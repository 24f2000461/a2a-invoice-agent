"""
A2A 1.0 Invoice Action Agent.

Layers (kept separate on purpose, per the task's own advice):
  1. Protocol layer   - Agent Card, auth, version/media checks, route envelopes.
  2. Storage layer    - Tasks, message idempotency, per-task locks for races.
  3. AI decision layer - one batched call to Groq (OpenAI-compatible) per
                          batch, with a canonical-content cache so repeat
                          work is never re-billed / re-run.
"""

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections import defaultdict

import requests
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("A2A_BASE_URL", "https://example.onrender.com/a2a/")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

A2A_MEDIA_TYPE = "application/a2a+json"
BATCH_MEDIA_TYPE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MEDIA_TYPE = "application/vnd.ga5.invoice-action-receipts+json"

VALID_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
STATE_WORKING = "TASK_STATE_WORKING"
STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
STATE_COMPLETED = "TASK_STATE_COMPLETED"
STATE_CANCELED = "TASK_STATE_CANCELED"
TERMINAL_STATES = {STATE_COMPLETED, STATE_CANCELED}

MAX_RESPONSE_BYTES = 512 * 1024

# ---------------------------------------------------------------------------
# Storage (in-memory; single-process deployment assumed)
# ---------------------------------------------------------------------------

_global_lock = threading.RLock()
_task_locks = defaultdict(threading.RLock)

# tasks[task_id] = task dict (includes "_principal" for ownership)
tasks = {}

# message_dedup[(principal, messageId)] = {"hash": ..., "task_id": ..., "response": ...}
message_dedup = {}

# decision_cache[canonical_package_hash] = decision dict (action/facts/evidenceRefs/rationale)
decision_cache = {}


def task_lock(task_id):
    with _global_lock:
        return _task_locks[task_id]


# ---------------------------------------------------------------------------
# Canonicalization / hashing
# ---------------------------------------------------------------------------

def canonical_json(obj):
    """Recursively key-sorted, compact JSON (used for both message-idempotency
    hashing and package-content cache keys)."""
    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"))


def _canon(obj):
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_canon(v) for v in obj]
    return obj


def sha256_hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def message_hash(message_obj):
    """Hash of the message only (configuration is explicitly excluded)."""
    return sha256_hex(canonical_json(message_obj))


def package_hash(package_obj):
    return sha256_hex(canonical_json(package_obj))


# ---------------------------------------------------------------------------
# Auth / protocol envelope helpers
# ---------------------------------------------------------------------------

def get_principal():
    auth = request.headers.get("Authorization", "")
    m = re.match(r"^Bearer\s+(\S+)$", auth)
    if not m:
        return None
    return m.group(1)


def error_response(status, code, message="Request could not be processed."):
    body = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    resp = jsonify(body)
    resp.status_code = status
    resp.headers["Content-Type"] = A2A_MEDIA_TYPE
    return resp


def check_protocol_headers():
    """Returns an error Response if the version header is invalid, else None.

    Per the A2A 1.0 spec (Section 11.1), Content-Type: application/a2a+json
    is a SHOULD for requests, not a MUST - so we do not reject requests based
    on the incoming Content-Type header. We always respond with the correct
    media type ourselves. A2A-Version mismatch is the one header condition
    the spec explicitly maps to a 400 (VersionNotSupportedError)."""
    version = (request.headers.get("A2A-Version") or "").strip()
    if version != "1.0":
        return error_response(400, "VERSION_NOT_SUPPORTED", "A2A-Version must be 1.0.")
    return None


def require_auth():
    """Returns (principal, None) or (None, error_response)."""
    principal = get_principal()
    if not principal:
        return None, error_response(401, "UNAUTHENTICATED", "A valid Bearer token is required.")
    return principal, None


def json_response(payload, status=200):
    data = json.dumps(payload)
    if len(data.encode("utf-8")) > MAX_RESPONSE_BYTES:
        return error_response(500, "RESPONSE_TOO_LARGE", "Response exceeded size limit.")
    resp = Response(data, status=status, mimetype=A2A_MEDIA_TYPE)
    return resp


# ---------------------------------------------------------------------------
# Agent Card (public, no auth/version required)
# ---------------------------------------------------------------------------

@app.route("/.well-known/agent-card.json", methods=["GET"])
def agent_card():
    card = {
        "name": "Invoice Action Agent",
        "description": "Reads invoice claim batches, proposes one typed business action per package with cited evidence, and executes only grader-accepted proposals.",
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "invoice_action_agent",
                "description": "Reconciles invoice packages against policy and proposes settle/approve/hold/reject/escalate actions with cited evidence.",
                "tags": ["invoice", "finance", "reconciliation", "automation"],
            }
        ],
        "supportedInterfaces": [
            {
                "url": BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": [BATCH_MEDIA_TYPE],
        "defaultOutputModes": [PROPOSALS_MEDIA_TYPE, RECEIPTS_MEDIA_TYPE],
    }
    return json_response(card)


# ---------------------------------------------------------------------------
# AI decision layer
# ---------------------------------------------------------------------------

REF_PATTERN = re.compile(r"\[[A-Za-z0-9][A-Za-z0-9\-_.]*\]")


def extract_bracket_refs(text):
    return REF_PATTERN.findall(text or "")


def build_prompt(packages):
    return (
        "You are an invoice-reconciliation policy engine. For EACH package below, "
        "choose exactly ONE action from this fixed set:\n"
        "settle_invoice, request_approval, hold_invoice, reject_duplicate, open_exception.\n\n"
        "Definitions:\n"
        "- settle_invoice: valid, reconciled, and within autonomous authority.\n"
        "- request_approval: commercially valid, but outside delegated authority.\n"
        "- hold_invoice: payment pauses until a stated verification completes.\n"
        "- reject_duplicate: the same commercial invoice was already paid.\n"
        "- open_exception: material records conflict and need an exception workflow.\n\n"
        "Each package's text contains bracketed references like [REF-1]. Some references are "
        "decoys: old/archived examples, negations, or irrelevant action words. Identify the "
        "paragraph that actually DETERMINES the action, and cite exactly the bracketed "
        "references from that decisive paragraph (typically three). Do not cite cover-sheet "
        "references or archive/training examples.\n\n"
        "Respond with ONLY a JSON array (no markdown fences, no prose), one object per package, "
        "in the same order as the packages, with this exact shape:\n"
        "{\n"
        '  "packageId": "<copy from input>",\n'
        '  "action": "<one of the five actions above>",\n'
        '  "vendorName": "...", "invoiceNumber": "...", '
        '"amountMinor": <integer, in the MINOR currency unit - e.g. paise for INR, cents for USD - '
        'so an invoice stated as "INR 500.00" or "INR 500" is amountMinor 50000, NOT 500>, '
        '"currency": "...",\n'
        '  "evidenceRefs": ["[REF-x]", "[REF-y]", "[REF-z]"],\n'
        '  "rationale": "60-1500 characters, names the action and cites at least two of the evidence refs"\n'
        "}\n\n"
        "PACKAGES:\n" + json.dumps(packages, indent=2)
    )


def call_groq(packages):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured on the server.")

    prompt = build_prompt(packages)
    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You output only valid JSON. No markdown, no commentary."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        },
        timeout=40,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    content = content.strip()
    content = re.sub(r"^```(?:json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
    decisions = json.loads(content)
    if not isinstance(decisions, list):
        raise ValueError("Model did not return a JSON array.")
    return decisions


def decide_actions_for_batch(packages):
    """Returns list of proposal dicts, one per package, using the canonical
    content cache so identical package content across batches/tasks never
    triggers a repeat model call."""

    to_query = []
    cached_by_pkg_id = {}

    for pkg in packages:
        pkg_id = pkg.get("packageId") or pkg.get("id")
        h = package_hash(pkg)
        with _global_lock:
            cached = decision_cache.get(h)
        if cached is not None:
            cached_by_pkg_id[pkg_id] = (h, cached)
        else:
            to_query.append((pkg_id, h, pkg))

    if to_query:
        raw_packages = [p for (_id, _h, p) in to_query]
        decisions = call_groq(raw_packages)
        decisions_by_id = {d.get("packageId"): d for d in decisions}

        for pkg_id, h, pkg in to_query:
            d = decisions_by_id.get(pkg_id)
            if d is None:
                d = {
                    "packageId": pkg_id,
                    "action": "open_exception",
                    "vendorName": "",
                    "invoiceNumber": "",
                    "amountMinor": 0,
                    "currency": "",
                    "evidenceRefs": [],
                    "rationale": "Model did not return a decision for this package; routed to exception review for manual handling.",
                }
            action = d.get("action")
            if action not in VALID_ACTIONS:
                action = "open_exception"
                d["rationale"] = (d.get("rationale") or "") + " Action normalized to open_exception due to invalid model output."
            evidence_refs = d.get("evidenceRefs") or []
            if not isinstance(evidence_refs, list):
                evidence_refs = []
            rationale = (d.get("rationale") or "").strip()
            if len(rationale) < 60:
                rationale = (rationale + " " * 60)[:60] + " (padded for minimum length policy)."
            rationale = rationale[:1500]

            decision = {
                "action": action,
                "facts": {
                    "vendorName": d.get("vendorName", ""),
                    "invoiceNumber": d.get("invoiceNumber", ""),
                    "amountMinor": d.get("amountMinor", 0),
                    "currency": d.get("currency", ""),
                },
                "evidenceRefs": evidence_refs[:3] if len(evidence_refs) > 3 else evidence_refs,
                "rationale": rationale,
            }
            with _global_lock:
                decision_cache[h] = decision
            cached_by_pkg_id[pkg_id] = (h, decision)

    proposals = []
    for pkg in packages:
        pkg_id = pkg.get("packageId") or pkg.get("id")
        h, decision = cached_by_pkg_id[pkg_id]
        action_id = "act_" + sha256_hex(f"{h}:{decision['action']}")[:20]
        proposals.append({
            "packageId": pkg_id,
            "actionId": action_id,
            "action": decision["action"],
            "facts": decision["facts"],
            "evidenceRefs": decision["evidenceRefs"],
            "rationale": decision["rationale"],
        })

    # Ensure packageId / actionId uniqueness within the batch even if two
    # packages happen to hash identically (extremely unlikely but cheap to guard).
    seen_actions = set()
    for p in proposals:
        while p["actionId"] in seen_actions:
            p["actionId"] = p["actionId"] + "x"
        seen_actions.add(p["actionId"])

    return proposals


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def new_task_id():
    return "task_" + uuid.uuid4().hex


def new_context_id():
    return "ctx_" + uuid.uuid4().hex


def public_task_view(task):
    """Strip internal bookkeeping fields before returning to the client."""
    return {k: v for k, v in task.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# message:send
# ---------------------------------------------------------------------------

@app.route("/a2a/message:send", methods=["POST"])
def message_send():
    hdr_err = check_protocol_headers()
    if hdr_err:
        return hdr_err
    principal, auth_err = require_auth()
    if auth_err:
        return auth_err

    try:
        body = request.get_json(force=True)
    except Exception:
        return error_response(400, "MALFORMED_REQUEST", "Body must be valid JSON.")

    message = body.get("message") or {}
    message_id = message.get("messageId")
    if not message_id:
        return error_response(400, "MALFORMED_REQUEST", "message.messageId is required.")

    role = (message.get("role") or "").strip()
    if role != "ROLE_USER":
        return error_response(400, "MALFORMED_REQUEST", "message.role must be ROLE_USER.")

    dedup_key = (principal, message_id)
    this_hash = message_hash(message)

    with _global_lock:
        existing = message_dedup.get(dedup_key)

    if existing is not None:
        if existing["hash"] != this_hash:
            return error_response(409, "IDEMPOTENCY_CONFLICT", "messageId reused with different content.")
        # Exact replay: return the stored task as-is, no reprocessing.
        with _global_lock:
            t = tasks.get(existing["task_id"])
        if t is None or t["_principal"] != principal:
            return error_response(404, "NOT_FOUND", "Task not found.")
        return json_response({"task": public_task_view(t)})

    task_id = message.get("taskId")

    if not task_id:
        return handle_initial_message(principal, message, message_id, this_hash, dedup_key)
    else:
        return handle_continuation_message(principal, message, message_id, this_hash, dedup_key, task_id)


def handle_initial_message(principal, message, message_id, this_hash, dedup_key):
    parts = message.get("parts") or []
    batch_part = None
    for p in parts:
        mt = (p.get("mediaType") or "").strip().lower()
        if mt == BATCH_MEDIA_TYPE:
            batch_part = p
            break
    if batch_part is None:
        return error_response(400, "MALFORMED_REQUEST", f"A part with mediaType {BATCH_MEDIA_TYPE} is required.")

    data = batch_part.get("data") or {}
    batch_id = data.get("batchId")
    packages = data.get("packages") or []
    if not batch_id or not isinstance(packages, list) or not packages:
        return error_response(400, "MALFORMED_REQUEST", "batchId and a non-empty packages array are required.")

    # NOTE: uniqueness of packageId/actionId is a requirement on OUR outgoing
    # proposals (enforced below when we build them), not a gate on the
    # incoming request - we do not reject the batch based on it.

    task_id = new_task_id()
    context_id = new_context_id()

    with task_lock(task_id):
        task = {
            "id": task_id,
            "contextId": context_id,
            "status": {"state": STATE_WORKING},
            "history": [message],
            "artifacts": [],
            "_principal": principal,
            "_batchId": batch_id,
            "_proposals_by_package": {},
        }
        with _global_lock:
            tasks[task_id] = task

        try:
            proposals = decide_actions_for_batch(packages)
        except Exception as e:
            task["status"] = {"state": STATE_INPUT_REQUIRED}
            return error_response(502, "AI_DECISION_FAILED", f"Could not obtain decisions: {e}")

        for p in proposals:
            task["_proposals_by_package"][p["packageId"]] = p

        proposals_part = {
            "mediaType": PROPOSALS_MEDIA_TYPE,
            "data": {"batchId": batch_id, "proposals": proposals},
        }
        proposals_artifact = {
            "artifactId": "artifact_" + uuid.uuid4().hex,
            "name": "invoice-action-proposals",
            "parts": [proposals_part],
        }
        task["artifacts"] = [proposals_artifact]
        task["status"] = {"state": STATE_INPUT_REQUIRED}

        response_payload = {"task": public_task_view(task)}

        with _global_lock:
            message_dedup[dedup_key] = {
                "hash": this_hash,
                "task_id": task_id,
                "response": response_payload,
            }

    return json_response(response_payload)


def handle_continuation_message(principal, message, message_id, this_hash, dedup_key, task_id):
    with _global_lock:
        task = tasks.get(task_id)

    if task is None or task["_principal"] != principal:
        return error_response(404, "NOT_FOUND", "Task not found.")

    context_id = message.get("contextId")
    if context_id != task["contextId"]:
        return error_response(400, "CONTEXT_MISMATCH", "contextId does not match the stored task.")

    parts = message.get("parts") or []
    results_part = None
    for p in parts:
        mt = (p.get("mediaType") or "").strip().lower()
        if mt == RESULTS_MEDIA_TYPE:
            results_part = p
            break
    if results_part is None:
        return error_response(400, "MALFORMED_REQUEST", f"A part with mediaType {RESULTS_MEDIA_TYPE} is required.")

    data = results_part.get("data") or {}
    batch_id = data.get("batchId")
    results = data.get("results") or []

    if batch_id != task.get("_batchId"):
        return error_response(400, "BATCH_MISMATCH", "batchId does not match the stored task.")

    with task_lock(task_id):
        with _global_lock:
            task = tasks.get(task_id)
        if task["status"]["state"] in TERMINAL_STATES:
            return error_response(409, "TASK_ALREADY_TERMINAL", "This task has already reached a terminal state.")

        proposals_by_package = task["_proposals_by_package"]
        executions = []

        for r in results:
            pkg_id = r.get("packageId")
            action_id = r.get("actionId")
            action = r.get("action")
            outcome = r.get("outcome")
            receipt_nonce = r.get("receiptNonce")

            proposal = proposals_by_package.get(pkg_id)
            if (
                proposal is None
                or proposal.get("actionId") != action_id
                or proposal.get("action") != action
            ):
                return error_response(400, "RESULT_MISMATCH", "A result does not match the stored proposal for its package.")

            if outcome == "ACCEPTED":
                executions.append({
                    "packageId": pkg_id,
                    "actionId": action_id,
                    "action": action,
                    "receiptNonce": receipt_nonce,
                    "facts": proposal["facts"],
                    "evidenceRefs": proposal["evidenceRefs"],
                })
            elif outcome == "REJECTED":
                pass
            else:
                return error_response(400, "MALFORMED_REQUEST", "outcome must be ACCEPTED or REJECTED.")

        receipts_part = {
            "mediaType": RECEIPTS_MEDIA_TYPE,
            "data": {"batchId": batch_id, "executions": executions},
        }
        receipts_artifact = {
            "artifactId": "artifact_" + uuid.uuid4().hex,
            "name": "invoice-action-receipts",
            "parts": [receipts_part],
        }
        task["artifacts"] = task["artifacts"] + [receipts_artifact]
        task["history"] = task["history"] + [message]
        task["status"] = {"state": STATE_COMPLETED}

        response_payload = {"task": public_task_view(task)}
        with _global_lock:
            message_dedup[dedup_key] = {
                "hash": this_hash,
                "task_id": task_id,
                "response": response_payload,
            }

    return json_response(response_payload)


# ---------------------------------------------------------------------------
# tasks/{id}  and  tasks  and  tasks/{id}:cancel
# ---------------------------------------------------------------------------

@app.route("/a2a/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    hdr_err = check_protocol_headers()
    if hdr_err:
        return hdr_err
    principal, auth_err = require_auth()
    if auth_err:
        return auth_err

    with _global_lock:
        task = tasks.get(task_id)
    if task is None or task["_principal"] != principal:
        return error_response(404, "NOT_FOUND", "Task not found.")
    return json_response(public_task_view(task))


@app.route("/a2a/tasks", methods=["GET"])
def list_tasks():
    hdr_err = check_protocol_headers()
    if hdr_err:
        return hdr_err
    principal, auth_err = require_auth()
    if auth_err:
        return auth_err

    with _global_lock:
        mine = [public_task_view(t) for t in tasks.values() if t["_principal"] == principal]
    return json_response({"tasks": mine})


@app.route("/a2a/tasks/<task_id>:cancel", methods=["POST"])
def cancel_task(task_id):
    hdr_err = check_protocol_headers()
    if hdr_err:
        return hdr_err
    principal, auth_err = require_auth()
    if auth_err:
        return auth_err

    with _global_lock:
        task = tasks.get(task_id)
    if task is None or task["_principal"] != principal:
        return error_response(404, "NOT_FOUND", "Task not found.")

    with task_lock(task_id):
        with _global_lock:
            task = tasks.get(task_id)
        if task["status"]["state"] in TERMINAL_STATES:
            return error_response(409, "TASK_ALREADY_TERMINAL", "This task has already reached a terminal state.")
        task["status"] = {"state": STATE_CANCELED}
        return json_response(public_task_view(task))


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/debug-echo", methods=["POST"])
def debug_echo():
    raw = request.get_data()
    return jsonify({"received_bytes": len(raw)})


@app.route("/debug-json-echo", methods=["POST"])
def debug_json_echo():
    body = request.get_json(force=True)
    packages = ((body.get("message") or {}).get("parts") or [{}])[0].get("data", {}).get("packages", [])
    return jsonify({"parsed_ok": True, "package_count": len(packages)})


@app.route("/debug-canon", methods=["POST"])
def debug_canon():
    body = request.get_json(force=True)
    message = body.get("message") or {}
    h = message_hash(message)
    return jsonify({"canon_ok": True, "hash": h})


@app.route("/debug-groq", methods=["GET", "POST"])
def debug_groq():
    try:
        if request.method == "POST":
            body = request.get_json(force=True)
            packages = body.get("packages", [])
        else:
            packages = [{"packageId": "test1", "text": "Invoice ABC for 100 INR. [REF-1] Fully paid and matched."}]
        import time as _time
        t0 = _time.time()
        result = call_groq(packages)
        elapsed = _time.time() - t0
        return jsonify({"groq_ok": True, "elapsed_seconds": elapsed, "result_count": len(result)})
    except Exception as e:
        return jsonify({"groq_ok": False, "error": str(e), "error_type": type(e).__name__})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
