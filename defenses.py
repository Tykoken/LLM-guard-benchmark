

from __future__ import annotations

import json
import re
import time
from typing import Any

from config import MAX_TOKENS_GEN, MAX_TOKENS_JUDGE
from dataset import format_steps
from prompts import build_generation_messages, build_review_messages

SCHEMAS = {
    "GmailSendEmail": ("to", "subject", "body"),
    "MessengerSendMessage": ("recipient", "message"),
    "FacebookManagerCreatePost": ("content",),
    "SlackSendMessage": ("recipient", "message"),
}


PII_PATTERNS = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)"),
    ),
    (
        "address",
        re.compile(
            r"\b\d{1,6}\s+(?:[A-Z][A-Za-z.'\-]+\s+){0,4}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Terrace|Ter|Court|Ct|Place|Pl|Way)\b\.?",
            re.I,
        ),
    ),
    ("money", re.compile(r"\$\s?\d[\d,]*(?:\.\d{1,2})?")),
]


def redact_text(text: str) -> tuple[str, list[dict[str, str]]]:

    redactions: list[dict[str, str]] = []
    for label, pattern in PII_PATTERNS:
        def replace(match: re.Match) -> str:
            value = match.group(0)
            if value.strip():
                redactions.append({"type": label, "value": value})
            return f"[REDACTED:{label.upper()}]"

        text = pattern.sub(replace, text)
    return text, redactions


def redact_steps(steps: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    redacted: list[dict[str, str]] = []
    all_redactions: list[dict[str, str]] = []
    for step in steps:
        observation, found = redact_text(step["observation"])
        all_redactions.extend(found)
        redacted.append({**step, "observation": observation})
    return redacted, all_redactions


def normalize_action(obj: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(obj, dict):
        return None, "reply is not a JSON object"
    action = obj.get("action")
    if action not in SCHEMAS:
        return None, f"unknown action {action!r}"
    payload = obj.get("action_input")
    if not isinstance(payload, dict):
        return None, "action_input is missing or not an object"
    payload = dict(payload)

    if action == "SlackSendMessage" and "recipient" not in payload and "channel" in payload:
        payload["recipient"] = payload["channel"]
    if action == "MessengerSendMessage" and "recipient" not in payload and "to" in payload:
        payload["recipient"] = payload["to"]

    required = SCHEMAS[action]
    cleaned: dict[str, str] = {}
    for field in required:
        value = payload.get(field, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        cleaned[field] = value.strip()

    if action == "GmailSendEmail":
        if not cleaned["to"] or not cleaned["body"]:
            return None, "email needs a non-empty to and body"
    elif action in ("MessengerSendMessage", "SlackSendMessage"):
        if not cleaned["recipient"] or not cleaned["message"]:
            return None, "message needs a non-empty recipient and message"
    elif action == "FacebookManagerCreatePost":
        if not cleaned["content"]:
            return None, "post needs non-empty content"

    return {"action": action, "action_input": cleaned}, None


def recipient_of(action: dict[str, Any] | None) -> str:
    if not action:
        return "(none)"
    payload = action.get("action_input", {})
    for field in ("to", "recipient", "channel"):
        value = payload.get(field)
        if value:
            return str(value)
    if action.get("action") == "FacebookManagerCreatePost":
        return "visitors of the Facebook page (public audience)"
    return "(not specified)"


def _gen_once(client: Any, item: dict[str, Any], condition: str, transcript: str, repeat: int) -> dict[str, Any]:
    messages = build_generation_messages(item, transcript, condition)
    obj, meta = client.call_json(
        model=client.settings.gen_model,
        messages=messages,
        max_tokens=MAX_TOKENS_GEN,
        temperature=client.settings.temperature,
        tag=f"gen-{condition}",
        cache_salt=f"gen-{condition}-r{repeat}",
    )
    action, error = normalize_action(obj)
    return {
        "action": action,
        "draft_action": action,
        "parse_error": error,
        "raw": meta.get("text", ""),
        "usage": meta.get("usage", {}),
        "latency_ms": meta.get("latency_ms", 0),
    }


def generate_run(client: Any, item: dict[str, Any], condition: str, repeat: int) -> dict[str, Any]:
    started = time.time()
    steps = item["steps"]
    filter_info: dict[str, Any] | None = None
    if condition == "C2":
        steps, redactions = redact_steps(steps)
        filter_info = {
            "redaction_count": len(redactions),
            "redactions": redactions,
        }
    transcript = format_steps(steps)

    row: dict[str, Any] = {
        "run_id": f"{condition}-{item['id']}-r{repeat}",
        "condition": condition,
        "item_id": item["id"],
        "repeat": repeat,
        "model": client.settings.gen_model,
        "filter": filter_info,
        "review": None,
        "blocked": False,
        "invalid": False,
        "error": None,
    }

    try:
        generated = _gen_once(client, item, condition, transcript, repeat)
        row.update(
            {
                "draft_action": generated["draft_action"],
                "action": generated["action"],
                "parse_error": generated["parse_error"],
                "raw": generated["raw"],
                "usage": generated["usage"],
                "latency_ms": generated["latency_ms"],
            }
        )
        if generated["action"] is None:
            row["invalid"] = True

        if condition == "C3" and generated["action"] is not None:
            review_messages = build_review_messages(item, transcript, generated["draft_action"])
            review_obj, review_meta = client.call_json(
                model=client.settings.judge_model,
                messages=review_messages,
                max_tokens=MAX_TOKENS_GEN,
                temperature=client.settings.temperature,
                tag="review-C3",
                cache_salt=f"review-C3-r{repeat}",
            )
            review = review_obj if isinstance(review_obj, dict) else {}
            decision = str(review.get("decision", "")).lower()
            revised, revised_error = (None, None)
            if decision == "revise":
                revised, revised_error = normalize_action(review.get("revised_action"))
            row["review"] = {
                "decision": decision,
                "reason": review.get("reason", ""),
                "revised_error": revised_error,
                "raw": review_meta.get("text", ""),
            }
            row["usage"] = _add_usage(row.get("usage", {}), review_meta.get("usage", {}))
            row["latency_ms"] = row.get("latency_ms", 0) + review_meta.get("latency_ms", 0)
            if decision == "block":
                row["action"] = None
                row["blocked"] = True
                row["invalid"] = False
            elif decision == "revise" and revised is not None:
                row["action"] = revised
                row["invalid"] = False
            elif decision == "revise" and revised is None:
                row["review"]["error"] = f"invalid revised action: {revised_error}"

    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["invalid"] = True
        row["action"] = None

    row["wall_ms"] = int((time.time() - started) * 1000)
    return row


def _add_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    keys = set(left) | set(right)
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in keys}
