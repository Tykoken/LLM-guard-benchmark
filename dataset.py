

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(
    r"Action:\s*(.+?)\s*Action Input:\s*(.+?)\s*Observation:\s*(.*?)(?=\s*Action:\s|\Z)",
    re.S,
)


def _clean(text: str) -> str:
    return text.strip().replace("\r\n", "\n")


def parse_steps(trajectory: str) -> list[dict[str, str]]:
    steps = []
    for action, action_input, observation in STEP_RE.findall(trajectory):
        steps.append(
            {
                "action": _clean(action),
                "input": _clean(action_input),
                "observation": _clean(observation),
            }
        )
    if not steps:
        raise ValueError("could not parse any ReAct step")
    return steps


def format_steps(steps: list[dict[str, str]]) -> str:
    blocks = []
    for index, step in enumerate(steps, 1):
        blocks.append(
            f"{index}. Action: {step['action']}\n"
            f"   Action Input: {step['input']}\n"
            f"   Observation: {step['observation']}"
        )
    return "\n\n".join(blocks)


def _derive_id(record: dict[str, Any], fallback: int) -> int:

    raw_id = record.get("id")
    if raw_id is not None:
        try:
            return int(raw_id)
        except (TypeError, ValueError):
            pass
    match = re.fullmatch(r"main(\d+)", str(record.get("name", "")))
    if match:
        return int(match.group(1))
    return fallback


def load_items(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    for index, record in enumerate(raw, 1):
        trajectory = record["trajectory"]
        steps = parse_steps(trajectory["executable_trajectory"])
        item_id = _derive_id(record, index)
        items.append(
            {
                "id": item_id,
                "name": record["name"],
                "seed": record["seed"],
                "vignette": record["vignette"],
                "story": record["vignette"]["story"],
                "instruction": trajectory["user_instruction"],
                "toolkits": trajectory["toolkits"],
                "final_action": trajectory["final_action"],
                "sensitive": list(trajectory["sensitive_info_items"]),
                "steps": steps,
            }
        )
    return items


def action_as_text(action: dict[str, Any] | None) -> str:
    if action is None:
        return "(no action; the assistant did not perform one)"
    return json.dumps(action, ensure_ascii=False, indent=2)
