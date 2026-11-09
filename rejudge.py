import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Lenovo\Desktop\AgentPrivacy")
sys.path.insert(0, str(ROOT))

from config import Settings
from dataset import load_items
from defenses import recipient_of
from judge import _parse_labels
from llm import LLMClient
from prompts import build_leak_messages

ITEM = int(sys.argv[1]) if len(sys.argv) > 1 else 386

settings = Settings(root=ROOT, data=ROOT / "main_data.json", key=ROOT / "DSAPIKey.txt")
client = LLMClient(settings)

items = {it["id"]: it for it in load_items(ROOT / "main_data.json")}
item = items[ITEM]

gens = [
    json.loads(l)
    for l in (ROOT / "runs" / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    if l.strip()
]
rows = {r["condition"]: r for r in gens if r["item_id"] == ITEM}

out = {"item": ITEM, "rubric": "rule 9 — partial needs a specific element", "arms": {}}

for cond in ("C0", "C1", "C2", "C3"):
    row = rows[cond]
    action = row.get("action")
    recipient = recipient_of(action) if action else ""
    messages = build_leak_messages(item, action, recipient)
    obj, meta = client.call_json(
        model=settings.judge_model,
        messages=messages,
        max_tokens=2500,
        temperature=settings.temperature,
        tag=f"rejudge-{cond}",
    )
    labels = _parse_labels(obj, item) if obj else []
    out["arms"][cond] = {
        "run_id": row["run_id"],
        "labels": labels,
        "usage": meta.get("usage", {}),
        "latency_ms": meta.get("latency_ms", 0),
    }
    bad = [lb for lb in labels if lb.get("disclosed")]
    print(
        f"{cond}: {len(bad )}/{len(labels )} leaked  "
        + ", ".join(f"f{lb['index']}={lb['leak_type']}" for lb in bad)
    )

path = ROOT / "demo" / f"rejudged_{ITEM}.json"
path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("saved:", path.name)
print("tokens:", sum(out["arms"][c]["usage"].get("total_tokens", 0) for c in out["arms"]))
