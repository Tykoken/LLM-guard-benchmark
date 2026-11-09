

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from config import CONDITIONS, Settings
from dataset import load_items
from defenses import generate_run
from judge import (
    annotate_item,
    fallback_leakage,
    judge_leakage,
    judge_refusal,
    judge_utility,
    load_annotations,
    save_annotations,
)
from llm import LLMClient
from metrics import read_jsonl, write_results
import prompts


def _prompt_hashes() -> dict[str, str]:
    names = [
        "SYSTEM_BASE",
        "C1_POLICY",
        "REVIEW_SYSTEM",
        "LEAK_SYSTEM",
        "UTILITY_SYSTEM",
        "REFUSAL_SYSTEM",
        "ANNOTATE_SYSTEM",
    ]
    return {
        name: hashlib.sha256(getattr(prompts, name).encode("utf-8")).hexdigest()[:16]
        for name in names
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _print_progress(label: str, done: int, total: int) -> None:
    if done == total or done % 10 == 0:
        print(f"  {label}: {done}/{total}", flush=True)


def ensure_annotations(client: LLMClient, items: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    annotations = load_annotations(settings.annotations_path)
    missing = [item for item in items if str(item["id"]) not in annotations]
    if missing:
        print(f"annotating {len(missing)} items with {settings.judge_model} ...", flush=True)
        with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
            futures = {pool.submit(annotate_item, client, item): item for item in missing}
            done = 0
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "benign_facts": [],
                        "reference_action": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "source": "auto",
                    }
                annotations[str(item["id"])] = record
                done += 1
                _print_progress("annotations", done, len(missing))
        save_annotations(settings.annotations_path, annotations)
    return annotations


def run_generations(
    client: LLMClient,
    items: list[dict[str, Any]],
    settings: Settings,
    conditions: list[str],
    force: bool,
) -> None:
    if force and settings.generations_path.exists():
        settings.generations_path.unlink()
    existing = {row.get("run_id") for row in read_jsonl(settings.generations_path)}
    tasks = []
    for condition in conditions:
        for item in items:
            for repeat in range(settings.repeats):
                run_id = f"{condition}-{item['id']}-r{repeat}"
                if run_id not in existing:
                    tasks.append((condition, item, repeat))
    if not tasks:
        print("generations: nothing to do", flush=True)
        return
    print(f"generations: {len(tasks)} runs ({len(conditions)} conditions, {len(items)} items, {settings.repeats} repeats)", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
        futures = {
            pool.submit(generate_run, client, item, condition, repeat): (condition, item, repeat)
            for condition, item, repeat in tasks
        }
        for future in as_completed(futures):
            condition, item, repeat = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "run_id": f"{condition}-{item['id']}-r{repeat}",
                    "condition": condition,
                    "item_id": item["id"],
                    "repeat": repeat,
                    "model": settings.gen_model,
                    "action": None,
                    "invalid": True,
                    "blocked": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "usage": {},
                    "latency_ms": 0,
                    "wall_ms": 0,
                }
            _append_jsonl(settings.generations_path, row)
            done += 1
            _print_progress("generations", done, len(tasks))
    print("generations: done", flush=True)


def run_judgments(
    client: LLMClient,
    items: list[dict[str, Any]],
    settings: Settings,
    annotations: dict[str, Any],
    force: bool,
) -> None:
    if force and settings.judgments_path.exists():
        settings.judgments_path.unlink()
    generations = read_jsonl(settings.generations_path)
    existing = {row.get("judgment_id") for row in read_jsonl(settings.judgments_path)}
    item_by_id = {item["id"]: item for item in items}
    kinds = ("leakage", "utility", "refusal")
    pending = []
    for row in generations:
        run_id = row.get("run_id")
        if all(f"{run_id}:{kind}" in existing for kind in kinds):
            continue
        item = item_by_id.get(row.get("item_id"))
        if item is None:
            continue
        pending.append((row, item, annotations.get(str(item["id"]), {})))

    if not pending:
        print("judgments: nothing to do", flush=True)
        return
    print(f"judgments: {len(pending)} generation rows", flush=True)

    def judge_one(row: dict[str, Any], item: dict[str, Any], annotation: dict[str, Any]) -> list[dict[str, Any]]:
        run_id = row["run_id"]
        base = {
            "run_id": run_id,
            "condition": row["condition"],
            "item_id": item["id"],
            "repeat": row.get("repeat", 0),
        }
        records = []
        if f"{run_id}:leakage" not in existing:
            try:
                leak = judge_leakage(client, item, row, annotation)
            except Exception as exc:
                leak = {
                    "labels": fallback_leakage(item, row.get("action")),
                    "method": f"fallback-error: {type(exc).__name__}",
                    "usage": {},
                    "latency_ms": 0,
                    "raw": str(exc),
                }
            records.append({"judgment_id": f"{run_id}:leakage", "kind": "leakage", **base, **leak})
        if f"{run_id}:utility" not in existing:
            try:
                utility = judge_utility(client, item, row, annotation)
            except Exception as exc:
                utility = {
                    "outcome": "invalid",
                    "utility": 0,
                    "retained": [],
                    "missing": list(range(1, len(annotation.get("benign_facts", [])) + 1)),
                    "reason": f"judge error: {type(exc).__name__}: {exc}",
                    "method": "fallback-error",
                    "usage": {},
                    "latency_ms": 0,
                    "raw": str(exc),
                }
            records.append({"judgment_id": f"{run_id}:utility", "kind": "utility", **base, **utility})
        if f"{run_id}:refusal" not in existing:
            try:
                refusal = judge_refusal(client, item, row, annotation)
            except Exception as exc:
                refusal = {
                    "refusal": "judge_error",
                    "evidence": "",
                    "reason": f"judge error: {type(exc).__name__}: {exc}",
                    "confidence": 0.0,
                    "method": "fallback-error",
                    "usage": {},
                    "latency_ms": 0,
                    "raw": str(exc),
                }
            records.append({"judgment_id": f"{run_id}:refusal", "kind": "refusal", **base, **refusal})
        return records

    done = 0
    with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
        futures = {
            pool.submit(judge_one, row, item, annotation): row
            for row, item, annotation in pending
        }
        for future in as_completed(futures):
            try:
                records = future.result()
            except Exception as exc:
                row = futures[future]
                records = [
                    {
                        "judgment_id": f"{row['run_id']}:leakage",
                        "kind": "leakage",
                        "run_id": row["run_id"],
                        "condition": row["condition"],
                        "item_id": row["item_id"],
                        "repeat": row.get("repeat", 0),
                        "labels": [],
                        "method": f"fatal: {type(exc).__name__}",
                        "usage": {},
                        "latency_ms": 0,
                        "raw": str(exc),
                    },
                    {
                        "judgment_id": f"{row['run_id']}:utility",
                        "kind": "utility",
                        "run_id": row["run_id"],
                        "condition": row["condition"],
                        "item_id": row["item_id"],
                        "repeat": row.get("repeat", 0),
                        "outcome": "invalid",
                        "utility": 0,
                        "retained": [],
                        "missing": [],
                        "reason": f"fatal: {type(exc).__name__}",
                        "method": "fatal",
                        "usage": {},
                        "latency_ms": 0,
                        "raw": str(exc),
                    },
                    {
                        "judgment_id": f"{row['run_id']}:refusal",
                        "kind": "refusal",
                        "run_id": row["run_id"],
                        "condition": row["condition"],
                        "item_id": row["item_id"],
                        "repeat": row.get("repeat", 0),
                        "refusal": "judge_error",
                        "evidence": "",
                        "reason": f"fatal: {type(exc).__name__}",
                        "confidence": 0.0,
                        "method": "fatal",
                        "usage": {},
                        "latency_ms": 0,
                        "raw": str(exc),
                    },
                ]
            for record in records:
                _append_jsonl(settings.judgments_path, record)
            done += 1
            _print_progress("judgments", done, len(pending))
    print("judgments: done", flush=True)


def print_summary(summary: list[dict[str, Any]]) -> None:
    if not summary:
        return
    header = (
        f"{'arm':<5} {'n':>5} {'item_leak':>10} {'sens_leak':>10} {'recall':>8} "
        f"{'prec':>7} {'util':>6} {'complete':>9} {'false_ref':>10} {'block':>7} "
        f"{'invalid':>8} {'p50_ms':>8} {'tokens':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['condition']:<5} {row['n_runs']:>5} {row['item_leak_rate']:>10.3f} "
            f"{row['sensitive_item_leak_rate']:>10.3f} {row['privacy_recall']:>8.3f} "
            f"{row['privacy_precision']:>7.3f} {row['mean_utility']:>6.2f} "
            f"{row['completion_rate']:>9.3f} {row['false_refusal_rate']:>10.3f} "
            f"{row['block_rate']:>7.3f} {row['invalid_rate']:>8.3f} "
            f"{row['latency_p50_ms']:>8.0f} {row['total_tokens']:>10}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the PrivacyLens safeguard experiment")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--data", default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--limit", type=int, default=0, help="only the first N items (pilot)")
    parser.add_argument("--force", action="store_true", help="delete prior runs/judgments and rerun")
    parser.add_argument("--annotate-only", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root)
    settings = Settings(
        root=root,
        data=Path(args.data) if args.data else root / "privacylens_25.json",
        key=Path(args.key) if args.key else root / "DSAPIKey.txt",
        repeats=args.repeats,
        concurrency=args.concurrency,
    )
    conditions = [part.strip() for part in args.conditions.split(",") if part.strip()]
    invalid = [condition for condition in conditions if condition not in CONDITIONS]
    if invalid:
        parser.error(f"unknown conditions: {invalid}")

    items = load_items(settings.data)
    if args.limit:
        items = items[: args.limit]
    print(f"items: {len(items)}; conditions: {conditions}; repeats: {settings.repeats}", flush=True)
    print(f"generator: {settings.gen_model}; judge: {settings.judge_model}", flush=True)

    client = LLMClient(settings)
    if args.force and settings.annotations_path.exists():
        settings.annotations_path.unlink()
    annotations = ensure_annotations(client, items, settings)
    if args.annotate_only:
        print(f"annotations saved to {settings.annotations_path}", flush=True)
        return 0

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_generations(client, items, settings, conditions, args.force)
    if not args.skip_judge:
        run_judgments(client, items, settings, annotations, args.force)

    generations = read_jsonl(settings.generations_path)
    judgments = read_jsonl(settings.judgments_path)
    result = write_results(
        settings,
        items,
        generations,
        judgments,
        annotations,
        _prompt_hashes(),
        extra_metadata={
            "created_at": started,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "conditions": conditions,
            "llm_stats": client.stats,
        },
    )
    print_summary(result["summary"])
    print(f"\nwrote {settings.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
