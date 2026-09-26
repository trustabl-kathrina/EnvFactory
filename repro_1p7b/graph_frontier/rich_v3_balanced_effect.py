"""Small, task-disjoint Rich-v3 frontier pilot using already audited tasks.

This prepares and CPU-replays gold frontiers only. It never trains or reads
Frozen300 examples. Model sampling/training must pass later, separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.active_frontier_v2 import (
    inventory_record,
    replay_one,
    write_jsonl,
)
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256


SCHEMA = "graph_frontier_rich_v3_balanced_effect_v1"


def read_rows(source_run: Path) -> list[dict[str, Any]]:
    path = source_run / "valid/graph_frontier_rich_train_v1.jsonl"
    report = json.loads((source_run / "audit/validation_report.json").read_text())
    if report["verdict"] != "PARTIAL-STRUCTURAL-DATA-READY":
        raise ValueError("bounded pilot requires audited partial pool, not a claimed formal-ready pool")
    if report["funnel"]["final_valid_tasks"] != 192 or report["funnel"]["replay_pass"] != 192:
        raise ValueError("pilot source task count or full replay gate changed")
    if report["frozen300"]["exact_overlap"] != 0:
        raise ValueError("source overlaps Frozen300")
    if report["artifacts"]["valid_data_sha256"] != file_sha256(path):
        raise ValueError("source data hash changed")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 192 or len({row["task_id"] for row in rows}) != 192:
        raise ValueError("expected 192 independent validated pilot tasks")
    if any(row.get("frozen_300_overlap") is not False for row in rows):
        raise ValueError("source task overlap flag is not false")
    return rows


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite quick-effect pilot")
    rows = read_rows(args.source_run)
    quotas = {
        "depth1": (61, 8, 9),
        "depth2": (44, 6, 6),
        "depth3plus": (45, 7, 6),
    }
    splits = {"train": [], "preference_val": [], "heldout_eval": []}
    for bucket, quota in quotas.items():
        subset = [row for row in rows
                  if row["extra_info"]["graph_frontier"]["target_bucket"] == bucket]
        subset = sorted(subset, key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['task_id']}".encode()).hexdigest())
        if len(subset) != sum(quota):
            raise ValueError(f"source depth bucket changed: {bucket}")
        train_n, val_n, heldout_n = quota
        splits["train"].extend(subset[:train_n])
        splits["preference_val"].extend(subset[train_n:train_n + val_n])
        splits["heldout_eval"].extend(subset[train_n + val_n:])
    for name in splits:
        splits[name].sort(key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['task_id']}".encode()).hexdigest())
    if [len(splits[name]) for name in splits] != [150, 21, 21]:
        raise ValueError("task split quotas changed")
    records = []
    for row in rows:
        edges = row["extra_info"]["graph_frontier"].get("dependency_edges", [])
        records.extend(inventory_record(row, edge, index) for index, edge in enumerate(edges))
    if len(records) != 364 or not all(record["static_eligible"] for record in records):
        raise ValueError("source static frontier count changed")
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "frontier_edge_inventory.jsonl", records)
    for name, subset in splits.items():
        write_jsonl(args.output_dir / name / "source_tasks.jsonl", subset)
    manifest = {
        "schema_version": SCHEMA,
        "source_run": str(args.source_run.resolve()),
        "source_data_sha256": file_sha256(args.source_run / "valid/graph_frontier_rich_train_v1.jsonl"),
        "source_gold_sha256": file_sha256(args.source_run / "valid/graph_frontier_rich_gold_v1.jsonl"),
        "source_verdict": "PARTIAL-STRUCTURAL-DATA-READY",
        "frozen300_exact_overlap": 0,
        "split_seed": args.seed,
        "task_ids": {name: [row["task_id"] for row in subset] for name, subset in splits.items()},
        "source_tasks": len(rows),
        "static_eligible_edges": len(records),
        "static_depths": dict(Counter(str(record["dependency_depth"]) for record in records)),
        "model_training_started": False,
    }
    (args.output_dir / "quick_effect_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def replay(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((args.output_dir / "quick_effect_manifest.json").read_text())
    rows = read_rows(args.source_run)
    if manifest["source_data_sha256"] != file_sha256(
        args.source_run / "valid/graph_frontier_rich_train_v1.jsonl"
    ):
        raise ValueError("source changed after split freeze")
    by_id = {row["task_id"]: row for row in rows}
    records = [json.loads(line) for line in (args.output_dir / "frontier_edge_inventory.jsonl").read_text().splitlines()]
    cache = args.output_dir / "replay_cache"
    cache.mkdir(exist_ok=True)
    for index, record in enumerate(records):
        identity = f"{record['task_id']}:{record['edge_index']}:{record['edge_id']}"
        key = hashlib.sha256(identity.encode()).hexdigest()[:24]
        path = cache / f"{key}.json"
        if path.exists():
            continue
        try:
            state = replay_one(by_id[record["task_id"]], record, args.output_dir / "replay_runtime", args.seed + index * 17)
            payload = {"status": "valid", "identity": identity, "state": state}
        except Exception as exc:
            payload = {"status": "dropped", "identity": identity, "reason": f"{type(exc).__name__}:{str(exc)[:300]}"}
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        print(f"RICH_V3_FRONTIER_REPLAY {index + 1}/{len(records)} {payload['status']}", flush=True)
    entries = [json.loads(path.read_text()) for path in sorted(cache.glob("*.json"))]
    if len(entries) != len(records):
        raise RuntimeError("replay cache is incomplete")
    states = [entry["state"] for entry in entries if entry["status"] == "valid"]
    if len({state["state_id"] for state in states}) != len(states):
        raise ValueError("duplicate replay frontier state")
    split_of = {task_id: name for name, ids in manifest["task_ids"].items() for task_id in ids}
    for name in manifest["task_ids"]:
        write_jsonl(args.output_dir / name / "frontier_states.jsonl", [
            state for state in states if split_of[state["task_id"]] == name
        ])
    report = {
        "schema_version": SCHEMA,
        "source_tasks": 192,
        "static_eligible_edges": len(records),
        "replay_valid_states": len(states),
        "replay_dropped": len(records) - len(states),
        "drop_reasons": dict(Counter(entry["reason"] for entry in entries if entry["status"] == "dropped")),
        "split_states": dict(Counter(split_of[state["task_id"]] for state in states)),
        "split_tasks_with_states": {
            name: len({state["task_id"] for state in states if split_of[state["task_id"]] == name})
            for name in manifest["task_ids"]
        },
        "chosen_executable": sum(state["chosen_executable"] is True for state in states),
        "frozen300_exact_overlap": 0,
        "model_training_started": False,
    }
    (args.output_dir / "replay_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "replay"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    report = prepare(args) if args.command == "prepare" else replay(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
