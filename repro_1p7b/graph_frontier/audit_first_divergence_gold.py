"""Inventory replay-certified, single-call, unmasked gold tasks for v1 collection."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
from repro_1p7b.graph_frontier.build_first_divergence_v1 import (
    AUDIT, FROZEN, TRAIN, EXPECTED_FROZEN_SHA, EXPECTED_TRAIN_SHA, jsonl, sha256,
)
from repro_1p7b.graph_frontier.first_divergence_v1 import reference_from_raw

def inventory(output: Path) -> dict:
    if sha256(TRAIN) != EXPECTED_TRAIN_SHA or sha256(FROZEN) != EXPECTED_FROZEN_SHA:
        raise RuntimeError("source or Frozen300 hash mismatch")
    rows = json.loads(TRAIN.read_text())
    frozen_ids = {row["task_id"] for row in jsonl(FROZEN)}
    audits = {row["task_id"]: row for row in jsonl(AUDIT)}
    reasons = collections.Counter()
    depths = collections.Counter()
    eligible = []
    raw_cache = {}
    for row in rows:
        task_id = row["task_id"]
        audit = audits.get(task_id)
        if not audit:
            reasons["missing_audit"] += 1
            continue
        if task_id in frozen_ids:
            reasons["frozen_overlap"] += 1
            continue
        path = audit["raw_file"]
        if path not in raw_cache:
            raw_cache[path] = json.loads(Path(path).read_text())
        reference, reason = reference_from_raw(row, audit, raw_cache[path])
        reasons[reason] += 1
        if reference is None:
            continue
        sidecar = row["extra_info"]["graph_frontier"]
        depth = sidecar.get("dependency_depth", "unknown")
        depths[str(depth)] += 1
        eligible.append({
            "task_id": task_id, "depth": depth,
            "environment_identifiers": sidecar.get("environment_identifiers", []),
            "gold_step_count": len(reference["actions"]),
            "gold_trajectory_id": reference["gold_trajectory_id"],
            "raw_file": path, "gold_replay_valid": True,
            "frozen_overlap": False,
        })
    output.mkdir(parents=True, exist_ok=False)
    (output / "eligible_tasks.json").write_text(
        json.dumps(eligible, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    report = {"schema_version": "first_divergence_gold_inventory_v1",
              "train_tasks": len(rows), "eligible_tasks": len(eligible),
              "reasons": dict(sorted(reasons.items())),
              "depth_counts": dict(sorted(depths.items())),
              "frozen_exact_id_overlap": len({r["task_id"] for r in rows} & frozen_ids),
              "frozen_manifest_sha256": EXPECTED_FROZEN_SHA,
              "train_sha256": EXPECTED_TRAIN_SHA}
    (output / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inventory(args.output), indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
