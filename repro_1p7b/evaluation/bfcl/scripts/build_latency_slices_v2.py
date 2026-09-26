#!/usr/bin/env python3
"""Build latency-balanced BFCL shards from the completed Dynamic-v1 run."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

CATEGORY_SECONDS = {
    "multi_turn_base": 4708.0,
    "multi_turn_long_context": 13613.0,
    "multi_turn_miss_func": 5908.0,
    "multi_turn_miss_param": 398.0,
}
FILE_CATEGORY = {
    "BFCL_v3_multi_turn_base_result.json": "multi_turn_base",
    "BFCL_v3_multi_turn_long_context_result.json": "multi_turn_long_context",
    "BFCL_v3_multi_turn_miss_func_result.json": "multi_turn_miss_func",
    "BFCL_v3_multi_turn_miss_param_result.json": "multi_turn_miss_param",
}


def nested_sum(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return sum(nested_sum(item) for item in value)
    return 0.0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-result-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()

    weighted: list[dict[str, Any]] = []
    category_audit: dict[str, Any] = {}
    for file_name, category in FILE_CATEGORY.items():
        rows = read_jsonl(args.prior_result_dir / file_name)
        if len(rows) != 200:
            raise RuntimeError(f"{category}: expected 200 prior rows, got {len(rows)}")
        raw = {
            row["id"]: max(nested_sum(row.get("latency")), 0.001)
            for row in rows
        }
        total_raw = sum(raw.values())
        if total_raw <= 0:
            raise RuntimeError(f"{category}: no usable latency")
        scale = CATEGORY_SECONDS[category] / total_raw
        for task_id, latency in raw.items():
            weighted.append({
                "id": task_id,
                "category": category,
                "prior_latency_seconds": latency,
                "estimated_wall_seconds": latency * scale,
            })
        category_audit[category] = {
            "count": len(rows),
            "observed_wall_seconds": CATEGORY_SECONDS[category],
            "sum_prior_latency_seconds": total_raw,
            "wall_scale": scale,
        }

    shards = [
        {"estimated_wall_seconds": 0.0, "rows": []},
        {"estimated_wall_seconds": 0.0, "rows": []},
    ]
    for row in sorted(
        weighted,
        key=lambda item: (-item["estimated_wall_seconds"], item["id"]),
    ):
        target = min(
            range(2),
            key=lambda index: (shards[index]["estimated_wall_seconds"], index),
        )
        shards[target]["rows"].append(row)
        shards[target]["estimated_wall_seconds"] += row["estimated_wall_seconds"]

    plan_shards = []
    for index, shard in enumerate(shards):
        ids: dict[str, list[str]] = defaultdict(list)
        for row in shard["rows"]:
            ids[row["category"]].append(row["id"])
        payload = {
            category: sorted(ids[category])
            for category in CATEGORY_SECONDS
        }
        root = args.output_root / f"gpu{index}"
        root.mkdir(parents=True, exist_ok=False)
        (root / "test_case_ids_to_generate.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        plan_shards.append({
            "gpu": index,
            "estimated_wall_seconds": shard["estimated_wall_seconds"],
            "estimated_hours": shard["estimated_wall_seconds"] / 3600,
            "total_cases": len(shard["rows"]),
            "category_counts": {
                category: len(payload[category])
                for category in CATEGORY_SECONDS
            },
            "ids_path": str(root / "test_case_ids_to_generate.json"),
        })

    all_ids = [row["id"] for shard in shards for row in shard["rows"]]
    if len(all_ids) != 800 or len(set(all_ids)) != 800:
        raise RuntimeError("slice partition is not an exact 800-case partition")
    result = {
        "schema_version": "bfcl_latency_balanced_slices_v1",
        "source": str(args.prior_result_dir),
        "method": (
            "LPT over per-case prior latency, scaled to observed category wall time"
        ),
        "prior_category_wall_seconds": category_audit,
        "shards": plan_shards,
        "expected_makespan_seconds": max(
            shard["estimated_wall_seconds"] for shard in shards
        ),
        "expected_makespan_hours": max(
            shard["estimated_wall_seconds"] for shard in shards
        ) / 3600,
        "count": len(all_ids),
        "unique_ids": len(set(all_ids)),
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
