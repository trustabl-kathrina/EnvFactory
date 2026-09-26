#!/usr/bin/env python3
"""Merge two BFCL run-id shards into a complete official-evaluation root."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

FILES = (
    "BFCL_v3_multi_turn_base_result.json",
    "BFCL_v3_multi_turn_long_context_result.json",
    "BFCL_v3_multi_turn_miss_func_result.json",
    "BFCL_v3_multi_turn_miss_param_result.json",
)


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sort_key(row: dict[str, Any]) -> tuple[str, int]:
    task_id = row["id"]
    prefix, suffix = task_id.rsplit("_", 1)
    return prefix, int(suffix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-dir", default="Qwen_Qwen3-1.7B-FC")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise RuntimeError(f"refusing existing output: {args.output_root}")
    output = args.output_root / "result" / args.model_dir
    output.mkdir(parents=True)
    summary = {}
    all_ids: set[str] = set()
    for name in FILES:
        merged = []
        for index in range(2):
            merged.extend(
                rows(
                    args.shard_root
                    / f"gpu{index}"
                    / "result"
                    / args.model_dir
                    / name
                )
            )
        ids = [row["id"] for row in merged]
        if len(merged) != 200 or len(set(ids)) != 200:
            raise RuntimeError(
                f"{name}: expected 200 unique rows, got {len(merged)}/{len(set(ids))}"
            )
        overlap = all_ids.intersection(ids)
        if overlap:
            raise RuntimeError(f"cross-category duplicate IDs: {sorted(overlap)[:5]}")
        all_ids.update(ids)
        merged.sort(key=sort_key)
        (output / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged)
        )
        summary[name] = {"count": len(merged), "unique": len(set(ids))}
    if len(all_ids) != 800:
        raise RuntimeError(f"expected 800 unique IDs, got {len(all_ids)}")
    shutil.copy2(args.plan, args.output_root / "slice_plan.json")
    report = {
        "status": "PASS",
        "count": 800,
        "unique_ids": len(all_ids),
        "files": summary,
    }
    (args.output_root / "merge_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
