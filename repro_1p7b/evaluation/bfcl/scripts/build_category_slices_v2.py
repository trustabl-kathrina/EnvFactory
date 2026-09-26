#!/usr/bin/env python3
"""Create the fixed Dynamic-v2 BFCL split requested by the experiment owner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FILES = {
    "multi_turn_base": "BFCL_v3_multi_turn_base_result.json",
    "multi_turn_long_context": "BFCL_v3_multi_turn_long_context_result.json",
    "multi_turn_miss_func": "BFCL_v3_multi_turn_miss_func_result.json",
    "multi_turn_miss_param": "BFCL_v3_multi_turn_miss_param_result.json",
}
OBSERVED_SECONDS = {
    "multi_turn_base": 4708,
    "multi_turn_long_context": 13613,
    "multi_turn_miss_func": 5908,
    "multi_turn_miss_param": 398,
}


def ids(path: Path) -> list[str]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    result = sorted(row["id"] for row in rows)
    if len(result) != 200 or len(set(result)) != 200:
        raise RuntimeError(f"{path}: expected 200 unique IDs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-result-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    all_ids = {
        category: ids(args.prior_result_dir / name)
        for category, name in FILES.items()
    }
    payloads = (
        {"multi_turn_long_context": all_ids["multi_turn_long_context"]},
        {
            category: all_ids[category]
            for category in (
                "multi_turn_base",
                "multi_turn_miss_func",
                "multi_turn_miss_param",
            )
        },
    )
    shards = []
    for gpu, payload in enumerate(payloads):
        root = args.output_root / f"gpu{gpu}"
        root.mkdir(parents=True, exist_ok=False)
        path = root / "test_case_ids_to_generate.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        seconds = sum(OBSERVED_SECONDS[category] for category in payload)
        shards.append({
            "gpu": gpu,
            "categories": list(payload),
            "category_counts": {
                category: len(values) for category, values in payload.items()
            },
            "total_cases": sum(len(values) for values in payload.values()),
            "prior_observed_seconds": seconds,
            "prior_observed_hours": seconds / 3600,
            "ids_path": str(path),
        })
    flat = [item for payload in payloads for values in payload.values() for item in values]
    if len(flat) != 800 or len(set(flat)) != 800:
        raise RuntimeError("fixed split is not an exact 800-case partition")
    result = {
        "schema_version": "bfcl_fixed_category_slices_v2",
        "method": "GPU0 long_context; GPU1 base + miss_func + miss_param",
        "source": str(args.prior_result_dir),
        "shards": shards,
        "expected_makespan_seconds": max(x["prior_observed_seconds"] for x in shards),
        "expected_makespan_hours": max(x["prior_observed_hours"] for x in shards),
        "count": len(flat),
        "unique_ids": len(set(flat)),
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
