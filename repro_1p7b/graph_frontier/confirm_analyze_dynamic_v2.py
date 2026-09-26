#!/usr/bin/env python3
"""Analyze Dynamic-v2 on the frozen 300-probe set without changing v1 reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from repro_1p7b.graph_frontier.confirm_analyze import (
    aggregate,
    load_manifest,
    metric_maps,
    paired_metric,
    task_rows_for_model,
)

LABELS = ("base", "original_sft", "parameter_aware", "dynamic_v1", "dynamic_v2")
COMPARATORS = ("original_sft", "parameter_aware", "dynamic_v1")
BFCL_DIR = {
    "base": "base",
    "original_sft": "sft",
    "parameter_aware": "parameter_aware",
    "dynamic_v1": "dynamic_v1",
    "dynamic_v2": "dynamic_v2",
}


def score(root: Path, label: str) -> dict[str, Any]:
    path = (
        root
        / BFCL_DIR[label]
        / "score"
        / "data_multi_turn.csv"
    )
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    def percent(name: str) -> float:
        return float(row[name].rstrip("%"))
    return {
        "status": "available",
        "path": str(path),
        "overall": percent("Multi Turn Overall Acc"),
        "base": percent("Base"),
        "miss_func": percent("Miss Func"),
        "miss_param": percent("Miss Param"),
        "long_context": percent("Long Context"),
    }


def pairs(task_maps: Mapping[str, Mapping[str, Mapping[str, Any]]], split=None):
    result = {}
    right = metric_maps(task_maps["dynamic_v2"], split=split)
    for left_label in COMPARATORS:
        left = metric_maps(task_maps[left_label], split=split)
        result[f"{left_label}_vs_dynamic_v2"] = {
            name: paired_metric(left[name], right[name])
            for name in (
                "semantic_task_success",
                "reference_path_complete_success",
                "task_level_internal_edge_complete",
            )
        }
    return result


def fmt(metric: Mapping[str, Any]) -> str:
    rate = metric.get("raw_rate")
    return (
        f"{metric.get('successes', 0)}/{metric.get('attempts', 0)}="
        + ("n/a" if rate is None else f"{100 * rate:.2f}%")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("repro_1p7b/results/graph_frontier/confirm_300"),
    )
    parser.add_argument(
        "--bfcl-root",
        type=Path,
        default=Path("repro_1p7b/evaluation/bfcl/artifacts/full"),
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=Path("repro_1p7b/graph_frontier/reports"),
    )
    args = parser.parse_args()

    _, manifests = load_manifest(args.root)
    task_maps = {}
    runtime = {}
    models = {}
    for label in LABELS:
        task_maps[label], runtime[label] = task_rows_for_model(
            args.root, manifests, label
        )
        models[label] = (
            aggregate(task_maps[label]) if task_maps[label] else {}
        )
    gate = all(runtime[label]["valid"] >= 285 for label in LABELS)
    report = {
        "schema_version": "graph_frontier_dynamic_v2_confirmation_v1",
        "validity_gate": {
            "minimum_valid_per_model": 285,
            "passed": gate,
        },
        "runtime": runtime,
        "bfcl": {label: score(args.bfcl_root, label) for label in LABELS},
        "models": models,
        "paired_statistics": pairs(task_maps) if gate else {},
        "split_paired_statistics": (
            {
                split: pairs(task_maps, split=split)
                for split in ("diagnosis", "heldout")
            }
            if gate else {}
        ),
        "dynamic_v2_target_metrics": (
            {
                "semantic": models["dynamic_v2"]["semantic_task_success"],
                "reference_path": models["dynamic_v2"][
                    "reference_path_complete_success"
                ],
                "internal_reach": models["dynamic_v2"]["internal_reach"],
                "conditional_propagation": models["dynamic_v2"][
                    "conditional_propagation_accuracy"
                ],
                "internal_end_to_end": models["dynamic_v2"][
                    "internal_end_to_end"
                ],
                "root_failure_attribution": models["dynamic_v2"][
                    "root_failure_attribution"
                ],
                "efficiency": models["dynamic_v2"]["efficiency"],
            }
            if models["dynamic_v2"] else {}
        ),
        "limitations": [
            "Task-level McNemar tests are primary.",
            "The held-out split has only 60 tasks and is directional.",
            "BFCL and executable probes test different capability surfaces.",
        ],
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    json_path = args.reports / "confirm_300_dynamic_v2_comparison.json"
    md_path = args.reports / "confirm_300_dynamic_v2_comparison.md"
    failure_path = args.reports / "confirm_300_dynamic_v2_failure_patterns.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    failures = {
        label: {
            "root_failure_attribution": models[label].get(
                "root_failure_attribution", {}
            ),
            "call_audit": models[label].get("call_audit", {}),
            "efficiency": models[label].get("efficiency", {}),
        }
        for label in LABELS
    }
    failure_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Dynamic Graph-Frontier v2 Confirmation",
        "",
        f"Validity gate: {gate}",
        "",
        "| Model | BFCL | Semantic | Reference | Reach | Conditional | Internal E2E |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        model = models[label]
        bfcl = report["bfcl"][label]
        bfcl_text = (
            f"{bfcl['overall']:.2f}%"
            if bfcl["status"] == "available"
            else "missing"
        )
        lines.append(
            f"| {label} | {bfcl_text} | "
            f"{fmt(model['semantic_task_success'])} | "
            f"{fmt(model['reference_path_complete_success'])} | "
            f"{fmt(model['internal_reach'])} | "
            f"{fmt(model['conditional_propagation_accuracy'])} | "
            f"{fmt(model['internal_end_to_end'])} |"
        )
    dynamic = models["dynamic_v2"]
    lines += [
        "",
        "## Dynamic v2 failure and efficiency",
        "",
        f"- Root failures: {dynamic.get('root_failure_attribution', {})}",
        f"- Efficiency: {dynamic.get('efficiency', {})}",
        "",
        "## Paired statistics",
        "",
        "See the JSON report for exact McNemar counts and p-values.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": "PASS" if gate else "FAIL",
        "report": str(json_path),
        "markdown": str(md_path),
        "failure_patterns": str(failure_path),
    }, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
