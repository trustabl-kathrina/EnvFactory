"""Build and assemble an immutable replacement for a failed rich-v3 smoke.

The original successful chains are reused without changing their data. New
depth-three paths are selected before any additional model generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from repro_1p7b.graph_frontier.rich_generation_v3 import (
    canonical,
    file_sha256,
    path_environment,
    profiler_unambiguous_path,
    read_jsonl,
    read_paths,
    select_balanced,
    write_json,
    write_jsonl,
)


def path_signature(path: list[dict[str, Any]]) -> str:
    return canonical([edge["edge_id"] for edge in path])


def plan_summary(name: str, plans: list[dict[str, Any]], edge_limit: int, **extra: Any) -> dict[str, Any]:
    edge_counts = Counter(edge["edge_id"] for plan in plans for edge in plan["edge_path"])
    return {
        "schema_version": "graph_frontier_rich_plan_v1",
        "name": name,
        "plans": len(plans),
        "selected": dict(Counter(str(plan["target_depth"]) for plan in plans)),
        "internal_edge_instances": sum(len(plan["edge_path"]) for plan in plans),
        "unique_path_signatures": len({path_signature(plan["edge_path"]) for plan in plans}),
        "unique_inventory_edge_ids": len(edge_counts),
        "max_inventory_edge_reuse_config": edge_limit,
        "max_inventory_edge_reuse_observed": max(edge_counts.values(), default=0),
        "environment_distribution": dict(Counter(
            env for plan in plans for env in plan["environment_identifiers"]
        )),
        **extra,
    }


def build(args: argparse.Namespace) -> None:
    if args.combined_plan.exists() or args.supplemental_plan.exists():
        raise FileExistsError("refusing to overwrite a plan")
    original = read_jsonl(args.original_plan)
    ledger = {row["task_id"]: row for row in read_jsonl(args.original_ledger)}
    if len(original) != 120 or Counter(plan["target_depth"] for plan in original) != {1: 40, 2: 40, 3: 40}:
        raise ValueError("original smoke must contain exactly 40 plans per depth")
    if set(ledger) != {plan["plan_id"] for plan in original}:
        raise ValueError("original ledger and plan disagree")
    kept = [
        plan for plan in original
        if plan["target_depth"] < 3 or ledger[plan["plan_id"]]["status"] == "FINAL_VALID"
    ]
    missing = 40 - sum(plan["target_depth"] == 3 for plan in kept)
    if missing <= 0:
        raise ValueError("depth-three gate is already satisfied by original plan")
    edge_counts = Counter(edge["edge_id"] for plan in kept for edge in plan["edge_path"])
    used_paths = {path_signature(plan["edge_path"]) for plan in original}
    registry = json.loads(args.registry.read_text())["mcpServers"]
    candidates = [
        path for path in read_paths(args.inventory)
        if path_signature(path) not in used_paths
        and profiler_unambiguous_path(path)
        and len(path_environment(path)) <= args.max_environments
        and all(
            edge["producer_environment"] in registry and edge["consumer_environment"] in registry
            for edge in path
        )
    ]
    selected = select_balanced(
        candidates, missing, args.seed, edge_counts, args.max_edge_reuse
    )
    if len(selected) != missing:
        raise ValueError(f"only {len(selected)}/{missing} replacement paths available")
    supplemental = []
    for index, path in enumerate(selected):
        plan = {
            "bucket": "depth3plus",
            "target_depth": 3,
            "edge_path": path,
            "tool_sequence": [path[0]["producer_tool"]] + [edge["consumer_tool"] for edge in path],
            "environment_identifiers": list(path_environment(path)),
            "generation_seed": args.seed + 30_000_000 + index * 7919,
        }
        plan["plan_id"] = "gf-rich-" + hashlib.sha256(canonical(plan).encode()).hexdigest()[:20]
        supplemental.append(plan)
    combined = [plan for plan in kept if plan["target_depth"] == 3] + supplemental
    combined.extend(plan for plan in kept if plan["target_depth"] < 3)
    if Counter(plan["target_depth"] for plan in combined) != {1: 40, 2: 40, 3: 40}:
        raise AssertionError("replacement plan is not 40x3")
    if len({plan["plan_id"] for plan in combined}) != len(combined):
        raise AssertionError("duplicate task id")
    if len({plan["generation_seed"] for plan in combined}) != len(combined):
        raise AssertionError("duplicate seed")
    if max(edge_counts.values()) > args.max_edge_reuse:
        raise AssertionError("inventory edge reuse exceeded")
    args.combined_plan.parent.mkdir(parents=True, exist_ok=True)
    args.supplemental_plan.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.combined_plan, combined)
    write_jsonl(args.supplemental_plan, supplemental)
    lineage = {
        "original_plan": str(args.original_plan),
        "original_plan_sha256": file_sha256(args.original_plan),
        "original_ledger": str(args.original_ledger),
        "original_ledger_sha256": file_sha256(args.original_ledger),
        "reused_original_plans": len(kept),
        "reused_original_depth3_valid": 40 - missing,
        "new_depth3_plans": missing,
        "new_depth3_max_environments": args.max_environments,
        "selection_seed": args.seed,
        "requested": {"1": 40, "2": 40, "3": 40},
    }
    for path, plans in ((args.combined_plan, combined), (args.supplemental_plan, supplemental)):
        summary = plan_summary(path.stem, plans, args.max_edge_reuse, **lineage)
        write_json(path.with_suffix(".summary.json"), summary)
        print(json.dumps({"path": str(path), "plans": len(plans), "selected": summary["selected"]}))


def source_files(run_dir: Path) -> tuple[dict[int, Path], dict[str, Path]]:
    raw = {}
    for path in (run_dir / "raw").glob("*.json"):
        seed = int(json.loads(path.read_text())["seed"])
        if seed in raw:
            raise ValueError(f"duplicate raw seed {seed}")
        raw[seed] = path
    sidecars = {}
    for path in (run_dir / "sidecar").glob("*.gold.json"):
        task_id = str(json.loads(path.read_text())["task_id"])
        if task_id in sidecars:
            raise ValueError(f"duplicate sidecar task id {task_id}")
        sidecars[task_id] = path
    return raw, sidecars


def assemble(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite assembled smoke")
    plans = read_jsonl(args.combined_plan)
    if Counter(plan["target_depth"] for plan in plans) != {1: 40, 2: 40, 3: 40}:
        raise ValueError("assembled plan must contain exactly 40 tasks per depth")
    old_ledger = {row["task_id"]: row for row in read_jsonl(args.original_run / "audit/final_ledger.jsonl")}
    new_ledger = {row["task_id"]: row for row in read_jsonl(args.supplemental_run / "audit/final_ledger.jsonl")}
    if (args.original_run / "audit_status").read_text().strip() != "AUDIT_DONE":
        raise ValueError("original audit incomplete")
    if (args.supplemental_run / "audit_status").read_text().strip() != "AUDIT_DONE":
        raise ValueError("supplemental audit incomplete")
    old_raw, old_sidecars = source_files(args.original_run)
    new_raw, new_sidecars = source_files(args.supplemental_run)
    copies = []
    for plan in plans:
        task_id = plan["plan_id"]
        if task_id in old_ledger:
            record, raw, sidecars, source = old_ledger[task_id], old_raw, old_sidecars, args.original_run
        else:
            record, raw, sidecars, source = new_ledger[task_id], new_raw, new_sidecars, args.supplemental_run
        if record["status"] != "FINAL_VALID":
            continue
        seed = int(plan["generation_seed"])
        copies.append((raw[seed], sidecars[task_id], source, task_id))
    (args.output_dir / "raw").mkdir(parents=True)
    (args.output_dir / "sidecar").mkdir()
    lineage = []
    for raw_path, sidecar_path, source, task_id in copies:
        shutil.copy2(raw_path, args.output_dir / "raw" / raw_path.name)
        shutil.copy2(sidecar_path, args.output_dir / "sidecar" / sidecar_path.name)
        lineage.append({"task_id": task_id, "source_run": str(source)})
    write_json(args.output_dir / "assembly_manifest.json", {
        "schema_version": "graph_frontier_rich_smoke_assembly_v1",
        "plan": str(args.combined_plan),
        "plan_sha256": file_sha256(args.combined_plan),
        "original_run": str(args.original_run),
        "supplemental_run": str(args.supplemental_run),
        "retained_copied": len(copies),
        "lineage": lineage,
    })
    (args.output_dir / "status").write_text("GENERATED\n")
    print(json.dumps({"output_dir": str(args.output_dir), "retained_copied": len(copies)}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--original-plan", type=Path, required=True)
    build_parser.add_argument("--original-ledger", type=Path, required=True)
    build_parser.add_argument("--inventory", type=Path, required=True)
    build_parser.add_argument("--registry", type=Path, required=True)
    build_parser.add_argument("--combined-plan", type=Path, required=True)
    build_parser.add_argument("--supplemental-plan", type=Path, required=True)
    build_parser.add_argument("--seed", type=int, default=20260925)
    build_parser.add_argument("--max-environments", type=int, default=2)
    build_parser.add_argument("--max-edge-reuse", type=int, default=2)
    assemble_parser = sub.add_parser("assemble")
    assemble_parser.add_argument("--combined-plan", type=Path, required=True)
    assemble_parser.add_argument("--original-run", type=Path, required=True)
    assemble_parser.add_argument("--supplemental-run", type=Path, required=True)
    assemble_parser.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "build":
        build(args)
    else:
        assemble(args)
