"""Build a task-disjoint, depth-2/3 stop-targeted Rich-v3 plan.

Only new graph paths ending in retrieval/inspection or state-changing terminal
tools are selected. This is a bounded data-yield experiment, not a training set.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from repro_1p7b.graph_frontier.rich_generation_v3 import (
    ROOT, canonical, file_sha256, path_environment, profiler_unambiguous_path,
    read_paths, sha256_bytes, write_json, write_jsonl,
)

TERMINAL_HINTS = tuple(os.getenv("STOP_PLAN_HINTS", "get,retrieve,list,suspend,resolve").split(","))
SEED = int(os.getenv("STOP_PLAN_SEED", "20260927"))
PER_DEPTH = int(os.getenv("STOP_PLAN_PER_DEPTH", "32"))
TARGET = {3: PER_DEPTH, 2: PER_DEPTH}
MAX_TOOL = int(os.getenv("STOP_PLAN_MAX_TOOL", "4"))
MAX_ENV = int(os.getenv("STOP_PLAN_MAX_ENV", "8"))
MAX_EDGE_REUSE = int(os.getenv("STOP_PLAN_MAX_EDGE_REUSE", "2"))


def edge_ids(path: list[dict]) -> tuple[str, ...]:
    return tuple(str(edge["edge_id"]) for edge in path)


def select(paths: list[list[dict]], count: int, depth: int,
           edge_use: Counter[str], tool_use: Counter[str],
           env_use: Counter[str]) -> list[list[dict]]:
    buckets: dict[str, list[list[dict]]] = defaultdict(list)
    for path in paths:
        buckets[str(path[-1]["consumer_environment"])].append(path)
    for env, values in buckets.items():
        values.sort(key=lambda path: sha256_bytes(
            canonical([SEED, depth, env, edge_ids(path)]).encode()))
    envs = sorted(buckets, key=lambda env: sha256_bytes(canonical([SEED, env]).encode()))
    cursors = Counter()
    chosen: list[list[dict]] = []
    while len(chosen) < count:
        progressed = False
        for env in envs:
            if env_use[env] >= MAX_ENV:
                continue
            values = buckets[env]
            while cursors[env] < len(values):
                path = values[cursors[env]]
                cursors[env] += 1
                terminal = str(path[-1]["consumer_tool"])
                if tool_use[terminal] >= MAX_TOOL or any(edge_use[key] >= MAX_EDGE_REUSE for key in edge_ids(path)):
                    continue
                chosen.append(path)
                edge_use.update(edge_ids(path))
                tool_use[terminal] += 1
                env_use[env] += 1
                progressed = True
                break
            if len(chosen) == count:
                break
        if not progressed:
            raise ValueError(f"only {len(chosen)}/{count} eligible depth-{depth} paths")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/mcp_server.json")
    parser.add_argument("--exclude-plan", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError("stop-targeted plan already exists")
    old = [json.loads(line) for line in args.formal_plan.read_text().splitlines() if line.strip()]
    old_paths = {edge_ids(row["edge_path"]) for row in old}
    for exclusion in args.exclude_plan:
        old_paths.update(edge_ids(json.loads(line)["edge_path"])
                         for line in exclusion.read_text().splitlines() if line.strip())
    registered = set(json.loads(args.registry.read_text())["mcpServers"])
    edge_use: Counter[str] = Counter()
    tool_use: Counter[str] = Counter()
    env_use: Counter[str] = Counter()
    selected: list[dict] = []
    screened = {}
    for depth, count in TARGET.items():
        paths = read_paths(args.source_dir / "audit" / f"paths_depth{depth}.jsonl")
        eligible = [path for path in paths
                    if edge_ids(path) not in old_paths
                    and profiler_unambiguous_path(path)
                    and all(edge["producer_environment"] in registered
                            and edge["consumer_environment"] in registered for edge in path)
                    and any(hint in str(path[-1]["consumer_tool"]).lower()
                            for hint in TERMINAL_HINTS)]
        screened[str(depth)] = {"inventory_paths": len(paths), "eligible": len(eligible)}
        for index, path in enumerate(select(eligible, count, depth, edge_use, tool_use, env_use)):
            row = {"bucket": f"depth{depth}" if depth < 3 else "depth3plus",
                   "target_depth": depth, "edge_path": path,
                   "tool_sequence": [path[0]["producer_tool"]] +
                                    [edge["consumer_tool"] for edge in path],
                   "environment_identifiers": list(path_environment(path)),
                   "generation_seed": SEED + depth * 10_000_000 + index * 7919}
            row["plan_id"] = "gf-rich-" + sha256_bytes(canonical(row).encode())[:20]
            selected.append(row)
    if len(selected) != 2 * PER_DEPTH or len({row["plan_id"] for row in selected}) != 2 * PER_DEPTH:
        raise ValueError("stop-targeted plan count or identity failed")
    if any(edge_ids(row["edge_path"]) in old_paths for row in selected):
        raise ValueError("formal plan path overlap")
    write_jsonl(args.output, selected)
    summary = {"schema_version": "graph_frontier_rich_plan_v1",
               "name": args.output.stem, "requested": {"1": 0, "2": PER_DEPTH, "3": PER_DEPTH},
               "selected": {"2": PER_DEPTH, "3": PER_DEPTH}, "shortfalls": {"1": 0, "2": 0, "3": 0},
               "plans": len(selected), "unique_path_signatures": len(selected),
               "internal_edge_instances": sum(len(row["edge_path"]) for row in selected),
               "unique_inventory_edge_ids": len(edge_use),
               "max_inventory_edge_reuse_config": MAX_EDGE_REUSE,
               "max_inventory_edge_reuse_observed": max(edge_use.values()),
               "registered_environments": len(registered),
               "registry_filtered_path_candidates": {},
               "profiler_unambiguous_only": True,
               "profiler_ambiguous_path_candidates_filtered": {},
               "environment_distribution": dict(Counter(
                   env for row in selected for env in row["environment_identifiers"])),
               "terminal_tool_distribution": dict(tool_use),
               "terminal_environment_distribution": dict(env_use),
               "screened": screened, "terminal_hints": TERMINAL_HINTS,
               "formal_plan_sha256": file_sha256(args.formal_plan),
               "formal_path_overlap": 0, "selection_seed": SEED,
               "excluded_plan_sha256": [file_sha256(p) for p in args.exclude_plan]}
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
