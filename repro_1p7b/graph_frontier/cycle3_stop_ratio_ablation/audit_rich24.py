"""Locked six-model Rich24 differential audit; writes only JSON/JSONL."""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

from repro_1p7b.graph_frontier.audit_rich_v3_trajectories import (
    analyze_model, canonical, load_inputs, read_json,
)
from repro_1p7b.graph_frontier.cycle3_pilot.audit_rich24 import bootstrap, mcnemar_exact
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_stop_ratio_ablation.prepare import OUT

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "repro_1p7b/logs/rich_v3_trajectory_differential_audit/per_task_diff.jsonl"
EOS1 = ROOT / "repro_1p7b/graph_frontier/cycle3_pilot/evaluation/rich24/audit/per_task.jsonl"
PLAN = ROOT / "repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_heldout_greedy_plan.json"
OUTPUT = OUT / "evaluation/rich24_ablation_audit"
LABELS = ("Dynamic-v1", "FD-only", "old 1:1", "EOS 1:1", "EOS 2:1", "EOS 4:1")


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite Rich24 ablation audit")
    plan, _, references, old_collections, _ = load_inputs()
    if sha256(PLAN) != "0cb86095c5675146b02bf1e5951d16f7c0016b8e1007e0b13e2a42edda69e38d":
        raise RuntimeError("Rich24 plan changed")
    historical = {r["task_id"]: r for r in (json.loads(line) for line in OLD.read_text().splitlines() if line.strip())}
    eos1 = {r["task_id"]: r for r in (json.loads(line) for line in EOS1.read_text().splitlines() if line.strip())}
    if set(historical) != set(plan["task_ids"]) or set(eos1) != set(plan["task_ids"]):
        raise RuntimeError("Rich24 historical task alignment failed")
    new_collections = {}
    for ratio in (2, 4):
        directory = OUT / f"evaluation/eos_{ratio}to1/rich24/trained"
        manifest = read_json(directory / "collection_manifest.json")
        if manifest["plan_sha256"] != sha256(PLAN) or manifest["task_count"] != 24:
            raise RuntimeError(f"Rich24 ratio {ratio} collection protocol mismatch")
        new_collections[ratio] = {task: read_json(directory / f"{task}.r0.json")
                                  for task in plan["task_ids"]}
    rows = []
    for index, task in enumerate(plan["task_ids"]):
        base = old_collections["Dynamic-v1"][task]
        models = {"Dynamic-v1": historical[task]["models"]["Dynamic-v1"],
                  "FD-only": historical[task]["models"]["FD-only"],
                  "old 1:1": historical[task]["models"]["v2-1to1"],
                  "EOS 1:1": eos1[task]["Cycle-3"]}
        for ratio in (2, 4):
            new = new_collections[ratio][task]
            if new["task_id"] != task or new["rollout_index"] != 0:
                raise RuntimeError(f"{task} ratio {ratio} rollout identity mismatch")
            if new["sampling_seed"] != plan["sampling_seed"] + index * 1009:
                raise RuntimeError(f"{task} ratio {ratio} sampling seed mismatch")
            for field in ("source_seed", "initial_prompt", "tools"):
                if canonical(new[field]) != canonical(base[field]):
                    raise RuntimeError(f"{task} ratio {ratio} {field} mismatch")
            if canonical(new["steps"][0]["state_before_action"]) != canonical(base["steps"][0]["state_before_action"]):
                raise RuntimeError(f"{task} ratio {ratio} initial state mismatch")
            official = new["terminal_info"]["reward_parts"]["official_reward"]
            if bool(new["semantic_success"]) != (official["trace_score"] == 1 and official["state_score"] == 1):
                raise RuntimeError(f"{task} ratio {ratio} semantic rule mismatch")
            models[f"EOS {ratio}:1"] = analyze_model(new, references[task], set())
        rows.append({"task_id": task, "environment": historical[task]["environment"],
                     "depth": historical[task]["depth"], "models": models})
    summary = {"schema_version": "cycle3_eos_ratio_rich24_audit_v1",
               "task_count": 24, "plan_sha256": sha256(PLAN),
               "historical_per_task_sha256": sha256(OLD), "eos1_per_task_sha256": sha256(EOS1),
               "models": {}, "paired": {}}
    for label in LABELS:
        items = [row["models"][label] for row in rows]
        steps = [x["first_failure_step"] for x in items if x["first_failure_step"] is not None]
        summary["models"][label] = {
            "official_reward_mean": statistics.mean(x["reward"] for x in items),
            "semantic_success": sum(bool(x["semantic_success"]) for x in items),
            "official_trace_success": sum(x["official_reward_parts"]["trace_score"] == 1 for x in items),
            "official_state_success": sum(x["official_reward_parts"]["state_score"] == 1 for x in items),
            "tool_events": sum(x["tool_efficiency"]["tool_calls_total"] for x in items),
            "repeated_exact_calls": sum(x["tool_efficiency"]["repeated_exact_calls"] for x in items),
            "extra_calls_over_gold_length": sum(x["tool_efficiency"]["calls_over_gold_length"] for x in items),
            "required_tool_matches": sum(x["matched_required_call_count"] for x in items),
            "correct_parameter_propagations": sum(x["graph_frontier"]["correct_propagations"] or 0 for x in items),
            "wrong_argument": sum(x["primary_failure"] == "WRONG_ARGUMENT" for x in items),
            "premature_stop": sum(x["primary_failure"] == "PREMATURE_STOP" for x in items),
            "primary_failure_counts": dict(collections.Counter(x["primary_failure"] for x in items)),
            "first_failure_step_mean": statistics.mean(steps) if steps else None,
            "first_failure_step_median": statistics.median(steps) if steps else None,
            "first_failure_step_distribution": dict(sorted(collections.Counter(steps).items())),
            "max_turn_exhaustion": sum(x["termination"]["step_count"] >= 8 for x in items),
        }
    for ratio in (2, 4):
        label = f"EOS {ratio}:1"
        for other in ("Dynamic-v1", "old 1:1", "EOS 1:1"):
            differences = [row["models"][label]["reward"] - row["models"][other]["reward"] for row in rows]
            fixed = sum(row["models"][other]["primary_failure"] == "WRONG_ARGUMENT"
                        and row["models"][label]["primary_failure"] != "WRONG_ARGUMENT" for row in rows)
            added = sum(row["models"][other]["primary_failure"] != "WRONG_ARGUMENT"
                        and row["models"][label]["primary_failure"] == "WRONG_ARGUMENT" for row in rows)
            summary["paired"][f"{label}_vs_{other}"] = {
                "reward_delta_mean": statistics.mean(differences),
                "reward_delta_bootstrap_95ci": bootstrap(differences, seed=20260926 + ratio),
                "better": sum(x > 1e-12 for x in differences),
                "same": sum(abs(x) <= 1e-12 for x in differences),
                "worse": sum(x < -1e-12 for x in differences),
                "wrong_argument_fixed": fixed, "wrong_argument_added": added,
                "wrong_argument_mcnemar_exact_p": mcnemar_exact(fixed, added)}
    output.mkdir(parents=True, exist_ok=False)
    (output / "per_task.jsonl").write_text("".join(stable(r) + "\n" for r in rows))
    (output / "summary.json").write_text(stable(summary) + "\n")
    return summary


if __name__ == "__main__":
    print(stable(run()), flush=True)
