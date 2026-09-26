"""CPU-only locked Rich24 differential audit; outputs JSON, never Markdown."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from repro_1p7b.graph_frontier.audit_rich_v3_trajectories import (
    analyze_model, canonical, load_inputs, read_json,
)

ROOT = Path(__file__).resolve().parents[3]
OLD_AUDIT = ROOT / "repro_1p7b/logs/rich_v3_trajectory_differential_audit"
TRAINED = ROOT / "repro_1p7b/graph_frontier/cycle3_pilot/evaluation/rich24/trained"
OUTPUT = ROOT / "repro_1p7b/graph_frontier/cycle3_pilot/evaluation/rich24/audit"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(deltas: list[float], seed: int = 20260926) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(deltas, k=len(deltas)))
                   for _ in range(10000))
    return means[250], means[9749]


def mcnemar_exact(changed_a: int, changed_b: int) -> float:
    n = changed_a + changed_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(changed_a, changed_b) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def run(trained: Path, output: Path) -> dict:
    plan, _, references, old_collections, _ = load_inputs()
    manifest = read_json(trained / "collection_manifest.json")
    if manifest["plan_sha256"] != sha(ROOT / "repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_heldout_greedy_plan.json"):
        raise RuntimeError("Cycle-3 Rich24 plan hash mismatch")
    if manifest["task_count"] != 24:
        raise RuntimeError("Cycle-3 Rich24 task count mismatch")
    historical = {json.loads(line)["task_id"]: json.loads(line)
                  for line in (OLD_AUDIT / "per_task_diff.jsonl").read_text().splitlines()
                  if line.strip()}
    if set(historical) != set(plan["task_ids"]):
        raise RuntimeError("historical Rich24 task identity mismatch")
    rows = []
    for index, task in enumerate(plan["task_ids"]):
        new = read_json(trained / f"{task}.r0.json")
        base = old_collections["Dynamic-v1"][task]
        if new["task_id"] != task or new["rollout_index"] != 0:
            raise RuntimeError(f"{task} rollout identity mismatch")
        if new["sampling_seed"] != plan["sampling_seed"] + index * 1009:
            raise RuntimeError(f"{task} sampling seed mismatch")
        for key in ("source_seed", "initial_prompt", "tools"):
            if canonical(new[key]) != canonical(base[key]):
                raise RuntimeError(f"{task} {key} mismatch")
        if canonical(new["steps"][0]["state_before_action"]) != canonical(base["steps"][0]["state_before_action"]):
            raise RuntimeError(f"{task} initial state mismatch")
        official = new["terminal_info"]["reward_parts"]["official_reward"]
        if bool(new["semantic_success"]) != (official["trace_score"] == 1 and official["state_score"] == 1):
            raise RuntimeError(f"{task} semantic rule mismatch")
        item = analyze_model(new, references[task], set())
        rows.append({"task_id": task, "environment": historical[task]["environment"],
                     "depth": historical[task]["depth"], "Cycle-3": item,
                     "Dynamic-v1": historical[task]["models"]["Dynamic-v1"],
                     "FD-only": historical[task]["models"]["FD-only"],
                     "v2-1to1": historical[task]["models"]["v2-1to1"]})
    summary = {"schema_version": "cycle3_rich24_differential_audit_v1",
               "task_count": 24, "plan_sha256": manifest["plan_sha256"],
               "trained_collection_manifest_sha256": sha(trained / "collection_manifest.json"),
               "historical_audit_sha256": sha(OLD_AUDIT / "per_task_diff.jsonl"),
               "models": {}, "paired": {}}
    for name in ("Dynamic-v1", "FD-only", "v2-1to1", "Cycle-3"):
        items = [r[name] for r in rows]
        steps = [x["first_failure_step"] for x in items if x["first_failure_step"] is not None]
        reference_steps = [x["first_reference_divergence_step"] for x in items
                           if x["first_reference_divergence_step"] is not None]
        summary["models"][name] = {
            "reward_mean": statistics.mean(x["reward"] for x in items),
            "semantic_success": sum(bool(x["semantic_success"]) for x in items),
            "official_trace_success": sum(x["official_reward_parts"]["trace_score"] == 1 for x in items),
            "official_state_success": sum(x["official_reward_parts"]["state_score"] == 1 for x in items),
            "tool_events": sum(x["tool_efficiency"]["tool_calls_total"] for x in items),
            "repeated_exact_calls": sum(x["tool_efficiency"]["repeated_exact_calls"] for x in items),
            "extra_calls_over_gold_length": sum(x["tool_efficiency"]["calls_over_gold_length"] for x in items),
            "required_calls_matched": sum(x["matched_required_call_count"] for x in items),
            "correct_propagations": sum(x["graph_frontier"]["correct_propagations"] or 0 for x in items),
            "primary_failure_counts": dict(collections.Counter(x["primary_failure"] for x in items)),
            "first_failure_step_mean": statistics.mean(steps) if steps else None,
            "first_failure_step_median": statistics.median(steps) if steps else None,
            "first_failure_step_distribution": dict(sorted(collections.Counter(steps).items())),
            "first_reference_divergence_step_mean": statistics.mean(reference_steps) if reference_steps else None,
            "final_action_count": sum(x["termination"]["last_action_kind"] == "final" for x in items),
            "max_turn_exhaustion": sum(x["termination"]["step_count"] >= 8 for x in items),
            "premature_stop": sum(x["primary_failure"] == "PREMATURE_STOP" for x in items),
            "wrong_argument": sum(x["primary_failure"] == "WRONG_ARGUMENT" for x in items),
        }
    for comparator in ("Dynamic-v1", "v2-1to1"):
        deltas = [r["Cycle-3"]["reward"] - r[comparator]["reward"] for r in rows]
        improved = sum(bool(r["Cycle-3"]["semantic_success"]) and not bool(r[comparator]["semantic_success"]) for r in rows)
        regressed = sum(not bool(r["Cycle-3"]["semantic_success"]) and bool(r[comparator]["semantic_success"]) for r in rows)
        wrong_fixed = sum(r[comparator]["primary_failure"] == "WRONG_ARGUMENT" and r["Cycle-3"]["primary_failure"] != "WRONG_ARGUMENT" for r in rows)
        wrong_added = sum(r[comparator]["primary_failure"] != "WRONG_ARGUMENT" and r["Cycle-3"]["primary_failure"] == "WRONG_ARGUMENT" for r in rows)
        summary["paired"][f"Cycle-3_vs_{comparator}"] = {
            "reward_delta_mean": statistics.mean(deltas),
            "reward_delta_bootstrap_95ci": bootstrap(deltas),
            "better": sum(d > 1e-12 for d in deltas),
            "same": sum(abs(d) <= 1e-12 for d in deltas),
            "worse": sum(d < -1e-12 for d in deltas),
            "semantic_improved": improved, "semantic_regressed": regressed,
            "semantic_mcnemar_exact_p": mcnemar_exact(improved, regressed),
            "wrong_argument_fixed": wrong_fixed, "wrong_argument_added": wrong_added,
            "wrong_argument_mcnemar_exact_p": mcnemar_exact(wrong_fixed, wrong_added),
        }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "per_task.jsonl").open("w", encoding="utf-8") as stream:
        for item in rows:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained", type=Path, default=TRAINED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.trained, args.output), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
