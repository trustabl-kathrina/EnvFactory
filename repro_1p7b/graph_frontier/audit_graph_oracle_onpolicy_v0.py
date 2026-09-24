"""Audit existing Dynamic-v1 student rollouts; never create training labels from gold prefixes."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
from repro_1p7b.graph_frontier.graph_oracle_onpolicy_v0 import get_local_supervision

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROWS = ROOT / "repro_1p7b/data/graph_frontier_rl_v1_frozen_seed20260920/train.json"
DEFAULT_ROLLOUTS = ROOT / "repro_1p7b/graph_frontier/preference_runtime/dynamic_v1_train_v1"

def decoded(value):
    return json.loads(value) if isinstance(value, str) else value

def audit(rows_path: Path, rollout_dir: Path) -> dict:
    rows = {row["task_id"]: row for row in json.loads(rows_path.read_text())}
    counts = collections.Counter()
    reasons = collections.Counter()
    buckets = collections.Counter()
    depths = collections.Counter()
    task_ids = set()
    for path in sorted(rollout_dir.glob("*.r*.json")):
        rollout = json.loads(path.read_text())
        task_id = rollout.get("task_id")
        if task_id not in rows:
            counts["unknown_task_rollouts"] += 1
            continue
        row = rows[task_id]
        sidecar = row["extra_info"]["graph_frontier"]
        reference = decoded(row["reward_model"]["ground_truth"])
        history = []
        task_ids.add(task_id)
        counts["rollouts"] += 1
        for step in rollout.get("steps", []):
            counts["decision_states"] += 1
            action = step.get("parsed_action") or {}
            is_final = action.get("kind") == "final"
            # Final is state-preserving in EnvFactoryBatchEnv.step. Only then
            # may the post-final verifier certify the pre-final state.
            verifier = (bool(rollout.get("semantic_success")) if is_final and step.get("done")
                        else "unknown")
            result = get_local_supervision(
                sidecar=sidecar, history=history, reference_actions=reference,
                current_environment_state=step.get("state_before_action", "unknown"),
                current_verifier_success=verifier,
                certified_enabled_actions=None,
                replay_valid=True,
            )
            counts["tier_" + result["tier"]] += 1
            counts["action_" + str(action.get("kind", "unknown"))] += 1
            if result["target_type"] == "stop":
                counts["stop_targets"] += 1
            if result["target_type"] == "tool_call":
                counts["tool_targets"] += 1
            if result["target_type"] == "skip":
                counts["skipped"] += 1
            reasons[result["reason"]] += 1
            buckets[str(sidecar.get("target_bucket", "unknown"))] += 1
            depths[str(sidecar.get("dependency_depth", "unknown"))] += 1
            event = step.get("typed_event")
            if isinstance(event, dict):
                history.append(event)
    return {
        "schema_version": "graph_oracle_onpolicy_v0_coverage",
        "source": "existing_actual_Dynamic_v1_student_rollouts",
        "rows_path": str(rows_path),
        "rollout_dir": str(rollout_dir),
        "unique_tasks": len(task_ids),
        "counts": dict(sorted(counts.items())),
        "skip_reasons": dict(sorted(reasons.items())),
        "bucket_decision_counts": dict(sorted(buckets.items())),
        "depth_decision_counts": dict(sorted(depths.items())),
        "independent_replay_performed": False,
        "train_ready": False,
        "train_ready_reason": "no_exhaustive_action_oracle_or_verified_tool_argument_serialization",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.rows, args.rollouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
