"""Offline v2 counterfactual over the 512 saved Graph-v1 rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repro_1p7b.graph_frontier.guided_rl_v2_transport import compute_group_signals


EPS = 1e-12


def retry_opportunities(events):
    opportunities = 0
    for previous, current in zip(events, events[1:]):
        state_known = previous.get("state_after", "unknown") != "unknown" and current.get("state_before", "unknown") != "unknown"
        no_state_change = state_known and json.dumps(previous["state_after"], sort_keys=True) == json.dumps(current["state_before"], sort_keys=True)
        opportunities += int(previous.get("execution_success") is False and no_state_change)
    return opportunities


def v2_graph_parts(payload):
    graph = payload["reward_parts"]["graph_frontier"]
    events = payload["trace"].get("events") or []
    consumer_opp = int(graph.get("consumer_opportunities", 0) or 0)
    consumer_done = int(graph.get("completed_consumer_opportunities", 0) or 0)
    prop_opp = int(graph.get("propagation_checks", 0) or 0)
    prop_correct = int(graph.get("correct_propagations", 0) or 0)
    retry_opp = retry_opportunities(events)
    invalid = int(graph.get("invalid_retry_count", 0) or 0)
    consumer = consumer_done / consumer_opp if consumer_opp else 0.0
    prop = prop_correct / prop_opp if prop_opp else 0.0
    retry = -(invalid / retry_opp) if retry_opp else 0.0
    score = float(np.clip(0.30 * prop + 0.40 * consumer + 0.10 * retry, -1.0, 1.0))
    return {
        "eligible_consumer_opportunities": consumer_opp,
        "consumer_completed": consumer_done,
        "consumer_score": consumer,
        "eligible_prop_opportunities": prop_opp,
        "prop_correct": prop_correct,
        "prop_score": prop,
        "retry_opportunities": retry_opp,
        "invalid_retry": invalid,
        "retry_score": retry,
        "graph_score": score,
        "graph_available": bool(consumer_opp or prop_opp or retry_opp),
    }


def sign(value):
    return int(value > EPS) - int(value < -EPS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/u2024311031/workspace/envfactory_repro_1p7b"))
    args = parser.parse_args()
    report_dir = args.root / "repro_1p7b/graph_frontier/reports"
    audit = pd.read_parquet(report_dir / "guided_rl_v1_rollout_audit.parquet")
    frame = audit[audit.experiment.eq("graph")].copy().sort_values(["train_step", "task_id", "rollout_index"])
    parts = {}
    for row in frame.itertuples():
        payload = json.loads((args.root / row.rollout_file).read_text())
        parts[(row.group_id, int(row.rollout_index))] = v2_graph_parts(payload)

    records = []
    strict_inversions = success_inversions = harmful = 0
    applied_groups = 0
    for group_id, group in frame.groupby("group_id", sort=True):
        group = group.sort_values("rollout_index")
        result = compute_group_signals(
            uids=[group_id] * len(group),
            traj_uids=[str(value) for value in group.rollout_index],
            success=group.official_success.astype(float).tolist(),
            official_reward=group.official_reward.tolist(),
            graph_score=[parts[(group_id, int(index))]["graph_score"] for index in group.rollout_index],
            graph_available=[parts[(group_id, int(index))]["graph_available"] for index in group.rollout_index],
            mode="graph",
            tolerance=1e-6,
        )["trajectory_records"]
        result = sorted(result, key=lambda item: int(item["traj_uid"]))
        signals = np.asarray([item["final_group_reward"] for item in result], dtype=float)
        std = signals.std(ddof=1)
        advantages = (signals - signals.mean()) / (std + 1e-6) if std > 0 else np.zeros_like(signals)
        applied_groups += int(any(item["graph_tiebreak_applied"] for item in result))
        for left in range(len(result)):
            for right in range(left + 1, len(result)):
                a, b = result[left], result[right]
                final_rel = sign(a["final_group_reward"] - b["final_group_reward"])
                official_rel = sign(a["official_reward"] - b["official_reward"])
                if official_rel * final_rel == -1:
                    strict_inversions += 1
                if a["success"] != b["success"]:
                    success_row = a if a["success"] else b
                    failure_row = b if a["success"] else a
                    if failure_row["final_group_reward"] > success_row["final_group_reward"] + EPS:
                        success_inversions += 1
                primary_rel = sign((a["success"] - b["success"]) or (a["official_reward"] - b["official_reward"]))
                if primary_rel * final_rel == -1:
                    harmful += 1
        for position, (source, transported, advantage) in enumerate(zip(group.to_dict("records"), result, advantages)):
            extra = parts[(group_id, int(source["rollout_index"]))]
            records.append({
                "group_id": group_id,
                "train_step": int(source["train_step"]),
                "task_id": source["task_id"],
                "rollout_index": int(source["rollout_index"]),
                "success": bool(source["official_success"]),
                "official_reward": float(source["official_reward"]),
                **extra,
                "primary_class": transported["primary_class"],
                "graph_tiebreak_eligible": bool(transported["graph_tiebreak_eligible"]),
                "graph_tiebreak_applied": bool(transported["graph_tiebreak_applied"]),
                "final_group_reward": float(transported["final_group_reward"]),
                "episode_level_advantage": float(advantage),
            })

    output = pd.DataFrame(records)
    step53 = output[(output.train_step == 53) & output.task_id.eq("gf-rl-v1-s966029525-t0")].to_dict("records")
    official_values = np.sort(frame.official_reward.unique())
    meaningful_gaps = np.diff(np.unique(np.round(official_values, 6)))
    report = {
        "schema_version": "guided_rl_v2_offline_replay_v1",
        "source_rollouts": 512,
        "official_reward_distribution": {
            "unique_serialized_values": official_values.tolist(),
            "minimum_meaningful_gap": float(meaningful_gaps[meaningful_gaps > 0].min()),
            "tie_tolerance": 1e-6,
        },
        "safety": {
            "harmful_inversion_pairs": harmful,
            "success_over_failure_inversion_pairs": success_inversions,
            "strict_official_inversion_pairs": strict_inversions,
        },
        "coverage": {
            "graph_available_rollouts": int(output.graph_available.sum()),
            "graph_tiebreak_eligible_rollouts": int(output.graph_tiebreak_eligible.sum()),
            "graph_tiebreak_applied_rollouts": int(output.graph_tiebreak_applied.sum()),
            "graph_tiebreak_applied_groups": applied_groups,
        },
        "step53": step53,
        "step53_summary": {
            "old_short_failure_advantage": -3.3281840345825304,
            "old_long_failure_advantage": 0.2773486698459244,
            "v2_short_failure_advantage": min(row["episode_level_advantage"] for row in step53),
            "v2_long_failure_advantage": max(row["episode_level_advantage"] for row in step53),
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "guided_rl_v2_offline_replay.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.to_csv(report_dir / "guided_rl_v2_offline_replay.csv", index=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
