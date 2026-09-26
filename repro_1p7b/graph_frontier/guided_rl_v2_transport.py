"""Episode-level, rank-preserving reward transport for Guided RL v2.

The transport is deliberately group-aware.  Runtime task success is primary,
official EnvFactory reward is secondary, and the bounded Graph score is used
only inside a success/official equivalence class.  A trajectory contributes
one scalar to GRPO group statistics irrespective of its number of active
turns; the resulting scalar advantage is then attached to every active turn
for the unchanged token-level policy loss.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_TIE_TOLERANCE = 1e-6


def _official_bin(value: float, tolerance: float) -> int:
    return int(round(float(value) / tolerance))


def _unique_trajectory_indices(uids: Sequence[Any], traj_uids: Sequence[Any]) -> list[int]:
    seen = set()
    result = []
    for index, pair in enumerate(zip(uids, traj_uids)):
        if pair in seen:
            continue
        seen.add(pair)
        result.append(index)
    return result


def compute_group_signals(
    *,
    uids: Sequence[Any],
    traj_uids: Sequence[Any],
    success: Sequence[float],
    official_reward: Sequence[float],
    graph_score: Sequence[float],
    graph_available: Sequence[bool],
    mode: str,
    tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Build safe scalar group signals and broadcast them to active-turn rows.

    ``mode='official'`` uses the same success/official primary ordering but no
    Graph perturbation. ``mode='graph'`` permits Graph only inside an exact
    success/official class.  The perturbation is at most one quarter of the
    nearest primary-class gap, so crossing a strict primary class is
    impossible by construction.
    """

    if mode not in {"official", "graph"}:
        raise ValueError("mode must be official or graph")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    arrays = [uids, traj_uids, success, official_reward, graph_score, graph_available]
    sizes = {len(item) for item in arrays}
    if len(sizes) != 1:
        raise ValueError(f"transport inputs have different lengths: {sorted(sizes)}")

    first_indices = _unique_trajectory_indices(uids, traj_uids)
    trajectory_records: dict[tuple[Any, Any], dict[str, Any]] = {}
    grouped: dict[Any, list[tuple[Any, Any]]] = defaultdict(list)
    for index in first_indices:
        pair = (uids[index], traj_uids[index])
        record = {
            "uid": uids[index],
            "traj_uid": traj_uids[index],
            "success": int(float(success[index]) >= 0.5),
            "official_reward": float(official_reward[index]),
            "official_bin": _official_bin(float(official_reward[index]), tolerance),
            "graph_score": float(np.clip(float(graph_score[index]), -1.0, 1.0)),
            "graph_available": bool(graph_available[index]),
        }
        trajectory_records[pair] = record
        grouped[uids[index]].append(pair)

    for uid, pairs in grouped.items():
        records = [trajectory_records[pair] for pair in pairs]
        official_values = [record["official_reward"] for record in records]
        success_offset = max(official_values) - min(official_values) + 1.0
        classes: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            class_key = (record["success"], record["official_bin"])
            record["primary_class"] = (
                f"success={record['success']}|official_bin={record['official_bin']}"
            )
            record["primary_signal"] = (
                record["official_reward"] + record["success"] * success_offset
            )
            classes[class_key].append(record)

        primary_levels = sorted({record["primary_signal"] for record in records})
        positive_gaps = [
            right - left
            for left, right in zip(primary_levels, primary_levels[1:])
            if right - left > tolerance
        ]
        perturbation_scale = 0.25 * min(positive_gaps) if positive_gaps else 0.25

        for members in classes.values():
            effective = [record["graph_score"] if record["graph_available"] else 0.0 for record in members]
            varies = max(effective) - min(effective) > tolerance if len(effective) > 1 else False
            class_has_graph = any(record["graph_available"] for record in members)
            apply_graph = mode == "graph" and varies and class_has_graph
            for record, effective_graph in zip(members, effective):
                record["graph_tiebreak_eligible"] = bool(len(members) > 1 and record["graph_available"])
                record["graph_tiebreak_applied"] = bool(apply_graph)
                record["graph_perturbation"] = perturbation_scale * effective_graph if apply_graph else 0.0
                record["final_group_reward"] = record["primary_signal"] + record["graph_perturbation"]
                record["primary_gap_guard"] = perturbation_scale

        # Construction-time safety assertions.
        for left in records:
            for right in records:
                if left["success"] > right["success"]:
                    assert left["final_group_reward"] > right["final_group_reward"]
                elif left["success"] == right["success"] and (
                    left["official_reward"] > right["official_reward"] + tolerance
                ):
                    assert left["final_group_reward"] > right["final_group_reward"]

    row_records = [trajectory_records[(uid, traj_uid)] for uid, traj_uid in zip(uids, traj_uids)]
    return {
        "row_signals": np.asarray([record["final_group_reward"] for record in row_records], dtype=np.float32),
        "row_metadata": {
            key: np.asarray([record[key] for record in row_records])
            for key in (
                "primary_class",
                "primary_signal",
                "graph_tiebreak_eligible",
                "graph_tiebreak_applied",
                "graph_perturbation",
                "final_group_reward",
                "primary_gap_guard",
            )
        },
        "trajectory_records": list(trajectory_records.values()),
    }


def apply_rank_preserving_transport(
    *,
    reward_tensor,
    response_mask,
    non_tensor_batch: Mapping[str, Any],
    mode: str,
    tolerance: float = DEFAULT_TIE_TOLERANCE,
):
    """Replace repeated episode scores with safe per-trajectory group signals."""

    result = compute_group_signals(
        uids=non_tensor_batch["uid"],
        traj_uids=non_tensor_batch["traj_uid"],
        success=non_tensor_batch["episode_success"],
        official_reward=non_tensor_batch["episode_official_reward"],
        graph_score=non_tensor_batch["episode_graph_score"],
        graph_available=non_tensor_batch["episode_graph_available"],
        mode=mode,
        tolerance=tolerance,
    )
    transported = reward_tensor.new_zeros(reward_tensor.shape)
    for row_index, scalar in enumerate(result["row_signals"]):
        valid = response_mask[row_index].bool().nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            raise RuntimeError("cannot place episode reward on an empty response")
        transported[row_index, int(valid[-1])] = float(scalar)
    return transported, result["row_metadata"]


def audit_transport_batch(batch, *, train_step: int) -> list[dict[str, Any]]:
    """Return one audit record per trajectory after advantage computation."""

    mask = batch.batch["response_mask"].bool()
    advantages = batch.batch["advantages"]
    non_tensor = batch.non_tensor_batch
    first_indices = _unique_trajectory_indices(non_tensor["uid"], non_tensor["traj_uid"])
    rollout_indices: dict[Any, int] = defaultdict(int)
    records = []
    for index in first_indices:
        uid = non_tensor["uid"][index]
        traj_uid = non_tensor["traj_uid"][index]
        same_traj = [
            row
            for row, pair in enumerate(zip(non_tensor["uid"], non_tensor["traj_uid"]))
            if pair == (uid, traj_uid)
        ]
        valid_advantage = advantages[index][mask[index]]
        scalar_advantage = float(valid_advantage[0].detach().cpu()) if valid_advantage.numel() else None
        record = {
            "train_step": int(train_step),
            "prompt_id": str(non_tensor["episode_task_id"][index]),
            "task_id": str(non_tensor["episode_task_id"][index]),
            "group_uid": str(uid),
            "traj_uid": str(traj_uid),
            "rollout_index": int(rollout_indices[uid]),
            "success": float(non_tensor["episode_success"][index]),
            "official_reward": float(non_tensor["episode_official_reward"][index]),
            "eligible_consumer_opportunities": int(non_tensor["episode_eligible_consumer_opportunities"][index]),
            "consumer_completed": int(non_tensor["episode_consumer_completed"][index]),
            "consumer_score": float(non_tensor["episode_consumer_score"][index]),
            "eligible_prop_opportunities": int(non_tensor["episode_eligible_prop_opportunities"][index]),
            "prop_correct": int(non_tensor["episode_prop_correct"][index]),
            "prop_score": float(non_tensor["episode_prop_score"][index]),
            "retry_opportunities": int(non_tensor["episode_retry_opportunities"][index]),
            "invalid_retry": int(non_tensor["episode_invalid_retry"][index]),
            "retry_score": float(non_tensor["episode_retry_score"][index]),
            "graph_score": float(non_tensor["episode_graph_score"][index]),
            "graph_available": bool(non_tensor["episode_graph_available"][index]),
            "primary_class": str(non_tensor["primary_class"][index]),
            "graph_tiebreak_eligible": bool(non_tensor["graph_tiebreak_eligible"][index]),
            "graph_tiebreak_applied": bool(non_tensor["graph_tiebreak_applied"][index]),
            "final_group_reward": float(non_tensor["final_group_reward"][index]),
            "advantage": scalar_advantage,
            "trajectory_length": len(same_traj),
            "generated_tokens": int(sum(mask[row].sum().item() for row in same_traj)),
            "tool_call_count": float(non_tensor["tool_callings"][index]),
        }
        rollout_indices[uid] += 1
        records.append(record)
    return records


def print_transport_audit(batch, *, train_step: int) -> None:
    for record in audit_transport_batch(batch, train_step=train_step):
        print("GUIDED_RL_V2_AUDIT " + json.dumps(record, sort_keys=True), flush=True)
