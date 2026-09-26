from __future__ import annotations

import numpy as np
import torch

from repro_1p7b.graph_frontier.guided_rl_v2_transport import (
    apply_rank_preserving_transport,
    compute_group_signals,
)
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def _runtime_advantages(lengths, rewards):
    uids = []
    traj_uids = []
    for trajectory, length in enumerate(lengths):
        uids.extend(["prompt"] * length)
        traj_uids.extend([f"traj-{trajectory}"] * length)
    rows = len(uids)
    response_mask = torch.ones((rows, 1), dtype=torch.float32)
    reward_tensor = torch.zeros((rows, 1), dtype=torch.float32)
    official = []
    for reward, length in zip(rewards, lengths):
        official.extend([reward] * length)
    metadata = {
        "uid": np.asarray(uids, dtype=object),
        "traj_uid": np.asarray(traj_uids, dtype=object),
        "episode_success": np.zeros(rows),
        "episode_official_reward": np.asarray(official),
        "episode_graph_score": np.zeros(rows),
        "episode_graph_available": np.zeros(rows, dtype=bool),
    }
    transported, _ = apply_rank_preserving_transport(
        reward_tensor=reward_tensor,
        response_mask=response_mask,
        non_tensor_batch=metadata,
        mode="official",
    )
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=transported,
        response_mask=response_mask,
        index=metadata["uid"],
        traj_index=metadata["traj_uid"],
        compute_mean_std_cross_steps=False,
    )
    return [float(advantages[sum(lengths[:index]), 0]) for index in range(len(lengths))]


def test_same_reward_different_trajectory_length_has_same_episode_advantage():
    short_long = _runtime_advantages([1, 4], [0.7, 0.7])
    equal_length = _runtime_advantages([1, 1], [0.7, 0.7])
    assert np.allclose(short_long, [0.0, 0.0], atol=1e-6)
    assert np.allclose(short_long, equal_length, atol=1e-6)


def test_zero_vs_positive_is_not_multiplied_by_longer_trajectory():
    short_long = _runtime_advantages([1, 4], [0.0, 0.7])
    equal_length = _runtime_advantages([1, 1], [0.0, 0.7])
    assert short_long[1] > short_long[0]
    assert np.allclose(short_long, equal_length, atol=1e-6)


def test_all_equal_group_is_neutral():
    result = _runtime_advantages([1, 2, 3, 4], [0.7, 0.7, 0.7, 0.7])
    assert np.allclose(result, [0.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_graph_never_crosses_success_or_strict_official_class():
    result = compute_group_signals(
        uids=["p"] * 4,
        traj_uids=["a", "b", "c", "d"],
        success=[1, 0, 0, 0],
        official_reward=[0.0, 1.0, 0.5, 0.5],
        graph_score=[-1.0, 1.0, -1.0, 1.0],
        graph_available=[True] * 4,
        mode="graph",
    )["trajectory_records"]
    by_id = {row["traj_uid"]: row for row in result}
    assert by_id["a"]["final_group_reward"] > by_id["b"]["final_group_reward"]
    assert by_id["b"]["final_group_reward"] > by_id["c"]["final_group_reward"]
    assert by_id["b"]["final_group_reward"] > by_id["d"]["final_group_reward"]
    assert by_id["d"]["final_group_reward"] > by_id["c"]["final_group_reward"]


def test_float_serialization_noise_is_an_exact_tie_at_one_e_minus_six():
    result = compute_group_signals(
        uids=["p", "p"],
        traj_uids=["a", "b"],
        success=[0, 0],
        official_reward=[0.9, 0.8999999999999999],
        graph_score=[0.0, 0.7],
        graph_available=[False, True],
        mode="graph",
        tolerance=1e-6,
    )["trajectory_records"]
    assert result[0]["primary_class"] == result[1]["primary_class"]
    assert result[1]["final_group_reward"] > result[0]["final_group_reward"]
