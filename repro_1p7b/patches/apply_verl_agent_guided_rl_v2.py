"""Apply the minimal Guided RL v2 transport hooks to the project-only verl-agent.

The patch is intentionally limited to trajectory metadata transport and the
existing pre-advantage extension point. It does not touch PPO ratios, clipping,
KL, optimizer, FSDP, checkpointing, or rollout generation.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MARKER = "guided_rl_v2_transport"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_rollout_loop(path: Path) -> bool:
    text = path.read_text()
    if "episode_graph_available" in text and "trajectory_metadata" in text:
        return False

    old_summary = '''        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
'''
    new_summary = '''        # Keep legacy batch-level success-rate logging, but preserve v2
        # per-trajectory reward metadata without averaging it across the group.
        success_rate = {
            key: np.mean(value)
            for key, value in success.items()
            if "success_rate" in key
        }
        trajectory_metadata = {
            key: value
            for key, value in success.items()
            if "success_rate" not in key
        }

        effective_batch = []
'''
    text = replace_once(text, old_summary, new_summary, "rollout summary split")

    old_attach = '''                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
'''
    new_attach = '''                    for key, value in success_rate.items():
                        data[key] = value
                    for key, value in trajectory_metadata.items():
                        data[key] = value[bs]

                    effective_batch.append(data)
'''
    text = replace_once(text, old_attach, new_attach, "trajectory metadata attach")

    old_success = '''        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
'''
    new_success = '''        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )

        # Guided RL v2 needs the terminal decomposition at the group-aware
        # transport hook. Pull only structured fields from the final active
        # environment info; no response prose is reparsed.
        final_infos = []
        for bs in range(batch_size):
            final_info = None
            for turn in reversed(range(len(total_batch_list[bs]))):
                if total_batch_list[bs][turn]['active_masks']:
                    final_info = total_infos[bs][turn]
                    break
            if final_info is None or not isinstance(final_info.get('reward_parts'), dict):
                raise RuntimeError('Guided RL v2 terminal reward metadata is missing')
            final_infos.append(final_info)

        def terminal_values(extractor, dtype=None):
            values = [extractor(info, info['reward_parts'], info['reward_parts']['graph_v2']) for info in final_infos]
            return np.asarray(values, dtype=dtype)

        success['episode_task_id'] = terminal_values(lambda info, parts, graph: info['task_id'], object)
        success['episode_success'] = terminal_values(lambda info, parts, graph: info['won'], np.float32)
        success['episode_official_reward'] = terminal_values(
            lambda info, parts, graph: parts['official_reward']['score'], np.float32
        )
        success['episode_eligible_consumer_opportunities'] = terminal_values(
            lambda info, parts, graph: graph['eligible_consumer_opportunities'], np.int64
        )
        success['episode_consumer_completed'] = terminal_values(
            lambda info, parts, graph: graph['consumer_completed'], np.int64
        )
        success['episode_consumer_score'] = terminal_values(
            lambda info, parts, graph: graph['consumer_score'], np.float32
        )
        success['episode_eligible_prop_opportunities'] = terminal_values(
            lambda info, parts, graph: graph['eligible_prop_opportunities'], np.int64
        )
        success['episode_prop_correct'] = terminal_values(
            lambda info, parts, graph: graph['prop_correct'], np.int64
        )
        success['episode_prop_score'] = terminal_values(
            lambda info, parts, graph: graph['prop_score'], np.float32
        )
        success['episode_retry_opportunities'] = terminal_values(
            lambda info, parts, graph: graph['retry_opportunities'], np.int64
        )
        success['episode_invalid_retry'] = terminal_values(
            lambda info, parts, graph: graph['invalid_retry'], np.int64
        )
        success['episode_retry_score'] = terminal_values(
            lambda info, parts, graph: graph['retry_score'], np.float32
        )
        success['episode_graph_score'] = terminal_values(
            lambda info, parts, graph: graph['graph_score'], np.float32
        )
        success['episode_graph_available'] = terminal_values(
            lambda info, parts, graph: graph['graph_available'], bool
        )

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
'''
    text = replace_once(text, old_success, new_success, "terminal metadata extraction")
    path.write_text(text)
    return True


def patch_ray_trainer(path: Path) -> bool:
    text = path.read_text()
    if "guided_rl_v2_transport =" in text and "compute_mean_std_cross_steps_in_grpo" in text:
        return False

    old_signature = "def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode=\"mean_std_norm\", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):"
    new_signature = "def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, compute_mean_std_cross_steps_in_grpo=True, step_advantage_w=1.0, gigpo_mode=\"mean_std_norm\", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):"
    text = replace_once(text, old_signature, new_signature, "compute_advantage signature")

    old_core_call = '''        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
'''
    new_core_call = '''        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            compute_mean_std_cross_steps=compute_mean_std_cross_steps_in_grpo,
        )
'''
    text = replace_once(text, old_core_call, new_core_call, "GRPO episode-level flag")

    old_reward = '''                            batch.batch["token_level_scores"] = reward_tensor

                            print(f"{list(reward_extra_infos_dict.keys())=}")
'''
    new_reward = '''                            guided_rl_v2_transport = self.config.algorithm.get("guided_rl_v2_transport", "disabled")
                            if guided_rl_v2_transport != "disabled":
                                from repro_1p7b.graph_frontier.guided_rl_v2_transport import apply_rank_preserving_transport

                                reward_tensor, transport_metadata = apply_rank_preserving_transport(
                                    reward_tensor=reward_tensor,
                                    response_mask=batch.batch["response_mask"],
                                    non_tensor_batch=batch.non_tensor_batch,
                                    mode=guided_rl_v2_transport,
                                    tolerance=float(self.config.algorithm.get("guided_rl_v2_tie_tolerance", 1e-6)),
                                )
                                batch.non_tensor_batch.update(transport_metadata)
                            batch.batch["token_level_scores"] = reward_tensor

                            print(f"{list(reward_extra_infos_dict.keys())=}")
'''
    text = replace_once(text, old_reward, new_reward, "pre-advantage transport hook")

    old_call_arg = '''                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
'''
    new_call_arg = '''                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                compute_mean_std_cross_steps_in_grpo=self.config.algorithm.get(
                                    "compute_mean_std_cross_steps_in_grpo", True
                                ),
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
'''
    text = replace_once(text, old_call_arg, new_call_arg, "trainer flag transport")

    old_after_adv = '''                                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                            )

                    # update critic
'''
    new_after_adv = '''                                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                            )
                            if guided_rl_v2_transport != "disabled":
                                from repro_1p7b.graph_frontier.guided_rl_v2_transport import print_transport_audit

                                print_transport_audit(batch, train_step=self.global_steps)

                    # update critic
'''
    text = replace_once(text, old_after_adv, new_after_adv, "v2 audit logging")
    path.write_text(text)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", type=Path, default=Path("/home/u2024311031/verl-agent"))
    args = parser.parse_args()
    targets = {
        "rollout": args.verl_root / "agent_system/multi_turn_rollout/rollout_loop.py",
        "trainer": args.verl_root / "verl/trainer/ppo/ray_trainer.py",
    }
    for path in targets.values():
        if not path.exists():
            raise FileNotFoundError(path)
        backup = path.with_suffix(path.suffix + ".guided_rl_v2.orig")
        if not backup.exists():
            shutil.copy2(path, backup)
    changed = {
        "rollout": patch_rollout_loop(targets["rollout"]),
        "trainer": patch_ray_trainer(targets["trainer"]),
    }
    print(changed)


if __name__ == "__main__":
    main()
