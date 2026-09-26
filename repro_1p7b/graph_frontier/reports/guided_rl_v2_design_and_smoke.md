# Guided RL v2 ºw^~)Þt Runtime Fix, Design, and Smoke

## Status

Implementation, CPU tests, offline replay, and both 8-task G=2 smokes are complete. No full RL workload was started.

## Locked protocol

- Frozen pilot train SHA256: `5cef60450c57478df241da3438053fcc34ebcacfc2edc2c7ab761ccf4b27ee2f`
- Dynamic v1 model SHA256: `0231573c95e939a3dcc52518c15a7241edb32bcca65729f49230f593cec7b970`
- Dynamic v1 config SHA256: `2eb1c7d9ec4d1dde3651c2257b17d042489c589b16290501e0fbb5243851e03d`
- Both arms independently initialize from the original Dynamic v1 checkpoint.
- EnvFactory official reward remains the primary reward in both arms.
- Graph v2 is the only mechanism difference and can act only inside a success/official tie.
- Official tie tolerance is `1e-6`; the observed minimum meaningful official gap is approximately `0.05`.

## Runtime root cause and fix

The real runtime flow was traced rather than inferred:

1. `EpisodeRewardManager` writes the terminal episode reward onto the last token of every active-turn row.
2. `TrajectoryCollector.gather_rollout_data` carries the same episode reward across those active-turn rows.
3. With `compute_mean_std_cross_steps_in_grpo=True`, GRPO group statistics include all repeated active-turn rows, so longer trajectories receive more weight in normalization.
4. The token-level policy loss then consumes the resulting repeated turn-row advantage.

Guided RL v2 uses `compute_mean_std_cross_steps_in_grpo=False`. The group statistics deduplicate by `(uid, traj_uid)`, so each trajectory contributes exactly one episode scalar; the resulting episode advantage is broadcast back to its active turns for the unchanged token loss. Token count can still naturally affect policy loss, but trajectory length no longer duplicates the reward during group comparison.

The narrow transport hook is in:

- Runtime: `/home/u2024311031/verl-agent/verl/trainer/ppo/ray_trainer.py`, immediately after reward collection and before `token_level_scores` and `compute_advantage`.
- Structured rollout metadata: `/home/u2024311031/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py`.
- Reproducible patcher: `repro_1p7b/patches/apply_verl_agent_guided_rl_v2.py`.
- Transport implementation: `repro_1p7b/graph_frontier/guided_rl_v2_transport.py`.

The official EnvFactory `src/` tree was not modified. A minimal external verl-agent runtime patch was required and is reproduced by the patcher.

## Rank-preserving transport

For each prompt group, v2 orders trajectories by:

1. runtime task success;
2. official reward;
3. Graph score only within the same success and `1e-6` official bin.

When the trainer requires a scalar, the Graph perturbation is bounded to at most one quarter of the nearest primary-class gap. Construction-time assertions prove that Graph cannot reverse a strict success or official ordering.

Standard-v2 uses the same corrected episode-level transport with Graph disabled. Graph-v2 uses the same transport with the tertiary tie-break enabled.

## Opportunity-normalized Graph score

The existing profiler remains unchanged. The terminal reward adapter exposes:

- `consumer_score = completed / eligible_consumer_opportunities`;
- `prop_score = correct / eligible_prop_opportunities`;
- bounded high-confidence retry rate;
- neutral/unavailable, not a free success, when a denominator is zero.

The retained relative weights are propagation `0.30`, consumer `0.40`, retry `-0.10`, clipped to `[-1, 1]`. This score is diagnostic in Standard-v2 and tertiary-only in Graph-v2.

## CPU tests

Command:

```text
PYTHONPATH=/home/u2024311031/verl-agent:/home/u2024311031/workspace/envfactory_repro_1p7b \
/home/u2024311031/.conda/envs/verl-agent-webshop/bin/python -m pytest -q \
  repro_1p7b/graph_frontier/tests/test_guided_rl_v2_transport.py \
  repro_1p7b/graph_frontier/tests/test_guided_rl_reward.py
```

Result: `11 passed in 50.76s`.

Required invariants all pass:

- same reward, different trajectory length: PASS;
- zero versus positive reward without length multiplication: PASS;
- all rewards equal produce neutral group advantage: PASS;
- strict success/official ordering cannot be inverted: PASS.

## Offline replay of the 512 v1 Graph rollouts

| Check | Result |
|---|---:|
| Harmful inversions | 0 |
| Success-over-failure inversions | 0 |
| Strict official-order inversions | 0 |
| Graph-available rollouts | 67 / 512 |
| Tie-break eligible rollouts | 53 / 512 |
| Tie-break applied rollouts | 18 / 512 |
| Tie-break applied groups | 5 |

For step 53 (`gf-rl-v1-s966029525-t0`):

| Trajectory | v1 advantage | v2 counterfactual advantage |
|---|---:|---:|
| Short failure, graph 0 | -3.328184 | -1.499983 |
| Each long failure, graph +0.7 | +0.277349 | +0.499994 |

The Graph distinction is preserved inside the official tie, but each trajectory enters normalization once; four active turns no longer multiply its comparison weight.

## 8-task G=2 smoke

Both smokes used 8 frozen tasks, 16 rollouts, four optimizer steps, two A100 40GB GPUs, context 6144, max response 768, full-parameter BF16 FSDP, per-GPU microbatch 1, and independent Dynamic v1 initialization.

| Metric | Standard-v2 smoke | Graph-v2 smoke |
|---|---:|---:|
| Status | DONE | DONE |
| Optimizer steps / audits | 4 / 16 | 4 / 16 |
| Final checkpoint | step 4 | step 4 |
| Official reward mean | 0.58750 | 0.62187 |
| Success rate | 18.75% | 25.00% |
| Advantage range | [-0.7071, +0.7071] | [-0.7071, +0.7071] |
| Grad norm range | [0.006, 1.969] | [0.006, 2.748] |
| Actor KL loss | 0.001 each step | 0.001 each step |
| GPU sampled peak per card | 29,896 MiB | 29,896 MiB |
| Harmful inversions | 0 | 0 |

The smoke subset produced two Graph-available rollouts, both with Graph score 0, so no tie-break was applied in the smoke itself. This is a coverage limitation, not a transport failure: the offline replay exercised 18 applied rollouts across five groups, and the formal pilot later exercised the mechanism again.

## Artifacts

- `repro_1p7b/graph_frontier/guided_rl_v2_transport.py`
- `repro_1p7b/graph_frontier/replay_guided_rl_v2.py`
- `repro_1p7b/graph_frontier/run_guided_rl_v2_train.sh`
- `repro_1p7b/graph_frontier/analyze_guided_rl_v2.py`
- `repro_1p7b/graph_frontier/tests/test_guided_rl_v2_transport.py`
- `repro_1p7b/graph_frontier/reports/guided_rl_v2_offline_replay.json`
- `repro_1p7b/graph_frontier/reports/guided_rl_v2_offline_replay.csv`

