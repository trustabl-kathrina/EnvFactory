# Graph-Frontier Guided RL v1 — 128-task Pilot Comparison

## Decision

**Status: PILOT_COMPLETE — DO NOT PROMOTE TO FULL RL**

Both matched 128-task, G=4 pilots completed from the original Dynamic v1 checkpoint. The Graph reward increased the optimized reward, but it did not improve the official capability signal in this run. No full RL workload was started.

## Locked protocol

- Source model: `repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b`
- Frozen dataset status: `RL_DATASET_V1_FROZEN`
- Replay-valid pool: 349 tasks
- Frozen split: train 314 / validation 35
- Shared pilot: 128 tasks, four rollouts per task, 512 reward events per arm
- Optimizer steps: 64 per arm
- Frozen-300 exact train overlap: 0
- Pilot manifest SHA256: `1f97f8fea69dbe007d73f531291137892ca0df34e9ba7669b2934ab98768cfa2`
- Pilot train parquet SHA256: `5cef60450c57478df241da3438053fcc34ebcacfc2edc2c7ab761ccf4b27ee2f`
- Pilot validation parquet SHA256: `824bd9230b8a9e226bcd8826944a96b1cf8e749babd587b3989372508f70a547`
- Contamination report SHA256: `71fb88f724bcf2078d64c3374ca0de3aad59021271066d7c04fb7517c1082239`

Data order, seed, G, steps, prompt budget, rollout budget, model initialization, and optimizer configuration were matched. The principal intended variable was reward mode.

## Completion

| Arm | Status | Steps | Wall time |
|---|---:|---:|---:|
| Standard / official reward | DONE | 64/64 | 2:04:42 |
| Graph-Frontier reward | DONE | 64/64 | 1:58:40 |

The ignored `multiprocess.resource_tracker` RLock traceback occurred during interpreter teardown after successful completion and did not change either DONE status.

## Main result

| Metric over 512 rollout events | Standard | Graph | Graph − Standard |
|---|---:|---:|---:|
| Official score mean | 0.56084 | 0.54102 | **-0.01982** |
| Optimized/final reward mean | 0.56084 | 0.57471 | +0.01387 |
| Official success rate | 0.32617 | 0.27930 | **-0.04688** |
| State score mean | 0.80469 | 0.80664 | +0.00195 |
| Trace score mean | 0.33789 | 0.29102 | **-0.04687** |
| Mean graph delta | 0.03418 (diagnostic only) | 0.03369 (applied) | — |

The Graph arm's higher optimized reward is explained by the applied mean graph delta (+0.03369). After removing that shaping bonus, the official score is lower than Standard by 0.01982 and official success is lower by 4.69 percentage points.

At task level, Graph official score was better on 34 tasks, equal on 53, and worse on 41.

## Graph-signal coverage

- Tasks with at least one internal dependency edge: 28 / 128
- Reward events with internal edges: 112 / 512
- Positive graph delta events: 25 / 512
- Negative graph delta events: 1 / 512
- Zero graph delta events: 486 / 512
- Eligible internal-edge occurrences: 124
- Consumer opportunities: 56
- Completed consumer opportunities: 26
- Propagation checks: 26
- Correct propagations: 26
- Invalid-retry count: 1
- Maximum graph delta: +0.70
- Minimum graph delta: -0.05

The signal is sparse: only 4.9% of rollout events received a positive delta. Correct propagation was perfect when it was checked, but only 26 of 56 consumer opportunities reached a completed consumer call.

## Paired task subsets

| Subset | Tasks | Standard official | Graph official | Difference | Standard success | Graph success | Difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 128 | 0.56084 | 0.54102 | -0.01982 | 0.32617 | 0.27930 | -0.04688 |
| Internal-edge tasks | 28 | 0.43438 | 0.42188 | -0.01250 | 0.18750 | 0.16071 | -0.02679 |
| No-edge tasks | 100 | 0.59625 | 0.57438 | -0.02188 | 0.36500 | 0.31250 | -0.05250 |

The intended edge subset did not show an official-score or success-rate gain. The no-edge majority also regressed, despite receiving essentially no graph bonus.

## Training dynamics

| Metric (64-step mean) | Standard | Graph |
|---|---:|---:|
| Episode reward mean | 0.56091 | 0.57477 |
| Episode success rate | 0.32617 | 0.27930 |
| Actor KL loss | 0.00567 | 0.00733 |
| Actor gradient norm | 2.70441 | 4.41953 |
| Tool calls per episode | 1.26953 | 1.09180 |

Last-16-step actor gradient norm averaged 10.2051 for Graph versus 2.3732 for Standard. Graph also reached a gradient norm of 129.599 at step 53. This is a stability warning, not proof of divergence, but it argues against scaling the current reward unchanged.

## Interpretation

The current Graph reward is technically functional: typed edge checks fire, positive and negative deltas are produced, and both arms train to completion. However, the reward is too sparse and can be large relative to the official reward. In this single matched pilot it improved the shaped objective without improving the underlying official capability metric.

Therefore the result is **no-go for full Guided RL v1**.

## Limitations

- This is one matched 128-task pilot, not a multi-seed confirmation.
- Metrics are online training-rollout metrics on the shared pilot tasks, not held-out post-training evaluation.
- `save_freq=-1` was used, so no final model weights were saved for either arm; only complete logs and typed rollout evidence remain.
- Reward-dependent updates make later sampled trajectories differ even with matched initial seed and schedule.

## Required changes before another pilot

1. Increase useful graph-signal density, preferably with an edge-enriched but still shared and auditable pilot.
2. Bound or normalize graph delta relative to official reward so shaping cannot dominate the target.
3. Preserve a final checkpoint and run a held-out executable evaluation.
4. Re-run a matched pilot, then require an official-score/success improvement before considering full RL.
