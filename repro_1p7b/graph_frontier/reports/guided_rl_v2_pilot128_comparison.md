# Graph-Frontier Guided RL v2"È›y¯ßy‘ Matched 128-task Pilot

## Verdict

**PARTIAL-GO**

The v2 transport fixed the v1 failure mode: global capability no longer regressed, strict ordering remained intact, and the Graph gradient distribution is close to Standard rather than showing the v1 spike. However, this single pilot does not establish a clear local mechanism gain. Consumer completion improved only 3.00 percentage points on small and changing opportunity counts, while invalid retry worsened from 0/6 to 2/10. This is sufficient to retain v2 for confirmation, but not sufficient to call the full-RL gate passed.

No full RL workload was started.

## Completion and protocol lock

| Arm | Status | Steps | Rollouts / audits | Final checkpoint | Wall time |
|---|---:|---:|---:|---:|---:|
| Standard-v2 control | DONE | 64 | 512 / 512 | step 64 | 2:02:20 |
| Graph-v2 | DONE | 64 | 512 / 512 | step 64 | 2:09:20 |

Matched settings:

- Same frozen 128-task data, prompt order, seed 20260917, G=4, and 64 optimizer steps.
- Same Dynamic v1 model SHA256: `0231573c95e939a3dcc52518c15a7241edb32bcca65729f49230f593cec7b970`.
- Same train parquet SHA256: `5cef60450c57478df241da3438053fcc34ebcacfc2edc2c7ab761ccf4b27ee2f`.
- Same Qwen3-1.7B BF16 full-parameter FSDP, two A100 40GB GPUs.
- Context 6144; prompt 5200; response 768; train batch 2; per-GPU microbatch 1.
- Learning rate `1e-6`; KL coefficient `0.01`; no new clipping or optimizer change.
- Same corrected episode-level normalization with `compute_mean_std_cross_steps_in_grpo=False`.
- Only primary mechanism difference: Graph-v2 enables rank-preserving tertiary tie-break; Standard-v2 disables it.

## Main comparison

| Metric | Standard-v2 | Graph-v2 | Graph ∫w^~)ﬁv Standard |
|---|---:|---:|---:|
| Official reward | 0.54336 | 0.54893 | **+0.00557** |
| Task success | 28.71% | 29.69% | **+0.98 pp** |
| State score | 80.08% | 81.15% | +1.07 pp |
| Trace score | 29.69% | 30.47% | +0.78 pp |
| Consumer completion | 46.15% (30/65) | 49.15% (29/59) | **+3.00 pp** |
| Propagation accuracy | 90.00% (27/30) | 96.55% (28/29) | **+6.55 pp** |
| Invalid retry rate | 0.00% (0/6) | 20.00% (2/10) | **+20.00 pp worse** |
| Grad P95 | 5.559 | 5.440 | -0.119 |
| Grad max | 5.810 | 7.313 | +1.503 |
| Actor KL loss mean | 0.00784 | 0.00841 | +0.00056 |
| Trajectory length mean | 2.137 | 2.219 | +0.082 |
| Generated tokens mean | 1104.2 | 1119.7 | +15.5 |
| Tool calls mean | 1.086 | 1.152 | +0.066 |

The global direction is positive, unlike v1. The local denominator changed between online arms: Graph completed 29 consumers versus Standard's 30, but did so over 59 rather than 65 eligible opportunities. Therefore the +3.00 pp normalized improvement is directionally useful but not yet a clear effect.

## Task-paired uncertainty

| Metric | Mean delta | Approximate 95% interval | Graph wins / ties / losses |
|---|---:|---:|---:|
| Official reward | +0.00557 | [-0.01503, +0.02616] | 30 / 62 / 36 |
| Task success | +0.00977 | [-0.02604, +0.04557] | 17 / 94 / 17 |

The intervals cross zero. These are online, reward-dependent training rollouts rather than held-out evaluations, so they are evidence about mechanism behavior, not a final capability claim.

## Graph coverage and safety

| Metric | Standard-v2 diagnostic | Graph-v2 |
|---|---:|---:|
| Graph available | 65 / 512 (12.70%) | 67 / 512 (13.09%) |
| Tie-break eligible | 60 / 512 (11.72%) | 58 / 512 (11.33%) |
| Tie-break applied | 0 / 512 | 17 / 512 (3.32%) |
| Applied groups | 0 | 5 / 128 |
| Mean Graph score when available | 0.2831 | 0.2821 |
| Success-over-failure inversions | 0 | 0 |
| Strict official-order inversions | 0 | 0 |
| Harmful inversions | 0 | 0 |

The applied coverage remains intentionally sparse: Graph is blocked whenever success or official reward already distinguishes the rollouts.

## Stability and resources

| Metric | Standard-v2 | Graph-v2 |
|---|---:|---:|
| Grad median | 2.0665 | 2.5970 |
| Grad P90 | 4.8805 | 4.6645 |
| Grad P95 | 5.5593 | 5.4401 |
| Grad P99 | 5.7924 | 6.2918 |
| Grad max | 5.8100 | 7.3130 |
| Mean step time | 113.39 s | 118.18 s |
| Sampled GPU peak per card | 29,896 MiB | 29,896 MiB |
| Torch max allocated | 28.715 GB | 28.715 GB |
| Torch max reserved | 37.559 GB | 37.559 GB |

For historical context, Graph v1 had grad median about 2.83, P99 about 51.745, and max 129.599. Graph-v2 max 7.313, with P95 slightly below Standard-v2, is strong evidence that reward-duplication-induced instability was fixed.

## Historical context

| Arm | Official reward | Task success |
|---|---:|---:|
| Old Standard v1 | 0.56084 | 32.62% |
| Old Graph v1 | 0.54102 | 27.93% |
| Standard-v2 control | 0.54336 | 28.71% |
| Graph-v2 | 0.54893 | 29.69% |

The fair v2 comparison is Standard-v2 versus Graph-v2. Cross-version rows are background only because online reward-dependent updates change later sampled trajectories.

## Gate assessment

| Gate | Status | Evidence |
|---|---|---|
| A"È›y¯ßy‘ no global regression | PASS | Official +0.00557; success +0.98 pp |
| B"È›y¯ßy‘ local mechanism improvement | INCONCLUSIVE | Consumer +3.00 pp and propagation +6.55 pp, but small support, one fewer completed consumer, and retry worsened |
| C+ßuÁ‚ùÁT propagation preserved | PASS | 96.55% versus 90.00% |
| D ∫w^~)ﬁt stability fixed | PASS | No inversions; grad P95 5.440; max 7.313 versus v1 129.599 |

Because Gate B is not clearly established, the verdict is **PARTIAL-GO**, not GO.

## Saved checkpoints

- Standard-v2: `repro_1p7b/checkpoints/standard_rl_v2_control_pilot128/global_step_64`
- Graph-v2: `repro_1p7b/checkpoints/graph_frontier_rl_v2_pilot128/global_step_64`

Both contain rank-0/rank-1 model, optimizer, and extra-state shards, and both have `latest_checkpointed_iteration.txt = 64`.

## Evaluation status

- `FROZEN_300_NOT_RUN_FOR_PILOT`
- `BFCL_NOT_RUN_FOR_PILOT`

The mechanism report was prioritized as requested. A later confirmation should evaluate both saved v2 checkpoints under the same frozen-300/BFCL protocol; neither should be evaluated alone.

## Limitations

- One matched 128-task online pilot, not a multi-seed confirmation.
- Graph tie-break applied to only 17/512 rollouts across five prompt groups.
- Consumer and retry denominators are small and differ because later online trajectories diverge.
- Task-paired intervals cross zero.
- The ignored `multiprocess.resource_tracker` RLock traceback occurred after successful completion and checkpoint save in both arms; it did not change either `DONE` status.

