# Rich-v3 50-task quick model-effect pilot (2026-09-23)

## Scope and frozen protocol

This is a short diagnostic, not the formal 1219-task generation or a matched Standard-vs-Graph DPO study. Source is the previously audited `smoke_120_unambiguous_v3_repaired_seed20260925` with verdict `SMOKE_READY`, 50 valid tasks, zero exact Frozen300 overlap, and 100% source executable replay. Deterministic task-disjoint split (seed 20260923): 36 train, 4 preference validation, 10 heldout evaluation. The full resumable 1219-task run remained paused throughout.

CPU replay accepted 78/97 static eligible frontiers: 53 train, 5 preference validation, 20 heldout. The 19 dropped states were not used. Dynamic-v1 sampled four actions per train/validation state; initial 8k serving window produced HTTP 400 for a 10.7k-token validation task. A 12k-window validation-only top-up added one successful real model sample without changing task split, original samples, or quality thresholds. This yielded 70 clean train pairs from 27 tasks and 5 validation pairs from 2 tasks, with 100% chosen-action executable and zero Frozen300 overlap.

The final micro-training set contains 16 pairs from 15 tasks: 10 premature-stop, 4 wrong-tool, 1 useless-retry, 1 wrong-value/parameter-flow failure. A copy of the initial 16-pair selection was retained before replacing one premature-stop pair by the real wrong-value pair. Dynamic-v1 initialized full-parameter Graph DPO, 2×A100 40 GB, 16 steps, learning rate 1e-6, beta 0.1, max length 12288; validation used the 5 separate pairs. Both pair and serialization gates passed. The 16-step training and checkpoint save completed; train loss 0.4078 and preference validation loss 0.0662 are optimization diagnostics, not task-success claims.

## Independent heldout results

All heldout comparisons used the same 10 fixed tasks, seeds, model sampling temperature 0, eight-step tool budget, and 384-token per-step budget; model servers ran serially for full task rollouts to avoid environment-state interference.

| Full executable rollout | Dynamic-v1 | Graph DPO |
|---|---:|---:|
| Semantic success | 0/10 | 1/10 |
| Mean official reward | 0.185 | 0.125 |

On the same heldout tasks, 20 replay-validated gold-prefix frontiers were independently evaluated with one deterministic 512-token next-action sample per model. The two models ran on separate GPUs; no environment execution occurred during this next-action comparison.

| Gold-prefix next action | Dynamic-v1 | Graph DPO |
|---|---:|---:|
| Exactly correct action | 3/20 | 6/20 |
| Correct tool and propagated value, allowing other argument differences | 5/20 | 10/20 |
| Paired flow improvements / regressions | — | 5 / 0 |
| Premature stop | 12/20 | 0/20 |
| Wrong tool | 3/20 | 7/20 |
| Useless retry | 0/20 | 2/20 |
| Wrong propagated value | 0/20 | 1/20 |
| Model request errors | 0 | 0 |

## Interpretation

There is a real local next-action signal: the short Graph DPO run reduced premature stopping and doubled correct internal-value propagation on the fixed heldout frontiers. However, errors shifted toward wrong tools and retries, while mean full-rollout official reward fell. One extra terminal success on only 10 tasks is too small to establish robust end-to-end improvement. The selected training data are also dominated by premature-stop negatives, so this pilot does not isolate graph-specific parameter-flow learning. There is no matched Standard DPO control. Treat the direction as **promising at the gold-prefix frontier, unproven for complete tasks**; do not escalate to full training solely on these numbers.

## Artifacts and safety

- Pilot root: `repro_1p7b/graph_frontier/preference_generation_v3/quick_effect_pilot_50_seed20260923/`
- Frozen split: `quick_effect_manifest.json`
- Replay audit: `replay_summary.json`
- Pair and serialization gates: `quick_pair_audit.json`, `quick_dpo_materialization_audit.json`, `quick_pair_selection_audit.json`
- Paired full-rollout result: `quick_effect_result.json`
- Paired gold-prefix result: `quick_frontier_effect_result.json`
- Checkpoint and training manifest: `repro_1p7b/checkpoints/rich_v3_quick_graph_dpo_seed20260923/`
- CPU Rich-v3 regression suite: 13/13 passed.
- No full 1219-task restart, Frozen300 evaluation, BFCL, full RL, Git commit, push, pull, or PR.

All generated rollouts, runtime logs, and checkpoints remain untracked and must not be staged.
