# Guided RL v1 reward/rank post-mortem

## Verdict

This is **Case E: a combination dominated by Case C (signal scarcity) and Case D (normalization/scale), with a partial Case A local/global objective mismatch and secondary Case B instability evidence**. The additive Graph reward should **not** be used for full RL in its current form.

- Graph reward events were positive on only **25/512 (4.88%)**, negative on **1**, and exactly zero on **486**.
- Inside the 128 actual Graph G=4 groups, adding `graph_delta` changed at least one pairwise relation in **5 groups**, changed the deterministic top-1 winner in **4**, removed every official-best trajectory from the additive top set in **0**, and created **0 strict lower-official-over-higher-official inversions across 0 groups**. In this pilot, all observed rank changes were official-tie breaks; no strict inversion occurred.
- It created **0 failure-over-success inversions across 0 groups**.
- The Graph arm's rollout-level official mean/success were **0.541016/0.279297**, versus Standard **0.560840/0.326172** (differences **-0.019824/-0.046875**).
- The run did **not save a model checkpoint**, so there is no held-out post-training model evaluation. These results diagnose the training signal and on-policy behavior; they do not claim a causal held-out model delta.

## Artifact and reconstruction basis

- Standard log: `repro_1p7b/logs/graph_frontier_rl_v1/frozen_v1_pilot128_standard_g4_canonicalfix2/train.log`
- Graph log: `repro_1p7b/logs/graph_frontier_rl_v1/frozen_v1_pilot128_graph_g4_canonicalfix2/train.log`
- Standard typed rollouts: `repro_1p7b/logs/graph_frontier_rl_v1/frozen_v1_pilot128_standard_g4_canonicalfix2/rollouts/` (512 JSON)
- Graph typed rollouts: `repro_1p7b/logs/graph_frontier_rl_v1/frozen_v1_pilot128_graph_g4_canonicalfix2/rollouts/` (512 JSON)
- Frozen shared pilot data: `repro_1p7b/data/graph_frontier_rl_v1_frozen_seed20260920/parquet_pilot128/train.parquet` (128 prompts, identical order)
- Exact GRPO code inspected: `/home/u2024311031/verl-agent/verl/trainer/ppo/core_algos.py` and `/home/u2024311031/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py`.

`task_id` is the stable prompt ID; `group_id = experiment + train_step + task_id`; filename suffix gives exact rollout index. `trajectory_length = typed tool event count + one terminal assistant action`. This equality was checked against every logged step. The maximum reconstruction discrepancies on exactly recoverable steps (logs are rounded to 3 decimals) are:

```json
{
  "standard": {
    "episode_length_mean": 0.0,
    "tool_call_mean": 0.0,
    "episode_reward_mean": 0.000500000000000056,
    "advantage_min": 0.0004933237339082019,
    "advantage_max": 0.0004913459043238699,
    "steps_with_all_rollout_advantages_exact": 64,
    "steps_with_ambiguous_rollout_advantages": 0
  },
  "graph": {
    "episode_length_mean": 0.0,
    "tool_call_mean": 0.0,
    "episode_reward_mean": 0.000500000000000056,
    "advantage_min": 0.0004933237339082019,
    "advantage_max": 0.0004947243775850207,
    "steps_with_all_rollout_advantages_exact": 62,
    "steps_with_ambiguous_rollout_advantages": 2
  }
}
```

`generated_tokens` remains null because exact per-trajectory token counts were not persisted. `semantic_success` is the persisted runtime `verifier_result`; it is diagnostic only and was not added as another reward term. When total active-turn rows are odd, `adjust_batch(mode="copy")` randomly duplicates one turn before advantage computation; the duplicated UID was not logged. Per-rollout `advantage` is therefore null whenever logged extrema do not uniquely determine it. `advantage_reconstruction` records this status for every row.

## Why actual GRPO amplification is larger than a naive G=4 z-score

The runtime repeats each trajectory's **episode reward on every active turn row**, then `compute_grpo_outcome_advantage(..., compute_mean_std_cross_steps=True)` computes mean/std over all turn rows sharing a prompt UID. Therefore a four-rollout group is length-weighted, not four-scalar-normalized. A long locally rewarded trajectory appears several times in the normalization pool, while a short failure can become a very large negative outlier. This is faithfully reconstructed in `advantage`/`normalized_group_reward`; it is not the naive four-terminal-score z-score.

## Rank audit (Graph arm, 128 actual G=4 groups)

| Metric | Count |
| --- | ---: |
| Groups with any pairwise rank/tie relation change | 5 |
| Deterministic top-1 winner changed | 4 |
| Official-best set entirely lost | 0 |
| Groups with strict harmful inversions | 0 |
| Strict harmful inversion pairs | 0 |
| Groups with success-over-failure inversions | 0 |
| Success-over-failure inversion pairs | 0 |
| Tie-break-only groups | 5 |
| Zero official spread but nonzero Graph spread | 4 |

All harmful inversion records, including both rollout rewards and success labels, are in the JSON report under `rank_audit.harmful_inversion_cases`.

## Positive Graph-delta classification (all 25 cases)

Definitions are evidence based: D means arithmetic/evidence inconsistency; C means verified local structure but official failure or harmful promotion; B means a non-harmful official tie-break; A means verified structure aligns with official success without harmful promotion.

| class | count |
| --- | --- |
| A_genuinely_better | 18 |
| C_local_good_global_bad | 7 |

No field was inferred from response prose. The full per-case list is in `positive_delta_classification.cases`.

## Correlation with global outcomes

| target | N | Pearson | Spearman |
| --- | --- | --- | --- |
| official_reward | 512 | 0.1596 | 0.1449 |
| official_success | 512 | 0.2267 | 0.2240 |
| semantic_success | 512 | 0.2267 | 0.2240 |
| internal_edge_complete | 112 | 0.9605 | 0.9580 |
| consumer_executed | 112 | 0.9979 | 0.9988 |
| correct_propagation | 25 | null | null |
| retry_after_error | 512 | -0.0381 | -0.0880 |
| invalid_retry | 512 | -0.0248 | -0.2011 |

Nulls are dropped pairwise. Boolean structural fields are null when the rollout had no relevant opportunity, so their N is intentionally smaller. Correlation does not establish causality, but it exposes whether the sparse local bonus aligns with the official objective.

## Sparsity and opportunity-normalized incidence

| Quantity | Value |
| --- | ---: |
| Positive / all rollouts | 25 / 512 = 4.88% |
| Eligible internal edges (rollout-edge opportunities) | 124 |
| Producer-success opportunities | 56 |
| Completed consumers | 26 |
| Propagation checks / correct | 26 / 26 |
| Invalid retries | 1 |
| Positive / eligible internal edges | 0.2016 |
| Positive / producer-success opportunities | 0.4464 |
| Positive / completed consumers | 0.9615 |

### By dependency depth

| experiment | dependency_depth | n | official_reward_mean | official_success_rate | graph_positive_n | tool_call_mean |
| --- | --- | --- | --- | --- | --- | --- |
| graph | 0 | 384 | 0.5818 | 0.3229 | 0 | 1.0286 |
| graph | 1 | 116 | 0.4065 | 0.1552 | 22 | 1.3276 |
| graph | 2 | 12 | 0.5375 | 0.0833 | 3 | 0.8333 |
| standard | 0 | 384 | 0.5978 | 0.3646 | 0 | 1.2344 |
| standard | 1 | 116 | 0.4371 | 0.2155 | 23 | 1.3621 |
| standard | 2 | 12 | 0.5750 | 0.1667 | 2 | 1.5000 |

### Internal-edge vs no-edge tasks

| experiment | internal_edge_subset | n | official_reward_mean | official_success_rate | graph_positive_n | tool_call_mean | unexpected_call_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| graph | internal_edge | 112 | 0.4219 | 0.1607 | 25 | 1.1964 | 0.0804 |
| graph | no_internal_edge | 400 | 0.5744 | 0.3125 | 0 | 1.0625 | 0.1425 |
| standard | internal_edge | 112 | 0.4344 | 0.1875 | 25 | 1.3393 | 0.1071 |
| standard | no_internal_edge | 400 | 0.5962 | 0.3650 | 0 | 1.2500 | 0.2100 |

The full environment table is stored in JSON. Environments with the largest Graph-minus-Standard official shifts are shown below (small-N rows are descriptive, not inferential):

| environment | n_graph | official_diff | success_diff | graph_positive_n |
| --- | --- | --- | --- | --- |
| Didi | 4 | -0.3875 | -0.7500 | 0 |
| WuWa | 4 | -0.2625 | -0.5000 | 0 |
| KospiKosdaqStock | 16 | -0.1562 | -0.3125 | 0 |
| Whois | 4 | -0.1375 | -0.2500 | 0 |
| AttomRealEstate\|RentCast | 4 | -0.1250 | -0.2500 | 0 |
| Canvas | 8 | -0.1250 | -0.2500 | 0 |
| PayPalPaymentProcessor\|StripePaymentServer | 4 | -0.1250 | 0.0000 | 0 |
| RentCast | 4 | -0.1250 | -0.2500 | 0 |
| HotelBooking | 12 | -0.1167 | -0.2500 | 3 |
| LeagueOfLegends | 4 | -0.1125 | -0.2500 | 0 |
| ShopifyEcommerce | 4 | -0.1125 | -0.2500 | 0 |
| Valorant | 4 | -0.1125 | -0.2500 | 0 |
| VehicleControl | 24 | 0.0479 | 0.0833 | 3 |
| FatSecretPlatform | 4 | 0.0500 | 0.0000 | 0 |
| Retail | 8 | 0.0500 | 0.0000 | 0 |
| FakeStoreServer\|EbayServer | 4 | 0.0500 | 0.0000 | 0 |
| TFTServer | 12 | 0.0500 | 0.0833 | 0 |
| UUPaoTui | 8 | 0.0563 | 0.1250 | 0 |
| GoogleTasks | 4 | 0.1250 | 0.2500 | 0 |
| StripePaymentServer | 8 | 0.1813 | 0.3750 | 0 |

## Training-stage behavior drift

| experiment | stage | n | official_reward_mean | official_success_rate | graph_delta_mean | graph_positive_n | tool_call_mean | unexpected_call_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph | early_1_21 | 168 | 0.5798 | 0.3036 | 0.0488 | 12 | 1.0298 | 0.0833 |
| graph | middle_22_42 | 168 | 0.5149 | 0.2738 | 0.0375 | 9 | 1.1667 | 0.1905 |
| graph | late_43_64 | 176 | 0.5290 | 0.2614 | 0.0156 | 4 | 1.0795 | 0.1136 |
| standard | early_1_21 | 168 | 0.5854 | 0.3155 | 0.0542 | 13 | 1.1786 | 0.1667 |
| standard | middle_22_42 | 168 | 0.5533 | 0.3571 | 0.0375 | 9 | 1.4464 | 0.2500 |
| standard | late_43_64 | 176 | 0.5446 | 0.3068 | 0.0119 | 3 | 1.1875 | 0.1477 |

Because each arm generated its own stochastic on-policy rollouts, these stage comparisons are behavioral diagnostics rather than paired counterfactual outcomes.

## Step 53 forensic

Step 53 has `grad_norm=129.599`, logged advantage range **[-3.328, 0.667]**, `KL=0.013`, response max **768.0** and clip ratio **0.182**.

| task_id | rollout_index | official_reward | official_success | graph_delta | total_reward | trajectory_length | advantage | tool_sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gf-rl-v1-s924096046-t0 | 0 | 0.5000 | False | 0.0000 | 0.5000 | 1 | -1.3333 |  |
| gf-rl-v1-s924096046-t0 | 1 | 0.5000 | False | 0.0000 | 0.5000 | 2 | -1.3333 | HotelBooking-search_destinations |
| gf-rl-v1-s924096046-t0 | 2 | 1.0000 | True | 0.0000 | 1.0000 | 3 | 0.6667 | HotelBooking-search_destinations -> HotelBooking-get_hotels |
| gf-rl-v1-s924096046-t0 | 3 | 1.0000 | True | 0.0000 | 1.0000 | 3 | 0.6667 | HotelBooking-search_destinations -> HotelBooking-get_hotels |
| gf-rl-v1-s966029525-t0 | 0 | 0.0000 | False | 0.0000 | 0.0000 | 1 | -3.3282 |  |
| gf-rl-v1-s966029525-t0 | 1 | 0.0000 | False | 0.7000 | 0.7000 | 4 | 0.2773 | VehicleControl-get_city_zipcode -> VehicleControl-get_city_zipcode -> VehicleControl-estimate_distance_between_cities |
| gf-rl-v1-s966029525-t0 | 2 | 0.0000 | False | 0.7000 | 0.7000 | 4 | 0.2773 | VehicleControl-get_city_zipcode -> VehicleControl-get_city_zipcode -> VehicleControl-estimate_distance_between_cities |
| gf-rl-v1-s966029525-t0 | 3 | 0.0000 | False | 0.7000 | 0.7000 | 4 | 0.2773 | VehicleControl-get_city_zipcode -> VehicleControl-get_city_zipcode -> VehicleControl-estimate_distance_between_cities |

The extreme `-3.328` is reproduced exactly (up to log rounding): task `gf-rl-v1-s966029525-t0` has one short zero-reward rollout and three four-turn rollouts with `official=0`, `graph_delta=+0.7`. Length-weighted normalization repeats each +0.7 score four times, making the one zero score the extreme negative outlier. The other task supplies the logged +0.667 maximum. Thus the spike is consistent with the combination of **sparse local bonuses + length-weighted group normalization + long/clipped outputs**. It cannot be attributed to one factor alone from aggregate logs, but it is not an unexplained random spike.

Graph grad-norm percentiles: `{"0": 0.004, "1": 0.00652, "5": 0.012150000000000001, "25": 0.034749999999999996, "50": 2.83, "75": 4.2885, "90": 4.9043, "95": 5.611399999999998, "99": 51.74548999999968, "100": 129.599}`. Reward/bonus/advantage percentiles and Graph-spread/official-spread ratios are in `reward_scale` in the JSON report.

## Counterfactual ranking rules

| scheme | official_inversion_pairs | success_over_failure | tie_break_pairs | graph_groups | structural_best_rate |
| --- | --- | --- | --- | --- | --- |
| A_additive | 0 | 0 | 13 | 5 | 1.0000 |
| B_official_only | 0 | 0 | 0 | 0 | 0.7000 |
| C_rank_preserving_tie_break | 0 | 0 | 13 | 5 | 1.0000 |
| D_success_gated_lexicographic | 0 | 0 | 13 | 5 | 1.0000 |

- **A additive** is the current rule and directly permits local bonus to overpower official differences.
- **B official only** is safe but discards structural information.
- **C rank-preserving/tie-break** orders by `(official_reward, graph_delta)`: Graph can choose among official ties but cannot reverse a strict official preference.
- **D success-gated lexicographic** orders by `(official_success, official_reward, graph_delta)`: a verified successful rollout always beats a failure, then Graph resolves remaining ties.

For the next reward iteration, use **D if the runtime success bit is available at training time; otherwise C**. In both cases: normalize Graph incidence per eligible edge/opportunity, cap its contribution, and change GRPO grouping to episode-level deduplication (`compute_mean_std_cross_steps=False` or an equivalent `traj_uid` dedupe) so trajectory length cannot multiply a terminal reward's influence.

## Final mechanism classification

- **Case A — reward misspecification: partial evidence.** Seven positive local-structure cases are official failures, so local and global objectives can disagree. However, the realized G=4 groups show zero strict official-rank or success-over-failure inversions; all five rank changes are tie-only.
- **Case B — training instability: secondary evidence.** Graph grad norms have a severe tail and official behavior drifts downward, but no saved final model exists for a controlled held-out causal comparison.
- **Case C — signal too sparse: observed.** Only 25/512 rollouts receive a positive Graph delta; most tasks and many eligible edges never provide a differentiating signal.
- **Case D — normalization/scale problem: observed.** Graph deltas reach +0.7 on an official 0–1 scale and are repeated per active turn before within-group standardization; step 53 demonstrates the resulting extreme advantage.
- **Case E — combination: final diagnosis.** C + D are directly established; B is consistent with the severe gradient tail and behavioral drift; A exists as a local/global mismatch but did not produce strict within-group inversions in this pilot. The training reward rises because sparse locally compliant, often longer trajectories receive a large additive bonus and length-amplified advantages, while official success does not improve.

## Recommendation

**Do not run full RL with the current additive reward.** The next bounded experiment should preserve official/success rank, use Graph only as a conditional tie-breaker, deduplicate reward normalization by trajectory, and save at least a final checkpoint for held-out comparison. No training or inference was launched for this audit.
