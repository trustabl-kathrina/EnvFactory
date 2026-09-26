# Guided RL v2 Benchmark Evaluation

## Executive Summary

**Verdict: NO-GO**

Frozen300 and BFCL completed under locked protocols. FULL_RL_NOT_STARTED.

# 1. Checkpoint integrity

- Standard/Graph FSDP rank-0 and rank-1 model shards present; step: 64/64.
- Same config: True; same tokenizer: True.
- Shared Dynamic-v1 init model SHA256: `0231573c95e939a3dcc52518c15a7241edb32bcca65729f49230f593cec7b970`.
- Shared Dynamic-v1 init config SHA256: `2eb1c7d9ec4d1dde3651c2257b17d042489c589b16290501e0fbb5243851e03d`.
- Shared pilot parquet SHA256: `5cef60450c57478df241da3438053fcc34ebcacfc2edc2c7ab761ccf4b27ee2f`.
- Both checkpoints were merged into independent HF eval directories; originals were not modified.

# 2. Frozen 300 protocol

- Manifest SHA256: `4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c`
- 300 tasks; seed 20260914; diagnosis 240 / held-out 60; 710 gold internal edges.
- MCP/state dual-process isolation validated: True.
- GPU0=Standard-v2, GPU1=Graph-v2; identical frozen inference config.

# 3. Frozen 300 overall

| Model | Semantic | RefPath | Internal complete | Producer reach | Consumer reach | CondProp | Internal E2E | Consumer not executed | Retry |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 9.33% (28/300) | 0.00% (0/300) | 4.67% (14/300) | 29.15% (207/710) | 26.48% (188/710) | 7.45% (14/188) | 1.97% (14/710) | 0 | 138 |
| Original SFT | 13.67% (41/300) | 2.33% (7/300) | 24.33% (73/300) | 95.07% (675/710) | 54.93% (390/710) | 50.26% (196/390) | 27.61% (196/710) | 129 | 68 |
| Static PA | 18.33% (55/300) | 6.33% (19/300) | 10.67% (32/300) | 94.65% (672/710) | 57.32% (407/710) | 15.72% (64/407) | 9.01% (64/710) | 106 | 115 |
| Dynamic v1 | 19.00% (57/300) | 7.67% (23/300) | 32.67% (98/300) | 94.79% (673/710) | 60.28% (428/710) | 55.37% (237/428) | 33.38% (237/710) | 102 | 94 |
| Dynamic v2 | 18.33% (55/300) | 5.67% (17/300) | 31.67% (95/300) | 94.51% (671/710) | 59.44% (422/710) | 54.03% (228/422) | 32.11% (228/710) | 102 | 101 |
| Standard-v2 | 11.00% (33/300) | 0.67% (2/300) | 22.00% (66/300) | 95.07% (675/710) | 51.41% (365/710) | 47.67% (174/365) | 24.51% (174/710) | 109 | 82 |
| Graph-v2 | 10.00% (30/300) | 1.67% (5/300) | 23.00% (69/300) | 94.93% (674/710) | 52.39% (372/710) | 50.00% (186/372) | 26.20% (186/710) | 145 | 81 |

# 4. Frozen 300 depth breakdown

## Depth 1

| Model | Semantic | Producer reach | Consumer reach | CondProp | Internal E2E | Consumer complete |
|---|---:|---:|---:|---:|---:|---:|
| Standard-v2 | 25.83% (31/120) | 100.00% (172/172) | 19.77% (34/172) | 58.82% (20/34) | 11.63% (20/172) | 26.67% (32/120) |
| Graph-v2 | 20.00% (24/120) | 100.00% (172/172) | 16.28% (28/172) | 60.71% (17/28) | 9.88% (17/172) | 22.50% (27/120) |

## Depth 2

| Model | Semantic | Producer reach | Consumer reach | CondProp | Internal E2E | Consumer complete |
|---|---:|---:|---:|---:|---:|---:|
| Standard-v2 | 1.82% (2/110) | 100.00% (293/293) | 65.19% (191/293) | 25.65% (49/191) | 16.72% (49/293) | 50.91% (56/110) |
| Graph-v2 | 5.45% (6/110) | 99.66% (292/293) | 69.62% (204/293) | 31.37% (64/204) | 21.84% (64/293) | 58.18% (64/110) |

## Depth 3+

| Model | Semantic | Producer reach | Consumer reach | CondProp | Internal E2E | Consumer complete |
|---|---:|---:|---:|---:|---:|---:|
| Standard-v2 | 0.00% (0/70) | 85.71% (210/245) | 57.14% (140/245) | 75.00% (105/140) | 42.86% (105/245) | 50.00% (35/70) |
| Graph-v2 | 0.00% (0/70) | 85.71% (210/245) | 57.14% (140/245) | 75.00% (105/140) | 42.86% (105/245) | 50.00% (35/70) |

# 5. Internal-edge subset

All frozen-300 tasks contain at least one internal dependency edge.
Therefore this subset is identical to full300; metrics and paired results are retained in JSON.

# 6. Diagnosis240 / Heldout60

## diagnosis

| Model | Semantic | Consumer complete | CondProp | Internal E2E |
|---|---:|---:|---:|---:|
| Standard-v2 | 10.00% (24/240) | 42.08% (101/240) | 48.16% (144/299) | 25.31% (144/569) |
| Graph-v2 | 10.00% (24/240) | 43.75% (105/240) | 51.47% (158/307) | 27.77% (158/569) |

## heldout

| Model | Semantic | Consumer complete | CondProp | Internal E2E |
|---|---:|---:|---:|---:|
| Standard-v2 | 15.00% (9/60) | 36.67% (22/60) | 45.45% (30/66) | 21.28% (30/141) |
| Graph-v2 | 10.00% (6/60) | 35.00% (21/60) | 43.08% (28/65) | 19.86% (28/141) |

# 7. BFCL V3 Multi-Turn 800

| Model | Overall | Base | Missing Function | Missing Parameter | Long Context |
|---|---:|---:|---:|---:|---:|
| Base | 8.75% | 12.50% | 7.50% | 8.50% | 6.50% |
| Original SFT | 9.75% | 13.50% | 11.00% | 8.00% | 6.50% |
| Static PA | 11.00% | 12.00% | 9.50% | 15.00% | 7.50% |
| Dynamic v1 | 10.38% | 14.00% | 11.00% | 10.50% | 6.00% |
| Dynamic v2 | 11.12% | 13.00% | 10.50% | 13.00% | 8.00% |
| Standard-v2 | 11.50% | 16.00% | 12.00% | 11.50% | 6.50% |
| Graph-v2 | 10.50% | 13.00% | 9.50% | 11.00% | 8.50% |

# 8. Paired statistics

- Frozen300 semantic_task_success: delta=-1.00 pp, 95% CI=[-4.0, 2.0], exact McNemar p=0.663624, Graph-only=9, Standard-only=12.
- Frozen300 reference_path_complete_success: delta=1.00 pp, 95% CI=[-0.3333333333333333, 2.6666666666666665], exact McNemar p=0.375, Graph-only=4, Standard-only=1.
- Frozen300 task_level_internal_edge_complete: delta=1.00 pp, 95% CI=[-2.0, 4.0], exact McNemar p=0.663624, Graph-only=12, Standard-only=9.
- Frozen300 consumer_task_completion: delta=1.00 pp, 95% CI=[-2.3333333333333335, 4.333333333333333], exact McNemar p=0.690038, Graph-only=14, Standard-only=11.
- Frozen300 consumer_not_executed: delta=12.00 pp, 95% CI=[7.333333333333333, 16.666666666666668], exact McNemar p=7.28843e-07, Graph-only=45, Standard-only=9.
- Frozen300 producer_reach: delta=-0.14 pp, clustered 95% CI=[-0.4322766570605152, 0.0], permutation p=1.
- Frozen300 consumer_reach: delta=0.99 pp, clustered 95% CI=[-0.8759124087591275, 3.0055427685659226], permutation p=0.392361.
- Frozen300 conditional_propagation: delta=2.33 pp, clustered 95% CI=[-0.12417257264672214, 4.899478669231046], permutation p=0.079992.
- Frozen300 internal_e2e: delta=1.69 pp, clustered 95% CI=[-0.4121305418719252, 3.8135881863944956], permutation p=0.152385.
- Frozen300 retry_after_error: total delta=-1, mean/task delta=-0.0033, 95% CI=[-0.08333333333333333, 0.08666666666666667], sign-flip p=1.
- BFCL overall: delta=-1.00 pp, 95% CI=[-3.625, 1.625], exact McNemar p=0.508513; both correct=32, Standard-only=60, Graph-only=52, both wrong=656.

# 9. Historical comparison

Tables above use the actual stored historical reports/results, not prompt approximations.

# 10. Standard RL effect

Standard-v2 vs Dynamic-v1: Frozen semantic delta=-8.00 pp; consumer-reach delta=-8.87 pp; internal-E2E delta=-8.87 pp; BFCL delta=1.12 pp.
Conclusion: No overall improvement on Frozen300; only the BFCL point estimate improved. This is a historical, unpaired checkpoint comparison; no causal significance claim is made.

# 11. Additional Graph effect

Graph-v2 vs Standard-v2 verdict: NO-GO. The causal comparison is paired: full300 internal-E2E +1.69 pp, diagnosis240 +2.46 pp, heldout60 -1.42 pp, BFCL -1.00 pp.
Conclusion: No reliable general benefit over the paired Standard-v2 control.

# 12. Mechanism interpretation

- consumer_not_executed: Standard=109, Graph=145 (delta=+12.00 pp).
- conditional propagation: Standard=47.67% (174/365), Graph=50.00% (186/372).
- retry_after_error: Standard=82, Graph=81.
- Pilot transfer: The diagnosis240 propagation/E2E gain did not reproduce on heldout60; heldout directions reversed.
- Attribution nuance: Graph has significantly more failures whose first attributed root is consumer_not_executed. This is partly an attribution redistribution (especially at depth 3+, where edge outcomes are identical but missing_internal shifts to consumer_not_executed), not evidence that every count difference represents a newly unexecuted consumer.
- Mechanistic conclusion: The Graph reward shows a diagnosis-local parameter-flow signal, but it is not robust enough to justify full RL.

# 13. Final verdict

**NO-GO**

FULL_RL_NOT_STARTED

## Research questions

- Q1 Standard GRPO vs Dynamic-v1: No overall improvement on Frozen300; only the BFCL point estimate improved.
- Q2 Additional Graph reward: No reliable general benefit over the paired Standard-v2 control.
- Q3 Enter full RL: No. Gate=NO-GO; FULL_RL_NOT_STARTED.

## Limitations

- Paired task-level binary outcomes use exact McNemar tests and deterministic paired bootstrap confidence intervals.
- Edge metrics use task-cluster bootstrap and paired task-label permutation; edges within a task are not treated as independent.
- The held-out split has 60 tasks, so its confidence intervals are wide and directions are not over-interpreted.
- Conditional propagation uses reached consumers as its protocol-defined denominator; reach and internal E2E use the identical 710 gold-edge denominator.
