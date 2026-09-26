# Graph-Targeted Preference Data Audit

## Verdict

**PIPELINE-INVALID — preference training was not started.**

The frozen TRAIN pool did not yield enough task-disjoint Graph-targeted pairs for the preregistered 16-pair smoke. Using preference-val examples, removing the task-level split, or retaining more than two rejected actions per normalized state would violate the protocol.

## Locked source data

- Source: frozen EnvFactory generated-RL **TRAIN** split only.
- Train tasks: 314; validation tasks: 35.
- Train SHA256: `1faabac69216efd36b4bb02822cec641fd623a6fefa5d581aa123fb441e66eeb`.
- Frozen300 manifest SHA256: `4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c`.
- Frozen300 exact overlap: **0**.
- Initialization/rollout model: Dynamic-v1.

## Rollout collection

| Phase | Tasks | Rollouts | Temperature | Purpose |
|---|---:|---:|---:|---|
| Full internal-edge pool | 34 | 544 | 0.7 | 16 rollouts per task |
| Near-frontier extension | 2 | 64 | 0.7 | 32 extra rollouts per task |
| Final diversity extension | 2 | 64 | 0.9 | 32 extra rollouts per task |
| **Total** | — | **672** | — | Structured SGLang tool calls only |

Every tool action came from the OpenAI-compatible response's structured `message.tool_calls`; raw assistant text was never regex-parsed into a call. Length-truncated generations were labeled invalid and could not become premature-stop negatives.

## Pair audit

```text
Dynamic-v1 rollouts       = 672
candidate frontier states = 8
valid Graph pairs         = 16
dropped decisions         = 1194
Standard pairs            = 24
```

Drop reasons:

| Reason | Count |
|---|---:|
| producer_not_successful | 848 |
| next_action_not_target_failure | 184 |
| internal_value_not_unique_or_mismatch | 70 |
| duplicate_state_capped | 46 |
| missing_unique_ground_truth_action | 30 |
| possible_legal_reordering | 16 |
| ambiguous_multiple_executable_consumers | 0 |

Graph failure types:

| Type | Count |
|---|---:|
| premature_stop | 16 |
| wrong_tool | 0 |
| useless_retry | 0 |
| wrong_value | 0 |

All retained pairs are depth 1. No clean depth2 or depth3+ pair was available. Environment counts are AirtableMcpServer 2, BestBuyServer 2, DrugBank 2, GoogleSheets 2, Kuaidi100 2, PriceComparison 4, and ResendEmailService 2.

## Pair quality

| Check | Result |
|---|---:|
| chosen executable | 16/16 (100%) |
| ambiguous drops | 0 |
| normalized frontier states | 8 |
| max rejected per state | 2 |
| states with two rejected actions | 8 |
| exact duplicate chosen/rejected ratio | 0% |
| unique task ratio | 50% (8/16) |
| unique prompt-state ratio | 50% (8/16) |
| unique consumer-tool ratio | 43.75% |
| Frozen300 overlap | 0 |

The human-readable sanity artifact contains all 16 available Graph pairs. The requested minimum of 20 could not be met without violating the two-rejected-per-state cap.

## Task-grouped split

| Dataset | Train pairs/tasks | Val pairs/tasks |
|---|---:|---:|
| Graph-targeted | 14 / 7 | 2 / 1 |
| Standard trajectory | 22 / 11 | 2 / 1 |

No task appears in both train and preference-val.

## Serialization and budget gate

Each row was pre-serialized with the Dynamic-v1 Qwen tool-call template and its own tool schema. Token-boundary equivalence and the 8192-token limit passed for every retained pair; serialization dropped zero rows.

| Arm | Available smoke pairs | Unique tasks | Preference tokens | Max sequence |
|---|---:|---:|---:|---:|
| Graph-targeted | 14 | 7 | 41,275 | 1,920 |
| Standard | 16 | 10 | 71,029 | 4,474 |

The Graph arm misses the required 16 train pairs, and the smallest permitted Standard smoke has approximately 72% more preference tokens. The smoke comparison is therefore not valid.

## Gate decision

- No DPO smoke.
- No Standard or Graph pilot.
- No Frozen300/BFCL checkpoint evaluation.
- No full preference training.

The data artifacts remain useful for redesigning the task generator or collecting a larger independent EnvFactory TRAIN pool, but the current frozen pool cannot answer the causal question under the preregistered protocol.

