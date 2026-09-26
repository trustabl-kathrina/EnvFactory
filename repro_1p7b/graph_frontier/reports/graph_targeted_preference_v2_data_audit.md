# Graph-targeted preference v2 data audit

## Verdict

**PARTIAL-DATA-READY**

Active construction used only the frozen EnvFactory generated-RL TRAIN split.  It replayed real gold prefixes, captured the producer observation, and validated every chosen consumer call in a cloned real environment state.  Frozen300 and BFCL were not used as data.

## Retention funnel

| Stage | Count |
|---|---:|
| Frozen TRAIN tasks | 314 |
| All gold dependency edges | 37 |
| Internal required edges | 33 |
| Static high-confidence edges | 20 |
| Replay-valid / chosen-executable states | 17 |
| States with at least one retained negative | 9 |
| Dynamic-v1 actions sampled | 68 |
| Valid preference pairs | 18 |

## Gates

| Gate | Observed | Required | Pass |
|---|---:|---:|---|
| Unique frontier states | 9 | 128 | False |
| Preference pairs | 18 | 256 | False |
| Chosen executable rate | 1.000 | 1.000 | True |
| Frozen300 overlap | 0 | 0 | True |

## Distributions

- Failure types: `{"premature_stop": 18}`
- Depth: `{"1": 18}`
- Environments: `{"DrugBank": 2, "GoogleSheets": 2, "Kuaidi100": 4, "PayPalPaymentProcessor": 2, "PriceComparison": 4, "ResendEmailService": 2, "WhatsApp": 2}`
- Sample classifications: `{"ambiguous_same_tool": 10, "correct": 25, "premature_stop": 33}`

## Hard bottleneck

The source TRAIN pool contains only **33 internal required edges** and only **20 static high-confidence immediate frontiers**.  Therefore the protocol's 128-state/256-pair gate cannot be reached without generating new graph-bearing EnvFactory tasks or fabricating duplicate states; neither is permitted in this round.

## Protocol locks

- Frozen TRAIN SHA256: `1faabac69216efd36b4bb02822cec641fd623a6fefa5d581aa123fb441e66eeb`
- Frozen300 manifest SHA256: `4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c`
- Frozen300 exact overlap: `0`
- Training started: `false`
