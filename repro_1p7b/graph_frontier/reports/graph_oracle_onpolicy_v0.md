# Graph-Oracle On-Policy Imitation v0 — feasibility gate

## Executive summary

**VERDICT: DATA-NOT-READY.** The current selected-reference sidecar and executable environment can certify a small number of terminal STOP states, but cannot certify an exhaustive set of legal next actions or full tool arguments on arbitrary student-visited off-gold states. A hard-target CE run would therefore mostly imitate a chosen gold path or invent arguments. No GPU smoke, 64-step pilot, Frozen300 model evaluation, DPO, GRPO, PPO, or external teacher was run.

This is a bounded negative feasibility result on existing student rollouts, not proof that a stronger graph oracle is impossible. The decisive next engineering step is an executable transition oracle with typed provenance and replay certificates, not more stop preference pairs.

## Repository capability audit

- Dynamic-v1 initialization is the actual evaluated HF checkpoint at `repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b` (model weights, trainer state, frozen evaluator path, and the completed 300-probe comparison all agree). The frozen comparison measured Dynamic-v1 semantic success 57/300 (19.0%), original SFT 41/300 (13.7%), and parameter-aware SFT 55/300 (18.3%). Dynamic-v1 internal end-to-end was 237/710 (33.4%).
- `gold_sidecar.py` preserves generation-time selected-reference graph nodes/edges; `raw_tool_call` is otherwise lost on save. The sidecar is a relevant *selected path*, not a complete world model.
- `rollout_trace.py`, `FastMCPTraceAdapter`, and the existing `collect_preference_rollouts.py` capture structured calls, arguments, responses, state snapshots, and actual Dynamic-v1 actions. The audited source here is actual student rollout files; no gold prefix was relabeled as on-policy.
- `official_envs.py` in the existing verl-agent environment has isolated MCP sessions and typed events. Its terminal reward compares trace and final scenario; it does not expose a pre-action exhaustive legal-action or argument oracle.
- Existing `active_frontier_v2.legal_next_tools` computes a set from selected gold actions. That alone cannot certify alternative actions or off-gold recovery.
- Existing VeRL `algorithm.opd` requires `teacher_backend=base_no_lora`, positive `top_k`, LoRA, and teacher logits. It is not a graph-label trainer and was not repurposed. The SFT/TRL path could eventually provide assistant-only CE once complete serialized targets exist.
- Frontier v1 weakness diagnosis and resampling can choose **where** to collect tasks. It does not establish **what** local action is correct. The proposed Uniform and Frontier-guided variants would use the same oracle and objective with different task sampling distributions; neither was trained at this gate.

## Why Balanced Preference v3 stops

The last DPO data route had 513 candidate pairs but only seven observed stop pairs at the 24-chunk gate. Subsequent targeted stop collection did not make that mechanism reliable. The new question is whether a graph/environment can supervise student-visited states directly; no DPO pair gate is carried forward.

## Graph supervision capability table

| Signal (user audit A–L) | Reliable now? | Source / boundary | Hard target? |
|---|---|---|---|
| A completed required nodes | Conditional | Successful typed events **and** reference argument/provenance match; repeated names ambiguous | No if off-path |
| B remaining required nodes | Conditional | Selected sidecar minus certified prefix | No if off-path |
| C prerequisites satisfied | Conditional | Resolved selected dependency edges, certified prefix | Not sufficient alone |
| D executable/enabled nodes | No exhaustive certificate | Selected path is not full ToolGraph/world state | No |
| E terminal-ready | Yes only if current executable verifier succeeds; otherwise unknown | Env terminal verifier or exact trusted final-state match | STOP only |
| F stop correct / premature | Correct stop can be certified; false verifier alone does not identify a unique continuation | Current verifier, required prefix | STOP when true; otherwise skip |
| G next required consumer | Selected path can nominate one, not prove global uniqueness | Sidecar | No without transition certificate |
| H producer exists, consumer absent | Conditional typed observation | Event + edge provenance | Diagnostic only |
| I argument provenance | Partial; masked nondeterministic values need unique typed source certificate | Typed response/argument fields | No general argument target |
| J proposed call validity/progress/redundancy | Duplicate name detectable; semantic progress not generally certified | History + verifier | No general target |
| K multiple valid next actions | Not exhaustively enumerated | Selected-reference graph omits alternatives | Never force one hard label |
| L off-gold recovery | Unknown | No full graph transition oracle | Skip |

A tool-name-only label cannot be silently serialized into a full assistant tool call with invented arguments. A STOP action-class label is not yet a complete natural-language answer target either. Therefore current Tier-A counts are *local action-level certificates*, not CE-ready examples.

## Method and implementation

`graph_oracle_onpolicy_v0.get_local_supervision` returns Tier A deterministic, Tier B multi-valid, or Tier C skip with evidence, completed/remaining/enabled required nodes, terminal readiness, and weight. It first requires an actual student state and valid typed history. It refuses unresolved/cross-turn/repeated selected references and mismatched or masked arguments. Current executable verifier success can certify STOP even when a different valid action order completed the task. For tool selection it requires an **external exhaustive** `certified_enabled_actions` input; the selected sidecar never supplies that certificate. Multi-valid states store a positive set and no hard target. Tool observations are masked from CE by `assistant_only_labels`.

The read-only coverage auditor consumes existing Dynamic-v1 executable rollout JSON. It never fabricates a gold state. It uses the post-final verifier only for a pre-final state because the runtime's final action does not mutate the environment. It does **not** claim independent replay.

Relative to Dynamic SFT, this selects actual student-visited states rather than expert prefixes. Relative to GRPO, there is no reward/advantage optimization. Relative to DPO, there are no chosen/rejected pairs. Relative to LLM-teacher OPD, there are no teacher logits or external labels. If eventually trained, the accurate name is Graph-Oracle On-Policy Imitation with hard-target CE.

## Dataset contract and gate

A future immutable state record must contain task ID, generation/environment seed, checkpoint and policy version, trajectory ID, step index, full trajectory prefix, current observation/state, student action/type, graph sidecar identity, certified completed/remaining/enabled required nodes, terminal status, Frontier bucket/failure/depth, target type/action/positive set, oracle source/tier/evidence, replay result, serialization mask, and training weight. The current auditor emits **aggregate coverage only**, not a trainable dataset.

Conservative minimum before any GPU smoke: at least 128 unique replay-certified train decision states from at least 40 task IDs; at least 16 verified STOP and 64 complete serialized tool-call targets with both tool name and arguments certified; zero Frozen300/existing heldout overlap; no unknown provenance in Tier A; assistant-only loss mask verified. These are feasibility thresholds, not DPO stop-pair thresholds. Uniform and Frontier-guided tasks must share the same oracle and evaluation rules. Current data fail by a wide margin, so no trainer adaptation was attempted.

## Measured on-policy coverage

Source: 31 existing Dynamic-v1 executable rollout files from eight train tasks; 64 actual student decision states (depth 1: 54, depth 2: 10). Actions: 33 tool, 14 final, 17 invalid. The source contains ten semantically successful rollouts, but only nine have a state-preserving final decision suitable for the current-state STOP certificate, across four unique tasks.

| Result | Count |
|---|---:|
| Tier A action-level | 9/64 (14.1%) |
| Tier B | 0/64 |
| Tier C skipped | 55/64 (85.9%) |
| Verified STOP | 9 |
| Certified complete tool-call target | 0 |
| Selected path but no exhaustive action oracle | 42 |
| Off-reference path or unverified argument | 13 |

These 31 rollouts cover only eight task IDs and are a feasibility sample, not a statistical estimate of the 289-task Rich v3 pool. Independent replay of these old rollouts was **not** performed; the source execution was real, but no replay-valid claim is made for training. Bucket metadata in this source are unknown.

Frozen300 manifest SHA256: `4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c`, matching the locked evaluator constant. Exact task-ID overlap between the 314-task generated train set and the 300 Frozen tasks is **0**. This is an identity-level check only; a new generated training set would need the established stronger content/hash contamination audit before use.

## Tests, smoke, pilot, evaluation, regression

The 13 new CPU unit tests pass, covering terminal STOP, unique certified consumer, premature STOP, producer/consumer state, duplicate call, multi-valid no hard label, unknown action-space skip, Frozen overlap, replay-invalid filtering, observation masking, off-gold arguments, masked provenance, and unknown terminal state. CPU coverage logs are preserved under `repro_1p7b/logs/graph_oracle_onpolicy_v0/`.

The broader 102-test discovery attempt had 101 passing tests and one unrelated import error: legacy `test_guided_rl_v2_transport` imports VeRL, whose `ray` dependency is absent in the EnvFactory CPU environment. This was logged; no unrelated environment installation or test weakening was performed.

GPU smoke: **not started** (data and serialization gate failed). Two-GPU preference is recorded for a future eligible smoke/pilot. Pilot: not started. Frozen300 model evaluation: not started. Uniform vs Frontier-guided and bucket regression: no trained variants exist; not measurable. The historical Dynamic-v1 Frozen result above is context, not a new comparison.

## Limitations and next step

1. Generation-time sidecar is selected-path metadata; it cannot prove all legal or progress-making actions at an arbitrary student state.
2. Exact full tool-call CE targets require typed argument reconstruction, especially masked runtime IDs. That certificate is absent here.
3. The existing terminal verifier is terminal, not an efficient statewise oracle; successful final states are the only directly demonstrated STOP labels.
4. Current historical on-policy sample is small and lacks independent replay; no quality or effect claim is made from the 9 STOP labels.

Next: implement a *separate*, CPU-testable executable transition probe on isolated env snapshots. It must enumerate available actions or certify a unique action, validate argument provenance and state progress, and replay a stratified student sample. Re-run this coverage gate before any training. If still low, treat it as a substantive negative result rather than relaxing certainty.

## Files, logs, Git discipline

New source: `graph_oracle_onpolicy_v0.py`, `audit_graph_oracle_onpolicy_v0.py`; tests: `tests/test_graph_oracle_onpolicy_v0.py`; report: this file. Logs: `repro_1p7b/logs/graph_oracle_onpolicy_v0/coverage.json`, `coverage_stdout.log`, `cpu_tests.log` (ignored generated artifacts, retained on disk). No checkpoint or generated data was staged. Historical staged state (153 files, about 61k inserted lines) predates this task and was preserved; it was not blindly committed or reformatted. No push/pull/PR.
