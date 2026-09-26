VERDICT = FORMAT-AUDIT-UNCERTAIN

# Cycle-3 FORMAT/PARSER Semantic Validity Audit

## Protocol finding

The actual Cycle-3 collector treated any turn with more than one parsed tool call as `parallel_tool_calls` and executed none. It was not the verl AgentLoop. The installed verl/SGLang rollout does support multiple calls in one turn via `asyncio.gather`; the inspected execution block has no explicit `max_parallel_calls` guard. Therefore collector rejection alone does not prove semantic error in a parallel-capable agent loop. Only server-parsed structured calls, not raw generation tokens, were persisted; parser-extraction artifacts cannot be confirmed or excluded.

## FORMAT AUDIT

The original all-task taxonomy had 1,051 primary `PARSER_OR_FORMAT_ERROR` cases. Of these, 1,049 were in the 1,936-row FD dataset: 1,040 multiple-tool-call turns and 9 generation-length/no-call cases. Two other format cases were not FD samples.

| Semantic class | Count | Evidence / treatment |
| --- | ---: | --- |
| TRUE_FAILURE | 24 | 9 truncated/no executable call, 14 schema-invalid multi-calls, 1 future-observation dependency |
| SEMANTICALLY_VALID_PARALLEL | 123 | Exact, distinct, read-only gold-prefix calls; independent concurrent replay and verified suffix passed |
| PARTIALLY_VALID | 7 | At least one exact gold call plus a schema-invalid proposed call |
| PARSER_ARTIFACT | 0 confirmed | Raw token stream unavailable; zero is not proof of absence |
| UNCERTAIN | 895 | Alternative path, conflict, or correctness not provable from saved evidence |

Disposition within the 1,049 old format FD rows: `KEEP_AS_FD=9`, `DROP_FROM_FD=123`, `RECLASSIFY_AND_KEEP=22`, `UNCERTAIN_EXCLUDE=895`. The 123 drops are a proven lower bound of false-positive exact-sequential FD labels **for a parallel-capable execution policy**. The 895 uncertain cases are not counted as false positives or true failures.

## Counterfactual replay

An automatic, conservative screen selected 125 cases whose proposed calls exactly matched distinct read-only gold-prefix actions and had no detected same-turn future-observation dependency. Each was replayed from a fresh `initial_config` with the common gold prefix restored, then calls were executed concurrently using the installed verl-style `asyncio.gather` semantics. A pass required all tool calls to succeed, exact response and state agreement with gold progress, and successful replay of the remaining gold suffix to the verified final state. `123/125` passed. The other two had response mismatch despite successful calls and matching saved state; they remain uncertain. Mutating or non-gold alternatives were not certified by this rule.

## DATASET EFFECT

| Metric | Original | Corrected conservative subset |
| --- | ---: | ---: |
| FD rows | 1,936 | 918 |
| WRONG_ARGUMENT | 398 | 407 (+9 local-next-gold-tool schema errors) |
| Step 1 FD | 1,734 | 716 |
| Step 2 / 3 / 4+ FD | 192 / 8 / 2 | 192 / 8 / 2 |
| Depth >=2 | 202 | 202 |
| Mean first-divergence step | 1.111 | 1.234 |

Corrected kept taxonomy: `WRONG_ARGUMENT=398`, `WRONG_ARGUMENT_IN_MULTI_CALL=9`, `WRONG_TOOL=333`, `PREMATURE_STOP=153`, `TRUE_PROTOCOL_FORMAT_ERROR=9`, `TOOL_SCHEMA_INVALID_IN_MULTI_CALL=12`, `DEPENDENCY_VIOLATION=1`, `LOOP_OR_REPEAT=3`. The 918-row result is a **certified conservative manifest**, not an estimate that all excluded rows were harmless. It has 716/192/8/2 rows at steps 1/2/3/4+.

## Sanity and integrity

Deterministic stratified structural review selected 20 true-failure, 20 replay-proven valid-parallel, and 20 reclassified rows; structural evidence checks found `0/60` errors. All 9 reclassified wrong-argument rows were additionally checked to use the local next gold tool name and have concrete schema errors (`0/9` failures). There were no confirmed parser artifacts to sample. This is a structural review, not human proof of semantic correctness for uncertain paths.

The original FD file SHA256 remains `5304b0b4f4b3ee8527a56d7c74a0e0b338bc6db0b8a2d8f830856318bc71b7ac`. The 125 replay records are hash-locked (`89926f3e2b5adf8d9b58551f509c428b8b6dd3436ccd5d17a375aca189fd2127`). Corrected manifest has 1,936 unique decisions and exactly 918 kept; classification counts sum to 1,049. Original rollouts and first-divergence files were not changed. Outputs are in `cycle3_official_rl/run_dynamic_v1_t0/semantic_audit_v2/`; the first-pass v1 audit is preserved separately.

## Decision

- Q1 — Are many step-1 multi-tool outputs genuinely policy failures? **PARTIAL / UNRESOLVED**: 31 format rows have positive failure or partial-error evidence, 123 are replay-proven parallel-valid, and 895 remain uncertain. Under the actual one-call Cycle-3 collector, all multi-call turns were rejected; under the installed parallel-capable verl executor, that rejection is not a semantic verdict.
- Q2 — Did exact-gold sequential alignment create clear false-positive FD? **YES, at least 123** for the parallel-capable policy. It is not defensible to call all 895 uncertain rows false positives.
- Q3 — Is the corrected dataset ready for Frontier-balanced training design? **NOT YET**. The 918 certified rows are real data, but 895 ambiguous format rows and missing raw-token evidence prevent a trustworthy full taxonomy. No sampling ratio or training is proposed here.

No GPU, model inference, new task generation, training, commit, push, pull, or PR was used. Existing staged Git contents were preserved.
