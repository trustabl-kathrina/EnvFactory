VERDICT = EXEC-VERIFIED-PARTIAL

# Cycle-3 Execution-Verified FD v2

## REGRESSION

known candidates = 125; known valid parallel = 123; validator reproduced = 125; disagreements = 0.

## OLD VS NEW

| Metric | Old | Exec-Verified v2 |
|---|---:|---:|
| FD total | 1936 | 998 |
| Conservative usable FD | 918 | 989 |
| step1 | 1734 | 796 |
| step2 | 192 | 192 |
| step3 | 8 | 8 |
| step4+ | 2 | 2 |
| depth>=2 | 202 | 202 |
| WRONG_ARGUMENT | 407 | 413 |
| uncertain multi-call | 895 previous semantic audit | 615 |

Mean/median gold divergence step = 1.215/1.0.
New WRONG_ARGUMENT = 398 unchanged direct + 15 multi-call reclassified.

## MULTI-CALL

Candidates = 1040; valid parallel = 314; valid fixed-suffix alternatives = 9; jointly state-equivalent extra-call cases = 165 (individual read-only effects not separately proven).
Dependency violations = 1; wrong arguments = 15; wrong tools = 5; execution failures = 81; still uncertain = 615.
Previous 895 uncertain resolved as valid parallel = 191, valid fixed-suffix alternative = 9, execution failure = 72, wrong argument = 3, wrong tool = 5, still uncertain = 615.

## FRONTIER MIGRATION

Newly exposed deeper FD = 0.
The saved Cycle-3 collector rejects multi-call output and stops immediately: zero multi-call episodes contain a later saved student turn. Re-alignment is implemented and CPU-tested, but deeper model failures cannot be observed from these fixed artifacts without new inference.

## LIMITATIONS AND ANSWERS

Q1: YES. Independent replay removes many exact-sequence false positives.
Q2: PARTIAL. Same-state gold-compatible cases are validated without search; unproved alternative paths remain excluded.
Q3: NO observable deeper FD in these saved traces, because the collector terminated every multi-call turn.
Q4: NOT YET for a complete Cycle-3 training composition; the high-confidence FD subset may inform a later design, but uncertainty and truncated continuations remain.
Raw generation tokens were not saved, so parser artifacts cannot be conclusively excluded even though none was confirmed. Nine finish_reason=length budget truncations remain in the FD taxonomy but are excluded from training. No model inference, training, or search was run.
Terminal-stop and original FD SHA256 values remained unchanged: 806597e4ba414607eaa2ebeaf0195ebee111cf0449276660f5f7ebe5fb118697 and 5304b0b4f4b3ee8527a56d7c74a0e0b338bc6db0b8a2d8f830856318bc71b7ac.
