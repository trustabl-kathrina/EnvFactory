VERDICT = FORMAT-AUDIT-UNCERTAIN

# Cycle-3 format/parser semantic validity audit

## FORMAT AUDIT

Old format/parser in FD = 1049; all-task primary count = 1051.
TRUE_FAILURE = 24; SEMANTICALLY_VALID_PARALLEL = 123; PARTIALLY_VALID = 7; PARSER_ARTIFACT = 0; UNCERTAIN = 895.
KEEP_AS_FD = 9; DROP_FROM_FD = 123; RECLASSIFY_AND_KEEP = 22; UNCERTAIN_EXCLUDE = 895.
Independent counterfactual replays passed = 123/125.

## DATASET EFFECT

Old FD total = 1936; corrected conservative FD total = 918.
Old WRONG_ARGUMENT = 398; corrected = 407; added from format = 9.
Corrected step1/2/3/4+ = 716/192/8/2; depth>=2 = 202; mean/median/p75 = 1.234/1.0/1.0.

## PROTOCOL AND LIMITATIONS

The actual Cycle-3 collector rejected all multi-call turns; the installed verl/SGLang rollout executes parsed calls concurrently with asyncio.gather. The audit therefore separates collector behavior from semantic validity.
Positive parallel-valid labels require exact distinct read-only gold-prefix actions, no detected future-observation dependency, successful independent concurrent replay, matching responses/progress, and a gold-compatible suffix reaching the verified final state.
Other paths remain UNCERTAIN_EXCLUDE unless tool-schema or provenance evidence proves an error. This is a lower bound on valid parallel alternatives, not an estimate of all such paths.
Only server-parsed tool calls were saved, not raw token text; parser extraction artifacts cannot be proven or ruled out from these artifacts. No new model inference or training was run.
