VERDICT = FRONTIER-DATA-READY

# Cycle-3 Official EnvFactory-RL completion review

## Source and protocol

- Authoritative clean source: 2,158 tasks; attempted: 2,158; student rollouts completed: 2,157; runtime failure: 1.
- Original Dynamic-v1 model SHA256: `0231573c95e939a3dcc52518c15a7241edb32bcca65729f49230f593cec7b970`.
- Official source SHA256: `0283caf6487f8790df972f192a94cdec7d17b1ccb94aceda59d21efd562105c4`.
- One deterministic T=0 rollout per task, two isolated GPU workers, fresh MCP subprocess clients for gold and student.
- No training, optimizer update, BFCL, Frozen300/Rich24 evaluation, or 14B generation.

## Direct outcome

- FD_READY 1,936; NO_DIVERGENCE 206; TERMINAL_ONLY 12; ALIGNMENT_UNCERTAIN 3; ENV_FAILURE 1.
- First-divergence step: mean 1.111, median 1, p75 1, p90 2; step 1/2/3/4+ counts = 1,734/192/8/2.
- Primary WRONG_ARGUMENT 398 (18.44% of all tasks; 20.56% of FD_READY), WRONG_TOOL 333, PREMATURE_STOP 153, PARSER_OR_FORMAT_ERROR 1,051, LOOP_OR_REPEAT 3, EXTRA_TOOL_AFTER_COMPLETION 10.
- Conservative wrong-argument primary subtypes: extra arg 126, partial multi-arg 87, wrong entity/ID 64, wrong literal 57, user-query extraction 40, unknown 14, upstream provenance 5, type/format 5, missing required 0, stale 0. Across 398 calls: 600 wrong slots / 683 gold slots; 178 multi-argument cases.
- Independent gold replay/final-state valid 2,157. EOS-only terminal-stop encoded 2,154 from 197 environments. Three ClinicalTrialsGov terminal histories exceeded the current 16k serialization context; one other ClinicalTrialsGov gold replay hit temporary DNS resolution failure.
- FD and terminal dataset rows: 1,936 and 2,154. Frozen300/Rich24 exact overlap 0/0, inherited from the hash-locked authoritative clean-source exclusion audit. Suggested task-group split: 1,693 train / 465 validation; no training was performed.

## Quality and limitations

- All seven generated audit artifacts passed SHA256 verification; JSONL counts match the manifest. Deterministic structural spot audit: 20 wrong-argument, 10 wrong-tool, 10 premature-stop cases, 0/40 structural failures. This is not a manual semantic review.
- Of 1,058 invalid model action steps, 1,043 involved multiple tool calls in a single assistant turn and 15 hit generation length. The protocol treats multi-call output as invalid rather than executing a speculative sequence. Thus the 1,051 primary PARSER_OR_FORMAT_ERROR count is largely a one-action-per-turn protocol failure, not proof of a JSON parser defect.
- 1,734/1,936 FD samples diverge at step 1. Although the numeric data-readiness gate is met, this pool alone is not evidence that deep parameter-flow behavior improved. The 398 true wrong-argument cases are a substantial but smaller frontier.
- Observed rollout wall span 5,098 seconds (1.42 h), 1,523 successful episodes/hour. Logged GPU utilization mean: GPU0 21.4%, GPU1 23.3% across 164 samples; both held about 30 GiB while loaded. MCP/tool runtime and per-task process setup limited utilization. Both GPUs and owned processes were released after completion.

## Decision

- Q1: YES, reliable first-divergence supervision has high numeric yield, but depth coverage is shallow.
- Q2: YES, 398 wrong-argument cases make it a material frontier; it is not the dominant first failure on this distribution.
- Q3: YES for data availability only. No Cycle-3 training or model-effectiveness claim follows from this audit.
- Q4: NO immediate 14B generation is required to meet the stated Cycle-3 data gate; any later targeted generation decision should use the protocol-failure and depth caveats above.

## Reproducibility and Git

- Primary outputs: `repro_1p7b/graph_frontier/cycle3_official_rl/run_dynamic_v1_t0/audit_v1/`.
- CPU regression: 14 tests passed; `git diff --check` passed.
- Existing staged set remained 154 files. New Cycle-3 source/test files are untracked and unstaged; runtime artifacts are ignored. No commit, push, pull, PR, or reset.
