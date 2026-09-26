VERDICT = TERMINAL-STOP-PIPELINE-READY

# Terminal-Stop EOS-only CE smoke (2026-09-25)

This is a pipeline smoke, not an effectiveness evaluation. No formal Cycle-3,
Official-RL mass rollout, Frozen300/Rich24 benchmark, commit, push, or full
training was performed.

## Implementation and integrity

- Current Dynamic-v1 Qwen3-1.7B tokenizer resolves the assistant end token
  `<|im_end|>` to ID 151645. The terminal-stop branch asserts this checkpoint
  identity but obtains the token from the tokenizer; no new token or tokenizer
  or inference-protocol change was made.
- `terminal_stop` is separate from the unchanged ordinary FD mask. The
  terminal history includes the user query, gold assistant tool calls, typed
  tool observations, and the final successful observation. The empty assistant
  turn is the target. Exactly one active label, 151645, is required; the
  template's trailing newline token 198 is masked to -100.
- 16 independent Official EnvFactory-RL replay-success clean terminal tasks
  were re-executed with isolated MCP clients, observed responses captured,
  initial/final scenario checks repeated, and final-state verifier passed.
  They span 16 distinct environments. One additional candidate exceeded the
  16k token gate and was rejected. Official release SHA256:
  `0283caf6487f8790df972f192a94cdec7d17b1ccb94aceda59d21efd562105c4`.
- 16 existing verified FD examples: wrong_arguments 6, wrong_tool 5,
  premature_stop 5. Total 32 unique task IDs, 25 environments across both
  types. Frozen300 overlap 0; Rich24 overlap 0.
- Whole-dataset CPU collator audit: shape [32, 2744], no truncation,
  all 16 EOS labels present, prompt/history and padding masked. Token tail
  samples are recorded in the immutable smoke manifest. New tests 6/6 and
  existing targeted tests 20/20; Python compile check passed.

## Two-GPU training smoke

- Initialization: original Dynamic-v1 full-parameter checkpoint
  `repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b`
  (weights SHA256 `0231573c95e939a3dcc52518c15a7241edb32bcca65729f49230f593cec7b970`).
- GPUs: A100 40GB 0 and 1, DDP, BF16, AdamW, LR 2e-6, per-rank batch 1,
  8 optimizer steps. Each step had one verified FD example on rank 0 and one
  EOS-only terminal-stop example on rank 1: 8 of each actually seen.
- All 8 rank-1 terminal-stop losses were finite (range 8.533–25.167);
  backward/optimizer completed, all gradient norms finite. Peak GPU memory
  19.275 GiB. Mean loss first/last 5.783/10.239; this is not a learning
  curve claim. Temporary checkpoint saved at
  `repro_1p7b/checkpoints/terminal_stop_smoke_20260925_v1`
  (weights SHA256 `bcd965cc6c3631ca494b032c16fbf9f455d08aa5afd7cc2010f8ab5f2d1e2fb8`).

## Deterministic inference sanity

The first 64-token diagnostic capped two lengthy natural-language responses;
it was inconclusive for end-of-turn behavior. A bounded retry used two
different unexposed terminal states (AmazonSES, AttomRealEstate) and a 256-token
cap. Base and smoke checkpoints both reloaded. All 4 generations ended with
`<|im_end|>`; parser errors 0. The existing AgentLoop-style classifier
returned final for three no-tool-call outputs. The smoke checkpoint's fourth
output was a valid typed tool call and classified as tool, not final. Thus the
integration works, but this smoke does not show improved stopping ability.

Q1: YES — terminal-state EOS-only CE is now a legal sample in the existing
trainer. Q2: YES — under the current EnvFactory evaluator/AgentLoop semantics,
gold natural-language final text is not required to teach cessation of tool
calling. Q3: YES, technically ready to design the next gated Official-RL
on-policy FD plus replay-backed stop collection; do not infer method efficacy
or launch Cycle-3 from this smoke alone.

Evidence: `repro_1p7b/graph_frontier/terminal_stop_smoke/official_fd16_stop16_20260925_v1/manifest.json`,
`repro_1p7b/logs/terminal_stop_smoke_20260925_v1/result.json`,
`repro_1p7b/logs/terminal_stop_smoke_20260925_v1/inference_sanity_v2.json`.
Dataset, logs, and checkpoint remain unstaged. The existing 154-file staged
index was not changed; no commit or push occurred.
