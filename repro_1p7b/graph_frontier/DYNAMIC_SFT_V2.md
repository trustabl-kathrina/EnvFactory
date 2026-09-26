# Dynamic Graph-Frontier SFT v2

Dynamic v2 keeps the v1 Stage-1 boundary and changes only the Stage-2 sampler.
It uses the frozen 240-task Stage-1 diagnosis split. The held-out 60 tasks and
the final v1 300-task evaluation are forbidden allocation inputs.

## Frontier

For each depth bucket, v2 preserves raw numerators and denominators and applies
a Beta(1,1) posterior mean:

- F_prop = reach * (1 - conditional propagation)
- F_consumer = producer success * (1 - consumer after producer success)
- F_retry = smoothed tasks with retry-after-tool-error / tasks in bucket
- F_semantic = 4 * p_semantic * (1 - p_semantic)
- priority = 0.30 F_prop + 0.35 F_consumer + 0.20 F_retry + 0.15 F_semantic

Each of the four existing buckets retains a 5 percent exploration floor.

## Completion-efficiency policy

The source audit labels examples as clean_success, retry_present,
redundant_present, loop_present, or malformed_or_failed. Selection consumes
clean examples before redundant examples. Retry, loop, and malformed examples
are excluded, so the Stage-2 target does not imitate retry-after-error behavior.

## Continuation safety

Formal v2 must resume the complete native DeepSpeed state at v1 checkpoint-207:
model weights, both optimizer shards, both model-state shards, scheduler,
trainer state, and both RNG states. The launcher rejects checkpoint-390,
checkpoint-414, v1 final weights, weights-only resume, and a fresh optimizer.

If that exact state is unavailable, the machine-readable result is
BLOCKED_STAGE1_STATE_MISSING. No smoke or formal GPU process may start.
