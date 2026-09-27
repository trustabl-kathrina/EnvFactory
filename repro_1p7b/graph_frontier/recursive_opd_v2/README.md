# Recursive OPD v2 — CPU preparation

This isolated module adds exactly one natural terminal EOS retention target for
each execution-verified `SUCCESS` train task. `FIRST_ERROR` START/ADVANCE and
TERMINATE remain V1 corrections; `UNCERTAIN` yields no row. There is no class
weighting, oversampling, historical replay, or static-gold substitution.

## Immutable input and lineage

- D0 only: V1 cycle0 is the same frozen original Dynamic-v1 policy as V2 pi0.
- V1 cycle1–3 metrics are `RETROSPECTIVE_ONLY`. `source_policy_gate` refuses
  their use as V2 D1–D3 without a previous **V2** trained-checkpoint hash.
- The builder reuses V1 execution-verified verdicts and gold replay locks,
  checks episode/gold hashes, then reconstructs the exact student decision
  prefix from `prompt_message_count`. A SUCCESS final assistant message is
  excluded from the input. EOS encoding and loss-mask validation reuse the
  existing shared encoder.
- The existing 16,384-token CE limit is retained. No prefix truncation is
  performed. The original candidate `d0/` remains `NOT_READY`. A separate
  `d0_context16k_v2/` build explicitly accounts for one `CONTEXT_OVERFLOW`:
  1,115 semantic-eligible = 1,114 exact-prefix trainable + 1 overflow.
  Its independently audited manifest is `READY_FOR_GPU_TRAINING`.

## CPU commands actually permitted

From the repository root with the existing EnvFactory Python environment:

```bash
python -m repro_1p7b.graph_frontier.recursive_opd_v2.prepare prepare
python -m repro_1p7b.graph_frontier.recursive_opd_v2.audit
python -m repro_1p7b.graph_frontier.recursive_opd_v2.context_forensics
python -m repro_1p7b.graph_frontier.recursive_opd_v2.prepare prepare-context
python -m repro_1p7b.graph_frontier.recursive_opd_v2.audit --context-accounted
python -m pytest -q repro_1p7b/graph_frontier/tests/test_recursive_opd_v2.py repro_1p7b/graph_frontier/tests/test_recursive_opd_v1.py
```

`prepare` and `audit` refuse to overwrite their outputs. Runtime data lives
under ignored `recursive_opd_v2_run/`; small audit JSON is in `reports/`.

`future.py` prepares the later V2-only `lock → worker(shard 0/1) → mine-build`
sequence. It wraps V1's executable collector, gold source, and FD-v2 verifier
but writes exclusively to the separate V2 run directory and checks that the
rollout model is the previous **V2** checkpoint. Its worker needs future model
endpoints and is **not invoked** in this CPU phase. For pi1/pi2 the resulting
fresh V2 dataset can feed the same trainer; pi3 is audit-only (no Cycle4).

## GPU handoff — not executed

The future training entry requires explicit `--steps`, `--learning-rate`,
`--seed`, and `--exposure-budget` (exactly two exposures per optimizer step),
and refuses any dataset whose manifest is not `READY_FOR_GPU_TRAINING` with
all gates `PASS`. It inherits V1 full-parameter BF16 two-A100 DDP, SDPA,
gradient checkpointing, AdamW, constant LR, and grad clipping at 1.0.

After the context-accounted D0 audit and explicit GPU budget approval, the
shortest launch template is:

```bash
torchrun --standalone --nproc_per_node=2 -m repro_1p7b.graph_frontier.recursive_opd_v2.train \
  --cycle 0 \
  --dataset repro_1p7b/graph_frontier/recursive_opd_v2_run/d0_context16k_v2 \
  --model /home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b \
  --output /home/u2024311031/workspace/envfactory_repro_1p7b/repro_1p7b/graph_frontier/recursive_opd_v2_run/cycle1/checkpoint \
  --steps 64 --learning-rate 2e-6 \
  --seed 20260927 --exposure-budget 128
```

This handoff command is documented but was **not run in CPU preparation**.
The GPU-stage protocol must separately check hardware, source hashes and
optimizer-free smoke before executing it. After V2 pi1 exists, a **new**
on-policy V2 pi1 rollout is required before D1 mining. V1
pi1/pi2/pi3 rollouts cannot be substituted. No V2 later-cycle rollout or
training was performed in this CPU preparation stage.
