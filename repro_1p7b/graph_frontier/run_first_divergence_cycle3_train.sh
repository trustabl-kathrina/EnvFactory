#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
LOG=$ROOT/repro_1p7b/logs/first_divergence_onpolicy_v1
DATA=$LOG/cycle3_cumulative_ce_dataset
BASE=$ROOT/repro_1p7b/checkpoints/first_divergence_onpolicy_v1_cycle2_smoke
OUT=$ROOT/repro_1p7b/checkpoints/first_divergence_onpolicy_v1_cycle3_ddp64
TRAIN_LOG=$LOG/cycle3_train
STATUS=$LOG/cycle3_train_status
cd "$ROOT"
if [[ -e "$OUT" || -e "$TRAIN_LOG" || -e "$STATUS" ]]; then
  echo "Cycle-3 training artifact exists; refusing overwrite" >&2
  exit 2
fi
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$ENV/bin/python" - "$DATA" "$BASE" <<'PY'
import sys,subprocess
from pathlib import Path
from repro_1p7b.graph_frontier.train_first_divergence_ddp import load_and_schedule
selected,plan=load_and_schedule(Path(sys.argv[1]),Path(sys.argv[2]))
assert len(selected)==128 and plan["global_steps"]==64 and plan["world_size"]==2
out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
used=[int(x.strip()) for x in out.splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]): raise SystemExit(f"GPUs busy: {used}")
print(f"DDP gate passed: 50 unique states, 64x2 schedule, GPU memory {used[:2]}",flush=True)
PY
printf '%s\n' TRAINING > "$STATUS"
"$ENV/bin/python" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set \
  --cycle 3 --phase TRAIN --last-completed-action cumulative_50_replay_verified --current-job fd_v1_cycle3_ddp64 \
  --latest-checkpoint "$BASE" --latest-report "$DATA/manifest.json"
set +e
CUDA_VISIBLE_DEVICES=0,1 "$ENV/bin/torchrun" --standalone --nproc-per-node=2 \
  -m repro_1p7b.graph_frontier.train_first_divergence_ddp \
  --dataset-dir "$DATA" --model-path "$BASE" --output "$OUT" --logs "$TRAIN_LOG"
RC=$?
set -e
if [[ "$RC" -ne 0 ]]; then
  printf 'FAILED_TRAIN:%s\n' "$RC" > "$STATUS"
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set \
    --cycle 3 --phase DIAGNOSE --last-completed-action ddp64_failed --current-job none \
    --error "ddp_exit_$RC" --latest-report "$TRAIN_LOG/plan.json"
  exit "$RC"
fi
"$ENV/bin/python" - "$TRAIN_LOG/result.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="TRAINED" and r["global_steps"]==64 and r["world_size"]==2
assert r["all_loss_finite"] and r["all_grad_finite"]
print("DDP64 checkpoint gate passed",flush=True)
PY
printf '%s\n' TRAINED > "$STATUS"
"$ENV/bin/python" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set \
  --cycle 3 --phase EVALUATE --last-completed-action ddp64_checkpoint_saved \
  --current-job none --latest-checkpoint "$OUT" --latest-report "$TRAIN_LOG/result.json"
