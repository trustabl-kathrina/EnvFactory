#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
BASE=$ROOT/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
FROZEN=$ROOT/repro_1p7b/logs/first_divergence_terminal_v2/frozen_ce_v2
LOG_ROOT=$ROOT/repro_1p7b/logs/first_divergence_terminal_v2
RATIO=${1:?ratio must be 1 or 2}
STEPS=${2:?steps must be 16 or 64}
if [[ "$RATIO" != 1 && "$RATIO" != 2 ]]; then echo "invalid ratio" >&2; exit 2; fi
if [[ "$STEPS" != 16 && "$STEPS" != 64 ]]; then echo "invalid steps" >&2; exit 2; fi
if [[ "$STEPS" == 16 ]]; then
  NAME=smoke_1to${RATIO}_16
else
  NAME=run_1to${RATIO}_64
  if [[ ! -f "$LOG_ROOT/smoke_1to${RATIO}_16/SMOKE_PASS.json" ]]; then
    echo "16-step smoke inference gate not passed for ratio $RATIO" >&2
    exit 2
  fi
fi
OUT=$ROOT/repro_1p7b/checkpoints/fd_terminal_v2_${NAME}
TRAIN_LOG=$LOG_ROOT/$NAME
STATUS=$LOG_ROOT/${NAME}_status
cd "$ROOT"
if [[ -e "$OUT" || -e "$TRAIN_LOG" || -e "$STATUS" ]]; then
  echo "training artifact exists; refusing overwrite: $NAME" >&2
  exit 2
fi
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$ENV/bin/python" - "$FROZEN" "$BASE" "$RATIO" "$STEPS" <<'PY'
import subprocess,sys
from pathlib import Path
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import load_and_schedule
rows,plan=load_and_schedule(Path(sys.argv[1]),Path(sys.argv[2]),ratio=int(sys.argv[3]),steps=int(sys.argv[4]))
assert len(rows)==plan["global_steps"]*2 and plan["world_size"]==2
out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
used=[int(x.strip()) for x in out.splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]):
    raise SystemExit(f"GPUs busy: {used}")
print(f"v2 preflight PASS ratio={plan['ratio_target']} steps={plan['global_steps']} "
      f"exposure={plan['exposure_counts']} GPUs={used[:2]}",flush=True)
PY
printf '%s\n' TRAINING > "$STATUS"
set +e
CUDA_VISIBLE_DEVICES=0,1 "$ENV/bin/torchrun" --standalone --nproc-per-node=2 \
  -m repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp \
  --frozen-dir "$FROZEN" --model-path "$BASE" --output "$OUT" --logs "$TRAIN_LOG" \
  --ratio "$RATIO" --steps "$STEPS"
RC=$?
set -e
if [[ "$RC" -ne 0 ]]; then
  printf 'TRAINING_FAILED:%s\n' "$RC" > "$STATUS"
  exit "$RC"
fi
"$ENV/bin/python" - "$TRAIN_LOG/result.json" "$STEPS" "$RATIO" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="TRAINED" and r["global_steps"]==int(sys.argv[2])
assert r["ratio_target"]==f"1:{sys.argv[3]}" and r["world_size"]==2
assert r["all_loss_finite"] and r["all_grad_finite"] and r["full_parameter"]
print("v2 checkpoint gate PASS",flush=True)
PY
if [[ "$STEPS" == 16 ]]; then
  printf '%s\n' TRAINED_AWAITING_INFERENCE > "$STATUS"
else
  printf '%s\n' TRAINED > "$STATUS"
fi
