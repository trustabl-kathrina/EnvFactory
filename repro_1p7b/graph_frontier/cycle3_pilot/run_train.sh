#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MODE=${1:?mode must be smoke or formal}
if [[ "$MODE" != smoke && "$MODE" != formal ]]; then
  echo 'mode must be smoke or formal' >&2
  exit 2
fi
BASE=$ROOT/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
DATA=$ROOT/repro_1p7b/graph_frontier/cycle3_pilot/data_execfd_stop_64_v1
PILOT=$ROOT/repro_1p7b/graph_frontier/cycle3_pilot
LOG=$PILOT/${MODE}_train
if [[ "$MODE" == smoke ]]; then
  OUT=$ROOT/repro_1p7b/checkpoints/cycle3_execfd_terminalstop_pilot_smoke
else
  OUT=$ROOT/repro_1p7b/checkpoints/cycle3_execfd_terminalstop_pilot
  "$ENV/bin/python" - "$PILOT/smoke_train/result.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['status']=='SMOKE_PASS' and r['steps']==2 and r['world_size']==2
assert r['all_loss_finite'] and r['all_grad_finite']
print('2-step memory sanity gate PASS',flush=True)
PY
fi
STATUS=$PILOT/${MODE}_status
cd "$ROOT"
if [[ -e "$OUT" || -e "$LOG" || -e "$STATUS" ]]; then
  echo "pilot artifact already exists; refusing overwrite: $MODE" >&2
  exit 2
fi
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$ENV/bin/python" - "$DATA" "$BASE" "$MODE" <<'PY'
import subprocess,sys
from pathlib import Path
from repro_1p7b.graph_frontier.cycle3_pilot.train import load_plan
pairs,plan=load_plan(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
assert len(pairs)==plan['steps'] and plan['world_size']==2
used=[int(x.strip()) for x in subprocess.check_output(
    ['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True
).splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]):
    raise SystemExit(f'GPU busy: {used[:2]}')
print('Cycle-3 preflight PASS',plan['mode'],plan['steps'],plan['exposures'],used[:2],flush=True)
PY
printf '%s\n' TRAINING > "$STATUS"
set +e
CUDA_VISIBLE_DEVICES=0,1 "$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  -m repro_1p7b.graph_frontier.cycle3_pilot.train \
  --data "$DATA" --model-path "$BASE" --output "$OUT" --logs "$LOG" --mode "$MODE"
RC=$?
set -e
if [[ "$RC" -ne 0 ]]; then
  printf 'TRAINING_FAILED:%s\n' "$RC" > "$STATUS"
  exit "$RC"
fi
"$ENV/bin/python" - "$LOG/result.json" "$MODE" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['status']==('SMOKE_PASS' if sys.argv[2]=='smoke' else 'TRAINED')
assert r['steps']==(2 if sys.argv[2]=='smoke' else 64)
assert r['world_size']==2 and r['full_parameter'] and r['precision']=='bfloat16'
assert r['all_loss_finite'] and r['all_grad_finite']
print('Cycle-3 checkpoint gate PASS',flush=True)
PY
printf '%s\n' COMPLETE > "$STATUS"
