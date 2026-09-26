#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/recursive_opd_v1_run
BASE=$ROOT/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
cd "$ROOT"
exec 9>"$RUN/.pipeline.lock"
flock -n 9 || { echo 'Recursive OPD pipeline already active' >&2; exit 2; }
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

gpu_free() {
  "$ENV/bin/python" - <<'PY'
import subprocess
used=[int(x.strip()) for x in subprocess.check_output(
    ['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]):
    raise SystemExit(f'GPUs busy: {used[:2]}')
print('two GPUs available',used[:2],flush=True)
PY
}

"$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v1.protocol check
printf 'WAIT_CYCLE0\n' > "$RUN/pipeline_status"
while true; do
  current=$(<"$RUN/cycle0/status")
  if [[ $current == AUDITED_full ]]; then break; fi
  if [[ $current == FAILED_* ]]; then
    printf 'BLOCKED_CYCLE0:%s\n' "$current" > "$RUN/pipeline_status"
    exit 3
  fi
  sleep 60
done

for cycle in 0 1 2; do
  current_model=$BASE
  if [[ $cycle -gt 0 ]]; then current_model=$RUN/cycle$cycle/checkpoint; fi
  next=$((cycle+1))
  if [[ $(<"$RUN/cycle$cycle/status") != AUDITED_full ]]; then
    printf 'BLOCKED_CYCLE_%s_NOT_AUDITED\n' "$cycle" > "$RUN/pipeline_status"
    exit 4
  fi
  if [[ ! -f $RUN/cycle$next/checkpoint/trainer_state.json ]]; then
    "$ENV/bin/python" - "$cycle" "$current_model" <<'PY'
import sys
from pathlib import Path
from repro_1p7b.graph_frontier.recursive_opd_v1.train import load
rows,order,plan=load(int(sys.argv[1]),Path(sys.argv[2]))
assert len(order)==128 and plan['steps']==64 and plan['world_size']==2
print('Recursive Dk data/mask/current-policy gate PASS',len(rows),flush=True)
PY
    gpu_free
    printf 'TRAIN_PI%s\n' "$next" > "$RUN/pipeline_status"
    CUDA_VISIBLE_DEVICES=0,1 "$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
      -m repro_1p7b.graph_frontier.recursive_opd_v1.train \
      --cycle "$cycle" --model "$current_model" \
      > "$RUN/cycle$cycle/train_launcher.log" 2>&1 || {
        printf 'FAILED_TRAIN_PI%s\n' "$next" > "$RUN/pipeline_status"; exit 5;
      }
  fi
  "$ENV/bin/python" - "$RUN/cycle$next/checkpoint/trainer_state.json" "$cycle" <<'PY'
import json,sys
from pathlib import Path
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256
p=Path(sys.argv[1]); state=json.loads(p.read_text())
assert state['status']=='TRAINED' and state['cycle']==int(sys.argv[2]) and state['steps']==64
assert state['model_sha256']==sha256(p.parent/'model.safetensors')
print('checkpoint gate PASS',p,flush=True)
PY
  if [[ ! -f $RUN/cycle$next/metrics/summary.json ]]; then
    printf 'COLLECT_PI%s\n' "$next" > "$RUN/pipeline_status"
    bash repro_1p7b/graph_frontier/recursive_opd_v1/run_cycle.sh \
      "$next" "$RUN/cycle$next/checkpoint" \
      > "$RUN/cycle$next/launcher.log" 2>&1 || {
        printf 'FAILED_CYCLE_PI%s\n' "$next" > "$RUN/pipeline_status"; exit 6;
      }
  fi
  if [[ ! -f $RUN/cycle$next/metrics/paired_from_cycle$cycle.json ]]; then
    "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v1.compare \
      --from-cycle "$cycle" > "$RUN/cycle$next/compare.log" 2>&1 || {
        printf 'FAILED_PAIRED_PI%s\n' "$next" > "$RUN/pipeline_status"; exit 7;
      }
  fi
  printf 'AUDITED_PI%s\n' "$next" > "$RUN/pipeline_status"
done

printf 'BUILD_STATIC_GOLD\n' > "$RUN/pipeline_status"
if [[ ! -f $RUN/static_gold/manifest.json ]]; then
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v1.static_gold \
    > "$RUN/static_gold_build.log" 2>&1 || {
      printf 'FAILED_STATIC_GOLD\n' > "$RUN/pipeline_status"; exit 8;
    }
fi
for cycle in 0 1 2; do
  current_model=$BASE
  if [[ $cycle -gt 0 ]]; then current_model=$RUN/static/cycle$cycle/checkpoint; fi
  next=$((cycle+1))
  if [[ ! -f $RUN/static/cycle$next/checkpoint/trainer_state.json ]]; then
    "$ENV/bin/python" - "$cycle" "$current_model" <<'PY'
import sys
from pathlib import Path
from repro_1p7b.graph_frontier.recursive_opd_v1.train_static import load
rows,plan=load(int(sys.argv[1]),Path(sys.argv[2]))
assert len(rows)==128 and plan['steps']==64 and plan['world_size']==2
print('fixed static SFT data gate PASS',flush=True)
PY
    gpu_free
    mkdir -p "$RUN/static/cycle$cycle"
    printf 'TRAIN_STATIC_PI%s\n' "$next" > "$RUN/pipeline_status"
    CUDA_VISIBLE_DEVICES=0,1 "$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
      -m repro_1p7b.graph_frontier.recursive_opd_v1.train_static \
      --cycle "$cycle" --model "$current_model" \
      > "$RUN/static/cycle$cycle/train_launcher.log" 2>&1 || {
        printf 'FAILED_STATIC_PI%s\n' "$next" > "$RUN/pipeline_status"; exit 9;
      }
  fi
done
if [[ ! -f $RUN/static/evaluation/metrics/summary.json ]]; then
  printf 'EVALUATE_STATIC\n' > "$RUN/pipeline_status"
  mkdir -p "$RUN/static/evaluation"
  bash repro_1p7b/graph_frontier/recursive_opd_v1/run_static_eval.sh \
    > "$RUN/static/evaluation/launcher.log" 2>&1 || {
      printf 'FAILED_STATIC_EVAL\n' > "$RUN/pipeline_status"; exit 10;
    }
fi
printf 'STATIC_EVALUATED_AWAITING_EXTERNAL\n' > "$RUN/pipeline_status"
