#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
BASE=$ROOT/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
NEW=$ROOT/repro_1p7b/checkpoints/first_divergence_onpolicy_v1_cycle2_smoke
LOG=$ROOT/repro_1p7b/logs/first_divergence_onpolicy_v1
EVAL=$LOG/cycle2_heldout_eval
PLAN=$LOG/cycle2_heldout_plan.json
STATUS=$LOG/cycle2_eval_status
cd "$ROOT"
if [[ -e "$EVAL" || -e "$STATUS" ]]; then echo "eval artifacts already exist" >&2; exit 2; fi
mkdir -p "$EVAL"
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$ENV/bin/python" - "$PLAN" "$BASE" "$NEW" "$LOG/cycle2_train/result.json" <<'PY'
import json,sys,socket,subprocess
from pathlib import Path
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import checked_plan
from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256
plan,_,_,_=checked_plan(Path(sys.argv[1]))
result=json.loads(Path(sys.argv[4]).read_text())
assert plan["task_count"]==24 and plan["train_task_overlap"]==0
assert sha256(Path(sys.argv[2])/"model.safetensors")==plan["model_weights_sha256"]
assert sha256(Path(sys.argv[3])/"model.safetensors")==result["model_weights_sha256"]
for port in (8130,8131):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1",port))==0: raise SystemExit(f"port {port} in use")
out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
used=[int(x.strip()) for x in out.splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]): raise SystemExit(f"GPUs busy: {used}")
print("plan, checkpoint hashes, ports, GPUs verified",flush=True)
PY
SERVER0= SERVER1= WORKER0= WORKER1=
cleanup() {
  for pid in "$WORKER0" "$WORKER1" "$SERVER0" "$SERVER1"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
  for pid in "$WORKER0" "$WORKER1" "$SERVER0" "$SERVER1"; do
    if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT INT TERM
printf '%s\n' STARTING_SERVERS > "$STATUS"
for gpu in 0 1; do
  if [[ "$gpu" -eq 0 ]]; then MODEL=$BASE; NAME=dynamic-v1; else MODEL=$NEW; NAME=first-divergence-v1; fi
  CUDA_VISIBLE_DEVICES=$gpu "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$NAME" \
    --host 127.0.0.1 --port "$((8130 + gpu))" --context-length 32768 \
    --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "$EVAL/server_gpu$gpu.log" 2>&1 &
  if [[ "$gpu" -eq 0 ]]; then SERVER0=$!; else SERVER1=$!; fi
done
"$ENV/bin/python" - "$SERVER0" "$SERVER1" <<'PY'
import os,sys,time,urllib.request
for port,pid in zip((8130,8131),map(int,sys.argv[1:])):
    for _ in range(420):
        try: os.kill(pid,0)
        except OSError: raise SystemExit(f"server {port} exited before readiness")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as response:
                if response.status==200: print(f"server {port} ready",flush=True); break
        except Exception: time.sleep(1)
    else: raise SystemExit(f"server {port} readiness timeout")
PY
printf '%s\n' COLLECTING > "$STATUS"
for gpu in 0 1; do
  if [[ "$gpu" -eq 0 ]]; then MODEL=$BASE; NAME=dynamic-v1; LABEL=base; else MODEL=$NEW; NAME=first-divergence-v1; LABEL=trained; fi
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.evaluate_first_divergence_ce collect \
    --plan "$PLAN" --model-path "$MODEL" --served-name "$NAME" \
    --endpoint "http://127.0.0.1:$((8130 + gpu))/v1" \
    --output "$EVAL/$LABEL" > "$EVAL/collect_$LABEL.log" 2>&1 &
  if [[ "$gpu" -eq 0 ]]; then WORKER0=$!; else WORKER1=$!; fi
done
set +e
wait "$WORKER0"; RC0=$?
wait "$WORKER1"; RC1=$?
set -e
WORKER0= WORKER1=
if [[ "$RC0" -ne 0 || "$RC1" -ne 0 ]]; then
  printf 'FAILED_COLLECT:%s:%s\n' "$RC0" "$RC1" > "$STATUS"
  exit 3
fi
printf '%s\n' SUMMARIZING > "$STATUS"
if ! "$ENV/bin/python" -m repro_1p7b.graph_frontier.evaluate_first_divergence_ce summarize \
  --plan "$PLAN" --base "$EVAL/base" --trained "$EVAL/trained" \
  --output "$EVAL/comparison" > "$EVAL/summarize.log" 2>&1; then
  printf '%s\n' FAILED_SUMMARIZE > "$STATUS"
  exit 4
fi
printf '%s\n' COMPLETE > "$STATUS"
"$ENV/bin/python" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set --cycle 2 --phase DIAGNOSE --last-completed-action heldout_eval_complete --current-job none --latest-checkpoint "$NEW" --latest-report "$EVAL/comparison/metrics.json"
