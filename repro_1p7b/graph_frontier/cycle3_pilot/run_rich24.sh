#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
PILOT=$ROOT/repro_1p7b/graph_frontier/cycle3_pilot
OLD=$ROOT/repro_1p7b/logs/first_divergence_onpolicy_v1
MODEL=$ROOT/repro_1p7b/checkpoints/cycle3_execfd_terminalstop_pilot
PLAN=$OLD/cycle3_heldout_greedy_plan.json
BASE=$OLD/cycle3_heldout_greedy_eval/base
OUT=$PILOT/evaluation/rich24
STATUS=$PILOT/rich24_status
PORT=8270
GPU=${1:-0}
[[ "$GPU" == 0 || "$GPU" == 1 ]] || { echo 'GPU must be 0 or 1' >&2; exit 2; }
cd "$ROOT"
[[ ! -e "$OUT" && ! -e "$STATUS" ]] || { echo 'Rich24 pilot output exists' >&2; exit 2; }
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$ENV/bin/python" - "$MODEL" "$PILOT/formal_train/result.json" "$PLAN" "$BASE" "$PORT" "$GPU" <<'PY'
import hashlib,json,socket,subprocess,sys
from pathlib import Path
model,result_path,plan_path,base_dir=map(Path,sys.argv[1:5])
port,gpu=map(int,sys.argv[5:7])
result=json.loads(result_path.read_text());plan=json.loads(plan_path.read_text())
base=json.loads((base_dir/'collection_manifest.json').read_text())
assert result['status']=='TRAINED' and result['steps']==64 and result['world_size']==2
assert hashlib.sha256((model/'model.safetensors').read_bytes()).hexdigest()==result['model_sha256']
assert plan['task_count']==24 and plan['train_task_overlap']==0
assert base['plan_sha256']==hashlib.sha256(plan_path.read_bytes()).hexdigest() and base['task_count']==24
with socket.socket() as s:
    if s.connect_ex(('127.0.0.1',port))==0: raise SystemExit('port busy')
out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
used=[int(x.strip()) for x in out.splitlines()]
if used[gpu]>=1500: raise SystemExit(f'GPU{gpu} busy: {used[gpu]}')
print('Cycle-3 Rich24 plan/checkpoint/baseline/GPU gate PASS',flush=True)
PY
mkdir -p "$OUT"
SERVER=
cleanup() {
  if [[ -n "$SERVER" ]]; then kill "$SERVER" 2>/dev/null || true; wait "$SERVER" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
printf '%s\n' STARTING_SERVER > "$STATUS"
CUDA_VISIBLE_DEVICES=$GPU "$ENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name cycle3-execfd-terminalstop-pilot \
  --host 127.0.0.1 --port "$PORT" --context-length 32768 \
  --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
  --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
  --api-key placeholder > "$OUT/server.log" 2>&1 &
SERVER=$!
"$ENV/bin/python" - "$SERVER" "$PORT" <<'PY'
import os,sys,time,urllib.request
pid=int(sys.argv[1]);port=int(sys.argv[2])
for _ in range(420):
    try: os.kill(pid,0)
    except OSError: raise SystemExit('server exited before readiness')
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
            if r.status==200: print('Rich server ready',flush=True); break
    except Exception: time.sleep(1)
else: raise SystemExit('Rich server readiness timeout')
PY
printf '%s\n' COLLECTING > "$STATUS"
"$ENV/bin/python" -m repro_1p7b.graph_frontier.evaluate_first_divergence_ce collect \
  --plan "$PLAN" --model-path "$MODEL" --served-name cycle3-execfd-terminalstop-pilot \
  --endpoint "http://127.0.0.1:$PORT/v1" --output "$OUT/trained" \
  > "$OUT/collect.log" 2>&1
printf '%s\n' SUMMARIZING > "$STATUS"
"$ENV/bin/python" -m repro_1p7b.graph_frontier.evaluate_first_divergence_ce summarize \
  --plan "$PLAN" --base "$BASE" --trained "$OUT/trained" --output "$OUT/comparison" \
  > "$OUT/summarize.log" 2>&1
printf '%s\n' COMPLETE > "$STATUS"
