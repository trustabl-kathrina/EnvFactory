#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN_PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
RUN=$ROOT/repro_1p7b/graph_frontier/recursive_opd_v1_run
LABEL=${1:?recursive_pi3 or static_pi3 required}
case $LABEL in
  recursive_pi3) MODEL=$RUN/cycle3/checkpoint; STATE=$MODEL/trainer_state.json;;
  static_pi3) MODEL=$RUN/static/cycle3/checkpoint; STATE=$MODEL/trainer_state.json;;
  *) echo 'invalid model label' >&2; exit 2;;
esac
OUT=$RUN/evaluation/$LABEL
FROZEN=$ROOT/repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl
FROZEN_BASE=$ROOT/repro_1p7b/results/graph_frontier/confirm_300/dynamic_v1
RICH_PLAN=$ROOT/repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_heldout_greedy_plan.json
RICH_BASE=$ROOT/repro_1p7b/logs/first_divergence_onpolicy_v1/cycle3_heldout_greedy_eval/base
cd "$ROOT"
mkdir -p "$OUT"
exec 9>"$OUT/.launcher.lock"
flock -n 9 || { echo 'external evaluation already running' >&2; exit 2; }
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official
export MCP_CONFIG_PATH=$ROOT/configs/mcp_server.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LITELLM_LOCAL_MODEL_COST_MAP=True PYTHONUNBUFFERED=1
"$ENV/bin/python" - "$LABEL" "$MODEL" "$STATE" "$FROZEN" "$FROZEN_BASE" "$RICH_PLAN" "$RICH_BASE" <<'PY'
import hashlib,json,socket,subprocess,sys
from pathlib import Path
label=sys.argv[1];model,state,frozen,base,rich_plan,rich_base=map(Path,sys.argv[2:])
r=json.loads(state.read_text())
assert r['status']=='TRAINED' and r['steps']==64
assert (r.get('cycle')==2 if label=='recursive_pi3' else r.get('static_phase')==2)
assert hashlib.sha256((model/'model.safetensors').read_bytes()).hexdigest()==r['model_sha256']
assert hashlib.sha256(frozen.read_bytes()).hexdigest()=='4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c'
assert json.loads((base/'run_summary.json').read_text())['valid']==300
p=json.loads(rich_plan.read_text());b=json.loads((rich_base/'collection_manifest.json').read_text())
assert p['task_count']==24 and p['train_task_overlap']==0
assert b['plan_sha256']==hashlib.sha256(rich_plan.read_bytes()).hexdigest() and b['task_count']==24
for port in (8370,8371,8380):
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',port))==0: raise SystemExit(f'port {port} busy')
used=[int(x.strip()) for x in subprocess.check_output(
    ['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]): raise SystemExit(f'GPUs busy: {used[:2]}')
print('external evaluation checkpoint/manifest/GPU gate PASS',label,flush=True)
PY

if [[ ! -f $OUT/frozen300/comparison/summary.json ]]; then
  [[ ! -e $OUT/frozen300 ]] || { echo 'partial Frozen300 output: inspect before resume' >&2; exit 3; }
  mkdir -p "$OUT/frozen300"
  printf 'FROZEN300\n' > "$OUT/status"
  run_shard() (
    set -euo pipefail
    gpu=$1;port=$((8370+gpu));key=recursive-opd-$LABEL-$gpu
    server=
    cleanup() { [[ -z $server ]] || { kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; }; }
    trap cleanup EXIT INT TERM
    CUDA_VISIBLE_DEVICES=$gpu "$ENV/bin/python" -m sglang.launch_server \
      --model-path "$MODEL" --host 127.0.0.1 --port "$port" --api-key "$key" \
      --dtype bfloat16 --tp-size 1 --mem-fraction-static 0.70 \
      --disable-cuda-graph --attention-backend triton --sampling-backend pytorch \
      --context-length 32768 --max-total-tokens 49152 --trust-remote-code \
      > "$OUT/frozen300/server_shard$gpu.log" 2>&1 &
    server=$!
    "$RUN_PY" - "$port" "$key" "$server" <<'PY'
import os,requests,sys,time
port,key,pid=sys.argv[1],sys.argv[2],int(sys.argv[3])
for _ in range(300):
    try: os.kill(pid,0)
    except OSError: raise SystemExit(f'server {port} died')
    try:
        r=requests.get(f'http://127.0.0.1:{port}/v1/models',headers={'Authorization':'Bearer '+key},timeout=2)
        if r.status_code==200: print(f'server {port} ready',flush=True); break
    except Exception: pass
    time.sleep(1)
else: raise SystemExit(f'server {port} timeout')
PY
    export SGLANG_BASE_URL=http://127.0.0.1:$port/v1
    export SGLANG_API_KEY=$key SGLANG_MODEL=$MODEL
    "$RUN_PY" -m repro_1p7b.graph_frontier.eval_first_divergence_frozen300_shard \
      --label "$LABEL" --model-path "$MODEL" --manifest "$FROZEN" \
      --output-dir "$OUT/frozen300" --shard-index "$gpu" \
      > "$OUT/frozen300/eval_shard$gpu.log" 2>&1
  )
  run_shard 0 & P0=$!
  run_shard 1 & P1=$!
  set +e
  wait "$P0"; RC0=$?
  wait "$P1"; RC1=$?
  set -e
  if [[ $RC0 -ne 0 || $RC1 -ne 0 ]]; then
    printf 'FAILED_FROZEN_SHARD:%s:%s\n' "$RC0" "$RC1" > "$OUT/status"; exit 4
  fi
  "$RUN_PY" -m repro_1p7b.graph_frontier.compare_first_divergence_frozen300 \
    --manifest "$FROZEN" --base "$FROZEN_BASE" --trained "$OUT/frozen300" \
    --output "$OUT/frozen300/comparison" > "$OUT/frozen300/comparison.log" 2>&1 || {
      printf 'FAILED_FROZEN_COMPARE\n' > "$OUT/status"; exit 5;
    }
fi

if [[ ! -f $OUT/rich24/comparison/metrics.json ]]; then
  [[ ! -e $OUT/rich24 ]] || { echo 'partial Rich24 output: inspect before resume' >&2; exit 6; }
  mkdir -p "$OUT/rich24"
  printf 'RICH24\n' > "$OUT/status"
  server=
  rich_cleanup() { [[ -z $server ]] || { kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; }; }
  trap rich_cleanup EXIT INT TERM
  CUDA_VISIBLE_DEVICES=0 "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$LABEL" \
    --host 127.0.0.1 --port 8380 --context-length 32768 \
    --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "$OUT/rich24/server.log" 2>&1 &
  server=$!
  "$ENV/bin/python" - "$server" <<'PY'
import os,sys,time,urllib.request
pid=int(sys.argv[1])
for _ in range(420):
    try: os.kill(pid,0)
    except OSError: raise SystemExit('Rich24 server died')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8380/health',timeout=2) as r:
            if r.status==200: print('Rich24 server ready',flush=True); break
    except Exception: time.sleep(1)
else: raise SystemExit('Rich24 server timeout')
PY
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.evaluate_first_divergence_ce collect \
    --plan "$RICH_PLAN" --model-path "$MODEL" --served-name "$LABEL" \
    --endpoint http://127.0.0.1:8380/v1 --output "$OUT/rich24/trained" \
    > "$OUT/rich24/collect.log" 2>&1
  rich_cleanup
  server=
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.evaluate_first_divergence_ce summarize \
    --plan "$RICH_PLAN" --base "$RICH_BASE" --trained "$OUT/rich24/trained" \
    --output "$OUT/rich24/comparison" > "$OUT/rich24/summarize.log" 2>&1 || {
      printf 'FAILED_RICH_SUMMARY\n' > "$OUT/status"; exit 7;
    }
fi
printf 'AUDITED\n' > "$OUT/status"
