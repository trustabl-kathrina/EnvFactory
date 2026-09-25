#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
SERVER_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN_PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
MODEL=$ROOT/repro_1p7b/checkpoints/first_divergence_onpolicy_v1_cycle2_smoke
MANIFEST=$ROOT/repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl
BASE=$ROOT/repro_1p7b/results/graph_frontier/confirm_300/dynamic_v1
LOG=$ROOT/repro_1p7b/logs/first_divergence_onpolicy_v1
OUT=$LOG/frozen300_cycle2
STATUS=$LOG/cycle2_frozen300_status
cd "$ROOT"
if [[ -e "$OUT" || -e "$STATUS" ]]; then echo "Frozen300 output exists" >&2; exit 2; fi
export CUDA_HOME=$SERVER_ENV
export PATH=$SERVER_ENV/bin:$PATH
export LD_LIBRARY_PATH=$SERVER_ENV/lib:$SERVER_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT
export MCP_CONFIG_PATH=$ROOT/configs/mcp_server.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LITELLM_LOCAL_MODEL_COST_MAP=True
"$RUN_PY" - "$MANIFEST" "$MODEL" "$BASE" <<'PY'
import hashlib,json,sys,socket,subprocess
from pathlib import Path
manifest,model,base=map(Path,sys.argv[1:])
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=="4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c"
r=json.loads(Path("repro_1p7b/logs/first_divergence_onpolicy_v1/cycle2_train/result.json").read_text())
assert hashlib.sha256((model/"model.safetensors").read_bytes()).hexdigest()==r["model_weights_sha256"]
assert json.loads((base/"run_summary.json").read_text())["valid"]==300
for port in (8160,8161):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1",port))==0: raise SystemExit(f"port {port} busy")
out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
used=[int(x.strip()) for x in out.splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]): raise SystemExit(f"GPUs busy: {used}")
print("Frozen300 manifest, checkpoint, baseline, ports and GPU gate PASS",flush=True)
PY
mkdir -p "$OUT"
printf '%s\n' EVALUATING > "$STATUS"
"$RUN_PY" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set \
  --cycle 2 --phase EXTERNAL_EVAL --last-completed-action cycle2_checkpoint_independent_confirmation \
  --current-job fd_v1_frozen300 --latest-checkpoint "$MODEL" \
  --latest-report "$LOG/cycle3_heldout_eval/comparison/metrics.json"
run_shard() (
  set -euo pipefail
  gpu=$1; shard=$1; port=$((8160 + gpu)); key=fd-cycle3-frozen-$gpu
  SERVER_PID=
  cleanup() {
    if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi
  }
  trap cleanup EXIT INT TERM
  CUDA_VISIBLE_DEVICES=$gpu "$SERVER_ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port "$port" --api-key "$key" \
    --dtype bfloat16 --tp-size 1 --mem-fraction-static 0.70 \
    --disable-cuda-graph --attention-backend triton --sampling-backend pytorch \
    --context-length 32768 --max-total-tokens 49152 --trust-remote-code \
    > "$OUT/server_shard$shard.log" 2>&1 &
  SERVER_PID=$!
  "$RUN_PY" - "$port" "$key" "$SERVER_PID" <<'PY'
import os,sys,time,requests
port,key,pid=sys.argv[1],sys.argv[2],int(sys.argv[3])
url=f"http://127.0.0.1:{port}/v1/models"
for _ in range(300):
    try: os.kill(pid,0)
    except OSError: raise SystemExit(f"server {port} died")
    try:
        if requests.get(url,headers={"Authorization":"Bearer "+key},timeout=2).status_code==200:
            print(f"server {port} ready",flush=True); break
    except Exception: pass
    time.sleep(1)
else: raise SystemExit(f"server {port} timeout")
PY
  export SGLANG_BASE_URL=http://127.0.0.1:$port/v1
  export SGLANG_API_KEY=$key
  export SGLANG_MODEL=$MODEL
  "$RUN_PY" -m repro_1p7b.graph_frontier.eval_first_divergence_frozen300_shard \
    --label first_divergence_cycle2 --model-path "$MODEL" \
    --manifest "$MANIFEST" --output-dir "$OUT" --shard-index "$shard" \
    > "$OUT/eval_shard$shard.log" 2>&1
)
run_shard 0 &
P0=$!
run_shard 1 &
P1=$!
set +e
wait "$P0"; RC0=$?
wait "$P1"; RC1=$?
set -e
printf '%s %s\n' "$RC0" "$RC1" > "$OUT/shard_exit_codes"
if [[ "$RC0" -ne 0 || "$RC1" -ne 0 ]]; then
  printf 'FAILED_SHARD:%s:%s\n' "$RC0" "$RC1" > "$STATUS"
  "$RUN_PY" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set \
    --cycle 2 --phase DIAGNOSE --last-completed-action frozen300_shard_failed \
    --current-job none --error "frozen300_exit_$RC0:$RC1"
  exit 3
fi
printf '%s\n' COMPARING > "$STATUS"
if ! "$RUN_PY" -m repro_1p7b.graph_frontier.compare_first_divergence_frozen300 \
  --manifest "$MANIFEST" --base "$BASE" --trained "$OUT" --output "$OUT/comparison" \
  > "$OUT/comparison.log" 2>&1; then
  printf '%s\n' FAILED_AUDIT > "$STATUS"
  exit 4
fi
printf '%s\n' COMPLETE > "$STATUS"
"$RUN_PY" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set \
  --cycle 2 --phase DIAGNOSE --last-completed-action frozen300_comparison_complete \
  --current-job none --latest-checkpoint "$MODEL" --latest-report "$OUT/comparison/summary.json"
