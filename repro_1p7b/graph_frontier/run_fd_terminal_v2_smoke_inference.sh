#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
LOG=$ROOT/repro_1p7b/logs/first_divergence_terminal_v2
RATIO=${1:?ratio 1 or 2 required}
if [[ "$RATIO" != 1 && "$RATIO" != 2 ]]; then exit 2; fi
NAME=smoke_1to${RATIO}_16
RUN=$LOG/$NAME
MODEL=$ROOT/repro_1p7b/checkpoints/fd_terminal_v2_${NAME}
STATUS=$LOG/${NAME}_status
PORT=$((8170 + RATIO))
cd "$ROOT"
if [[ "$(cat "$STATUS" 2>/dev/null)" != TRAINED_AWAITING_INFERENCE ]]; then
  echo "training checkpoint not ready for smoke inference" >&2; exit 2
fi
if [[ -e "$RUN/SMOKE_PASS.json" || ! -f "$RUN/result.json" ]]; then
  echo "smoke inference already completed or training result missing" >&2; exit 2
fi
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$ENV/bin/python" - "$MODEL" "$RUN/result.json" "$PORT" <<'PY'
import json,socket,subprocess,sys
from pathlib import Path
from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256
model=Path(sys.argv[1]);result=json.loads(Path(sys.argv[2]).read_text());port=int(sys.argv[3])
assert result["status"]=="TRAINED" and result["global_steps"]==16
assert sha256(model/"model.safetensors")==result["model_weights_sha256"]
with socket.socket() as s:
    if s.connect_ex(("127.0.0.1",port))==0: raise SystemExit(f"port {port} busy")
out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
used=[int(x.strip()) for x in out.splitlines()]
if used[0]>=1500: raise SystemExit(f"GPU0 busy: {used[0]}")
print("smoke checkpoint/hash/port/GPU gate PASS",flush=True)
PY
SERVER=
cleanup() {
  if [[ -n "$SERVER" ]]; then
    kill "$SERVER" 2>/dev/null || true
    wait "$SERVER" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
printf '%s\n' INFERENCE_STARTING > "$STATUS"
CUDA_VISIBLE_DEVICES=0 "$ENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "fd-terminal-v2-smoke-$RATIO" \
  --host 127.0.0.1 --port "$PORT" --context-length 32768 \
  --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
  --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
  --api-key placeholder > "$RUN/inference_server.log" 2>&1 &
SERVER=$!
"$ENV/bin/python" - "$SERVER" "$PORT" <<'PY'
import os,sys,time,urllib.request
pid=int(sys.argv[1]);port=int(sys.argv[2])
for _ in range(420):
    try: os.kill(pid,0)
    except OSError: raise SystemExit("server exited before readiness")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as response:
            if response.status==200: print("SGLang smoke server ready",flush=True); break
    except Exception: time.sleep(1)
else: raise SystemExit("smoke server readiness timeout")
PY
printf '%s\n' INFERENCE_CHECKING > "$STATUS"
"$ENV/bin/python" -m repro_1p7b.graph_frontier.smoke_fd_terminal_v2_inference \
  --dataset "$LOG/terminal_dataset_attempt4/dataset.jsonl" \
  --endpoint "http://127.0.0.1:$PORT/v1" --model "fd-terminal-v2-smoke-$RATIO" \
  --output "$RUN/SMOKE_PASS.json"
printf '%s\n' SMOKE_PASS > "$STATUS"
