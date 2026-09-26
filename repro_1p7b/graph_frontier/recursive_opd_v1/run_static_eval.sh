#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/recursive_opd_v1_run
OUT=$RUN/static/evaluation
MODEL=$RUN/static/cycle3/checkpoint
cd "$ROOT"
mkdir -p "$OUT"
exec 9>"$OUT/.launcher.lock"
flock -n 9 || { echo 'static evaluation already running' >&2; exit 2; }
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v1.eval_static manifest
"$ENV/bin/python" - <<'PY'
import socket,subprocess
for port in (8360,8361):
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',port))==0: raise SystemExit(f'port {port} busy')
used=[int(x.strip()) for x in subprocess.check_output(
    ['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]): raise SystemExit(f'GPUs busy: {used[:2]}')
print('static evaluation GPU preflight PASS',flush=True)
PY
attempt=$(date -u +%Y%m%dT%H%M%SZ)_$$
SERVER0= SERVER1= WORKER0= WORKER1=
cleanup() {
  for pid in "$WORKER0" "$WORKER1" "$SERVER0" "$SERVER1"; do
    [[ -z $pid ]] || kill "$pid" 2>/dev/null || true
  done
  for pid in "$WORKER0" "$WORKER1" "$SERVER0" "$SERVER1"; do
    [[ -z $pid ]] || wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM
printf 'STARTING\n' > "$OUT/status"
for gpu in 0 1; do
  port=$((8360 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name recursive-opd-static-pi3 \
    --host 127.0.0.1 --port "$port" --context-length 32768 \
    --mem-fraction-static 0.72 --attention-backend triton \
    --sampling-backend pytorch --disable-cuda-graph \
    --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "$OUT/server_gpu${gpu}_${attempt}.log" 2>&1 &
  if [[ $gpu -eq 0 ]]; then SERVER0=$!; else SERVER1=$!; fi
done
"$ENV/bin/python" - "$SERVER0" "$SERVER1" <<'PY'
import os,sys,time,urllib.request
for port,pid in zip((8360,8361),map(int,sys.argv[1:])):
    for _ in range(420):
        try: os.kill(pid,0)
        except OSError: raise SystemExit(f'server {port} exited')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                if r.status==200: print(f'server {port} ready',flush=True); break
        except Exception: time.sleep(1)
    else: raise SystemExit(f'server {port} readiness timeout')
PY
printf 'COLLECTING\n' > "$OUT/status"
for gpu in 0 1; do
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v1.eval_static worker \
    --shard "$gpu" --endpoint "http://127.0.0.1:$((8360 + gpu))/v1" \
    > "$OUT/worker_gpu${gpu}_${attempt}.log" 2>&1 &
  if [[ $gpu -eq 0 ]]; then WORKER0=$!; else WORKER1=$!; fi
done
set +e
wait "$WORKER0"; RC0=$?
wait "$WORKER1"; RC1=$?
set -e
WORKER0= WORKER1=
if [[ $RC0 -ne 0 || $RC1 -ne 0 ]]; then
  printf 'FAILED_WORKERS:%s:%s\n' "$RC0" "$RC1" > "$OUT/status"
  exit 3
fi
cleanup
SERVER0= SERVER1=
printf 'SUMMARIZING\n' > "$OUT/status"
"$ENV/bin/python" -m repro_1p7b.graph_frontier.recursive_opd_v1.eval_static summarize \
  > "$OUT/summarize_${attempt}.log" 2>&1 || {
    printf 'FAILED_SUMMARIZE\n' > "$OUT/status"; exit 4;
  }
printf 'AUDITED\n' > "$OUT/status"
