#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/cycle3_official_rl/run_dynamic_v1_t0
MODEL=$ROOT/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
cd "$ROOT"
mkdir -p "$RUN"
exec 9>"$RUN/.launcher.lock"
flock -n 9 || { echo "Cycle-3 launcher already running"; exit 2; }

export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT
export ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$ENV/bin/python" -m repro_1p7b.graph_frontier.cycle3_official_rl.plan check
"$ENV/bin/python" - <<'PY'
import socket, subprocess
for port in (8250, 8251):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"Cycle-3 port {port} is occupied")
used = [int(x.strip()) for x in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
    text=True).splitlines()]
if len(used) < 2 or any(x >= 1500 for x in used[:2]):
    raise SystemExit(f"GPUs busy: {used[:2]}")
print(f"isolated ports and GPUs free: {used[:2]}", flush=True)
PY

TAG=full
LIMIT=()
if [[ -n "${CYCLE3_SMOKE_LIMIT:-}" ]]; then
  [[ "$CYCLE3_SMOKE_LIMIT" =~ ^[1-9][0-9]*$ ]] ||
    { echo "invalid CYCLE3_SMOKE_LIMIT" >&2; exit 2; }
  TAG=smoke
  LIMIT=(--limit "$CYCLE3_SMOKE_LIMIT")
fi
SERVER0= SERVER1= WORKER0= WORKER1= MONITOR=
cleanup() {
  for pid in "$MONITOR" "$WORKER0" "$WORKER1" "$SERVER0" "$SERVER1"; do
    [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
  done
  for pid in "$MONITOR" "$WORKER0" "$WORKER1" "$SERVER0" "$SERVER1"; do
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM
printf '%s\n' "STARTING_SERVERS_$TAG" > "$RUN/status"
(
  while true; do
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw \
      --format=csv,noheader || true
    sleep 30
  done
) > "$RUN/gpu_utilization_${TAG}.log" 2>&1 &
MONITOR=$!
for gpu in 0 1; do
  port=$((8250 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name dynamic-v1 \
    --host 127.0.0.1 --port "$port" --context-length 32768 \
    --mem-fraction-static 0.72 --attention-backend triton \
    --sampling-backend pytorch --disable-cuda-graph \
    --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "$RUN/server_gpu${gpu}_${TAG}.log" 2>&1 &
  if [[ "$gpu" -eq 0 ]]; then SERVER0=$!; else SERVER1=$!; fi
done
"$ENV/bin/python" - "$SERVER0" "$SERVER1" <<'PY'
import os, sys, time, urllib.request
for port, pid in zip((8250, 8251), map(int, sys.argv[1:])):
    for _ in range(420):
        try:
            os.kill(pid, 0)
        except OSError:
            raise SystemExit(f"server {port} exited before readiness")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    print(f"server {port} ready", flush=True)
                    break
        except Exception:
            time.sleep(1)
    else:
        raise SystemExit(f"server {port} readiness timeout")
PY
printf '%s\n' "COLLECTING_$TAG" > "$RUN/status"
for gpu in 0 1; do
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.cycle3_official_rl.runtime \
    --plan "$RUN/rollout_manifest.json" \
    --endpoint "http://127.0.0.1:$((8250 + gpu))/v1" \
    --shard-index "$gpu" --shard-count 2 "${LIMIT[@]}" \
    > "$RUN/worker_gpu${gpu}_${TAG}.log" 2>&1 &
  if [[ "$gpu" -eq 0 ]]; then WORKER0=$!; else WORKER1=$!; fi
done
set +e
wait "$WORKER0"; RC0=$?
wait "$WORKER1"; RC1=$?
set -e
WORKER0= WORKER1=
if [[ "$RC0" -ne 0 || "$RC1" -ne 0 ]]; then
  printf 'FAILED_WORKERS:%s:%s\n' "$RC0" "$RC1" > "$RUN/status"
  exit 3
fi
printf '%s\n' "ROLLOUT_COMPLETE_$TAG" > "$RUN/status"
if [[ "$TAG" == full ]]; then
  printf '%s\n' AUDITING > "$RUN/status"
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.cycle3_official_rl.audit \
    --plan "$RUN/rollout_manifest.json" \
    > "$RUN/audit.log" 2>&1 || {
      printf '%s\n' FAILED_AUDIT > "$RUN/status"
      exit 4
    }
  printf '%s\n' AUDITED > "$RUN/status"
fi
