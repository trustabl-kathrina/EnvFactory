#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MODEL=$ROOT/repro_1p7b/checkpoints/first_divergence_onpolicy_v1_cycle2_smoke
LOG=$ROOT/repro_1p7b/logs/first_divergence_onpolicy_v1
PLAN=$LOG/cycle3_plan.json
ROLL=$LOG/cycle3_rollouts
DATASET=$LOG/cycle3_dataset
STATUS=$LOG/cycle3_status
cd "$ROOT"
mkdir -p "$LOG"
if [[ -e "$DATASET" ]]; then echo "cycle3 dataset exists" >&2; exit 2; fi
if [[ -e "$STATUS" && "$(<"$STATUS")" != FAILED_* ]]; then echo "cycle3 status exists" >&2; exit 2; fi
export CUDA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=$ROOT
export ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$ENV/bin/python" - "$PLAN" <<'PY'
import sys
from pathlib import Path
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import checked_plan
plan, _, _, _ = checked_plan(Path(sys.argv[1]))
assert plan["task_count"] == 48 and plan["target_depth_counts"] == {"1":12,"2":20,"3":16}
print("plan and hashes verified", flush=True)
PY
"$ENV/bin/python" - <<'PY'
import socket, subprocess
for port in (8140,8141):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1",port)) == 0:
            raise SystemExit(f"port {port} in use")
out=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
used=[int(x.strip()) for x in out.splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]):
    raise SystemExit(f"GPUs busy: {used}")
print(f"ports and GPUs free: {used[:2]}",flush=True)
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
"$ENV/bin/python" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set --cycle 3 --phase COLLECT --last-completed-action plan_verified --current-job fd_v1_cycle3_collect
for gpu in 0 1; do
  port=$((8140 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu "$ENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name first-divergence-v1 \
    --host 127.0.0.1 --port "$port" --context-length 32768 \
    --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "$LOG/cycle3_server_gpu$gpu.log" 2>&1 &
  if [[ "$gpu" -eq 0 ]]; then SERVER0=$!; else SERVER1=$!; fi
done
"$ENV/bin/python" - "$SERVER0" "$SERVER1" <<'PY'
import os,sys,time,urllib.request
for port,pid in zip((8140,8141),map(int,sys.argv[1:])):
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
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.rich_first_divergence_v1 collect \
    --plan "$PLAN" --output "$ROLL" --endpoint "http://127.0.0.1:$((8140 + gpu))/v1" \
    --shard-index "$gpu" --shard-count 2 > "$LOG/cycle3_worker_gpu$gpu.log" 2>&1 &
  if [[ "$gpu" -eq 0 ]]; then WORKER0=$!; else WORKER1=$!; fi
done
set +e
wait "$WORKER0"; RC0=$?
wait "$WORKER1"; RC1=$?
set -e
WORKER0= WORKER1=
if [[ "$RC0" -ne 0 || "$RC1" -ne 0 ]]; then
  printf 'FAILED_COLLECT:%s:%s\n' "$RC0" "$RC1" > "$STATUS"
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set --status ERROR --cycle 3 --phase COLLECT --error "worker_exit:$RC0:$RC1" --current-job none
  exit 3
fi
printf '%s\n' BUILDING > "$STATUS"
if ! "$ENV/bin/python" -m repro_1p7b.graph_frontier.rich_first_divergence_v1 build --plan "$PLAN" --rollouts "$ROLL" --output "$DATASET" > "$LOG/cycle3_build.log" 2>&1; then
  printf '%s\n' FAILED_BUILD > "$STATUS"
  "$ENV/bin/python" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set --status ERROR --cycle 3 --phase DATA_AUDIT --error build_failed --current-job none
  exit 4
fi
printf '%s\n' AUDITED > "$STATUS"
"$ENV/bin/python" -m repro_1p7b.graph_frontier.first_divergence_heartbeat set --cycle 3 --phase DATA_AUDIT --last-completed-action cycle3_audited --current-job none --latest-report "$DATASET/audit.json"
