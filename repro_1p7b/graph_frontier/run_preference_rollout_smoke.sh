#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
LOG_ROOT=${ROOT}/repro_1p7b/logs/preference/smoke_v1
ROLLOUTS=${ROOT}/repro_1p7b/graph_frontier/preference_runtime/smoke_v1
PORT=8020

if [[ -e "${LOG_ROOT}/status" || -d "${ROLLOUTS}" ]]; then
  echo "refusing to overwrite preference smoke artifacts" >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}" "${ROLLOUTS}"
printf '%s\n' STARTING_SERVER > "${LOG_ROOT}/status"

export CUDA_HOME=${ENV}
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=${ROOT}
export ENVFACTORY_REWARD_MODE=official
export ENVFACTORY_RL_ROLLOUT_DIR=${LOG_ROOT}/env_traces
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SERVER_PID=
cleanup() {
  [[ -z "${SERVER_PID}" ]] || kill "${SERVER_PID}" 2>/dev/null || true
  [[ -z "${SERVER_PID}" ]] || wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 "${ENV}/bin/python" -m sglang.launch_server \
  --model-path "${MODEL}" \
  --served-model-name dynamic-v1 \
  --host 127.0.0.1 --port "${PORT}" \
  --context-length 8192 \
  --mem-fraction-static 0.72 \
  --attention-backend triton --sampling-backend pytorch \
  --disable-cuda-graph \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen25 \
  --api-key placeholder \
  > "${LOG_ROOT}/server.log" 2>&1 &
SERVER_PID=$!
"${ENV}/bin/python" - "${PORT}" "${SERVER_PID}" <<'PY'
import os
import sys
import time
import urllib.request

port, pid = int(sys.argv[1]), int(sys.argv[2])
for _ in range(300):
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise SystemExit(f"server process exited before readiness: {exc}")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            if response.status == 200:
                break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("server readiness timeout")
PY

printf '%s\n' COLLECTING > "${LOG_ROOT}/status"
cd "${ROOT}"
set +e
"${ENV}/bin/python" repro_1p7b/graph_frontier/collect_preference_rollouts.py \
  --endpoint "http://127.0.0.1:${PORT}/v1" \
  --model dynamic-v1 \
  --output-dir "${ROLLOUTS}" \
  --internal-only --limit 8 --rollouts-per-task 2 \
  --max-steps 8 --max-tokens 384 --temperature 0.7 \
  > "${LOG_ROOT}/collector.log" 2>&1
RC=$?
set -e
printf '%s\n' "${RC}" > "${LOG_ROOT}/exit_code"
if [[ ${RC} -eq 0 ]]; then
  printf '%s\n' DONE > "${LOG_ROOT}/status"
else
  printf 'FAILED:%s\n' "${RC}" > "${LOG_ROOT}/status"
fi
exit "${RC}"

