#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
OUTPUT=${ROOT}/repro_1p7b/graph_frontier/preference_v2
LOG_ROOT=${ROOT}/repro_1p7b/logs/preference/active_frontier_v2
PORT=8020

if [[ -e "${LOG_ROOT}/status" ]]; then
  echo "refusing to overwrite active-frontier v2 runtime logs" >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}" "${OUTPUT}/sampled_actions"
printf '%s\n' STARTING_SERVER > "${LOG_ROOT}/status"

export CUDA_HOME=${ENV}
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=${ROOT}
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
for _ in range(420):
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise SystemExit(f"server exited before readiness: {exc}")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            if response.status == 200:
                break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("server readiness timeout")
PY

printf '%s\n' SAMPLING > "${LOG_ROOT}/status"
cd "${ROOT}"
set +e
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.active_frontier_v2 \
  --output-dir "${OUTPUT}" sample \
  --endpoint "http://127.0.0.1:${PORT}/v1" \
  --model dynamic-v1 --samples-per-state 4 \
  --temperature 0.7 --max-tokens 512 \
  > "${LOG_ROOT}/sampler.log" 2>&1
SAMPLE_RC=$?
set -e
if [[ ${SAMPLE_RC} -ne 0 ]]; then
  printf 'FAILED_SAMPLING:%s\n' "${SAMPLE_RC}" > "${LOG_ROOT}/status"
  exit "${SAMPLE_RC}"
fi

printf '%s\n' FINALIZING > "${LOG_ROOT}/status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.active_frontier_v2 \
  --output-dir "${OUTPUT}" finalize \
  > "${LOG_ROOT}/finalize.log" 2>&1
FINALIZE_RC=$?
printf '%s\n' "${FINALIZE_RC}" > "${LOG_ROOT}/exit_code"
if [[ ${FINALIZE_RC} -eq 0 ]]; then
  printf '%s\n' DONE > "${LOG_ROOT}/status"
else
  printf 'FAILED_FINALIZE:%s\n' "${FINALIZE_RC}" > "${LOG_ROOT}/status"
fi
exit "${FINALIZE_RC}"
