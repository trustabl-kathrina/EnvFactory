#!/usr/bin/env bash
set -euo pipefail

MAIN_ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
SERVER_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
SERVER_PYTHON=${SERVER_ENV}/bin/python
QUERY_ENV=/home/u2024311031/.conda/envs/envfactory_repro_1p7b
QUERY_PYTHON=${QUERY_ENV}/bin/python
MODEL=/home/u2024311031/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
GRAPH=${MAIN_ROOT}/repro_1p7b/data/graph_frontier_rl_v1_generated/tool_graph.pkl
MCP_CONFIG=${MAIN_ROOT}/configs/mcp_server.json
OUTPUT=${1:?usage: run_guided_rl_generation.sh OUTPUT_DIR [COUNT]}
COUNT=${2:-1000}
SEED=${GUIDED_RL_GENERATION_SEED:-20260917}
MAX_FAILURES=${GUIDED_RL_MAX_FAILURES:-20}
MIN_VALID_TASKS=${GUIDED_RL_MIN_VALID_TASKS:-0}

export CUDA_HOME=${SERVER_ENV}
export PATH=${SERVER_ENV}/bin:${QUERY_ENV}/bin:${PATH}
export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${MCP_CONFIG}
export LD_LIBRARY_PATH=${SERVER_ENV}/lib:${SERVER_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${SERVER_ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}

if [[ -e "${OUTPUT}/generation_run.json" || -d "${OUTPUT}/raw" || -d "${OUTPUT}/gold" ]]; then
  echo "refusing to overwrite an existing generation run: ${OUTPUT}" >&2
  exit 2
fi
for required in "${SERVER_PYTHON}" "${QUERY_PYTHON}" "${MODEL}/config.json" "${GRAPH}" "${MCP_CONFIG}"; do
  if [[ ! -e "${required}" ]]; then
    echo "missing required generation input: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT}"
STATUS=${OUTPUT}/status
echo STARTING_SERVERS > "${STATUS}"

SERVER0_PID=
SERVER1_PID=
cleanup() {
  [[ -z "${SERVER0_PID}" ]] || kill "${SERVER0_PID}" 2>/dev/null || true
  [[ -z "${SERVER1_PID}" ]] || kill "${SERVER1_PID}" 2>/dev/null || true
  [[ -z "${SERVER0_PID}" ]] || wait "${SERVER0_PID}" 2>/dev/null || true
  [[ -z "${SERVER1_PID}" ]] || wait "${SERVER1_PID}" 2>/dev/null || true
}
trap cleanup EXIT

COMMON_ARGS=(
  --model-path "${MODEL}"
  --host 127.0.0.1
  --served-model-name Qwen2.5-14B-Instruct
  --context-length 8192
  --mem-fraction-static 0.85
  --attention-backend triton
  --sampling-backend pytorch
  --disable-cuda-graph
  --api-key placeholder
)
wait_ready() {
  local port=$1
  local pid=$2
  for _ in $(seq 1 600); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "generation server on port ${port} exited before ready" >&2
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "generation server on port ${port} did not become ready" >&2
  return 1
}

# Cold-start the two replicas sequentially.  Concurrent imports and two
# simultaneous 28-GB shard reads contend heavily on this host; inference is
# still fully parallel after both replicas are ready.
CUDA_VISIBLE_DEVICES=0 "${SERVER_PYTHON}" -m sglang.launch_server \
  "${COMMON_ARGS[@]}" --port 8010 --random-seed "${SEED}" \
  > "${OUTPUT}/server_gpu0.log" 2>&1 &
SERVER0_PID=$!
wait_ready 8010 "${SERVER0_PID}"
CUDA_VISIBLE_DEVICES=1 "${SERVER_PYTHON}" -m sglang.launch_server \
  "${COMMON_ARGS[@]}" --port 8011 --random-seed "$((SEED + 1))" \
  > "${OUTPUT}/server_gpu1.log" 2>&1 &
SERVER1_PID=$!
wait_ready 8011 "${SERVER1_PID}"

export SGLANG_BASE_URL=http://127.0.0.1:8010/v1
export SGLANG_API_KEY=placeholder
export SGLANG_MODEL=Qwen2.5-14B-Instruct
export SGLANG_BASE_URL1=http://127.0.0.1:8011/v1
export SGLANG_API_KEY1=placeholder
export SGLANG_MODEL1=Qwen2.5-14B-Instruct
export CHAT_URL=http://127.0.0.1:8010/v1
export CHAT_API_KEY=placeholder
export CHAT_MODEL=Qwen2.5-14B-Instruct

echo GENERATING > "${STATUS}"
cd "${MAIN_ROOT}"
set +e
"${QUERY_PYTHON}" repro_1p7b/graph_frontier/guided_rl_generate.py generate \
  --graph "${GRAPH}" \
  --output-dir "${OUTPUT}" \
  --count "${COUNT}" \
  --seed "${SEED}" \
  --pass-k 2 \
  --concurrency 4 \
  --model-name sglang,sglang1 \
  --max-failures "${MAX_FAILURES}" \
  --min-valid-tasks "${MIN_VALID_TASKS}" \
  > "${OUTPUT}/generation.log" 2>&1
EXIT_CODE=$?
set -e
if [[ ${EXIT_CODE} -eq 0 ]]; then
  echo GENERATED > "${STATUS}"
else
  echo "GENERATION_FAILED:${EXIT_CODE}" > "${STATUS}"
fi
exit "${EXIT_CODE}"
