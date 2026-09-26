#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
SERVER_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
QUERY_ENV=/home/u2024311031/.conda/envs/envfactory_repro_1p7b
MODEL=/home/u2024311031/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
GRAPH=${ROOT}/repro_1p7b/data/graph_frontier_rl_v1_generated/tool_graph.pkl
PLAN=${2:-${ROOT}/repro_1p7b/graph_frontier/preference_generation_v3/manifests/smoke_plan.jsonl}
OUTPUT=${1:-${ROOT}/repro_1p7b/graph_frontier/preference_generation_v3/smoke_30_seed20260922}
MCP_CONFIG=${ROOT}/configs/mcp_server.json
MAX_FAILURES=${RICH_V3_MAX_FAILURES:-100000}
ATTEMPTS=${RICH_V3_ATTEMPTS:-3}
CONCURRENCY=${RICH_V3_CONCURRENCY:-4}

if [[ -e "${OUTPUT}/status" || -e "${OUTPUT}/generation_run.json" || -d "${OUTPUT}/raw" ]]; then
  echo "refusing to overwrite Graph-Frontier-rich smoke" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"
exec 3> "${OUTPUT}/controller.log"
export BASH_XTRACEFD=3
set -x
printf '%s\n' STARTING_SERVERS > "${OUTPUT}/status"

export CUDA_HOME=${SERVER_ENV}
export PATH=${SERVER_ENV}/bin:${QUERY_ENV}/bin:${PATH}
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${MCP_CONFIG}
export LD_LIBRARY_PATH=${SERVER_ENV}/lib:${SERVER_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${SERVER_ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SERVER_PIDS=()
cleanup() {
  for pid in "${SERVER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    printf 'SCRIPT_FAILED:%s:line=%s\n' "${rc}" "${BASH_LINENO[0]:-unknown}" > "${OUTPUT}/status"
  fi
  cleanup
  exit "${rc}"
}
trap on_exit EXIT

COMMON_ARGS=(
  --model-path "${MODEL}"
  --host 127.0.0.1
  --served-model-name Qwen2.5-14B-Instruct
  --context-length 20000
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
    env -u LD_LIBRARY_PATH /usr/bin/curl -fsS \
      "http://127.0.0.1:${port}/health" >/dev/null && return 0
    kill -0 "${pid}" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

# Reuse the previously validated two-replica 14B setup: one replica per A100.
CUDA_VISIBLE_DEVICES=0 "${SERVER_ENV}/bin/python" -m sglang.launch_server \
  "${COMMON_ARGS[@]}" --port 8010 --random-seed 20260922 \
  > "${OUTPUT}/server_gpu0.log" 2>&1 &
SERVER_PIDS+=("$!")
wait_ready 8010 "${SERVER_PIDS[0]}"

CUDA_VISIBLE_DEVICES=1 "${SERVER_ENV}/bin/python" -m sglang.launch_server \
  "${COMMON_ARGS[@]}" --port 8011 --random-seed 20260923 \
  > "${OUTPUT}/server_gpu1.log" 2>&1 &
SERVER_PIDS+=("$!")
wait_ready 8011 "${SERVER_PIDS[1]}"

export SGLANG_BASE_URL=http://127.0.0.1:8010/v1
export SGLANG_API_KEY=placeholder
export SGLANG_MODEL=Qwen2.5-14B-Instruct
export SGLANG_BASE_URL1=http://127.0.0.1:8011/v1
export SGLANG_API_KEY1=placeholder
export SGLANG_MODEL1=Qwen2.5-14B-Instruct
export CHAT_URL=http://127.0.0.1:8010/v1
export CHAT_API_KEY=placeholder
export CHAT_MODEL=Qwen2.5-14B-Instruct

printf '%s\n' GENERATING > "${OUTPUT}/status"
cd "${ROOT}"
set +e
"${QUERY_ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_generation_v3 \
  generate --graph "${GRAPH}" --plan "${PLAN}" --run-dir "${OUTPUT}" \
  --model-name sglang,sglang1 --pass-k 2 --concurrency "${CONCURRENCY}" \
  --attempts "${ATTEMPTS}" --max-failures "${MAX_FAILURES}" \
  > "${OUTPUT}/generation.log" 2>&1
RC=$?
set -e
printf '%s\n' "${RC}" > "${OUTPUT}/exit_code"
if [[ ${RC} -eq 0 ]]; then
  printf '%s\n' GENERATED > "${OUTPUT}/status"
else
  printf 'GENERATION_FAILED:%s\n' "${RC}" > "${OUTPUT}/status"
fi
exit "${RC}"
