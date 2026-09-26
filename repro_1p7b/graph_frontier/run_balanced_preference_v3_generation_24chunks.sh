#!/usr/bin/env bash
# Resume the immutable 1219-plan Rich-v3 run to a bounded 24-chunk milestone.
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
GF=${ROOT}/repro_1p7b/graph_frontier
V3=${GF}/preference_generation_v3
SERVER_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
QUERY_ENV=/home/u2024311031/.conda/envs/envfactory_repro_1p7b
MODEL=/home/u2024311031/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
GRAPH=${ROOT}/repro_1p7b/data/graph_frontier_rl_v1_generated/tool_graph.pkl
PLAN=${V3}/manifests/formal_plan_unambiguous_v2_scaled_deep.jsonl
OUTPUT=${V3}/formal_1219_unambiguous_v2_resumable
STATUS=${GF}/balanced_preference_v3_generation_status
cd "${ROOT}"
[[ -f "${OUTPUT}/resume_manifest.json" && ! -f "${OUTPUT}/generation_run.json" ]] || { echo 'resume lock missing or run complete' >&2; exit 2; }
if [[ -e "${STATUS}" ]]; then
  stage=$(cat "${STATUS}")
  [[ "${stage}" == GENERATING_TO_24_CHUNKS || "${stage}" == STARTING_SERVERS || "${stage}" == FAILED:* ]] || {
    echo "cannot resume v3 generation from status ${stage}" >&2; exit 2;
  }
  if pgrep -f '[r]ich_generation_v3_resumable run' >/dev/null; then
    echo 'v3 generation is already active; refusing duplicate' >&2; exit 2
  fi
fi
complete=$(find "${OUTPUT}" -name COMPLETE.json | wc -l)
[[ "${complete}" -ge 16 && "${complete}" -lt 24 ]] || {
  echo "unexpected completed chunk count: ${complete}" >&2; exit 2;
}
"${QUERY_ENV}/bin/python" - "${OUTPUT}/resume_manifest.json" "${PLAN}" "${GRAPH}" <<'PY'
import json,sys
from pathlib import Path
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256
m=json.loads(Path(sys.argv[1]).read_text())
if m['plan_sha256']!=file_sha256(Path(sys.argv[2])) or m['graph_sha256']!=file_sha256(Path(sys.argv[3])) or m['chunk_size']!=32 or m['total_plans']!=1219:
    raise SystemExit('resume plan/graph/chunk identity mismatch')
PY
for port in 8010 8011; do
  if env -u LD_LIBRARY_PATH /usr/bin/curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "port ${port} already hosts a server" >&2; exit 2
  fi
done
mkdir -p "${OUTPUT}/launcher_attempts"
ATTEMPT=$(mktemp -d "${OUTPUT}/launcher_attempts/v3-24-XXXXXXXX")
export CUDA_HOME=${SERVER_ENV}
export PATH=${SERVER_ENV}/bin:${QUERY_ENV}/bin:${PATH}
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${ROOT}/configs/mcp_server.json
export LD_LIBRARY_PATH=${SERVER_ENV}/lib:${SERVER_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${SERVER_ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SERVER_PIDS=()
cleanup() {
  for pid in "${SERVER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
on_exit() {
  rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then printf 'FAILED:%s\n' "${rc}" > "${STATUS}"; fi
  cleanup
  exit "${rc}"
}
trap on_exit EXIT
wait_ready() {
  local port=$1 pid=$2
  for _ in $(seq 1 600); do
    env -u LD_LIBRARY_PATH /usr/bin/curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && return 0
    kill -0 "${pid}" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
COMMON_ARGS=(
  --model-path "${MODEL}" --host 127.0.0.1
  --served-model-name Qwen2.5-14B-Instruct --context-length 20000
  --mem-fraction-static 0.85 --attention-backend triton
  --sampling-backend pytorch --disable-cuda-graph --api-key placeholder
)
printf '%s\n' STARTING_SERVERS > "${STATUS}"
CUDA_VISIBLE_DEVICES=0 "${SERVER_ENV}/bin/python" -m sglang.launch_server \
  "${COMMON_ARGS[@]}" --port 8010 --random-seed 20260922 > "${ATTEMPT}/server_gpu0.log" 2>&1 &
SERVER_PIDS+=("$!")
wait_ready 8010 "${SERVER_PIDS[0]}"
CUDA_VISIBLE_DEVICES=1 "${SERVER_ENV}/bin/python" -m sglang.launch_server \
  "${COMMON_ARGS[@]}" --port 8011 --random-seed 20260923 > "${ATTEMPT}/server_gpu1.log" 2>&1 &
SERVER_PIDS+=("$!")
wait_ready 8011 "${SERVER_PIDS[1]}"
export SGLANG_BASE_URL=http://127.0.0.1:8010/v1 SGLANG_API_KEY=placeholder SGLANG_MODEL=Qwen2.5-14B-Instruct
export SGLANG_BASE_URL1=http://127.0.0.1:8011/v1 SGLANG_API_KEY1=placeholder SGLANG_MODEL1=Qwen2.5-14B-Instruct
export CHAT_URL=http://127.0.0.1:8010/v1 CHAT_API_KEY=placeholder CHAT_MODEL=Qwen2.5-14B-Instruct
printf '%s\n' GENERATING_TO_24_CHUNKS > "${STATUS}"
"${QUERY_ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_generation_v3_resumable run \
  --plan "${PLAN}" --graph "${GRAPH}" --output-dir "${OUTPUT}" --chunk-size 32 \
  --model-name sglang,sglang1 --pass-k 2 --concurrency 4 --attempts 4 \
  --balanced-order --pause-after-completed-chunks 24 > "${ATTEMPT}/generation.log" 2>&1
[[ "$(cat "${OUTPUT}/status")" == PAUSED_FOR_PILOT ]] || { echo 'expected bounded pause after chunk24' >&2; exit 3; }
printf '%s\n' PAUSED_AT_24_CHUNKS > "${STATUS}"
