#!/usr/bin/env bash
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
STATUS=${V3}/resumable_pipeline_status

cd "${ROOT}"
test -f "${OUTPUT}/resume_manifest.json" || { echo 'missing resume bootstrap' >&2; exit 2; }
mkdir -p "${OUTPUT}/launcher_attempts"
ATTEMPT=$(mktemp -d "${OUTPUT}/launcher_attempts/launch-XXXXXXXX")
printf 'launcher=%s\n' "${ATTEMPT}" > "${ATTEMPT}/info"

export CUDA_HOME=${SERVER_ENV}
export PATH=${SERVER_ENV}/bin:${QUERY_ENV}/bin:${PATH}
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${ROOT}/configs/mcp_server.json
export LD_LIBRARY_PATH=${SERVER_ENV}/lib:${SERVER_ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=${SERVER_ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SERVER_PIDS=()
cleanup() {
  for pid in "${SERVER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
  SERVER_PIDS=()
}
on_exit() {
  rc=$?
  trap - EXIT
  if [[ ${rc} -ne 0 ]]; then
    printf 'RUN_FAILED:%s\n' "${rc}" > "${ATTEMPT}/status"
    printf 'RUN_FAILED:%s\n' "${rc}" > "${STATUS}"
  fi
  cleanup
  exit "${rc}"
}
trap on_exit EXIT

if [[ ! -f "${OUTPUT}/generation_run.json" ]]; then
  printf '%s\n' STARTING_SERVERS > "${ATTEMPT}/status"
  printf '%s\n' STARTING_SERVERS > "${STATUS}"
  COMMON_ARGS=(
    --model-path "${MODEL}" --host 127.0.0.1
    --served-model-name Qwen2.5-14B-Instruct --context-length 20000
    --mem-fraction-static 0.85 --attention-backend triton
    --sampling-backend pytorch --disable-cuda-graph --api-key placeholder
  )
  wait_ready() {
    local port=$1 pid=$2
    for _ in $(seq 1 600); do
      env -u LD_LIBRARY_PATH /usr/bin/curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && return 0
      kill -0 "${pid}" 2>/dev/null || return 1
      sleep 1
    done
    return 1
  }
  CUDA_VISIBLE_DEVICES=0 "${SERVER_ENV}/bin/python" -m sglang.launch_server \
    "${COMMON_ARGS[@]}" --port 8010 --random-seed 20260922 \
    > "${ATTEMPT}/server_gpu0.log" 2>&1 &
  SERVER_PIDS+=("$!")
  wait_ready 8010 "${SERVER_PIDS[0]}"
  CUDA_VISIBLE_DEVICES=1 "${SERVER_ENV}/bin/python" -m sglang.launch_server \
    "${COMMON_ARGS[@]}" --port 8011 --random-seed 20260923 \
    > "${ATTEMPT}/server_gpu1.log" 2>&1 &
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
  printf '%s\n' GENERATING > "${ATTEMPT}/status"
  printf '%s\n' GENERATING > "${OUTPUT}/status"
  printf '%s\n' GENERATING > "${STATUS}"
  "${QUERY_ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_generation_v3_resumable run \
    --plan "${PLAN}" --graph "${GRAPH}" --output-dir "${OUTPUT}" --chunk-size 32 \
    --model-name sglang,sglang1 --pass-k 2 --concurrency 4 --attempts 4 \
    --balanced-order --pause-after-completed-chunks 16 \
    > "${ATTEMPT}/generation.log" 2>&1
  cleanup
  if [[ "$(cat "${OUTPUT}/status")" == "PAUSED_FOR_PILOT" ]]; then
    printf "%s\n" PAUSED_FOR_PILOT > "${ATTEMPT}/status"
    printf "%s\n" PAUSED_FOR_PILOT > "${STATUS}"
    exit 0
  fi
fi

"${QUERY_ENV}/bin/python" -c 'import json,sys; from pathlib import Path; from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256; p=Path(sys.argv[1]); r=json.loads(Path(sys.argv[2]).read_text()); assert r["requested_plans"] == 1219 and r["plan_sha256"] == file_sha256(p); assert r["raw_chain_files"] == r["sidecar_files"] == r["completed_chains"]' "${PLAN}" "${OUTPUT}/generation_run.json"
printf '%s\n' GENERATED > "${OUTPUT}/status"
printf '%s\n' AUDITING_FORMAL > "${ATTEMPT}/status"
printf '%s\n' AUDITING_FORMAL > "${STATUS}"
if [[ ! -e "${OUTPUT}/audit_status" ]]; then
  bash "${GF}/run_rich_generation_v3_audit.sh" "${OUTPUT}" "${PLAN}" formal \
    > "${ATTEMPT}/audit.log" 2>&1
fi
test "$(cat "${OUTPUT}/audit_status")" = AUDIT_DONE
"${QUERY_ENV}/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' \
  "${OUTPUT}/audit/validation_report.json" | tee "${STATUS}" "${ATTEMPT}/status"
