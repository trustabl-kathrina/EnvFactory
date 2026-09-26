#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
RUN_TAG=${RUN_TAG:-dynamic_v1_train_v2}
ROLLOUT_TAG=${ROLLOUT_TAG:-${RUN_TAG}}
ALLOW_EXISTING_ROLLOUTS=${ALLOW_EXISTING_ROLLOUTS:-0}
LOG_ROOT=${ROOT}/repro_1p7b/logs/preference/${RUN_TAG}
ROLLOUTS=${ROOT}/repro_1p7b/graph_frontier/preference_runtime/${ROLLOUT_TAG}
ROLLOUTS_PER_TASK=${ROLLOUTS_PER_TASK:-4}
ROLLOUT_START_INDEX=${ROLLOUT_START_INDEX:-0}
MAX_TOKENS=${MAX_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0.7}
TASK_IDS=${TASK_IDS:-}

if [[ -e "${LOG_ROOT}/status" ]]; then
  echo "refusing to overwrite preference collection logs" >&2
  exit 2
fi
if [[ -d "${ROLLOUTS}" && "${ALLOW_EXISTING_ROLLOUTS}" != 1 ]]; then
  echo "refusing to reuse preference rollouts without ALLOW_EXISTING_ROLLOUTS=1" >&2
  exit 2
fi
if [[ ! -d "${ROLLOUTS}" && "${ALLOW_EXISTING_ROLLOUTS}" == 1 ]]; then
  echo "requested resume but rollout directory does not exist" >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}" "${ROLLOUTS}"
printf '%s\n' STARTING_SERVERS > "${LOG_ROOT}/status"

export CUDA_HOME=${ENV}
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=${ROOT}
export ENVFACTORY_REWARD_MODE=official
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SERVER_PIDS=()
COLLECTOR_PIDS=()
cleanup() {
  for pid in "${COLLECTOR_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${COLLECTOR_PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT

for shard in 0 1; do
  port=$((8020 + shard))
  CUDA_VISIBLE_DEVICES=${shard} "${ENV}/bin/python" -m sglang.launch_server \
    --model-path "${MODEL}" \
    --served-model-name dynamic-v1 \
    --host 127.0.0.1 --port "${port}" \
    --context-length 8192 \
    --mem-fraction-static 0.72 \
    --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen25 \
    --api-key placeholder \
    > "${LOG_ROOT}/server.shard${shard}.log" 2>&1 &
  SERVER_PIDS+=("$!")
done

"${ENV}/bin/python" - "${SERVER_PIDS[0]}" "${SERVER_PIDS[1]}" <<'PY'
import os
import sys
import time
import urllib.request

pids = [int(value) for value in sys.argv[1:]]
for port, pid in zip((8020, 8021), pids):
    for _ in range(420):
        try:
            os.kill(pid, 0)
        except OSError as exc:
            raise SystemExit(f"server {port} exited before readiness: {exc}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        raise SystemExit(f"server {port} readiness timeout")
PY

printf '%s\n' COLLECTING > "${LOG_ROOT}/status"
cd "${ROOT}"
TASK_ARGS=()
if [[ -n "${TASK_IDS}" ]]; then
  IFS=',' read -r -a requested_task_ids <<< "${TASK_IDS}"
  for task_id in "${requested_task_ids[@]}"; do
    TASK_ARGS+=(--task-id "${task_id}")
  done
fi
for shard in 0 1; do
  port=$((8020 + shard))
  (
    export ENVFACTORY_RL_ROLLOUT_DIR=${LOG_ROOT}/env_traces/shard${shard}
    "${ENV}/bin/python" repro_1p7b/graph_frontier/collect_preference_rollouts.py \
      --endpoint "http://127.0.0.1:${port}/v1" \
      --model dynamic-v1 \
      --output-dir "${ROLLOUTS}" \
      --internal-only --rollouts-per-task "${ROLLOUTS_PER_TASK}" \
      --rollout-start-index "${ROLLOUT_START_INDEX}" \
      "${TASK_ARGS[@]}" \
      --shard-count 2 --shard-index "${shard}" \
      --max-steps 8 --max-tokens "${MAX_TOKENS}" --temperature "${TEMPERATURE}" \
      > "${LOG_ROOT}/collector.shard${shard}.log" 2>&1
    printf '%s\n' "$?" > "${LOG_ROOT}/exit_code.shard${shard}"
  ) &
  COLLECTOR_PIDS+=("$!")
done

set +e
wait "${COLLECTOR_PIDS[0]}"; RC0=$?
wait "${COLLECTOR_PIDS[1]}"; RC1=$?
set -e
printf '%s %s\n' "${RC0}" "${RC1}" > "${LOG_ROOT}/exit_codes"
if [[ ${RC0} -eq 0 && ${RC1} -eq 0 ]]; then
  printf '%s\n' DONE > "${LOG_ROOT}/status"
else
  printf 'FAILED:%s:%s\n' "${RC0}" "${RC1}" > "${LOG_ROOT}/status"
  exit 2
fi

