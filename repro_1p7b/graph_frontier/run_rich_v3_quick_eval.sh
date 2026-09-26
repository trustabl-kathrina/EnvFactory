#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
BASE=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
TRAINED=${ROOT}/repro_1p7b/checkpoints/rich_v3_quick_graph_dpo_seed20260923/final_model
PILOT=${ROOT}/repro_1p7b/graph_frontier/preference_generation_v3/quick_effect_pilot_50_seed20260923
cd "${ROOT}"
[[ "$(cat "${PILOT}/dpo_status")" == TRAINED && -d "${TRAINED}" ]] || { echo 'DPO checkpoint not ready' >&2; exit 2; }
[[ ! -e "${PILOT}/eval_status" ]] || { echo 'refusing duplicate quick eval launcher' >&2; exit 2; }
export CUDA_HOME=${ENV}
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${ROOT}/configs/mcp_server.json
export ENVFACTORY_ROOT=${ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SERVER_PID=
cleanup() {
  [[ -z "${SERVER_PID}" ]] || { kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
}
trap cleanup EXIT
for label in base graph_dpo; do
  if [[ "${label}" == base ]]; then model_path="${BASE}"; else model_path="${TRAINED}"; fi
  printf 'EVAL_%s\n' "${label}" > "${PILOT}/eval_status"
  CUDA_VISIBLE_DEVICES=0 "${ENV}/bin/python" -m sglang.launch_server \
    --model-path "${model_path}" --served-model-name "${label}" \
    --host 127.0.0.1 --port 8020 --context-length 24576 \
    --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "${PILOT}/eval_${label}_server.log" 2>&1 &
  SERVER_PID=$!
  "${ENV}/bin/python" - "${SERVER_PID}" <<'PY'
import os, sys, time, urllib.request
pid = int(sys.argv[1])
for _ in range(420):
    try: os.kill(pid, 0)
    except OSError: raise SystemExit('eval server exited before readiness')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8020/health', timeout=2) as response:
            if response.status == 200: break
    except Exception: time.sleep(1)
else: raise SystemExit('eval server readiness timeout')
PY
  "${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_quick_eval collect \
    --root "${PILOT}" --label "${label}" \
    --endpoint http://127.0.0.1:8020/v1 --model "${label}" \
    --seed 20260923 --max-steps 8 --max-tokens 384 \
    > "${PILOT}/eval_${label}.log" 2>&1
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=
  sleep 5
done
printf '%s\n' COMPARING > "${PILOT}/eval_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_quick_eval compare \
  --root "${PILOT}" > "${PILOT}/quick_effect_result.log" 2>&1
printf '%s\n' DONE > "${PILOT}/eval_status"
