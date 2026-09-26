#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
BASE=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
TRAINED=${ROOT}/repro_1p7b/checkpoints/rich_v3_quick_graph_dpo_seed20260923/final_model
PILOT=${ROOT}/repro_1p7b/graph_frontier/preference_generation_v3/quick_effect_pilot_50_seed20260923
cd "${ROOT}"
[[ "$(cat "${PILOT}/eval_status")" == DONE ]] || { echo 'full heldout eval not done' >&2; exit 2; }
[[ ! -e "${PILOT}/frontier_eval_status" ]] || { echo 'refusing duplicate frontier eval' >&2; exit 2; }
export CUDA_HOME=${ENV}
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PID0= PID1=
cleanup() {
  [[ -z "${PID0}" ]] || { kill "${PID0}" 2>/dev/null || true; wait "${PID0}" 2>/dev/null || true; }
  [[ -z "${PID1}" ]] || { kill "${PID1}" 2>/dev/null || true; wait "${PID1}" 2>/dev/null || true; }
}
trap cleanup EXIT
printf '%s\n' STARTING_SERVERS > "${PILOT}/frontier_eval_status"
CUDA_VISIBLE_DEVICES=0 "${ENV}/bin/python" -m sglang.launch_server \
  --model-path "${BASE}" --served-model-name base --host 127.0.0.1 --port 8020 \
  --context-length 24576 --mem-fraction-static 0.72 \
  --attention-backend triton --sampling-backend pytorch --disable-cuda-graph \
  --reasoning-parser qwen3 --tool-call-parser qwen25 --api-key placeholder \
  > "${PILOT}/frontier_base_server.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 "${ENV}/bin/python" -m sglang.launch_server \
  --model-path "${TRAINED}" --served-model-name graph_dpo --host 127.0.0.1 --port 8021 \
  --context-length 24576 --mem-fraction-static 0.72 \
  --attention-backend triton --sampling-backend pytorch --disable-cuda-graph \
  --reasoning-parser qwen3 --tool-call-parser qwen25 --api-key placeholder \
  > "${PILOT}/frontier_graph_server.log" 2>&1 &
PID1=$!
"${ENV}/bin/python" - "${PID0}" "${PID1}" <<'PY'
import os, sys, time, urllib.request
for port, pid in zip((8020, 8021), map(int, sys.argv[1:])):
    for _ in range(420):
        try: os.kill(pid, 0)
        except OSError: raise SystemExit(f'server {port} exited before readiness')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                if response.status == 200: break
        except Exception: time.sleep(1)
    else: raise SystemExit(f'server {port} readiness timeout')
PY
printf '%s\n' EVALUATING > "${PILOT}/frontier_eval_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_quick_frontier_eval collect \
  --root "${PILOT}" --label base --endpoint http://127.0.0.1:8020/v1 \
  --model base --seed 20260923 --max-tokens 512 \
  > "${PILOT}/frontier_base.log" 2>&1 &
EVAL0=$!
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_quick_frontier_eval collect \
  --root "${PILOT}" --label graph_dpo --endpoint http://127.0.0.1:8021/v1 \
  --model graph_dpo --seed 20260923 --max-tokens 512 \
  > "${PILOT}/frontier_graph.log" 2>&1 &
EVAL1=$!
set +e
wait "${EVAL0}"; rc0=$?
wait "${EVAL1}"; rc1=$?
set -e
if [[ "${rc0}" -ne 0 || "${rc1}" -ne 0 ]]; then
  printf 'FAILED:%s:%s\n' "${rc0}" "${rc1}" > "${PILOT}/frontier_eval_status"
  exit 3
fi
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_quick_frontier_eval compare \
  --root "${PILOT}" > "${PILOT}/frontier_compare.log" 2>&1
printf '%s\n' DONE > "${PILOT}/frontier_eval_status"
