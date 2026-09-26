#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
PILOT=${ROOT}/repro_1p7b/graph_frontier/preference_generation_v3/quick_effect_pilot_50_seed20260923
cd "${ROOT}"
[[ "$(cat "${PILOT}/sampling_status")" == PAIR_GATE_FAILED ]] || { echo 'unexpected pilot state' >&2; exit 2; }
if "${ENV}/bin/python" - <<'PY'
import socket, sys
s = socket.socket()
result = s.connect_ex(('127.0.0.1', 8020))
s.close()
sys.exit(0 if result != 0 else 1)
PY
then :; else echo 'port 8020 occupied' >&2; exit 2; fi
export CUDA_HOME=${ENV}
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=${ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SERVER_PID=
cleanup() {
  [[ -z "${SERVER_PID}" ]] || { kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
}
trap cleanup EXIT
printf '%s\n' STARTING_12K_SERVER > "${PILOT}/val_topup_status"
CUDA_VISIBLE_DEVICES=0 "${ENV}/bin/python" -m sglang.launch_server \
  --model-path "${MODEL}" --served-model-name dynamic-v1 \
  --host 127.0.0.1 --port 8020 --context-length 12288 \
  --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
  --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
  --api-key placeholder > "${PILOT}/val_topup_server.log" 2>&1 &
SERVER_PID=$!
"${ENV}/bin/python" - "${SERVER_PID}" <<'PY'
import os, sys, time, urllib.request
pid = int(sys.argv[1])
for _ in range(420):
    try: os.kill(pid, 0)
    except OSError: raise SystemExit('12k server exited before readiness')
    try:
        with urllib.request.urlopen('http://127.0.0.1:8020/health', timeout=2) as response:
            if response.status == 200: break
    except Exception: time.sleep(1)
else: raise SystemExit('12k server readiness timeout')
PY
printf '%s\n' TOPPING_UP > "${PILOT}/val_topup_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_quick_val_topup topup \
  --root "${PILOT}" --endpoint http://127.0.0.1:8020/v1 --model dynamic-v1 \
  --max-extra-per-state 12 > "${PILOT}/val_topup.log" 2>&1
printf '%s\n' REAUDITING > "${PILOT}/val_topup_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_quick_pairs audit-samples \
  --root "${PILOT}" > "${PILOT}/quick_pair_audit_after_topup.log" 2>&1
if ! grep -q '"verdict": "PAIR_READY"' "${PILOT}/quick_pair_audit.json"; then
  printf '%s\n' PAIR_GATE_FAILED > "${PILOT}/val_topup_status"
  exit 4
fi
printf '%s\n' MATERIALIZING > "${PILOT}/val_topup_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_quick_pairs materialize \
  --root "${PILOT}" --model "${MODEL}" --max-length 12288 \
  > "${PILOT}/quick_dpo_materialization.log" 2>&1
if ! grep -q '"verdict": "DPO_READY"' "${PILOT}/quick_dpo_materialization_audit.json"; then
  printf '%s\n' SERIALIZATION_GATE_FAILED > "${PILOT}/val_topup_status"
  exit 5
fi
printf '%s\n' DPO_READY > "${PILOT}/val_topup_status"
