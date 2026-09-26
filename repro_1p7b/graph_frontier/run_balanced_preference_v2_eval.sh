#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
OUT=${ROOT}/repro_1p7b/graph_frontier/balanced_preference_v2
MODEL=${ROOT}/repro_1p7b/checkpoints/balanced_graph_dpo_v2_pilot/final_model
cd "${ROOT}"
[[ ! -e "${OUT}/evaluation/eval_status" ]] || { echo 'evaluation controller already started' >&2; exit 2; }
mkdir -p "${OUT}/evaluation"
printf '%s\n' WAITING_FOR_PILOT > "${OUT}/evaluation/eval_status"
for _ in $(seq 1 480); do
  stage=$(cat "${OUT}/training_status" 2>/dev/null || true)
  [[ "${stage}" != FAILED:* ]] || { echo "training failed: ${stage}" >&2; exit 3; }
  [[ "${stage}" != PILOT_TRAINED ]] || break
  sleep 15
done
[[ "$(cat "${OUT}/training_status")" == PILOT_TRAINED && -d "${MODEL}" ]] || { echo 'pilot not ready after wait' >&2; exit 3; }
for port in 8020 8021; do
  "${ENV}/bin/python" - "${port}" <<'PY'
import socket, sys
s=socket.socket(); busy=s.connect_ex(('127.0.0.1',int(sys.argv[1])))==0; s.close()
sys.exit(1 if busy else 0)
PY
done
export CUDA_HOME=${ENV} PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${ROOT}/configs/mcp_server.json ENVFACTORY_ROOT=${ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PID0= PID1= JOB0= JOB1=
cleanup() {
  [[ -z "${JOB0}" ]] || { kill "${JOB0}" 2>/dev/null || true; wait "${JOB0}" 2>/dev/null || true; }
  [[ -z "${JOB1}" ]] || { kill "${JOB1}" 2>/dev/null || true; wait "${JOB1}" 2>/dev/null || true; }
  [[ -z "${PID0}" ]] || { kill "${PID0}" 2>/dev/null || true; wait "${PID0}" 2>/dev/null || true; }
  [[ -z "${PID1}" ]] || { kill "${PID1}" 2>/dev/null || true; wait "${PID1}" 2>/dev/null || true; }
}
trap cleanup EXIT
trap 'rc=$?; printf "FAILED:%s:line%s\n" "${rc}" "${LINENO}" > "${OUT}/evaluation/eval_status"' ERR
printf '%s\n' STARTING_SERVERS > "${OUT}/evaluation/eval_status"
for gpu in 0 1; do
  port=$((8020 + gpu))
  CUDA_VISIBLE_DEVICES="${gpu}" "${ENV}/bin/python" -m sglang.launch_server \
    --model-path "${MODEL}" --served-model-name balanced-v2 \
    --host 127.0.0.1 --port "${port}" --context-length 24576 \
    --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "${OUT}/evaluation/server_gpu${gpu}.log" 2>&1 &
  if [[ "${gpu}" -eq 0 ]]; then PID0=$!; else PID1=$!; fi
done
"${ENV}/bin/python" - "${PID0}" "${PID1}" <<'PY'
import os,sys,time,urllib.request
for port,pid in zip((8020,8021),map(int,sys.argv[1:])):
    for _ in range(420):
        try: os.kill(pid,0)
        except OSError: raise SystemExit(f'eval server {port} exited')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                if r.status==200: break
        except Exception: time.sleep(1)
    else: raise SystemExit(f'eval server {port} readiness timeout')
PY
printf '%s\n' EVALUATING > "${OUT}/evaluation/eval_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.eval_balanced_preference_v2 isolated \
  --endpoint http://127.0.0.1:8020/v1 --model balanced-v2 \
  > "${OUT}/evaluation/isolated.log" 2>&1 & JOB0=$!
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.eval_balanced_preference_v2 full \
  --endpoint http://127.0.0.1:8021/v1 --model balanced-v2 \
  > "${OUT}/evaluation/full.log" 2>&1 & JOB1=$!
set +e
wait "${JOB0}"; rc0=$?
wait "${JOB1}"; rc1=$?
set -e
JOB0= JOB1=
[[ "${rc0}" -eq 0 && "${rc1}" -eq 0 ]] || { printf 'FAILED_EVAL:%s:%s\n' "${rc0}" "${rc1}" > "${OUT}/evaluation/eval_status"; exit 4; }
printf '%s\n' ANALYZING > "${OUT}/evaluation/eval_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.eval_balanced_preference_v2 report \
  > "${OUT}/evaluation/report.log" 2>&1
printf '%s\n' COMPLETE > "${OUT}/evaluation/eval_status"
