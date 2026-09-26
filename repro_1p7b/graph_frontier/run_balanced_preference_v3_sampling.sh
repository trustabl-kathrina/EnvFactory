#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MILESTONE=${BALANCED_V3_MILESTONE:-milestone_24}
OUT=${ROOT}/repro_1p7b/graph_frontier/balanced_preference_v3/${MILESTONE}
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
GEN_STATUS=${ROOT}/repro_1p7b/graph_frontier/balanced_preference_v3_generation_status
cd "${ROOT}"
[[ -f "${OUT}/source_manifest.json" && -f "${OUT}/heldout_manifest.json" ]] || {
  echo 'audited milestone source and heldout reservation required' >&2; exit 2;
}
[[ ! -f "${OUT}/manifest.json" ]] || { echo 'data already frozen' >&2; exit 2; }
[[ "$(cat "${GEN_STATUS}")" == PAUSED_AT_* ]] || { echo 'generator must be paused' >&2; exit 2; }
for port in 8020 8021; do
  "${ENV}/bin/python" - "${port}" <<'PY'
import socket,sys
s=socket.socket(); busy=s.connect_ex(('127.0.0.1',int(sys.argv[1])))==0; s.close()
sys.exit(1 if busy else 0)
PY
done
export CUDA_HOME=${ENV} PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PID0= PID1= JOB0= JOB1=
cleanup() {
  [[ -z "${JOB0}" ]] || { kill "${JOB0}" 2>/dev/null || true; wait "${JOB0}" 2>/dev/null || true; }
  [[ -z "${JOB1}" ]] || { kill "${JOB1}" 2>/dev/null || true; wait "${JOB1}" 2>/dev/null || true; }
  [[ -z "${PID0}" ]] || { kill "${PID0}" 2>/dev/null || true; wait "${PID0}" 2>/dev/null || true; }
  [[ -z "${PID1}" ]] || { kill "${PID1}" 2>/dev/null || true; wait "${PID1}" 2>/dev/null || true; }
}
trap cleanup EXIT
trap 'rc=$?; printf "FAILED:%s:line%s\n" "${rc}" "${LINENO}" > "${OUT}/sampling_status"' ERR
printf '%s\n' STARTING_SERVERS > "${OUT}/sampling_status"
for gpu in 0 1; do
  port=$((8020 + gpu))
  CUDA_VISIBLE_DEVICES="${gpu}" "${ENV}/bin/python" -m sglang.launch_server \
    --model-path "${MODEL}" --served-model-name dynamic-v1 \
    --host 127.0.0.1 --port "${port}" --context-length 24576 \
    --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "${OUT}/server_gpu${gpu}.log" 2>&1 &
  if [[ "${gpu}" -eq 0 ]]; then PID0=$!; else PID1=$!; fi
done
"${ENV}/bin/python" - "${PID0}" "${PID1}" <<'PY'
import os,sys,time,urllib.request
for port,pid in zip((8020,8021),map(int,sys.argv[1:])):
    for _ in range(420):
        try: os.kill(pid,0)
        except OSError: raise SystemExit(f'v3 sampling server {port} exited')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
                if r.status==200: break
        except Exception: time.sleep(1)
    else: raise SystemExit(f'v3 sampling server {port} readiness timeout')
PY
sample_pass() {
  local k=$1 type=$2 rc0 rc1 opts=()
  [[ "${type}" == all ]] || opts+=(--only-missing-type "${type}")
  for gpu in 0 1; do
    "${ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v3_pairs sample \
      --endpoint "http://127.0.0.1:$((8020 + gpu))/v1" --shard-count 2 \
      --shard-index "${gpu}" --target-k "${k}" "${opts[@]}" \
      > "${OUT}/sample_k${k}_${type}_gpu${gpu}.log" 2>&1 &
    if [[ "${gpu}" -eq 0 ]]; then JOB0=$!; else JOB1=$!; fi
  done
  set +e
  wait "${JOB0}"; rc0=$?
  wait "${JOB1}"; rc1=$?
  set -e
  JOB0= JOB1=
  [[ "${rc0}" -eq 0 && "${rc1}" -eq 0 ]] || {
    echo "sampling failed for k=${k} type=${type}: ${rc0}/${rc1}" >&2; exit 3;
  }
}
printf '%s\n' SAMPLING_K4 > "${OUT}/sampling_status"
sample_pass 4 all
printf '%s\n' AUDITING_K4 > "${OUT}/sampling_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v3_pairs build > "${OUT}/audit_k4.log" 2>&1
if [[ -f "${OUT}/manifest.json" ]]; then
  printf '%s\n' DATA_READY > "${OUT}/sampling_status"
  exit 0
fi
for type in stop_required downstream_continue continue_required; do
  need=$("${ENV}/bin/python" - "${OUT}/audit.json" "${type}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); t=sys.argv[2]
total=max(280,r.get('train_pairs',0)); count=r.get('candidate',{}).get('candidate_types',{}).get(t,0)
threshold={'stop_required':.25,'downstream_continue':.20,'continue_required':.30}[t]
v=r.get('validation_type_unique_states',{}).get(t,0)
vmin={'stop_required':10,'downstream_continue':8,'continue_required':10}[t]
print('yes' if count < threshold*total or v < vmin else 'no')
PY
)
  if [[ "${need}" == yes ]]; then
    printf 'SAMPLING_K8_%s\n' "${type}" > "${OUT}/sampling_status"
    sample_pass 8 "${type}"
  fi
done
printf '%s\n' AUDITING_K8 > "${OUT}/sampling_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v3_pairs build > "${OUT}/audit_k8.log" 2>&1
if [[ -f "${OUT}/manifest.json" ]]; then
  printf '%s\n' DATA_READY > "${OUT}/sampling_status"
else
  printf '%s\n' DATA_NOT_READY > "${OUT}/sampling_status"
fi
