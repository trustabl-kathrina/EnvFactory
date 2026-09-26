#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
OUT=${ROOT}/repro_1p7b/graph_frontier/balanced_preference_v2
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
cd "${ROOT}"
[[ -f "${OUT}/preparation.json" ]] || { echo 'preparation missing' >&2; exit 2; }
[[ ! -e "${OUT}/manifest.json" ]] || { echo 'data already frozen' >&2; exit 2; }
for port in 8020 8021; do
  "${ENV}/bin/python" - "${port}" <<'PY'
import socket, sys
s=socket.socket(); busy=s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0; s.close()
sys.exit(1 if busy else 0)
PY
done
export CUDA_HOME=${ENV} PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PID0= PID1=
cleanup() {
  [[ -z "${PID0}" ]] || { kill "${PID0}" 2>/dev/null || true; wait "${PID0}" 2>/dev/null || true; }
  [[ -z "${PID1}" ]] || { kill "${PID1}" 2>/dev/null || true; wait "${PID1}" 2>/dev/null || true; }
}
trap cleanup EXIT
printf '%s\n' STARTING_SERVERS > "${OUT}/sampling_status"
for gpu in 0 1; do
  port=$((8020 + gpu))
  CUDA_VISIBLE_DEVICES="${gpu}" "${ENV}/bin/python" -m sglang.launch_server \
    --model-path "${MODEL}" --served-model-name dynamic-v1 \
    --host 127.0.0.1 --port "${port}" --context-length 8192 \
    --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "${OUT}/server_gpu${gpu}.log" 2>&1 &
  if [[ "${gpu}" -eq 0 ]]; then PID0=$!; else PID1=$!; fi
done
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
sample_split() {
  local split=$1 k=$2 only_missing=$3 p0 p1 rc0 rc1
  for gpu in 0 1; do
    local opts=()
    [[ "${only_missing}" == no ]] || opts+=(--only-missing)
    "${ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v2_pipeline sample-stop \
      --split "${split}" --endpoint "http://127.0.0.1:$((8020 + gpu))/v1" --model dynamic-v1 \
      --shard-count 2 --shard-index "${gpu}" --target-k "${k}" "${opts[@]}" \
      > "${OUT}/${split}_k${k}_gpu${gpu}.log" 2>&1 &
    if [[ "${gpu}" -eq 0 ]]; then p0=$!; else p1=$!; fi
  done
  set +e
  wait "${p0}"; rc0=$?
  wait "${p1}"; rc1=$?
  set -e
  [[ "${rc0}" -eq 0 && "${rc1}" -eq 0 ]] || { printf 'FAILED_SAMPLING_%s:%s:%s\n' "${split}" "${rc0}" "${rc1}" > "${OUT}/sampling_status"; exit 3; }
}
printf '%s\n' SAMPLING_K4 > "${OUT}/sampling_status"
sample_split train 4 no
sample_split preference_val 4 no
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v2_pipeline count-stop --split train > "${OUT}/train_stop_count_k4.json"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v2_pipeline count-stop --split preference_val > "${OUT}/val_stop_count_k4.json"
if ! "${ENV}/bin/python" - "${OUT}" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); train=json.loads((p/'train_stop_count_k4.json').read_text()); val=json.loads((p/'val_stop_count_k4.json').read_text())
sys.exit(0 if train['states_with_observed_extra_tool']>=16 and val['states_with_observed_extra_tool']>=1 else 1)
PY
then
  printf '%s\n' SAMPLING_K8_MISSING > "${OUT}/sampling_status"
  sample_split train 8 yes
  sample_split preference_val 8 yes
fi
printf '%s\n' SAMPLING_DONE > "${OUT}/sampling_status"
