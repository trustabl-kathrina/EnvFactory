#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
PILOT=${ROOT}/repro_1p7b/graph_frontier/preference_generation_v3/quick_effect_pilot_50_seed20260923

cd "${ROOT}"
[[ -f "${PILOT}/replay_summary.json" ]] || { echo 'CPU replay incomplete' >&2; exit 2; }
"${ENV}/bin/python" - "${PILOT}" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
r = json.loads((p / 'replay_summary.json').read_text())
if r['chosen_executable'] != r['replay_valid_states'] or r['frozen300_exact_overlap'] != 0:
    raise SystemExit('replay provenance gate failed')
if r['split_tasks_with_states']['train'] < 10 or r['split_tasks_with_states']['preference_val'] < 2:
    raise SystemExit('too few independent replay-valid tasks')
PY

for port in 8020 8021; do
  if "${ENV}/bin/python" - "${port}" <<'PY'
import socket, sys
s = socket.socket()
result = s.connect_ex(('127.0.0.1', int(sys.argv[1])))
s.close()
sys.exit(0 if result != 0 else 1)
PY
  then :; else echo "port ${port} already occupied" >&2; exit 2; fi
done

export CUDA_HOME=${ENV}
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export ENVFACTORY_ROOT=${ROOT}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PID0= PID1=
cleanup() {
  [[ -z "${PID0}" ]] || { kill "${PID0}" 2>/dev/null || true; wait "${PID0}" 2>/dev/null || true; }
  [[ -z "${PID1}" ]] || { kill "${PID1}" 2>/dev/null || true; wait "${PID1}" 2>/dev/null || true; }
}
trap cleanup EXIT

printf '%s\n' STARTING_SERVERS > "${PILOT}/sampling_status"
for gpu in 0 1; do
  port=$((8020 + gpu))
  CUDA_VISIBLE_DEVICES="${gpu}" "${ENV}/bin/python" -m sglang.launch_server \
    --model-path "${MODEL}" --served-model-name dynamic-v1 \
    --host 127.0.0.1 --port "${port}" --context-length 8192 \
    --mem-fraction-static 0.72 --attention-backend triton --sampling-backend pytorch \
    --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser qwen25 \
    --api-key placeholder > "${PILOT}/server_gpu${gpu}.log" 2>&1 &
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

printf '%s\n' SAMPLING > "${PILOT}/sampling_status"
for split in train preference_val; do
  for gpu in 0 1; do
    "${ENV}/bin/python" -m repro_1p7b.graph_frontier.active_frontier_v2 \
      --output-dir "${PILOT}/${split}" sample \
      --endpoint "http://127.0.0.1:$((8020 + gpu))/v1" --model dynamic-v1 \
      --samples-per-state 4 --temperature 0.7 --max-tokens 512 \
      --seed 20260923 --shard-count 2 --shard-index "${gpu}" \
      > "${PILOT}/${split}.sample_gpu${gpu}.log" 2>&1 &
    if [[ "${gpu}" -eq 0 ]]; then SAMPLE0=$!; else SAMPLE1=$!; fi
  done
  set +e
  wait "${SAMPLE0}"; rc0=$?
  wait "${SAMPLE1}"; rc1=$?
  set -e
  if [[ "${rc0}" -ne 0 || "${rc1}" -ne 0 ]]; then
    printf 'FAILED_SAMPLING_%s:%s:%s\n' "${split}" "${rc0}" "${rc1}" > "${PILOT}/sampling_status"
    exit 3
  fi
done
printf '%s\n' AUDITING_PAIRS > "${PILOT}/sampling_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_quick_pairs audit-samples \
  --root "${PILOT}" > "${PILOT}/quick_pair_audit.log" 2>&1
if ! grep -q '"verdict": "PAIR_READY"' "${PILOT}/quick_pair_audit.json"; then
  printf '%s\n' PAIR_GATE_FAILED > "${PILOT}/sampling_status"
  exit 4
fi
printf '%s\n' MATERIALIZING > "${PILOT}/sampling_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_quick_pairs materialize \
  --root "${PILOT}" --model "${MODEL}" --max-length 8192 \
  > "${PILOT}/quick_dpo_materialization.log" 2>&1
if ! grep -q '"verdict": "DPO_READY"' "${PILOT}/quick_dpo_materialization_audit.json"; then
  printf '%s\n' SERIALIZATION_GATE_FAILED > "${PILOT}/sampling_status"
  exit 5
fi
printf '%s\n' DPO_READY > "${PILOT}/sampling_status"
