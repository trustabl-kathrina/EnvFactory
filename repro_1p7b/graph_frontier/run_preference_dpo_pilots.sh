#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
DATA=${ROOT}/repro_1p7b/graph_frontier/preference/dpo_v1
LOG_ROOT=${ROOT}/repro_1p7b/logs/preference/dpo_pilot_v1
OUTPUT_ROOT=${ROOT}/repro_1p7b/checkpoints
DEEPSPEED=${ROOT}/repro_1p7b/configs/ds_z3_config.json
SMOKE_STATUS=${ROOT}/repro_1p7b/logs/preference/dpo_smoke_v1/status

[[ -f "${SMOKE_STATUS}" && "$(cat "${SMOKE_STATUS}")" == DONE ]] || {
  echo "DPO smoke gate is not DONE" >&2
  exit 2
}
[[ ! -e "${LOG_ROOT}/status" ]] || { echo "refusing to overwrite pilot logs" >&2; exit 2; }
for path in \
  "${OUTPUT_ROOT}/dynamic_v1_standard_dpo_pilot" \
  "${OUTPUT_ROOT}/dynamic_v1_graph_targeted_dpo_pilot"; do
  [[ ! -e "${path}" ]] || { echo "refusing to overwrite ${path}" >&2; exit 2; }
done
GRAPH_PAIRS=$(wc -l < "${DATA}/graph_pilot_train.jsonl")
STANDARD_PAIRS=$(wc -l < "${DATA}/standard_pilot_train.jsonl")
[[ ${GRAPH_PAIRS} -ge 16 && ${GRAPH_PAIRS} -le 256 ]] || { echo "invalid graph pilot size" >&2; exit 2; }
[[ ${STANDARD_PAIRS} -ge 16 && ${STANDARD_PAIRS} -le 256 ]] || { echo "invalid standard pilot size" >&2; exit 2; }
# Three passes over the Graph pilot at global batch 2; both arms receive the
# exact same optimizer update budget even when their pair counts differ.
PILOT_STEPS=${PILOT_STEPS:-$(( (GRAPH_PAIRS + 1) / 2 * 3 ))}
mkdir -p "${LOG_ROOT}"

export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
cd "${ROOT}"

run_arm() {
  kind=$1; output=$2; run_name=$3
  CUDA_VISIBLE_DEVICES=0,1 "${ENV}/bin/torchrun" --standalone --nproc_per_node=2 \
    repro_1p7b/graph_frontier/train_preference_dpo.py \
    --model "${MODEL}" \
    --train-file "${DATA}/${kind}_pilot_train.jsonl" \
    --eval-file "${DATA}/${kind}_val.jsonl" \
    --output-dir "${output}" \
    --deepspeed "${DEEPSPEED}" \
    --max-steps "${PILOT_STEPS}" --gradient-accumulation-steps 1 \
    --learning-rate 1e-6 --beta 0.1 --max-length 8192 \
    --run-name "${run_name}" \
    > "${LOG_ROOT}/${kind}.log" 2>&1
}

printf '%s\n' STANDARD_RUNNING > "${LOG_ROOT}/status"
run_arm standard "${OUTPUT_ROOT}/dynamic_v1_standard_dpo_pilot" dynamic_v1_standard_dpo_pilot
printf '%s\n' GRAPH_RUNNING > "${LOG_ROOT}/status"
run_arm graph "${OUTPUT_ROOT}/dynamic_v1_graph_targeted_dpo_pilot" dynamic_v1_graph_targeted_dpo_pilot
printf '%s\n' DONE > "${LOG_ROOT}/status"

