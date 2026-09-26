#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
DATA=${ROOT}/repro_1p7b/graph_frontier/preference/dpo_v1
LOG_ROOT=${ROOT}/repro_1p7b/logs/preference/dpo_smoke_v1
OUTPUT_ROOT=${ROOT}/repro_1p7b/checkpoints/preference_smoke_v1
DEEPSPEED=${ROOT}/repro_1p7b/configs/ds_z3_config.json

if [[ -e "${LOG_ROOT}/status" || -d "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite DPO smoke artifacts" >&2
  exit 2
fi
for kind in standard graph; do
  count=$(wc -l < "${DATA}/${kind}_smoke_train.jsonl")
  if [[ "${count}" -ne 16 ]]; then
    echo "${kind} smoke requires exactly 16 pairs, got ${count}" >&2
    exit 2
  fi
done
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

printf '%s\n' STANDARD_RUNNING > "${LOG_ROOT}/status"
cd "${ROOT}"
CUDA_VISIBLE_DEVICES=0,1 "${ENV}/bin/torchrun" --standalone --nproc_per_node=2 \
  repro_1p7b/graph_frontier/train_preference_dpo.py \
  --model "${MODEL}" \
  --train-file "${DATA}/standard_smoke_train.jsonl" \
  --eval-file "${DATA}/standard_val.jsonl" \
  --output-dir "${OUTPUT_ROOT}/standard" \
  --deepspeed "${DEEPSPEED}" \
  --max-steps 16 --gradient-accumulation-steps 1 \
  --learning-rate 1e-6 --beta 0.1 --max-length 8192 \
  --run-name dynamic_v1_standard_dpo_smoke \
  > "${LOG_ROOT}/standard.log" 2>&1

printf '%s\n' GRAPH_RUNNING > "${LOG_ROOT}/status"
CUDA_VISIBLE_DEVICES=0,1 "${ENV}/bin/torchrun" --standalone --nproc_per_node=2 \
  repro_1p7b/graph_frontier/train_preference_dpo.py \
  --model "${MODEL}" \
  --train-file "${DATA}/graph_smoke_train.jsonl" \
  --eval-file "${DATA}/graph_val.jsonl" \
  --output-dir "${OUTPUT_ROOT}/graph" \
  --deepspeed "${DEEPSPEED}" \
  --max-steps 16 --gradient-accumulation-steps 1 \
  --learning-rate 1e-6 --beta 0.1 --max-length 8192 \
  --run-name dynamic_v1_graph_targeted_dpo_smoke \
  > "${LOG_ROOT}/graph.log" 2>&1

printf '%s\n' DONE > "${LOG_ROOT}/status"

