#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
PILOT=${ROOT}/repro_1p7b/graph_frontier/preference_generation_v3/balanced_effect_pilot_192_v1
OUTPUT=${ROOT}/repro_1p7b/checkpoints/rich_v3_balanced_graph_dpo64_seed20260925
DEEPSPEED=${ROOT}/repro_1p7b/configs/ds_z3_config.json
cd "${ROOT}"
[[ "$(cat "${PILOT}/sampling_status")" == DPO_READY ]] || { echo 'DPO gate not ready' >&2; exit 2; }
[[ ! -e "${OUTPUT}" && ! -e "${PILOT}/dpo_status" ]] || { echo 'refusing to overwrite DPO output' >&2; exit 2; }
"${ENV}/bin/python" - "${PILOT}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
pairs=json.loads((root/'quick_pair_audit.json').read_text())
material=json.loads((root/'quick_dpo_materialization_audit.json').read_text())
if pairs['verdict']!='PAIR_READY' or material['verdict']!='DPO_READY':
    raise SystemExit('DPO gates failed')
if pairs['train_tasks']<40 or pairs['preference_val_tasks']<4 or pairs['train_failure_types'].get('wrong_value',0)<1 or material['train_rows']!=64 or material['val_rows']<8:
    raise SystemExit('DPO task or pair gate failed')
if pairs['frozen300_overlap']!=0 or pairs['chosen_executable_rate']!=1.0:
    raise SystemExit('DPO contamination or replay gate failed')
PY
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
printf '%s\n' TRAINING > "${PILOT}/dpo_status"
set +e
CUDA_VISIBLE_DEVICES=0,1 "${ENV}/bin/torchrun" --standalone --nproc_per_node=2 \
  repro_1p7b/graph_frontier/train_preference_dpo.py \
  --model "${MODEL}" --train-file "${PILOT}/dpo_train64.jsonl" \
  --eval-file "${PILOT}/dpo_val.jsonl" --output-dir "${OUTPUT}" \
  --deepspeed "${DEEPSPEED}" --max-steps 64 \
  --gradient-accumulation-steps 1 --learning-rate 1e-6 --beta 0.1 \
  --max-length 12288 --seed 20260925 --run-name rich_v3_balanced_graph_dpo64 \
  > "${PILOT}/dpo_train.log" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" > "${PILOT}/dpo_exit_code"
if [[ "${rc}" -eq 0 && -f "${OUTPUT}/run_manifest.json" && -d "${OUTPUT}/final_model" ]]; then
  printf '%s\n' TRAINED > "${PILOT}/dpo_status"
else
  printf 'FAILED:%s\n' "${rc}" > "${PILOT}/dpo_status"
fi
exit "${rc}"
