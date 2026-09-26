#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
GF=${ROOT}/repro_1p7b/graph_frontier
OUT=${GF}/balanced_preference_v2
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
SMOKE=${ROOT}/repro_1p7b/checkpoints/balanced_graph_dpo_v2_smoke24
PILOT=${ROOT}/repro_1p7b/checkpoints/balanced_graph_dpo_v2_pilot
DEEPSPEED=${ROOT}/repro_1p7b/configs/ds_z3_config.json
cd "${ROOT}"
trap 'rc=$?; printf "FAILED:%s:line%s\n" "${rc}" "${LINENO}" > "${OUT}/training_status"' ERR
[[ "$(cat "${OUT}/sampling_status")" == SAMPLING_DONE ]] || { echo 'stop sampling incomplete' >&2; exit 2; }
[[ ! -e "${OUT}/training_status" && ! -e "${SMOKE}" && ! -e "${PILOT}" ]] || { echo 'training output already exists' >&2; exit 2; }
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
printf '%s\n' DATA_AUDIT > "${OUT}/training_status"
if [[ ! -f "${OUT}/manifest.json" ]]; then
  "${ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v2_pipeline build > "${OUT}/build.log" 2>&1
fi
"${ENV}/bin/python" - "${OUT}" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); audit=json.loads((p/'audit.json').read_text())
if audit['verdict']!='DATA_READY' or not (p/'manifest.json').exists():
    raise SystemExit(f"data gate failed: {audit}")
PY
printf '%s\n' SERIALIZING > "${OUT}/training_status"
if [[ ! -f "${OUT}/serialization_audit.json" ]]; then
  "${ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v2_pipeline materialize > "${OUT}/materialize.log" 2>&1
fi
"${ENV}/bin/python" - "${OUT}" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); audit=json.loads((p/'serialization_audit.json').read_text())
if audit['verdict']!='DPO_READY' or audit['smoke_types']!={'continue_required':8,'stop_required':8,'downstream_continue':8}:
    raise SystemExit(f"serialization gate failed: {audit}")
PY
printf '%s\n' SMOKE_TRAINING > "${OUT}/training_status"
CUDA_VISIBLE_DEVICES=0,1 "${ENV}/bin/torchrun" --standalone --nproc_per_node=2 \
  repro_1p7b/graph_frontier/train_balanced_preference_v2.py \
  --model "${MODEL}" --train-file "${OUT}/dpo_smoke24.jsonl" \
  --eval-file "${OUT}/dpo_val.jsonl" --preference-manifest "${OUT}/manifest.json" \
  --output-dir "${SMOKE}" --deepspeed "${DEEPSPEED}" --max-steps 12 \
  --seed 20260925 --run-name balanced_preference_v2_smoke24 \
  > "${OUT}/smoke_train.log" 2>&1
"${ENV}/bin/python" - "${SMOKE}/run_manifest.json" "${OUT}/smoke_verdict.json" <<'PY'
import json, math, pathlib, sys
p=pathlib.Path(sys.argv[1]); m=json.loads(p.read_text()); hist=[x for x in m['log_history'] if isinstance(x.get('step'),int) and 'loss' in x]
numeric=[v for x in hist for v in x.values() if isinstance(v,float)]
end=hist[-1] if hist else {}
passed=(m['max_steps']==12 and len(hist)>=12 and all(math.isfinite(v) for v in numeric)
        and end.get('rewards/margins',-1)>0 and end.get('rewards/accuracies',0)>=0.5
        and len(list((p.parent/'checkpoint-12'/'global_step12').glob('*optim_states.pt')))==2
        and all(t in m['type_eval_metrics'] for t in ('continue_required','stop_required','downstream_continue')))
verdict={'verdict':'SMOKE_PASS' if passed else 'SMOKE_FAILED','last_train_step':end,'train_metrics':m['train_metrics'],'type_eval_metrics':m['type_eval_metrics']}
pathlib.Path(sys.argv[2]).write_text(json.dumps(verdict,indent=2,sort_keys=True)+'\n')
if not passed: raise SystemExit('smoke gate failed; pilot not started')
PY
printf '%s\n' PILOT_TRAINING > "${OUT}/training_status"
CUDA_VISIBLE_DEVICES=0,1 "${ENV}/bin/torchrun" --standalone --nproc_per_node=2 \
  repro_1p7b/graph_frontier/train_balanced_preference_v2.py \
  --model "${MODEL}" --train-file "${OUT}/dpo_train.jsonl" \
  --eval-file "${OUT}/dpo_val.jsonl" --preference-manifest "${OUT}/manifest.json" \
  --output-dir "${PILOT}" --deepspeed "${DEEPSPEED}" --max-steps 64 \
  --seed 20260925 --run-name balanced_preference_v2_pilot64 \
  > "${OUT}/pilot_train.log" 2>&1
"${ENV}/bin/python" - "${PILOT}" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); m=json.loads((p/'run_manifest.json').read_text())
if m['max_steps']!=64 or m['initialization_model']!=m['reference_model'] or len(list((p/'checkpoint-64'/'global_step64').glob('*optim_states.pt')))!=2 or not (p/'final_model').is_dir():
    raise SystemExit('pilot checkpoint/identity gate failed')
PY
printf '%s\n' PILOT_TRAINED > "${OUT}/training_status"
