#!/usr/bin/env bash
# Same v2 DPO trainer/configuration; only the audited v3 data are changed.
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
GF=${ROOT}/repro_1p7b/graph_frontier
MILESTONE=${BALANCED_V3_MILESTONE:-milestone_24}
OUT=${GF}/balanced_preference_v3/${MILESTONE}
MODEL=${ROOT}/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b
SMOKE=${ROOT}/repro_1p7b/checkpoints/balanced_graph_dpo_v3_replication_smoke24
PILOT=${ROOT}/repro_1p7b/checkpoints/balanced_graph_dpo_v3_replication
DEEPSPEED=${ROOT}/repro_1p7b/configs/ds_z3_config.json
export BALANCED_V3_MILESTONE=${MILESTONE}
cd "${ROOT}"
trap 'rc=$?; printf "FAILED:%s:line%s\n" "${rc}" "${LINENO}" > "${OUT}/training_status"' ERR
[[ "$(cat "${OUT}/sampling_status")" == DATA_READY ]] || { echo 'v3 data sampling gate not ready' >&2; exit 2; }
[[ ! -e "${OUT}/training_status" && ! -e "${SMOKE}" && ! -e "${PILOT}" ]] || {
  echo 'v3 training output already exists; refusing overwrite' >&2; exit 2;
}
export PATH=${ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NCCL_DEBUG=WARN
printf '%s\n' DATA_AUDIT > "${OUT}/training_status"
"${ENV}/bin/python" - "${OUT}" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); a=json.loads((p/'audit.json').read_text()); m=json.loads((p/'manifest.json').read_text())
assert a['verdict']==m['verdict']=='DATA_READY'
assert a['train_unique_states']>=180 and a['train_pairs']>=280
assert a['chosen_final_answer_fraction']>=.25
assert a['overlap']=={'train_val_tasks':0,'train_new_heldout_tasks':0,'val_new_heldout_tasks':0,
                      'frozen300_exact':0,'historical_eval_exact':0}
assert a['chosen_validation_rate']==1.0 and a['systematic_issues']==0
assert a['validation_unique_states']>=30
assert a['validation_type_unique_states']['continue_required']>=10
assert a['validation_type_unique_states']['stop_required']>=10
assert a['validation_type_unique_states']['downstream_continue']>=8
PY
printf '%s\n' SERIALIZING > "${OUT}/training_status"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v3_materialize \
  > "${OUT}/materialize.log" 2>&1
"${ENV}/bin/python" - "${OUT}" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); a=json.loads((p/'serialization_audit.json').read_text())
assert a['verdict']=='DPO_READY' and a['smoke_types']=={
    'continue_required':8,'stop_required':8,'downstream_continue':8}
assert not a['drops']
PY
printf '%s\n' SMOKE_TRAINING > "${OUT}/training_status"
CUDA_VISIBLE_DEVICES=0,1 "${ENV}/bin/torchrun" --standalone --nproc_per_node=2 \
  repro_1p7b/graph_frontier/train_balanced_preference_v2.py \
  --model "${MODEL}" --train-file "${OUT}/dpo_smoke24.jsonl" \
  --eval-file "${OUT}/dpo_val.jsonl" --preference-manifest "${OUT}/manifest.json" \
  --output-dir "${SMOKE}" --deepspeed "${DEEPSPEED}" --max-steps 2 \
  --seed 20260925 --run-name balanced_preference_v3_smoke24 \
  > "${OUT}/smoke_train.log" 2>&1
"${ENV}/bin/python" - "${SMOKE}/run_manifest.json" "${OUT}/smoke_verdict.json" <<'PY'
import json,math,pathlib,sys
p=pathlib.Path(sys.argv[1]); m=json.loads(p.read_text()); hist=[x for x in m['log_history'] if 'loss' in x]
numeric=[v for x in hist for v in x.values() if isinstance(v,(int,float)) and not isinstance(v,bool)]
ready=(m['max_steps']==2 and len(hist)>=2 and all(math.isfinite(v) for v in numeric)
       and len(list((p.parent/'checkpoint-2'/'global_step2').glob('*optim_states.pt')))==2
       and all(t in m['type_eval_metrics'] for t in
               ('continue_required','stop_required','downstream_continue')))
verdict={'verdict':'SMOKE_PASS' if ready else 'SMOKE_FAILED',
         'steps':len(hist),'last_step':hist[-1] if hist else {},
         'type_eval_metrics':m['type_eval_metrics']}
pathlib.Path(sys.argv[2]).write_text(json.dumps(verdict,indent=2,sort_keys=True)+'\n')
if not ready: raise SystemExit('2-GPU v3 memory/serialization smoke failed')
PY
printf '%s\n' PILOT_TRAINING > "${OUT}/training_status"
CUDA_VISIBLE_DEVICES=0,1 "${ENV}/bin/torchrun" --standalone --nproc_per_node=2 \
  repro_1p7b/graph_frontier/train_balanced_preference_v2.py \
  --model "${MODEL}" --train-file "${OUT}/dpo_train.jsonl" \
  --eval-file "${OUT}/dpo_val.jsonl" --preference-manifest "${OUT}/manifest.json" \
  --output-dir "${PILOT}" --deepspeed "${DEEPSPEED}" --max-steps 64 \
  --seed 20260925 --run-name balanced_preference_v3_replication64 \
  > "${OUT}/pilot_train.log" 2>&1
"${ENV}/bin/python" - "${PILOT}" "${OUT}/manifest.json" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); m=json.loads((p/'run_manifest.json').read_text()); pref=pathlib.Path(sys.argv[2])
assert m['max_steps']==64 and m['full_parameter'] is True and m['gpus']==[0,1]
assert m['initialization_model']==m['reference_model']
assert m['preference_manifest_sha256']==hashlib.sha256(pref.read_bytes()).hexdigest()
assert len(list((p/'checkpoint-64'/'global_step64').glob('*optim_states.pt')))==2
assert (p/'final_model').is_dir()
PY
printf '%s\n' PILOT_TRAINED > "${OUT}/training_status"
