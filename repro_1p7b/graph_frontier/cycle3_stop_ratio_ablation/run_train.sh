#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RATIO=${1:?ratio 2 or 4 required}
[[ "$RATIO" == 2 || "$RATIO" == 4 ]] || exit 2
ABL=$ROOT/repro_1p7b/graph_frontier/cycle3_stop_ratio_ablation
OUT=$ROOT/repro_1p7b/checkpoints/cycle3_execfd_eos_${RATIO}to1_64
LOG=$ABL/train_${RATIO}to1
STATUS=$ABL/train_${RATIO}to1_status
cd "$ROOT"
[[ ! -e "$OUT" && ! -e "$LOG" && ! -e "$STATUS" ]] || { echo 'ablation training output exists' >&2; exit 2; }
export CUDA_HOME=$ENV PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$ENV/bin/python" - "$RATIO" <<'PY'
import subprocess,sys
from repro_1p7b.graph_frontier.cycle3_stop_ratio_ablation.train import load_plan
ratio=int(sys.argv[1]); pairs,plan=load_plan(ratio)
assert len(pairs)==64 and plan['world_size']==2 and plan['exposures']['FD']+plan['exposures']['terminal_stop']==128
used=[int(x.strip()) for x in subprocess.check_output(
    ['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
if len(used)<2 or any(x>=1500 for x in used[:2]):
    raise SystemExit(f'GPU busy: {used[:2]}')
print('EOS ratio training preflight PASS',ratio,plan['exposures'],used[:2],flush=True)
PY
printf '%s\n' TRAINING > "$STATUS"
set +e
CUDA_VISIBLE_DEVICES=0,1 "$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  -m repro_1p7b.graph_frontier.cycle3_stop_ratio_ablation.train \
  --ratio "$RATIO" --output "$OUT" --logs "$LOG"
RC=$?
set -e
if [[ "$RC" -ne 0 ]]; then
  printf 'TRAINING_FAILED:%s\n' "$RC" > "$STATUS"
  exit "$RC"
fi
"$ENV/bin/python" - "$LOG/result.json" "$RATIO" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text()); ratio=int(sys.argv[2])
assert r['status']=='TRAINED' and r['steps']==64 and r['world_size']==2
assert r['ratio']==ratio and r['full_parameter'] and r['precision']=='bfloat16'
assert r['all_loss_finite'] and r['all_grad_finite']
assert r['exposures']==({'FD':85,'terminal_stop':43} if ratio==2 else {'FD':102,'terminal_stop':26})
print('EOS ratio checkpoint gate PASS',ratio,flush=True)
PY
cp -n "$ABL/schedule_${RATIO}.json" "$OUT/sampling_manifest.json"
cp -n "$ABL/train_${RATIO}to1/plan.json" "$OUT/optimizer_plan.json"
cp -n "$ROOT/repro_1p7b/graph_frontier/cycle3_pilot/data_execfd_stop_64_v1/train_manifest.json" "$OUT/train_manifest.json"
printf '%s\n' COMPLETE > "$STATUS"
