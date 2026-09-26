#!/usr/bin/env bash
set -uo pipefail

MODE=${1:-launch}
[[ "$MODE" == "launch" || "$MODE" == "worker" ]] || {
  echo "usage: $0 [launch|worker]" >&2
  exit 2
}

WORKTREE=/home/u2024311031/workspace/envfactory_repro_1p7b
TRAIN_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN_PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
SESSION=graph_frontier_dynamic_v2
OUTPUT="$WORKTREE/repro_1p7b/checkpoints/graph_frontier_dynamic_v2_8k_1p7b"
BOUNDARY="$OUTPUT/checkpoint-207"
SAFE_BOUNDARY="$WORKTREE/repro_1p7b/checkpoints/preserved_boundaries/graph_frontier_dynamic_v2_stage1_step207"
SAFE_TMP="$WORKTREE/repro_1p7b/checkpoints/preserved_boundaries/.graph_frontier_dynamic_v2_stage1_step207.incomplete"
LOG_DIR="$WORKTREE/repro_1p7b/logs/graph_frontier_dynamic_v2"
STATUS="$LOG_DIR/status"
STAGE1_CONFIG="$WORKTREE/repro_1p7b/configs/llamafactory_dynamic_v2_stage1.yaml"
STAGE2_CONFIG="$WORKTREE/repro_1p7b/configs/llamafactory_dynamic_v2_stage2.yaml"
STAGE1="$WORKTREE/repro_1p7b/data/graph_frontier_dynamic_v1/stage1_general.json"
STAGE2="$WORKTREE/repro_1p7b/data/graph_frontier_dynamic_v2/stage2_completion_efficiency.json"
PLAN="$WORKTREE/repro_1p7b/data/graph_frontier_dynamic_v2/stage2_allocation_plan_v2.json"
STATS="$WORKTREE/repro_1p7b/data/graph_frontier_dynamic_v2/stage2_dataset_stats_v2.json"
VALIDATION="$WORKTREE/repro_1p7b/data/graph_frontier_dynamic_v2/stage2_validation_v2.json"

cd "$WORKTREE" || exit 2
mkdir -p "$LOG_DIR"
set_status() {
  printf '%s\n' "$1" > "$STATUS"
  printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG_DIR/state_history.log"
}
fail() {
  set_status "$1"
  printf '%s\n' "$1" >&2
  exit "$2"
}
gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
    sed '/^[[:space:]]*$/d'
}

if [[ "$MODE" == "launch" ]]; then
  [[ ! -e "$OUTPUT" ]] || fail REFUSE_OVERWRITE_FORMAL_OUTPUT 20
  [[ ! -e "$SAFE_BOUNDARY" ]] || fail REFUSE_OVERWRITE_PRESERVED_BOUNDARY 21
  [[ ! -e "$SAFE_TMP" ]] || fail REFUSE_INCOMPLETE_PRESERVED_BOUNDARY 22
  [[ -z "$(gpu_pids)" ]] || fail GPU_BUSY 23
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    fail REFUSE_EXISTING_TMUX_SESSION 24
  fi
  set_status LAUNCHING_STAGE1_RETRAIN
  tmux new-session -d -s "$SESSION" "$0 worker"
  exit 0
fi

[[ ! -e "$OUTPUT" ]] || fail REFUSE_OVERWRITE_FORMAL_OUTPUT 30
[[ ! -e "$SAFE_BOUNDARY" ]] || fail REFUSE_OVERWRITE_PRESERVED_BOUNDARY 31
[[ ! -e "$SAFE_TMP" ]] || fail REFUSE_INCOMPLETE_PRESERVED_BOUNDARY 32
[[ -z "$(gpu_pids)" ]] || fail GPU_BUSY 33
[[ -s "$STAGE1" ]] || fail MISSING_STAGE1_DATA 34
[[ -s "$STAGE2" ]] || fail MISSING_STAGE2_DATA 35

"$RUN_PY" -m repro_1p7b.graph_frontier.dynamic_v2 validate \
  --stage1 "$STAGE1" --stage2 "$STAGE2" --plan "$PLAN" --stats "$STATS" \
  --report "$VALIDATION" > "$LOG_DIR/preflight_validation.log" 2>&1 \
  || fail FAILED_DATA_VALIDATION 36

set +u
source /opt/conda/etc/profile.d/conda.sh || fail FAILED_CONDA_SOURCE 40
conda activate "$TRAIN_ENV" || fail FAILED_CONDA_ACTIVATE 41
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

set_status STAGE1_RETRAINING
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 --master-port=29527 \
  -m repro_1p7b.graph_frontier.train_boundary "$STAGE1_CONFIG" \
  --stop-after-step 207 2>&1 | tee "$LOG_DIR/stage1_retrain.log"
stage1_rc=${PIPESTATUS[0]}
printf '%s\n' "$stage1_rc" > "$LOG_DIR/stage1_retrain.exit_code"
[[ "$stage1_rc" -eq 0 ]] || fail FAILED_STAGE1_RETRAIN "$stage1_rc"

"$RUN_PY" -m repro_1p7b.graph_frontier.continuity_audit checkpoint \
  --checkpoint "$BOUNDARY" --step 207 --horizon 414 \
  --report "$LOG_DIR/stage1_checkpoint_audit_retrained.json" \
  || fail FAILED_STAGE1_BOUNDARY_AUDIT 50

set_status PRESERVING_STAGE1_BOUNDARY
mkdir -p "$(dirname "$SAFE_BOUNDARY")"
mkdir "$SAFE_TMP" || fail FAILED_CREATE_PRESERVE_TMP 51
cp -a --reflink=auto "$BOUNDARY/." "$SAFE_TMP/" \
  || fail FAILED_COPY_PRESERVED_BOUNDARY 52
"$RUN_PY" -m repro_1p7b.graph_frontier.continuity_audit checkpoint \
  --checkpoint "$SAFE_TMP" --step 207 --horizon 414 \
  --report "$LOG_DIR/preserved_stage1_checkpoint_audit.json" \
  || fail FAILED_PRESERVED_BOUNDARY_AUDIT 53
mv "$SAFE_TMP" "$SAFE_BOUNDARY" || fail FAILED_FINALIZE_PRESERVED_BOUNDARY 54
printf '%s\n' "$SAFE_BOUNDARY" > "$LOG_DIR/preserved_stage1_boundary_path"
set_status STAGE1_BOUNDARY_PRESERVED

set_status STAGE2_TRAINING
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 --master-port=29528 \
  -m repro_1p7b.graph_frontier.train_boundary "$STAGE2_CONFIG" \
  --resume-from-checkpoint "$SAFE_BOUNDARY" --ignore-data-skip \
  2>&1 | tee "$LOG_DIR/stage2.log"
stage2_rc=${PIPESTATUS[0]}
printf '%s\n' "$stage2_rc" > "$LOG_DIR/stage2.exit_code"
[[ "$stage2_rc" -eq 0 ]] || fail FAILED_STAGE2_TRAINING "$stage2_rc"

final_step=$("$RUN_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("global_step",0))' "$OUTPUT/trainer_state.json")
[[ "$final_step" -eq 414 ]] || fail FAILED_FINAL_STEP_GATE 60
[[ -s "$SAFE_BOUNDARY/trainer_state.json" ]] || fail PRESERVED_BOUNDARY_MISSING_AFTER_TRAIN 61
set_status TRAINING_DONE
