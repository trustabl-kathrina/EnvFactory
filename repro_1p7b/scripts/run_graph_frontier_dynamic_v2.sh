#!/usr/bin/env bash
set -uo pipefail

MODE=${1:-audit}
[[ "$MODE" == "audit" || "$MODE" == "smoke" || "$MODE" == "formal" || "$MODE" == "launch" ]] || {
  echo "usage: $0 [audit|smoke|formal|launch]" >&2
  exit 2
}

WORKTREE=/home/u2024311031/workspace/envfactory_repro_1p7b
TRAIN_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN_PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
DATA_DIR="$WORKTREE/repro_1p7b/data/graph_frontier_dynamic_v2"
LOG_DIR="$WORKTREE/repro_1p7b/logs/graph_frontier_dynamic_v2"
BOUNDARY="$WORKTREE/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b/checkpoint-207"
STAGE1="$WORKTREE/repro_1p7b/data/graph_frontier_dynamic_v1/stage1_general.json"
STAGE2="$DATA_DIR/stage2_completion_efficiency.json"
PLAN="$DATA_DIR/stage2_allocation_plan_v2.json"
STATS="$DATA_DIR/stage2_dataset_stats_v2.json"
VALIDATION="$DATA_DIR/stage2_validation_v2.json"
CONFIG="$WORKTREE/repro_1p7b/configs/llamafactory_dynamic_v2_stage2.yaml"
SMOKE_CONFIG="$WORKTREE/repro_1p7b/configs/llamafactory_dynamic_v2_smoke.yaml"
FORMAL_OUTPUT="$WORKTREE/repro_1p7b/checkpoints/graph_frontier_dynamic_v2_8k_1p7b"
SMOKE_OUTPUT="$WORKTREE/repro_1p7b/checkpoints/graph_frontier_dynamic_v2_smoke"
AUDIT="$LOG_DIR/stage1_continuation_audit_v2.json"
STATUS="$LOG_DIR/status"

cd "$WORKTREE" || exit 2
mkdir -p "$LOG_DIR"
set_status() {
  printf '%s\n' "$1" > "$STATUS"
  printf '%s %s\n' "$(date -Is)" "$1" >> "$LOG_DIR/state_history.log"
}
fail() {
  set_status "$1"
  echo "$1" >&2
  exit "$2"
}

"$RUN_PY" -m repro_1p7b.graph_frontier.dynamic_v2 validate \
  --stage1 "$STAGE1" --stage2 "$STAGE2" --plan "$PLAN" --stats "$STATS" \
  --report "$VALIDATION" || fail FAILED_DATA_VALIDATION 10

"$RUN_PY" -m repro_1p7b.graph_frontier.dynamic_v2 continuation-audit \
  --boundary "$BOUNDARY" --stage2 "$STAGE2" --config "$CONFIG" --report "$AUDIT"
audit_rc=$?
if [[ "$audit_rc" -ne 0 ]]; then
  [[ "$audit_rc" -eq 20 ]] && fail BLOCKED_STAGE1_STATE_MISSING 20
  fail FAILED_CONTINUATION_AUDIT "$audit_rc"
fi
set_status READY

[[ "$MODE" == "audit" ]] && exit 0

gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
[[ -z "$gpu_pids" ]] || fail GPU_BUSY 25

if [[ "$MODE" == "launch" ]]; then
  [[ ! -e "$FORMAL_OUTPUT" ]] || fail REFUSE_OVERWRITE_FORMAL_OUTPUT 21
  [[ ! -e "$SMOKE_OUTPUT" ]] || fail REFUSE_OVERWRITE_SMOKE_OUTPUT 32
  if tmux has-session -t graph_frontier_dynamic_v2 2>/dev/null; then
    fail REFUSE_EXISTING_TMUX_SESSION 22
  fi
  "$WORKTREE/repro_1p7b/scripts/run_graph_frontier_dynamic_v2.sh" smoke \
    || fail FAILED_CONTINUATION_SMOKE 33
  set_status SMOKE_STEP_208_CONFIRMED
  tmux new-session -d -s graph_frontier_dynamic_v2 \
    "$WORKTREE/repro_1p7b/scripts/run_graph_frontier_dynamic_v2.sh formal"
  set_status FORMAL_LAUNCHED
  for _ in $(seq 1 180); do
    if [[ -s "$FORMAL_OUTPUT/continuity_audit/step_trace.jsonl" ]] \
      && grep -q '"global_step": 208' "$FORMAL_OUTPUT/continuity_audit/step_trace.jsonl"; then
      set_status FORMAL_STEP_208_CONFIRMED
      exit 0
    fi
    tmux has-session -t graph_frontier_dynamic_v2 2>/dev/null \
      || fail FORMAL_EXITED_BEFORE_STEP_208 23
    sleep 10
  done
  fail FORMAL_STEP_208_TIMEOUT 24
fi

set +u
source /opt/conda/etc/profile.d/conda.sh || fail FAILED_CONDA_SOURCE 30
conda activate "$TRAIN_ENV" || fail FAILED_CONDA_ACTIVATE 31
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

if [[ "$MODE" == "smoke" ]]; then
  [[ ! -e "$SMOKE_OUTPUT" ]] || fail REFUSE_OVERWRITE_SMOKE_OUTPUT 32
  set_status SMOKE_TRAINING
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 --master-port=29525 \
    -m repro_1p7b.graph_frontier.train_boundary "$SMOKE_CONFIG" \
    --stop-after-step 208 --resume-from-checkpoint "$BOUNDARY" --ignore-data-skip \
    2>&1 | tee "$LOG_DIR/continuation_smoke.log"
  rc=${PIPESTATUS[0]}
  [[ "$rc" -eq 0 ]] || fail FAILED_CONTINUATION_SMOKE "$rc"
  [[ -s "$SMOKE_OUTPUT/continuity_audit/step_trace.jsonl" ]] \
    && grep -q '"global_step": 208' "$SMOKE_OUTPUT/continuity_audit/step_trace.jsonl" \
    || fail FAILED_SMOKE_STEP_208_GATE 33
  set_status SMOKE_STEP_208_CONFIRMED
  exit 0
fi

[[ ! -e "$FORMAL_OUTPUT" ]] || fail REFUSE_OVERWRITE_FORMAL_OUTPUT 40
set_status FORMAL_TRAINING
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 --master-port=29526 \
  -m repro_1p7b.graph_frontier.train_boundary "$CONFIG" \
  --resume-from-checkpoint "$BOUNDARY" --ignore-data-skip \
  2>&1 | tee "$LOG_DIR/stage2.log"
rc=${PIPESTATUS[0]}
[[ "$rc" -eq 0 ]] || fail FAILED_FORMAL_TRAINING "$rc"
set_status FORMAL_TRAINING_DONE
