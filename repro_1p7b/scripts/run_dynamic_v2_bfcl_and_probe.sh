#!/usr/bin/env bash
set -uo pipefail

MODE=${1:-launch}
[[ "$MODE" == "launch" || "$MODE" == "worker" ]] || {
  echo "usage: $0 [launch|worker]" >&2
  exit 2
}

WORKTREE=/home/u2024311031/workspace/envfactory_repro_1p7b
RUN_LABEL=${RUN_LABEL:-dynamic_v2}
EVAL_DIR="$WORKTREE/repro_1p7b/evaluation/bfcl"
MODEL=${MODEL_OVERRIDE:-"$WORKTREE/repro_1p7b/checkpoints/graph_frontier_dynamic_v2_8k_1p7b"}
BFCL_ENV=/home/u2024311031/.conda/envs/envfactory_bfcl_v1p3
RUN_PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
BFCL_PY="$BFCL_ENV/bin/python"
BFCL_BIN="$BFCL_ENV/bin/bfcl"
BFCL_MODEL_ID=Qwen/Qwen3-1.7B-FC
BFCL_MODEL_DIR=Qwen_Qwen3-1.7B-FC
PRIOR_RESULT="$EVAL_DIR/artifacts/full/dynamic_v1/result/$BFCL_MODEL_DIR"
SHARD_ROOT="$EVAL_DIR/artifacts/full/${RUN_LABEL}_shards"
FINAL_ROOT="$EVAL_DIR/artifacts/full/${RUN_LABEL}"
LOG_DIR=${EVAL_LOG_DIR:-"$WORKTREE/repro_1p7b/logs/graph_frontier_${RUN_LABEL}_eval"}
BFCL_LOG_DIR="$EVAL_DIR/logs/full"
PLAN="$LOG_DIR/bfcl_slice_plan.json"
PROBE_ROOT="$WORKTREE/repro_1p7b/results/graph_frontier/confirm_300"
PROBE_OUTPUT="$PROBE_ROOT/$RUN_LABEL"
MANIFEST="$PROBE_ROOT/frozen/confirm_300_seed_20260914.jsonl"
REPORT_DIR="$WORKTREE/repro_1p7b/graph_frontier/reports"
SESSION=${EVAL_SESSION:-graph_frontier_dynamic_v2_eval}
STATUS="$LOG_DIR/status"

cd "$WORKTREE" || exit 2
mkdir -p "$LOG_DIR" "$BFCL_LOG_DIR"

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
port_free() {
  "$RUN_PY" -c 'import socket,sys; s=socket.socket(); s.settimeout(.2); sys.exit(0 if s.connect_ex(("127.0.0.1",int(sys.argv[1]))) else 1)' "$1"
}

if [[ "$MODE" == "launch" ]]; then
  [[ -z "$(gpu_pids)" ]] || fail GPU_BUSY 20
  [[ ! -e "$SHARD_ROOT" ]] || fail REFUSE_EXISTING_BFCL_SHARDS 21
  [[ ! -e "$FINAL_ROOT" ]] || fail REFUSE_EXISTING_BFCL_RESULTS 22
  [[ ! -e "$PROBE_OUTPUT" ]] || fail REFUSE_EXISTING_PROBE_RESULTS 23
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    fail REFUSE_EXISTING_TMUX_SESSION 24
  fi
  set_status LAUNCHING_BFCL
  tmux new-session -d -s "$SESSION" "$0 worker"
  exit 0
fi

source "$EVAL_DIR/scripts/common.sh"
require_clean_bfcl || fail BFCL_ENVIRONMENT_DIRTY 30
require_model "$MODEL" || fail MODEL_INCOMPLETE 31
[[ -z "$(gpu_pids)" ]] || fail GPU_BUSY 32
for port in 1074 1075 1076 1077; do
  port_free "$port" || fail PORT_BUSY_$port 33
done
[[ ! -e "$SHARD_ROOT" ]] || fail REFUSE_EXISTING_BFCL_SHARDS 34
[[ ! -e "$FINAL_ROOT" ]] || fail REFUSE_EXISTING_BFCL_RESULTS 35
[[ ! -e "$PROBE_OUTPUT" ]] || fail REFUSE_EXISTING_PROBE_RESULTS 36

"$RUN_PY" "$EVAL_DIR/scripts/build_category_slices_v2.py" \
  --prior-result-dir "$PRIOR_RESULT" \
  --output-root "$SHARD_ROOT" \
  --plan "$PLAN" > "$LOG_DIR/bfcl_slice_build.log" 2>&1 \
  || fail BFCL_SLICE_BUILD_FAILED 40

run_bfcl_shard() (
  set -uo pipefail
  gpu="$1"
  port="$2"
  root="$3"
  prefix="$4"
  export CUDA_HOME="$BFCL_ENV"
  export PATH="$CUDA_HOME/bin:$PATH"
  export CC="$BFCL_ENV/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$BFCL_ENV/bin/x86_64-conda-linux-gnu-c++"
  export GCC="$BFCL_ENV/bin/x86_64-conda-linux-gnu-gcc"
  export GXX="$BFCL_ENV/bin/x86_64-conda-linux-gnu-g++"
  export NVCC_PREPEND_FLAGS="-ccbin=$CXX"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export BFCL_PROJECT_ROOT="$root"
  export VLLM_ENDPOINT=127.0.0.1
  export VLLM_PORT="$port"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  "$BFCL_PY" -m sglang.launch_server \
    --host 127.0.0.1 --model-path "$MODEL" --port "$port" \
    --dtype bfloat16 --tp-size 1 --mem-fraction-static 0.75 \
    --trust-remote-code --disable-cuda-graph \
    --attention-backend triton --sampling-backend pytorch \
    --context-length 40960 --max-total-tokens 65536 \
    > "$BFCL_LOG_DIR/${prefix}_server.log" 2>&1 &
  server_pid=$!
  cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM
  "$BFCL_PY" -c 'import sys,time,requests; u="http://127.0.0.1:"+sys.argv[1]+"/v1/models"; end=time.time()+300
while time.time()<end:
    try:
        if requests.get(u,timeout=2).status_code==200: raise SystemExit(0)
    except Exception: pass
    time.sleep(2)
raise SystemExit(1)' "$port" || exit 51
  "$BFCL_BIN" generate \
    --model "$BFCL_MODEL_ID" --backend sglang --skip-server-setup \
    --num-gpus 1 --gpu-memory-utilization 0.75 --temperature 0.7 \
    --include-input-log --allow-overwrite --local-model-path "$MODEL" \
    --run-ids > "$BFCL_LOG_DIR/${prefix}_generate.log" 2>&1
)
set_status BFCL_GENERATING
run_bfcl_shard 0 1074 "$SHARD_ROOT/gpu0" "${RUN_LABEL}_gpu0" &
bfcl0=$!
run_bfcl_shard 1 1075 "$SHARD_ROOT/gpu1" "${RUN_LABEL}_gpu1" &
bfcl1=$!
set +e
wait "$bfcl0"; rc0=$?
wait "$bfcl1"; rc1=$?
set -e
printf '%s\n' "$rc0" > "$LOG_DIR/bfcl_gpu0.exit_code"
printf '%s\n' "$rc1" > "$LOG_DIR/bfcl_gpu1.exit_code"
[[ "$rc0" -eq 0 && "$rc1" -eq 0 ]] || fail BFCL_GENERATION_FAILED 50

set_status BFCL_MERGING
"$RUN_PY" "$EVAL_DIR/scripts/merge_latency_slices_v2.py" \
  --shard-root "$SHARD_ROOT" --output-root "$FINAL_ROOT" --plan "$PLAN" \
  > "$LOG_DIR/bfcl_merge.log" 2>&1 || fail BFCL_MERGE_FAILED 51

set_status BFCL_SCORING
export BFCL_PROJECT_ROOT="$FINAL_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$BFCL_BIN" evaluate --model "$BFCL_MODEL_ID" --test-category multi_turn \
  > "$BFCL_LOG_DIR/${RUN_LABEL}_score.log" 2>&1 \
  || fail BFCL_SCORING_FAILED 52
set_status BFCL_DONE

run_probe_shard() (
  set -uo pipefail
  gpu="$1"
  port="$2"
  shard="$3"
  key="${RUN_LABEL}-probe-$shard"
  export CUDA_HOME="$BFCL_ENV"
  export PATH="$CUDA_HOME/bin:$PATH"
  export CC="$BFCL_ENV/bin/x86_64-conda-linux-gnu-cc"
  export CXX="$BFCL_ENV/bin/x86_64-conda-linux-gnu-c++"
  export GCC="$BFCL_ENV/bin/x86_64-conda-linux-gnu-gcc"
  export GXX="$BFCL_ENV/bin/x86_64-conda-linux-gnu-g++"
  export NVCC_PREPEND_FLAGS="-ccbin=$CXX"
  export CUDA_VISIBLE_DEVICES="$gpu"
  "$BFCL_PY" -m sglang.launch_server \
    --model-path "$MODEL" --host 127.0.0.1 --port "$port" --api-key "$key" \
    --dtype bfloat16 --tp-size 1 --mem-fraction-static 0.70 \
    --disable-cuda-graph --attention-backend triton --sampling-backend pytorch \
    --context-length 32768 --max-total-tokens 49152 --trust-remote-code \
    > "$PROBE_OUTPUT/server_shard${shard}.log" 2>&1 &
  server_pid=$!
  cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM
  "$BFCL_PY" -c 'import sys,time,requests; u="http://127.0.0.1:"+sys.argv[1]+"/v1/models"; h={"Authorization":"Bearer "+sys.argv[2]}; end=time.time()+300
while time.time()<end:
    try:
        if requests.get(u,headers=h,timeout=2).status_code==200: raise SystemExit(0)
    except Exception: pass
    time.sleep(2)
raise SystemExit(1)' "$port" "$key" || exit 61
  export LITELLM_LOCAL_MODEL_COST_MAP=True
  export SGLANG_BASE_URL="http://127.0.0.1:$port/v1"
  export SGLANG_API_KEY="$key"
  export SGLANG_MODEL="$MODEL"
  "$RUN_PY" -m repro_1p7b.graph_frontier.confirm_300 run \
    --manifest "$MANIFEST" --model "$RUN_LABEL" --output-dir "$PROBE_OUTPUT" \
    --shard-count 2 --shard-index "$shard" \
    > "$PROBE_OUTPUT/eval${shard}.log" 2>&1
)
set_status PROBE_RUNNING
mkdir -p "$PROBE_OUTPUT"
run_probe_shard 0 1076 0 &
probe0=$!
run_probe_shard 1 1077 1 &
probe1=$!
set +e
wait "$probe0"; prc0=$?
wait "$probe1"; prc1=$?
set -e
printf '%s\n' "$prc0" > "$LOG_DIR/probe_gpu0.exit_code"
printf '%s\n' "$prc1" > "$LOG_DIR/probe_gpu1.exit_code"
[[ "$prc0" -eq 0 && "$prc1" -eq 0 ]] || fail PROBE_FAILED 60

"$RUN_PY" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path
label=os.environ.get("RUN_LABEL","dynamic_v2")
root=Path("repro_1p7b/results/graph_frontier/confirm_300")/label
rows=[json.loads((root/f"run_summary.shard{i}.json").read_text()) for i in range(2)]
summary={
    "model":label,
    "completed":sum(x["completed"] for x in rows),
    "valid":sum(x["valid"] for x in rows),
    "reference_path_complete_success":sum(x["reference_path_complete_success"] for x in rows),
    "runtime_seconds_parallel_makespan":max(x["runtime_seconds"] for x in rows),
    "runtime_seconds_sum":sum(x["runtime_seconds"] for x in rows),
    "system_error_counts":dict(sum((Counter(x["system_error_counts"]) for x in rows),Counter())),
    "shards":rows,
}
(root/"run_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2))
PY

set_status PROBE_ANALYZING
if [[ "$RUN_LABEL" == dynamic_v2 ]]; then
  "$RUN_PY" -m repro_1p7b.graph_frontier.confirm_analyze_dynamic_v2 \
    --root "$PROBE_ROOT" --bfcl-root "$EVAL_DIR/artifacts/full" \
    --reports "$REPORT_DIR" > "$LOG_DIR/probe_analysis.log" 2>&1 || fail PROBE_ANALYSIS_FAILED 61
else
  printf '%s\n' 'BFCL scored; frozen-300 run_summary.json aggregated' > "$LOG_DIR/probe_analysis.log"
fi

set_status DONE
