#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-launch}
[[ "$MODE" == launch || "$MODE" == worker ]] || { echo "usage: $0 [launch|worker]" >&2; exit 2; }
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
RUN_PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
BFCL_ENV=/home/u2024311031/.conda/envs/envfactory_bfcl_v1p3
BFCL_PY=$BFCL_ENV/bin/python
BFCL_BIN=$BFCL_ENV/bin/bfcl
BFCL_REPO=/home/u2024311031/benchmarks/gorilla_bfcl_v1p3/berkeley-function-call-leaderboard
EXPECTED_BFCL_COMMIT=ea13468e4423454d0c213704fb87cf7cb3990433
MANIFEST=$ROOT/repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl
EXPECTED_MANIFEST_HASH=4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c
STD_MODEL=$ROOT/repro_1p7b/checkpoints_eval/standard_rl_v2_control_pilot128_step64_hf
GRAPH_MODEL=$ROOT/repro_1p7b/checkpoints_eval/graph_frontier_rl_v2_pilot128_step64_hf
STD_ROOT=$ROOT/repro_1p7b/graph_frontier/evals/standard_rl_v2_control_step64
GRAPH_ROOT=$ROOT/repro_1p7b/graph_frontier/evals/graph_frontier_rl_v2_step64
CONTROL=$ROOT/repro_1p7b/graph_frontier/evals/_guided_rl_v2_control
STATUS=$CONTROL/status
SESSION=gf_v2_benchmark_eval
ADAPTER=$ROOT/repro_1p7b/graph_frontier/eval_checkpoint_frozen300.py
BFCL_MODEL_ID=Qwen/Qwen3-1.7B-FC

mkdir -p "$CONTROL"
set_status() { printf '%s\n' "$1" > "$STATUS"; printf '%s %s\n' "$(date -Is)" "$1" >> "$CONTROL/state_history.log"; }
fail() { set_status "$1"; echo "$1" >&2; exit "${2:-1}"; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'; }
port_free() { "$RUN_PY" -c 'import socket,sys; s=socket.socket(); s.settimeout(.2); sys.exit(0 if s.connect_ex(("127.0.0.1",int(sys.argv[1]))) else 1)' "$1"; }
model_complete() { test -f "$1/config.json" && test -f "$1/tokenizer_config.json" && test -f "$1/tokenizer.json" && compgen -G "$1/*.safetensors" >/dev/null; }

if [[ "$MODE" == launch ]]; then
  [[ -z "$(gpu_pids)" ]] || fail REFUSE_GPU_BUSY 20
  [[ ! -e "$STD_ROOT" && ! -e "$GRAPH_ROOT" ]] || fail REFUSE_EXISTING_OUTPUT 21
  ! tmux has-session -t "$SESSION" 2>/dev/null || fail REFUSE_EXISTING_TMUX 22
  set_status LAUNCHING
  tmux new-session -d -s "$SESSION" "$0 worker"
  echo "launched $SESSION"
  exit 0
fi

on_exit() { rc=$?; if [[ $rc -ne 0 ]]; then set_status "FAILED_EXIT_$rc"; fi; }
trap on_exit EXIT
cd "$ROOT"
model_complete "$STD_MODEL" || fail STANDARD_MODEL_INCOMPLETE 30
model_complete "$GRAPH_MODEL" || fail GRAPH_MODEL_INCOMPLETE 31
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$EXPECTED_MANIFEST_HASH" ]] || fail MANIFEST_HASH_MISMATCH 32
[[ "$(git -C "$BFCL_REPO" rev-parse HEAD)" == "$EXPECTED_BFCL_COMMIT" ]] || fail BFCL_COMMIT_MISMATCH 33
[[ -z "$(git -C "$BFCL_REPO" status --porcelain)" ]] || fail BFCL_REPO_DIRTY 34
[[ -z "$(gpu_pids)" ]] || fail GPU_BUSY 35
for port in 1082 1083 1084 1085; do port_free "$port" || fail "PORT_BUSY_$port" 36; done
[[ ! -e "$STD_ROOT" && ! -e "$GRAPH_ROOT" ]] || fail OUTPUT_EXISTS 37

run_frozen_arm() (
  set -euo pipefail
  gpu=$1; port=$2; key=$3; label=$4; model=$5; arm_root=$6
  output=$arm_root/frozen_300
  mkdir -p "$output/logs"
  export CUDA_VISIBLE_DEVICES=$gpu
  export CUDA_HOME=$BFCL_ENV
  export PATH=$CUDA_HOME/bin:$PATH
  export CC=$BFCL_ENV/bin/x86_64-conda-linux-gnu-cc
  export CXX=$BFCL_ENV/bin/x86_64-conda-linux-gnu-c++
  export GCC=$BFCL_ENV/bin/x86_64-conda-linux-gnu-gcc
  export GXX=$BFCL_ENV/bin/x86_64-conda-linux-gnu-g++
  export NVCC_PREPEND_FLAGS="-ccbin=$CXX"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export MCP_CONFIG_PATH=$ROOT/configs/mcp_server.json
  "$BFCL_PY" -m sglang.launch_server --model-path "$model" --host 127.0.0.1 --port "$port" --api-key "$key" \
    --dtype bfloat16 --tp-size 1 --mem-fraction-static 0.70 --disable-cuda-graph \
    --attention-backend triton --sampling-backend pytorch --context-length 32768 \
    --max-total-tokens 49152 --trust-remote-code > "$output/logs/server.log" 2>&1 &
  server_pid=$!
  cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
  "$BFCL_PY" -c 'import sys,time,requests; u="http://127.0.0.1:"+sys.argv[1]+"/v1/models"; h={"Authorization":"Bearer "+sys.argv[2]}; end=time.time()+300
while time.time()<end:
  try:
    if requests.get(u,headers=h,timeout=2).status_code==200: raise SystemExit(0)
  except Exception: pass
  time.sleep(2)
raise SystemExit(1)' "$port" "$key"
  export LITELLM_LOCAL_MODEL_COST_MAP=True
  export SGLANG_BASE_URL="http://127.0.0.1:$port/v1" SGLANG_API_KEY="$key" SGLANG_MODEL="$model"
  "$RUN_PY" -m repro_1p7b.graph_frontier.eval_checkpoint_frozen300 \
    --label "$label" --model-path "$model" --manifest "$MANIFEST" --output-dir "$output" \
    > "$output/logs/eval.log" 2>&1
)

set_status FROZEN300_STARTING
run_frozen_arm 0 1082 gf-v2-standard standard_rl_v2_control_step64 "$STD_MODEL" "$STD_ROOT" & p0=$!
run_frozen_arm 1 1083 gf-v2-graph graph_frontier_rl_v2_step64 "$GRAPH_MODEL" "$GRAPH_ROOT" & p1=$!
set +e
wait "$p0"; rc0=$?
wait "$p1"; rc1=$?
set -e
printf '%s\n' "$rc0" > "$CONTROL/frozen_standard.exit_code"
printf '%s\n' "$rc1" > "$CONTROL/frozen_graph.exit_code"
[[ $rc0 -eq 0 && $rc1 -eq 0 ]] || fail FROZEN300_FAILED 40
"$RUN_PY" -c 'import json,sys; from pathlib import Path
for p in map(Path,sys.argv[1:]):
 s=json.loads((p/"run_summary.json").read_text()); assert s["completed"]==300 and s["valid"]==300 and not s["system_error_counts"],(p,s)
print("frozen300_valid_300_300")' "$STD_ROOT/frozen_300" "$GRAPH_ROOT/frozen_300" > "$CONTROL/frozen_validation.log"
set_status FROZEN300_DONE_BFCL_STARTING

run_bfcl_arm() (
  set -euo pipefail
  gpu=$1; port=$2; label=$3; model=$4; arm_root=$5
  output=$arm_root/bfcl
  mkdir -p "$output/logs"
  export CUDA_VISIBLE_DEVICES=$gpu
  export CUDA_HOME=$BFCL_ENV
  export PATH=$CUDA_HOME/bin:$PATH
  export CC=$BFCL_ENV/bin/x86_64-conda-linux-gnu-cc
  export CXX=$BFCL_ENV/bin/x86_64-conda-linux-gnu-c++
  export GCC=$BFCL_ENV/bin/x86_64-conda-linux-gnu-gcc
  export GXX=$BFCL_ENV/bin/x86_64-conda-linux-gnu-g++
  export NVCC_PREPEND_FLAGS="-ccbin=$CXX"
  export BFCL_PROJECT_ROOT=$output
  export VLLM_ENDPOINT=127.0.0.1 VLLM_PORT=$port
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  "$BFCL_PY" -m sglang.launch_server --host 127.0.0.1 --model-path "$model" --port "$port" \
    --dtype bfloat16 --tp-size 1 --mem-fraction-static 0.75 --trust-remote-code \
    --disable-cuda-graph --attention-backend triton --sampling-backend pytorch \
    --context-length 40960 --max-total-tokens 65536 > "$output/logs/server.log" 2>&1 &
  server_pid=$!
  cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
  trap cleanup EXIT INT TERM
  "$BFCL_PY" -c 'import sys,time,requests; u="http://127.0.0.1:"+sys.argv[1]+"/v1/models"; end=time.time()+300
while time.time()<end:
  try:
    if requests.get(u,timeout=2).status_code==200: raise SystemExit(0)
  except Exception: pass
  time.sleep(2)
raise SystemExit(1)' "$port"
  "$BFCL_BIN" generate --model "$BFCL_MODEL_ID" --backend sglang --skip-server-setup \
    --num-gpus 1 --gpu-memory-utilization 0.75 --temperature 0.7 --include-input-log \
    --local-model-path "$model" --test-category multi_turn > "$output/logs/generate.log" 2>&1
  "$BFCL_BIN" evaluate --model "$BFCL_MODEL_ID" --test-category multi_turn > "$output/logs/score.log" 2>&1
)

run_bfcl_arm 0 1084 standard_rl_v2_control_step64 "$STD_MODEL" "$STD_ROOT" & b0=$!
run_bfcl_arm 1 1085 graph_frontier_rl_v2_step64 "$GRAPH_MODEL" "$GRAPH_ROOT" & b1=$!
set +e
wait "$b0"; brc0=$?
wait "$b1"; brc1=$?
set -e
printf '%s\n' "$brc0" > "$CONTROL/bfcl_standard.exit_code"
printf '%s\n' "$brc1" > "$CONTROL/bfcl_graph.exit_code"
[[ $brc0 -eq 0 && $brc1 -eq 0 ]] || fail BFCL_FAILED 50
set_status BENCHMARKS_DONE