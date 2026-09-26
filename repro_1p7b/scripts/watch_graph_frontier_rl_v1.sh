#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
VERL=/home/u2024311031/verl-agent
PY=/home/u2024311031/.conda/envs/verl-agent/bin/python
TRAIN_STATUS="$ROOT/repro_1p7b/logs/graph_frontier_rl_v1/status"
EVAL_LOG="$ROOT/repro_1p7b/logs/graph_frontier_rl_v1_eval"
STATUS="$EVAL_LOG/status"
STATE="$EVAL_LOG/state_history.log"
RUN_ROOT="$ROOT/repro_1p7b/checkpoints/graph_frontier_rl_v1_1p7b"
BASE="$ROOT/repro_1p7b/checkpoints/graph_frontier_dynamic_v1_8k_1p7b"
FINAL="$RUN_ROOT/hf_final"
MANIFEST="$ROOT/repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl"
EXPECTED_SHA=4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c

mkdir -p "$EVAL_LOG"
status() { printf '%s\n' "$1" > "$STATUS"; printf '%s %s\n' "$(date -Is)" "$1" >> "$STATE"; }
fail() { status "$1"; exit "${2:-1}"; }

status WAITING_FOR_RL
while true; do
  train_state=$(cat "$TRAIN_STATUS" 2>/dev/null || true)
  case "$train_state" in
    RL_DONE) break ;;
    RL_FAILED*) fail RL_FAILED 20 ;;
  esac
  sleep 30
done

status CHECKPOINT_EXPORTING
latest=$(find "$RUN_ROOT/verl_state" -maxdepth 1 -type d -name 'global_step_*' -print 2>/dev/null | sort -V | tail -1)
[[ -n "$latest" ]] || fail CHECKPOINT_INCOMPLETE 21
adapter="$latest/actor/lora_adapter"
[[ -s "$adapter/adapter_config.json" ]] || fail CHECKPOINT_INCOMPLETE 22
[[ ! -e "$FINAL" ]] || fail REFUSE_EXISTING_FINAL_MODEL 23

"$PY" - "$BASE" "$adapter" "$FINAL" > "$EVAL_LOG/checkpoint_export.log" 2>&1 <<'PY'
import sys
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, adapter, output = sys.argv[1:]
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(base).save_pretrained(output)
assert (Path(output) / "config.json").is_file()
assert list(Path(output).glob("*.safetensors"))
PY
[[ $? -eq 0 ]] || fail CHECKPOINT_INCOMPLETE 24

status GPU_TEARDOWN
/home/u2024311031/.conda/envs/verl-agent/bin/ray stop --force > "$EVAL_LOG/ray_stop.log" 2>&1 || true
for _ in $(seq 1 30); do
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] && break
  sleep 10
done
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]] || fail GPU_TEARDOWN_FAILED 25
actual_sha=$(sha256sum "$MANIFEST" | awk '{print $1}')
[[ "$actual_sha" == "$EXPECTED_SHA" ]] || fail PROBE_MANIFEST_MISMATCH 26

status EVALUATION_STARTING
export RUN_LABEL=graph_frontier_rl_v1
export MODEL_OVERRIDE="$FINAL"
export EVAL_SESSION=graph_frontier_rl_v1_eval
export EVAL_LOG_DIR="$EVAL_LOG"
cd "$ROOT" || fail EVAL_CHDIR_FAILED 27
bash repro_1p7b/scripts/run_dynamic_v2_bfcl_and_probe.sh worker >> "$EVAL_LOG/watcher.log" 2>&1
rc=$?
[[ $rc -eq 0 ]] || exit $rc
status DONE
