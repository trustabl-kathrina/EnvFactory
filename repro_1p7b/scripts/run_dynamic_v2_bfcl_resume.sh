#!/usr/bin/env bash
set -uo pipefail

MODE=${1:-launch}
[[ "$MODE" == "launch" || "$MODE" == "worker" ]] || {
  echo "usage: $0 [launch|worker]" >&2
  exit 2
}

WORKTREE=/home/u2024311031/workspace/envfactory_repro_1p7b
EVAL_DIR="$WORKTREE/repro_1p7b/evaluation/bfcl"
MODEL="$WORKTREE/repro_1p7b/checkpoints/graph_frontier_dynamic_v2_8k_1p7b"
BFCL_ENV=/home/u2024311031/.conda/envs/envfactory_bfcl_v1p3
RUN_PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
BFCL_PY="$BFCL_ENV/bin/python"
BFCL_BIN="$BFCL_ENV/bin/bfcl"
BFCL_MODEL_ID=Qwen/Qwen3-1.7B-FC
MODEL_DIR=Qwen_Qwen3-1.7B-FC
PRIOR="$EVAL_DIR/artifacts/full/dynamic_v1/result/$MODEL_DIR"
INTERRUPTED="$EVAL_DIR/artifacts/full/dynamic_v2_shards"
RESUME_ROOT="$EVAL_DIR/artifacts/full/dynamic_v2_resume_shards"
FINAL_ROOT="$EVAL_DIR/artifacts/full/dynamic_v2"
LOG_DIR="$WORKTREE/repro_1p7b/logs/graph_frontier_dynamic_v2_eval"
BFCL_LOG_DIR="$EVAL_DIR/logs/full"
PLAN="$LOG_DIR/bfcl_resume_slice_plan.json"
PROBE_ROOT="$WORKTREE/repro_1p7b/results/graph_frontier/confirm_300"
PROBE_OUTPUT="$PROBE_ROOT/dynamic_v2"
MANIFEST="$PROBE_ROOT/frozen/confirm_300_seed_20260914.jsonl"
REPORT_DIR="$WORKTREE/repro_1p7b/graph_frontier/reports"
SESSION=graph_frontier_dynamic_v2_eval_resume
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
  [[ -z "$(gpu_pids)" ]] || fail RESUME_GPU_BUSY 70
  [[ ! -e "$RESUME_ROOT" ]] || fail REFUSE_EXISTING_RESUME_SHARDS 71
  [[ ! -e "$FINAL_ROOT" ]] || fail REFUSE_EXISTING_BFCL_RESULTS 72
  [[ ! -e "$PROBE_OUTPUT" ]] || fail REFUSE_EXISTING_PROBE_RESULTS 73
  tmux has-session -t "$SESSION" 2>/dev/null && fail REFUSE_EXISTING_RESUME_TMUX 74
  set_status BFCL_RESUME_LAUNCHING
  tmux new-session -d -s "$SESSION" "$0 worker"
  exit 0
fi

source "$EVAL_DIR/scripts/common.sh"
require_clean_bfcl || fail BFCL_ENVIRONMENT_DIRTY 75
require_model "$MODEL" || fail MODEL_INCOMPLETE 76
[[ -z "$(gpu_pids)" ]] || fail RESUME_GPU_BUSY 77
for port in 1074 1075 1076 1077; do
  port_free "$port" || fail PORT_BUSY_$port 78
done
[[ ! -e "$RESUME_ROOT" ]] || fail REFUSE_EXISTING_RESUME_SHARDS 79
[[ ! -e "$FINAL_ROOT" ]] || fail REFUSE_EXISTING_BFCL_RESULTS 80
[[ ! -e "$PROBE_OUTPUT" ]] || fail REFUSE_EXISTING_PROBE_RESULTS 81

BASE_RESULT="$INTERRUPTED/gpu1/result/$MODEL_DIR"
LONG_RESULT="$INTERRUPTED/gpu0/result/$MODEL_DIR"
"$RUN_PY" - "$PRIOR" "$BASE_RESULT" "$LONG_RESULT" "$RESUME_ROOT" "$PLAN" <<'PY' \
  > "$LOG_DIR/bfcl_resume_slice_build.log" 2>&1 || fail BFCL_RESUME_SLICE_BUILD_FAILED 82
import json
import sys
from collections import defaultdict
from pathlib import Path

prior, base_result, long_result, output_root, plan_path = map(Path, sys.argv[1:])
files = {
    "multi_turn_base": "BFCL_v3_multi_turn_base_result.json",
    "multi_turn_long_context": "BFCL_v3_multi_turn_long_context_result.json",
    "multi_turn_miss_func": "BFCL_v3_multi_turn_miss_func_result.json",
    "multi_turn_miss_param": "BFCL_v3_multi_turn_miss_param_result.json",
}
category_seconds = {
    "multi_turn_base": 4708.0,
    "multi_turn_long_context": 13613.0,
    "multi_turn_miss_func": 5908.0,
    "multi_turn_miss_param": 398.0,
}

def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

completed = defaultdict(set)
seen = set()
for result_dir in (base_result, long_result):
    for category, name in files.items():
        for row in rows(result_dir / name):
            task_id = row["id"]
            if task_id in seen:
                raise RuntimeError(f"duplicate completed task: {task_id}")
            seen.add(task_id)
            completed[category].add(task_id)

expected_completed = {
    "multi_turn_base": 200,
    "multi_turn_long_context": 200,
    "multi_turn_miss_func": 35,
    "multi_turn_miss_param": 0,
}
actual_completed = {key: len(completed[key]) for key in files}
if actual_completed != expected_completed:
    raise RuntimeError(
        f"interrupted snapshot changed: expected {expected_completed}, got {actual_completed}"
    )

payloads = [defaultdict(list), defaultdict(list)]
remaining_counts = {}
all_prior = set()
for category, name in files.items():
    prior_rows = rows(prior / name)
    ids = sorted(
        (row["id"] for row in prior_rows),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if len(ids) != 200 or len(set(ids)) != 200:
        raise RuntimeError(f"{category}: prior source is not 200 unique IDs")
    all_prior.update(ids)
    unknown = completed[category] - set(ids)
    if unknown:
        raise RuntimeError(f"{category}: unknown completed IDs: {sorted(unknown)[:5]}")
    remaining = [task_id for task_id in ids if task_id not in completed[category]]
    split = (len(remaining) + 1) // 2
    payloads[0][category].extend(remaining[:split])
    payloads[1][category].extend(remaining[split:])
    remaining_counts[category] = len(remaining)

assigned = set()
shards = []
for gpu, payload in enumerate(payloads):
    clean = {key: value for key, value in payload.items() if value}
    root = output_root / f"gpu{gpu}"
    root.mkdir(parents=True, exist_ok=False)
    ids_path = root / "test_case_ids_to_generate.json"
    ids_path.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")
    flat = [task_id for values in clean.values() for task_id in values]
    if assigned.intersection(flat):
        raise RuntimeError("duplicate ID across resume shards")
    assigned.update(flat)
    estimate = sum(
        category_seconds[key] * len(value) / 200 for key, value in clean.items()
    )
    shards.append({
        "gpu": gpu,
        "total_cases": len(flat),
        "category_counts": {key: len(value) for key, value in clean.items()},
        "estimated_wall_seconds": estimate,
        "estimated_hours": estimate / 3600,
        "ids_path": str(ids_path),
    })

if len(all_prior) != 800 or len(seen) != 435 or len(assigned) != 365:
    raise RuntimeError(
        f"partition mismatch prior={len(all_prior)} completed={len(seen)} remaining={len(assigned)}"
    )
if assigned != all_prior - seen:
    raise RuntimeError("resume shards do not exactly cover unfinished IDs")
plan = {
    "schema_version": "bfcl_dynamic_v2_coarse_resume_v1",
    "method": "sorted IDs split into contiguous halves within each remaining category",
    "completed_count": len(seen),
    "completed_counts": actual_completed,
    "remaining_count": len(assigned),
    "remaining_counts": remaining_counts,
    "shards": shards,
    "expected_makespan_seconds": max(item["estimated_wall_seconds"] for item in shards),
    "expected_makespan_hours": max(item["estimated_hours"] for item in shards),
}
plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
print(json.dumps(plan, indent=2))
PY

run_bfcl_shard() (
  set -uo pipefail
  gpu="$1"; port="$2"; root="$3"; prefix="$4"
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
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
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
raise SystemExit(1)' "$port" || exit 83
  "$BFCL_BIN" generate --model "$BFCL_MODEL_ID" --backend sglang \
    --skip-server-setup --num-gpus 1 --gpu-memory-utilization 0.75 \
    --temperature 0.7 --include-input-log --allow-overwrite \
    --local-model-path "$MODEL" --run-ids \
    > "$BFCL_LOG_DIR/${prefix}_generate.log" 2>&1
)

set_status BFCL_RESUME_GENERATING
run_bfcl_shard 0 1074 "$RESUME_ROOT/gpu0" dynamic_v2_resume_gpu0 &
p0=$!
run_bfcl_shard 1 1075 "$RESUME_ROOT/gpu1" dynamic_v2_resume_gpu1 &
p1=$!
set +e
wait "$p0"; rc0=$?
wait "$p1"; rc1=$?
set -e
printf '%s\n' "$rc0" > "$LOG_DIR/bfcl_resume_gpu0.exit_code"
printf '%s\n' "$rc1" > "$LOG_DIR/bfcl_resume_gpu1.exit_code"
[[ "$rc0" -eq 0 && "$rc1" -eq 0 ]] || fail BFCL_RESUME_GENERATION_FAILED 84

set_status BFCL_MERGING
"$RUN_PY" - "$BASE_RESULT" "$LONG_RESULT" \
  "$RESUME_ROOT/gpu0/result/$MODEL_DIR" "$RESUME_ROOT/gpu1/result/$MODEL_DIR" \
  "$FINAL_ROOT" "$PLAN" <<'PY' > "$LOG_DIR/bfcl_resume_merge.log" 2>&1 \
  || fail BFCL_MERGE_FAILED 85
import json
import shutil
import sys
from pathlib import Path

base, long_result, resume0, resume1, output_root, plan = map(Path, sys.argv[1:])
files = (
    "BFCL_v3_multi_turn_base_result.json",
    "BFCL_v3_multi_turn_long_context_result.json",
    "BFCL_v3_multi_turn_miss_func_result.json",
    "BFCL_v3_multi_turn_miss_param_result.json",
)
def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if output_root.exists():
    raise RuntimeError(f"refusing existing output: {output_root}")
output = output_root / "result" / "Qwen_Qwen3-1.7B-FC"
output.mkdir(parents=True)
all_ids = set()
summary = {}
for name in files:
    merged = []
    for result_dir in (base, long_result, resume0, resume1):
        merged.extend(rows(result_dir / name))
    ids = [row["id"] for row in merged]
    unique = set(ids)
    if len(merged) != 200 or len(unique) != 200:
        raise RuntimeError(f"{name}: expected 200 unique rows, got {len(merged)}/{len(unique)}")
    if all_ids.intersection(unique):
        raise RuntimeError(f"cross-category duplicate in {name}")
    all_ids.update(unique)
    merged.sort(key=lambda row: int(row["id"].rsplit("_", 1)[1]))
    (output / name).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged)
        )
    summary[name] = {"count": len(merged), "unique": len(unique)}
if len(all_ids) != 800:
    raise RuntimeError(f"expected 800 unique IDs, got {len(all_ids)}")
shutil.copy2(plan, output_root / "resume_slice_plan.json")
audit = {"status": "PASS", "count": 800, "unique_ids": 800, "files": summary}
(output_root / "merge_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(audit, indent=2))
PY

set_status BFCL_SCORING
export BFCL_PROJECT_ROOT="$FINAL_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$BFCL_BIN" evaluate --model "$BFCL_MODEL_ID" --test-category multi_turn \
  > "$BFCL_LOG_DIR/dynamic_v2_score.log" 2>&1 || fail BFCL_SCORING_FAILED 86
set_status BFCL_DONE

run_probe_shard() (
  set -uo pipefail
  gpu="$1"; port="$2"; shard="$3"; key="dynamic-v2-probe-$shard"
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
raise SystemExit(1)' "$port" "$key" || exit 87
  export LITELLM_LOCAL_MODEL_COST_MAP=True
  export SGLANG_BASE_URL="http://127.0.0.1:$port/v1"
  export SGLANG_API_KEY="$key"
  export SGLANG_MODEL="$MODEL"
  "$RUN_PY" -m repro_1p7b.graph_frontier.confirm_300 run \
    --manifest "$MANIFEST" --model dynamic_v2 --output-dir "$PROBE_OUTPUT" \
    --shard-count 2 --shard-index "$shard" \
    > "$PROBE_OUTPUT/eval${shard}.log" 2>&1
)

set_status PROBE_RUNNING
mkdir -p "$PROBE_OUTPUT"
run_probe_shard 0 1076 0 &
q0=$!
run_probe_shard 1 1077 1 &
q1=$!
set +e
wait "$q0"; qr0=$?
wait "$q1"; qr1=$?
set -e
printf '%s\n' "$qr0" > "$LOG_DIR/probe_gpu0.exit_code"
printf '%s\n' "$qr1" > "$LOG_DIR/probe_gpu1.exit_code"
[[ "$qr0" -eq 0 && "$qr1" -eq 0 ]] || fail PROBE_FAILED 88

"$RUN_PY" - <<'PY'
import json
from collections import Counter
from pathlib import Path
root=Path("repro_1p7b/results/graph_frontier/confirm_300/dynamic_v2")
rows=[json.loads((root/f"run_summary.shard{i}.json").read_text()) for i in range(2)]
summary={
    "model":"dynamic_v2",
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
"$RUN_PY" -m repro_1p7b.graph_frontier.confirm_analyze_dynamic_v2 \
  --root "$PROBE_ROOT" --bfcl-root "$EVAL_DIR/artifacts/full" \
  --reports "$REPORT_DIR" > "$LOG_DIR/probe_analysis.log" 2>&1 \
  || fail PROBE_ANALYSIS_FAILED 89
set_status DONE
