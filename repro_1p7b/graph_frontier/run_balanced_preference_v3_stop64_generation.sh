#!/usr/bin/env bash
# Isolated two-GPU, resumable stop-yield experiment (64-plan default).
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
GF=$ROOT/repro_1p7b/graph_frontier
V3=$GF/preference_generation_v3
SERVER_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
QUERY_ENV=/home/u2024311031/.conda/envs/envfactory_repro_1p7b
MODEL=/home/u2024311031/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
GRAPH=$ROOT/repro_1p7b/data/graph_frontier_rl_v1_generated/tool_graph.pkl
PLAN=${STOP_TARGET_PLAN:-$V3/manifests/stop_target_depth23_64_seed20260927_v2.jsonl}
PREFLIGHT=${STOP_TARGET_PREFLIGHT:-$V3/audit/stop_target_depth23_64_seed20260927_v2.json}
OUT=${STOP_TARGET_OUTPUT:-$V3/stop_target_depth23_64_seed20260927_v2_resumable}
STATUS=${STOP_TARGET_STATUS:-$GF/balanced_preference_v3/stop64_generation_status}
COUNT=${STOP_TARGET_COUNT:-64}
LABEL=${STOP_TARGET_LABEL:-STOP64}
export STOP_TARGET_COUNT=$COUNT
cd "$ROOT"
export OPENAI_API_KEY=placeholder
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
"$QUERY_ENV/bin/python" - "$PREFLIGHT" <<'PY'
import json,os,sys
report=json.load(open(sys.argv[1]))
assert report["plans"]==report["passed"]==int(os.environ["STOP_TARGET_COUNT"]) and report["failed"]==0
PY
if [[ ! -e "$OUT" ]]; then
  "$QUERY_ENV/bin/python" - "$PLAN" "$GRAPH" "$OUT" <<'PY'
import json,os,sys
from pathlib import Path
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256
from repro_1p7b.graph_frontier.rich_generation_v3_resumable import SCHEMA, atomic_json, plan_maps
plan,graph,out=map(Path,sys.argv[1:])
plans,_,_=plan_maps(plan)
assert len(plans)==int(os.environ["STOP_TARGET_COUNT"]) and not out.exists()
out.mkdir(parents=True)
atomic_json(out/"resume_manifest.json", {
 "schema_version":SCHEMA,"source_run":None,"source_status":"EMPTY",
 "source_pairs":0,"source_artifacts":{},"plan":str(plan.resolve()),
 "plan_sha256":file_sha256(plan),"graph":str(graph.resolve()),
 "graph_sha256":file_sha256(graph),"total_plans":int(os.environ["STOP_TARGET_COUNT"]),"chunk_size":32})
(out/"status").write_text("BOOTSTRAPPED\n")
PY
fi
"$QUERY_ENV/bin/python" - "$OUT/resume_manifest.json" "$PLAN" "$GRAPH" <<'PY'
import json,os,sys
from pathlib import Path
from repro_1p7b.graph_frontier.rich_generation_v3 import file_sha256
m=json.loads(Path(sys.argv[1]).read_text())
assert m["plan_sha256"]==file_sha256(sys.argv[2])
assert m["graph_sha256"]==file_sha256(sys.argv[3])
assert m["total_plans"]==int(os.environ["STOP_TARGET_COUNT"]) and m["chunk_size"]==32
PY
[[ ! -e "$OUT/generation_run.json" ]] || { echo 'stop64 generation already complete' >&2; exit 2; }
for port in 8010 8011; do
  if env -u LD_LIBRARY_PATH /usr/bin/curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "port $port already occupied" >&2; exit 2
  fi
done
mkdir -p "$OUT/launcher_attempts"
ATTEMPT=$(mktemp -d "$OUT/launcher_attempts/stop64-XXXXXXXX")
export CUDA_HOME=$SERVER_ENV
export PATH=$SERVER_ENV/bin:$QUERY_ENV/bin:$PATH
export MCP_CONFIG_PATH=$ROOT/configs/mcp_server.json
export LD_LIBRARY_PATH=$SERVER_ENV/lib:$SERVER_ENV/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$SERVER_ENV/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pid0= pid1=
cleanup() {
  [[ -z "$pid0" ]] || { kill "$pid0" 2>/dev/null || true; wait "$pid0" 2>/dev/null || true; }
  [[ -z "$pid1" ]] || { kill "$pid1" 2>/dev/null || true; wait "$pid1" 2>/dev/null || true; }
}
on_exit() {
  rc=$?
  trap - EXIT
  if [[ "$rc" -ne 0 ]]; then printf 'FAILED:%s\n' "$rc" > "$STATUS"; fi
  cleanup
  exit "$rc"
}
trap on_exit EXIT
wait_ready() {
  local port=$1 pid=$2
  for _ in $(seq 1 600); do
    env -u LD_LIBRARY_PATH /usr/bin/curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
COMMON=(--model-path "$MODEL" --host 127.0.0.1
  --served-model-name Qwen2.5-14B-Instruct --context-length 20000
  --mem-fraction-static 0.85 --attention-backend triton
  --sampling-backend pytorch --disable-cuda-graph --api-key placeholder)
printf '%s\n' STARTING_SERVERS > "$STATUS"
CUDA_VISIBLE_DEVICES=0 "$SERVER_ENV/bin/python" -m sglang.launch_server "${COMMON[@]}" \
  --port 8010 --random-seed 20260922 > "$ATTEMPT/server_gpu0.log" 2>&1 &
pid0=$!
wait_ready 8010 "$pid0"
CUDA_VISIBLE_DEVICES=1 "$SERVER_ENV/bin/python" -m sglang.launch_server "${COMMON[@]}" \
  --port 8011 --random-seed 20260923 > "$ATTEMPT/server_gpu1.log" 2>&1 &
pid1=$!
wait_ready 8011 "$pid1"
export SGLANG_BASE_URL=http://127.0.0.1:8010/v1 SGLANG_API_KEY=placeholder SGLANG_MODEL=Qwen2.5-14B-Instruct
export SGLANG_BASE_URL1=http://127.0.0.1:8011/v1 SGLANG_API_KEY1=placeholder SGLANG_MODEL1=Qwen2.5-14B-Instruct
export CHAT_URL=http://127.0.0.1:8010/v1 CHAT_API_KEY=placeholder CHAT_MODEL=Qwen2.5-14B-Instruct
printf 'GENERATING_%s\n' "$LABEL" > "$STATUS"
"$QUERY_ENV/bin/python" -m repro_1p7b.graph_frontier.rich_generation_v3_resumable run \
  --plan "$PLAN" --graph "$GRAPH" --output-dir "$OUT" --chunk-size 32 \
  --model-name sglang,sglang1 --pass-k 2 --concurrency 4 --attempts 4 \
  > "$ATTEMPT/generation.log" 2>&1
"$QUERY_ENV/bin/python" - "$OUT/generation_run.json" <<'PY'
import json,os,sys
r=json.load(open(sys.argv[1]))
assert r["requested_plans"]==int(os.environ["STOP_TARGET_COUNT"])
assert r["raw_chain_files"]==r["sidecar_files"]==r["completed_chains"]
PY
printf 'GENERATED_%s\n' "$LABEL" > "$STATUS"
