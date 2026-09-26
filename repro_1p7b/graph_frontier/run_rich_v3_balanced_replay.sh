#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
V3=${ROOT}/repro_1p7b/graph_frontier/preference_generation_v3
SOURCE=${V3}/pilot_balanced_16chunks_v1
OUTPUT=${V3}/balanced_effect_pilot_192_v1
STATUS=${OUTPUT}/replay_status
cd "${ROOT}"
test -f "${OUTPUT}/quick_effect_manifest.json"
test ! -e "${OUTPUT}/replay_summary.json"
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${ROOT}/configs/mcp_server.json
export OPENAI_API_KEY=placeholder
trap 'rc=$?; if [[ $rc -ne 0 ]]; then printf "REPLAY_FAILED:%s\n" "$rc" > "${STATUS}"; fi' EXIT
printf '%s\n' REPLAYING > "${STATUS}"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_balanced_effect replay \
  --source-run "${SOURCE}" --output-dir "${OUTPUT}" --seed 20260925 \
  > "${OUTPUT}/frontier_replay.log" 2>&1
printf '%s\n' REPLAY_DONE > "${STATUS}"
