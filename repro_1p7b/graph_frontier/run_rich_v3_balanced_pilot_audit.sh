#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_repro_1p7b
PILOT=${ROOT}/repro_1p7b/graph_frontier/preference_generation_v3/pilot_balanced_16chunks_v1
PLAN=${PILOT}/pilot_plan.jsonl
STATUS=${PILOT}/audit_status
test -f "${PILOT}/snapshot_manifest.json"
test -f "${PLAN}"
test ! -e "${STATUS}"
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${ROOT}/configs/mcp_server.json
export OPENAI_API_KEY=placeholder
cd "${ROOT}"
printf '%s\n' STATIC_AUDIT > "${STATUS}"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_generation_v3_validate prepare \
  --run-dir "${PILOT}" --plan "${PLAN}" > "${PILOT}/audit_prepare.log" 2>&1
printf '%s\n' GOLD_REPLAY > "${STATUS}"
"${ENV}/bin/python" repro_1p7b/graph_frontier/replay_generated_guided_rl.py \
  --source-dir "${PILOT}/converted" --audit-ledger "${PILOT}/audit/static_ledger.jsonl" \
  --output-dir "${PILOT}/replay" > "${PILOT}/audit_replay.log" 2>&1
printf '%s\n' FINALIZING > "${STATUS}"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_generation_v3_validate finalize \
  --run-dir "${PILOT}" --replay-dir "${PILOT}/replay" --mode formal \
  --plan "${PLAN}" --plan-summary "${PILOT}/pilot_plan.summary.json" \
  --audit-json "${PILOT}/graph_frontier_rich_generation_audit.json" \
  --audit-md "${PILOT}/graph_frontier_rich_generation_audit.md" \
  > "${PILOT}/audit_finalize.log" 2>&1
printf '%s\n' AUDIT_DONE > "${STATUS}"
