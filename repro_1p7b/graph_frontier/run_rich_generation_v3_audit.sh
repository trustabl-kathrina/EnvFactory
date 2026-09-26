#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_repro_1p7b
REPLAY_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
RUN_DIR=${1:?usage: run_rich_generation_v3_audit.sh RUN_DIR PLAN MODE}
PLAN=${2:?usage: run_rich_generation_v3_audit.sh RUN_DIR PLAN MODE}
MODE=${3:?usage: run_rich_generation_v3_audit.sh RUN_DIR PLAN MODE}
PLAN_SUMMARY=${PLAN%.jsonl}.summary.json
STATUS=${RUN_DIR}/audit_status
REPLAY_DIR=${RUN_DIR}/replay

if [[ "${MODE}" != smoke && "${MODE}" != formal ]]; then
  echo "MODE must be smoke or formal" >&2
  exit 2
fi
if [[ -e "${STATUS}" ]]; then
  echo "refusing to overwrite existing audit_status: ${STATUS}" >&2
  exit 2
fi
if [[ ! -f "${PLAN_SUMMARY}" ]]; then
  echo "missing plan summary: ${PLAN_SUMMARY}" >&2
  exit 2
fi

export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${ROOT}/configs/mcp_server.json
export OPENAI_API_KEY=placeholder
cd "${ROOT}"

printf '%s\n' STATIC_AUDIT > "${STATUS}"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_generation_v3_validate prepare \
  --run-dir "${RUN_DIR}" --plan "${PLAN}" \
  > "${RUN_DIR}/audit_prepare.log" 2>&1

printf '%s\n' GOLD_REPLAY > "${STATUS}"
"${REPLAY_ENV}/bin/python" repro_1p7b/graph_frontier/replay_generated_guided_rl.py \
  --source-dir "${RUN_DIR}/converted" \
  --audit-ledger "${RUN_DIR}/audit/static_ledger.jsonl" \
  --output-dir "${REPLAY_DIR}" \
  > "${RUN_DIR}/audit_replay.log" 2>&1

printf '%s\n' FINALIZING > "${STATUS}"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_generation_v3_validate finalize \
  --run-dir "${RUN_DIR}" --replay-dir "${REPLAY_DIR}" --mode "${MODE}" \
  --plan "${PLAN}" --plan-summary "${PLAN_SUMMARY}" \
  --audit-json "${RUN_DIR}/graph_frontier_rich_generation_audit.json" \
  --audit-md "${RUN_DIR}/graph_frontier_rich_generation_audit.md" \
  > "${RUN_DIR}/audit_finalize.log" 2>&1

printf '%s\n' AUDIT_DONE > "${STATUS}"
