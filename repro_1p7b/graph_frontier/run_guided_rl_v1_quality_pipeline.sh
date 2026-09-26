#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
SOURCE=${1:?usage: run_guided_rl_v1_quality_pipeline.sh SOURCE_DIR FROZEN_DIR [SEED]}
FROZEN=${2:?usage: run_guided_rl_v1_quality_pipeline.sh SOURCE_DIR FROZEN_DIR [SEED]}
PILOT_SIZE=${GUIDED_RL_PILOT_SIZE:-128}
SEED=${3:-20260920}
AUDIT=${SOURCE}/audit_final
REPLAY=${SOURCE}/replay_final
MCP_CONFIG=${ROOT}/configs/mcp_server.json

cd "${ROOT}"
export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${MCP_CONFIG}

if [[ ! -x "${PY}" ]]; then
  echo "missing query environment python: ${PY}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE}/status" ]] || [[ "$(cat "${SOURCE}/status")" != GENERATED ]]; then
  echo "generation is not complete: ${SOURCE}/status" >&2
  exit 2
fi
if [[ -e "${AUDIT}" || -e "${REPLAY}" || -e "${FROZEN}" ]]; then
  echo "refusing to overwrite audit, replay, or frozen output" >&2
  exit 2
fi

"${PY}" repro_1p7b/graph_frontier/audit_generated_guided_rl.py \
  --source-dir "${SOURCE}" \
  --output-dir "${AUDIT}" \
  --registry "${MCP_CONFIG}" \
  > "${SOURCE}/audit_final.log" 2>&1

if [[ "$(cat "${AUDIT}/status")" != STATIC_PASS ]]; then
  echo "static quality gate failed" >&2
  exit 3
fi

"${PY}" repro_1p7b/graph_frontier/guided_rl_generate.py convert-audit \
  --output-dir "${SOURCE}" \
  --audit-ledger "${AUDIT}/rl_data_v1_audit.jsonl" \
  --seed "${SEED}" \
  --train-ratio 0.90 \
  > "${SOURCE}/conversion_final.log" 2>&1

"${PY}" repro_1p7b/graph_frontier/replay_generated_guided_rl.py \
  --source-dir "${SOURCE}" \
  --audit-ledger "${AUDIT}/rl_data_v1_audit.jsonl" \
  --output-dir "${REPLAY}" \
  --registry "${MCP_CONFIG}" \
  > "${SOURCE}/replay_final.log" 2>&1

if [[ "$(cat "${REPLAY}/status")" != REPLAY_COMPLETE ]]; then
  echo "executable replay did not complete" >&2
  exit 4
fi

"${PY}" repro_1p7b/graph_frontier/freeze_generated_guided_rl.py \
  --source-dir "${SOURCE}" \
  --static-ledger "${AUDIT}/rl_data_v1_audit.jsonl" \
  --static-report "${AUDIT}/quality_report.json" \
  --replay-dir "${REPLAY}" \
  --output-dir "${FROZEN}" \
  --seed "${SEED}" \
  --pilot-size "${PILOT_SIZE}" \
  --smoke-size 8 \
  > "${SOURCE}/freeze_final.log" 2>&1

"${PY}" repro_1p7b/graph_frontier/prepare_generated_guided_rl.py \
  --source-dir "${FROZEN}" \
  --output-dir "${FROZEN}/parquet_pilot128" \
  --train-json pilot_128.json \
  --val-json val.json \
  > "${SOURCE}/prepare_pilot128.log" 2>&1

"${PY}" repro_1p7b/graph_frontier/prepare_generated_guided_rl.py \
  --source-dir "${FROZEN}" \
  --output-dir "${FROZEN}/parquet_smoke8" \
  --train-json smoke_8.json \
  --val-json val.json \
  > "${SOURCE}/prepare_smoke8.log" 2>&1

test "$(cat "${FROZEN}/status")" = RL_DATASET_V1_FROZEN
echo "RL_DATASET_V1_FROZEN"
