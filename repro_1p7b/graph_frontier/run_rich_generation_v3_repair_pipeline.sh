#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
GF=${ROOT}/repro_1p7b/graph_frontier
V3=${GF}/preference_generation_v3
ORIGINAL_RUN=${V3}/smoke_120_unambiguous_v2_seed20260924
SUPPLEMENTAL_RUN=${V3}/smoke_depth3_supplement_v3_33_seed20260925
SUPPLEMENTAL_PLAN=${V3}/manifests/smoke_depth3_supplement_v3_33.jsonl
COMBINED_RUN=${V3}/smoke_120_unambiguous_v3_repaired_seed20260925
COMBINED_PLAN=${V3}/manifests/smoke_plan_unambiguous_v3_repaired_40x3.jsonl
PIPELINE_STATUS=${V3}/pipeline_unambiguous_v3_repaired_status
STATUS=${V3}/repair_pipeline_v3_status

if [[ -e "${STATUS}" || -e "${PIPELINE_STATUS}" || -e "${COMBINED_RUN}" ]]; then
  echo "refusing to overwrite rich-v3 repair pipeline state" >&2
  exit 2
fi

on_exit() {
  local rc=$?
  local current
  current=$(cat "${STATUS}" 2>/dev/null || true)
  if [[ ${rc} -ne 0 && "${current}" != *FAILED* ]]; then
    printf 'SCRIPT_FAILED:%s:line=%s\n' "${rc}" "${BASH_LINENO[0]:-unknown}" > "${STATUS}"
  fi
}
trap on_exit EXIT

printf '%s\n' WAITING_FOR_SUPPLEMENTAL_GENERATION > "${STATUS}"
while true; do
  current=$(cat "${SUPPLEMENTAL_RUN}/status" 2>/dev/null || printf '%s\n' MISSING)
  if [[ "${current}" == GENERATED ]]; then
    break
  fi
  if [[ "${current}" != GENERATING && "${current}" != STARTING_SERVERS ]]; then
    printf 'SUPPLEMENTAL_GENERATION_FAILED:%s\n' "${current}" > "${STATUS}"
    exit 3
  fi
  sleep 30
done

printf '%s\n' AUDITING_SUPPLEMENTAL > "${STATUS}"
bash "${GF}/run_rich_generation_v3_audit.sh" \
  "${SUPPLEMENTAL_RUN}" "${SUPPLEMENTAL_PLAN}" smoke

if ! "${PY}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["funnel"]["final_valid_tasks"] >= 3; assert r["hard_gates"]["retained_replay_rate_100pct"]; assert r["hard_gates"]["internal_value_leakage_zero"]; assert r["frozen300"]["exact_overlap"] == 0' \
  "${SUPPLEMENTAL_RUN}/audit/validation_report.json"; then
  printf '%s\n' SUPPLEMENTAL_QUALITY_GATE_FAILED > "${STATUS}"
  exit 4
fi

printf '%s\n' ASSEMBLING_SMOKE > "${STATUS}"
"${PY}" -m repro_1p7b.graph_frontier.rich_generation_v3_repair_smoke assemble \
  --combined-plan "${COMBINED_PLAN}" \
  --original-run "${ORIGINAL_RUN}" \
  --supplemental-run "${SUPPLEMENTAL_RUN}" \
  --output-dir "${COMBINED_RUN}"

printf '%s\n' RUNNING_40X3_PIPELINE > "${STATUS}"
RICH_V3_SMOKE_RUN=${COMBINED_RUN} \
RICH_V3_SMOKE_PLAN=${COMBINED_PLAN} \
RICH_V3_PIPELINE_STATUS=${PIPELINE_STATUS} \
  bash "${GF}/run_rich_generation_v3_pipeline.sh"
cat "${PIPELINE_STATUS}" > "${STATUS}"
