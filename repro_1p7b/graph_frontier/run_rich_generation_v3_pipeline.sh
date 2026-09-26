#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
PY=/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python
GF=${ROOT}/repro_1p7b/graph_frontier
V3=${GF}/preference_generation_v3
SMOKE_RUN=${RICH_V3_SMOKE_RUN:-${V3}/smoke_120_unambiguous_v2_seed20260924}
SMOKE_PLAN=${RICH_V3_SMOKE_PLAN:-${V3}/manifests/smoke_plan_unambiguous_v2.jsonl}
FORMAL_RUN=${RICH_V3_FORMAL_RUN:-${V3}/formal_1219_unambiguous_v2_seed20260924}
FORMAL_PLAN=${RICH_V3_FORMAL_PLAN:-${V3}/manifests/formal_plan_unambiguous_v2_scaled_deep.jsonl}
FORMAL_PREFLIGHT=${RICH_V3_FORMAL_PREFLIGHT:-${V3}/audit/formal_plan_unambiguous_v2_scaled_deep_preflight.json}
STATUS=${RICH_V3_PIPELINE_STATUS:-${V3}/pipeline_unambiguous_v2_status}

if [[ -e "${STATUS}" ]]; then
  echo "refusing to overwrite pipeline status: ${STATUS}" >&2
  exit 2
fi
if ! "${PY}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["plans"] == d["passed"] and d["failed"] == 0' "${FORMAL_PREFLIGHT}"; then
  printf '%s\n' FORMAL_PREFLIGHT_FAILED > "${STATUS}"
  exit 2
fi

read_status() {
  cat "$1/status" 2>/dev/null || printf '%s\n' MISSING
}

verdict() {
  "${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$1"
}

printf '%s\n' WAITING_FOR_SMOKE_GENERATION > "${STATUS}"
while true; do
  current=$(read_status "${SMOKE_RUN}")
  if [[ "${current}" == GENERATED ]]; then
    break
  fi
  if [[ "${current}" != GENERATING && "${current}" != STARTING_SERVERS ]]; then
    printf 'SMOKE_GENERATION_FAILED:%s\n' "${current}" > "${STATUS}"
    exit 3
  fi
  sleep 30
done

printf '%s\n' AUDITING_SMOKE > "${STATUS}"
bash "${GF}/run_rich_generation_v3_audit.sh" "${SMOKE_RUN}" "${SMOKE_PLAN}" smoke
smoke_verdict=$(verdict "${SMOKE_RUN}/audit/validation_report.json")
if [[ "${smoke_verdict}" != SMOKE_READY ]]; then
  printf 'SMOKE_GATE_FAILED:%s\n' "${smoke_verdict}" > "${STATUS}"
  exit 4
fi

printf '%s\n' FORMAL_GENERATING > "${STATUS}"
RICH_V3_MAX_FAILURES=100000 RICH_V3_ATTEMPTS=4 RICH_V3_CONCURRENCY=4 \
  bash "${GF}/run_rich_generation_v3_smoke.sh" "${FORMAL_RUN}" "${FORMAL_PLAN}"
formal_generation=$(read_status "${FORMAL_RUN}")
if [[ "${formal_generation}" != GENERATED ]]; then
  printf 'FORMAL_GENERATION_FAILED:%s\n' "${formal_generation}" > "${STATUS}"
  exit 5
fi

printf '%s\n' AUDITING_FORMAL > "${STATUS}"
bash "${GF}/run_rich_generation_v3_audit.sh" "${FORMAL_RUN}" "${FORMAL_PLAN}" formal
formal_verdict=$(verdict "${FORMAL_RUN}/audit/validation_report.json")
printf '%s\n' "${formal_verdict}" > "${STATUS}"
