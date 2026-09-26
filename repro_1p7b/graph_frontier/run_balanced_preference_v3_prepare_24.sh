#!/usr/bin/env bash
# Snapshot a bounded Rich-v3 milestone and prepare replay-verified v3 states.
set -euo pipefail
ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
ENV=/home/u2024311031/.conda/envs/envfactory_repro_1p7b
REPLAY_ENV=/home/u2024311031/.conda/envs/envfactory_sglang_1p7b
GF=${ROOT}/repro_1p7b/graph_frontier
V3=${GF}/preference_generation_v3
SOURCE=${V3}/formal_1219_unambiguous_v2_resumable
PLAN=${V3}/manifests/formal_plan_unambiguous_v2_scaled_deep.jsonl
SNAP=${V3}/pilot_balanced_24chunks_v3
MILESTONE=${BALANCED_V3_MILESTONE:-milestone_24}
OUT=${GF}/balanced_preference_v3/${MILESTONE}
export BALANCED_V3_MILESTONE=${MILESTONE}
export PYTHONPATH=${ROOT}:/home/u2024311031/verl-agent:${PYTHONPATH:-}
export MCP_CONFIG_PATH=${ROOT}/configs/mcp_server.json
export ENVFACTORY_ROOT=${ROOT}
export OPENAI_API_KEY=placeholder
cd "${ROOT}"
[[ "$(cat "${GF}/balanced_preference_v3_generation_status")" == PAUSED_AT_24_CHUNKS ]] || {
  echo '24-chunk generator not paused cleanly' >&2; exit 2;
}
[[ "$(find "${SOURCE}" -name COMPLETE.json | wc -l)" -eq 24 ]] || {
  echo 'expected exactly 24 complete chunks' >&2; exit 2;
}
mkdir -p "${OUT}"
STATUS=${OUT}/preparation_status
trap 'rc=$?; if [[ ${rc} -ne 0 ]]; then printf "FAILED:%s:line%s\n" "${rc}" "${LINENO}" > "${STATUS}"; fi' EXIT
if [[ ! -d "${SNAP}" ]]; then
  printf '%s\n' FREEZING_24_CHUNKS > "${STATUS}"
  "${ENV}/bin/python" -m repro_1p7b.graph_frontier.rich_v3_milestone_snapshot \
    --source "${SOURCE}" --plan "${PLAN}" --output "${SNAP}" \
    --completed-chunks --min-completed-chunks 24 > "${OUT}/snapshot.log" 2>&1
fi
"${ENV}/bin/python" - "${SNAP}/snapshot_manifest.json" "${SOURCE}/resume_manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text()); r=json.loads(Path(sys.argv[2]).read_text())
digest=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert s['selection_mode']=='completed_chunks' and len(s['selected_chunk_indices'])==24
assert s['source_resume_manifest_sha256']==digest(sys.argv[2])
assert s['source_plan_sha256']==r['plan_sha256'] and s['source_graph_sha256']==r['graph_sha256']
PY
if [[ ! -e "${SNAP}/audit_status" ]]; then
  printf '%s\n' AUDITING_SNAPSHOT > "${STATUS}"
  bash "${GF}/run_rich_generation_v3_audit.sh" "${SNAP}" "${SNAP}/pilot_plan.jsonl" formal \
    > "${OUT}/source_audit_controller.log" 2>&1
fi
[[ "$(cat "${SNAP}/audit_status")" == AUDIT_DONE ]] || {
  echo 'Rich-v3 executable audit did not finish' >&2; exit 3;
}
printf '%s\n' REPLAYING_FRONTIERS > "${STATUS}"
"${REPLAY_ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v3_source \
  --snapshot "${SNAP}" --output "${OUT}" > "${OUT}/source_prepare.log" 2>&1
printf '%s\n' RESERVING_HELDOUT > "${STATUS}"
"${ENV}/bin/python" -m repro_1p7b.graph_frontier.balanced_preference_v3_pairs reserve \
  > "${OUT}/heldout_reservation.log" 2>&1
printf '%s\n' SOURCE_READY > "${STATUS}"
