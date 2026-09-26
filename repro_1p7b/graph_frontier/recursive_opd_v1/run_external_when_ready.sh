#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/u2024311031/workspace/envfactory_repro_1p7b
RUN=$ROOT/repro_1p7b/graph_frontier/recursive_opd_v1_run
STATUS=$RUN/external_pipeline_status
cd "$ROOT"

exec 9>"$RUN/.external_pipeline.lock"
flock -n 9 || { echo 'external watcher already active' >&2; exit 2; }

printf 'WAIT_STATIC\n' > "$STATUS"
while true; do
  if [[ -f $RUN/pipeline_status ]]; then
    current=$(<"$RUN/pipeline_status")
    case "$current" in
      STATIC_EVALUATED_AWAITING_EXTERNAL) break ;;
      FAILED_*|BLOCKED_*) printf 'BLOCKED_UPSTREAM:%s\n' "$current" > "$STATUS"; exit 3 ;;
    esac
  fi
  sleep 60
done

for label in recursive_pi3 static_pi3; do
  if [[ -f $RUN/evaluation/$label/status ]] &&
     [[ $(<"$RUN/evaluation/$label/status") == AUDITED ]]; then
    continue
  fi
  printf 'EVALUATE_%s\n' "$label" > "$STATUS"
  bash repro_1p7b/graph_frontier/recursive_opd_v1/run_external.sh "$label" \
    > "$RUN/evaluate_$label.launcher.log" 2>&1 || {
      printf 'FAILED_EVALUATE_%s\n' "$label" > "$STATUS"
      exit 4
    }
done

printf 'AUDITED\n' > "$STATUS"
