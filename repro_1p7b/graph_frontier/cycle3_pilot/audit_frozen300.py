"""Read-only four-model Frozen300 profile audit (machine-readable only)."""
from __future__ import annotations

import collections
import json
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "repro_1p7b/graph_frontier/cycle3_pilot"
MANIFEST = ROOT / "repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl"
DIRS = {
    "Dynamic-v1": ROOT / "repro_1p7b/results/graph_frontier/confirm_300/dynamic_v1",
    "FD-only": ROOT / "repro_1p7b/logs/first_divergence_onpolicy_v1/frozen300_cycle3",
    "v2-1to1": ROOT / "repro_1p7b/logs/first_divergence_terminal_v2/frozen300_1to1",
    "Cycle-3": PILOT / "evaluation/frozen300",
}
OUTPUT = PILOT / "evaluation/frozen300/profile_audit.json"


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite Frozen300 profile audit")
    if sha256(MANIFEST) != "4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c":
        raise RuntimeError("Frozen300 manifest changed")
    expected = {json.loads(line)["task_id"] for line in MANIFEST.read_text().splitlines() if line.strip()}
    if len(expected) != 300:
        raise RuntimeError("Frozen300 task identity count changed")
    result = {"schema_version": "cycle3_frozen300_profile_audit_v1",
              "manifest_sha256": sha256(MANIFEST), "task_count": 300,
              "models": {}}
    for name, directory in DIRS.items():
        rows = [json.loads(p.read_text()) for p in (directory / "tasks").glob("*.result.json")]
        if len(rows) != 300 or {r["task_id"] for r in rows} != expected:
            raise RuntimeError(f"{name} Frozen300 task identity mismatch")
        if not all(r["valid_capability_probe"] is True for r in rows):
            raise RuntimeError(f"{name} invalid capability probe")
        core = json.loads((directory / "comparison/summary.json").read_text()) if name != "Dynamic-v1" else None
        if core and (core["valid_probe_count"] != 300 or core["manifest_sha256"] != sha256(MANIFEST)):
            raise RuntimeError(f"{name} comparison lock mismatch")
        metrics = collections.Counter()
        for row in rows:
            p = row["profile"]
            if not isinstance(p, dict) or row.get("profiler_error") is not None:
                raise RuntimeError(f"{name} profile invalid")
            metrics["tool_events"] += row["tool_call_count"]
            metrics["redundant_calls"] += row["redundant_calls"]
            metrics["unexpected_calls"] += row["unexpected_calls"]
            metrics["zero_tool_tasks"] += row["tool_call_count"] == 0
            metrics["incomplete_final_turns"] += bool(p["missing_expected_tools"] and row["final_assistant_finish_reason"] == "stop")
            metrics["internal_e2e_tasks"] += bool(row["internal_param_complete"] and row["final_state_success"])
            edges = [e for e in p["dependency_edge_checks"] if e.get("internal_parameter") is True]
            metrics["eligible_internal_edges"] += len(edges)
            reached = [e for e in edges if isinstance(e.get("consumer_step"), int)]
            metrics["consumer_reached_edges"] += len(reached)
            metrics["conditional_propagation_checked"] += sum(e.get("source_value_available") is True
                                                               and e.get("target_value_available") is True
                                                               for e in reached)
            metrics["conditional_propagation_correct"] += sum(e.get("source_value_available") is True
                                                               and e.get("target_value_available") is True
                                                               and e.get("value_match") is True
                                                               for e in reached)
        for key in ("task_success", "final_state_success", "edge_complete", "gold_node_complete",
                    "internal_param_complete", "reference_path_complete_success"):
            metrics[key] = sum(bool(r[key]) for r in rows)
        if core and any(metrics[k] != core["trained"][k] for k in core["trained"] if k in metrics):
            raise RuntimeError(f"{name} core profile/summary mismatch")
        result["models"][name] = dict(metrics)
    output.write_text(stable(result) + "\n")
    return result


if __name__ == "__main__":
    print(stable(run()), flush=True)
