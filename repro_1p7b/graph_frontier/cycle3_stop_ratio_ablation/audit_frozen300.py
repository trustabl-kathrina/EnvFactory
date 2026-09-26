"""Read-only Frozen300 ablation summary with exact task/protocol identity gates."""
from __future__ import annotations

import json
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.cycle3_stop_ratio_ablation.prepare import OUT

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "repro_1p7b/results/graph_frontier/confirm_300/frozen/confirm_300_seed_20260914.jsonl"
EOS1_PROFILE = ROOT / "repro_1p7b/graph_frontier/cycle3_pilot/evaluation/frozen300/profile_audit.json"
OUTPUT = OUT / "evaluation/frozen300_ablation_summary.json"


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite Frozen300 ablation summary")
    if sha256(MANIFEST) != "4ea5304d6f76294d70166260986767795fa0bf3fcf54b6196ee0c3f71e70f51c":
        raise RuntimeError("Frozen300 manifest changed")
    expected = {json.loads(line)["task_id"] for line in MANIFEST.read_text().splitlines() if line.strip()}
    if len(expected) != 300:
        raise RuntimeError("Frozen300 task IDs changed")
    prior = json.loads(EOS1_PROFILE.read_text())["models"]
    labels = {"Dynamic-v1": prior["Dynamic-v1"], "FD-only": prior["FD-only"],
              "old 1:1": prior["v2-1to1"], "EOS 1:1": prior["Cycle-3"]}
    result = {"schema_version": "cycle3_eos_ratio_frozen300_summary_v1",
              "task_count": 300, "manifest_sha256": sha256(MANIFEST),
              "eos1_profile_sha256": sha256(EOS1_PROFILE), "models": labels}
    base_protocol = json.loads((ROOT / "repro_1p7b/logs/first_divergence_terminal_v2/frozen300_1to1/comparison/summary.json").read_text())["protocol"]
    for ratio in (2, 4):
        directory = OUT / f"evaluation/eos_{ratio}to1/frozen300"
        summary = json.loads((directory / "comparison/summary.json").read_text())
        if (summary["task_count"] != 300 or summary["valid_probe_count"] != 300
                or summary["manifest_sha256"] != sha256(MANIFEST)
                or summary["protocol"] != base_protocol):
            raise RuntimeError(f"EOS {ratio}:1 Frozen300 protocol/count mismatch")
        rows = [json.loads(p.read_text()) for p in (directory / "tasks").glob("*.result.json")]
        if len(rows) != 300 or {r["task_id"] for r in rows} != expected:
            raise RuntimeError(f"EOS {ratio}:1 Frozen300 task identity mismatch")
        if not all(r.get("valid_capability_probe") is True and r.get("profiler_error") is None for r in rows):
            raise RuntimeError(f"EOS {ratio}:1 Frozen300 invalid probe/profile")
        core = summary["trained"]
        metrics = {
            "task_success": sum(bool(r["task_success"]) for r in rows),
            "reference_path_complete_success": sum(bool(r["reference_path_complete_success"]) for r in rows),
            "final_state_success": sum(bool(r["final_state_success"]) for r in rows),
            "edge_complete": sum(bool(r["edge_complete"]) for r in rows),
            "gold_node_complete": sum(bool(r["gold_node_complete"]) for r in rows),
            "internal_param_complete": sum(bool(r["internal_param_complete"]) for r in rows),
            "tool_events": sum(r["tool_call_count"] for r in rows),
            "unexpected_calls": sum(r["unexpected_calls"] for r in rows),
            "redundant_calls": sum(r["redundant_calls"] for r in rows),
            "incomplete_final_turns": sum(bool(r["profile"]["missing_expected_tools"]
                                               and r["final_assistant_finish_reason"] == "stop") for r in rows),
        }
        if any(metrics[k] != core[k] for k in core if k in metrics):
            raise RuntimeError(f"EOS {ratio}:1 profile/summary mismatch")
        if metrics["tool_events"] != core["tool_call_count"]:
            raise RuntimeError(f"EOS {ratio}:1 tool count mismatch")
        labels[f"EOS {ratio}:1"] = metrics
    output.write_text(stable(result) + "\n")
    return result


if __name__ == "__main__":
    print(stable(run()), flush=True)
