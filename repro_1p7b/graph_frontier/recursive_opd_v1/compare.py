"""Task-paired Recursive OPD frontier and correction-retention accounting."""
from __future__ import annotations

import argparse
import collections
import json

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import RUN, check


def records(cycle: int) -> dict[str, dict]:
    base = RUN / f"cycle{cycle}/metrics"
    summary = json.loads((base / "summary.json").read_text())
    if summary["per_task_sha256"] != sha256(base / "per_task.jsonl"):
        raise RuntimeError("per-task metrics hash drift")
    rows = [json.loads(x) for x in (base / "per_task.jsonl").read_text().splitlines() if x]
    if len(rows) != 2158 or len({r["task_id"] for r in rows}) != 2158:
        raise RuntimeError("cycle metrics incomplete")
    return {r["task_id"]: r for r in rows}


def movement(old: dict[str, dict], new: dict[str, dict], tasks: list[str]) -> dict:
    outcome = collections.Counter()
    for task in tasks:
        a, b = old[task], new[task]
        if a["gold_tool_count"] != b["gold_tool_count"] or a["pool"] != b["pool"]:
            raise RuntimeError(f"paired task/gold drift: {task}")
        delta = b["frontier_depth"] - a["frontier_depth"]
        outcome["FORWARD" if delta > 0 else "BACKWARD" if delta < 0 else "SAME"] += 1
    return {key: outcome[key] for key in ("FORWARD", "SAME", "BACKWARD")}


def compare(previous: int) -> dict:
    if previous not in range(3):
        raise ValueError("compare 0->1, 1->2, or 2->3")
    protocol = check()
    old, new = records(previous), records(previous + 1)
    train = protocol["train_task_ids"]
    heldout = protocol["heldout_task_ids"]
    corrected = [task for task in train if old[task]["status"] == "FIRST_ERROR"]
    if len(corrected) != len({task for task in corrected}):
        raise RuntimeError("duplicate correction task")
    return {"transition": f"pi{previous}->pi{previous+1}",
            "train": movement(old, new, train),
            "heldout": movement(old, new, heldout),
            "correction_retention": movement(old, new, corrected),
            "corrected_task_count": len(corrected),
            "success": {"train_before": sum(old[t]["status"] == "SUCCESS" for t in train),
                        "train_after": sum(new[t]["status"] == "SUCCESS" for t in train),
                        "heldout_before": sum(old[t]["status"] == "SUCCESS" for t in heldout),
                        "heldout_after": sum(new[t]["status"] == "SUCCESS" for t in heldout)},
            "previous_metrics_sha256": sha256(RUN / f"cycle{previous}/metrics/per_task.jsonl"),
            "next_metrics_sha256": sha256(RUN / f"cycle{previous+1}/metrics/per_task.jsonl")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cycle", type=int, required=True)
    args = parser.parse_args()
    value = compare(args.from_cycle)
    output = RUN / f"cycle{args.from_cycle+1}/metrics/paired_from_cycle{args.from_cycle}.json"
    if output.exists():
        raise RuntimeError("refusing to overwrite paired metrics")
    output.write_text(stable(value) + "\n")
    print(stable(value))


if __name__ == "__main__":
    main()
