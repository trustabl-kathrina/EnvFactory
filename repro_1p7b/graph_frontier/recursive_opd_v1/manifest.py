"""Create/check immutable per-policy rollout identity before allocating GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import CONFIG, PROTOCOL, RUN, check


def expected(cycle: int, model: Path) -> dict:
    protocol = check()
    if cycle not in range(4):
        raise ValueError("cycle must be 0..3")
    if cycle == 0 and model.resolve() != Path(protocol["pi0_model"]).resolve():
        raise RuntimeError("cycle0 must use original Dynamic-v1")
    if cycle > 0:
        previous = RUN / f"cycle{cycle}/checkpoint/trainer_state.json"
        state = json.loads(previous.read_text())
        if state.get("status") != "TRAINED" or state.get("cycle") != cycle-1:
            raise RuntimeError("previous policy checkpoint is not trained")
        if model.resolve() != previous.parent.resolve():
            raise RuntimeError("cycle must use current policy checkpoint")
        if sha256(model / "model.safetensors") != state["model_sha256"]:
            raise RuntimeError("checkpoint weight drift")
    return {"schema_version": "recursive_opd_rollout_v1", "cycle": cycle,
            "model_path": str(model), "model_sha256": sha256(model / "model.safetensors"),
            "protocol_sha256": sha256(PROTOCOL), "served_model": f"recursive-opd-pi{cycle}",
            "config": CONFIG, "task_count": len(protocol["train_task_ids"]) + len(protocol["heldout_task_ids"]),
            "train_count": len(protocol["train_task_ids"]),
            "heldout_count": len(protocol["heldout_task_ids"])}


def lock(cycle: int, model: Path) -> dict:
    value = expected(cycle, model)
    path = RUN / f"cycle{cycle}/rollout_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("rollout manifest differs from frozen policy")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stable(value) + "\n")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(stable(lock(args.cycle, args.model)))


if __name__ == "__main__":
    main()
