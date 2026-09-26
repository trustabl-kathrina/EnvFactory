"""Matched 3x64-step fixed-gold SFT baseline from original Dynamic-v1."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v1.train import LR, STEPS
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask


def load(cycle: int, model: Path) -> tuple[list[dict], dict]:
    check()
    if cycle not in range(3):
        raise ValueError("static phase must be 0..2")
    base = RUN / "static_gold"
    manifest = json.loads((base / "manifest.json").read_text())
    rows = [json.loads(x) for x in (base / "dataset.jsonl").read_text().splitlines() if x]
    schedule = json.loads((base / "schedule.json").read_text())
    if (manifest.get("schema_version") != "recursive_opd_static_gold_v1"
            or manifest.get("status") != "DATA_READY"
            or manifest.get("protocol_sha256") != sha256(PROTOCOL)
            or manifest.get("dataset_sha256") != sha256(base / "dataset.jsonl")
            or manifest.get("schedule_sha256") != sha256(base / "schedule.json")
            or len(rows) != manifest["rows"] or len(schedule) != 384
            or manifest.get("heldout_overlap") != 0):
        raise RuntimeError("fixed static pool/schedule drift")
    if cycle == 0 and model.resolve() != MODEL.resolve():
        raise RuntimeError("static baseline must start from original Dynamic-v1")
    if cycle > 0:
        old = RUN / f"static/cycle{cycle}/checkpoint"
        state = json.loads((old / "trainer_state.json").read_text())
        if model.resolve() != old.resolve() or state.get("model_sha256") != sha256(model / "model.safetensors"):
            raise RuntimeError("static checkpoint chain drift")
    for row in rows:
        validate_mask(row)
    selected = []
    for i, item in enumerate(schedule[cycle * STEPS * 2:(cycle+1) * STEPS * 2]):
        row = rows[item["row_index"]]
        if (item["step"] != cycle * STEPS + i // 2 + 1
                or item["rank"] != i % 2
                or item["task_id"] != row["task_id"]
                or item["state_identity"] != row["state_identity"]):
            raise RuntimeError("fixed static schedule identity drift")
        selected.append(row)
    plan = {"schema_version": "recursive_opd_static_ddp64_v1", "cycle": cycle,
            "steps": STEPS, "world_size": 2, "full_parameter": True,
            "precision": "bfloat16", "optimizer": "AdamW", "scheduler": "constant",
            "learning_rate": LR, "weight_decay": 0.0,
            "gradient_checkpointing": True,
            "initialization": str(model),
            "initialization_sha256": sha256(model / "model.safetensors"),
            "dataset_sha256": manifest["dataset_sha256"],
            "schedule_sha256": manifest["schedule_sha256"],
            "selected_input_tokens": sum(x["total_tokens"] for x in selected),
            "selected_target_tokens": sum(x["target_tokens"] for x in selected),
            "schedule_state_ids": [x["state_identity"] for x in selected]}
    return selected, plan


def train(cycle: int, model_path: Path) -> None:
    os.environ.setdefault("CUDA_HOME", str(Path(sys.executable).resolve().parents[1]))
    os.environ["PATH"] = str(Path(os.environ["CUDA_HOME"]) / "bin") + ":" + os.environ.get("PATH", "")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, local, world = (int(os.environ[x]) for x in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 2 or local not in (0, 1):
        raise RuntimeError("requires two A100 GPUs")
    selected, plan = load(cycle, model_path)
    output = RUN / f"static/cycle{cycle+1}/checkpoint"
    logs = RUN / f"static/cycle{cycle}/train_manifest"
    if output.exists() or logs.exists():
        raise RuntimeError("refusing to overwrite static baseline checkpoint/log")
    torch.cuda.set_device(local)
    torch.manual_seed(20260927 + cycle)
    torch.cuda.manual_seed_all(20260927 + cycle)
    dist.init_process_group("nccl")
    if rank == 0:
        logs.mkdir(parents=True)
        (logs / "plan.json").write_text(stable(plan) + "\n")
    dist.barrier()
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(f"cuda:{local}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=LR, weight_decay=0.0)
    for step in range(STEPS):
        row = selected[step * 2 + rank]
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=local)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=local)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        finite = torch.tensor([int(torch.isfinite(loss).item())], device=local)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite loss at step {step+1}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        finite = torch.tensor([int(torch.isfinite(grad).item())], device=local)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite grad at step {step+1}")
        optimizer.step()
        torch.cuda.synchronize(local)
        losses = [torch.zeros((), dtype=torch.float32, device=local) for _ in range(2)]
        dist.all_gather(losses, loss.detach().float())
        if rank == 0:
            rec = {"step": step+1, "mean_loss": float(((losses[0]+losses[1])/2).cpu()),
                   "grad_norm": float(grad.detach().cpu()),
                   "rank0_peak_gb": round(torch.cuda.max_memory_allocated(local)/1024**3, 3),
                   "elapsed_seconds": round(time.monotonic()-started, 2)}
            with (logs / "training_metrics.jsonl").open("a") as stream:
                stream.write(stable(rec) + "\n")
            print(stable(rec), flush=True)
    peak = torch.tensor([torch.cuda.max_memory_allocated(local)], dtype=torch.float64, device=local)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    dist.barrier()
    if rank == 0:
        output.mkdir(parents=True)
        model.eval()
        model.save_pretrained(output, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
        tokenizer.save_pretrained(output)
        torch.save(optimizer.state_dict(), output / "optimizer_state.pt")
        result = {"status": "TRAINED", "static_phase": cycle,
                  "steps": STEPS, "model_sha256": sha256(output / "model.safetensors"),
                  "optimizer_sha256": sha256(output / "optimizer_state.pt"),
                  "dataset_sha256": plan["dataset_sha256"],
                  "plan_sha256": sha256(logs / "plan.json"),
                  "peak_gpu_gb": round(peak.item()/1024**3, 3),
                  "wall_clock_seconds": round(time.monotonic()-started, 2)}
        (output / "trainer_state.json").write_text(stable(result) + "\n")
        (logs / "result.json").write_text(stable(result) + "\n")
        print(stable(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    train(args.cycle, args.model)


if __name__ == "__main__":
    main()
