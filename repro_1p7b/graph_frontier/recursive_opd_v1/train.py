"""Exactly 64 full-parameter BF16 DDP updates from the current policy."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, RUN, check
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

STEPS = 64
LR = 2e-6
SEED = 20260927


def schedule(rows: list[dict], cycle: int) -> tuple[list[int], dict]:
    if len(rows) < 64 or len({r["task_id"] for r in rows}) != len(rows):
        raise RuntimeError("Dk too small or duplicated")
    rng = random.Random(SEED + cycle)
    order = []
    while len(order) < STEPS * 2:
        epoch = list(range(len(rows)))
        rng.shuffle(epoch)
        order.extend(epoch)
    order = order[:STEPS * 2]
    counts = {index: order.count(index) for index in set(order)}
    return order, {"unique_exposed_tasks": len(counts),
                   "repeated_exposures": len(order) - len(counts),
                   "max_exposure_per_task": max(counts.values()),
                   "selected_input_tokens": sum(rows[i]["total_tokens"] for i in order),
                   "selected_target_tokens": sum(rows[i]["target_tokens"] for i in order)}


def load(cycle: int, model: Path) -> tuple[list[dict], list[int], dict]:
    protocol = check()
    if cycle not in range(3):
        raise ValueError("only cycle 0,1,2 can train")
    data = RUN / f"cycle{cycle}/first_error_dataset"
    manifest = json.loads((data / "manifest.json").read_text())
    rows = [json.loads(x) for x in (data / "dataset.jsonl").read_text().splitlines() if x]
    if (manifest.get("schema_version") != "recursive_opd_first_error_dataset_v1"
            or manifest.get("status") != "DATA_READY"
            or manifest.get("cycle") != cycle
            or manifest.get("dataset_sha256") != sha256(data / "dataset.jsonl")
            or manifest.get("task_count") != len(rows)
            or manifest.get("heldout_overlap") != 0
            or manifest.get("observation_tokens_in_loss") != 0
            or manifest.get("source_model_sha256") != sha256(model / "model.safetensors")
            or (cycle == 0 and model.resolve() != Path(protocol["pi0_model"]).resolve())):
        raise RuntimeError("Recursive OPD data/current-policy lock failed")
    for row in rows:
        validate_mask(row)
    order, exposure = schedule(rows, cycle)
    return rows, order, {"schema_version": "recursive_opd_ddp64_plan_v1",
                          "cycle": cycle, "steps": STEPS, "world_size": 2,
                          "per_rank_batch_size": 1, "full_parameter": True,
                          "precision": "bfloat16", "optimizer": "AdamW",
                          "scheduler": "constant", "learning_rate": LR,
                          "weight_decay": 0.0, "gradient_checkpointing": True,
                          "seed": SEED + cycle, "initialization": str(model),
                          "initialization_sha256": sha256(model / "model.safetensors"),
                          "dataset_sha256": manifest["dataset_sha256"],
                          "protocol_sha256": sha256(PROTOCOL),
                          "schedule_task_ids": [rows[i]["task_id"] for i in order],
                          **exposure}


def train(cycle: int, model_path: Path) -> None:
    os.environ.setdefault("CUDA_HOME", str(Path(sys.executable).resolve().parents[1]))
    os.environ["PATH"] = str(Path(os.environ["CUDA_HOME"]) / "bin") + ":" + os.environ.get("PATH", "")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, local, world = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 2 or local not in (0, 1):
        raise RuntimeError("requires two A100 GPUs")
    rows, order, plan = load(cycle, model_path)
    output = RUN / f"cycle{cycle+1}/checkpoint"
    logs = RUN / f"cycle{cycle}/train_manifest"
    if output.exists() or logs.exists():
        raise RuntimeError("refusing to overwrite checkpoint or train manifest")
    torch.cuda.set_device(local)
    torch.manual_seed(SEED + cycle)
    torch.cuda.manual_seed_all(SEED + cycle)
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
        row = rows[order[step * 2 + rank]]
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
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        finite = torch.tensor([int(torch.isfinite(grad_norm).item())], device=local)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite gradient norm at step {step+1}")
        optimizer.step()
        losses = [torch.zeros((), dtype=torch.float32, device=local) for _ in range(2)]
        dist.all_gather(losses, loss.detach().float())
        torch.cuda.synchronize(local)
        if rank == 0:
            record = {"step": step+1,
                      "rank0_task": rows[order[step*2]]["task_id"],
                      "rank1_task": rows[order[step*2+1]]["task_id"],
                      "mean_loss": float(((losses[0] + losses[1]) / 2).cpu()),
                      "grad_norm": float(grad_norm.detach().cpu()),
                      "rank0_peak_gb": round(torch.cuda.max_memory_allocated(local)/1024**3, 3),
                      "elapsed_seconds": round(time.monotonic()-started, 2)}
            with (logs / "training_metrics.jsonl").open("a") as stream:
                stream.write(stable(record) + "\n")
            print(stable(record), flush=True)
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
        result = {"status": "TRAINED", "cycle": cycle, "steps": STEPS,
                  "model_sha256": sha256(output / "model.safetensors"),
                  "optimizer_sha256": sha256(output / "optimizer_state.pt"),
                  "dataset_sha256": plan["dataset_sha256"],
                  "train_plan_sha256": sha256(logs / "plan.json"),
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
