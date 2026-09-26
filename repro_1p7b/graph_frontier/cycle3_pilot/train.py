"""Two-GPU bounded CE pilot from original Dynamic-v1, never from smoke.

One verified FD target and one EOS-only terminal target are exposed per
optimizer step. This retains the previously smoke-tested full-parameter BF16
DDP/AdamW stack and its effective constant 2e-6 learning rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_pilot.prepare import (
    FD_SOURCE, MODEL, OUT as DATA, RUN, TERM_SOURCE, read_jsonl,
)
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

LR = 2e-6
SEED = 20260927


def load_plan(data: Path, model_path: Path, mode: str) -> tuple[list[dict], dict]:
    if mode not in ("smoke", "formal"):
        raise ValueError("only two-step smoke or exact 64-step formal pilot")
    manifest = json.loads((data / "train_manifest.json").read_text())
    sample = json.loads((data / "sampling_manifest.json").read_text())
    fd = read_jsonl(data / "fd_ce_dataset.jsonl")
    term = read_jsonl(TERM_SOURCE)
    if (manifest.get("status") != "DATA_READY"
            or manifest.get("schema_version") != "cycle3_execfd_eos_pilot_data_v1"
            or manifest.get("steps") != 64
            or manifest.get("fd_available") != 989
            or manifest.get("terminal_available") != 2154
            or len(fd) != 989 or len(term) != 2154
            or manifest.get("fd_ce_sha256") != sha256(data / "fd_ce_dataset.jsonl")
            or manifest.get("source_fd_sha256") != sha256(FD_SOURCE)
            or manifest.get("source_terminal_sha256") != sha256(TERM_SOURCE)
            or manifest.get("source_plan_sha256") != sha256(RUN / "rollout_manifest.json")
            or model_path.resolve() != MODEL.resolve()
            or manifest.get("dynamic_v1_model_sha256") != sha256(MODEL / "model.safetensors")
            or manifest.get("tokenizer_sha256") != sha256(MODEL / "tokenizer.json")
            or sample.get("schema_version") != "cycle3_execfd_eos_exposure_v1"
            or sample.get("counts", {}).get("FD") != 64
            or sample.get("counts", {}).get("terminal_stop") != 64):
        raise RuntimeError("pilot source/model/data lock failed")
    for row in fd + term:
        validate_mask(row)
    fd_by = {r["task_id"]: r for r in fd}
    term_by = {r["task_id"]: r for r in term}
    if len(fd_by) != 989 or len(term_by) != 2154:
        raise RuntimeError("duplicate training task identity")
    schedule = (sample["memory_smoke_schedule"] if mode == "smoke"
                else sample["formal_schedule"])
    if len(schedule) != (2 if mode == "smoke" else 64):
        raise RuntimeError("fixed step budget changed")
    pairs = []
    for i, item in enumerate(schedule):
        if item["step"] != i + 1:
            raise RuntimeError("schedule order drift")
        a, b = fd_by[item["fd_task_id"]], term_by[item["terminal_task_id"]]
        if a["target_type"] != "first_divergence_tool" or b["target_type"] != "terminal_stop":
            raise RuntimeError("FD:EOS exposure type drift")
        pairs.append((a, b))
    if mode == "formal":
        if (len({a["task_id"] for a, _ in pairs}) != 64
                or len({b["task_id"] for _, b in pairs}) != 64):
            raise RuntimeError("formal exposure must use 64 unique tasks per type")
    plan = {
        "schema_version": "cycle3_execfd_eos_ddp_plan_v1", "mode": mode,
        "steps": len(pairs), "world_size": 2, "per_rank_batch_size": 1,
        "ratio": "FD:terminal_stop=1:1", "initialization": str(model_path),
        "initialization_sha256": manifest["dynamic_v1_model_sha256"],
        "precision": "bfloat16", "full_parameter": True, "optimizer": "AdamW",
        "scheduler": "none_constant_lr_as_prior_CE_trainer",
        "learning_rate": LR, "weight_decay": 0.0,
        "gradient_checkpointing": True, "seed": SEED,
        "train_manifest_sha256": sha256(data / "train_manifest.json"),
        "sampling_manifest_sha256": sha256(data / "sampling_manifest.json"),
        "fd_ce_sha256": manifest["fd_ce_sha256"],
        "terminal_ce_sha256": manifest["source_terminal_sha256"],
        "exposures": {"FD": len(pairs), "terminal_stop": len(pairs)},
        "selected_input_tokens": sum(a["total_tokens"] + b["total_tokens"] for a, b in pairs),
        "selected_target_tokens": sum(a["target_tokens"] + b["target_tokens"] for a, b in pairs),
    }
    return pairs, plan


def train(data: Path, model_path: Path, output: Path, logs: Path, mode: str) -> None:
    os.environ.setdefault("CUDA_HOME", str(Path(sys.executable).resolve().parents[1]))
    os.environ["PATH"] = str(Path(os.environ["CUDA_HOME"]) / "bin") + ":" + os.environ.get("PATH", "")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, local_rank, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 2 or local_rank not in (0, 1):
        raise RuntimeError("pilot requires exactly CUDA GPUs 0 and 1")
    if output.exists() or logs.exists():
        raise RuntimeError("refusing to overwrite pilot training outputs")
    pairs, plan = load_plan(data, model_path, mode)
    torch.cuda.set_device(local_rank)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dist.init_process_group("nccl")
    if rank == 0:
        logs.mkdir(parents=True)
        (logs / "plan.json").write_text(stable(plan) + "\n")
    dist.barrier()
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(f"cuda:{local_rank}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
              find_unused_parameters=False)
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=LR, weight_decay=0.0)
    records = []
    for step, (fd, term) in enumerate(pairs, start=1):
        row = fd if rank == 0 else term
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=local_rank)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=local_rank)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        finite = torch.tensor([int(torch.isfinite(loss).item())], device=local_rank)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
        finite = torch.tensor([int(torch.isfinite(grad).item())], device=local_rank)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if finite.item() != 1:
            raise RuntimeError(f"nonfinite grad norm at step {step}")
        optimizer.step()
        losses = [torch.zeros((), dtype=torch.float32, device=local_rank) for _ in range(2)]
        dist.all_gather(losses, loss.detach().float())
        torch.cuda.synchronize(local_rank)
        if rank == 0:
            record = {
                "step": step, "fd_task_id": fd["task_id"],
                "terminal_task_id": term["task_id"],
                "fd_loss": float(losses[0].cpu()),
                "terminal_stop_loss": float(losses[1].cpu()),
                "mean_loss": float(((losses[0] + losses[1]) / 2).cpu()),
                "grad_norm": float(grad.detach().cpu()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "fd_input_tokens": fd["total_tokens"],
                "terminal_input_tokens": term["total_tokens"],
                "fd_target_tokens": fd["target_tokens"],
                "terminal_active_labels": 1,
                "rank0_peak_gb": round(torch.cuda.max_memory_allocated(local_rank) / 1024**3, 3),
                "elapsed_seconds": round(time.monotonic() - start, 2),
            }
            records.append(record)
            with (logs / "training_metrics.jsonl").open("a") as stream:
                stream.write(stable(record) + "\n")
            print(stable(record), flush=True)
    peak = torch.tensor([torch.cuda.max_memory_allocated(local_rank)],
                        dtype=torch.float64, device=local_rank)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    dist.barrier()
    if rank == 0:
        output.mkdir(parents=True)
        if mode == "formal":
            model.eval()
            model.save_pretrained(output, safe_serialization=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                                      local_files_only=True)
            tokenizer.save_pretrained(output)
            torch.save(optimizer.state_dict(), output / "optimizer_state.pt")
            (output / "scheduler_state.json").write_text(stable({
                "family": "none", "constant_learning_rate": LR,
                "last_completed_step": len(pairs),
            }) + "\n")
        result = {
            "schema_version": "cycle3_execfd_eos_ddp_result_v1",
            "status": "SMOKE_PASS" if mode == "smoke" else "TRAINED",
            "mode": mode, "steps": len(pairs), "world_size": 2,
            "full_parameter": True, "precision": "bfloat16",
            "all_loss_finite": all(r["mean_loss"] == r["mean_loss"] for r in records),
            "all_grad_finite": all(r["grad_norm"] == r["grad_norm"] for r in records),
            "fd_exposures": len(records), "terminal_exposures": len(records),
            "trained_input_tokens": sum(r["fd_input_tokens"] + r["terminal_input_tokens"]
                                        for r in records),
            "trained_target_tokens": sum(r["fd_target_tokens"] + 1 for r in records),
            "initialization_sha256": plan["initialization_sha256"],
            "final_loss": records[-1]["mean_loss"],
            "final_grad_norm": records[-1]["grad_norm"],
            "peak_gpu_gb": round(peak.item() / 1024**3, 3),
            "wall_clock_seconds": round(time.monotonic() - start, 2),
            "steps_per_second": round(len(pairs) / (time.monotonic() - start), 5),
            "model_sha256": (sha256(output / "model.safetensors") if mode == "formal" else None),
            "optimizer_sha256": (sha256(output / "optimizer_state.pt") if mode == "formal" else None),
            "schedule_sha256": plan["sampling_manifest_sha256"],
        }
        (output / "trainer_state.json").write_text(stable(result) + "\n")
        (logs / "result.json").write_text(stable(result) + "\n")
        print(stable(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    train(args.data, args.model_path, args.output, args.logs, args.mode)


if __name__ == "__main__":
    main()
