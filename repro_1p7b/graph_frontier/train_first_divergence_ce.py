"""Bounded full-parameter assistant-only CE smoke from Dynamic-v1."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from repro_1p7b.graph_frontier.build_first_divergence_v1 import sha256
from repro_1p7b.graph_frontier.rich_first_divergence_v1 import MODEL

STEPS = 16
LR = 1.0e-6
SEED = 20260925


def load_and_plan(dataset_dir: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    rows = [json.loads(line) for line in (dataset_dir / "dataset.jsonl").read_text().splitlines()]
    if (
        manifest.get("dataset_sha256") != sha256(dataset_dir / "dataset.jsonl")
        or manifest.get("model_weights_sha256") != sha256(MODEL / "model.safetensors")
        or manifest.get("example_count") != len(rows)
        or manifest.get("frozen_exact_id_overlap") != 0
        or manifest.get("tool_observation_tokens_in_loss") != 0
        or len(rows) < STEPS
        or len({row["state_identity"] for row in rows}) != len(rows)
    ):
        raise RuntimeError("CE materialization gate failed")
    for row in rows:
        p = row["prompt_tokens"]
        if (
            len(row["input_ids"]) != len(row["labels"])
            or row["labels"][:p] != [-100] * p
            or row["labels"][p:] != row["input_ids"][p:]
            or len(row["input_ids"]) > manifest["max_length"]
        ):
            raise RuntimeError("target-only loss masking failed")
    by_length = sorted(range(len(rows)), key=lambda i: rows[i]["total_tokens"])
    selected = [by_length[0], by_length[len(rows)//2], by_length[-1]]
    rest = [i for i in range(len(rows)) if i not in selected]
    random.Random(SEED).shuffle(rest)
    selected += rest[:STEPS-len(selected)]
    plan = {
        "schema_version": "first_divergence_ce_smoke_plan_v1",
        "base_model": str(MODEL),
        "base_model_weights_sha256": manifest["model_weights_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "steps": STEPS, "learning_rate": LR, "seed": SEED,
        "batch_size": 1, "gradient_accumulation_steps": 1,
        "optimizer": "AdamW", "weight_decay": 0,
        "precision": "bfloat16", "full_parameter": True,
        "gradient_checkpointing": True,
        "selected": [{
            "task_id": rows[i]["task_id"],
            "state_identity": rows[i]["state_identity"],
            "total_tokens": rows[i]["total_tokens"],
            "target_tokens": rows[i]["target_tokens"],
            "divergence_type": rows[i]["divergence_type"],
        } for i in selected],
    }
    return [rows[i] for i in selected], plan


def train(dataset_dir: Path, output: Path, logs: Path) -> dict:
    # Accelerate imports DeepSpeed while saving, which expects a CUDA toolkit.
    os.environ.setdefault("CUDA_HOME", str(Path(sys.executable).resolve().parents[1]))
    os.environ["PATH"] = str(Path(os.environ["CUDA_HOME"]) / "bin") + ":" + os.environ.get("PATH", "")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if output.exists() or logs.exists():
        raise RuntimeError("refusing to overwrite smoke checkpoint or logs")
    selected, plan = load_and_plan(dataset_dir)
    logs.mkdir(parents=True)
    (logs / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if torch.cuda.memory_allocated(0) > 100_000_000:
        raise RuntimeError("GPU0 unexpectedly occupied")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    metrics = []
    for step_index, row in enumerate(selected, start=1):
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda:0")
        labels = torch.tensor([row["labels"]], dtype=torch.long, device="cuda:0")
        mask = torch.ones_like(ids)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"nonfinite loss at step {step_index}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm).item():
            raise RuntimeError(f"nonfinite gradient at step {step_index}")
        optimizer.step()
        torch.cuda.synchronize()
        record = {
            "step": step_index, "task_id": row["task_id"],
            "tokens": row["total_tokens"], "target_tokens": row["target_tokens"],
            "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu()),
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 3),
        }
        metrics.append(record)
        with (logs / "metrics.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
    model.eval()
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(output)
    result = {
        "schema_version": "first_divergence_ce_smoke_result_v1",
        "status": "TRAINED", "steps": STEPS,
        "initial_loss": metrics[0]["loss"], "final_loss": metrics[-1]["loss"],
        "all_loss_finite": True, "all_grad_finite": True,
        "peak_gpu_gb": max(x["gpu_peak_gb"] for x in metrics),
        "model_path": str(output), "model_weights_sha256": sha256(output / "model.safetensors"),
        "dataset_sha256": plan["dataset_sha256"],
    }
    (logs / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset_dir, args.output, args.logs), sort_keys=True))


if __name__ == "__main__":
    main()
