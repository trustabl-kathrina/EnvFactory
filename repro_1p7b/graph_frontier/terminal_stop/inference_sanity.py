"""Two-state deterministic reload/parser smoke; this is not an evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from repro_1p7b.graph_frontier.rich_first_divergence_v1 import MODEL
from repro_1p7b.graph_frontier.smoke_fd_terminal_v2_inference import classify
from src.utils.utils import parse_structured_output


def run(dataset: Path, checkpoint: Path, logs: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite inference sanity")
    states = [json.loads(line) for line in (dataset / "terminal_states.jsonl").read_text().splitlines()
              if line.strip()]
    plan = json.loads((logs / "plan.json").read_text())
    exposed = {step["rank1_task"] for step in plan["schedule"]}
    preferred = ("AmazonSES", "AttomRealEstate")
    chosen = [next(state for state in states
                   if state["environment"] == env and state["task_id"] not in exposed)
              for env in preferred]
    if len(chosen) != 2:
        raise RuntimeError("need two distinct unexposed terminal states")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.eos_token_id != 151645:
        raise RuntimeError("assistant-end token drift")
    observations = []
    for name, path in (("dynamic_v1", MODEL), ("terminal_stop_smoke", checkpoint)):
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa",
            trust_remote_code=True, low_cpu_mem_usage=True,
        ).to("cuda:0")
        model.eval()
        for state in chosen:
            tokens = tokenizer.apply_chat_template(
                state["conversation_prefix"], tools=state["tool_schema"],
                tokenize=True, enable_thinking=False, add_generation_prompt=True,
                return_tensors="pt",
            ).to("cuda:0")
            with torch.no_grad():
                full = model.generate(
                    tokens, do_sample=False, max_new_tokens=256,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )[0]
            generated = full[tokens.shape[1]:].tolist()
            ended = bool(generated and generated[-1] == tokenizer.eos_token_id)
            decoded = tokenizer.decode(generated, skip_special_tokens=False)
            parsed = parse_structured_output(decoded)
            calls = parsed.get("tool_call")
            if calls is None:
                choice = {"message": {"content": parsed.get("non_think", ""),
                                      "tool_calls": []},
                          "finish_reason": "stop" if ended else "length"}
                try:
                    kind, _ = classify(choice)
                    parser_error = None
                except Exception as exc:
                    kind, parser_error = "unknown", str(exc)
            else:
                kind, parser_error = "tool", None
            observations.append({
                "model": name, "task_id": state["task_id"], "environment": state["environment"],
                "prompt_tokens": tokens.shape[1], "generated_tokens": len(generated),
                "assistant_end_generated": ended, "first_generated_id": generated[0] if generated else None,
                "parsed_kind": kind, "parser_error": parser_error,
                "generated_text_excerpt": decoded[:250],
            })
        del model
        torch.cuda.empty_cache()
    result = {
        "schema_version": "terminal_stop_smoke_inference_sanity_v1",
        "status": "COMPLETE", "sample_count": len(chosen),
        "checkpoint_count": 2, "observations": observations,
        "note": "Deterministic parser/reload sanity only; not an effectiveness evaluation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.checkpoint, args.logs, args.output),
                     ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

