"""Two-request local SGLang smoke: forced structured tool call and auto terminal turn."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def request(endpoint: str, model: str, messages: list[dict], tools: list[dict], choice: str) -> dict:
    response = requests.post(
        endpoint.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer placeholder"},
        json={
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": choice,
            "temperature": 0,
            "top_p": 0.95,
            "max_tokens": 256,
            "seed": 20260925,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]


def classify(choice: dict) -> tuple[str, str | None]:
    message = choice["message"]
    calls = message.get("tool_calls") or []
    if len(calls) == 1:
        function = calls[0]["function"]
        args = function["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict) or not isinstance(function.get("name"), str):
            raise RuntimeError("untyped tool call")
        return "tool", function["name"]
    if len(calls) > 1:
        raise RuntimeError("parallel tool calls in smoke")
    if choice.get("finish_reason") == "length":
        raise RuntimeError("truncated assistant output")
    return "final", None


def smoke(dataset: Path, endpoint: str, model: str, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("refusing to overwrite smoke inference")
    target_id = "gf-rich-7372af308535cceb8e11"
    sample = None
    for line in dataset.read_text().splitlines():
        row = json.loads(line)
        if row["task_id"] == target_id:
            sample = row
            break
    if sample is None:
        raise RuntimeError("locked smoke task missing from frozen terminal data")
    initial = request(endpoint, model, sample["conversation_prefix"][:1], sample["tool_schema"], "required")
    initial_kind, initial_tool = classify(initial)
    if initial_kind != "tool":
        raise RuntimeError("required tool call was not structured")
    terminal = request(endpoint, model, sample["conversation_prefix"], sample["tool_schema"], "auto")
    terminal_kind, terminal_tool = classify(terminal)
    result = {
        "schema_version": "fd_terminal_v2_smoke_inference",
        "status": "SMOKE_PASS",
        "task_id": target_id,
        "forced_tool_kind": initial_kind,
        "forced_tool_name": initial_tool,
        "terminal_auto_kind": terminal_kind,
        "terminal_auto_tool_name": terminal_tool,
        "server_model": model,
        "tool_schema_count": len(sample["tool_schema"]),
        "note": "Protocol and checkpoint reload test; not an evaluation score.",
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(smoke(args.dataset, args.endpoint, args.model, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
