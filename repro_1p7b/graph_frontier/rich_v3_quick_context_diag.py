"""Read-only tokenizer length diagnosis for Rich-v3 quick frontier states."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    for split in ("train", "preference_val"):
        path = args.root / split / "frontier_states.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            state = json.loads(line)
            if split == "train" and state["state_id"] != "frontier-9d80b9a0bbc260af875d67e0":
                continue
            try:
                ids = tokenizer.apply_chat_template(
                    state["prompt_messages"], tools=state["tools"],
                    tokenize=True, add_generation_prompt=True,
                    enable_thinking=False,
                )
                result = {"split": split, "task_id": state["task_id"], "state_id": state["state_id"], "tokens": len(ids), "tools": len(state["tools"])}
            except Exception as exc:
                result = {"split": split, "task_id": state["task_id"], "state_id": state["state_id"], "error": f"{type(exc).__name__}:{exc}"}
            print(json.dumps(result, sort_keys=True))

if __name__ == "__main__":
    main()
