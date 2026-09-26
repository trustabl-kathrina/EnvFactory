"""Fixed clean gold SFT pool, selected without consulting any current policy."""
from __future__ import annotations

import collections
import hashlib
import json
import random

from transformers import AutoTokenizer

from repro_1p7b.graph_frontier.assistant_target_ce_v2 import encode_first_divergence, encode_terminal_stop
from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for, terminal_history, valid_gold
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, OUT, query_from_row
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, RUN, PROTOCOL, check
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import GOLD_RUN
from repro_1p7b.graph_frontier.train_fd_terminal_v2_ddp import validate_mask

SEED = 20260928


def build() -> dict:
    protocol = check()
    output = RUN / "static_gold"
    if output.exists():
        raise RuntimeError("refusing to overwrite fixed static gold pool")
    source = json.loads(SOURCE.read_text())
    by_id = {r["audit_id"]: r for r in json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id != 151645:
        raise RuntimeError("tokenizer EOS drift")
    rows, skipped = [], []
    for task in protocol["train_task_ids"]:
        entry = by_id[task]
        source_row = source[entry["source_row_index"]]
        episode = json.loads((GOLD_RUN / "episodes" / f"{task}.json").read_text())
        if not valid_gold(entry, source_row, episode):
            cache = json.loads((RUN / "gold_cache" / f"{task}.json").read_text())
            if cache["task_id"] != task:
                raise RuntimeError("static gold cache identity drift")
            episode = {**episode, "gold": cache["gold"],
                       "schema_sha256": hashlib.sha256(stable(cache["schemas"]).encode()).hexdigest()}
            schema = schema_for(RUN, episode)
        else:
            schema = schema_for(GOLD_RUN, episode)
        if not valid_gold(entry, source_row, episode):
            raise RuntimeError(f"static gold replay invalid: {task}")
        gold = episode["gold"]
        full_history = terminal_history(query_from_row(source_row), gold)
        for pointer, action in enumerate(gold["actions"]):
            sample = {"task_id": task, "state_identity": f"{task}:static_gold:{pointer}",
                      "independent_prefix_replay_valid": True,
                      "conversation_prefix": full_history[:1 + 2 * pointer],
                      "gold_action": action}
            try:
                item = encode_first_divergence(tokenizer, sample, {"task_id": task, "tools": schema})
                validate_mask(item)
                item.update(source_task_id=task, gold_step=pointer + 1,
                            static_target="GOLD_TOOL", pool="train")
                rows.append(item)
            except ValueError as exc:
                skipped.append({"task_id": task, "gold_step": pointer + 1, "reason": str(exc)})
        terminal = {"task_id": task, "state_identity": f"{task}:static_gold:terminal",
                    "replay_valid": True, "final_state_match": True,
                    "conversation_prefix": full_history, "tool_schema": schema}
        try:
            item = encode_terminal_stop(tokenizer, terminal)
            validate_mask(item)
            item.update(source_task_id=task, gold_step=len(gold["actions"])+1,
                        static_target="GOLD_TERMINAL", pool="train")
            rows.append(item)
        except ValueError as exc:
            skipped.append({"task_id": task, "gold_step": "terminal", "reason": str(exc)})
    if len(rows) < 384 or len({r["state_identity"] for r in rows}) != len(rows):
        raise RuntimeError("static gold pool too small or duplicated")
    if {r["task_id"] for r in rows} & set(protocol["heldout_task_ids"]):
        raise RuntimeError("static gold leaks heldout task")
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    selected = order[:384]
    output.mkdir(parents=True)
    (output / "dataset.jsonl").write_text("".join(stable(r) + "\n" for r in rows))
    schedule = [{"step": i // 2 + 1, "rank": i % 2, "row_index": index,
                 "task_id": rows[index]["task_id"], "state_identity": rows[index]["state_identity"]}
                for i, index in enumerate(selected)]
    (output / "schedule.json").write_text(stable(schedule) + "\n")
    counts = collections.Counter(rows[i]["static_target"] for i in selected)
    manifest = {"schema_version": "recursive_opd_static_gold_v1", "status": "DATA_READY",
                "source": "fixed_clean_official_rl_gold_train_pool",
                "selection_policy_independent": True,
                "protocol_sha256": sha256(PROTOCOL),
                "dataset_sha256": sha256(output / "dataset.jsonl"),
                "schedule_sha256": sha256(output / "schedule.json"),
                "rows": len(rows), "selected_exposures": 384,
                "unique_selected_states": len(selected),
                "selected_input_tokens": sum(rows[i]["total_tokens"] for i in selected),
                "selected_target_tokens": sum(rows[i]["target_tokens"] for i in selected),
                "selected_types": dict(counts), "encoding_skipped": skipped,
                "heldout_overlap": 0}
    (output / "manifest.json").write_text(stable(manifest) + "\n")
    return manifest


if __name__ == "__main__":
    print(stable(build()))
