"""CPU-only attribution of V2 on-policy context length and original pi0 input.

No server is started. This uses saved model-response usage, the original
tokenizer/chat template, V1 execution-verified verdicts and persisted episodes.
"""
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.official_rl_audit_static import SOURCE, OUT
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, PROTOCOL, RUN as V1_RUN, check
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import (
    REPORT, RUN, checked_prefix, load_cycle, load_verified_episode,
)

LIMIT = 16384
BLOCKED_TASK = "official-rl-0281-627654eb4348"
THRESHOLDS = (8192, 12288, 16384, 24576, 32768)
KINDS = ("START", "ADVANCE", "TERMINATE", "SUCCESS_TERMINAL_POSITIVE")


def context_class(total_tokens: int, limit: int = LIMIT) -> str:
    if not isinstance(total_tokens, int) or total_tokens < 1 or limit < 1:
        raise ValueError("invalid context length/limit")
    return "CONTEXT_COMPATIBLE" if total_tokens <= limit else "CONTEXT_OVERFLOW"


def account_context(*, semantic_eligible: int, encoded: int, overflow: list[dict],
                    limit: int = LIMIT) -> None:
    if (semantic_eligible < 0 or encoded < 0 or encoded + len(overflow) != semantic_eligible
            or len({row["task_id"] for row in overflow}) != len(overflow)
            or any(row.get("semantic_eligible") is not True
                   or row.get("execution_verified") is not True
                   or row.get("train_context_compatible") is not False
                   or row.get("reason") != "CONTEXT_OVERFLOW"
                   or row.get("full_tokens", 0) <= limit for row in overflow)):
        raise RuntimeError("semantic eligible / compatible / overflow accounting invalid")


def percentile(sorted_values: list[int], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("empty length population")
    at = (len(sorted_values) - 1) * fraction
    low, high = math.floor(at), math.ceil(at)
    return round(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (at - low), 2)


def distribution(lengths: list[int]) -> dict:
    ordered = sorted(lengths)
    if not ordered:
        return {"count": 0}
    return {"count": len(ordered), "min": ordered[0], "p50": percentile(ordered, .5),
            "p90": percentile(ordered, .9), "p95": percentile(ordered, .95),
            "p99": percentile(ordered, .99), "max": ordered[-1],
            "above": {str(threshold): sum(length > threshold for length in ordered)
                      for threshold in THRESHOLDS}}


def original_server_config() -> dict:
    base = V1_RUN / "cycle0"
    values = []
    for gpu in (0, 1):
        path = base / f"server_gpu{gpu}_full.log"
        line = next((line for line in path.read_text(errors="replace").splitlines()
                     if "server_args=ServerArgs(" in line), None)
        if line is None:
            raise RuntimeError(f"original pi0 server_args missing: {path}")
        def field(name: str) -> str:
            match = re.search(rf"\b{name}=([^,)]*)", line)
            if match is None:
                raise RuntimeError(f"pi0 server arg {name} missing: {path}")
            return match.group(1)
        values.append({"gpu": gpu, "log": str(path), "log_sha256": sha256(path),
                       "context_length": int(field("context_length")),
                       "max_prefill_tokens": int(field("max_prefill_tokens")),
                       "chunked_prefill_size": int(field("chunked_prefill_size")),
                       "allow_auto_truncate": field("allow_auto_truncate")})
    if any({k: item[k] for k in ("context_length", "max_prefill_tokens",
                                  "chunked_prefill_size", "allow_auto_truncate")}
           != {k: values[0][k] for k in ("context_length", "max_prefill_tokens",
                                            "chunked_prefill_size", "allow_auto_truncate")}
           for item in values[1:]):
        raise RuntimeError("original pi0 server context config differs across GPUs")
    return {"per_server": values, "declared_context_length": values[0]["context_length"],
            "allow_auto_truncate": values[0]["allow_auto_truncate"],
            "max_prefill_tokens": values[0]["max_prefill_tokens"],
            "chunked_prefill_size": values[0]["chunked_prefill_size"],
            "launcher": str(V1_RUN.parent / "recursive_opd_v1/run_cycle.sh"),
            "launcher_sha256": sha256(V1_RUN.parent / "recursive_opd_v1/run_cycle.sh")}


def target_message(record: dict) -> dict:
    if record["status"] == "SUCCESS" or record.get("error_regime") == "TERMINATE":
        return {"role": "assistant", "content": ""}
    action = record["gold_next_action"]
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_0", "type": "function",
        "function": {"name": action["name"], "arguments": action["arguments"]},
    }]}


def sample_lengths(tokenizer, prefix: list[dict], schema: list[dict], record: dict) -> tuple[int, int]:
    kwargs = {"tools": schema, "tokenize": True, "enable_thinking": False}
    prompt = tokenizer.apply_chat_template(prefix, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(prefix + [target_message(record)],
                                         add_generation_prompt=False, **kwargs)
    if not full[:len(prompt)] == prompt or len(full) <= len(prompt):
        raise RuntimeError("target is not the exact prompt suffix")
    return len(prompt), len(full)


def attribution(tokenizer, prefix: list[dict], schema: list[dict], usage: dict) -> dict:
    def size(messages: list[dict], *, tools: bool = True, generation: bool = True) -> int:
        kwargs = {"tokenize": True, "enable_thinking": False,
                  "add_generation_prompt": generation}
        if tools:
            kwargs["tools"] = schema
        return len(tokenizer.apply_chat_template(messages, **kwargs))
    empty_user = [{"role": "user", "content": ""}]
    empty_no_tools = size(empty_user, tools=False)
    empty_with_tools = size(empty_user)
    cumulative = [size(prefix[:i]) for i in range(1, len(prefix) + 1)]
    decision_prompt = cumulative[-1] - size(prefix, generation=False)
    pieces = {"system_and_tool_schema_marginal": empty_with_tools - empty_no_tools,
              "user_query_marginal": cumulative[0] - empty_with_tools,
              "historical_assistant_marginal": sum(cumulative[i] - cumulative[i-1]
                                                  for i in range(1, len(prefix))
                                                  if prefix[i].get("role") == "assistant"),
              "tool_observation_marginal": sum(cumulative[i] - cumulative[i-1]
                                               for i in range(1, len(prefix))
                                               if prefix[i].get("role") == "tool"),
              "final_decision_prompt_marginal": decision_prompt,
              "other_template_scaffold": empty_no_tools - decision_prompt}
    if sum(pieces.values()) != cumulative[-1]:
        raise RuntimeError("token attribution does not sum to local prompt")
    observations = [{"message_index": i, "content_chars": len(str(prefix[i].get("content") or "")),
                     "marginal_tokens": cumulative[i] - cumulative[i - 1]}
                    for i in range(1, len(prefix)) if prefix[i].get("role") == "tool"]
    return {"method": "sequential chat-template marginal token lengths; boundary effects belong to the added segment",
            "local_prompt_tokens": cumulative[-1], "server_reported_prompt_tokens": usage.get("prompt_tokens"),
            "server_minus_local_tokens": usage.get("prompt_tokens", 0) - cumulative[-1],
            "components": pieces, "cumulative_by_message": cumulative,
            "tool_observations": observations,
            "raw_message_roles": [m.get("role") for m in prefix]}


def run() -> dict:
    from transformers import AutoTokenizer

    protocol = check()
    records, manifest, hashes = load_cycle(0, protocol)
    by_id = {entry["audit_id"]: entry for entry in json.loads((OUT / "opd_source_manifest.json").read_text())["rows"]}
    source = json.loads(SOURCE.read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    if tokenizer.eos_token_id != 151645 or sha256(MODEL / "tokenizer.json") != protocol["tokenizer_sha256"]:
        raise RuntimeError("original tokenizer identity drift")
    server = original_server_config()
    model_config = json.loads((MODEL / "config.json").read_text())
    tokenizer_config = json.loads((MODEL / "tokenizer_config.json").read_text())
    groups = collections.defaultdict(list)
    overflow = []
    case = None
    compared_usage = collections.Counter()
    for record in records:
        if record["pool"] != "train" or record["status"] == "UNCERTAIN":
            continue
        episode, schema, gold_sha = load_verified_episode(record, manifest,
                                                           hashes["manifest_sha256"], source, by_id)
        prefix = checked_prefix(record, episode)
        prompt_len, full_len = sample_lengths(tokenizer, prefix, schema, record)
        kind = record["error_regime"] if record["status"] == "FIRST_ERROR" else "SUCCESS_TERMINAL_POSITIVE"
        groups["ALL"].append(full_len)
        groups[kind].append(full_len)
        step = (episode["student"]["steps"][-1] if record["status"] == "SUCCESS"
                else episode["student"]["steps"][record["student_step"]])
        usage = step.get("usage") or {}
        if isinstance(usage.get("prompt_tokens"), int):
            compared_usage[usage["prompt_tokens"] - prompt_len] += 1
        if context_class(full_len) == "CONTEXT_OVERFLOW":
            overflow.append({"task_id": record["task_id"], "query_hash": record["query_hash"],
                             "kind": kind, "semantic_eligible": True, "execution_verified": True,
                             "train_context_compatible": False, "reason": "CONTEXT_OVERFLOW",
                             "prompt_tokens_local": prompt_len, "full_tokens": full_len,
                             "server_reported_prompt_tokens": usage.get("prompt_tokens"),
                             "gold_depth": record["gold_tool_count"]})
        if record["task_id"] == BLOCKED_TASK:
            case = {"task_id": record["task_id"], "query_hash": record["query_hash"],
                    "cycle": 0, "source_policy": "original Dynamic-v1 pi0",
                    "source_model_sha256": manifest["model_sha256"],
                    "reason": record["reason"], "status": record["status"],
                    "gold_depth": record["gold_tool_count"],
                    "gold_tool_names": [item["name"] for item in episode["gold"]["actions"]],
                    "servers": sorted({item["name"].split("-", 1)[0] for item in episode["gold"]["actions"]}),
                    "saved_rollout_path": str(V1_RUN / "cycle0/rollouts" / f"{record['task_id']}.json"),
                    "saved_rollout_sha256": sha256(V1_RUN / "cycle0/rollouts" / f"{record['task_id']}.json"),
                    "raw_conversation_message_count": len(episode["student"]["messages"]),
                    "pre_stop_student_message_count": len(prefix),
                    "final_assistant_content_chars_excluded": len(str(episode["student"]["messages"][-1].get("content") or "")),
                    "local_prompt_tokens": prompt_len, "local_encoded_total_tokens": full_len,
                    "server_reported_prompt_tokens": usage.get("prompt_tokens"),
                    "server_reported_completion_tokens": usage.get("completion_tokens"),
                    "server_reported_total_tokens": usage.get("total_tokens"),
                    "finish_reason": step.get("finish_reason"),
                    "token_attribution": attribution(tokenizer, prefix, schema, usage),
                    "first_request_prompt_tokens_server": episode["student"]["steps"][0]["usage"].get("prompt_tokens"),
                    "first_request_prompt_tokens_local": len(tokenizer.apply_chat_template(
                        prefix[:1], tools=schema, tokenize=True, enable_thinking=False,
                        add_generation_prompt=True)),
                    "gold_source_episode_sha256": gold_sha}
    if case is None or not groups["ALL"]:
        raise RuntimeError("blocked task or eligible D0 missing")
    lengths = {kind: distribution(groups[kind]) for kind in ("ALL", *KINDS)}
    if len(groups["ALL"]) != 1115 or len(overflow) != 1 or overflow[0]["task_id"] != BLOCKED_TASK:
        # A systematic long-context tail is a new protocol decision, not an
        # automatic change to 32k or a relaxed gate.
        decision = "TRAIN_CONTEXT_PROTOCOL_DECISION"
    else:
        decision = "SINGLE_CONTEXT_OVERFLOW"
    effective_tokens = case["server_reported_prompt_tokens"]
    no_auto = server["allow_auto_truncate"] == "False"
    under_service_cap = (effective_tokens + case["server_reported_completion_tokens"]
                         <= server["declared_context_length"])
    server_local_delta_stable = (case["first_request_prompt_tokens_server"]
                                 - case["first_request_prompt_tokens_local"]
                                 == case["token_attribution"]["server_minus_local_tokens"])
    if (isinstance(effective_tokens, int) and effective_tokens > LIMIT and no_auto
            and under_service_cap and server_local_delta_stable
            and case["finish_reason"] == "stop"):
        inference_case, truncation = "B", "NO"
    else:
        inference_case, truncation = "C", "UNKNOWN"
    result = {"schema_version": "recursive_opd_v2_context_forensics_1",
              "task": case, "inference_case": inference_case,
              "server_side_truncation": truncation,
              "effective_input_tokens": effective_tokens if inference_case == "B" else "UNKNOWN",
              "exact_effective_input_token_ids_available": False,
              "service_and_model_context": {
                  "server": server,
                  "request_max_new_tokens": manifest["config"]["max_tokens_per_turn"],
                  "model_max_position_embeddings": model_config.get("max_position_embeddings"),
                  "model_sliding_window": model_config.get("sliding_window"),
                  "tokenizer_config_model_max_length": tokenizer_config.get("model_max_length"),
                  "tokenizer_object_model_max_length": tokenizer.model_max_length,
                  "tokenizer_sha256": sha256(MODEL / "tokenizer.json"),
                  "chat_template_sha256": sha256(MODEL / "chat_template.jinja"),
                  "rollout_manifest_sha256": hashes["manifest_sha256"],
                  "client_source": str(V1_RUN.parent / "recursive_opd_v1/rollout.py"),
                  "client_source_sha256": sha256(V1_RUN.parent / "recursive_opd_v1/rollout.py"),
                  "request_source": str(V1_RUN.parent / "collect_preference_rollouts.py"),
                  "request_source_sha256": sha256(V1_RUN.parent / "collect_preference_rollouts.py"),
              },
              "length_distribution": lengths,
              "semantic_eligible_count": len(groups["ALL"]),
              "context_compatible_count": len(groups["ALL"]) - len(overflow),
              "context_overflow_count": len(overflow),
              "context_overflow_records": overflow,
              "usage_minus_local_prompt_token_histogram": dict(compared_usage),
              "protocol_decision": decision,
              "case_b_evidence": {"server_reported_prompt_over_16k": effective_tokens > LIMIT,
                                  "allow_auto_truncate_disabled": no_auto,
                                  "prompt_plus_completion_within_service_context": under_service_cap,
                                  "same_server_local_delta_on_first_and_last_request": server_local_delta_stable}}
    audit_dir = RUN / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    paths = {"case": audit_dir / "long_context_case.json",
             "lengths": audit_dir / "length_distribution.json",
             "summary": REPORT / "recursive_opd_v2_context_forensics.json"}
    if any(path.exists() for path in paths.values()):
        raise RuntimeError("refusing to overwrite V2 context forensic artifacts")
    paths["case"].write_text(stable({"task": case, "inference_case": inference_case,
                                     "server_side_truncation": truncation,
                                     "service_and_model_context": result["service_and_model_context"]}) + "\n")
    paths["lengths"].write_text(stable({"length_distribution": lengths,
                                        "context_overflow_records": overflow,
                                        "semantic_eligible_count": len(groups["ALL"])}) + "\n")
    paths["summary"].write_text(stable(result) + "\n")
    return result


if __name__ == "__main__":
    report = run()
    print(stable({"inference_case": report["inference_case"],
                  "server_side_truncation": report["server_side_truncation"],
                  "eligible": report["semantic_eligible_count"],
                  "compatible": report["context_compatible_count"],
                  "overflow": report["context_overflow_count"],
                  "lengths": report["length_distribution"]["ALL"]}), flush=True)
