"""Execution-verified same-state multi-call alignment for saved Cycle-3 traces.

Only a multi-call mismatch enters this slow path. This module deliberately does
not search for arbitrary alternative tool plans or generate new model actions.
"""
from __future__ import annotations

import asyncio
from typing import Any

from repro_1p7b.graph_frontier.cycle3_official_rl.alignment import (
    classify_wrong_arguments, typed_equal,
)
from repro_1p7b.graph_frontier.cycle3_official_rl.semantic_audit import (
    distinct_gold_matches, future_dependency, replay_candidate, schema_errors,
)
from repro_1p7b.graph_frontier.cycle3_official_rl.runtime import IsolatedEpisode
from repro_1p7b.graph_frontier.official_rl_audit_static import decode


def _result(status: str, reason: str, **extra: Any) -> dict:
    return {
        "status": status, "reason": reason, "gold_progress_consumed": 0,
        "state_equivalent": False, "suffix_verified": False,
        "validation_path": "uncertain", **extra,
    }


def _contiguous_progress(calls: list[dict], actions: list[dict], start: int) -> tuple[int, dict[int, int]]:
    matches = distinct_gold_matches(calls, actions, start)
    covered = set(matches.values())
    progress = 0
    while start + progress in covered:
        progress += 1
    return progress, matches


def _failure_from_evidence(calls: list[dict], actions: list[dict], start: int,
                           progress: int, schemas: list[dict], observations: list[Any],
                           query: str, errors: list[dict], execution_errors: list[Any]) -> dict:
    """Classify only a local, evidenced mismatch, never an arbitrary path."""
    target = actions[start + progress] if start + progress < len(actions) else None
    if errors:
        bad = errors[0]
        call = calls[bad["call_position"]]
        if target and call["name"] == target["name"]:
            detail = classify_wrong_arguments(target, call, next(
                (s for s in schemas if s["function"]["name"] == call["name"]), None),
                observations[:start], query) if not typed_equal(call["arguments"], target["arguments"]) else None
            return _result("RECLASSIFY_WRONG_ARGUMENT", "schema_invalid_local_gold_tool",
                           validation_path="execution_failure", argument_detail=detail,
                           schema_errors=errors, execution_errors=execution_errors)
        return _result("EXECUTION_FAILURE", "invalid_tool_schema",
                       validation_path="execution_failure", schema_errors=errors,
                       execution_errors=execution_errors)
    if execution_errors:
        return _result("EXECUTION_FAILURE", "parallel_tool_execution_failed",
                       validation_path="execution_failure", execution_errors=execution_errors)
    if target:
        remainder = [c for c in calls if not typed_equal(c, actions[start])] if progress == 0 else calls
        wrong_arg = next((c for c in remainder if c["name"] == target["name"]
                          and not typed_equal(c["arguments"], target["arguments"])), None)
        if wrong_arg and progress > 0:
            detail = classify_wrong_arguments(target, wrong_arg, next(
                (s for s in schemas if s["function"]["name"] == target["name"]), None),
                observations[:start], query)
            return _result("RECLASSIFY_WRONG_ARGUMENT", "local_gold_tool_wrong_arguments",
                           validation_path="execution_failure", argument_detail=detail)
        wrong_tool = next((c for c in remainder if c["name"] != target["name"]
                           and c["name"] not in {a["name"] for a in actions[start:start + progress]}), None)
        if wrong_tool and progress > 0:
            return _result("RECLASSIFY_WRONG_TOOL", "local_extra_tool_changes_goal_state",
                           validation_path="execution_failure", wrong_tool=wrong_tool["name"])
    return _result("UNCERTAIN", "no_verified_gold_equivalent_progress")


def resolve_evidence(*, calls: list[dict], actions: list[dict], start: int,
                     schemas: list[dict], observations: list[Any], query: str,
                     student_state: Any, gold_progress_states: list[Any],
                     student_results: list[tuple[Any, bool, Any]],
                     gold_progress_responses: list[Any], suffix_verified: dict[int, bool],
                     pre_state: Any, dependency: list[dict] | None = None) -> dict:
    """Pure decision core, also used by CPU-only synthetic regression tests."""
    progress, matches = _contiguous_progress(calls, actions, start)
    errors = [dict(call_position=i, error=err) for i, call in enumerate(calls)
              for err in schema_errors(call, schemas)]
    dependency = dependency if dependency is not None else future_dependency(
        calls, matches, {"actions": actions, "observations": observations,
                         "initial_state": pre_state,
                         "states_after": [pre_state] * start}, start, query)
    if dependency:
        return _result("DEPENDENCY_VIOLATION", "future_observation_not_available_at_shared_state",
                       validation_path="execution_failure", dependency_evidence=dependency)
    execution_errors = [result[2] for result in student_results if result[1] is not True]
    if errors or execution_errors:
        return _failure_from_evidence(calls, actions, start, progress, schemas,
                                      observations, query, errors, execution_errors)
    # A typed state alone is insufficient when read-only gold steps share a
    # state: require exact distinct gold-call identities and typed responses.
    any_state_match = False
    for consumed in range(progress, 0, -1):
        if consumed > len(gold_progress_states) or not typed_equal(
                student_state, gold_progress_states[consumed - 1]):
            continue
        any_state_match = True
        consumed_matches = {gold_i: pos for pos, gold_i in matches.items()
                            if start <= gold_i < start + consumed}
        if len(consumed_matches) != consumed or any(
                not typed_equal(student_results[consumed_matches[start + j]][0],
                                gold_progress_responses[j]) for j in range(consumed)):
            continue
        if not suffix_verified.get(consumed, False):
            continue
        redundant = len(calls) - consumed
        return _result("VALID_PARALLEL", "verified_gold_progress_and_suffix",
                       gold_progress_consumed=consumed, state_equivalent=True,
                       suffix_verified=True, validation_path="state_equivalence",
                       redundant_read_calls=redundant)
    # A nonidentical intermediate state may still be a verified compatible
    # suffix, but only after at least one exact local gold action was taken.
    for consumed in range(progress, 0, -1):
        if suffix_verified.get(consumed, False):
            return _result("VALID_ALTERNATIVE", "verified_gold_compatible_suffix",
                           gold_progress_consumed=consumed, suffix_verified=True,
                           validation_path="suffix_verified")
    if any_state_match:
        return _result("UNCERTAIN", "gold_state_matches_but_response_or_suffix_unverified")
    return _failure_from_evidence(calls, actions, start, progress, schemas,
                                  observations, query, [], [])


async def validate_multicall_transition(record: dict, episode: dict, source_row: dict,
                                        schemas: list[dict], query: str) -> dict:
    """Replay saved calls with the inspected verl/SGLang gather semantics."""
    gold, calls = episode["gold"], record["tool_calls"]
    start = record["gold_step"] - 1
    if len(calls) < 2 or start >= len(gold["actions"]):
        return _result("UNCERTAIN", "not_a_gold_path_multicall")
    progress, matches = _contiguous_progress(calls, gold["actions"], start)
    dependency = future_dependency(calls, matches, gold, start, query)
    if dependency:
        return _result("DEPENDENCY_VIOLATION", "future_observation_not_available_at_shared_state",
                       validation_path="execution_failure", dependency_evidence=dependency)
    errors = [dict(call_position=i, error=err) for i, call in enumerate(calls)
              for err in schema_errors(call, schemas)]
    if errors:
        return _failure_from_evidence(calls, gold["actions"], start, progress,
                                      schemas, gold["observations"], query, errors, [])
    if progress == 0:
        return _result("UNCERTAIN", "no_exact_local_gold_progress_general_path_out_of_scope")
    # Preserve the previously established strict replay implementation for
    # the 125 regression cases; this also tests the original logic verbatim.
    strict = record.get("replay_candidate") is True
    if strict:
        previous = await replay_candidate(record, episode, source_row)
        if previous.get("passed"):
            return _result("VALID_PARALLEL", "strict_replay_candidate_passed",
                           gold_progress_consumed=len(calls), state_equivalent=True,
                           suffix_verified=True, validation_path="state_equivalence",
                           regression_replay=previous, redundant_read_calls=0)
        return _result("UNCERTAIN", "strict_replay_candidate_not_verified",
                       regression_replay=previous)
    if len(calls) > 16:
        return _result("UNCERTAIN", "more_than_16_parallel_calls_not_bounded_for_replay")
    factory = source_row["extra_info"]["mcp_factory_kwargs"]
    servers, initial = decode(factory["mcp_servers"]), decode(factory["initial_config"])
    env = IsolatedEpisode(record["task_id"], servers, initial)
    await env.open()
    try:
        for action in gold["actions"][:start]:
            _, success, _ = await env.call(action["name"], action["arguments"])
            if not success:
                return _result("UNCERTAIN", "shared_prefix_execution_failed")
        before = await env.save()
        expected_before = gold["initial_state"] if start == 0 else gold["states_after"][start - 1]
        if not typed_equal(before, expected_before):
            return _result("UNCERTAIN", "shared_prefix_state_mismatch")
        results = await asyncio.gather(*(env.call(c["name"], c["arguments"]) for c in calls))
        after = await env.save()
    finally:
        await env.close()
    gold_states = gold["states_after"][start:start + progress]
    gold_responses = gold["observations"][start:start + progress]
    # Replay each continuation from the *student* post-call state. The fixed
    # gold suffix is used; no planning or search is attempted.
    suffix = {}
    for consumed in range(1, progress + 1):
        if not all(r[1] is True for r in results):
            break
        continuation = IsolatedEpisode(record["task_id"], servers, after)
        try:
            await continuation.open()
            ok = True
            for action in gold["actions"][start + consumed:]:
                _, success, _ = await continuation.call(action["name"], action["arguments"])
                if not success:
                    ok = False
                    break
            suffix[consumed] = ok and typed_equal(await continuation.save(), gold["final_state"])
        except Exception:
            suffix[consumed] = False
        finally:
            await continuation.close()
    return resolve_evidence(
        calls=calls, actions=gold["actions"], start=start, schemas=schemas,
        observations=gold["observations"], query=query, student_state=after,
        gold_progress_states=gold_states, student_results=results,
        gold_progress_responses=gold_responses, suffix_verified=suffix,
        pre_state=before, dependency=dependency)
