"""Finalize CPU-only machine handoff after D0, retrospective and token audits."""
from __future__ import annotations

import json
import subprocess

from transformers import AutoTokenizer

from repro_1p7b.graph_frontier.cycle3_official_rl.audit import schema_for
from repro_1p7b.graph_frontier.cycle3_official_rl.plan import sha256, stable
from repro_1p7b.graph_frontier.recursive_opd_v1.protocol import MODEL, RUN as V1_RUN
from repro_1p7b.graph_frontier.recursive_opd_v1.rollout import GOLD_RUN
from repro_1p7b.graph_frontier.recursive_opd_v2.prepare import REPORT, ROOT, RUN


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT.parents[1], text=True).strip()


def blocked_lengths(task_id: str) -> dict:
    episode = json.loads((V1_RUN / "cycle0/rollouts" / f"{task_id}.json").read_text())
    student = episode["student"]
    step = student["steps"][-1]
    count = step["prompt_message_count"]
    if (count != len(student["messages"]) - 1 or student["messages"][count] != step["model_message"]
            or student["messages"][count - 1].get("role") != "tool"):
        raise RuntimeError("blocked sample is not an exact pre-stop student prefix")
    prefix = student["messages"][:count]
    schema = schema_for(V1_RUN if episode.get("gold_source_kind") == "independent_replay_cache"
                        else GOLD_RUN, episode)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    kwargs = {"tools": schema, "tokenize": True, "enable_thinking": False}
    prompt = tokenizer.apply_chat_template(prefix, add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(prefix + [{"role": "assistant", "content": ""}],
                                         add_generation_prompt=False, **kwargs)
    return {"task_id": task_id, "prompt_tokens": len(prompt), "full_tokens": len(full),
            "limit_tokens": 16384, "suffix_tokens": len(full) - len(prompt),
            "student_prefix_messages": count}


def create() -> dict:
    path = REPORT / "recursive_opd_v2_handoff_final.json"
    if path.exists():
        raise RuntimeError("refusing to overwrite V2 handoff")
    prep_path = REPORT / "recursive_opd_v2_cpu_prep.json"
    audit_path = REPORT / "recursive_opd_v2_token_audit.json"
    prep, audit = json.loads(prep_path.read_text()), json.loads(audit_path.read_text())
    d0 = prep["d0"]
    if (d0["dataset_sha256"] != sha256(RUN / "d0/dataset.jsonl")
            or audit["candidate_dataset_sha256"] != d0["dataset_sha256"]
            or audit["status"] != "AUDITED"):
        raise RuntimeError("handoff input audit drift")
    command = ["/home/u2024311031/.conda/envs/envfactory_repro_1p7b/bin/python", "-m", "pytest", "-q",
               "repro_1p7b/graph_frontier/tests/test_recursive_opd_v2.py",
               "repro_1p7b/graph_frontier/tests/test_recursive_opd_v1.py"]
    tests = subprocess.run(command, cwd=ROOT.parents[1], text=True, capture_output=True, check=False)
    if tests.returncode:
        raise RuntimeError(f"CPU tests failed: {tests.stdout[-1000:]} {tests.stderr[-1000:]}")
    blocked = [blocked_lengths(row["task_id"]) for row in d0["encoding_failures"]]
    verdict = "READY_FOR_GPU_TRAINING" if not blocked and all(
        value == "PASS" for value in d0["gates"].values()) else "NOT_READY"
    if verdict != prep["verdict"] or verdict != audit["verdict"]:
        raise RuntimeError("V2 prep/audit/handoff verdict discrepancy")
    result = {"schema_version": "recursive_opd_v2_cpu_handoff_1", "verdict": verdict,
              "repo": {"path": str(ROOT.parents[1]), "branch": git("branch", "--show-current"),
                       "head": git("rev-parse", "HEAD"), "status_short": git("status", "--short")},
              "d0": {"eligible": d0["eligible_task_count"], "encoded": d0["sample_count"],
                     "classes_encoded": d0["class_counts"], "tool_targets_encoded": d0["tool_targets"],
                     "eos_targets_encoded": d0["eos_targets"],
                     "natural_eos_fraction_if_complete": d0["natural_eos_fraction"],
                     "dataset_sha256": d0["dataset_sha256"],
                     "rollout_manifest_sha256": d0["source_rollout_manifest_sha256"],
                     "pi0_model_sha256": d0["source_pi0_model_sha256"],
                     "gold_source_sha256": d0["gold_source_sha256"],
                     "split_sha256": d0["task_split_sha256"],
                     "tokenizer_sha256": d0["tokenizer_sha256"]},
              "retrospective": prep["retrospective"],
              "gates": {**d0["gates"], **audit["gates"]},
              "blocked_exact_prefixes": blocked,
              "cpu_tests": {"command": " ".join(command), "returncode": tests.returncode,
                            "summary": tests.stdout.strip().splitlines()[-1]},
              "cpu_token_audit": {"saved_verdicts": audit["all_saved_verdicts_checked"],
                                  "checked_prefixes": audit["execution_verified_prefixes_checked"],
                                  "all_encoded_masks": audit["d0_rows_full_mask_checked"],
                                  "random_reencoded": audit["random_reencoded"]},
              "gpu_used": False,
              "gpu_training_authorized": verdict == "READY_FOR_GPU_TRAINING",
              "unapproved_gpu_parameters": prep["unapproved_gpu_parameters"],
              "not_run": prep["not_run"],
              "cpu_prep_sha256": sha256(prep_path), "token_audit_sha256": sha256(audit_path)}
    path.write_text(stable(result) + "\n")
    return result


if __name__ == "__main__":
    result = create()
    print(stable({"verdict": result["verdict"], "blocked": result["blocked_exact_prefixes"],
                  "cpu_tests": result["cpu_tests"]["summary"]}), flush=True)
