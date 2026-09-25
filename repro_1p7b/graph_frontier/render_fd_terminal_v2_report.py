"""Render the v2 data/quality audit with verbatim ten-sample appendix."""
from __future__ import annotations

import json
from pathlib import Path

from repro_1p7b.graph_frontier.gold_terminal_builder import MANUAL_REJECT

ROOT = Path(__file__).resolve().parents[2]
GF = ROOT / "repro_1p7b/graph_frontier"
LOG = ROOT / "repro_1p7b/logs/first_divergence_terminal_v2"
SAMPLE_IDS = [
    "gf-rich-bb1c09eff616c479b72f",
    "gf-rich-51549205bfc988a9c805",
    "gf-rich-16250871ff92972a670f",
    "gf-rich-45e04e18655582d8fc92",
    "gf-rich-1b59497ce2bf2b133272",
    "gf-rich-9ec81b3b07cc5fdc5b47",
    "gf-rich-4e5587f0e4422d242c03",
    "gf-rich-4a15e6d4c5090c77537d",
    "gf-rich-0a06d52ed385e6019e82",
    "gf-rich-7372af308535cceb8e11",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def render() -> Path:
    source = LOG / "terminal_dataset_attempt3"
    final = LOG / "terminal_dataset_attempt4"
    frozen = LOG / "frozen_ce_v2"
    source_rows = {r["task_id"]: r for r in (
        json.loads(line) for line in (source / "dataset.jsonl").read_text().splitlines()
    )}
    final_rows = {r["task_id"]: r for r in (
        json.loads(line) for line in (final / "dataset.jsonl").read_text().splitlines()
    )}
    manifest = read_json(final / "manifest.json")
    lock = read_json(frozen / "manifest.json")
    if len(SAMPLE_IDS) != 10 or len(set(SAMPLE_IDS)) != 10 or not set(SAMPLE_IDS) <= set(source_rows):
        raise RuntimeError("ten-sample audit IDs changed")
    if set(SAMPLE_IDS) & set(read_json(
        ROOT / "repro_1p7b/logs/first_divergence_onpolicy_v1/cycle2_heldout_plan.json"
    )["task_ids"]):
        raise RuntimeError("manual sample audit includes heldout")
    counts = manifest["counts"]
    parts = [
        "# First-Divergence + Gold-Terminal Supervision v2",
        "",
        "## Executive Summary",
        "",
        "CPU data gate: **DATA_READY**. The strict 197 reference trajectories produced "
        "95 replay/verifier-consistent terminal candidates. Nine frozen Rich-v3 "
        "heldout tasks were removed before training. Of the remaining 86, two "
        "manually identified unsupported final answers, six MCP-schema source "
        "drifts, and three observed tool-error trajectories were excluded. "
        "Final Type B = **75**; existing locked Type A = **50**. This is "
        "structural supervision, not a claim of fully verified natural-language semantics.",
        "",
        "## Terminal dataset construction and schema recovery",
        "",
        f"- Strict references: {counts['strict_reference']}; strict replay/verifier intersection: {counts['strict_candidates']}.",
        f"- Heldout exclusion: {counts['heldout_candidate_excluded']}; manual quality exclusion: {counts['manual_quality_excluded']}.",
        f"- Candidate training tasks before quality/schema gate: {counts['strict_train_candidates']}; final usable: {counts['final_terminal_usable']}.",
        f"- Persisted same-task schema: {counts['schema_available_original']}; deterministic MCP schema recovery: {counts['schema_recovered_deterministically']}; unrecoverable: {counts['schema_unrecoverable']}; source drift: {counts['schema_source_drift']}.",
        "- MCP definitions were queried with the original server identities; missing parameters were never inferred from tool-call text. Each recovered schema stores the source tool-definition SHA256. Persisted schemas were compared to the current definitions and mismatches skipped.",
        "",
        "## Target quality audit",
        "",
        f"- Tool error response skipped: {counts['quality_tool_error_observed']}. Manual unsupported-answer skips: 2.",
        "- Automatic checks require a nonempty original assistant final, no placeholder/serialization residue, strict typed replay and final-state match, and Qwen3 chat-template encoding. All retained samples have quality_status=structurally_valid_unverified_semantics.",
        "- Generic final string “The requested operations were completed successfully.” occurs in 40/75 retained samples; this and unverified answer groundedness are material training risks.",
        "- Manual ten-sample verbatim inspection appears below. Two answers assert details unavailable in their tool observations and are excluded before model training. No external LLM judge or new STOP token was used.",
        "",
        "## Frozen datasets and SHA256",
        "",
        f"- Type A: 50 existing FD examples; dataset SHA256 {lock['fd_source_dataset_sha256']}.",
        f"- Type B: 75 terminal examples; source JSONL SHA256 {manifest['dataset_sha256']}; encoded CE JSONL SHA256 {lock['terminal_ce_dataset_sha256']}.",
        f"- Frozen manifest: {frozen / 'manifest.json'}; tokenizer.json SHA256 {lock['tokenizer_json_sha256']}.",
        f"- Rich heldout plan SHA256 {lock['heldout_plan_sha256']}; heldout overlap 0; Frozen300 overlap 0.",
        "",
        "## Unified trainer and CPU tests",
        "",
        "- Both targets use the same Qwen3 tokenizer/chat template and standard assistant-target CE. All user, historical assistant, and tool observation tokens have labels=-100; only the new assistant suffix contributes to loss. Type A appends a structured tool_call; Type B appends the original final natural-language response.",
        "- Deterministic 1:1 / approximately 1:2 exposure schedules use all frozen rows without physical duplication. Both models initialize from original Dynamic-v1; 16-step smoke includes the longest terminal context (11,529 tokens).",
        "- Scoped CPU tests: 20 PASS. No formal training or evaluation result should be inferred from this data audit alone.",
        "",
        "## Training, evaluation and supervision balance",
        "",
        "16-step smoke, 64-step RUN-A/RUN-B, Frozen300 and Rich-v3 evaluations: pending gated execution.",
        "",
        "| Model | FD:Terminal | Frozen success | Dep edge | Rich reward | Rich tool events |",
        "|---|---:|---:|---:|---:|---:|",
        "| Dynamic-v1 | 0:0 | 20/300 | 94/300 | 0.18611 (T=0.7) | 31 (T=0.7) |",
        "| Cycle-3 | 1:0 | 105/300 | 200/300 | 0.13750 (T=0.7) | 101 (T=0.7) |",
        "| v2-A | 1:1 | pending | pending | pending | pending |",
        "| v2-B | 1:2 | pending | pending | pending | pending |",
        "",
        "These baseline metrics are protocol-locked historical results, not newly rerun in this CPU phase. "
        "The T=0 diagnostic baseline is Dynamic-v1 0.18403/28 tool events and Cycle-3 0.11354/115 tool events.",
        "",
        "## Ten real terminal samples (verbatim source fields)",
        "",
        "These ten IDs were selected before manual exclusion across depth 1/2/3 and distinct environments. "
        "The original query, final tool observation, and original assistant final below are not rewritten. "
        "The two rejected rows are retained only in the diagnostic attempt3 audit source, not in the frozen training dataset.",
        "",
    ]
    for index, task_id in enumerate(SAMPLE_IDS, 1):
        row = source_rows[task_id]
        retained = task_id in final_rows
        query = row["conversation_prefix"][0]["content"]
        observation = row["final_tool_observation"]
        final_text = row["gold_final_response"]
        if not isinstance(observation, str):
            observation = json.dumps(observation, ensure_ascii=False, sort_keys=True, indent=2)
        parts.extend([
            f"### {index}. {task_id} — depth {row['gold_depth']}; environment {row['env_id']}",
            "",
            f"Quality decision: {'retained, semantics unverified' if retained else 'excluded: ' + MANUAL_REJECT[task_id]}.",
            "",
            "Original query:",
            "",
            "~~~text",
            query,
            "~~~",
            "",
            "Last tool observation:",
            "",
            "~~~text",
            observation,
            "~~~",
            "",
            "Original gold final assistant response:",
            "",
            "~~~text",
            final_text,
            "~~~",
            "",
        ])
    parts.extend([
        "## Limitations and current verdict",
        "",
        "Gold terminal states are reference-policy, not student on-policy; existing strict "
        "student-perfect terminal prefixes are 0/192. The 75 retained final answers pass "
        "structural checks, not independent full semantic verification. 40 generic answers "
        "could bias the model toward early generic termination. Six current MCP schema "
        "definitions differ from historical persisted schemas and were conservatively skipped. "
        "The previous failed/diagnostic dataset attempts are never training inputs.",
        "",
        "**Current stage: DATA_READY, experiment verdict pending smoke and matched evaluations.**",
        "",
    ])
    output = GF / "reports/first_divergence_terminal_v2.md"
    if output.exists() and "## Frozen300 detailed audit" in output.read_text(encoding="utf-8"):
        # The CPU audit renderer must not clobber the finalized experiment report.
        return output
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(render())
