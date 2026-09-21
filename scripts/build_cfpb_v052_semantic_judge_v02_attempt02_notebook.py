"""Build the rubric-recall CFPB v05.2 blind judge v02 attempt02 notebook."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from scripts import build_cfpb_v052_semantic_judge_v02_notebook as base
except ImportError:  # Direct execution adds scripts/, not the project root.
    import build_cfpb_v052_semantic_judge_v02_notebook as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "notebooks/30_generation/nemo_data_designer/seeded_dialogue/canonical"
    / "FinDisputeEval_CFPB_v052_blind_semantic_judge_v02_attempt02_colab.ipynb"
)
FILES = [
    ROOT / "scripts/generation/nemo_data_designer/cfpb_v052_pipeline_smoke_v02_common.py",
    ROOT / "scripts/generation/nemo_data_designer/prepare_cfpb_seed_v052_pipeline_smoke_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/validate_cfpb_v052_pipeline_smoke_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/run_cfpb_v052_blind_semantic_judge_v01.py",
    ROOT / "scripts/generation/nemo_data_designer/run_cfpb_v052_blind_semantic_judge_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/evaluate_cfpb_v052_semantic_judge_v01.py",
    ROOT / "scripts/generation/nemo_data_designer/freeze_cfpb_v052_semantic_judge_v01.py",
    ROOT / "configs/generation/judges/cfpb_v052_semantic_judge_v02_attempt02.md",
]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _set_source(cell: dict, value: str) -> None:
    cell["source"] = base.base.source_lines(value)


def _replace(cell: dict, old: str, new: str) -> None:
    value = _source(cell)
    if old not in value:
        raise ValueError(f"Notebook attempt02 replacement target missing: {old}")
    _set_source(cell, value.replace(old, new))


def build_notebook() -> dict:
    original_files, original_output = base.FILES, base.OUTPUT
    try:
        base.FILES = FILES
        base.OUTPUT = OUTPUT
        notebook = base.build_notebook()
    finally:
        base.FILES = original_files
        base.OUTPUT = original_output

    _set_source(
        notebook["cells"][0],
        """
        # CFPB Seed v05.2 — blind semantic judge v02 calibration attempt02

        This is a controlled rubric-recall ablation. It keeps the attempt01
        model, DeepInfra ZDR endpoint, non-thinking sampling, stable evidence
        references, strict schema, and bounded retry policy unchanged.

        Attempt01 completed 20/20 technically, but reached only 0.2917
        reason-level recall and 0.4000 exact reason-set accuracy against the
        development oracle. Attempt02 changes only the rubric: it requires an
        exhaustive internal reason checklist, distinguishes product policy from
        procedure and scenario facts, and no longer permits stopping after the
        first sufficient reject reason.

        Twelve attempt01 disagreements received an independent second GPT-5.6-SOL
        development pass. Eleven were confirmed and one reason set was amended.
        That review remains LLM evidence, not human gold. All outputs remain
        pipeline-smoke-only, `privacy_verified=false`, and
        `benchmark_eligible=false`.
        """,
    )
    _replace(
        notebook["cells"][2],
        "judges/openrouter_qwen3_5_35b_a3b_v02_attempt01",
        "judges/openrouter_qwen3_5_35b_a3b_v02_attempt02",
    )
    _replace(
        notebook["cells"][2],
        "# The new root preserves the NVIDIA/Mistral attempt and all judge-v01\n"
        "# attempts 01-05 as immutable lineage.",
        "# This independent root preserves all earlier attempts, including the\n"
        "# completed but calibration-rejected v02 attempt01, as immutable lineage.",
    )
    _replace(
        notebook["cells"][2],
        'ORACLE = REVALIDATION_ROOT / "oracle/gpt56sol_adjudication_20_v01.csv"',
        'ORACLE = REVALIDATION_ROOT / "oracle/gpt56sol_adjudication_20_v02.csv"\n'
        'READJUDICATION_REPORT = (\n'
        '    REVALIDATION_ROOT\n'
        '    / "oracle/gpt56sol_disagreement_readjudication_12_v02_report.json"\n'
        ')',
    )
    _replace(
        notebook["cells"][2],
        "for path in (RAW, PREPARED, INPUT_MANIFEST, PRIVACY_DISPOSITION, ORACLE, FROZEN_SOURCE_RUN):",
        "for path in (\n"
        "    RAW, PREPARED, INPUT_MANIFEST, PRIVACY_DISPOSITION, ORACLE,\n"
        "    READJUDICATION_REPORT, FROZEN_SOURCE_RUN,\n"
        "):",
    )
    _replace(
        notebook["cells"][6],
        'PROMPT = SNAPSHOT_ROOT / "cfpb_v052_semantic_judge_v02.md"',
        'PROMPT = SNAPSHOT_ROOT / "cfpb_v052_semantic_judge_v02_attempt02.md"',
    )
    _replace(
        notebook["cells"][8],
        "nonthinking_blind_v02_attempt01",
        "nonthinking_blind_v02_attempt02",
    )
    _replace(
        notebook["cells"][13],
        "This is the first cell that loads the GPT-5.6-SOL development\n"
        "adjudication. It measures agreement, not human accuracy.",
        "This is the first cell that loads the versioned GPT-5.6-SOL development\n"
        "adjudication v02. It measures agreement, not human accuracy. The v02\n"
        "oracle differs from v01 by one independently reviewed reason addition.",
    )

    score_source = _source(notebook["cells"][14])
    score_source += """

# Automatic development gates are necessary but not sufficient for freezing.
# They require attempt02 to exceed attempt01 and roughly match or exceed the
# earlier candidate-prefill development baseline. Human review remains separate.
FREEZE_THRESHOLDS = {
    "decision_reject_precision": 0.95,
    "decision_reject_recall": 0.80,
    "decision_reject_f1": 0.85,
    "exact_set_accuracy": 0.65,
    "micro_precision": 0.75,
    "micro_recall": 0.60,
    "micro_f1": 0.70,
}
judge_metrics = calibration_metrics["judge_only_metrics"]
FREEZE_METRIC_CHECKS = {
    name: float(judge_metrics[name]) >= threshold
    for name, threshold in FREEZE_THRESHOLDS.items()
}
AUTO_FREEZE_METRICS_PASSED = all(FREEZE_METRIC_CHECKS.values())
display(pd.DataFrame([
    {
        "metric": name,
        "observed": float(judge_metrics[name]),
        "threshold": FREEZE_THRESHOLDS[name],
        "passed": passed,
    }
    for name, passed in FREEZE_METRIC_CHECKS.items()
]))
print({"automatic_freeze_metrics_passed": AUTO_FREEZE_METRICS_PASSED})
"""
    _set_source(notebook["cells"][14], score_source)

    _set_source(
        notebook["cells"][15],
        """
        ## 7. Apply the explicit attempt02 freeze gate

        Freezing requires all automatic metric thresholds, a real human review
        of the development disagreements, an explicit calibration-review flag,
        and a separate approval flag with an approver ID. The independent
        GPT-5.6-SOL second pass is not a substitute for the human-review flag.
        """,
    )
    _set_source(
        notebook["cells"][16],
        """
        DEVELOPMENT_DISAGREEMENTS_REVIEWED_BY_HUMAN = False
        CALIBRATION_REVIEW_COMPLETED = False
        APPROVE_FREEZE = False
        APPROVER_ID = ""

        freeze_blockers = []
        if not AUTO_FREEZE_METRICS_PASSED:
            freeze_blockers.append("automatic metric thresholds failed")
        if not DEVELOPMENT_DISAGREEMENTS_REVIEWED_BY_HUMAN:
            freeze_blockers.append("development disagreements lack human review")
        if not CALIBRATION_REVIEW_COMPLETED:
            freeze_blockers.append("calibration review is not marked complete")
        if not APPROVE_FREEZE:
            freeze_blockers.append("freeze approval is false")
        if not APPROVER_ID.strip():
            freeze_blockers.append("approver ID is empty")

        if freeze_blockers:
            print({"frozen": False, "blockers": freeze_blockers})
        else:
            frozen = freeze_judge(
                judge_report_path=CALIBRATION_ROOT / "judge_run_report.json",
                judgments_path=CALIBRATION_ROOT / "semantic_judgments.jsonl",
                evaluation_path=EVALUATED / "validation_report_v02.json",
                metrics_path=CALIBRATION_ROOT / "calibration_metrics_v02.json",
                prompt_path=PROMPT,
                oracle_path=ORACLE,
                raw_path=RAW,
                prepared_path=PREPARED,
                runner_path=RUNNER_SOURCE,
                common_path=COMMON_SOURCE,
                validator_path=VALIDATOR_SOURCE,
                endpoint_preflight_path=ENDPOINT_PREFLIGHT,
                output_path=(
                    REVALIDATION_ROOT
                    / "judges/frozen/cfpb_v052_semantic_judge_v02_attempt02.json"
                ),
                approved_by=APPROVER_ID,
                approve_freeze=True,
            )
            print(json.dumps(frozen.model_dump(mode="json"), ensure_ascii=False, indent=2))
        """,
    )
    _set_source(
        notebook["cells"][17],
        """
        ## Stop point

        If any automatic or human gate is blocked, preserve attempt02 as
        development lineage and do not run a held-out smoke. If the judge is
        genuinely frozen, the next notebook must generate a new held-out
        pipeline-only smoke and apply the frozen manifest without calibration on
        held-out answers. A finite human audit follows that fixed evaluation.
        Formal 50–100-row generation remains blocked until a separately approved
        privacy protocol makes `benchmark_release_gate_passed=true`.
        """,
    )
    notebook["metadata"]["colab"]["name"] = OUTPUT.name
    return notebook


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
