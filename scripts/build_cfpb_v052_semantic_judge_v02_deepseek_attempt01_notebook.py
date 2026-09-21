"""Build the DeepSeek V4 Flash 0731 semantic-judge model ablation notebook."""

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
    / "FinDisputeEval_CFPB_v052_blind_semantic_judge_v02_deepseek_v4_flash_0731_attempt01_colab.ipynb"
)
FILES = [
    ROOT / "scripts/generation/nemo_data_designer/cfpb_v052_pipeline_smoke_v02_common.py",
    ROOT / "scripts/generation/nemo_data_designer/prepare_cfpb_seed_v052_pipeline_smoke_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/validate_cfpb_v052_pipeline_smoke_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/run_cfpb_v052_blind_semantic_judge_v01.py",
    ROOT / "scripts/generation/nemo_data_designer/run_cfpb_v052_blind_semantic_judge_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/run_cfpb_v052_blind_semantic_judge_v03.py",
    ROOT / "scripts/generation/nemo_data_designer/evaluate_cfpb_v052_semantic_judge_v01.py",
    ROOT / "configs/generation/judges/cfpb_v052_semantic_judge_v02.md",
]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _set_source(cell: dict, value: str) -> None:
    cell["source"] = base.base.source_lines(value)


def _replace(cell: dict, old: str, new: str) -> None:
    value = _source(cell)
    if old not in value:
        raise ValueError(f"DeepSeek notebook replacement target missing: {old}")
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
        # CFPB Seed v05.2 blind semantic judge: DeepSeek model ablation attempt01

        This notebook tests the exact pinned model
        `deepseek/deepseek-v4-flash-0731` through OpenRouter. It is a clean
        model-only comparison with Qwen v02 attempt01: the conservative v02
        rubric, immutable 20 cases, evidence-reference schema, validator, oracle,
        and bounded retry policy stay fixed.

        The DeepSeek-specific runner uses official non-thinking chat sampling
        (`temperature=1.0`, `top_p=1.0`) and does not send Qwen-only `top_k`,
        `min_p`, presence-penalty, repetition-penalty, or seed parameters. Before
        any paid call, it selects one live endpoint from OpenRouter's public ZDR
        inventory, pins its exact provider tag, and hash-binds the snapshot.

        This is development evidence only. The GPT-5.6-SOL oracle is not human
        gold, all source rows remain pipeline-smoke-only, `privacy_verified=false`,
        and `benchmark_eligible=false`. This notebook cannot freeze a judge.
        """,
    )

    _replace(
        notebook["cells"][2],
        "judges/openrouter_qwen3_5_35b_a3b_v02_attempt01",
        "judges/openrouter_deepseek_v4_flash_0731_v02_attempt01",
    )
    _replace(
        notebook["cells"][2],
        "# The new root preserves the NVIDIA/Mistral attempt and all judge-v01\n"
        "# attempts 01-05 as immutable lineage.",
        "# This independent root preserves all Qwen and NVIDIA/Mistral attempts\n"
        "# as immutable lineage; no existing cache or report is reused.",
    )
    _replace(
        notebook["cells"][2],
        'ORACLE = REVALIDATION_ROOT / "oracle/gpt56sol_adjudication_20_v01.csv"',
        'ORACLE = REVALIDATION_ROOT / "oracle/gpt56sol_adjudication_20_v02.csv"\n'
        'QWEN_ATTEMPT01_METRICS = (\n'
        '    REVALIDATION_ROOT\n'
        '    / "judges/openrouter_qwen3_5_35b_a3b_v02_attempt01"\n'
        '    / "calibration_20/calibration_metrics_v02.json"\n'
        ')',
    )

    _replace(
        notebook["cells"][6],
        '    "run_cfpb_v052_blind_semantic_judge_v02",\n',
        '    "run_cfpb_v052_blind_semantic_judge_v02",\n'
        '    "run_cfpb_v052_blind_semantic_judge_v03",\n',
    )
    _replace(
        notebook["cells"][6],
        '    "freeze_cfpb_v052_semantic_judge_v01",\n',
        "",
    )
    _replace(
        notebook["cells"][6],
        "from run_cfpb_v052_blind_semantic_judge_v01 import (\n"
        "    fetch_zdr_endpoint_preflight,\n"
        "    load_blind_cases,\n"
        ")\n"
        "from run_cfpb_v052_blind_semantic_judge_v02 import (\n"
        "    RunnerConfig,\n"
        "    build_evidence_units,\n"
        "    run_judge,\n"
        ")",
        "from run_cfpb_v052_blind_semantic_judge_v01 import load_blind_cases\n"
        "from run_cfpb_v052_blind_semantic_judge_v02 import build_evidence_units\n"
        "from run_cfpb_v052_blind_semantic_judge_v03 import (\n"
        "    RunnerConfig,\n"
        "    resolve_zdr_endpoint_preflight,\n"
        "    run_judge,\n"
        ")",
    )
    _replace(
        notebook["cells"][6],
        "from freeze_cfpb_v052_semantic_judge_v01 import freeze_judge\n",
        "",
    )
    _replace(
        notebook["cells"][6],
        'RUNNER_SOURCE = SNAPSHOT_ROOT / "run_cfpb_v052_blind_semantic_judge_v02.py"',
        'RUNNER_SOURCE = SNAPSHOT_ROOT / "run_cfpb_v052_blind_semantic_judge_v03.py"',
    )

    _set_source(
        notebook["cells"][8],
        """
        from getpass import getpass
        import pandas as pd

        cases = load_blind_cases(RAW, PREPARED)
        assert len(cases) == 20
        assert all(
            set(case.model_dump()) == {"seed_id", "labels", "grounding", "generated"}
            for case in cases
        )
        preview_units = build_evidence_units(cases[0])
        print({
            "rows": len(cases),
            "fields_visible_to_judge": ["case_id", "evidence_units"],
            "evidence_protocol": "stable_reference_ids_v01",
            "oracle_loaded_by_runner": False,
            "candidate_judgments_loaded_by_runner": False,
            "prompt_sha256": sha256_file(PROMPT),
            "first_case_evidence_units": len(preview_units),
        })
        display(pd.DataFrame([
            {
                "ref_id": unit.ref_id,
                "source": unit.source,
                "location": unit.location,
                "text_preview": unit.text[:180],
            }
            for unit in preview_units[:10]
        ]))

        CONFIG_TEMPLATE = RunnerConfig(
            model_name="deepseek/deepseek-v4-flash-0731",
            base_url="https://openrouter.ai/api/v1",
            judge_id=(
                "openrouter_deepseek_v4_flash_0731_zdr_"
                "nonthinking_blind_v02_attempt01"
            ),
            sampling_profile="deepseek_v4_flash_0731_official_chat",
            temperature=1.0,
            top_p=1.0,
            max_tokens=2048,
            provider_slug="auto",
            allow_fallbacks=False,
            require_parameters=True,
            data_collection="deny",
            zdr=True,
            reasoning_enabled=False,
            reasoning_exclude=True,
            max_attempts=2,
            request_timeout_seconds=120.0,
        )
        CONFIG, endpoint_preflight = resolve_zdr_endpoint_preflight(
            config=CONFIG_TEMPLATE,
            output_path=ENDPOINT_PREFLIGHT,
        )
        print(CONFIG.model_dump())
        print({
            "zdr_endpoint_preflight": "passed",
            "model": endpoint_preflight.model_id,
            "provider": endpoint_preflight.selected_endpoint.provider_name,
            "tag": endpoint_preflight.selected_endpoint.tag,
            "status": endpoint_preflight.selected_endpoint.status,
            "eligible_endpoint_count": endpoint_preflight.eligible_endpoint_count,
            "supported_parameters": endpoint_preflight.selected_endpoint.supported_parameters,
        })

        if not os.environ.get("OPENROUTER_API_KEY"):
            os.environ["OPENROUTER_API_KEY"] = getpass(
                "OPENROUTER_API_KEY (not persisted): "
            )
        if not os.environ["OPENROUTER_API_KEY"].strip():
            raise RuntimeError("OPENROUTER_API_KEY is empty")
        """,
    )

    _set_source(
        notebook["cells"][9],
        """
        ## 4. Two-row DeepSeek API/schema contract smoke

        Run this before the full calibration. It proves routing, non-thinking
        response content, strict JSON schema, and evidence-reference resolution.
        A second call is allowed only after a non-empty response fails the
        JSON/schema/reference contract; provider errors and empty content stop.
        """,
    )
    _set_source(
        notebook["cells"][13],
        """
        ## 6. Score the fixed model ablation

        This is the first stage that loads the versioned GPT-5.6-SOL development
        oracle v02. It measures development agreement, not human accuracy. The
        same conservative rubric allows direct comparison with Qwen attempt01.
        """,
    )

    score_source = _source(notebook["cells"][14])
    score_source += """

# Direct model-only comparison when the synced Qwen attempt01 metrics exist.
if QWEN_ATTEMPT01_METRICS.is_file():
    qwen_metrics = json.loads(
        QWEN_ATTEMPT01_METRICS.read_text(encoding="utf-8")
    )["judge_only_metrics"]
    deepseek_metrics = calibration_metrics["judge_only_metrics"]
    comparison_names = [
        "decision_accuracy",
        "decision_reject_precision",
        "decision_reject_recall",
        "decision_reject_f1",
        "exact_set_accuracy",
        "micro_precision",
        "micro_recall",
        "micro_f1",
    ]
    display(pd.DataFrame([
        {
            "metric": name,
            "qwen_attempt01": float(qwen_metrics[name]),
            "deepseek_attempt01": float(deepseek_metrics[name]),
            "delta": float(deepseek_metrics[name]) - float(qwen_metrics[name]),
        }
        for name in comparison_names
    ]))
else:
    print({"qwen_comparison_skipped_missing": str(QWEN_ATTEMPT01_METRICS)})
"""
    _set_source(notebook["cells"][14], score_source)

    _set_source(
        notebook["cells"][15],
        """
        ## 7. Preserve results; do not freeze from this notebook

        This development set has informed model and rubric choices, and its
        oracle is LLM-produced rather than human gold. Even excellent metrics
        are evidence for selecting the next experiment, not freeze approval.
        """,
    )
    _set_source(
        notebook["cells"][16],
        """
        ZERO_RECALL_CODES_TO_INSPECT = [
            "claim_type_changed",
            "factual_status_changed",
            "unsupported_scenario_detail",
            "indirect_sensitive_information_guidance",
        ]
        print({
            "frozen": False,
            "freeze_supported_by_this_notebook": False,
            "next_review_focus": ZERO_RECALL_CODES_TO_INSPECT,
            "development_only": True,
            "human_gold": False,
            "privacy_verified": False,
            "benchmark_eligible": False,
        })
        """,
    )
    _set_source(
        notebook["cells"][17],
        """
        ## Stop point

        Preserve the complete DeepSeek attempt as independent lineage and sync
        its output directory back to the local project. Compare it with Qwen
        attempt01 before deciding whether to run the attempt02 high-recall rubric
        on DeepSeek. Do not edit this notebook in place, do not freeze from these
        20 development rows, and do not start a formal pilot while the privacy
        release gate remains closed.
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
