"""Build the evidence-reference CFPB v05.2 blind judge v02 Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from scripts import build_cfpb_v052_semantic_judge_v01_notebook as base
except ImportError:  # Direct execution adds scripts/, not the project root.
    import build_cfpb_v052_semantic_judge_v01_notebook as base


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "notebooks/30_generation/nemo_data_designer/seeded_dialogue/canonical"
    / "FinDisputeEval_CFPB_v052_blind_semantic_judge_v02_attempt01_colab.ipynb"
)
FILES = [
    ROOT / "scripts/generation/nemo_data_designer/cfpb_v052_pipeline_smoke_v02_common.py",
    ROOT / "scripts/generation/nemo_data_designer/prepare_cfpb_seed_v052_pipeline_smoke_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/validate_cfpb_v052_pipeline_smoke_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/run_cfpb_v052_blind_semantic_judge_v01.py",
    ROOT / "scripts/generation/nemo_data_designer/run_cfpb_v052_blind_semantic_judge_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/evaluate_cfpb_v052_semantic_judge_v01.py",
    ROOT / "scripts/generation/nemo_data_designer/freeze_cfpb_v052_semantic_judge_v01.py",
    ROOT / "configs/generation/judges/cfpb_v052_semantic_judge_v02.md",
]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _set_source(cell: dict, value: str) -> None:
    cell["source"] = base.source_lines(value)


def _replace(cell: dict, old: str, new: str) -> None:
    value = _source(cell)
    if old not in value:
        raise ValueError(f"Notebook v02 builder replacement target missing: {old}")
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
        # CFPB Seed v05.2 — blind semantic judge v02 calibration attempt01

        This notebook calls `qwen/qwen3.5-35b-a3b` through OpenRouter in
        non-thinking mode and pins the DeepInfra Zero Data Retention endpoint.
        It replaces the brittle free-text evidence contract used by judge v01.

        The model receives only stable evidence units derived from labels,
        post-redaction grounding, and generated dialogue. It returns reference
        IDs, never evidence text. The runner validates those IDs and resolves
        them back to immutable exact source text before persistence.

        Judge v01 attempts 01-05 remain untouched as development lineage.
        Attempt05 passed its two-row contract and cached 8/20 calibration rows,
        but stopped on a harmless ASCII-versus-curly apostrophe quote mismatch.

        All outputs remain pipeline-smoke-only, `privacy_verified=false`, and
        `benchmark_eligible=false`. Development metrics are not human accuracy.
        """,
    )

    _replace(
        notebook["cells"][2],
        'judges/openrouter_qwen3_5_35b_a3b_v01_attempt05',
        'judges/openrouter_qwen3_5_35b_a3b_v02_attempt01',
    )
    _replace(
        notebook["cells"][2],
        "# The new root preserves the NVIDIA/Mistral attempt and OpenRouter\n"
        "# attempts 01-04 as immutable lineage.",
        "# The new root preserves the NVIDIA/Mistral attempt and all judge-v01\n"
        "# attempts 01-05 as immutable lineage.",
    )
    _replace(
        notebook["cells"][4],
        '"scikit-learn>=1.5,<2",',
        '"scikit-learn>=1.5,<2",\n    "pysbd==0.3.4",',
    )
    _replace(
        notebook["cells"][4],
        '"openai", "pydantic", "pandas", "pyarrow", "scikit-learn"',
        '"openai", "pydantic", "pandas", "pyarrow", "scikit-learn", "pysbd"',
    )

    _replace(
        notebook["cells"][6],
        '"run_cfpb_v052_blind_semantic_judge_v01",\n',
        '"run_cfpb_v052_blind_semantic_judge_v01",\n'
        '    "run_cfpb_v052_blind_semantic_judge_v02",\n',
    )
    _replace(
        notebook["cells"][6],
        "from run_cfpb_v052_blind_semantic_judge_v01 import (\n"
        "    RunnerConfig,\n"
        "    fetch_zdr_endpoint_preflight,\n"
        "    load_blind_cases,\n"
        "    run_judge,\n"
        ")",
        "from run_cfpb_v052_blind_semantic_judge_v01 import (\n"
        "    fetch_zdr_endpoint_preflight,\n"
        "    load_blind_cases,\n"
        ")\n"
        "from run_cfpb_v052_blind_semantic_judge_v02 import (\n"
        "    RunnerConfig,\n"
        "    build_evidence_units,\n"
        "    run_judge,\n"
        ")",
    )
    _replace(
        notebook["cells"][6],
        'PROMPT = SNAPSHOT_ROOT / "cfpb_v052_semantic_judge_v01.md"',
        'PROMPT = SNAPSHOT_ROOT / "cfpb_v052_semantic_judge_v02.md"',
    )
    _replace(
        notebook["cells"][6],
        'RUNNER_SOURCE = SNAPSHOT_ROOT / "run_cfpb_v052_blind_semantic_judge_v01.py"',
        'RUNNER_SOURCE = SNAPSHOT_ROOT / "run_cfpb_v052_blind_semantic_judge_v02.py"',
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

        CONFIG = RunnerConfig(
            model_name="qwen/qwen3.5-35b-a3b",
            base_url="https://openrouter.ai/api/v1",
            judge_id=(
                "openrouter_qwen3_5_35b_a3b_deepinfra_zdr_"
                "nonthinking_blind_v02_attempt01"
            ),
            sampling_profile="qwen3.5_official_instruct_general",
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            max_tokens=2048,
            provider_slug="deepinfra",
            allow_fallbacks=False,
            require_parameters=True,
            data_collection="deny",
            zdr=True,
            reasoning_enabled=False,
            reasoning_exclude=True,
            seed=20260722,
            max_attempts=2,
            request_timeout_seconds=120.0,
        )
        print(CONFIG.model_dump())

        endpoint_preflight = fetch_zdr_endpoint_preflight(
            config=CONFIG,
            output_path=ENDPOINT_PREFLIGHT,
        )
        print({
            "zdr_endpoint_preflight": "passed",
            "provider": endpoint_preflight.selected_endpoint.provider_name,
            "tag": endpoint_preflight.selected_endpoint.tag,
            "status": endpoint_preflight.selected_endpoint.status,
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
        ## 4. Two-row reference-contract smoke

        These fixed development IDs exercise one previously accepted row and one
        semantic-reject row. Expected labels are not sent or loaded. The model may
        return only reference IDs that exist in the supplied evidence-unit table.
        A second call is permitted only for a non-empty JSON/schema/reference
        contract failure.
        """,
    )
    _replace(
        notebook["cells"][13],
        "## 6. Score only after all blind judgments exist",
        "## 6. Score v02 only after all blind judgments exist",
    )
    _replace(
        notebook["cells"][14],
        '"calibration_metrics_v01.json"',
        '"calibration_metrics_v02.json"',
    )
    _replace(
        notebook["cells"][15],
        "## 7. Explicitly freeze judge v01",
        "## 7. Explicitly freeze judge v02",
    )
    _replace(
        notebook["cells"][16],
        '"calibration_metrics_v01.json"',
        '"calibration_metrics_v02.json"',
    )
    _replace(
        notebook["cells"][16],
        "cfpb_v052_semantic_judge_v01_attempt05.json",
        "cfpb_v052_semantic_judge_v02_attempt01.json",
    )
    _replace(
        notebook["cells"][17],
        "Do not revise the frozen prompt, reason definitions, model, inference",
        "Do not revise the frozen v02 prompt, reference protocol, model, inference",
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
