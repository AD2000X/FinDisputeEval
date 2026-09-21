"""Evaluate blind judge-only and combined-validator development agreement."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
        load_oracle_csv,
        load_semantic_judgments,
        sha256_file,
    )
    from scripts.generation.nemo_data_designer.validate_cfpb_v052_pipeline_smoke_v02 import (
        evaluate_against_oracle,
    )
except ModuleNotFoundError:  # Direct execution or embedded Colab runtime.
    from cfpb_v052_pipeline_smoke_v02_common import (  # type: ignore[no-redef]
        load_oracle_csv,
        load_semantic_judgments,
        sha256_file,
    )
    from validate_cfpb_v052_pipeline_smoke_v02 import (  # type: ignore[no-redef]
        evaluate_against_oracle,
    )


_PROJECT_ROOT_ENV = os.environ.get("FINDISPUTEEVAL_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_PROJECT_ROOT_ENV).resolve()
    if _PROJECT_ROOT_ENV
    else Path(__file__).resolve().parents[3]
)
ROOT = (
    PROJECT_ROOT
    / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override"
    / "revalidations/validator_v02/source_run_20260722T135306Z"
)
CALIBRATION_ROOT = (
    ROOT / "judges/openrouter_qwen3_5_35b_a3b_v01_attempt03/calibration_20"
)
DEFAULT_JUDGMENTS = CALIBRATION_ROOT / "semantic_judgments.jsonl"
DEFAULT_ORACLE = ROOT / "oracle/gpt56sol_adjudication_20_v01.csv"
DEFAULT_COMBINED = CALIBRATION_ROOT / "evaluated/validation_report_v02.json"
DEFAULT_OUTPUT = CALIBRATION_ROOT / "calibration_metrics_v01.json"


def evaluate_judge(
    *,
    judgments_path: Path,
    oracle_path: Path,
    combined_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    judgments = load_semantic_judgments(judgments_path)
    oracle = load_oracle_csv(oracle_path, require_complete=True)
    if set(judgments) != set(oracle):
        raise ValueError("Judge and oracle coverage differ")
    predicted_reasons = {
        seed_id: set(judgment.reasons) for seed_id, judgment in judgments.items()
    }
    predicted_routes = {
        seed_id: {
            "accept": "validated",
            "reject": "rejected",
            "review": "review",
        }[judgment.decision]
        for seed_id, judgment in judgments.items()
    }
    judge_only = evaluate_against_oracle(
        predicted_reasons, oracle, predicted_routes=predicted_routes
    )
    combined_report = json.loads(combined_report_path.read_text(encoding="utf-8"))
    combined_metrics = combined_report.get("oracle_metrics")
    if not isinstance(combined_metrics, dict) or combined_metrics.get("rows") != len(oracle):
        raise ValueError("Combined validator report lacks complete oracle metrics")
    report = {
        "report_version": "v01",
        "purpose": "blind_llm_judge_development_calibration_metrics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(oracle),
        "judge_only_metrics": judge_only,
        "combined_validator_metrics": combined_metrics,
        "inputs": {
            "judgments_sha256": sha256_file(judgments_path),
            "oracle_sha256": sha256_file(oracle_path),
            "combined_validation_report_sha256": sha256_file(combined_report_path),
        },
        "policy": {
            "development_calibration_only": True,
            "human_gold": False,
            "held_out": False,
            "privacy_verified": False,
            "benchmark_eligible": False,
        },
        "note": (
            "Judge-only metrics exclude deterministic validator findings. Combined "
            "metrics measure the fixed deterministic validator plus the LLM judgment. "
            "Both compare against GPT-5.6-SOL development adjudication, not human gold."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--combined-report", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_judge(
        judgments_path=args.judgments,
        oracle_path=args.oracle,
        combined_report_path=args.combined_report,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
