"""Freeze the calibrated semantic judge before any held-out smoke is evaluated."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

try:
    from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
        SEMANTIC_REASON_CODES,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct execution or embedded Colab runtime.
    from cfpb_v052_pipeline_smoke_v02_common import (  # type: ignore[no-redef]
        SEMANTIC_REASON_CODES,
        sha256_file,
    )


_PROJECT_ROOT_ENV = os.environ.get("FINDISPUTEEVAL_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_PROJECT_ROOT_ENV).resolve()
    if _PROJECT_ROOT_ENV
    else Path(__file__).resolve().parents[3]
)
SMOKE_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override"
SOURCE_RUN_ROOT = SMOKE_ROOT / "run_20260722T135306Z"
ROOT = SMOKE_ROOT / "revalidations/validator_v02/source_run_20260722T135306Z"
CALIBRATION_ROOT = (
    ROOT / "judges/openrouter_qwen3_5_35b_a3b_v01_attempt03/calibration_20"
)
DEFAULT_JUDGE_REPORT = CALIBRATION_ROOT / "judge_run_report.json"
DEFAULT_JUDGMENTS = CALIBRATION_ROOT / "semantic_judgments.jsonl"
DEFAULT_EVALUATION = CALIBRATION_ROOT / "evaluated/validation_report_v02.json"
DEFAULT_METRICS = CALIBRATION_ROOT / "calibration_metrics_v01.json"
DEFAULT_PROMPT = PROJECT_ROOT / "configs/generation/judges/cfpb_v052_semantic_judge_v01.md"
DEFAULT_ORACLE = ROOT / "oracle/gpt56sol_adjudication_20_v01.csv"
DEFAULT_RAW = SOURCE_RUN_ROOT / "raw/data_designer/dataset/parquet-files/batch_00000.parquet"
DEFAULT_PREPARED = SOURCE_RUN_ROOT / "prepared_inputs/nemo_seed_v052_pipeline_smoke_20.jsonl"
DEFAULT_ENDPOINT_PREFLIGHT = (
    ROOT
    / "judges/openrouter_qwen3_5_35b_a3b_v01_attempt03/zdr_endpoint_preflight.json"
)
DEFAULT_OUTPUT = ROOT / "judges/frozen/cfpb_v052_semantic_judge_v01_attempt03.json"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundFile(StrictModel):
    role: str
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class JudgeFreezeManifest(StrictModel):
    manifest_version: Literal["v01"] = "v01"
    judge_version: Literal["cfpb_v052_semantic_judge_v01"] = (
        "cfpb_v052_semantic_judge_v01"
    )
    status: Literal["development_frozen_pending_held_out_evaluation"] = (
        "development_frozen_pending_held_out_evaluation"
    )
    frozen_utc: str
    approved_by: str = Field(min_length=1)
    model_name: str
    judge_id: str
    base_url: str
    inference_config: dict[str, Any]
    allowed_reason_codes: list[str]
    development_metrics: dict[str, Any]
    bound_files: list[BoundFile]
    policy: dict[str, bool]
    note: str


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def bind(role: str, path: Path) -> BoundFile:
    if not path.is_file():
        raise FileNotFoundError(path)
    return BoundFile(
        role=role,
        path=display_path(path),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def freeze_judge(
    *,
    judge_report_path: Path,
    judgments_path: Path,
    evaluation_path: Path,
    metrics_path: Path,
    prompt_path: Path,
    oracle_path: Path,
    raw_path: Path,
    prepared_path: Path,
    runner_path: Path,
    common_path: Path,
    validator_path: Path,
    endpoint_preflight_path: Path,
    output_path: Path,
    approved_by: str,
    approve_freeze: bool,
) -> JudgeFreezeManifest:
    if not approve_freeze:
        raise ValueError(
            "Freeze requires explicit approve_freeze=True after reviewing calibration metrics"
        )
    if not approved_by.strip():
        raise ValueError("approved_by is required")

    judge_report = json.loads(judge_report_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    calibration_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if judge_report.get("semantic_coverage_complete") is not True:
        raise ValueError("Judge calibration coverage is incomplete")
    if judge_report.get("expected_rows") != 20 or judge_report.get("judgment_rows") != 20:
        raise ValueError("Judge freeze requires the complete 20-row development set")
    blind = judge_report.get("blind_input_contract", {})
    if blind.get("loaded_oracle") is not False or blind.get("loaded_candidate_judgments") is not False:
        raise ValueError("Judge report does not prove blind input isolation")
    if evaluation.get("semantic_coverage_complete") is not True:
        raise ValueError("Validator evaluation lacks complete semantic coverage")
    combined_metrics = evaluation.get("oracle_metrics")
    if not isinstance(combined_metrics, dict) or combined_metrics.get("rows") != 20:
        raise ValueError("Complete 20-row development metrics are required")
    if calibration_metrics.get("rows") != 20:
        raise ValueError("Calibration metrics must cover all 20 development rows")
    if calibration_metrics.get("combined_validator_metrics") != combined_metrics:
        raise ValueError("Calibration and validator metrics disagree")
    if not isinstance(calibration_metrics.get("judge_only_metrics"), dict):
        raise ValueError("Judge-only development metrics are required")

    config = judge_report.get("config", {})
    required_config = {
        "model_name",
        "base_url",
        "judge_id",
        "sampling_profile",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "max_tokens",
        "provider_slug",
        "allow_fallbacks",
        "require_parameters",
        "data_collection",
        "zdr",
        "reasoning_enabled",
        "reasoning_exclude",
        "seed",
        "max_attempts",
        "request_timeout_seconds",
    }
    missing = sorted(required_config - set(config))
    if missing:
        raise ValueError(f"Judge report lacks frozen configuration: {missing}")
    if config["allow_fallbacks"] is not False:
        raise ValueError("Frozen judge must disable provider fallbacks")
    if config["require_parameters"] is not True:
        raise ValueError("Frozen judge must require all requested parameters")
    if config["data_collection"] != "deny" or config["zdr"] is not True:
        raise ValueError("Frozen judge must enforce deny-data-collection and ZDR")
    expected_identity = {
        "model_name": "qwen/qwen3.5-35b-a3b",
        "provider_slug": "deepinfra",
        "judge_id": (
            "openrouter_qwen3_5_35b_a3b_deepinfra_zdr_"
            "nonthinking_blind_v01_attempt03"
        ),
    }
    actual_identity = {key: config.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise ValueError("Frozen attempt03 judge identity differs from the reviewed candidate")
    expected_nonthinking = {
        "sampling_profile": "qwen3.5_official_instruct_general",
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "reasoning_enabled": False,
        "reasoning_exclude": True,
    }
    actual_nonthinking = {
        key: config.get(key) for key in expected_nonthinking
    }
    if actual_nonthinking != expected_nonthinking:
        raise ValueError(
            "Frozen attempt03 must use the reviewed Qwen3.5 official "
            "non-thinking sampling profile"
        )
    report_preflight_sha = judge_report.get("inputs", {}).get(
        "endpoint_preflight_sha256"
    )
    if report_preflight_sha != sha256_file(endpoint_preflight_path):
        raise ValueError("Endpoint preflight differs from the completed judge run")

    files = [
        bind("rubric_prompt", prompt_path),
        bind("runner_source", runner_path),
        bind("schema_source", common_path),
        bind("validator_source", validator_path),
        bind("provider_endpoint_preflight", endpoint_preflight_path),
        bind("development_raw", raw_path),
        bind("development_prepared", prepared_path),
        bind("development_oracle_not_human_gold", oracle_path),
        bind("development_judgments", judgments_path),
        bind("judge_run_report", judge_report_path),
        bind("development_evaluation", evaluation_path),
        bind("development_calibration_metrics", metrics_path),
    ]
    failed_attempts = judge_report.get("outputs", {}).get("failed_attempts", {})
    if failed_attempts.get("rows", 0):
        failed_path = Path(str(failed_attempts.get("path", "")))
        files.append(bind("provider_failed_attempt_audit", failed_path))
    manifest = JudgeFreezeManifest(
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        approved_by=approved_by.strip(),
        model_name=str(config["model_name"]),
        judge_id=str(config["judge_id"]),
        base_url=str(config["base_url"]),
        inference_config={key: config[key] for key in sorted(required_config)},
        allowed_reason_codes=sorted(SEMANTIC_REASON_CODES),
        development_metrics={
            "judge_only": calibration_metrics["judge_only_metrics"],
            "combined_validator": combined_metrics,
        },
        bound_files=files,
        policy={
            "development_calibration_only": True,
            "held_out_evaluated": False,
            "human_gold": False,
            "privacy_verified": False,
            "benchmark_eligible": False,
            "formal_pilot_allowed": False,
        },
        note=(
            "This freezes the judge before held-out evaluation. Development metrics "
            "measure agreement with GPT-5.6-SOL adjudication, not human accuracy. "
            "Do not alter prompt, model, parameters, schemas, or bound code for the "
            "held-out smoke; any alteration creates a new judge version."
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def verify_judge_freeze(manifest_path: Path) -> dict[str, Any]:
    manifest = JudgeFreezeManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    missing: list[str] = []
    changed: list[str] = []
    for item in manifest.bound_files:
        candidate = PROJECT_ROOT / item.path
        if not candidate.is_file():
            # Embedded Colab source files are recorded by basename. Verification
            # outside that runtime can validate all persistent project files.
            if "/" not in item.path:
                continue
            missing.append(item.path)
            continue
        if candidate.stat().st_size != item.size_bytes or sha256_file(candidate) != item.sha256:
            changed.append(item.path)
    result = {
        "verified": not (missing or changed),
        "judge_version": manifest.judge_version,
        "status": manifest.status,
        "missing": missing,
        "changed": changed,
    }
    if not result["verified"]:
        raise ValueError(f"Judge freeze verification failed: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-report", type=Path, default=DEFAULT_JUDGE_REPORT)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_cfpb_v052_blind_semantic_judge_v01.py"),
    )
    parser.add_argument(
        "--common",
        type=Path,
        default=Path(__file__).with_name("cfpb_v052_pipeline_smoke_v02_common.py"),
    )
    parser.add_argument(
        "--validator",
        type=Path,
        default=Path(__file__).with_name("validate_cfpb_v052_pipeline_smoke_v02.py"),
    )
    parser.add_argument(
        "--endpoint-preflight",
        type=Path,
        default=DEFAULT_ENDPOINT_PREFLIGHT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approve-freeze", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_existing:
        print(json.dumps(verify_judge_freeze(args.output), ensure_ascii=False, indent=2))
        return
    manifest = freeze_judge(
        judge_report_path=args.judge_report,
        judgments_path=args.judgments,
        evaluation_path=args.evaluation,
        metrics_path=args.metrics,
        prompt_path=args.prompt,
        oracle_path=args.oracle,
        raw_path=args.raw,
        prepared_path=args.prepared,
        runner_path=args.runner,
        common_path=args.common,
        validator_path=args.validator,
        endpoint_preflight_path=args.endpoint_preflight,
        output_path=args.output,
        approved_by=args.approved_by,
        approve_freeze=args.approve_freeze,
    )
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
