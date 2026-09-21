"""Build the self-contained CFPB v05.2 blind semantic-judge Colab notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "notebooks/30_generation/nemo_data_designer/seeded_dialogue/canonical"
    / "FinDisputeEval_CFPB_v052_blind_semantic_judge_v01_attempt05_colab.ipynb"
)
FILES = [
    ROOT / "scripts/generation/nemo_data_designer/cfpb_v052_pipeline_smoke_v02_common.py",
    ROOT / "scripts/generation/nemo_data_designer/prepare_cfpb_seed_v052_pipeline_smoke_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/validate_cfpb_v052_pipeline_smoke_v02.py",
    ROOT / "scripts/generation/nemo_data_designer/run_cfpb_v052_blind_semantic_judge_v01.py",
    ROOT / "scripts/generation/nemo_data_designer/evaluate_cfpb_v052_semantic_judge_v01.py",
    ROOT / "scripts/generation/nemo_data_designer/freeze_cfpb_v052_semantic_judge_v01.py",
    ROOT / "configs/generation/judges/cfpb_v052_semantic_judge_v01.md",
]


def source_lines(value: str) -> list[str]:
    return (dedent(value).strip("\n") + "\n").splitlines(keepends=True)


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(value)}


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(value),
    }


def embedded_payload() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in FILES:
        raw = path.read_bytes()
        result[path.name] = {
            "payload": base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return result


def build_notebook() -> dict:
    embedded = json.dumps(embedded_payload(), sort_keys=True)
    cells = [
        markdown(
            """
            # CFPB Seed v05.2 — blind semantic judge v01 calibration attempt05

            This notebook calls `qwen/qwen3.5-35b-a3b` through OpenRouter in
            non-thinking mode and pins the DeepInfra Zero Data Retention endpoint
            to judge the immutable 20-row pipeline smoke. It is attempt05. The
            failed thinking-mode attempt01, unroutable Parasail attempt02,
            attempt03 label-evidence failure, and attempt04 non-contiguous quote
            failure remain untouched as lineage.

            Attempt05 retains canonical `key: value` labels and requires every
            grounding/generated quote to be one contiguous exact substring. It
            permits one correction retry only after a non-empty response violates
            JSON, schema, or evidence validation. Provider errors, timeouts, and
            empty responses are not blindly retried.

            The judge sees only labels, the post-redaction grounding excerpt, and
            the generated dialogue. Candidate findings and the GPT-5.6-SOL
            development adjudication are not loaded until the later scoring cell.

            All outputs remain pipeline-smoke-only, `privacy_verified=false`, and
            `benchmark_eligible=false`. Development metrics are not human accuracy.
            """
        ),
        markdown("## 0. Mount Drive and bind the immutable development inputs"),
        code(
            """
            from pathlib import Path
            import json, os, sys, time

            IN_COLAB = "google.colab" in sys.modules
            if IN_COLAB:
                from google.colab import drive
                DRIVE_MOUNT = Path("/content/drive")
                MY_DRIVE = DRIVE_MOUNT / "MyDrive"
                if MY_DRIVE.is_dir():
                    print(f"Reusing mounted Google Drive: {MY_DRIVE}")
                else:
                    mount_error = None
                    for mount_attempt in range(1, 3):
                        try:
                            drive.mount(str(DRIVE_MOUNT), timeout_ms=180000)
                            mount_error = None
                            break
                        except Exception as exc:
                            mount_error = exc
                            print(
                                f"Google Drive mount attempt {mount_attempt}/2 failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            if mount_attempt < 2:
                                time.sleep(3)
                    if mount_error is not None or not MY_DRIVE.is_dir():
                        raise RuntimeError(
                            "Google Drive authentication did not reach the Colab runtime. "
                            "No project file was accessed. In VS Code, disconnect the Colab "
                            "runtime, reconnect it, confirm the browser is signed into the "
                            "Google account that owns MyDrive/FinDisputeEval, and rerun this "
                            "cell. If Drive is already mounted in another notebook, close that "
                            "session first."
                        ) from mount_error
                PROJECT_ROOT = MY_DRIVE / "FinDisputeEval"
                if not PROJECT_ROOT.is_dir():
                    raise FileNotFoundError(
                        f"Drive mounted, but the project directory is missing: {PROJECT_ROOT}"
                    )
            else:
                here = Path.cwd().resolve()
                PROJECT_ROOT = next(
                    (p for p in (here, *here.parents) if (p / "WORK_PROGRESS.md").exists()),
                    None,
                )
                if PROJECT_ROOT is None:
                    raise FileNotFoundError("Open the FinDisputeEval repository in VS Code")

            os.environ["FINDISPUTEEVAL_PROJECT_ROOT"] = str(PROJECT_ROOT)
            SMOKE_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override"
            SOURCE_RUN = SMOKE_ROOT / "run_20260722T135306Z"
            REVALIDATION_ROOT = SMOKE_ROOT / "revalidations/validator_v02/source_run_20260722T135306Z"
            JUDGE_ROOT = REVALIDATION_ROOT / "judges/openrouter_qwen3_5_35b_a3b_v01_attempt05"
            # The new root preserves the NVIDIA/Mistral attempt and OpenRouter
            # attempts 01-04 as immutable lineage.
            CONTRACT_ROOT = JUDGE_ROOT / "contract_smoke_2"
            CALIBRATION_ROOT = JUDGE_ROOT / "calibration_20"
            SNAPSHOT_ROOT = CALIBRATION_ROOT / "source_snapshot"
            ENDPOINT_PREFLIGHT = JUDGE_ROOT / "zdr_endpoint_preflight.json"
            RAW = SOURCE_RUN / "raw/data_designer/dataset/parquet-files/batch_00000.parquet"
            PREPARED = SOURCE_RUN / "prepared_inputs/nemo_seed_v052_pipeline_smoke_20.jsonl"
            INPUT_MANIFEST = SOURCE_RUN / "prepared_inputs/pipeline_smoke_input_manifest.json"
            PRIVACY_DISPOSITION = (
                PROJECT_ROOT / "dataset/curated/annotations/cfpb_seed_v05_audit"
                / "run_20260713T145423Z/privacy_qa/full_v052_v02"
                / "pipeline_privacy_disposition_v01.json"
            )
            ORACLE = REVALIDATION_ROOT / "oracle/gpt56sol_adjudication_20_v01.csv"
            FROZEN_SOURCE_RUN = SMOKE_ROOT / "frozen_manifests/run_20260722T135306Z_files_v01.json"
            for path in (RAW, PREPARED, INPUT_MANIFEST, PRIVACY_DISPOSITION, ORACLE, FROZEN_SOURCE_RUN):
                if not path.is_file():
                    raise FileNotFoundError(path)
            print({"project_root": str(PROJECT_ROOT), "source_run": str(SOURCE_RUN)})
            """
        ),
        markdown("## 1. Install the small judge/evaluation environment"),
        code(
            """
            import importlib.metadata
            import subprocess

            REQUIRED = [
                "openai>=1.109,<3",
                "pydantic>=2.10,<3",
                "pandas>=2.2,<3",
                "pyarrow>=18,<23",
                "scikit-learn>=1.5,<2",
            ]
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *REQUIRED])
            print({name: importlib.metadata.version(name) for name in (
                "openai", "pydantic", "pandas", "pyarrow", "scikit-learn"
            )})
            """
        ),
        markdown("## 2. Materialize and hash-check the embedded judge source snapshot"),
        code(
            f"""
            import base64, gzip, hashlib, importlib

            embedded = json.loads({embedded!r})
            SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
            for name, item in embedded.items():
                raw = gzip.decompress(base64.b64decode(item["payload"]))
                if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                    raise ValueError(f"Embedded file hash mismatch: {{name}}")
                target = SNAPSHOT_ROOT / name
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError(f"Existing frozen source snapshot differs: {{target}}")
                target.write_bytes(raw)

            if str(SNAPSHOT_ROOT) not in sys.path:
                sys.path.insert(0, str(SNAPSHOT_ROOT))
            for module_name in (
                "cfpb_v052_pipeline_smoke_v02_common",
                "prepare_cfpb_seed_v052_pipeline_smoke_v02",
                "validate_cfpb_v052_pipeline_smoke_v02",
                "run_cfpb_v052_blind_semantic_judge_v01",
                "evaluate_cfpb_v052_semantic_judge_v01",
                "freeze_cfpb_v052_semantic_judge_v01",
            ):
                sys.modules.pop(module_name, None)
            from cfpb_v052_pipeline_smoke_v02_common import sha256_file, load_semantic_judgments
            from run_cfpb_v052_blind_semantic_judge_v01 import (
                RunnerConfig,
                fetch_zdr_endpoint_preflight,
                load_blind_cases,
                run_judge,
            )
            from validate_cfpb_v052_pipeline_smoke_v02 import validate_and_route
            from evaluate_cfpb_v052_semantic_judge_v01 import evaluate_judge
            from freeze_cfpb_v052_semantic_judge_v01 import freeze_judge

            PROMPT = SNAPSHOT_ROOT / "cfpb_v052_semantic_judge_v01.md"
            RUNNER_SOURCE = SNAPSHOT_ROOT / "run_cfpb_v052_blind_semantic_judge_v01.py"
            COMMON_SOURCE = SNAPSHOT_ROOT / "cfpb_v052_pipeline_smoke_v02_common.py"
            VALIDATOR_SOURCE = SNAPSHOT_ROOT / "validate_cfpb_v052_pipeline_smoke_v02.py"
            print({{name: item["sha256"] for name, item in embedded.items()}})
            """
        ),
        markdown("## 3. Inspect payload, verify the live ZDR endpoint, and provide the API key"),
        code(
            """
            from getpass import getpass
            import pandas as pd

            cases = load_blind_cases(RAW, PREPARED)
            assert len(cases) == 20
            assert all(set(case.model_dump()) == {"seed_id", "labels", "grounding", "generated"} for case in cases)
            print({
                "rows": len(cases),
                "fields_visible_to_judge": ["case_id", "labels", "grounding_excerpt", "generated_dialogue"],
                "oracle_loaded_by_runner": False,
                "candidate_judgments_loaded_by_runner": False,
                "prompt_sha256": sha256_file(PROMPT),
            })
            display(pd.DataFrame([
                {
                    "seed_id": case.seed_id,
                    "product": case.labels["product"],
                    "issue": case.labels["issue"],
                    "grounding_preview": case.grounding[:180],
                }
                for case in cases[:2]
            ]))

            CONFIG = RunnerConfig(
                model_name="qwen/qwen3.5-35b-a3b",
                base_url="https://openrouter.ai/api/v1",
                judge_id="openrouter_qwen3_5_35b_a3b_deepinfra_zdr_nonthinking_blind_v01_attempt05",
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
                os.environ["OPENROUTER_API_KEY"] = getpass("OPENROUTER_API_KEY (not persisted): ")
            if not os.environ["OPENROUTER_API_KEY"].strip():
                raise RuntimeError("OPENROUTER_API_KEY is empty")
            """
        ),
        markdown(
            """
            ## 4. Two-row API/schema contract smoke

            These fixed development IDs exercise one previously accepted row and
            one semantic-reject row. Their expected labels are not sent or loaded.
            Rerunning this cell uses the separate two-row cache. A second provider
            call is allowed only when a non-empty response fails JSON, schema, or
            exact-evidence validation; other failures stop after one call.
            """
        ),
        code(
            """
            CONTRACT_IDS = [
                "cfpb_v052_0d210ece132c66d3",
                "cfpb_v052_2c70d920e4cfccd5",
            ]
            contract_report = run_judge(
                raw_path=RAW,
                prepared_path=PREPARED,
                prompt_path=PROMPT,
                judgments_path=CONTRACT_ROOT / "semantic_judgments.jsonl",
                raw_responses_path=CONTRACT_ROOT / "raw_responses.jsonl",
                failed_attempts_path=CONTRACT_ROOT / "failed_attempts.jsonl",
                endpoint_preflight_path=ENDPOINT_PREFLIGHT,
                report_path=CONTRACT_ROOT / "judge_run_report.json",
                config=CONFIG,
                api_key=os.environ["OPENROUTER_API_KEY"],
                selected_seed_ids=CONTRACT_IDS,
            )
            assert contract_report["semantic_coverage_complete"] is True
            contract = load_semantic_judgments(CONTRACT_ROOT / "semantic_judgments.jsonl")
            display(pd.DataFrame([
                {"seed_id": key, "decision": value.decision, "reasons": value.reasons}
                for key, value in sorted(contract.items())
            ]))
            """
        ),
        markdown("## 5. Run/resume the blind 20-row development calibration"),
        code(
            """
            import shutil

            # Reuse the two contract-smoke judgments only when starting a fresh
            # calibration. They used the exact same prompt, config, and schema;
            # their separate originals remain as contract evidence.
            calibration_judgments = CALIBRATION_ROOT / "semantic_judgments.jsonl"
            calibration_responses = CALIBRATION_ROOT / "raw_responses.jsonl"
            calibration_failures = CALIBRATION_ROOT / "failed_attempts.jsonl"
            if not calibration_judgments.exists() and not calibration_responses.exists():
                CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
                shutil.copy2(CONTRACT_ROOT / "semantic_judgments.jsonl", calibration_judgments)
                shutil.copy2(CONTRACT_ROOT / "raw_responses.jsonl", calibration_responses)
                contract_failures = CONTRACT_ROOT / "failed_attempts.jsonl"
                if contract_failures.exists():
                    shutil.copy2(contract_failures, calibration_failures)
                print("Bootstrapped the fresh calibration cache with 2 contract-smoke rows")
            elif calibration_judgments.exists() != calibration_responses.exists():
                raise ValueError("Calibration judgment/audit cache is incomplete")

            calibration_report = run_judge(
                raw_path=RAW,
                prepared_path=PREPARED,
                prompt_path=PROMPT,
                judgments_path=calibration_judgments,
                raw_responses_path=calibration_responses,
                failed_attempts_path=calibration_failures,
                endpoint_preflight_path=ENDPOINT_PREFLIGHT,
                report_path=CALIBRATION_ROOT / "judge_run_report.json",
                config=CONFIG,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            assert calibration_report["semantic_coverage_complete"] is True
            print({
                "rows": calibration_report["judgment_rows"],
                "new_provider_calls": calibration_report["new_provider_calls"],
                "decision_counts": calibration_report["decision_counts"],
                "failed_attempt_rows": calibration_report["outputs"]["failed_attempts"]["rows"],
            })
            """
        ),
        markdown(
            """
            ## 6. Score only after all blind judgments exist

            This is the first cell that loads the GPT-5.6-SOL development
            adjudication. It measures agreement, not human accuracy.
            """
        ),
        code(
            """
            EVALUATED = CALIBRATION_ROOT / "evaluated"
            evaluation = validate_and_route(
                raw_output=RAW,
                prepared_input=PREPARED,
                input_manifest=INPUT_MANIFEST,
                disposition_path=PRIVACY_DISPOSITION,
                validated_path=EVALUATED / "validated/dialogues.jsonl",
                rejected_path=EVALUATED / "rejected/dialogues.jsonl",
                review_path=EVALUATED / "review/dialogues.jsonl",
                report_path=EVALUATED / "validation_report_v02.json",
                semantic_judgments_path=CALIBRATION_ROOT / "semantic_judgments.jsonl",
                oracle_path=ORACLE,
            )
            metrics = evaluation["oracle_metrics"]
            calibration_metrics = evaluate_judge(
                judgments_path=CALIBRATION_ROOT / "semantic_judgments.jsonl",
                oracle_path=ORACLE,
                combined_report_path=EVALUATED / "validation_report_v02.json",
                output_path=CALIBRATION_ROOT / "calibration_metrics_v01.json",
            )
            display(pd.DataFrame({
                "judge_only": pd.Series(calibration_metrics["judge_only_metrics"]),
                "combined_validator": pd.Series(calibration_metrics["combined_validator_metrics"]),
            }))
            print({
                "validated": evaluation["validated_rows"],
                "rejected": evaluation["rejected_rows"],
                "review": evaluation["review_rows"],
                "development_only": True,
                "human_gold": False,
            })
            """
        ),
        markdown(
            """
            ## 7. Explicitly freeze judge v01

            Review the metrics above first. Freezing records the exact model,
            prompt, parameters, schemas, runner, source inputs, judgments, and
            calibration report. It does not approve a benchmark or formal pilot.
            """
        ),
        code(
            """
            CALIBRATION_REVIEW_COMPLETED = False
            APPROVE_FREEZE = False
            APPROVER_ID = ""

            if not CALIBRATION_REVIEW_COMPLETED or not APPROVE_FREEZE:
                print(
                    "Not frozen. Non-thinking execution success alone is not freeze evidence. "
                    "Review the 20-row metrics and disagreements first; only then set both "
                    "CALIBRATION_REVIEW_COMPLETED=True and APPROVE_FREEZE=True with APPROVER_ID."
                )
            else:
                frozen = freeze_judge(
                    judge_report_path=CALIBRATION_ROOT / "judge_run_report.json",
                    judgments_path=CALIBRATION_ROOT / "semantic_judgments.jsonl",
                    evaluation_path=EVALUATED / "validation_report_v02.json",
                    metrics_path=CALIBRATION_ROOT / "calibration_metrics_v01.json",
                    prompt_path=PROMPT,
                    oracle_path=ORACLE,
                    raw_path=RAW,
                    prepared_path=PREPARED,
                    runner_path=RUNNER_SOURCE,
                    common_path=COMMON_SOURCE,
                    validator_path=VALIDATOR_SOURCE,
                    endpoint_preflight_path=ENDPOINT_PREFLIGHT,
                    output_path=REVALIDATION_ROOT / "judges/frozen/cfpb_v052_semantic_judge_v01_attempt05.json",
                    approved_by=APPROVER_ID,
                    approve_freeze=True,
                )
                print(json.dumps(frozen.model_dump(mode="json"), ensure_ascii=False, indent=2))
            """
        ),
        markdown(
            """
            ## Stop point

            Do not revise the frozen prompt, reason definitions, model, inference
            parameters, or parsing logic after this notebook. The next notebook
            must generate a new held-out smoke and apply this frozen judge without
            calibrating on the held-out answers. A finite human audit follows that
            fixed evaluation. Formal 50–100-row generation remains blocked until a
            separately approved privacy protocol makes
            `benchmark_release_gate_passed=true`.
            """
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
