"""Build the self-contained CFPB Seed v05.2 pipeline-smoke Colab notebook."""

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
    / "FinDisputeEval_NeMoDataDesigner_CFPB_v052_pipeline_smoke_colab.ipynb"
)
MODULES = [
    ROOT / "scripts/generation/nemo_data_designer/prepare_cfpb_seed_v052_pipeline_smoke.py",
    ROOT / "scripts/generation/nemo_data_designer/generate_multi_turn_dialogues_v052_pipeline_smoke.py",
    ROOT / "scripts/generation/nemo_data_designer/validate_cfpb_v052_pipeline_smoke.py",
]


def source_lines(value: str) -> list[str]:
    value = dedent(value).strip("\n") + "\n"
    return value.splitlines(keepends=True)


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
    payload: dict[str, dict[str, str]] = {}
    for path in MODULES:
        raw = path.read_bytes()
        payload[path.name] = {
            "payload": base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return payload


def build_notebook() -> dict:
    embedded = json.dumps(embedded_payload(), sort_keys=True)
    cells = [
        markdown(
            """
            # CFPB Seed v05.2 — NeMo Data Designer pipeline-only smoke

            This notebook runs **10–20 engineering-only dialogues** from Seed v05.2.
            The source privacy review was closed without manual verification. This
            notebook therefore requires `pipeline_privacy_disposition_v01.json`,
            applies additional grounding redaction before provider submission, and
            marks every output:

            - `provisional=true`
            - `privacy_verified=false`
            - `benchmark_eligible=false`

            It cannot create a formal benchmark, privacy claim, prevalence estimate,
            publication artifact, or distribution-calibration input.
            """
        ),
        markdown("## 0. Mount Drive, locate inputs, and create a persistent run"),
        code(
            """
            from pathlib import Path
            from datetime import datetime, timezone
            import hashlib, json, os, sys

            IN_COLAB = "google.colab" in sys.modules
            if IN_COLAB:
                from google.colab import drive
                drive.mount("/content/drive")
                PROJECT_ROOT = Path("/content/drive/MyDrive/FinDisputeEval")
            else:
                here = Path.cwd().resolve()
                PROJECT_ROOT = next(
                    (p for p in (here, *here.parents) if (p / "WORK_PROGRESS.md").exists()),
                    None,
                )
                if PROJECT_ROOT is None:
                    raise FileNotFoundError("Open the FinDisputeEval repository in VS Code")

            os.environ["FINDISPUTEEVAL_PROJECT_ROOT"] = str(PROJECT_ROOT)
            SEED_ROOT = PROJECT_ROOT / "dataset/curated/seed_pools/cfpb_dispute/seed_v052"
            PRIVACY_ROOT = (
                PROJECT_ROOT
                / "dataset/curated/annotations/cfpb_seed_v05_audit"
                / "run_20260713T145423Z/privacy_qa/full_v052_v02"
            )
            SEED_INPUT = SEED_ROOT / "cfpb_seed_v052_generation_input.jsonl"
            SEED_MANIFEST = SEED_ROOT / "seed_v052_manifest.json"
            PRIVACY_DISPOSITION = PRIVACY_ROOT / "pipeline_privacy_disposition_v01.json"
            for path in (SEED_INPUT, SEED_MANIFEST, PRIVACY_DISPOSITION):
                if not path.exists():
                    raise FileNotFoundError(path)

            SMOKE_ROOT = (
                PROJECT_ROOT
                / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override"
            )
            # Set this explicitly when resuming a known run. When blank, recover
            # the newest run that has raw Parquet output but no final run manifest.
            RUN_ID_OVERRIDE = ""
            if not RUN_ID_OVERRIDE:
                incomplete_runs = sorted(
                    (
                        run_root
                        for run_root in SMOKE_ROOT.glob("run_*")
                        if any((run_root / "raw/data_designer").rglob("*.parquet"))
                        and not (run_root / "pipeline_smoke_run_manifest.json").exists()
                    ),
                    key=lambda path: path.name,
                    reverse=True,
                )
                if incomplete_runs:
                    RUN_ID_OVERRIDE = incomplete_runs[0].name.removeprefix("run_")
            RUN_ID = globals().get(
                "RUN_ID",
                RUN_ID_OVERRIDE or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            )
            RUN_ROOT = SMOKE_ROOT / f"run_{RUN_ID}"
            PREPARED_DIR = RUN_ROOT / "prepared_inputs"
            ARTIFACT_DIR = RUN_ROOT / "raw/data_designer"
            VALIDATED = RUN_ROOT / "validated/dialogues.jsonl"
            REJECTED = RUN_ROOT / "rejected/dialogues.jsonl"
            HUMAN_REVIEW = RUN_ROOT / "review/human_review_10.jsonl"
            REPORT = RUN_ROOT / "pipeline_smoke_validation_report.json"
            RUN_MANIFEST = RUN_ROOT / "pipeline_smoke_run_manifest.json"
            RAW_POINTER = RUN_ROOT / "raw_output_pointer.json"
            for path in (PREPARED_DIR, ARTIFACT_DIR, VALIDATED.parent, REJECTED.parent, HUMAN_REVIEW.parent):
                path.mkdir(parents=True, exist_ok=True)

            def sha256_file(path):
                digest = hashlib.sha256()
                with Path(path).open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                return digest.hexdigest()

            print({
                "runtime": "Colab" if IN_COLAB else "local VS Code",
                "project_root": str(PROJECT_ROOT),
                "run_root": str(RUN_ROOT),
                "resuming_incomplete_run": bool(RUN_ID_OVERRIDE),
                "seed_manifest_sha256": sha256_file(SEED_MANIFEST),
                "privacy_disposition_sha256": sha256_file(PRIVACY_DISPOSITION),
            })
            """
        ),
        markdown("## 1. Install the pinned generation environment"),
        code(
            """
            import importlib.metadata, subprocess

            REQUIRED_PACKAGES = [
                "data-designer==0.7.0",
                "pydantic>=2.10,<3",
                "pandas>=2.2,<3",
                "pyarrow>=18,<23",
            ]
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *REQUIRED_PACKAGES])
            print({
                name: importlib.metadata.version(name)
                for name in ("data-designer", "pydantic", "pandas", "pyarrow")
            })
            """
        ),
        markdown("## 2. Load hash-verified embedded pipeline modules"),
        code(
            f"""
            import base64, gzip

            RUNTIME = Path("/content/findisputeeval_v052_pipeline_smoke") if IN_COLAB else PROJECT_ROOT / ".runtime/v052_pipeline_smoke"
            RUNTIME.mkdir(parents=True, exist_ok=True)
            embedded = json.loads({embedded!r})
            for name, item in embedded.items():
                raw = gzip.decompress(base64.b64decode(item["payload"]))
                if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                    raise ValueError(f"Embedded module hash mismatch: {{name}}")
                (RUNTIME / name).write_bytes(raw)
            sys.path.insert(0, str(RUNTIME))
            for module_name in (
                "prepare_cfpb_seed_v052_pipeline_smoke",
                "generate_multi_turn_dialogues_v052_pipeline_smoke",
                "validate_cfpb_v052_pipeline_smoke",
            ):
                sys.modules.pop(module_name, None)

            from prepare_cfpb_seed_v052_pipeline_smoke import prepare_smoke_inputs
            from generate_multi_turn_dialogues_v052_pipeline_smoke import (
                build_config,
                create_dataset,
                resolve_final_dataset_file,
            )
            from validate_cfpb_v052_pipeline_smoke import validate_and_route
            MODULE_HASHES = {{name: item["sha256"] for name, item in embedded.items()}}
            print(MODULE_HASHES)
            """
        ),
        markdown(
            """
            ## 3. Set NVIDIA_API_KEY and verify the provider alias

            The key remains only in the current runtime. Use the `nvidia-text`
            alias, which is backed by the OpenAI-compatible NVIDIA endpoint in
            Data Designer 0.7.0.
            """
        ),
        code(
            """
            import shutil
            from getpass import getpass
            from data_designer.interface import DataDesigner

            MODEL_ALIAS = "nvidia-text"
            if not os.environ.get("NVIDIA_API_KEY"):
                os.environ["NVIDIA_API_KEY"] = getpass("NVIDIA_API_KEY: ")
            if not os.environ["NVIDIA_API_KEY"].strip():
                raise RuntimeError("NVIDIA_API_KEY is required")

            def engine_aliases_and_provider_types():
                engine = DataDesigner(artifact_path=str(ARTIFACT_DIR))
                aliases = {config.alias for config in engine.get_default_model_configs()}
                provider_types = {
                    provider.name: provider.provider_type
                    for provider in engine.get_default_model_providers()
                }
                return engine, aliases, provider_types

            engine, aliases, provider_types = engine_aliases_and_provider_types()
            if MODEL_ALIAS not in aliases or provider_types.get("nvidia") != "openai":
                shutil.rmtree(Path.home() / ".data-designer", ignore_errors=True)
                engine, aliases, provider_types = engine_aliases_and_provider_types()
            if MODEL_ALIAS not in aliases:
                raise RuntimeError(f"Model alias {MODEL_ALIAS!r} unavailable; aliases={sorted(aliases)}")
            if provider_types.get("nvidia") != "openai":
                raise RuntimeError("NVIDIA provider_type must be 'openai'")
            print({
                "model_alias": MODEL_ALIAS,
                "nvidia_provider_type": "openai",
                "provider_health_check": "deferred_to_preview",
            })
            """
        ),
        markdown("## 4. Verify lineage and prepare 20 defense-in-depth redacted inputs"),
        code(
            """
            SAMPLE_SIZE = 20
            RANDOM_SEED = 20260721
            SAMPLE_PATH, INPUT_MANIFEST_PATH, input_manifest = prepare_smoke_inputs(
                seed_input=SEED_INPUT,
                seed_manifest=SEED_MANIFEST,
                disposition_path=PRIVACY_DISPOSITION,
                output_dir=PREPARED_DIR,
                sample_size=SAMPLE_SIZE,
                random_seed=RANDOM_SEED,
            )
            print({
                "sample": str(SAMPLE_PATH),
                "rows": input_manifest["selected_rows"],
                "grounding_redactions": input_manifest["grounding_redactions"],
                "privacy_verified": input_manifest["policy"]["privacy_verified"],
                "benchmark_eligible": input_manifest["policy"]["benchmark_eligible"],
            })
            """
        ),
        markdown(
            """
            ## 5. Preview 2 and generate 20 dialogues

            Provider calls begin in this cell. A valid cached raw pointer avoids
            accidental regeneration after reconnecting.
            """
        ),
        code(
            """
            FORCE_REGENERATE = False
            RAW_OUTPUT = None
            if RAW_POINTER.exists() and not FORCE_REGENERATE:
                pointer = json.loads(RAW_POINTER.read_text(encoding="utf-8"))
                candidate = resolve_final_dataset_file(Path(pointer["path"]))
                if candidate.exists() and sha256_file(candidate) == pointer["sha256"]:
                    RAW_OUTPUT = candidate
                    print("Generation cache hit:", RAW_OUTPUT)

            # Recover a completed Data Designer dataset when generation finished
            # but pointer creation was interrupted (for example, when version
            # 0.7.0 returned the parquet-files directory instead of its file).
            RECOVERABLE_DATASET = ARTIFACT_DIR / "dataset/parquet-files"
            if (
                RAW_OUTPUT is None
                and not FORCE_REGENERATE
                and RECOVERABLE_DATASET.exists()
            ):
                RAW_OUTPUT = resolve_final_dataset_file(RECOVERABLE_DATASET)
                RAW_POINTER.write_text(
                    json.dumps(
                        {"path": str(RAW_OUTPUT), "sha256": sha256_file(RAW_OUTPUT)},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print("Recovered completed generation:", RAW_OUTPUT)

            if RAW_OUTPUT is None and RUN_ID_OVERRIDE and not FORCE_REGENERATE:
                raise RuntimeError(
                    "An incomplete run was selected but no unique raw Parquet batch "
                    "could be recovered. Refusing to call the provider again."
                )

            if RAW_OUTPUT is None:
                config_builder = build_config(str(SAMPLE_PATH), MODEL_ALIAS)
                results = create_dataset(
                    config_builder,
                    input_manifest["selected_rows"],
                    ARTIFACT_DIR,
                    preview_records=2,
                )
                RAW_OUTPUT = resolve_final_dataset_file(
                    Path(results.artifact_storage.final_dataset_path)
                )
                RAW_POINTER.write_text(
                    json.dumps(
                        {"path": str(RAW_OUTPUT), "sha256": sha256_file(RAW_OUTPUT)},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print("Raw Data Designer output:", RAW_OUTPUT)
            """
        ),
        markdown("## 6. Route raw output into validated, rejected, and human-review artifacts"),
        code(
            """
            report = validate_and_route(
                raw_output=RAW_OUTPUT,
                prepared_input=SAMPLE_PATH,
                input_manifest=INPUT_MANIFEST_PATH,
                disposition_path=PRIVACY_DISPOSITION,
                validated_path=VALIDATED,
                rejected_path=REJECTED,
                review_path=HUMAN_REVIEW,
                report_path=REPORT,
            )
            print(json.dumps({
                "pipeline_plumbing_passed": report["pipeline_plumbing_passed"],
                "raw_rows": report["raw_rows"],
                "validated_rows": report["validated_rows"],
                "rejected_rows": report["rejected_rows"],
                "rejection_reasons": report["rejection_reasons"],
                "human_review_rows": report["human_review_rows"],
                "privacy_verified": report["policy"]["privacy_verified"],
                "benchmark_eligible": report["policy"]["benchmark_eligible"],
            }, ensure_ascii=False, indent=2))
            """
        ),
        markdown("## 7. Inspect the 10-row human-review pack and finalize the run manifest"),
        code(
            """
            import pandas as pd

            review_rows = [json.loads(line) for line in HUMAN_REVIEW.read_text(encoding="utf-8").splitlines() if line]
            display(pd.DataFrame([
                {
                    "seed_id": row.get("seed_id"),
                    "product": row.get("programmatic_labels", {}).get("product"),
                    "issue": row.get("programmatic_labels", {}).get("issue"),
                    "passed": row.get("pipeline_smoke_validation", {}).get("passed"),
                    "reasons": row.get("pipeline_smoke_validation", {}).get("reasons"),
                }
                for row in review_rows
            ]))

            run_manifest = {
                "manifest_version": "v01",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "cfpb_seed_v052_pipeline_smoke_only",
                "run_id": RUN_ID,
                "module_hashes": MODULE_HASHES,
                "inputs": {
                    "seed_manifest_sha256": sha256_file(SEED_MANIFEST),
                    "privacy_disposition_sha256": sha256_file(PRIVACY_DISPOSITION),
                    "prepared_input_manifest_sha256": sha256_file(INPUT_MANIFEST_PATH),
                },
                "validation_report_sha256": sha256_file(REPORT),
                "pipeline_plumbing_passed": report["pipeline_plumbing_passed"],
                "policy": report["policy"],
            }
            RUN_MANIFEST.write_text(
                json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print({
                "run_root": str(RUN_ROOT),
                "run_manifest": str(RUN_MANIFEST),
                "pipeline_plumbing_passed": report["pipeline_plumbing_passed"],
                "formal_benchmark_allowed": False,
            })
            """
        ),
        markdown(
            """
            ## Interpretation

            A successful run proves only that input preparation, provider calls,
            structured generation, persistence, routing, and engineering validators
            work end to end. Review the 10-row pack for naturalness and grounding.
            Nothing from this run may enter the formal benchmark.
            """
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
