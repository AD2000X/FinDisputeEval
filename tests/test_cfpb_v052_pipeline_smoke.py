from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from scripts.generation.nemo_data_designer.prepare_cfpb_seed_v052_pipeline_smoke import (
    redact_grounding_excerpt,
    stratified_sample,
)
from scripts.generation.nemo_data_designer.generate_multi_turn_dialogues_v052_pipeline_smoke import (
    resolve_final_dataset_file,
)
from scripts.generation.nemo_data_designer import validate_cfpb_v052_pipeline_smoke as validator


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def safe_payload() -> dict:
    return {
        "conversation": [
            {"role": "user", "content": "A recent transaction does not look right."},
            {"role": "assistant", "content": "Please describe what you noticed about the transaction."},
            {"role": "user", "content": "I did not authorize it and want to understand my options."},
            {"role": "assistant", "content": "Contact your financial institution through its official channel and keep your records."},
        ],
        "synthetic_case_summary": "A customer asks about a disputed transaction.",
        "privacy_notes": ["No identifying or contact details were used."],
    }


def test_stratified_sample_is_deterministic_and_bounded() -> None:
    rows = [
        {
            "seed_id": f"seed_{index}",
            "release_split": "population_core" if index % 2 else "enrichment",
            "product": f"product_{index % 4}",
            "issue": f"issue_{index % 7}",
        }
        for index in range(40)
    ]
    first = stratified_sample(rows, 20, 20260721)
    second = stratified_sample(rows, 20, 20260721)
    assert [row["seed_id"] for row in first] == [row["seed_id"] for row in second]
    assert len({row["seed_id"] for row in first}) == 20


def test_grounding_redaction_combines_ner_and_regex_spans() -> None:
    text = "Contact Maria at maria@example.com about the disputed transfer."
    findings = pd.DataFrame([{"start": 8, "end": 13}])
    redacted, count = redact_grounding_excerpt(text, findings)
    assert "Maria" not in redacted
    assert "maria@example.com" not in redacted
    assert redacted.count("[PII_CANDIDATE]") == 2
    assert count == 2


def test_project_root_env_has_no_eager_fallback() -> None:
    scripts = [
        Path("scripts/generation/nemo_data_designer") / name
        for name in (
            "prepare_cfpb_seed_v052_pipeline_smoke.py",
            "generate_multi_turn_dialogues_v052_pipeline_smoke.py",
            "validate_cfpb_v052_pipeline_smoke.py",
        )
    ]
    for source in scripts:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        project_root_gets = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "FINDISPUTEEVAL_PROJECT_ROOT"
        ]
        assert len(project_root_gets) == 1
        assert len(project_root_gets[0].args) == 1


def test_resolve_final_dataset_file_accepts_data_designer_directory() -> None:
    dataset_dir = Path("temp/test_cfpb_v052_final_dataset/parquet-files").resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    batch = dataset_dir / "batch_00000.parquet"
    batch.write_bytes(b"pipeline-smoke-test")
    assert resolve_final_dataset_file(dataset_dir) == batch
    assert resolve_final_dataset_file(batch) == batch


def test_validate_row_attaches_labels_programmatically() -> None:
    seed = {
        "seed_id": "seed_1",
        "product": "Checking account",
        "sub_product": "Checking account",
        "issue": "Disputed transaction",
        "sub_issue": "Cash withdrawal",
        "generation_grounding_excerpt": "The consumer noticed a disputed cash withdrawal.",
    }
    raw = {
        **{key: seed[key] for key in ("seed_id", "product", "sub_product", "issue", "sub_issue")},
        "conversation_length": 4,
        "dialogue": safe_payload(),
    }
    reasons, routed = validator.validate_row(raw, {"seed_1": seed})
    assert reasons == []
    assert routed["programmatic_labels"]["product"] == "Checking account"
    assert routed["pipeline_smoke_validation"]["labels_attached_programmatically"] is True
    assert routed["privacy_verified"] is False
    assert routed["benchmark_eligible"] is False


def test_validate_row_rejects_model_labels_pii_and_authority_claims() -> None:
    seed = {
        "seed_id": "seed_1",
        "product": "Card",
        "sub_product": "Credit card",
        "issue": "Billing dispute",
        "sub_issue": "Wrong amount",
        "generation_grounding_excerpt": "A consumer questions a disputed amount.",
    }
    payload = safe_payload()
    payload["label_product"] = "Card"
    payload["conversation"][1]["content"] = (
        "I'll refund it within 3 business days. Send your password to test@example.com."
    )
    raw = {
        **{key: seed[key] for key in ("seed_id", "product", "sub_product", "issue", "sub_issue")},
        "conversation_length": 4,
        "dialogue": payload,
    }
    reasons, _ = validator.validate_row(raw, {"seed_1": seed})
    assert "model_generated_label_field" in reasons
    assert "regex_pii:email" in reasons
    assert "assistant_sensitive_credential_request" in reasons
    assert "assistant_unavailable_action_claim" in reasons
    assert "assistant_outcome_or_deadline_promise" in reasons


def test_validate_and_route_keeps_all_outputs_nonbenchmark(monkeypatch) -> None:
    tmp_path = Path("temp/test_cfpb_v052_pipeline_smoke").resolve()
    smoke_root = tmp_path / "smoke"
    monkeypatch.setattr(validator, "SMOKE_ROOT", smoke_root)
    prepared = smoke_root / "prepared_inputs/input.jsonl"
    raw = smoke_root / "raw/final.jsonl"
    manifest_path = smoke_root / "prepared_inputs/manifest.json"
    disposition_path = tmp_path / "pipeline_privacy_disposition_v01.json"
    validated = smoke_root / "validated/dialogues.jsonl"
    rejected = smoke_root / "rejected/dialogues.jsonl"
    review = smoke_root / "review/human_review_10.jsonl"
    report_path = smoke_root / "report.json"

    seeds = []
    raws = []
    for index in range(10):
        seed = {
            "seed_id": f"seed_{index}",
            "product": "Checking account",
            "sub_product": "Checking account",
            "issue": "Disputed transaction",
            "sub_issue": "Cash withdrawal",
            "generation_grounding_excerpt": "A customer noticed a disputed transaction.",
        }
        seeds.append(seed)
        raws.append({**seed, "conversation_length": 4, "dialogue": safe_payload()})
    write_jsonl(prepared, seeds)
    write_jsonl(raw, raws)
    disposition = {
        "pipeline_smoke_only_passed": True,
        "manual_review_performed": False,
        "release_clearance_passed": False,
        "benchmark_eligible": False,
    }
    disposition_path.write_text(json.dumps(disposition), encoding="utf-8")
    manifest = {
        "purpose": "cfpb_seed_v052_pipeline_smoke_only",
        "selected_rows": 10,
        "pipeline_privacy_disposition_sha256": validator.sha256_file(disposition_path),
        "policy": {
            "pipeline_smoke_only": True,
            "provisional": True,
            "privacy_verified": False,
            "benchmark_eligible": False,
            "formal_pilot_allowed": False,
        },
        "output": {"sha256": validator.sha256_file(prepared)},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validator.validate_and_route(
        raw_output=raw,
        prepared_input=prepared,
        input_manifest=manifest_path,
        disposition_path=disposition_path,
        validated_path=validated,
        rejected_path=rejected,
        review_path=review,
        report_path=report_path,
    )
    assert report["pipeline_plumbing_passed"] is True
    assert report["validated_rows"] == 10
    assert report["rejected_rows"] == 0
    assert report["policy"]["privacy_verified"] is False
    assert report["policy"]["benchmark_eligible"] is False
    assert len(review.read_text(encoding="utf-8").splitlines()) == 10
