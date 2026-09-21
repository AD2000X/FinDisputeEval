from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src.findisputeeval.curation.cfpb_privacy_qa_v02 import (
    build_negative_control_sample,
    evaluate_challenge_set,
    finalize_privacy_review_v02,
)


@dataclass
class Finding:
    entity_type: str
    start: int
    end: int
    score: float = 0.9


class FakeAnalyzer:
    def analyze(self, *, text, language, entities):
        if "example.com" in text:
            value = "alex.taylor@example.com"
            start = text.index(value)
            return [Finding("EMAIL_ADDRESS", start, start + len(value))]
        if "078-05-1120" in text:
            value = "078-05-1120"
            start = text.index(value)
            return [Finding("PHONE_NUMBER", start, start + len(value))]
        return []


def _review(rows: int, *, id_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            id_column: [f"id_{index}" for index in range(rows)],
            "manual_pii_present": ["no"] * rows,
            "release_action": ["allow"] * rows,
            "reviewer": ["reviewer"] * rows,
            "selection_stratum": [f"stratum_{index % 4}" for index in range(rows)],
        }
    )


def test_negative_control_sample_is_deterministic_and_excludes_findings():
    frame = pd.DataFrame(
        {
            "seed_id": [f"s{index}" for index in range(30)],
            "seed_text": [f"narrative {index} " * (index + 1) for index in range(30)],
            "release_split": ["core" if index % 2 else "stress" for index in range(30)],
            "register_candidate": ["consumer" if index % 3 else "letter" for index in range(30)],
            "lid_status": ["english"] * 30,
            "pii_regex_risk": [False] * 30,
        }
    )
    findings = pd.DataFrame({"record_id": ["s0", "s1"]})
    first = build_negative_control_sample(
        frame,
        findings,
        id_column="seed_id",
        text_column="seed_text",
        split_column="release_split",
        sample_size=12,
    )
    second = build_negative_control_sample(
        frame,
        findings,
        id_column="seed_id",
        text_column="seed_text",
        split_column="release_split",
        sample_size=12,
    )
    assert first.equals(second)
    assert len(first) == 12
    assert set(first["release_record_id"]).isdisjoint({"s0", "s1"})
    assert first["release_record_id"].equals(first["source_record_id"])
    assert first["negative_control_id"].is_unique
    assert first["selection_stratum"].nunique() > 1


def test_challenge_evaluation_matches_expected_span_and_negative():
    challenges = pd.DataFrame(
        [
            {
                "challenge_id": "email",
                "text": "Send to alex.taylor@example.com.",
                "expected_entity_type": "EMAIL_ADDRESS",
                "expected_value": "alex.taylor@example.com",
                "required": True,
            },
            {
                "challenge_id": "ssn_mistyped",
                "text": "Synthetic SSN 078-05-1120.",
                "expected_entity_type": "US_SSN",
                "expected_value": "078-05-1120",
                "required": True,
            },
            {
                "challenge_id": "negative",
                "text": "The account is [REDACTED].",
                "expected_entity_type": "",
                "expected_value": "",
                "required": True,
            },
        ]
    )
    results = evaluate_challenge_set(challenges, FakeAnalyzer())
    assert results["challenge_passed"].tolist() == [True, True, True]
    assert results["expected_type_matched"].tolist() == [True, False, True]


def test_finalize_v02_passes_and_hash_binds_evidence():
    tmp_path = Path("temp/privacy_qa_v02_unit/pass")
    tmp_path.mkdir(parents=True, exist_ok=True)
    findings = _review(2, id_column="finding_id")
    negatives = _review(4, id_column="negative_control_id")
    challenges = pd.DataFrame(
        {"required": [True, True], "challenge_passed": [True, True]}
    )
    paths = {}
    for name, frame in {
        "findings": findings,
        "negative_controls": negatives,
        "challenge_results": challenges,
    }.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    paths["analyzer_config"] = config

    result = finalize_privacy_review_v02(
        findings,
        negatives,
        challenges,
        scope="full_v052",
        source_sha256="a" * 64,
        analyzer_model="en_core_web_trf",
        analyzer_config_sha256="b" * 64,
        scanned_records=20,
        evidence_paths=paths,
        minimum_negative_controls=4,
    )
    assert result["release_clearance_passed"] is True
    assert result["negative_control_review"]["confirmed_pii_records"] == 0
    assert set(result["evidence"]) == set(paths)


def test_finalize_v02_blocks_missed_pii():
    tmp_path = Path("temp/privacy_qa_v02_unit/block")
    tmp_path.mkdir(parents=True, exist_ok=True)
    findings = _review(1, id_column="finding_id")
    negatives = _review(4, id_column="negative_control_id")
    negatives.loc[0, ["manual_pii_present", "release_action"]] = ["yes", "exclude"]
    challenges = pd.DataFrame({"required": [True], "challenge_passed": [True]})
    paths = {}
    for name, frame in {
        "findings": findings,
        "negative_controls": negatives,
        "challenge_results": challenges,
    }.items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path

    result = finalize_privacy_review_v02(
        findings,
        negatives,
        challenges,
        scope="full_v052",
        source_sha256="a" * 64,
        analyzer_model="en_core_web_lg",
        analyzer_config_sha256="b" * 64,
        scanned_records=20,
        evidence_paths=paths,
        minimum_negative_controls=4,
    )
    assert result["release_clearance_passed"] is False
    assert result["negative_control_review"]["confirmed_pii_records"] == 1


def test_finalize_v02_rejects_small_model():
    tmp_path = Path("temp/privacy_qa_v02_unit/model")
    tmp_path.mkdir(parents=True, exist_ok=True)
    review = _review(1, id_column="finding_id")
    challenge = pd.DataFrame({"required": [True], "challenge_passed": [True]})
    with pytest.raises(ValueError, match="Formal privacy QA requires"):
        finalize_privacy_review_v02(
            review,
            review.assign(selection_stratum="one"),
            challenge,
            scope="full_v052",
            source_sha256="a" * 64,
            analyzer_model="en_core_web_sm",
            analyzer_config_sha256="b" * 64,
            scanned_records=1,
            evidence_paths={},
            minimum_negative_controls=1,
        )
