"""Formal CFPB privacy QA with negative controls and challenge tests.

Version 02 keeps the Presidio/spaCy finding review from v01 and adds two
independent checks which are required before benchmark release:

* a deterministic, stratified manual review of records with no NER finding;
* a synthetic (non-person) challenge set which verifies detector sensitivity.

The resulting clearance hash-binds every evidence file and fails closed when
any confirmed PII, incomplete review, weak model, or failed challenge remains.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .cfpb_privacy_qa import HIGH_RISK_ENTITIES, scan_frame, sha256_file


FORMAL_MODELS = {"en_core_web_lg", "en_core_web_trf"}
REVIEW_LABELS = {"yes", "no"}
POSITIVE_ACTIONS = {"exclude", "redact_and_rescan"}


def _stable_key(seed: int, namespace: str, value: object) -> str:
    payload = f"{seed}|{namespace}|{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _length_bucket(words: int) -> str:
    if words < 50:
        return "lt_50"
    if words < 150:
        return "50_149"
    if words < 400:
        return "150_399"
    if words < 800:
        return "400_799"
    return "ge_800"


def build_negative_control_sample(
    frame: pd.DataFrame,
    findings: pd.DataFrame,
    *,
    id_column: str,
    text_column: str,
    split_column: str,
    sample_size: int = 200,
    random_seed: int = 20260713,
) -> pd.DataFrame:
    """Select deterministic round-robin strata from zero-finding records."""

    required = {id_column, text_column, split_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Negative-control columns are missing: {missing}")
    if sample_size <= 0:
        raise ValueError("Negative-control sample_size must be positive")

    finding_id_column = (
        "release_record_id"
        if "release_record_id" in findings.columns
        else "record_id"
        if "record_id" in findings.columns
        else None
    )
    detected_ids = (
        set(findings[finding_id_column].astype(str))
        if finding_id_column is not None
        else set()
    )
    candidates = frame.loc[~frame[id_column].astype(str).isin(detected_ids)].copy()
    if len(candidates) < sample_size:
        raise ValueError(
            f"Only {len(candidates)} zero-finding records are available; "
            f"{sample_size} are required"
        )

    candidates["source_record_id"] = (
        candidates["record_id"].astype(str)
        if "record_id" in candidates.columns
        else candidates[id_column].astype(str)
    )
    candidates["release_record_id"] = candidates[id_column].astype(str)
    candidates["release_split"] = candidates[split_column].astype(str)
    candidates["review_text"] = candidates[text_column].fillna("").astype(str)
    if candidates["review_text"].str.strip().eq("").any():
        raise ValueError("Negative-control candidates contain empty text")

    if "word_count" in candidates:
        words = pd.to_numeric(candidates["word_count"], errors="coerce")
        fallback = candidates["review_text"].str.split().str.len()
        words = words.fillna(fallback).astype(int)
    else:
        words = candidates["review_text"].str.split().str.len().astype(int)
    candidates["word_count"] = words
    candidates["length_bucket"] = words.map(_length_bucket)

    for column, default in (
        ("register_candidate", "unknown"),
        ("lid_status", "unknown"),
        ("pii_regex_risk", False),
    ):
        if column not in candidates:
            candidates[column] = default
        candidates[column] = candidates[column].fillna(default).astype(str)

    strata = [
        "release_split",
        "register_candidate",
        "lid_status",
        "length_bucket",
        "pii_regex_risk",
    ]
    candidates["selection_stratum"] = candidates[strata].agg("|".join, axis=1)
    candidates["_selection_key"] = candidates["release_record_id"].map(
        lambda value: _stable_key(random_seed, "privacy_negative_control", value)
    )
    candidates = candidates.sort_values(
        ["selection_stratum", "_selection_key", "release_record_id"], kind="mergesort"
    )
    candidates["_within_stratum_rank"] = candidates.groupby(
        "selection_stratum", sort=True
    ).cumcount()
    selected = candidates.sort_values(
        ["_within_stratum_rank", "selection_stratum", "_selection_key"],
        kind="mergesort",
    ).head(sample_size).copy()
    selected["negative_control_id"] = selected["release_record_id"].map(
        lambda value: hashlib.sha256(
            f"negative_control_v02|{value}".encode("utf-8")
        ).hexdigest()[:24]
    )
    selected["manual_pii_present"] = "pending"
    selected["release_action"] = "pending"
    selected["reviewer"] = ""
    selected["reviewed_utc"] = ""
    selected["notes"] = ""
    columns = [
        "negative_control_id",
        "release_record_id",
        "source_record_id",
        "release_split",
        "register_candidate",
        "lid_status",
        "length_bucket",
        "pii_regex_risk",
        "word_count",
        "selection_stratum",
        "review_text",
        "manual_pii_present",
        "release_action",
        "reviewer",
        "reviewed_utc",
        "notes",
    ]
    return selected[columns].reset_index(drop=True)


def build_challenge_set() -> pd.DataFrame:
    """Return fixed, synthetic examples; none identify an intended real person."""

    rows = [
        ("email", "Send the notice to alex.taylor@example.com.", "EMAIL_ADDRESS", "alex.taylor@example.com", True),
        ("phone", "My callback number is (202) 555-0147.", "PHONE_NUMBER", "(202) 555-0147", True),
        ("ssn", "The synthetic SSN for this test is 078-05-1120.", "US_SSN", "078-05-1120", True),
        ("credit_card", "The test card number is 4111 1111 1111 1111.", "CREDIT_CARD", "4111 1111 1111 1111", True),
        ("bank", "The bank routing number is 021000021.", "US_BANK_NUMBER", "021000021", True),
        ("ip", "The example request came from 192.0.2.25.", "IP_ADDRESS", "192.0.2.25", True),
        ("iban", "The test IBAN is GB82 WEST 1234 5698 7654 32.", "IBAN_CODE", "GB82 WEST 1234 5698 7654 32", True),
        ("crypto", "The test wallet is 1BoatSLRHtKNngkdXEeobR76b53LETtpyT.", "CRYPTO", "1BoatSLRHtKNngkdXEeobR76b53LETtpyT", True),
        ("person_1", "The fictional complainant is Maria Hernandez.", "PERSON", "Maria Hernandez", True),
        ("person_2", "Please contact the fictional customer Jonathan Williams.", "PERSON", "Jonathan Williams", True),
        ("location_1", "The example address is 1600 Amphitheatre Parkway, Mountain View, California.", "LOCATION", "Mountain View, California", True),
        ("location_2", "The fictional customer moved to Seattle, Washington.", "LOCATION", "Seattle, Washington", True),
        ("placeholder_negative", "The consumer name and account are [REDACTED].", "", "", True),
        ("xxxx_negative", "The account ending in XXXX was disputed.", "", "", True),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "challenge_id",
            "text",
            "expected_entity_type",
            "expected_value",
            "required",
        ],
    )


def evaluate_challenge_set(
    challenges: pd.DataFrame,
    analyzer: Any,
    *,
    minimum_score: float = 0.35,
) -> pd.DataFrame:
    """Evaluate sensitive-span recall and report entity-type agreement.

    Privacy release depends on an expected sensitive span being surfaced by
    any configured high-risk detector. Exact entity typing is retained as a
    diagnostic because a mis-typed but surfaced span still reaches manual
    review and is not a privacy false negative.
    """

    rows: list[dict[str, Any]] = []
    for row in challenges.to_dict(orient="records"):
        text = str(row["text"])
        expected_type = str(row["expected_entity_type"])
        expected_value = str(row["expected_value"])
        detections = []
        for finding in analyzer.analyze(
            text=text,
            language="en",
            entities=sorted(HIGH_RISK_ENTITIES),
        ):
            if float(finding.score) < minimum_score:
                continue
            detections.append(
                {
                    "entity_type": str(finding.entity_type),
                    "score": float(finding.score),
                    "start": int(finding.start),
                    "end": int(finding.end),
                    "text": text[int(finding.start) : int(finding.end)],
                }
            )
        if expected_type:
            expected_start = text.index(expected_value)
            expected_end = expected_start + len(expected_value)
            overlapping = [
                finding
                for finding in detections
                if finding["start"] < expected_end
                and finding["end"] > expected_start
            ]
            overlap_passed = bool(overlapping)
            expected_type_matched = any(
                finding["entity_type"] == expected_type for finding in overlapping
            )
            passed = overlap_passed
        else:
            passed = len(detections) == 0
            overlap_passed = passed
            expected_type_matched = True
        rows.append(
            {
                **row,
                "detection_overlap_passed": bool(overlap_passed),
                "expected_type_matched": bool(expected_type_matched),
                "challenge_passed": bool(passed),
                "detections_json": json.dumps(detections, ensure_ascii=False, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _validate_manual_review(review: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = {"manual_pii_present", "release_action", "reviewer"}
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError(f"{label} review columns are missing: {missing}")
    labels = review["manual_pii_present"].astype(str).str.lower()
    invalid = review.loc[~labels.isin(REVIEW_LABELS)]
    missing_reviewer = review.loc[review["reviewer"].astype(str).str.strip().eq("")]
    positives = review.loc[labels.eq("yes")]
    invalid_positive = positives.loc[
        ~positives["release_action"].astype(str).str.lower().isin(POSITIVE_ACTIONS)
    ]
    invalid_negative = review.loc[
        labels.eq("no")
        & ~review["release_action"].astype(str).str.lower().eq("allow")
    ]
    if any(map(len, (invalid, missing_reviewer, invalid_positive, invalid_negative))):
        raise ValueError(
            f"{label} review incomplete: invalid_or_pending={len(invalid)}, "
            f"missing_reviewer={len(missing_reviewer)}, "
            f"invalid_positive_action={len(invalid_positive)}, "
            f"invalid_negative_action={len(invalid_negative)}"
        )
    return positives


def _wilson_upper_95(positives: int, total: int) -> float:
    if total <= 0:
        return 1.0
    z = 1.959963984540054
    p = positives / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre + margin) / denominator


def finalize_privacy_review_v02(
    findings_review: pd.DataFrame,
    negative_review: pd.DataFrame,
    challenge_results: pd.DataFrame,
    *,
    scope: str,
    source_sha256: str,
    analyzer_model: str,
    analyzer_config_sha256: str,
    scanned_records: int,
    evidence_paths: dict[str, Path],
    minimum_negative_controls: int = 200,
) -> dict[str, Any]:
    """Finalize a hash-bound, fail-closed formal privacy clearance."""

    if analyzer_model not in FORMAL_MODELS:
        raise ValueError(
            f"Formal privacy QA requires one of {sorted(FORMAL_MODELS)}; "
            f"received {analyzer_model}"
        )
    if scope != "full_v052":
        raise ValueError("Privacy clearance v02 is only valid for scope=full_v052")
    if scanned_records <= 0:
        raise ValueError("scanned_records must be positive")

    finding_positives = _validate_manual_review(findings_review, label="Finding")
    negative_positives = _validate_manual_review(negative_review, label="Negative-control")
    if len(negative_review) < minimum_negative_controls:
        raise ValueError(
            f"At least {minimum_negative_controls} negative controls are required; "
            f"received {len(negative_review)}"
        )
    if "challenge_passed" not in challenge_results:
        raise ValueError("Challenge results do not contain challenge_passed")
    required_challenges = challenge_results.loc[_bool_mask(challenge_results["required"])]
    challenge_failures = required_challenges.loc[
        ~_bool_mask(required_challenges["challenge_passed"])
    ]
    type_mismatches = (
        required_challenges.loc[
            required_challenges["expected_entity_type"].astype(str).str.strip().ne("")
            & ~_bool_mask(required_challenges["expected_type_matched"])
        ]
        if "expected_type_matched" in required_challenges
        else required_challenges.iloc[0:0]
    )

    evidence: dict[str, dict[str, Any]] = {}
    for name, raw_path in evidence_paths.items():
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        item: dict[str, Any] = {"path": path.name, "sha256": sha256_file(path)}
        if path.suffix.lower() == ".csv":
            item["rows"] = len(pd.read_csv(path, encoding="utf-8-sig"))
        evidence[name] = item

    confirmed = len(finding_positives) + len(negative_positives)
    challenge_passed = challenge_failures.empty and not required_challenges.empty
    release_passed = confirmed == 0 and challenge_passed
    negative_rate = len(negative_positives) / len(negative_review)
    return {
        "clearance_version": "v02",
        "scope": scope,
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha256,
        "scanned_records": int(scanned_records),
        "analyzer": {
            "engine": "presidio_spacy",
            "model": analyzer_model,
            "config_sha256": analyzer_config_sha256,
        },
        "finding_review": {
            "findings_reviewed": len(findings_review),
            "confirmed_pii_findings": len(finding_positives),
        },
        "negative_control_review": {
            "records_reviewed": len(negative_review),
            "strata_represented": int(negative_review["selection_stratum"].nunique()),
            "confirmed_pii_records": len(negative_positives),
            "observed_miss_rate": negative_rate,
            "wilson_upper_95": _wilson_upper_95(len(negative_positives), len(negative_review)),
        },
        "challenge_validation": {
            "required": len(required_challenges),
            "passed": int(_bool_mask(required_challenges["challenge_passed"]).sum()),
            "failed": len(challenge_failures),
            "type_mismatches": len(type_mismatches),
            "all_required_passed": challenge_passed,
        },
        "evidence": evidence,
        "release_clearance_passed": release_passed,
        "required_follow_up": (
            []
            if release_passed
            else [
                "resolve_confirmed_or_missed_pii",
                "repair_failed_challenge_detection",
                "rebuild_and_rescan_if_release_text_changes",
            ]
        ),
    }
