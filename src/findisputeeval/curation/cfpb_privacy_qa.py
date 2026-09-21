"""Presidio/spaCy privacy scan and fail-closed manual review helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


HIGH_RISK_ENTITIES = {
    "PERSON",
    "LOCATION",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "CREDIT_CARD",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "IP_ADDRESS",
    "IBAN_CODE",
    "CRYPTO",
    "MEDICAL_LICENSE",
    "NRP",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_presidio_analyzer(model_name: str = "en_core_web_sm"):
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": model_name}],
        }
    )
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


def scan_frame(
    frame: pd.DataFrame,
    analyzer: Any,
    *,
    id_column: str,
    text_column: str,
    split_column: str | None = None,
    minimum_score: float = 0.35,
) -> pd.DataFrame:
    missing = sorted({id_column, text_column} - set(frame.columns))
    if missing:
        raise ValueError(f"Privacy scan columns are missing: {missing}")
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        text = str(record.get(text_column, ""))
        for finding in analyzer.analyze(text=text, language="en", entities=sorted(HIGH_RISK_ENTITIES)):
            if float(finding.score) < minimum_score:
                continue
            start, end = int(finding.start), int(finding.end)
            detected = text[start:end]
            finding_id = hashlib.sha256(
                f"{record[id_column]}|{finding.entity_type}|{start}|{end}|{detected}".encode("utf-8")
            ).hexdigest()[:24]
            rows.append(
                {
                    "finding_id": finding_id,
                    "record_id": str(record[id_column]),
                    "release_split": str(record.get(split_column, "")) if split_column else "",
                    "entity_type": str(finding.entity_type),
                    "score": float(finding.score),
                    "start": start,
                    "end": end,
                    "detected_text": detected,
                    "context": text[max(0, start - 100) : min(len(text), end + 100)],
                    "manual_pii_present": "pending",
                    "release_action": "pending",
                    "reviewer": "",
                    "reviewed_utc": "",
                    "notes": "",
                }
            )
    columns = [
        "finding_id", "record_id", "release_split", "entity_type", "score", "start", "end",
        "detected_text", "context", "manual_pii_present", "release_action", "reviewer",
        "reviewed_utc", "notes",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["record_id", "start", "entity_type"], kind="mergesort"
    ).reset_index(drop=True)


def finalize_privacy_review(
    review: pd.DataFrame,
    *,
    scope: str,
    source_sha256: str,
) -> dict[str, Any]:
    allowed_present = {"yes", "no"}
    labels = review["manual_pii_present"].astype(str).str.lower()
    invalid = review.loc[~labels.isin(allowed_present)]
    missing_reviewer = review.loc[review["reviewer"].astype(str).str.strip().eq("")]
    if not invalid.empty or not missing_reviewer.empty:
        raise ValueError(
            f"Privacy review incomplete: invalid_or_pending={len(invalid)}, "
            f"missing_reviewer={len(missing_reviewer)}"
        )
    positives = review.loc[labels.eq("yes")]
    positive_actions = positives["release_action"].astype(str).str.lower()
    invalid_actions = positives.loc[~positive_actions.isin({"exclude", "redact_and_rescan"})]
    all_actions = review["release_action"].astype(str).str.lower()
    invalid_negative = review.loc[labels.eq("no") & ~all_actions.isin({"allow"})]
    if not invalid_actions.empty or not invalid_negative.empty:
        raise ValueError(
            f"Privacy actions invalid: positive={len(invalid_actions)}, negative={len(invalid_negative)}"
        )
    return {
        "clearance_version": "v01",
        "scope": scope,
        "finalized_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha256,
        "findings_reviewed": len(review),
        "confirmed_pii_findings": len(positives),
        "release_clearance_passed": positives.empty,
        "required_follow_up": (
            [] if positives.empty else ["apply_versioned_overrides_or_redaction", "rebuild_and_rescan"]
        ),
    }
