"""Shared, library-first contracts for CFPB Seed v05.2 smoke validator v02.

The module deliberately limits deterministic text matching to high-precision
privacy and integrity checks.  Semantic findings are represented as validated
Pydantic records supplied by a human review or a separately evaluated judge.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator


SEMANTIC_REASON_CODES = frozenset(
    {
        "authorization_state_changed",
        "claim_type_changed",
        "factual_status_changed",
        "indirect_sensitive_information_guidance",
        "insufficient_grounding",
        "label_grounding_conflict",
        "legal_or_rights_claim",
        "unsupported_product_policy",
        "unsupported_procedural_guidance",
        "unsupported_scenario_detail",
    }
)


class StrictModel(BaseModel):
    """Base model that rejects misspelled or unexpected evidence fields."""

    model_config = ConfigDict(extra="forbid")


class EvidenceSpan(StrictModel):
    source: Literal["grounding", "generated", "label", "note"]
    text: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)


class SemanticJudgment(StrictModel):
    """One human or model-judge semantic decision for a generated row."""

    seed_id: str = Field(min_length=1)
    reasons: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    decision: Literal["accept", "reject", "review"]
    judge_kind: Literal["human", "llm"]
    judge_id: str = Field(min_length=1)
    prompt_or_guideline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_name: str | None = None
    reviewed_utc: str

    @field_validator("reasons")
    @classmethod
    def validate_reason_codes(cls, reasons: list[str]) -> list[str]:
        unknown = sorted(set(reasons) - SEMANTIC_REASON_CODES)
        if unknown:
            raise ValueError(f"Unknown semantic reason codes: {unknown}")
        return sorted(set(reasons))


class OracleRow(StrictModel):
    """Machine-checkable adjudication row; pending rows are not an oracle."""

    seed_id: str = Field(min_length=1)
    review_status: Literal["pending", "reviewed"] = "pending"
    expected_decision: Literal["", "accept", "reject", "review"] = ""
    expected_v02_reasons: list[str] = Field(default_factory=list)
    candidate_v02_reasons: list[str] = Field(default_factory=list)
    grounding_evidence: str = ""
    generated_evidence: str = ""
    analysis_note: str = ""
    reviewer_id: str = ""
    reviewed_utc: str = ""

    @field_validator("expected_v02_reasons", "candidate_v02_reasons")
    @classmethod
    def validate_reason_codes(cls, reasons: list[str]) -> list[str]:
        unknown = sorted(set(reasons) - SEMANTIC_REASON_CODES)
        if unknown:
            raise ValueError(f"Unknown semantic reason codes: {unknown}")
        return sorted(set(reasons))

    def assert_review_complete(self) -> None:
        if self.review_status != "reviewed":
            raise ValueError(f"Oracle row {self.seed_id} is still pending")
        if not self.expected_decision or not self.reviewer_id or not self.reviewed_utc:
            raise ValueError(f"Oracle row {self.seed_id} lacks review provenance")


class FrozenFile(StrictModel):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: int | None = Field(default=None, ge=0)


class FrozenManifest(StrictModel):
    manifest_version: Literal["v01"] = "v01"
    purpose: Literal["immutable_pipeline_smoke_evidence"] = (
        "immutable_pipeline_smoke_evidence"
    )
    source_run_id: str
    source_run_path: str
    created_utc: str
    files: list[FrozenFile]
    file_count: int = Field(ge=1)
    total_size_bytes: int = Field(ge=0)
    note: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return [json_safe(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError(f"Input must be JSONL or Parquet: {path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list")
    return [str(item) for item in parsed]


def load_oracle_csv(path: Path, *, require_complete: bool) -> dict[str, OracleRow]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows: dict[str, OracleRow] = {}
    for record in frame.to_dict(orient="records"):
        record["expected_v02_reasons"] = parse_json_list(
            record.pop("expected_v02_reasons_json", "")
        )
        record["candidate_v02_reasons"] = parse_json_list(
            record.pop("candidate_v02_reasons_json", "")
        )
        row = OracleRow.model_validate(record)
        if require_complete:
            row.assert_review_complete()
        if row.seed_id in rows:
            raise ValueError(f"Duplicate oracle seed_id: {row.seed_id}")
        rows[row.seed_id] = row
    return rows


def load_semantic_judgments(path: Path) -> dict[str, SemanticJudgment]:
    judgments: dict[str, SemanticJudgment] = {}
    for record in load_records(path):
        judgment = SemanticJudgment.model_validate(record)
        if judgment.seed_id in judgments:
            raise ValueError(f"Duplicate semantic judgment seed_id: {judgment.seed_id}")
        judgments[judgment.seed_id] = judgment
    return judgments


def row_count(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if suffix == ".csv":
        return len(pd.read_csv(path))
    return None
