"""Validate and route CFPB Seed v05.2 pipeline-only smoke dialogues.

Raw Data Designer output is never modified.  Every routed row inherits labels
programmatically from the prepared seed and remains explicitly unverified,
provisional, and benchmark-ineligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_PROJECT_ROOT_ENV = os.environ.get("FINDISPUTEEVAL_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_PROJECT_ROOT_ENV).resolve()
    if _PROJECT_ROOT_ENV
    else Path(__file__).resolve().parents[3]
)
SMOKE_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override"
DEFAULT_INPUT = SMOKE_ROOT / "prepared_inputs/nemo_seed_v052_pipeline_smoke_20.jsonl"
DEFAULT_INPUT_MANIFEST = SMOKE_ROOT / "prepared_inputs/pipeline_smoke_input_manifest.json"
DEFAULT_DISPOSITION = (
    PROJECT_ROOT
    / "dataset/curated/annotations/cfpb_seed_v05_audit"
    / "run_20260713T145423Z/privacy_qa/full_v052_v02"
    / "pipeline_privacy_disposition_v01.json"
)
DEFAULT_VALIDATED = SMOKE_ROOT / "validated/dialogues.jsonl"
DEFAULT_REJECTED = SMOKE_ROOT / "rejected/dialogues.jsonl"
DEFAULT_REVIEW = SMOKE_ROOT / "review/human_review_10.jsonl"
DEFAULT_REPORT = SMOKE_ROOT / "pipeline_smoke_validation_report.json"

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "account_number": re.compile(
        r"\b(?:account|acct|card|routing)\s*(?:number|no\.?|#)?\s*[:#-]?\s*\d{6,}\b",
        re.I,
    ),
    "long_digit_run": re.compile(r"(?<!\d)\d{9,}(?!\d)"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "url": re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I),
}
SENSITIVE_REQUEST_PATTERNS = [
    re.compile(
        r"\b(?:provide|send|share|tell me|enter|confirm)\b.{0,35}"
        r"\b(?:password|pin|cvv|security code|one[- ]time code|login|full account|"
        r"full card|social security|ssn)\b",
        re.I,
    ),
]
AUTHORITY_CLAIM_PATTERNS = [
    re.compile(
        r"\b(?:I|we)(?:'ll| will| can| have| just| am going to)?\s*"
        r"(?:file|submit|open|investigate|monitor|freeze|block|cancel|reverse|refund|"
        r"credit|escalate|resolve|contact)\b",
        re.I,
    ),
]
PROMISE_PATTERNS = [
    re.compile(r"\b(?:guarantee|promise|definitely|certainly)\b", re.I),
    re.compile(
        r"\b(?:will|should)\s+(?:be\s+)?(?:resolved|refunded|credited|reversed|completed)\b",
        re.I,
    ),
    re.compile(r"\bwithin\s+\d+\s+(?:business\s+)?days?\b", re.I),
]


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
    if path.suffix.lower() == ".parquet":
        return [json_safe(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError(f"Input must be JSONL or Parquet: {path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")


def parse_structured(value: Any) -> dict[str, Any]:
    value = json_safe(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def word_ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {
        tuple(words[index : index + n])
        for index in range(max(0, len(words) - n + 1))
    }


def assistant_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    )


def validate_row(
    row: dict[str, Any],
    seed_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    seed_id = str(row.get("seed_id", ""))
    seed = seed_by_id.get(seed_id)
    if seed is None:
        return ["unknown_or_missing_seed_id"], {}

    for column in ("product", "sub_product", "issue", "sub_issue"):
        if column in row and str(row.get(column, "")) != str(seed.get(column, "")):
            reasons.append(f"seed_provenance_changed:{column}")

    structured = parse_structured(row.get("dialogue"))
    prohibited_label_fields = {
        "label_product",
        "label_sub_product",
        "label_issue",
        "label_sub_issue",
        "product",
        "sub_product",
        "issue",
        "sub_issue",
    }
    if prohibited_label_fields & set(structured):
        reasons.append("model_generated_label_field")
    messages = structured.get("conversation")
    if not isinstance(messages, list) or len(messages) not in {4, 6, 8}:
        reasons.append("conversation_message_count_or_schema")
        messages = []
    expected_count = row.get("conversation_length")
    try:
        expected_count = int(expected_count)
    except (TypeError, ValueError):
        expected_count = None
    if expected_count is None or len(messages) != expected_count:
        reasons.append("conversation_length_mismatch")

    contents: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            reasons.append("message_not_object")
            continue
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected_role:
            reasons.append("roles_not_alternating")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            reasons.append("empty_message")
        else:
            contents.append(content)

    summary = structured.get("synthetic_case_summary")
    privacy_notes = structured.get("privacy_notes")
    if not isinstance(summary, str) or not summary.strip():
        reasons.append("missing_case_summary")
        summary = ""
    if not isinstance(privacy_notes, list):
        reasons.append("missing_privacy_notes")
        privacy_notes = []
    generated_text = "\n".join(contents + [summary] + [str(item) for item in privacy_notes])
    assistant_generated = assistant_text(messages)

    for name, pattern in PII_PATTERNS.items():
        if pattern.search(generated_text):
            reasons.append(f"regex_pii:{name}")
    if re.search(r"\bCFPB\b|consumerfinance\.gov", generated_text, re.I):
        reasons.append("cfpb_reference")
    if seed_id and seed_id.lower() in generated_text.lower():
        reasons.append("seed_id_copied")
    if "[PII_CANDIDATE]" in generated_text:
        reasons.append("pii_candidate_placeholder_copied")
    if any(pattern.search(assistant_generated) for pattern in SENSITIVE_REQUEST_PATTERNS):
        reasons.append("assistant_sensitive_credential_request")
    if any(pattern.search(assistant_generated) for pattern in AUTHORITY_CLAIM_PATTERNS):
        reasons.append("assistant_unavailable_action_claim")
    if any(pattern.search(assistant_generated) for pattern in PROMISE_PATTERNS):
        reasons.append("assistant_outcome_or_deadline_promise")

    source_ngrams = word_ngrams(str(seed.get("generation_grounding_excerpt", "")))
    generated_ngrams = word_ngrams(generated_text)
    overlap = source_ngrams & generated_ngrams
    if overlap:
        reasons.append("source_8gram_copied")

    reasons = sorted(set(reasons))
    routed = dict(row)
    routed["programmatic_labels"] = {
        "product": str(seed.get("product", "")),
        "sub_product": str(seed.get("sub_product", "")),
        "issue": str(seed.get("issue", "")),
        "sub_issue": str(seed.get("sub_issue", "")),
    }
    routed.update(
        {
            "pipeline_smoke_only": True,
            "provisional": True,
            "privacy_verified": False,
            "benchmark_eligible": False,
            "publication_eligible": False,
            "may_enter_final_dataset": False,
            "distribution_calibration_eligible": False,
        }
    )
    routed["pipeline_smoke_validation"] = {
        "passed": not reasons,
        "reasons": reasons,
        "source_8gram_overlap_count": len(overlap),
        "labels_attached_programmatically": True,
        "privacy_verified": False,
        "benchmark_eligible": False,
    }
    return reasons, routed


def validate_and_route(
    *,
    raw_output: Path,
    prepared_input: Path,
    input_manifest: Path,
    disposition_path: Path,
    validated_path: Path,
    rejected_path: Path,
    review_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    policy = json.loads(input_manifest.read_text(encoding="utf-8"))
    if policy.get("purpose") != "cfpb_seed_v052_pipeline_smoke_only":
        raise ValueError("Input manifest is not a v05.2 pipeline-smoke manifest")
    required_policy = {
        "pipeline_smoke_only": True,
        "provisional": True,
        "privacy_verified": False,
        "benchmark_eligible": False,
        "formal_pilot_allowed": False,
    }
    for key, expected in required_policy.items():
        if policy.get("policy", {}).get(key) != expected:
            raise ValueError(f"Input manifest policy mismatch: {key}")
    if policy["output"]["sha256"] != sha256_file(prepared_input):
        raise ValueError("Prepared input hash mismatch")
    if policy.get("pipeline_privacy_disposition_sha256") != sha256_file(disposition_path):
        raise ValueError("Pipeline privacy disposition hash mismatch")
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    if not (
        disposition.get("pipeline_smoke_only_passed") is True
        and disposition.get("manual_review_performed") is False
        and disposition.get("release_clearance_passed") is False
        and disposition.get("benchmark_eligible") is False
    ):
        raise ValueError("Privacy disposition does not permit pipeline-only smoke")

    raw_resolved = raw_output.resolve()
    if SMOKE_ROOT.resolve() not in (raw_resolved, *raw_resolved.parents):
        raise ValueError("Raw output must remain under the smoke-only workspace")
    seeds = load_records(prepared_input)
    raw = load_records(raw_output)
    expected_rows = int(policy.get("selected_rows", -1))
    if not 10 <= len(raw) <= 20 or len(raw) != expected_rows:
        raise ValueError(
            f"Raw pipeline-smoke output must contain exactly {expected_rows} rows; received {len(raw)}"
        )
    seed_by_id = {str(row["seed_id"]): row for row in seeds}
    if len(seed_by_id) != len(seeds):
        raise ValueError("Prepared smoke input contains duplicate seed_id values")

    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    seen_ids: list[str] = []
    for row in raw:
        reasons, routed = validate_row(row, seed_by_id)
        if not routed:
            routed = dict(row)
            routed.update(
                {
                    "pipeline_smoke_only": True,
                    "provisional": True,
                    "privacy_verified": False,
                    "benchmark_eligible": False,
                    "pipeline_smoke_validation": {
                        "passed": False,
                        "reasons": reasons,
                        "privacy_verified": False,
                        "benchmark_eligible": False,
                    },
                }
            )
        seed_id = str(row.get("seed_id", ""))
        seen_ids.append(seed_id)
        if reasons:
            failed.append(routed)
            reason_counts.update(reasons)
        else:
            passed.append(routed)
    if len(set(seen_ids)) != len(seen_ids):
        raise ValueError("Raw output contains duplicate seed_id values")
    if set(seen_ids) != set(seed_by_id):
        raise ValueError("Raw output seed coverage differs from prepared input")

    write_jsonl(validated_path, passed)
    write_jsonl(rejected_path, failed)
    review_rows = sorted(
        passed + failed,
        key=lambda row: hashlib.sha256(
            f"human_review|{row.get('seed_id', '')}".encode("utf-8")
        ).hexdigest(),
    )[: min(10, len(raw))]
    write_jsonl(review_path, review_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "report_version": "v01",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "cfpb_seed_v052_pipeline_smoke_only",
        "pipeline_plumbing_passed": len(passed) + len(failed) == len(raw),
        "raw_path": raw_resolved.relative_to(SMOKE_ROOT.resolve()).as_posix(),
        "raw_sha256": sha256_file(raw_resolved),
        "prepared_input_sha256": sha256_file(prepared_input),
        "input_manifest_sha256": sha256_file(input_manifest),
        "pipeline_privacy_disposition_sha256": sha256_file(disposition_path),
        "raw_rows": len(raw),
        "validated_rows": len(passed),
        "rejected_rows": len(failed),
        "rejection_rate": len(failed) / len(raw),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "human_review_rows": len(review_rows),
        "outputs": {
            "validated": {
                "path": validated_path.resolve().relative_to(SMOKE_ROOT.resolve()).as_posix(),
                "rows": len(passed),
                "sha256": sha256_file(validated_path),
            },
            "rejected": {
                "path": rejected_path.resolve().relative_to(SMOKE_ROOT.resolve()).as_posix(),
                "rows": len(failed),
                "sha256": sha256_file(rejected_path),
            },
            "human_review": {
                "path": review_path.resolve().relative_to(SMOKE_ROOT.resolve()).as_posix(),
                "rows": len(review_rows),
                "sha256": sha256_file(review_path),
            },
        },
        "policy": {
            "pipeline_smoke_only": True,
            "provisional": True,
            "privacy_verified": False,
            "benchmark_eligible": False,
            "publication_eligible": False,
            "may_enter_final_dataset": False,
            "distribution_calibration_eligible": False,
            "formal_pilot_allowed": False,
        },
        "note": (
            "Schema, regex, authority, promise, credential, URL, and copying checks are "
            "engineering validators. The source privacy review was not performed."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-output",
        "--raw-jsonl",
        dest="raw_output",
        type=Path,
        required=True,
        help="Untouched Data Designer final JSONL or Parquet artifact.",
    )
    parser.add_argument("--prepared-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--privacy-disposition", type=Path, default=DEFAULT_DISPOSITION)
    parser.add_argument("--validated", type=Path, default=DEFAULT_VALIDATED)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_and_route(
        raw_output=args.raw_output,
        prepared_input=args.prepared_input,
        input_manifest=args.input_manifest,
        disposition_path=args.privacy_disposition,
        validated_path=args.validated,
        rejected_path=args.rejected,
        review_path=args.review,
        report_path=args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
