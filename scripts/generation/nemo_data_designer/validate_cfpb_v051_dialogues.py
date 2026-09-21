"""Validate and route provisional CFPB v05.1 smoke-test dialogues.

Raw input is never modified. Passing and failing rows are written separately,
and every row remains explicitly ineligible for benchmark use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SMOKE_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v051_candidate"
DEFAULT_INPUT = SMOKE_ROOT / "prepared_inputs/nemo_seed_v051_provisional_smoke_20.jsonl"
DEFAULT_INPUT_MANIFEST = SMOKE_ROOT / "prepared_inputs/provisional_smoke_input_manifest.json"
DEFAULT_VALIDATED = SMOKE_ROOT / "validated/dialogues.jsonl"
DEFAULT_REJECTED = SMOKE_ROOT / "rejected/dialogues.jsonl"
DEFAULT_REPORT = SMOKE_ROOT / "provisional_validation_report.json"

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "account_number": re.compile(r"\b(?:account|acct|card)\s*(?:number|no\.?|#)\s*\d{6,}\b", re.I),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Normalize Arrow/NumPy nested values into stable JSON-compatible types."""
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


def load_records(path: Path) -> list[dict]:
    """Load a Data Designer final dataset without changing the raw artifact."""
    if path.suffix.lower() == ".parquet":
        return [json_safe(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError(f"Raw Data Designer output must be JSONL or Parquet: {path}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def parse_structured(value: Any) -> dict:
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
    return {tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1))}


def validate_row(row: dict, seed_by_id: dict[str, dict]) -> tuple[list[str], dict]:
    reasons: list[str] = []
    seed_id = str(row.get("seed_id", ""))
    seed = seed_by_id.get(seed_id)
    if seed is None:
        return ["unknown_or_missing_seed_id"], {}

    structured = parse_structured(row.get("conversation"))
    messages = structured.get("conversation")
    if not isinstance(messages, list) or len(messages) not in {4, 6, 8}:
        reasons.append("conversation_message_count_or_schema")
        messages = []
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

    generated_text = "\n".join(contents)
    if str(structured.get("label_product", "")) != str(seed.get("product", "")):
        reasons.append("product_label_mismatch")
    if str(structured.get("label_issue", "")) != str(seed.get("issue", "")):
        reasons.append("issue_label_mismatch")
    if str(structured.get("label_sub_issue", "")) != str(seed.get("sub_issue", "")):
        reasons.append("sub_issue_label_mismatch")
    if not isinstance(structured.get("synthetic_case_summary"), str):
        reasons.append("missing_case_summary")
    if not isinstance(structured.get("privacy_notes"), list):
        reasons.append("missing_privacy_notes")

    for name, pattern in PII_PATTERNS.items():
        if pattern.search(generated_text):
            reasons.append(f"regex_pii:{name}")
    if re.search(r"\bCFPB\b|consumerfinance\.gov", generated_text, re.I):
        reasons.append("cfpb_reference")
    if seed_id and seed_id.lower() in generated_text.lower():
        reasons.append("seed_id_copied")

    source_ngrams = word_ngrams(str(seed.get("seed_narrative_excerpt", "")))
    generated_ngrams = word_ngrams(generated_text)
    overlap = source_ngrams & generated_ngrams
    if overlap:
        reasons.append("source_8gram_copied")

    reasons = sorted(set(reasons))
    routed = dict(row)
    routed["provisional_validation"] = {
        "passed": not reasons,
        "reasons": reasons,
        "source_8gram_overlap_count": len(overlap),
        "benchmark_eligible": False,
        "publication_eligible": False,
        "may_enter_final_dataset": False,
        "distribution_calibration_eligible": False,
    }
    return reasons, routed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-jsonl",
        dest="raw_output",
        type=Path,
        required=True,
        help="Untouched Data Designer final JSONL or Parquet artifact.",
    )
    parser.add_argument("--prepared-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--validated", type=Path, default=DEFAULT_VALIDATED)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    policy = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    if policy.get("purpose") != "engineering_smoke_test_only":
        raise ValueError("Input manifest is not a provisional smoke manifest")
    if policy["output"]["sha256"] != sha256_file(args.prepared_input):
        raise ValueError("Prepared input hash mismatch")
    raw_resolved = args.raw_output.resolve()
    if SMOKE_ROOT.resolve() not in (raw_resolved, *raw_resolved.parents):
        raise ValueError("Raw output must remain under the smoke-only workspace")

    seeds = load_records(args.prepared_input)
    raw = load_records(args.raw_output)
    if not 10 <= len(raw) <= 20:
        raise ValueError("Raw provisional output must contain 10 to 20 rows")
    seed_by_id = {str(row["seed_id"]): row for row in seeds}
    passed: list[dict] = []
    failed: list[dict] = []
    reason_counts: Counter[str] = Counter()
    for row in raw:
        reasons, routed = validate_row(row, seed_by_id)
        if reasons:
            failed.append(routed)
            reason_counts.update(reasons)
        else:
            passed.append(routed)
    write_jsonl(args.validated, passed)
    write_jsonl(args.rejected, failed)
    report = {
        "report_version": "v01",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "engineering_smoke_test_only",
        "raw_path": str(raw_resolved.relative_to(SMOKE_ROOT.resolve())),
        "raw_sha256": sha256_file(raw_resolved),
        "raw_rows": len(raw),
        "validated_rows": len(passed),
        "rejected_rows": len(failed),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "benchmark_eligible": False,
        "publication_eligible": False,
        "may_enter_final_dataset": False,
        "distribution_calibration_eligible": False,
        "note": "Regex/schema/copying checks do not replace final NER and human privacy review.",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
