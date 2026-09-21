"""Offline validator v02 for CFPB Seed v05.2 pipeline-smoke dialogues.

Raw provider output is immutable.  This validator writes to an external
revalidation directory, uses Pydantic for generated schemas, and treats
semantic judgments as explicit evidence rather than encoding them in broad
regular expressions.  Every output remains privacy-unverified and
benchmark-ineligible.
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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

try:
    from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
        OracleRow,
        SemanticJudgment,
        json_safe,
        load_oracle_csv,
        load_records,
        load_semantic_judgments,
        sha256_file,
        write_jsonl,
    )
    from scripts.generation.nemo_data_designer.prepare_cfpb_seed_v052_pipeline_smoke_v02 import (
        DEFAULT_MIN_GROUNDING_TOKENS,
        grounding_features,
    )
except ModuleNotFoundError:  # Direct execution from this script's directory.
    from cfpb_v052_pipeline_smoke_v02_common import (  # type: ignore[no-redef]
        OracleRow,
        SemanticJudgment,
        json_safe,
        load_oracle_csv,
        load_records,
        load_semantic_judgments,
        sha256_file,
        write_jsonl,
    )
    from prepare_cfpb_seed_v052_pipeline_smoke_v02 import (  # type: ignore[no-redef]
        DEFAULT_MIN_GROUNDING_TOKENS,
        grounding_features,
    )


_PROJECT_ROOT_ENV = os.environ.get("FINDISPUTEEVAL_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_PROJECT_ROOT_ENV).resolve()
    if _PROJECT_ROOT_ENV
    else Path(__file__).resolve().parents[3]
)
SMOKE_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override"
RUN_ROOT = SMOKE_ROOT / "run_20260722T135306Z"
REVALIDATION_ROOT = (
    SMOKE_ROOT / "revalidations/validator_v02/source_run_20260722T135306Z"
)
DEFAULT_RAW = RUN_ROOT / "raw/data_designer/dataset/parquet-files/batch_00000.parquet"
DEFAULT_INPUT = RUN_ROOT / "prepared_inputs/nemo_seed_v052_pipeline_smoke_20.jsonl"
DEFAULT_INPUT_MANIFEST = RUN_ROOT / "prepared_inputs/pipeline_smoke_input_manifest.json"
DEFAULT_DISPOSITION = (
    PROJECT_ROOT
    / "dataset/curated/annotations/cfpb_seed_v05_audit"
    / "run_20260713T145423Z/privacy_qa/full_v052_v02"
    / "pipeline_privacy_disposition_v01.json"
)
DEFAULT_VALIDATED = REVALIDATION_ROOT / "validated/dialogues.jsonl"
DEFAULT_REJECTED = REVALIDATION_ROOT / "rejected/dialogues.jsonl"
DEFAULT_REVIEW = REVALIDATION_ROOT / "review/dialogues.jsonl"
DEFAULT_REPORT = REVALIDATION_ROOT / "validation_report_v02.json"

# These patterns are intentionally narrow deterministic integrity checks.  NER
# and semantic consistency belong to Presidio/spaCy or a calibrated judge.
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "url": re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I),
    "long_digit_run": re.compile(r"(?<!\d)\d{9,}(?!\d)"),
}
SENSITIVE_REQUEST = re.compile(
    r"\b(?:provide|send|share|tell me|enter|confirm)\b.{0,35}"
    r"\b(?:password|pin|cvv|security code|one[- ]time code|login|full account|"
    r"full card|social security|ssn)\b",
    re.I,
)
FIRST_PERSON_ACTION = re.compile(
    r"\b(?:I|we)(?:'ll| will| can| have| am going to)\s*"
    r"(?:file|submit|open|investigate|monitor|freeze|block|cancel|reverse|refund|"
    r"credit|escalate|resolve|contact)\b",
    re.I,
)
EXPLICIT_RIGHTS_CLAIM = re.compile(r"\byou (?:have|retain) the right to\b", re.I)
DIRECT_PROMISE = re.compile(
    r"\b(?:(?:I|we|they)(?:'ll| will)|your (?:refund|credit) should)\b.{0,45}"
    r"\b(?:refund|credit|reverse|resolve|complete|arrive|completed|refunded|credited|"
    r"reversed|resolved)\b",
    re.I,
)
ABSOLUTE_PROMISE = re.compile(r"\b(?:guarantee|promise|definitely|certainly)\b", re.I)

STRUCTURAL_REASONS = frozenset(
    {
        "conversation_schema_invalid",
        "conversation_length_mismatch",
        "model_generated_label_field",
        "unknown_or_missing_seed_id",
    }
)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Dialogue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conversation: list[Message]
    synthetic_case_summary: str = Field(min_length=1)
    privacy_notes: list[str]

    @field_validator("conversation")
    @classmethod
    def validate_conversation(cls, messages: list[Message]) -> list[Message]:
        if len(messages) not in {4, 6, 8}:
            raise ValueError("conversation must contain 4, 6, or 8 messages")
        for index, message in enumerate(messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError("conversation roles must alternate from user")
        return messages


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


def validate_row(
    row: dict[str, Any],
    seed_by_id: dict[str, dict[str, Any]],
    semantic_judgment: SemanticJudgment | None = None,
) -> tuple[list[str], dict[str, Any], str]:
    """Return reasons, immutable-derived routed row, and route."""

    reasons: list[str] = []
    seed_id = str(row.get("seed_id", ""))
    seed = seed_by_id.get(seed_id)
    if seed is None:
        routed = dict(row)
        reasons = ["unknown_or_missing_seed_id"]
        return reasons, routed, "rejected"

    for column in ("product", "sub_product", "issue", "sub_issue"):
        if column in row and str(row.get(column, "")) != str(seed.get(column, "")):
            reasons.append(f"seed_provenance_changed:{column}")

    structured = parse_structured(row.get("dialogue"))
    prohibited_labels = {
        "label_product", "label_sub_product", "label_issue", "label_sub_issue",
        "product", "sub_product", "issue", "sub_issue",
    }
    if prohibited_labels & set(structured):
        reasons.append("model_generated_label_field")
    try:
        dialogue = Dialogue.model_validate(structured)
    except ValidationError:
        reasons.append("conversation_schema_invalid")
        dialogue = None

    try:
        expected_count = int(row.get("conversation_length"))
    except (TypeError, ValueError):
        expected_count = -1
    if dialogue is None or len(dialogue.conversation) != expected_count:
        reasons.append("conversation_length_mismatch")

    generated_text = ""
    assistant_text = ""
    if dialogue is not None:
        generated_text = "\n".join(
            [message.content for message in dialogue.conversation]
            + [dialogue.synthetic_case_summary]
            + dialogue.privacy_notes
        )
        assistant_text = "\n".join(
            message.content
            for message in dialogue.conversation
            if message.role == "assistant"
        )
        for name, pattern in PII_PATTERNS.items():
            if pattern.search(generated_text):
                reasons.append(f"regex_pii:{name}")
        if re.search(r"\bCFPB\b|consumerfinance\.gov", generated_text, re.I):
            reasons.append("cfpb_reference")
        if seed_id.lower() in generated_text.lower():
            reasons.append("seed_id_copied")
        if "[PII_CANDIDATE]" in generated_text:
            reasons.append("pii_candidate_placeholder_copied")
        if SENSITIVE_REQUEST.search(assistant_text):
            reasons.append("assistant_sensitive_credential_request")
        if FIRST_PERSON_ACTION.search(assistant_text):
            reasons.append("assistant_unavailable_action_claim")
        if EXPLICIT_RIGHTS_CLAIM.search(assistant_text):
            reasons.append("legal_or_rights_claim")
        if DIRECT_PROMISE.search(assistant_text) or ABSOLUTE_PROMISE.search(assistant_text):
            reasons.append("assistant_outcome_or_deadline_promise")

    source_ngrams = word_ngrams(str(seed.get("generation_grounding_excerpt", "")))
    overlap = source_ngrams & word_ngrams(generated_text)
    if overlap:
        reasons.append("source_8gram_copied")

    features = grounding_features(str(seed.get("generation_grounding_excerpt", "")))
    if features["grounding_non_placeholder_tokens"] < DEFAULT_MIN_GROUNDING_TOKENS:
        reasons.append("insufficient_grounding")

    if semantic_judgment is not None:
        reasons.extend(semantic_judgment.reasons)
    reasons = sorted(set(reasons))

    hard_failure = bool(reasons)
    if semantic_judgment is None and not hard_failure:
        route = "review"
    elif semantic_judgment is not None and semantic_judgment.decision == "review":
        route = "review"
    elif hard_failure or (
        semantic_judgment is not None and semantic_judgment.decision == "reject"
    ):
        route = "rejected"
    else:
        route = "validated"

    routed = dict(row)
    routed["programmatic_labels"] = {
        column: str(seed.get(column, ""))
        for column in ("product", "sub_product", "issue", "sub_issue")
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
    routed["pipeline_smoke_validation_v02"] = {
        "route": route,
        "reasons": reasons,
        "grounding_features": features,
        "source_8gram_overlap_count": len(overlap),
        "semantic_judgment_present": semantic_judgment is not None,
        "semantic_judgment": (
            semantic_judgment.model_dump(mode="json")
            if semantic_judgment is not None
            else None
        ),
        "privacy_verified": False,
        "benchmark_eligible": False,
    }
    return reasons, routed, route


def evaluate_against_oracle(
    predicted: dict[str, set[str]],
    oracle: dict[str, OracleRow],
    predicted_routes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Measure reason-level judge performance using scikit-learn."""

    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Complete-oracle evaluation requires scikit-learn; install the v02 requirements."
        ) from exc

    seed_ids = sorted(oracle)
    classes = sorted(
        set().union(*(set(oracle[seed_id].expected_v02_reasons) for seed_id in seed_ids))
        | set().union(*(predicted.get(seed_id, set()) for seed_id in seed_ids))
    )
    if not classes:
        reason_metrics = {
            "micro_precision": 1.0,
            "micro_recall": 1.0,
            "micro_f1": 1.0,
        }
    else:
        encoder = MultiLabelBinarizer(classes=classes)
        expected_matrix = encoder.fit_transform(
            [oracle[seed_id].expected_v02_reasons for seed_id in seed_ids]
        )
        predicted_matrix = encoder.transform(
            [sorted(predicted.get(seed_id, set())) for seed_id in seed_ids]
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            expected_matrix, predicted_matrix, average="micro", zero_division=0
        )
        reason_metrics = {
            "micro_precision": float(precision),
            "micro_recall": float(recall),
            "micro_f1": float(f1),
        }
    exact = sum(
        predicted.get(seed_id, set()) == set(oracle[seed_id].expected_v02_reasons)
        for seed_id in seed_ids
    ) / len(seed_ids)
    result: dict[str, Any] = {
        "rows": len(seed_ids),
        "classes": classes,
        **reason_metrics,
        "exact_set_accuracy": exact,
    }
    if predicted_routes is not None:
        unresolved = [
            seed_id
            for seed_id in seed_ids
            if oracle[seed_id].expected_decision not in {"accept", "reject"}
        ]
        if unresolved:
            raise ValueError(
                f"Decision metrics require accept/reject oracle rows: {unresolved}"
            )
        expected_decisions = [
            int(oracle[seed_id].expected_decision == "reject")
            for seed_id in seed_ids
        ]
        predicted_decisions = [
            int(predicted_routes.get(seed_id) == "rejected")
            for seed_id in seed_ids
        ]
        decision_precision, decision_recall, decision_f1, _ = (
            precision_recall_fscore_support(
                expected_decisions,
                predicted_decisions,
                average="binary",
                zero_division=0,
            )
        )
        result["decision_reject_precision"] = float(decision_precision)
        result["decision_reject_recall"] = float(decision_recall)
        result["decision_reject_f1"] = float(decision_f1)
        result["decision_accuracy"] = float(
            accuracy_score(expected_decisions, predicted_decisions)
        )
    return result


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
    semantic_judgments_path: Path | None = None,
    oracle_path: Path | None = None,
) -> dict[str, Any]:
    policy = json.loads(input_manifest.read_text(encoding="utf-8"))
    if policy.get("purpose") != "cfpb_seed_v052_pipeline_smoke_only":
        raise ValueError("Input manifest is not a v05.2 pipeline-smoke manifest")
    for key, expected in {
        "pipeline_smoke_only": True,
        "provisional": True,
        "privacy_verified": False,
        "benchmark_eligible": False,
        "formal_pilot_allowed": False,
    }.items():
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
    seed_by_id = {str(row["seed_id"]): row for row in seeds}
    if len(seed_by_id) != len(seeds):
        raise ValueError("Prepared input contains duplicate seed_id values")
    raw_ids = [str(row.get("seed_id", "")) for row in raw]
    if len(set(raw_ids)) != len(raw_ids) or set(raw_ids) != set(seed_by_id):
        raise ValueError("Raw output seed coverage differs from prepared input")

    judgments = (
        load_semantic_judgments(semantic_judgments_path)
        if semantic_judgments_path is not None
        else {}
    )
    unknown_judgments = sorted(set(judgments) - set(seed_by_id))
    if unknown_judgments:
        raise ValueError(f"Semantic judgments contain unknown seed IDs: {unknown_judgments}")

    routed_outputs: dict[str, list[dict[str, Any]]] = {
        "validated": [], "rejected": [], "review": []
    }
    reason_counts: Counter[str] = Counter()
    predicted: dict[str, set[str]] = {}
    predicted_routes: dict[str, str] = {}
    for row in raw:
        seed_id = str(row["seed_id"])
        reasons, routed, route = validate_row(row, seed_by_id, judgments.get(seed_id))
        routed_outputs[route].append(routed)
        reason_counts.update(reasons)
        predicted[seed_id] = set(reasons)
        predicted_routes[seed_id] = route

    for path, key in (
        (validated_path, "validated"),
        (rejected_path, "rejected"),
        (review_path, "review"),
    ):
        write_jsonl(path, routed_outputs[key])

    oracle_metrics = None
    if oracle_path is not None:
        oracle = load_oracle_csv(oracle_path, require_complete=True)
        if set(oracle) != set(seed_by_id):
            raise ValueError("Complete oracle coverage differs from prepared input")
        oracle_metrics = evaluate_against_oracle(
            predicted, oracle, predicted_routes=predicted_routes
        )

    report: dict[str, Any] = {
        "report_version": "v02",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "offline_revalidation_of_cfpb_seed_v052_pipeline_smoke",
        "source_run_immutable": True,
        "raw_path": raw_resolved.relative_to(SMOKE_ROOT.resolve()).as_posix(),
        "raw_sha256": sha256_file(raw_resolved),
        "prepared_input_sha256": sha256_file(prepared_input),
        "input_manifest_sha256": sha256_file(input_manifest),
        "pipeline_privacy_disposition_sha256": sha256_file(disposition_path),
        "raw_rows": len(raw),
        "validated_rows": len(routed_outputs["validated"]),
        "rejected_rows": len(routed_outputs["rejected"]),
        "review_rows": len(routed_outputs["review"]),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "semantic_judgment_rows": len(judgments),
        "semantic_coverage_complete": len(judgments) == len(raw),
        "oracle_metrics": oracle_metrics,
        "outputs": {},
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
            "Rows without a semantic judgment are routed to review unless a deterministic "
            "failure already rejects them. Candidate oracle prefill is never used as truth."
        ),
    }
    for name, path in (
        ("validated", validated_path), ("rejected", rejected_path), ("review", review_path)
    ):
        report["outputs"][name] = {
            "path": path.resolve().relative_to(SMOKE_ROOT.resolve()).as_posix(),
            "rows": len(routed_outputs[name]),
            "sha256": sha256_file(path),
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--prepared-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--privacy-disposition", type=Path, default=DEFAULT_DISPOSITION)
    parser.add_argument("--validated", type=Path, default=DEFAULT_VALIDATED)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--semantic-judgments", type=Path)
    parser.add_argument("--oracle", type=Path)
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
        semantic_judgments_path=args.semantic_judgments,
        oracle_path=args.oracle,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
