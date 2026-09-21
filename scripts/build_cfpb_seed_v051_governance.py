"""Build the accepted CFPB Seed v05.1 governance inputs and decision record v03.

This script never edits the frozen EDA run, the v02 decision record, or Seed v05.
It derives versioned row overrides and regex-risk clearance evidence from the
completed audit workspace, then mints a new build-gated decision record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "run_20260713T145423Z"
EDA_RUN = (
    PROJECT_ROOT
    / "outputs/data_pipeline/cfpb_seed_source_eda/eda_v051"
    / RUN_ID
)
AUDIT_ROOT = (
    PROJECT_ROOT
    / "dataset/curated/annotations/cfpb_seed_v05_audit"
    / RUN_ID
)

PARENT_DECISION = AUDIT_ROOT / "seed_v05_decision_record_v02.json"
QUALITY_AUDIT = AUDIT_ROOT / "quality_audit_master.csv"
PII_AUDIT = AUDIT_ROOT / "pii_presidio_validation_master.csv"
FUZZY_ADJUDICATION = AUDIT_ROOT / "fuzzy_duplicate_audit_master.csv"
ROW_OVERRIDES = AUDIT_ROOT / "seed_v051_row_overrides_v01.csv"
PII_CLEARANCE = AUDIT_ROOT / "pii_release_clearance_v01.csv"
DECISION_TABLE = AUDIT_ROOT / "seed_v051_decision_table_v03.csv"
DECISION_RECORD = AUDIT_ROOT / "seed_v051_decision_record_v03.json"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def stable_override_id(complaint_id: str, reason_code: str) -> str:
    payload = f"cfpb_seed_v051|{complaint_id}|{reason_code}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def build_governance_inputs(decided_utc: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus = pd.read_parquet(
        EDA_RUN / "analysis_ready_corpus.parquet",
        columns=["Complaint ID", "canonical_text_sha256", "pii_regex_risk"],
    )
    corpus["Complaint ID"] = corpus["Complaint ID"].astype("string")
    corpus_index = corpus.set_index("Complaint ID", drop=False)

    quality = read_csv(QUALITY_AUDIT)
    quality_excludes = quality.loc[quality["manual_quality"].eq("exclude")].copy()
    pii = read_csv(PII_AUDIT)
    regex_review = pii.loc[pii["pii_regex_types"].str.strip().ne("")].copy()

    corpus_regex_ids = set(
        corpus.loc[corpus["pii_regex_risk"].fillna(False), "Complaint ID"].astype(str)
    )
    audited_regex_ids = set(regex_review["Complaint ID"].astype(str))
    if corpus_regex_ids != audited_regex_ids:
        missing = sorted(corpus_regex_ids - audited_regex_ids)
        extra = sorted(audited_regex_ids - corpus_regex_ids)
        raise ValueError(
            "PII regex clearance must census the complete corpus risk set; "
            f"missing={missing}, extra={extra}"
        )

    clearance_rows: list[dict[str, object]] = []
    for row in regex_review.to_dict(orient="records"):
        complaint_id = str(row["Complaint ID"])
        manual = str(row["manual_pii_present"]).strip().lower()
        if manual == "no":
            status, action = "cleared_by_completed_audit", "allow_after_regex_redaction"
        elif manual == "yes":
            status, action = "blocked_confirmed_pii", "exclude"
        else:
            status, action = "blocked_uncertain", "exclude"
        clearance_rows.append(
            {
                "clearance_id": stable_override_id(complaint_id, "pii_clearance"),
                "Complaint ID": complaint_id,
                "canonical_text_sha256": str(
                    corpus_index.at[complaint_id, "canonical_text_sha256"]
                ),
                "pii_regex_types": str(row["pii_regex_types"]),
                "presidio_entity_types": str(row["presidio_entity_types"]),
                "presidio_max_score": str(row["presidio_max_score"]),
                "audit_manual_pii_present": manual,
                "audit_manual_pii_types": str(row["manual_pii_types"]),
                "clearance_status": status,
                "release_action": action,
                "source_audit": PII_AUDIT.name,
                "source_annotation_id": str(row["annotation_id"]),
                "source_labeler": "llm:claude-fable-5",
                "policy_approved_by": "chang",
                "decided_utc": decided_utc,
                "notes": (
                    "Completed audit label is governance evidence but not a substitute "
                    "for final full-release Presidio/NER QA."
                ),
            }
        )
    clearance = pd.DataFrame(clearance_rows).sort_values("Complaint ID")

    override_rows: list[dict[str, object]] = []
    for row in quality_excludes.to_dict(orient="records"):
        complaint_id = str(row["Complaint ID"])
        override_rows.append(
            {
                "override_id": stable_override_id(complaint_id, "quality_audit_exclude"),
                "Complaint ID": complaint_id,
                "canonical_text_sha256": str(
                    corpus_index.at[complaint_id, "canonical_text_sha256"]
                ),
                "action": "exclude",
                "reason_code": "quality_audit_exclude",
                "source_audit": QUALITY_AUDIT.name,
                "source_annotation_id": str(row["annotation_id"]),
                "source_labeler": "llm:claude-fable-5",
                "policy_approved_by": "chang",
                "decided_utc": decided_utc,
                "notes": str(row["exclusion_reason"]),
            }
        )
    for row in clearance.loc[clearance["release_action"].eq("exclude")].to_dict(
        orient="records"
    ):
        complaint_id = str(row["Complaint ID"])
        override_rows.append(
            {
                "override_id": stable_override_id(complaint_id, "pii_clearance_block"),
                "Complaint ID": complaint_id,
                "canonical_text_sha256": str(row["canonical_text_sha256"]),
                "action": "exclude",
                "reason_code": "pii_clearance_block",
                "source_audit": PII_AUDIT.name,
                "source_annotation_id": str(row["source_annotation_id"]),
                "source_labeler": str(row["source_labeler"]),
                "policy_approved_by": "chang",
                "decided_utc": decided_utc,
                "notes": (
                    f"Blocked from all text-bearing releases: {row['clearance_status']}; "
                    f"types={row['audit_manual_pii_types']}"
                ),
            }
        )
    overrides = pd.DataFrame(override_rows).sort_values(
        ["Complaint ID", "reason_code"]
    )
    if overrides["override_id"].duplicated().any():
        raise ValueError("Duplicate override IDs")
    if set(overrides["action"]) != {"exclude"}:
        raise ValueError("v01 overrides must be fail-closed exclusions")

    write_csv(clearance, PII_CLEARANCE)
    write_csv(overrides, ROW_OVERRIDES)
    return overrides, clearance


def build_decisions(decided_utc: str) -> pd.DataFrame:
    parent = json.loads(PARENT_DECISION.read_text(encoding="utf-8"))
    parent_rows = {row["decision_area"]: row for row in parent["decisions"]}

    rules = {
        area: json.loads(parent_rows[area]["machine_rule_json"])
        for area in parent_rows
    }
    rules["very_short_long"].update(
        {
            "stress_reserve_from_retained": ["very_short", "unusual_format"],
        }
    )
    rules["pii"].update(
        {
            "clearance_path": PII_CLEARANCE.name,
            "require_clearance_for_regex_risk": True,
            "confirmed_pii_action": "exclude",
            "uncleared_action": "exclude",
            "final_full_release_ner_qa_required": True,
        }
    )
    rules["sampling_quotas"].update(
        {
            "enrichment_shortfall_policy": {
                "mode": "allow_if_exhausted_after_eligibility_and_caps",
                "minimum_selected_by_frame": {
                    "prepaid_historical_enrichment": 300,
                    "zelle_historical_enrichment": 300,
                    "zelle_outside_population_enrichment": 45,
                    "zelle_prepaid_enrichment_overlap": 1,
                },
                "fail_on_unexplained_shortfall": True,
            },
            "stress_population_only": True,
            "stress_quotas": {
                "pii_regex_risk": 25,
                "redaction_above_threshold": 35,
                "language_status_not_allowed": 40,
                "very_long": 50,
                "very_short": 20,
                "unusual_format": 30,
            },
            "stress_priority": [
                "pii_regex_risk",
                "redaction_above_threshold",
                "language_status_not_allowed",
                "very_long",
                "very_short",
                "unusual_format",
            ],
        }
    )
    rules["row_overrides"] = {
        "path": ROW_OVERRIDES.name,
        "required": True,
        "allowed_actions": ["exclude", "stress_only"],
        "missing_complaint_id_action": "error",
        "hash_mismatch_action": "error",
    }
    rules["release_schema"] = {
        "record_id_prefix": "cfpb_v051_",
        "stress_id_required": True,
        "drop_internal_columns": ["_selection_key"],
        "manifest_paths": "relative_to_manifest",
        "generation_splits": ["population", "enrichment"],
        "generation_excerpt_chars": 1400,
        "generation_columns": [
            "seed_id",
            "release_split",
            "product",
            "sub_product",
            "issue",
            "sub_issue",
            "sampling_frame",
            "enrichment_reason",
            "month",
            "register_candidate",
            "lid_status",
            "redaction_placeholder_proportion",
            "seed_narrative_excerpt",
            "source_release",
            "source_release_status",
            "benchmark_eligible",
            "publication_eligible",
            "may_enter_final_dataset",
        ],
    }

    evidence = {
        "row_overrides": (
            "seed_v051_row_overrides_v01.csv: completed audit exclusions and "
            "blocked PII rows with source hashes"
        ),
        "release_schema": (
            "Seed v05 verification: internal selection key shipped; stress had no "
            "stable ID; manifest paths were Colab-absolute"
        ),
    }
    rationale = {
        "row_overrides": (
            "Apply completed audit decisions before any sampling and fail closed on "
            "missing IDs or source-text hash mismatch."
        ),
        "release_schema": (
            "Provide portable lineage artifacts and a minimal model-facing view while "
            "keeping internal sampling fields out of the release interface."
        ),
    }

    rows: list[dict[str, object]] = []
    for area in [
        "exact_duplicates",
        "template_families",
        "fuzzy_duplicates",
        "register",
        "very_short_long",
        "redaction",
        "pii",
        "language",
        "linguistic_features",
        "sampling_quotas",
        "row_overrides",
        "release_schema",
    ]:
        if area in parent_rows:
            row = dict(parent_rows[area])
            row["machine_rule_json"] = json.dumps(
                rules[area], ensure_ascii=False, separators=(",", ":")
            )
            row["decision_status"] = "accepted"
            row["decided_by"] = "chang"
            row["decided_utc"] = decided_utc
            if area == "sampling_quotas":
                row["proposed_rule"] = (
                    "Keep requested enrichment demand separate from post-rule supply; "
                    "allow only explained exhaustion above accepted minima; reserve a "
                    "reason-stratified 200-row population stress set."
                )
                row["threshold_or_quota"] = (
                    "primary=2000; enrichment requested=651 with explained minimum=646; "
                    "stress=200 across six reason quotas"
                )
                row["audit_metric_reference"] = (
                    "Seed v05 replay: Zelle outside 50 source -> 49 eligible -> 45 "
                    "after exact cap; stress 173/21/4/2 showed reason imbalance"
                )
                row["research_rationale"] = (
                    "Preserve demand as 50 without silently claiming it was achieved; "
                    "fail on unexplained shortfall and cover rare robustness risks."
                )
            elif area == "pii":
                row["proposed_rule"] = (
                    "Require a hash-bound clearance disposition for every regex-risk "
                    "row; exclude confirmed or uncleared PII from all text-bearing "
                    "outputs; retain final full-release NER QA as a benchmark gate."
                )
                row["audit_metric_reference"] = (
                    "pii_presidio_validation_master.csv: complete census of 35 regex "
                    "risks; 33 no and 2 yes"
                )
                row["research_rationale"] = (
                    "Regex redaction misses contextual identifiers such as names and "
                    "addresses; sampling luck is not a privacy control."
                )
        else:
            row = {
                "decision_area": area,
                "required_evidence": evidence[area],
                "proposed_rule": rationale[area],
                "threshold_or_quota": "fail closed",
                "machine_rule_json": json.dumps(
                    rules[area], ensure_ascii=False, separators=(",", ":")
                ),
                "audit_metric_reference": evidence[area],
                "research_rationale": rationale[area],
                "decision_status": "accepted",
                "decided_by": "chang",
                "decided_utc": decided_utc,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def validate_decisions(decisions: pd.DataFrame) -> dict[str, dict[str, object]]:
    if len(decisions) != 12 or decisions["decision_area"].duplicated().any():
        raise ValueError("Decision v03 requires exactly 12 unique areas")
    rules: dict[str, dict[str, object]] = {}
    for row in decisions.to_dict(orient="records"):
        area = str(row["decision_area"])
        if row["decision_status"] != "accepted":
            raise ValueError(f"Decision is not accepted: {area}")
        for field in [
            "proposed_rule",
            "machine_rule_json",
            "audit_metric_reference",
            "research_rationale",
            "decided_by",
            "decided_utc",
        ]:
            if not str(row[field]).strip():
                raise ValueError(f"Missing {field}: {area}")
        value = json.loads(str(row["machine_rule_json"]))
        if not isinstance(value, dict):
            raise ValueError(f"Rule must be an object: {area}")
        rules[area] = value
    quotas = rules["sampling_quotas"]
    if sum(int(value) for value in quotas["stress_quotas"].values()) != int(
        quotas["stress_test_size"]
    ):
        raise ValueError("Stress reason quotas must sum to stress_test_size")
    if quotas["stress_priority"] != list(quotas["stress_quotas"]):
        raise ValueError("Stress priority and quota keys must have the same order")
    return rules


def main() -> None:
    for required in [
        EDA_RUN / "manifest.json",
        EDA_RUN / "analysis_ready_corpus.parquet",
        PARENT_DECISION,
        QUALITY_AUDIT,
        PII_AUDIT,
    ]:
        if not required.exists():
            raise FileNotFoundError(required)
    parent = json.loads(PARENT_DECISION.read_text(encoding="utf-8"))
    if not parent.get("release_gate", {}).get("passed"):
        raise ValueError("Parent decision v02 release gate is not passed")
    source_manifest_hash = sha256_file(EDA_RUN / "manifest.json")
    if parent.get("source_manifest_sha256") != source_manifest_hash:
        raise ValueError("Parent decision and frozen EDA manifest disagree")

    if DECISION_RECORD.exists():
        previous = json.loads(DECISION_RECORD.read_text(encoding="utf-8"))
        decided_utc = previous["created_utc"]
    else:
        decided_utc = datetime.now(timezone.utc).isoformat()

    overrides, clearance = build_governance_inputs(decided_utc)
    decisions = build_decisions(decided_utc)
    rules = validate_decisions(decisions)
    write_csv(decisions, DECISION_TABLE)

    record = {
        "record_version": "v03",
        "release_target": "CFPB Seed v05.1 candidate",
        "source_eda_version": "v05.1",
        "source_run": RUN_ID,
        "source_manifest_sha256": source_manifest_hash,
        "parent_decision_record": PARENT_DECISION.name,
        "parent_decision_record_sha256": sha256_file(PARENT_DECISION),
        "created_utc": decided_utc,
        "prepared_by": "llm:codex-5.6-sol",
        "approved_by": "chang",
        "approval_basis": (
            "User explicitly instructed execution of the v03 governance and v05.1 "
            "builder scope in the 2026-07-15 session."
        ),
        "governance_inputs": {
            ROW_OVERRIDES.name: {
                "rows": len(overrides),
                "sha256": sha256_file(ROW_OVERRIDES),
            },
            PII_CLEARANCE.name: {
                "rows": len(clearance),
                "cleared_rows": int(
                    clearance["release_action"].eq("allow_after_regex_redaction").sum()
                ),
                "blocked_rows": int(clearance["release_action"].eq("exclude").sum()),
                "sha256": sha256_file(PII_CLEARANCE),
            },
            FUZZY_ADJUDICATION.name: {
                "rows": len(read_csv(FUZZY_ADJUDICATION)),
                "sha256": sha256_file(FUZZY_ADJUDICATION),
            },
        },
        "decisions": decisions.to_dict(orient="records"),
        "rules": rules,
        "release_gate": {
            "scope": "build_seed_v051_candidate",
            "parent_v02_passed": True,
            "decision_rows_accepted": 12,
            "governance_inputs_hash_bound": True,
            "machine_rules_valid": True,
            "passed": True,
        },
        "benchmark_release_gate": {
            "passed": False,
            "blocking_requirements": [
                "full_seed_and_stress_presidio_or_equivalent_ner_scan",
                "manual_review_of_final_privacy_scan_positives",
                "final_release_validator_pass",
            ],
        },
        "limitations": [
            "PII and quality source labels are LLM audit labels approved for policy use; they are not independent human validation.",
            "The local environment lacks Presidio/spaCy, so final NER privacy QA remains a separate fail-closed benchmark gate.",
            "Canonical XXXX strings are not modified by this decision and remain under separate generation-input review.",
        ],
    }
    DECISION_RECORD.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "row_overrides": str(ROW_OVERRIDES),
                "row_override_rows": len(overrides),
                "pii_clearance": str(PII_CLEARANCE),
                "pii_clearance_rows": len(clearance),
                "decision_table": str(DECISION_TABLE),
                "decision_record": str(DECISION_RECORD),
                "decision_record_sha256": sha256_file(DECISION_RECORD),
                "build_gate_passed": True,
                "benchmark_release_gate_passed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
