"""Fail-closed CFPB Seed v05.1 candidate builder.

The v05.1 builder preserves the frozen EDA v05.1 parent and never overwrites
Seed v05. It consumes an accepted decision record v03, hash-bound audit row
overrides, and a complete clearance census for every regex-risk row.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .cfpb_seed_v05 import (
    PII_PATTERNS,
    SOURCE_TEXT_COLUMNS,
    _redact_regex_pii,
    apply_eligibility_rules,
    cap_confirmed_fuzzy_clusters,
    proportional_stratified_sample,
    sha256_file,
)


REQUIRED_DECISION_AREAS = {
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
}


@dataclass
class SeedV051Config:
    eda_run_dir: Path
    decision_record_path: Path
    output_dir: Path
    random_seed: int = 20260713

    def __post_init__(self) -> None:
        self.eda_run_dir = Path(self.eda_run_dir).resolve()
        self.decision_record_path = Path(self.decision_record_path).resolve()
        self.output_dir = Path(self.output_dir).resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def _resolve_governance_input(
    decision_path: Path,
    record: dict[str, Any],
    filename: str,
) -> Path:
    metadata = record.get("governance_inputs", {}).get(filename)
    if not metadata:
        raise ValueError(f"Decision record does not hash-bind governance input: {filename}")
    path = decision_path.parent / filename
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != metadata.get("sha256"):
        raise ValueError(
            f"Governance input hash mismatch: {filename}; "
            f"expected={metadata.get('sha256')}, actual={actual}"
        )
    return path


def load_accepted_v03(
    decision_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    record = json.loads(Path(decision_path).read_text(encoding="utf-8"))
    if record.get("record_version") != "v03":
        raise ValueError("Seed v05.1 requires decision record v03")
    gate = record.get("release_gate", {})
    if gate.get("scope") != "build_seed_v051_candidate" or not gate.get("passed"):
        raise ValueError("Seed v05.1 build gate is not passed")
    if record.get("source_eda_version") != "v05.1":
        raise ValueError("Seed v05.1 requires EDA v05.1 governance")
    decisions = record.get("decisions", [])
    by_area = {str(row.get("decision_area")): row for row in decisions}
    if set(by_area) != REQUIRED_DECISION_AREAS or len(decisions) != len(by_area):
        raise ValueError("Decision v03 areas are missing or duplicated")
    parsed: dict[str, dict[str, Any]] = {}
    for area in sorted(REQUIRED_DECISION_AREAS):
        row = by_area[area]
        if str(row.get("decision_status", "")).lower() != "accepted":
            raise ValueError(f"Decision is not accepted: {area}")
        parsed[area] = json.loads(str(row.get("machine_rule_json", "")))
        if parsed[area] != record.get("rules", {}).get(area):
            raise ValueError(f"Decision row and consolidated rules disagree: {area}")
    validate_v051_rules(parsed)
    return record, parsed


def validate_v051_rules(rules: dict[str, dict[str, Any]]) -> None:
    quotas = rules["sampling_quotas"]
    stress_quotas = quotas.get("stress_quotas")
    priority = quotas.get("stress_priority")
    if not isinstance(stress_quotas, dict) or not stress_quotas:
        raise ValueError("Reason-stratified stress quotas are required")
    if list(stress_quotas) != list(priority or []):
        raise ValueError("Stress priority must exactly match ordered stress quota keys")
    if sum(int(value) for value in stress_quotas.values()) != int(
        quotas["stress_test_size"]
    ):
        raise ValueError("Stress quotas must sum to stress_test_size")
    shortfall = quotas.get("enrichment_shortfall_policy", {})
    if shortfall.get("mode") != "allow_if_exhausted_after_eligibility_and_caps":
        raise ValueError("Unsupported enrichment shortfall policy")
    if shortfall.get("fail_on_unexplained_shortfall") is not True:
        raise ValueError("Unexplained enrichment shortfalls must fail closed")
    pii = rules["pii"]
    if pii.get("require_clearance_for_regex_risk") is not True:
        raise ValueError("Every regex-risk row must require clearance")
    if pii.get("confirmed_pii_action") != "exclude":
        raise ValueError("Confirmed PII must be excluded")
    if pii.get("uncleared_action") != "exclude":
        raise ValueError("Uncleared PII must be excluded")
    overrides = rules["row_overrides"]
    if overrides.get("required") is not True:
        raise ValueError("Row overrides must be required")
    schema = rules["release_schema"]
    if schema.get("stress_id_required") is not True:
        raise ValueError("Stress stable IDs are required")
    if schema.get("manifest_paths") != "relative_to_manifest":
        raise ValueError("Manifest paths must be portable")


def _stable_record_id(prefix: str, complaint_id: object) -> str:
    digest = hashlib.sha256(str(complaint_id).encode("utf-8")).hexdigest()[:16]
    return prefix + digest


def _selection_key(seed: int, namespace: str, complaint_id: object) -> str:
    payload = f"{seed}|{namespace}|{complaint_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def apply_pii_clearance(
    corpus: pd.DataFrame,
    clearance: pd.DataFrame,
) -> pd.DataFrame:
    result = corpus.copy()
    result["Complaint ID"] = result["Complaint ID"].astype("string")
    clearance = clearance.copy()
    clearance["Complaint ID"] = clearance["Complaint ID"].astype("string")
    if clearance["Complaint ID"].duplicated().any():
        raise ValueError("Duplicate Complaint IDs in PII clearance")
    risk_ids = set(
        result.loc[result["pii_regex_risk"].fillna(False), "Complaint ID"].astype(str)
    )
    clearance_ids = set(clearance["Complaint ID"].astype(str))
    if risk_ids != clearance_ids:
        raise ValueError(
            "PII clearance must exactly census corpus regex risks; "
            f"missing={sorted(risk_ids-clearance_ids)}, "
            f"extra={sorted(clearance_ids-risk_ids)}"
        )
    corpus_hashes = result.set_index("Complaint ID")["canonical_text_sha256"].astype(str)
    for row in clearance.to_dict(orient="records"):
        complaint_id = str(row["Complaint ID"])
        if corpus_hashes.at[complaint_id] != str(row["canonical_text_sha256"]):
            raise ValueError(f"PII clearance source hash mismatch: {complaint_id}")
    result = result.merge(
        clearance[
            [
                "Complaint ID",
                "clearance_status",
                "release_action",
                "audit_manual_pii_present",
                "audit_manual_pii_types",
            ]
        ].rename(
            columns={
                "clearance_status": "pii_clearance_status",
                "release_action": "pii_clearance_action",
            }
        ),
        on="Complaint ID",
        how="left",
        validate="one_to_one",
    )
    result["pii_clearance_status"] = result["pii_clearance_status"].fillna(
        "not_regex_risk"
    )
    result["pii_clearance_action"] = result["pii_clearance_action"].fillna(
        "not_required"
    )
    return result


def apply_row_overrides(
    corpus: pd.DataFrame,
    overrides: pd.DataFrame,
) -> pd.DataFrame:
    result = corpus.copy()
    overrides = overrides.copy()
    overrides["Complaint ID"] = overrides["Complaint ID"].astype("string")
    missing_ids = sorted(set(overrides["Complaint ID"]) - set(result["Complaint ID"]))
    if missing_ids:
        raise ValueError(f"Override Complaint IDs are absent from corpus: {missing_ids}")
    corpus_hashes = result.set_index("Complaint ID")["canonical_text_sha256"].astype(str)
    for row in overrides.to_dict(orient="records"):
        complaint_id = str(row["Complaint ID"])
        if corpus_hashes.at[complaint_id] != str(row["canonical_text_sha256"]):
            raise ValueError(f"Row override source hash mismatch: {complaint_id}")
    grouped = overrides.groupby("Complaint ID", sort=True)
    conflicts = [
        complaint_id
        for complaint_id, group in grouped
        if group["action"].nunique() != 1
    ]
    if conflicts:
        raise ValueError(f"Conflicting row overrides: {conflicts}")
    result["override_applied"] = False
    result["override_reason"] = ""
    for complaint_id, group in grouped:
        action = str(group["action"].iloc[0])
        mask = result["Complaint ID"].eq(str(complaint_id))
        result.loc[mask, "seed_action"] = action
        reasons = "|".join(sorted(set(group["reason_code"].astype(str))))
        result.loc[mask, "seed_action_reason"] = f"row_override:{reasons}"
        result.loc[mask, "override_applied"] = True
        result.loc[mask, "override_reason"] = reasons
    return result


def _stress_masks(
    corpus: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
) -> dict[str, pd.Series]:
    allowed_languages = set(rules["language"]["allowed_statuses"])
    redaction_threshold = float(rules["redaction"]["max_population_proportion"])
    return {
        "pii_regex_risk": (
            corpus["pii_regex_risk"].fillna(False)
            & corpus["pii_clearance_action"].eq("allow_after_regex_redaction")
        ),
        "redaction_above_threshold": corpus[
            "redaction_placeholder_proportion"
        ].gt(redaction_threshold),
        "language_status_not_allowed": ~corpus["lid_status"].isin(allowed_languages),
        "very_long": corpus["quality_candidate_very_long"].fillna(False),
        "very_short": corpus["quality_candidate_very_short"].fillna(False),
        "unusual_format": (
            corpus["quality_candidate_mostly_uppercase"].fillna(False)
            | corpus["byte_literal_parse_failed"].fillna(False)
            | corpus["byte_literal_invalid_escape"].fillna(False)
            | corpus["repeated_punctuation"].fillna(False)
        ),
    }


def select_reason_stratified_stress(
    corpus: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
    random_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    quotas = rules["sampling_quotas"]
    masks = _stress_masks(corpus, rules)
    selected_parts: list[pd.DataFrame] = []
    selected_ids: set[str] = set()
    selected_hashes: set[str] = set()
    selected_families: set[str] = set()
    accounting: list[dict[str, Any]] = []
    base = corpus.loc[~corpus["seed_action"].eq("exclude")].copy()
    if quotas.get("stress_population_only"):
        base = base.loc[base["population_eligible"]].copy()
    strata = [
        column
        for column in quotas["stratify_by"]
        if column in base.columns
    ]

    for offset, reason in enumerate(quotas["stress_priority"]):
        requested = int(quotas["stress_quotas"][reason])
        mask = masks[reason].reindex(base.index).fillna(False)
        candidates = base.loc[mask].copy()
        available_before_dedup = len(candidates)
        candidates = candidates.loc[
            ~candidates["Complaint ID"].astype(str).isin(selected_ids)
            & ~candidates["canonical_text_sha256"].astype(str).isin(selected_hashes)
            & ~candidates["family_signature"].astype(str).isin(selected_families)
        ].copy()
        candidates["_stress_key"] = candidates["Complaint ID"].map(
            lambda value: _selection_key(random_seed, reason, value)
        )
        candidates = (
            candidates.sort_values("_stress_key")
            .drop_duplicates("canonical_text_sha256")
            .drop_duplicates("family_signature")
            .drop(columns="_stress_key")
        )
        available_after_dedup = len(candidates)
        if available_after_dedup < requested:
            raise ValueError(
                f"Stress quota unavailable for {reason}: "
                f"requested={requested}, available={available_after_dedup}"
            )
        selected = proportional_stratified_sample(
            candidates,
            requested,
            strata,
            random_seed + 20_000 + offset,
        )
        if len(selected) != requested:
            raise ValueError(f"Stress quota underfilled: {reason}")
        selected["stress_reason"] = reason
        selected["seed_action"] = "stress_only"
        selected["seed_action_reason"] = f"stress_quota:{reason}"
        selected_parts.append(selected)
        selected_ids.update(selected["Complaint ID"].astype(str))
        selected_hashes.update(selected["canonical_text_sha256"].astype(str))
        selected_families.update(selected["family_signature"].astype(str))
        accounting.append(
            {
                "reason": reason,
                "requested": requested,
                "available_before_cross_reason_dedup": available_before_dedup,
                "available_after_cross_reason_dedup": available_after_dedup,
                "selected": len(selected),
                "shortfall": requested - len(selected),
            }
        )
    stress = pd.concat(selected_parts, ignore_index=True)
    if len(stress) != int(quotas["stress_test_size"]):
        raise ValueError("Stress set does not match stress_test_size")
    if not stress["Complaint ID"].astype(str).is_unique:
        raise ValueError("Stress Complaint IDs are not unique")
    return stress, accounting


def _frame_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame["sampling_frame"].value_counts().items()
    }


def cap_seed_candidates(
    retained: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
    random_seed: int,
    decision_record_path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    working = retained.copy()
    working["_selection_key"] = working["Complaint ID"].map(
        lambda value: _selection_key(random_seed, "seed_cap", value)
    )
    working = working.sort_values("_selection_key")
    accounting = {"before_caps": _frame_counts(working)}
    exact_cap = int(rules["exact_duplicates"]["max_per_exact_hash"])
    working = working.loc[
        working.groupby("canonical_text_sha256").cumcount().lt(exact_cap)
    ].copy()
    accounting["after_exact_cap"] = _frame_counts(working)
    family_cap = int(rules["template_families"]["max_per_family"])
    working = working.loc[
        working.groupby("family_signature").cumcount().lt(family_cap)
    ].copy()
    accounting["after_family_cap"] = _frame_counts(working)
    working = cap_confirmed_fuzzy_clusters(
        working,
        rules["fuzzy_duplicates"],
        decision_record_path,
    )
    accounting["after_fuzzy_cap"] = _frame_counts(working)
    return working, accounting


def _count_at_stage(
    accounting: dict[str, dict[str, int]], stage: str, frame: str
) -> int:
    return int(accounting.get(stage, {}).get(frame, 0))


def select_enrichment(
    candidates: pd.DataFrame,
    corpus: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
    cap_accounting: dict[str, dict[str, int]],
    random_seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    quotas = rules["sampling_quotas"]
    policy = quotas["enrichment_shortfall_policy"]
    minima = policy["minimum_selected_by_frame"]
    strata = list(quotas["stratify_by"])
    parts: list[pd.DataFrame] = []
    accounting: list[dict[str, Any]] = []
    for offset, (frame, value) in enumerate(sorted(quotas["enrichment_quotas"].items())):
        requested = int(value)
        available = candidates.loc[
            ~candidates["population_eligible"]
            & candidates["sampling_frame"].eq(frame)
        ].copy()
        selected = proportional_stratified_sample(
            available,
            requested,
            [column for column in strata if column in available.columns],
            random_seed + 10_000 + offset,
        )
        actual = len(selected)
        minimum = int(minima.get(frame, requested))
        explained_exhaustion = actual == min(requested, len(available))
        accepted_shortfall = (
            actual >= minimum
            and explained_exhaustion
            and policy["mode"] == "allow_if_exhausted_after_eligibility_and_caps"
        )
        if actual < requested and not accepted_shortfall:
            raise ValueError(
                f"Unaccepted enrichment shortfall: {frame}; "
                f"requested={requested}, available={len(available)}, selected={actual}, "
                f"minimum={minimum}"
            )
        source_frame = corpus.loc[corpus["sampling_frame"].eq(frame)]
        eligibility_reasons = (
            source_frame.loc[~source_frame["seed_action"].eq("retain"), "seed_action_reason"]
            .value_counts()
            .to_dict()
        )
        accounting.append(
            {
                "sampling_frame": frame,
                "requested": requested,
                "source_available": len(source_frame),
                "eligible_before_caps": _count_at_stage(
                    cap_accounting, "before_caps", frame
                ),
                "after_exact_cap": _count_at_stage(
                    cap_accounting, "after_exact_cap", frame
                ),
                "after_family_cap": _count_at_stage(
                    cap_accounting, "after_family_cap", frame
                ),
                "after_fuzzy_cap": _count_at_stage(
                    cap_accounting, "after_fuzzy_cap", frame
                ),
                "selected": actual,
                "shortfall": requested - actual,
                "minimum_acceptable": minimum,
                "shortfall_accepted": bool(actual == requested or accepted_shortfall),
                "eligibility_loss_reasons": {
                    str(key): int(count) for key, count in eligibility_reasons.items()
                },
                "exact_cap_losses": max(
                    0,
                    _count_at_stage(cap_accounting, "before_caps", frame)
                    - _count_at_stage(cap_accounting, "after_exact_cap", frame),
                ),
                "family_cap_losses": max(
                    0,
                    _count_at_stage(cap_accounting, "after_exact_cap", frame)
                    - _count_at_stage(cap_accounting, "after_family_cap", frame),
                ),
                "fuzzy_cap_losses": max(
                    0,
                    _count_at_stage(cap_accounting, "after_family_cap", frame)
                    - _count_at_stage(cap_accounting, "after_fuzzy_cap", frame),
                ),
            }
        )
        parts.append(selected)
    return pd.concat(parts, ignore_index=True), accounting


def _prepare_text_release(
    frame: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
    include_text: bool,
) -> pd.DataFrame:
    result = frame.copy()
    pii_rule = rules["pii"]
    if include_text:
        source_column = pii_rule["released_text_column"]
        result["seed_text"] = result[source_column].fillna("").astype(str)
        if pii_rule["apply_regex_redaction"]:
            result["seed_text"] = result["seed_text"].map(_redact_regex_pii)
    result = result.drop(
        columns=[column for column in SOURCE_TEXT_COLUMNS if column in result],
        errors="ignore",
    )
    result = result.drop(columns=rules["release_schema"]["drop_internal_columns"], errors="ignore")
    return result


def _validate_text_release(frame: pd.DataFrame, name: str) -> None:
    if "seed_text" not in frame:
        raise ValueError(f"Text-bearing release lacks seed_text: {name}")
    if frame["seed_text"].fillna("").str.strip().eq("").any():
        raise ValueError(f"Empty seed_text in {name}")
    leaked = sorted(SOURCE_TEXT_COLUMNS & set(frame.columns))
    if leaked:
        raise ValueError(f"Unreleased source columns in {name}: {leaked}")
    for entity, pattern in PII_PATTERNS.items():
        if frame["seed_text"].str.contains(pattern, na=False).any():
            raise ValueError(f"Residual regex PII in {name}: {entity}")
    uncleared = frame.loc[
        frame["pii_regex_risk"].fillna(False)
        & ~frame["pii_clearance_action"].eq("allow_after_regex_redaction")
    ]
    if not uncleared.empty:
        raise ValueError(f"Uncleared regex-risk rows in {name}")


def _generation_view(
    seed: pd.DataFrame,
    rules: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    schema = rules["release_schema"]
    view = pd.DataFrame(
        {
            "seed_id": seed["seed_id"],
            "release_split": seed["release_split"],
            "product": seed["Product"],
            "sub_product": seed["Sub-product"],
            "issue": seed["Issue"],
            "sub_issue": seed["Sub-issue"],
            "sampling_frame": seed["sampling_frame"],
            "enrichment_reason": seed["enrichment_reason"],
            "month": seed["month"],
            "register_candidate": seed["register_candidate"],
            "lid_status": seed["lid_status"],
            "redaction_placeholder_proportion": seed[
                "redaction_placeholder_proportion"
            ],
            "seed_narrative_excerpt": seed["seed_text"].str.slice(
                0, int(schema["generation_excerpt_chars"])
            ),
            "source_release": "CFPB Seed v05.1 candidate",
            "source_release_status": "privacy_qa_pending",
            "benchmark_eligible": False,
            "publication_eligible": False,
            "may_enter_final_dataset": False,
        }
    )
    expected = list(schema["generation_columns"])
    if list(view.columns) != expected:
        raise ValueError("Generation view columns disagree with decision v03")
    return view


def build_seed_v051(config: SeedV051Config) -> dict[str, Path]:
    decision_record, rules = load_accepted_v03(config.decision_record_path)
    eda_manifest_path = config.eda_run_dir / "manifest.json"
    if sha256_file(eda_manifest_path) != decision_record["source_manifest_sha256"]:
        raise ValueError("Decision v03 does not match the frozen EDA manifest")
    eda_manifest = json.loads(eda_manifest_path.read_text(encoding="utf-8"))
    if eda_manifest.get("version") != "v05.1":
        raise ValueError("Seed v05.1 requires EDA v05.1")

    override_path = _resolve_governance_input(
        config.decision_record_path,
        decision_record,
        rules["row_overrides"]["path"],
    )
    clearance_path = _resolve_governance_input(
        config.decision_record_path,
        decision_record,
        rules["pii"]["clearance_path"],
    )
    # The inherited fuzzy-pair adjudication affects which rows survive a
    # release cap, so v03 must pin it just like overrides and PII clearance.
    _resolve_governance_input(
        config.decision_record_path,
        decision_record,
        rules["fuzzy_duplicates"]["adjudicated_pairs_path"],
    )
    overrides = _read_csv(override_path)
    clearance = _read_csv(clearance_path)

    corpus = pd.read_parquet(config.eda_run_dir / "analysis_ready_corpus.parquet")
    if len(corpus) != 205_589:
        raise ValueError(f"Expected 205,589 corpus rows, got {len(corpus):,}")
    corpus["Complaint ID"] = corpus["Complaint ID"].astype("string")
    corpus["month"] = pd.to_datetime(
        corpus["date_received_parsed"], errors="coerce", utc=True
    ).dt.strftime("%Y-%m")
    corpus = apply_eligibility_rules(corpus, rules)
    corpus = apply_pii_clearance(corpus, clearance)
    corpus = apply_row_overrides(corpus, overrides)

    stress, stress_accounting = select_reason_stratified_stress(
        corpus, rules, config.random_seed
    )
    stress_ids = set(stress["Complaint ID"].astype(str))
    retained = corpus.loc[
        corpus["seed_action"].eq("retain")
        & ~corpus["Complaint ID"].astype(str).isin(stress_ids)
    ].copy()
    capped, cap_accounting = cap_seed_candidates(
        retained,
        rules,
        config.random_seed,
        config.decision_record_path,
    )

    quotas = rules["sampling_quotas"]
    strata = list(quotas["stratify_by"])
    population_candidates = capped.loc[capped["population_eligible"]].copy()
    primary = proportional_stratified_sample(
        population_candidates,
        int(quotas["primary_seed_size"]),
        strata,
        config.random_seed,
    )
    if len(primary) != int(quotas["primary_seed_size"]):
        raise ValueError("Primary seed quota was not met")
    enrichment, enrichment_accounting = select_enrichment(
        capped,
        corpus,
        rules,
        cap_accounting,
        config.random_seed,
    )

    prefix = str(rules["release_schema"]["record_id_prefix"])
    primary["record_id"] = primary["Complaint ID"].map(
        lambda value: _stable_record_id(prefix, value)
    )
    primary["seed_id"] = primary["record_id"]
    primary["release_split"] = "population"
    enrichment["record_id"] = enrichment["Complaint ID"].map(
        lambda value: _stable_record_id(prefix, value)
    )
    enrichment["seed_id"] = enrichment["record_id"]
    enrichment["release_split"] = "enrichment"
    stress["record_id"] = stress["Complaint ID"].map(
        lambda value: _stable_record_id(prefix, value)
    )
    stress["stress_id"] = stress["record_id"]
    stress["release_split"] = "stress"

    seed = pd.concat([primary, enrichment], ignore_index=True)
    excluded = corpus.loc[corpus["seed_action"].eq("exclude")].copy()
    excluded["record_id"] = excluded["Complaint ID"].map(
        lambda value: _stable_record_id(prefix, value)
    )
    excluded["release_split"] = "excluded"

    seed = _prepare_text_release(seed, rules, include_text=True)
    stress = _prepare_text_release(stress, rules, include_text=True)
    excluded = _prepare_text_release(excluded, rules, include_text=False)
    _validate_text_release(seed, "seed")
    _validate_text_release(stress, "stress")
    if not set(seed["Complaint ID"].astype(str)).isdisjoint(
        set(stress["Complaint ID"].astype(str))
    ):
        raise ValueError("Seed and stress Complaint IDs overlap")
    if not seed["seed_id"].is_unique or not stress["stress_id"].is_unique:
        raise ValueError("Stable release IDs are not unique")
    if seed.groupby("canonical_text_sha256").size().max() > int(
        rules["exact_duplicates"]["max_per_exact_hash"]
    ):
        raise ValueError("Exact duplicate cap failed")
    if seed.groupby("family_signature").size().max() > int(
        rules["template_families"]["max_per_family"]
    ):
        raise ValueError("Template family cap failed")

    population_release = seed.loc[seed["release_split"].eq("population")].copy()
    enrichment_release = seed.loc[seed["release_split"].eq("enrichment")].copy()
    generation = _generation_view(seed, rules)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "seed_parquet": config.output_dir / "cfpb_seed_v051.parquet",
        "seed_jsonl": config.output_dir / "cfpb_seed_v051.jsonl",
        "population_parquet": config.output_dir / "cfpb_seed_v051_population.parquet",
        "enrichment_parquet": config.output_dir / "cfpb_seed_v051_enrichment.parquet",
        "stress_parquet": config.output_dir / "cfpb_seed_v051_stress_test.parquet",
        "excluded_parquet": config.output_dir / "cfpb_seed_v051_excluded.parquet",
        "generation_parquet": config.output_dir
        / "cfpb_seed_v051_generation_input.parquet",
        "generation_jsonl": config.output_dir / "cfpb_seed_v051_generation_input.jsonl",
    }
    seed.to_parquet(paths["seed_parquet"], index=False)
    seed.to_json(paths["seed_jsonl"], orient="records", lines=True, force_ascii=False)
    population_release.to_parquet(paths["population_parquet"], index=False)
    enrichment_release.to_parquet(paths["enrichment_parquet"], index=False)
    stress.to_parquet(paths["stress_parquet"], index=False)
    excluded.to_parquet(paths["excluded_parquet"], index=False)
    generation.to_parquet(paths["generation_parquet"], index=False)
    generation.to_json(
        paths["generation_jsonl"], orient="records", lines=True, force_ascii=False
    )

    output_rows = {
        "seed_parquet": len(seed),
        "seed_jsonl": len(seed),
        "population_parquet": len(population_release),
        "enrichment_parquet": len(enrichment_release),
        "stress_parquet": len(stress),
        "excluded_parquet": len(excluded),
        "generation_parquet": len(generation),
        "generation_jsonl": len(generation),
    }
    manifest = {
        "release": "CFPB Seed v05.1 candidate",
        "release_status": "privacy_qa_pending",
        "benchmark_eligible": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.random_seed,
        "parent_eda_manifest_sha256": sha256_file(eda_manifest_path),
        "parent_decision_record_v03_sha256": sha256_file(
            config.decision_record_path
        ),
        "governance_inputs": decision_record["governance_inputs"],
        "source_rows": len(corpus),
        "primary_rows": len(primary),
        "enrichment_rows": len(enrichment),
        "stress_rows": len(stress),
        "excluded_rows": len(excluded),
        "row_overrides_applied": int(corpus["override_applied"].sum()),
        "pii_clearance": {
            "regex_risk_rows": int(corpus["pii_regex_risk"].fillna(False).sum()),
            "cleared_rows": int(
                corpus["pii_clearance_action"].eq(
                    "allow_after_regex_redaction"
                ).sum()
            ),
            "blocked_rows": int(corpus["pii_clearance_action"].eq("exclude").sum()),
            "final_full_release_ner_qa_required": True,
            "final_full_release_ner_qa_passed": False,
        },
        "enrichment_accounting": enrichment_accounting,
        "stress_accounting": stress_accounting,
        "rules": rules,
        "outputs": {
            name: {
                "path": path.name,
                "rows": output_rows[name],
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "benchmark_release_gate": {
            "passed": False,
            "blocking_requirements": decision_record["benchmark_release_gate"][
                "blocking_requirements"
            ],
        },
    }
    manifest_path = config.output_dir / "seed_v051_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["manifest"] = manifest_path
    return paths
