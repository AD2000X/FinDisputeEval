"""Decision-gated CFPB Seed v05 sampler.

The sampler intentionally has no methodological defaults. It consumes only a
passed v05 audit decision record whose accepted rows contain machine-readable
JSON rules. It never reads or inherits the v04 seed release.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
}

RELEASE_TEXT_COLUMNS = {
    "narrative_unwrapped",
    "narrative_core",
    "narrative_canonical",
    "narrative_lexical",
}
SOURCE_TEXT_COLUMNS = {
    "Consumer complaint narrative",
    "narrative_raw",
    "narrative_unwrapped",
    "narrative_core",
    "narrative_canonical",
    "narrative_match",
    "narrative_lexical",
    "ZIP code",
}
PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "PHONE": re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"
    ),
    "SSN": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "ACCOUNT": re.compile(
        r"\b(?:account|acct|card)\s*(?:number|no\.?|#)\s*\d{6,}\b",
        re.I,
    ),
    "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


@dataclass
class SeedV05Config:
    eda_run_dir: Path
    decision_record_path: Path
    output_dir: Path
    random_seed: int = 20260713

    def __post_init__(self) -> None:
        self.eda_run_dir = Path(self.eda_run_dir).resolve()
        self.decision_record_path = Path(self.decision_record_path).resolve()
        self.output_dir = Path(self.output_dir).resolve()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_key(seed: int, complaint_id: str) -> str:
    payload = f"{seed}|{complaint_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_accepted_rules(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if not record.get("release_gate", {}).get("passed", False):
        raise ValueError("Seed v05 release gate has not passed")
    if record.get("source_eda_version") != "v05.1":
        raise ValueError("Seed v05 requires a v05.1 decision record")
    decisions = record.get("decisions", [])
    by_area = {row["decision_area"]: row for row in decisions}
    missing = sorted(REQUIRED_DECISION_AREAS - set(by_area))
    if missing:
        raise ValueError(f"Decision record is missing areas: {missing}")
    rules: dict[str, Any] = {}
    for area in sorted(REQUIRED_DECISION_AREAS):
        row = by_area[area]
        if str(row.get("decision_status", "")).lower() != "accepted":
            raise ValueError(f"Decision is not accepted: {area}")
        value = str(row.get("machine_rule_json", "")).strip()
        if not value:
            raise ValueError(f"Decision has no machine_rule_json: {area}")
        try:
            rules[area] = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid machine_rule_json for {area}") from error
    return record, rules


def _validate_rule_contract(rules: dict[str, Any]) -> None:
    required_keys = {
        "exact_duplicates": {"max_per_exact_hash"},
        "template_families": {"max_per_family"},
        "fuzzy_duplicates": {"apply_confirmed_pairs"},
        "register": {"allowed_registers", "other_action"},
        "very_short_long": {"very_short_action", "very_long_action"},
        "redaction": {"max_population_proportion", "above_threshold_action"},
        "pii": {
            "regex_risk_action",
            "released_text_column",
            "apply_regex_redaction",
            "drop_unreleased_text_columns",
        },
        "language": {"allowed_statuses", "other_action"},
        "linguistic_features": {"sampling_features"},
        "sampling_quotas": {
            "primary_seed_size",
            "enrichment_quotas",
            "stress_test_size",
            "stratify_by",
        },
    }
    for area, keys in required_keys.items():
        missing = sorted(keys - set(rules[area]))
        if missing:
            raise ValueError(f"{area} rule is missing keys: {missing}")
    actions = {"retain", "exclude", "stress_only"}
    for area, keys in {
        "very_short_long": ["very_short_action", "very_long_action"],
        "redaction": ["above_threshold_action"],
        "pii": ["regex_risk_action"],
        "language": ["other_action"],
        "register": ["other_action"],
    }.items():
        for key in keys:
            if rules[area][key] not in actions:
                raise ValueError(f"Unsupported action {rules[area][key]!r} for {area}.{key}")
    pii_rule = rules["pii"]
    if pii_rule["released_text_column"] not in RELEASE_TEXT_COLUMNS:
        raise ValueError("pii.released_text_column is not an approved text view")
    if not isinstance(pii_rule["apply_regex_redaction"], bool):
        raise ValueError("pii.apply_regex_redaction must be boolean")
    if pii_rule["drop_unreleased_text_columns"] is not True:
        raise ValueError("Seed v05 must drop unreleased source-text columns")
    if (
        pii_rule["regex_risk_action"] != "exclude"
        and not pii_rule["apply_regex_redaction"]
    ):
        raise ValueError(
            "Retained or stress-only regex risks require regex redaction"
        )
    redaction_threshold = float(rules["redaction"]["max_population_proportion"])
    if not 0.0 <= redaction_threshold <= 1.0:
        raise ValueError("redaction.max_population_proportion must be bounded [0, 1]")
    if not rules["register"]["allowed_registers"]:
        raise ValueError("register.allowed_registers must not be empty")
    if not rules["language"]["allowed_statuses"]:
        raise ValueError("language.allowed_statuses must not be empty")
    if not isinstance(rules["linguistic_features"]["sampling_features"], list):
        raise ValueError("linguistic_features.sampling_features must be a list")
    if not isinstance(rules["fuzzy_duplicates"]["apply_confirmed_pairs"], bool):
        raise ValueError("fuzzy_duplicates.apply_confirmed_pairs must be boolean")
    for area, key in {
        "exact_duplicates": "max_per_exact_hash",
        "template_families": "max_per_family",
    }.items():
        if int(rules[area][key]) < 1:
            raise ValueError(f"{area}.{key} must be a positive integer")
    quotas = rules["sampling_quotas"]
    for key in ("primary_seed_size", "stress_test_size"):
        if int(quotas[key]) < 0:
            raise ValueError(f"sampling_quotas.{key} must be non-negative")
    if not isinstance(quotas["enrichment_quotas"], dict):
        raise ValueError("sampling_quotas.enrichment_quotas must be an object")
    if any(int(value) < 0 for value in quotas["enrichment_quotas"].values()):
        raise ValueError("Enrichment quotas must be non-negative")
    if not isinstance(quotas["stratify_by"], list) or not quotas["stratify_by"]:
        raise ValueError("sampling_quotas.stratify_by must be a non-empty list")


def _set_action(
    frame: pd.DataFrame, mask: pd.Series, action: str, reason: str
) -> None:
    affected = mask & frame["seed_action"].eq("retain")
    frame.loc[affected, "seed_action"] = action
    frame.loc[affected, "seed_action_reason"] = reason


def apply_eligibility_rules(
    frame: pd.DataFrame, rules: dict[str, Any]
) -> pd.DataFrame:
    result = frame.copy()
    result["seed_action"] = "retain"
    result["seed_action_reason"] = ""
    length_rule = rules["very_short_long"]
    _set_action(
        result,
        result["quality_candidate_very_short"].fillna(False),
        length_rule["very_short_action"],
        "very_short",
    )
    _set_action(
        result,
        result["quality_candidate_very_long"].fillna(False),
        length_rule["very_long_action"],
        "very_long",
    )
    redaction_rule = rules["redaction"]
    _set_action(
        result,
        result["redaction_placeholder_proportion"].gt(
            float(redaction_rule["max_population_proportion"])
        ),
        redaction_rule["above_threshold_action"],
        "redaction_above_threshold",
    )
    _set_action(
        result,
        result["pii_regex_risk"].fillna(False),
        rules["pii"]["regex_risk_action"],
        "pii_regex_risk",
    )
    allowed_languages = set(rules["language"]["allowed_statuses"])
    _set_action(
        result,
        ~result["lid_status"].isin(allowed_languages),
        rules["language"]["other_action"],
        "language_status_not_allowed",
    )
    allowed_registers = set(rules["register"]["allowed_registers"])
    _set_action(
        result,
        ~result["register_candidate"].isin(allowed_registers),
        rules["register"].get("other_action", "stress_only"),
        "register_not_allowed",
    )
    return result


def cap_duplicate_groups(
    frame: pd.DataFrame, rules: dict[str, Any], random_seed: int
) -> pd.DataFrame:
    result = frame.assign(
        _selection_key=frame["Complaint ID"].map(
            lambda value: _selection_key(random_seed, str(value))
        )
    ).sort_values("_selection_key")
    exact_cap = int(rules["exact_duplicates"]["max_per_exact_hash"])
    family_cap = int(rules["template_families"]["max_per_family"])
    if exact_cap < 1 or family_cap < 1:
        raise ValueError("Duplicate caps must be positive integers")
    result = result.loc[
        result.groupby("canonical_text_sha256").cumcount().lt(exact_cap)
    ]
    result = result.loc[
        result.groupby("family_signature").cumcount().lt(family_cap)
    ]
    return result


def cap_confirmed_fuzzy_clusters(
    frame: pd.DataFrame,
    rule: dict[str, Any],
    decision_record_path: Path,
) -> pd.DataFrame:
    if not bool(rule["apply_confirmed_pairs"]):
        return frame
    pair_value = str(rule.get("adjudicated_pairs_path", "")).strip()
    if not pair_value:
        raise ValueError(
            "fuzzy_duplicates requires adjudicated_pairs_path when confirmed pairs are applied"
        )
    pair_path = Path(pair_value)
    if not pair_path.is_absolute():
        pair_path = decision_record_path.parent / pair_path
    if not pair_path.exists():
        raise FileNotFoundError(pair_path)
    pairs = (
        pd.read_parquet(pair_path)
        if pair_path.suffix.lower() == ".parquet"
        else pd.read_csv(pair_path, dtype="string", keep_default_na=False)
    )
    required = {"complaint_id_a", "complaint_id_b", "manual_same_template"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"Adjudicated fuzzy-pair file is missing columns: {missing}")
    confirmed = pairs.loc[
        pairs["manual_same_template"].astype(str).str.lower().isin(
            ["true", "yes", "1"]
        )
    ]
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left, right in confirmed[
        ["complaint_id_a", "complaint_id_b"]
    ].itertuples(index=False, name=None):
        union(str(left), str(right))
    result = frame.copy()
    result["confirmed_fuzzy_cluster"] = result["Complaint ID"].map(
        lambda value: find(str(value)) if str(value) in parent else f"singleton:{value}"
    )
    cap = int(rule.get("max_per_confirmed_cluster", 1))
    if cap < 1:
        raise ValueError("max_per_confirmed_cluster must be positive")
    return result.loc[
        result.groupby("confirmed_fuzzy_cluster").cumcount().lt(cap)
    ]


def _redact_regex_pii(text: object) -> str:
    result = str(text)
    for entity, pattern in PII_PATTERNS.items():
        result = pattern.sub(f"[REDACTED_{entity}]", result)
    return result


def prepare_release_frame(
    frame: pd.DataFrame,
    pii_rule: dict[str, Any],
    include_text: bool = True,
) -> pd.DataFrame:
    result = frame.copy()
    source_column = pii_rule["released_text_column"]
    if include_text:
        if source_column not in result.columns:
            raise ValueError(f"Release text column is missing: {source_column}")
        result["seed_text"] = result[source_column].fillna("").astype(str)
        if pii_rule["apply_regex_redaction"]:
            result["seed_text"] = result["seed_text"].map(_redact_regex_pii)
    return result.drop(
        columns=[column for column in SOURCE_TEXT_COLUMNS if column in result],
        errors="ignore",
    )


def proportional_stratified_sample(
    frame: pd.DataFrame,
    n: int,
    strata: list[str],
    random_seed: int,
) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.head(0).copy()
    if n >= len(frame):
        return frame.copy()
    missing = sorted(set(strata) - set(frame.columns))
    if missing:
        raise ValueError(f"Sampling strata are missing from corpus: {missing}")
    working = frame.copy()
    working["_stratum"] = working[strata].astype("string").fillna("").agg("|".join, axis=1)
    counts = working["_stratum"].value_counts().rename("available").to_frame()
    counts["ideal"] = counts["available"] / len(working) * n
    counts["allocated"] = np.floor(counts["ideal"]).astype(int)
    counts["allocated"] = counts[["allocated", "available"]].min(axis=1)
    remainder = n - int(counts["allocated"].sum())
    order = (counts["ideal"] - counts["allocated"]).sort_values(ascending=False).index
    for stratum in order:
        if remainder <= 0:
            break
        if counts.at[stratum, "allocated"] < counts.at[stratum, "available"]:
            counts.at[stratum, "allocated"] += 1
            remainder -= 1
    parts = []
    for offset, (stratum, row) in enumerate(counts.iterrows()):
        take = int(row["allocated"])
        if take:
            group = working.loc[working["_stratum"].eq(stratum)]
            parts.append(group.sample(n=take, random_state=random_seed + offset))
    return pd.concat(parts, ignore_index=True).drop(columns="_stratum")


def build_seed_v05(config: SeedV05Config) -> dict[str, Path]:
    decision_record, rules = load_accepted_rules(config.decision_record_path)
    _validate_rule_contract(rules)
    eda_manifest_path = config.eda_run_dir / "manifest.json"
    eda_manifest = json.loads(eda_manifest_path.read_text(encoding="utf-8"))
    if eda_manifest.get("version") != "v05.1":
        raise ValueError("Seed v05 requires a v05.1 EDA manifest")
    expected_decision_parent = decision_record.get("source_manifest_sha256")
    if expected_decision_parent != sha256_file(eda_manifest_path):
        raise ValueError("Decision record does not match the selected EDA manifest")
    corpus = pd.read_parquet(config.eda_run_dir / "analysis_ready_corpus.parquet")
    if len(corpus) != 205_589:
        raise ValueError(f"Expected 205,589 corpus rows, got {len(corpus):,}")
    corpus["month"] = pd.to_datetime(
        corpus["date_received_parsed"], errors="coerce", utc=True
    ).dt.strftime("%Y-%m")
    corpus = apply_eligibility_rules(corpus, rules)
    retained = corpus.loc[corpus["seed_action"].eq("retain")].copy()
    retained = cap_duplicate_groups(retained, rules, config.random_seed)
    retained = cap_confirmed_fuzzy_clusters(
        retained,
        rules["fuzzy_duplicates"],
        config.decision_record_path,
    )
    quotas = rules["sampling_quotas"]
    strata = list(quotas["stratify_by"])
    population = retained.loc[retained["population_eligible"]]
    primary = proportional_stratified_sample(
        population,
        int(quotas["primary_seed_size"]),
        strata,
        config.random_seed,
    )
    enrichment_parts = []
    for offset, (sampling_frame, requested) in enumerate(
        sorted(dict(quotas["enrichment_quotas"]).items())
    ):
        candidates = retained.loc[
            ~retained["population_eligible"]
            & retained["sampling_frame"].eq(sampling_frame)
        ]
        enrichment_parts.append(
            proportional_stratified_sample(
                candidates,
                int(requested),
                [column for column in strata if column in candidates.columns],
                config.random_seed + 10_000 + offset,
            )
        )
    enrichment = (
        pd.concat(enrichment_parts, ignore_index=True)
        if enrichment_parts
        else retained.head(0).copy()
    )
    seed = pd.concat([primary, enrichment], ignore_index=True)
    seed["seed_id"] = seed["Complaint ID"].map(
        lambda value: "cfpb_v05_" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    )
    stress_candidates = corpus.loc[corpus["seed_action"].eq("stress_only")]
    stress = proportional_stratified_sample(
        stress_candidates,
        int(quotas["stress_test_size"]),
        [column for column in strata if column in stress_candidates.columns],
        config.random_seed + 20_000,
    )
    seed = prepare_release_frame(seed, rules["pii"], include_text=True)
    stress = prepare_release_frame(stress, rules["pii"], include_text=True)
    primary_release = seed.loc[seed["population_eligible"]].copy()
    enrichment_release = seed.loc[~seed["population_eligible"]].copy()
    excluded = prepare_release_frame(
        corpus.loc[corpus["seed_action"].eq("exclude")],
        rules["pii"],
        include_text=False,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "seed_parquet": config.output_dir / "cfpb_seed_v05.parquet",
        "seed_jsonl": config.output_dir / "cfpb_seed_v05.jsonl",
        "population_parquet": (
            config.output_dir / "cfpb_seed_v05_population.parquet"
        ),
        "enrichment_parquet": (
            config.output_dir / "cfpb_seed_v05_enrichment.parquet"
        ),
        "stress_parquet": config.output_dir / "cfpb_seed_v05_stress_test.parquet",
        "excluded_parquet": config.output_dir / "cfpb_seed_v05_excluded.parquet",
    }
    seed.to_parquet(paths["seed_parquet"], index=False)
    seed.to_json(paths["seed_jsonl"], orient="records", lines=True, force_ascii=False)
    primary_release.to_parquet(paths["population_parquet"], index=False)
    enrichment_release.to_parquet(paths["enrichment_parquet"], index=False)
    stress.to_parquet(paths["stress_parquet"], index=False)
    excluded.to_parquet(paths["excluded_parquet"], index=False)
    manifest = {
        "release": "CFPB Seed v05",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.random_seed,
        "parent_eda_manifest_sha256": sha256_file(eda_manifest_path),
        "parent_decision_record_sha256": sha256_file(config.decision_record_path),
        "source_rows": len(corpus),
        "primary_rows": len(primary),
        "enrichment_rows": len(enrichment),
        "stress_rows": len(stress),
        "rules": rules,
        "outputs": {},
    }
    output_rows = {
        "seed_parquet": len(seed),
        "seed_jsonl": len(seed),
        "population_parquet": len(primary_release),
        "enrichment_parquet": len(enrichment_release),
        "stress_parquet": len(stress),
        "excluded_parquet": len(excluded),
    }
    manifest["outputs"] = {
        name: {
            "path": str(path),
            "rows": output_rows[name],
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    manifest_path = config.output_dir / "seed_v05_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["manifest"] = manifest_path
    return paths
