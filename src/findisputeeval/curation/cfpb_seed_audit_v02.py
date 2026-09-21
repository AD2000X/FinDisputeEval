"""Immutable, decision-gated CFPB Seed v05 audit workflow.

The EDA run is treated as a frozen parent.  Human labels live only in a
versioned annotation workspace.  Double-annotation files are reconciled into
adjudication tables and then merged into workspace master tables without ever
changing an EDA output.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SOURCE_AUDIT_FILES = {
    "template_family_audit": "template_family_audit.csv",
    "register_audit": "register_audit.csv",
    "quality_audit": "quality_audit.csv",
    "pii_presidio_audit": "pii_presidio_audit.csv",
    "lid_audit": "lid_audit.csv",
    "linguistic_pattern_audit": "linguistic_pattern_audit.csv",
}

MASTER_FILENAMES = {
    "template_family_audit": "template_family_audit_master.csv",
    "register_audit": "register_audit_master.csv",
    "quality_audit": "quality_audit_master.csv",
    "pii_presidio_audit": "pii_presidio_validation_master.csv",
    "lid_audit": "lid_audit_master.csv",
    "linguistic_pattern_audit": "linguistic_pattern_audit_master.csv",
    "fuzzy_duplicate_audit": "fuzzy_duplicate_audit_master.csv",
}

DECISION_AREAS = [
    ("exact_duplicates", "Deterministic hash accounting and representative-selection policy"),
    ("template_families", "Family audit agreement and false-merge assessment"),
    ("fuzzy_duplicates", "Manual pair precision by similarity band"),
    ("register", "Double-annotation agreement and accepted register taxonomy"),
    ("very_short_long", "Quality audit outcomes by length stratum"),
    ("redaction", "Quality audit using bounded redaction-placeholder proportion"),
    ("pii", "Entity-type and score-stratified PII validation"),
    ("language", "Language audit by confidence and status stratum"),
    ("linguistic_features", "Validated precision/recall for features used by sampling"),
    ("sampling_quotas", "Research objective and population/enrichment separation"),
]

REGISTER_LABELS = {
    "consumer_narrative",
    "template_letter_family",
    "template_form",
    "pasted_correspondence",
    "legal_formal",
    "other",
    "uncertain",
}
YES_NO_UNCERTAIN = {"yes", "no", "uncertain"}
QUALITY_LABELS = {"retain", "exclude", "stress_only", "uncertain"}
VALID_ACTIONS = {"retain", "exclude", "stress_only"}
RELEASE_TEXT_COLUMNS = {
    "narrative_unwrapped",
    "narrative_core",
    "narrative_canonical",
    "narrative_lexical",
}
HIGH_RISK_PRESIDIO_TYPES = {
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
    "US_BANK_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "CRYPTO",
}


@dataclass
class AuditConfig:
    eda_run_dir: Path
    audit_root: Path
    random_seed: int = 20260713
    template_master_size: int = 500
    template_large_family_census_n: int = 50
    fuzzy_master_size: int = 500
    pii_generic_sample_n: int = 300
    pii_negative_control_n: int = 200
    template_double_n: int = 150
    register_double_n: int = 150
    linguistic_double_n: int = 200

    def __post_init__(self) -> None:
        self.eda_run_dir = Path(self.eda_run_dir).resolve()
        self.audit_root = Path(self.audit_root).resolve()
        if self.template_large_family_census_n > self.template_master_size:
            raise ValueError("Template census cannot exceed template master size")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def membership_sha256(values: Iterable[object]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_audit_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def write_audit_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def stable_annotation_id(audit_name: str, values: Iterable[object]) -> str:
    payload = "|".join([audit_name, *[str(value) for value in values]])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def add_annotation_ids(
    audit_name: str,
    frame: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    result = frame.copy()
    ids = [
        stable_annotation_id(audit_name, row)
        for row in result[key_columns].itertuples(index=False, name=None)
    ]
    if "annotation_id" in result:
        existing = result["annotation_id"].astype(str).tolist()
        if existing != ids:
            raise ValueError(f"Unstable annotation IDs in {audit_name}")
    else:
        result.insert(0, "annotation_id", ids)
    if result["annotation_id"].duplicated().any():
        raise ValueError(f"Duplicate annotation IDs in {audit_name}")
    return result


def verify_eda_run(config: AuditConfig) -> dict[str, Any]:
    manifest_path = config.eda_run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "v05.1":
        raise ValueError("Audit workflow requires EDA v05.1")
    design = manifest.get("source_design", {})
    expected = {
        "population_rows": 197_489,
        "enrichment_rows": 8_100,
        "universe_unique_ids": 205_589,
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise ValueError(f"Unexpected {key}: {design.get(key)!r}")
    for filename, metadata in manifest.get("outputs", {}).items():
        path = config.eda_run_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size != metadata["size_bytes"]:
            raise ValueError(f"EDA output size mismatch: {filename}")
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"EDA output hash mismatch: {filename}")
    return manifest


def snapshot_source_audits(
    config: AuditConfig,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    snapshot_dir = config.audit_root / "source_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for audit_name, filename in SOURCE_AUDIT_FILES.items():
        source = config.eda_run_dir / filename
        destination = snapshot_dir / filename
        expected_hash = manifest["outputs"][filename]["sha256"]
        if sha256_file(source) != expected_hash:
            raise ValueError(f"Frozen EDA source changed: {filename}")
        if not destination.exists():
            shutil.copy2(source, destination)
        if sha256_file(destination) != expected_hash:
            raise ValueError(f"Workspace snapshot mismatch: {filename}")
        result[audit_name] = destination
    return result


def round_robin_stratified_sample(
    frame: pd.DataFrame,
    strata: list[str],
    n: int,
    random_seed: int,
) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.head(0).copy()
    n = min(n, len(frame))
    missing = sorted(set(strata) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing sampling strata: {missing}")
    groups = []
    for offset, (_, group) in enumerate(
        frame.groupby(strata, dropna=False, sort=True)
    ):
        groups.append(
            group.sample(frac=1.0, random_state=random_seed + offset).reset_index()
        )
    selected_indices: list[int] = []
    layer = 0
    while len(selected_indices) < n:
        added = False
        for group in groups:
            if layer < len(group):
                selected_indices.append(int(group.iloc[layer]["index"]))
                added = True
                if len(selected_indices) == n:
                    break
        if not added:
            break
        layer += 1
    return frame.loc[selected_indices].copy()


def _family_size_band(value: object) -> str:
    size = int(value)
    if size == 2:
        return "2"
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    if size <= 25:
        return "11-25"
    if size <= 99:
        return "26-99"
    return "100+"


def build_template_family_master(config: AuditConfig) -> pd.DataFrame:
    path = config.audit_root / MASTER_FILENAMES["template_family_audit"]
    if path.exists():
        return read_audit_csv(path)
    columns = [
        "Complaint ID",
        "family_signature",
        "family_group_size",
        "sampling_frame",
        "population_eligible",
        "narrative_canonical",
    ]
    corpus = pd.read_parquet(
        config.eda_run_dir / "analysis_ready_corpus.parquet",
        columns=columns,
    )
    members = corpus.loc[corpus["family_group_size"].gt(1)].copy()
    members["Complaint ID"] = members["Complaint ID"].astype(str)
    members = members.sort_values(["family_signature", "Complaint ID"])
    grouped = members.groupby("family_signature", sort=True, dropna=False)
    first = grouped.nth(0).reset_index()
    second = grouped.nth(1).reset_index()
    pairs = first[
        [
            "family_signature",
            "Complaint ID",
            "narrative_canonical",
            "family_group_size",
            "sampling_frame",
            "population_eligible",
        ]
    ].merge(
        second[["family_signature", "Complaint ID", "narrative_canonical"]],
        on="family_signature",
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    pairs = pairs.rename(
        columns={
            "Complaint ID_a": "complaint_id_a",
            "Complaint ID_b": "complaint_id_b",
            "narrative_canonical_a": "text_a",
            "narrative_canonical_b": "text_b",
            "family_group_size": "family_size",
            "sampling_frame": "representative_sampling_frame",
        }
    )
    pairs["family_size_band"] = pairs["family_size"].map(_family_size_band)
    pairs = pairs.sort_values(
        ["family_size", "family_signature"], ascending=[False, True]
    )
    census = pairs.head(config.template_large_family_census_n).copy()
    census["audit_selection"] = "large_family_census"
    remaining = pairs.loc[~pairs["family_signature"].isin(census["family_signature"])]
    stratified = round_robin_stratified_sample(
        remaining,
        [
            "population_eligible",
            "representative_sampling_frame",
            "family_size_band",
        ],
        config.template_master_size - len(census),
        config.random_seed,
    )
    stratified["audit_selection"] = "family_size_stratified"
    master = pd.concat([census, stratified], ignore_index=True)
    master = add_annotation_ids(
        "template_family_audit",
        master,
        ["family_signature", "complaint_id_a", "complaint_id_b"],
    )
    master["double_annotation_required"] = False
    master["manual_same_template"] = ""
    master["notes"] = ""
    write_audit_csv(master, path)
    return master


def _copy_master(
    config: AuditConfig,
    snapshots: dict[str, Path],
    audit_name: str,
    key_columns: list[str],
) -> pd.DataFrame:
    path = config.audit_root / MASTER_FILENAMES[audit_name]
    if path.exists():
        return read_audit_csv(path)
    master = read_audit_csv(snapshots[audit_name])
    master = add_annotation_ids(audit_name, master, key_columns)
    master["double_annotation_required"] = False
    write_audit_csv(master, path)
    return master


def build_fuzzy_master(config: AuditConfig) -> pd.DataFrame:
    path = config.audit_root / MASTER_FILENAMES["fuzzy_duplicate_audit"]
    if path.exists():
        return read_audit_csv(path)
    frame = pd.read_parquet(config.eda_run_dir / "fuzzy_duplicate_candidates.parquet")
    similarity = pd.to_numeric(frame["shingle_jaccard"], errors="coerce")
    frame["similarity_band"] = pd.cut(
        similarity,
        bins=[-float("inf"), 0.50, 0.70, 0.80, 0.90, float("inf")],
        labels=["<=0.50", "0.50-0.70", "0.70-0.80", "0.80-0.90", ">0.90"],
    ).astype("string")
    master = round_robin_stratified_sample(
        frame,
        ["similarity_band"],
        config.fuzzy_master_size,
        config.random_seed + 10_000,
    ).reset_index(drop=True)
    master = add_annotation_ids(
        "fuzzy_duplicate_audit",
        master,
        ["complaint_id_a", "complaint_id_b"],
    )
    master["manual_same_template"] = ""
    master["notes"] = ""
    write_audit_csv(master, path)
    return master


def _entity_set(value: object) -> set[str]:
    return {part.strip() for part in str(value).split("|") if part.strip()}


def _primary_pii_entity_type(row: pd.Series) -> str:
    regex_types = sorted(_entity_set(row["pii_regex_types"]))
    if regex_types:
        return f"REGEX:{regex_types[0]}"
    entities = _entity_set(row["presidio_entity_types"])
    for entity_type in sorted(HIGH_RISK_PRESIDIO_TYPES):
        if entity_type in entities:
            return entity_type
    for entity_type in ("PERSON", "LOCATION", "NRP"):
        if entity_type in entities:
            return entity_type
    return sorted(entities)[0] if entities else "NONE"


def _pii_stratum(row: pd.Series) -> str:
    if str(row["pii_regex_types"]).strip():
        return "regex_positive"
    entities = _entity_set(row["presidio_entity_types"])
    if entities & HIGH_RISK_PRESIDIO_TYPES:
        return "high_risk_presidio"
    if entities & {"PERSON", "LOCATION", "NRP"}:
        return "person_location_nrp"
    if entities:
        return "generic_entity"
    return "no_detection_control"


def build_pii_master(
    config: AuditConfig,
    snapshots: dict[str, Path],
) -> pd.DataFrame:
    path = config.audit_root / MASTER_FILENAMES["pii_presidio_audit"]
    if path.exists():
        master = read_audit_csv(path)
        current_design = (
            "primary_entity_type" in master.columns
            and master.get("sampling_design_version", pd.Series(dtype="string"))
            .astype(str)
            .eq("entity_type_score_v02")
            .all()
        )
        if current_design:
            if "narrative_canonical" not in master.columns:
                narratives = pd.read_parquet(
                    config.eda_run_dir / "analysis_ready_corpus.parquet",
                    columns=["Complaint ID", "narrative_canonical"],
                )
                narratives["Complaint ID"] = narratives["Complaint ID"].astype(str)
                master = master.merge(
                    narratives,
                    on="Complaint ID",
                    how="left",
                    validate="one_to_one",
                )
                if master["narrative_canonical"].isna().any():
                    raise ValueError(
                        "PII master contains Complaint IDs absent from the corpus"
                    )
                write_audit_csv(master, path)
            return master
        annotation_columns = [
            column
            for column in ("manual_pii_present", "manual_pii_types", "notes")
            if column in master.columns
        ]
        if annotation_columns and master[annotation_columns].apply(
            lambda column: column.astype(str).str.strip().ne("").any()
        ).any():
            raise ValueError(
                "Cannot migrate an annotated PII master to entity_type_score_v02"
            )
    source = read_audit_csv(snapshots["pii_presidio_audit"])
    narratives = pd.read_parquet(
        config.eda_run_dir / "analysis_ready_corpus.parquet",
        columns=["Complaint ID", "narrative_canonical"],
    )
    narratives["Complaint ID"] = narratives["Complaint ID"].astype(str)
    source = source.merge(
        narratives,
        on="Complaint ID",
        how="left",
        validate="one_to_one",
    )
    if source["narrative_canonical"].isna().any():
        raise ValueError("PII audit source contains Complaint IDs absent from the corpus")
    source["primary_entity_type"] = source.apply(_primary_pii_entity_type, axis=1)
    source["pii_entity_stratum"] = source.apply(_pii_stratum, axis=1)
    scores = pd.to_numeric(source["presidio_max_score"], errors="coerce").fillna(0.0)
    source["presidio_score_band"] = pd.cut(
        scores,
        bins=[-float("inf"), 0.49, 0.79, 0.89, float("inf")],
        labels=["<0.50", "0.50-0.79", "0.80-0.89", ">=0.90"],
    ).astype("string")
    census_mask = source["pii_entity_stratum"].isin(
        {"regex_positive", "high_risk_presidio"}
    )
    census = source.loc[census_mask].copy()
    census["audit_selection"] = "high_risk_census"
    generic = source.loc[
        source["pii_entity_stratum"].isin(
            {"person_location_nrp", "generic_entity"}
        )
    ]
    generic_sample = round_robin_stratified_sample(
        generic,
        ["primary_entity_type", "presidio_score_band", "sampling_frame"],
        config.pii_generic_sample_n,
        config.random_seed + 20_000,
    )
    generic_sample["audit_selection"] = "entity_score_stratified"
    negatives = source.loc[
        source["pii_entity_stratum"].eq("no_detection_control")
    ]
    negative_sample = round_robin_stratified_sample(
        negatives,
        ["sampling_frame"],
        config.pii_negative_control_n,
        config.random_seed + 30_000,
    )
    negative_sample["audit_selection"] = "negative_control"
    master = pd.concat(
        [census, generic_sample, negative_sample],
        ignore_index=True,
    ).drop_duplicates("Complaint ID")
    master = add_annotation_ids(
        "pii_presidio_audit",
        master,
        ["Complaint ID"],
    )
    master["manual_pii_present"] = ""
    master["manual_pii_types"] = ""
    master["notes"] = ""
    master["double_annotation_required"] = False
    master["sampling_design_version"] = "entity_type_score_v02"
    write_audit_csv(master, path)
    return master


def initialize_masters(
    config: AuditConfig,
    snapshots: dict[str, Path],
) -> dict[str, pd.DataFrame]:
    config.audit_root.mkdir(parents=True, exist_ok=True)
    masters = {
        "template_family_audit": build_template_family_master(config),
        "register_audit": _copy_master(
            config, snapshots, "register_audit", ["Complaint ID"]
        ),
        "quality_audit": _copy_master(
            config, snapshots, "quality_audit", ["Complaint ID"]
        ),
        "pii_presidio_audit": build_pii_master(config, snapshots),
        "lid_audit": _copy_master(
            config, snapshots, "lid_audit", ["Complaint ID"]
        ),
        "linguistic_pattern_audit": _copy_master(
            config,
            snapshots,
            "linguistic_pattern_audit",
            ["Complaint ID", "feature", "detector_output"],
        ),
        "fuzzy_duplicate_audit": build_fuzzy_master(config),
    }
    return masters


def decision_table_path(config: AuditConfig) -> Path:
    return config.audit_root / "seed_v05_decision_table.csv"


def initialize_decision_table(config: AuditConfig) -> pd.DataFrame:
    path = decision_table_path(config)
    if not path.exists():
        frame = pd.DataFrame(
            [
                {
                    "decision_area": area,
                    "required_evidence": evidence,
                    "proposed_rule": "",
                    "threshold_or_quota": "",
                    "machine_rule_json": "",
                    "audit_metric_reference": "",
                    "research_rationale": "",
                    "decision_status": "pending",
                    "decided_by": "",
                    "decided_utc": "",
                }
                for area, evidence in DECISION_AREAS
            ]
        )
        write_audit_csv(frame, path)
    return read_audit_csv(path)


def linguistic_decision_state(
    decisions: pd.DataFrame,
    known_features: set[str],
) -> tuple[str, list[str], list[str]]:
    rows = decisions.loc[decisions["decision_area"].eq("linguistic_features")]
    if len(rows) != 1:
        return "pending_decision", [], ["Missing unique linguistic_features decision"]
    row = rows.iloc[0]
    if str(row["decision_status"]).strip().lower() != "accepted":
        return "pending_decision", [], []
    value = str(row["machine_rule_json"]).strip()
    if not value:
        return "pending_decision", [], ["Accepted linguistic decision has no JSON rule"]
    try:
        rule = json.loads(value)
    except json.JSONDecodeError:
        return "pending_decision", [], ["Invalid linguistic_features JSON"]
    features = rule.get("sampling_features")
    if not isinstance(features, list):
        return "pending_decision", [], ["sampling_features must be a list"]
    features = [str(feature) for feature in features]
    unknown = sorted(set(features) - known_features)
    if unknown:
        return "pending_decision", [], [f"Unknown linguistic features: {unknown}"]
    return ("not_required" if not features else "required"), features, []


def _mark_double_rows(master_path: Path, annotation_ids: set[str]) -> None:
    master = read_audit_csv(master_path)
    required = master["annotation_id"].isin(annotation_ids)
    current = (
        master["double_annotation_required"]
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    if not current.equals(required):
        master["double_annotation_required"] = required
        write_audit_csv(master, master_path)


def create_double_assignment(
    config: AuditConfig,
    audit_name: str,
    master: pd.DataFrame,
    label_field: str,
    target: int,
    strata: list[str],
) -> dict[str, Path]:
    selected = round_robin_stratified_sample(
        master,
        strata,
        target,
        config.random_seed + int(hashlib.sha256(audit_name.encode()).hexdigest()[:6], 16),
    ).sort_values("annotation_id")
    paths: dict[str, Path] = {}
    for annotator in ("A", "B"):
        path = config.audit_root / f"{audit_name}_double_annotator_{annotator}.csv"
        paths[annotator] = path
        if not path.exists():
            assignment = selected.copy()
            assignment[label_field] = ""
            assignment["annotator_id"] = annotator
            write_audit_csv(assignment, path)
        existing = read_audit_csv(path)
        if set(existing["annotation_id"]) != set(selected["annotation_id"]):
            raise ValueError(f"Existing {audit_name} assignment membership changed")
    _mark_double_rows(
        config.audit_root / MASTER_FILENAMES[audit_name],
        set(selected["annotation_id"]),
    )
    return paths


def sync_adjudication(
    config: AuditConfig,
    audit_name: str,
    label_field: str,
) -> Path:
    paths = {
        annotator: config.audit_root
        / f"{audit_name}_double_annotator_{annotator}.csv"
        for annotator in ("A", "B")
    }
    frames = {annotator: read_audit_csv(path) for annotator, path in paths.items()}
    context_columns = [
        column
        for column in frames["A"].columns
        if column not in {label_field, "annotator_id", "notes"}
    ]
    merged = frames["A"][context_columns].merge(
        frames["A"][["annotation_id", label_field]],
        on="annotation_id",
        validate="one_to_one",
    ).merge(
        frames["B"][["annotation_id", label_field]],
        on="annotation_id",
        suffixes=("_A", "_B"),
        validate="one_to_one",
    )
    destination = config.audit_root / f"{audit_name}_adjudication.csv"
    preserve_columns = ["adjudicated_value", "adjudicator_id", "adjudication_notes"]
    if destination.exists():
        previous = read_audit_csv(destination)
        preserved = previous[["annotation_id", *preserve_columns]]
        merged = merged.merge(
            preserved,
            on="annotation_id",
            how="left",
            validate="one_to_one",
        )
    else:
        for column in preserve_columns:
            merged[column] = ""
    for column in preserve_columns:
        merged[column] = merged[column].fillna("").astype("string")
    write_audit_csv(merged, destination)
    return destination


def _normalise_label(value: object) -> str:
    return str(value).strip().lower()


def merge_adjudication_into_master(
    config: AuditConfig,
    audit_name: str,
    label_field: str,
    allowed_labels: set[str],
) -> dict[str, int]:
    master_path = config.audit_root / MASTER_FILENAMES[audit_name]
    adjudication_path = config.audit_root / f"{audit_name}_adjudication.csv"
    master = read_audit_csv(master_path)
    adjudication = read_audit_csv(adjudication_path)
    resolved: dict[str, str] = {}
    disagreement_rows = 0
    pending_adjudication = 0
    for row in adjudication.itertuples(index=False):
        annotation_id = str(row.annotation_id)
        value_a = _normalise_label(getattr(row, f"{label_field}_A"))
        value_b = _normalise_label(getattr(row, f"{label_field}_B"))
        adjudicated = _normalise_label(row.adjudicated_value)
        if value_a in allowed_labels and value_b in allowed_labels:
            if value_a == value_b:
                resolved[annotation_id] = value_a
            else:
                disagreement_rows += 1
                if adjudicated in allowed_labels:
                    resolved[annotation_id] = adjudicated
                else:
                    pending_adjudication += 1
    changed = 0
    master_index = master.set_index("annotation_id", drop=False)
    for annotation_id, value in resolved.items():
        current = _normalise_label(master_index.at[annotation_id, label_field])
        if not current:
            master_index.at[annotation_id, label_field] = value
            changed += 1
        elif current != value:
            raise ValueError(
                f"Resolved label conflicts with master: {audit_name}/{annotation_id}"
            )
    if changed:
        write_audit_csv(master_index.reset_index(drop=True), master_path)
    return {
        "resolved_rows": len(resolved),
        "changed_master_rows": changed,
        "disagreement_rows": disagreement_rows,
        "pending_adjudication": pending_adjudication,
    }


def agreement_metrics(
    config: AuditConfig,
    audit_name: str,
    label_field: str,
    allowed_labels: set[str],
) -> dict[str, Any]:
    adjudication = read_audit_csv(
        config.audit_root / f"{audit_name}_adjudication.csv"
    )
    a = adjudication[f"{label_field}_A"].map(_normalise_label)
    b = adjudication[f"{label_field}_B"].map(_normalise_label)
    valid = a.isin(allowed_labels) & b.isin(allowed_labels)
    a_valid, b_valid = a[valid], b[valid]
    raw_agreement: float | None = None
    kappa: float | None = None
    if len(a_valid):
        raw_agreement = float(a_valid.eq(b_valid).mean())
        labels = sorted(allowed_labels)
        expected = sum(
            float(a_valid.eq(label).mean()) * float(b_valid.eq(label).mean())
            for label in labels
        )
        kappa = (
            float((raw_agreement - expected) / (1.0 - expected))
            if expected < 1.0
            else None
        )
    disagreements = valid & a.ne(b)
    adjudicated = adjudication["adjudicated_value"].map(_normalise_label)
    pending_adjudication = int(
        (disagreements & ~adjudicated.isin(allowed_labels)).sum()
    )
    return {
        "audit": audit_name,
        "field": label_field,
        "assigned_rows": len(adjudication),
        "completed_rows": int(valid.sum()),
        "invalid_or_blank_rows": int((~valid).sum()),
        "disagreement_rows": int(disagreements.sum()),
        "pending_adjudication": pending_adjudication,
        "complete": bool(valid.all() and pending_adjudication == 0),
        "raw_agreement": raw_agreement,
        "cohen_kappa": kappa,
    }


def validate_label_field(
    frame: pd.DataFrame,
    field: str,
    allowed: set[str] | None = None,
    language_code: bool = False,
    notes_field: str = "notes",
) -> dict[str, int | str | float]:
    values = frame[field].map(_normalise_label)
    blank = values.eq("")
    if language_code:
        valid_nonblank = values.str.fullmatch(r"[a-z]{2,3}|mixed|unknown|other")
    else:
        if allowed is None:
            raise ValueError("Allowed labels are required")
        valid_nonblank = values.isin(allowed)
    invalid = ~blank & ~valid_nonblank
    uncertain = values.isin({"uncertain", "unknown", "other"})
    if notes_field in frame:
        missing_required_note = uncertain & frame[notes_field].astype(str).str.strip().eq("")
    else:
        missing_required_note = uncertain
    complete = valid_nonblank & ~missing_required_note
    return {
        "field": field,
        "rows": len(frame),
        "completed": int(complete.sum()),
        "pending": int(blank.sum()),
        "invalid": int(invalid.sum()),
        "missing_required_note": int(missing_required_note.sum()),
        "completion_rate": float(complete.mean()) if len(frame) else 1.0,
        "status": (
            "complete"
            if bool(complete.all())
            else "invalid_labels"
            if bool(invalid.any() or missing_required_note.any())
            else "pending"
        ),
    }


def build_completion_table(
    masters: dict[str, pd.DataFrame],
    linguistic_state: str,
    sampling_features: list[str],
) -> pd.DataFrame:
    specs = [
        ("template_family_audit", "manual_same_template", YES_NO_UNCERTAIN, False),
        ("register_audit", "manual_register", REGISTER_LABELS, False),
        ("quality_audit", "manual_quality", QUALITY_LABELS, False),
        ("pii_presidio_audit", "manual_pii_present", YES_NO_UNCERTAIN, False),
        ("lid_audit", "manual_language", None, True),
        ("lid_audit", "manual_english", YES_NO_UNCERTAIN, False),
        ("fuzzy_duplicate_audit", "manual_same_template", YES_NO_UNCERTAIN, False),
    ]
    rows = []
    for audit_name, field, allowed, language_code in specs:
        result = validate_label_field(
            masters[audit_name],
            field,
            allowed,
            language_code,
        )
        rows.append({"audit": audit_name, **result})
    pii = masters["pii_presidio_audit"]
    pii_values = pii["manual_pii_present"].map(_normalise_label)
    missing_types = pii_values.eq("yes") & pii["manual_pii_types"].astype(str).str.strip().eq("")
    if missing_types.any():
        for row in rows:
            if row["audit"] == "pii_presidio_audit":
                row["status"] = "invalid_labels"
                row["invalid"] = int(row["invalid"]) + int(missing_types.sum())
    linguistic = masters["linguistic_pattern_audit"]
    if linguistic_state == "pending_decision":
        rows.append(
            {
                "audit": "linguistic_pattern_audit",
                "field": "manual_present",
                "rows": 0,
                "completed": 0,
                "pending": 0,
                "invalid": 0,
                "missing_required_note": 0,
                "completion_rate": 0.0,
                "status": "pending_decision",
            }
        )
    elif linguistic_state == "not_required":
        rows.append(
            {
                "audit": "linguistic_pattern_audit",
                "field": "manual_present",
                "rows": 0,
                "completed": 0,
                "pending": 0,
                "invalid": 0,
                "missing_required_note": 0,
                "completion_rate": 1.0,
                "status": "not_required",
            }
        )
    else:
        subset = linguistic.loc[linguistic["feature"].isin(sampling_features)]
        result = validate_label_field(
            subset,
            "manual_present",
            YES_NO_UNCERTAIN,
        )
        rows.append({"audit": "linguistic_pattern_audit", **result})
    return pd.DataFrame(rows)


def _binary_detector_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    completed = frame.loc[
        frame["manual_present"].map(_normalise_label).isin({"yes", "no"})
    ].copy()
    if completed.empty:
        return rows
    completed["gold"] = completed["manual_present"].map(_normalise_label).eq("yes")
    completed["predicted"] = (
        completed["detector_output"].astype(str).str.lower().isin({"true", "1", "yes"})
    )
    for feature, group in completed.groupby("feature"):
        tp = int((group["gold"] & group["predicted"]).sum())
        fp = int((~group["gold"] & group["predicted"]).sum())
        fn = int((group["gold"] & ~group["predicted"]).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "feature": feature,
                "support": len(group),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def build_audit_metrics(masters: dict[str, pd.DataFrame]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    template = masters["template_family_audit"].copy()
    template["label"] = template["manual_same_template"].map(_normalise_label)
    result["template_family"] = (
        template.loc[template["label"].isin(YES_NO_UNCERTAIN)]
        .groupby(["audit_selection", "family_size_band", "label"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .to_dict(orient="records")
    )
    fuzzy = masters["fuzzy_duplicate_audit"].copy()
    fuzzy["label"] = fuzzy["manual_same_template"].map(_normalise_label)
    result["fuzzy_duplicate"] = (
        fuzzy.loc[fuzzy["label"].isin(YES_NO_UNCERTAIN)]
        .groupby(["similarity_band", "label"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .to_dict(orient="records")
    )
    register = masters["register_audit"].copy()
    register["label"] = register["manual_register"].map(_normalise_label)
    valid_register = register["label"].isin(REGISTER_LABELS - {"uncertain"})
    result["register"] = {
        "labeled_rows": int(valid_register.sum()),
        "candidate_accuracy": (
            float(
                register.loc[valid_register, "register_candidate"]
                .astype(str)
                .str.lower()
                .eq(register.loc[valid_register, "label"])
                .mean()
            )
            if valid_register.any()
            else None
        ),
    }
    pii = masters["pii_presidio_audit"].copy()
    pii["label"] = pii["manual_pii_present"].map(_normalise_label)
    result["pii"] = (
        pii.loc[pii["label"].isin(YES_NO_UNCERTAIN)]
        .groupby(["primary_entity_type", "presidio_score_band", "label"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .to_dict(orient="records")
    )
    lid = masters["lid_audit"].copy()
    lid["label"] = lid["manual_english"].map(_normalise_label)
    result["language"] = (
        lid.loc[lid["label"].isin(YES_NO_UNCERTAIN)]
        .groupby(["lid_status", "label"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .to_dict(orient="records")
    )
    result["linguistic_detector"] = _binary_detector_metrics(
        masters["linguistic_pattern_audit"]
    )
    return result


def validate_machine_rules(
    decisions: pd.DataFrame,
    known_linguistic_features: set[str],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    expected = {area for area, _ in DECISION_AREAS}
    if set(decisions["decision_area"]) != expected or decisions["decision_area"].duplicated().any():
        errors.append("Decision areas are missing or duplicated")
        return {}, errors
    rules: dict[str, Any] = {}
    for row in decisions.to_dict(orient="records"):
        area = row["decision_area"]
        if _normalise_label(row["decision_status"]) != "accepted":
            errors.append(f"Decision not accepted: {area}")
        if not str(row["proposed_rule"]).strip():
            errors.append(f"Missing proposed_rule: {area}")
        if not str(row["research_rationale"]).strip():
            errors.append(f"Missing research_rationale: {area}")
        if not str(row["decided_by"]).strip() or not str(row["decided_utc"]).strip():
            errors.append(f"Missing decision provenance: {area}")
        try:
            value = json.loads(str(row["machine_rule_json"]))
            if not isinstance(value, dict):
                raise TypeError
            rules[area] = value
        except (json.JSONDecodeError, TypeError):
            errors.append(f"Invalid machine_rule_json: {area}")
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
        if area in rules:
            missing = sorted(keys - set(rules[area]))
            if missing:
                errors.append(f"Missing {area} keys: {missing}")
    if errors:
        return rules, errors
    action_fields = [
        ("register", "other_action"),
        ("very_short_long", "very_short_action"),
        ("very_short_long", "very_long_action"),
        ("redaction", "above_threshold_action"),
        ("pii", "regex_risk_action"),
        ("language", "other_action"),
    ]
    for area, field in action_fields:
        if rules[area][field] not in VALID_ACTIONS:
            errors.append(f"Invalid action: {area}.{field}")
    pii_rule = rules["pii"]
    if pii_rule["released_text_column"] not in RELEASE_TEXT_COLUMNS:
        errors.append("Invalid PII release text column")
    if not isinstance(pii_rule["apply_regex_redaction"], bool):
        errors.append("apply_regex_redaction must be boolean")
    if pii_rule["drop_unreleased_text_columns"] is not True:
        errors.append("drop_unreleased_text_columns must be true")
    if pii_rule["regex_risk_action"] != "exclude" and not pii_rule["apply_regex_redaction"]:
        errors.append("Retained PII regex risks require redaction")
    try:
        if not 0.0 <= float(rules["redaction"]["max_population_proportion"]) <= 1.0:
            errors.append("Redaction proportion must be in [0, 1]")
        if int(rules["exact_duplicates"]["max_per_exact_hash"]) < 1:
            errors.append("Exact duplicate cap must be positive")
        if int(rules["template_families"]["max_per_family"]) < 1:
            errors.append("Template family cap must be positive")
    except (TypeError, ValueError):
        errors.append("Invalid numeric duplicate/redaction rule")
    features = rules["linguistic_features"]["sampling_features"]
    if not isinstance(features, list):
        errors.append("sampling_features must be a list")
    elif set(map(str, features)) - known_linguistic_features:
        errors.append("Unknown linguistic sampling feature")
    quotas = rules["sampling_quotas"]
    try:
        valid_quotas = (
            int(quotas["primary_seed_size"]) >= 0
            and int(quotas["stress_test_size"]) >= 0
            and isinstance(quotas["enrichment_quotas"], dict)
            and all(int(value) >= 0 for value in quotas["enrichment_quotas"].values())
            and isinstance(quotas["stratify_by"], list)
            and bool(quotas["stratify_by"])
        )
        if not valid_quotas:
            errors.append("Invalid sampling quotas")
    except (TypeError, ValueError):
        errors.append("Invalid sampling quotas")
    return rules, errors


def write_workspace_manifest(
    config: AuditConfig,
    eda_manifest: dict[str, Any],
    snapshots: dict[str, Path],
    masters: dict[str, pd.DataFrame],
) -> Path:
    path = config.audit_root / "audit_workspace_manifest.json"
    created_utc = utc_now()
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        created_utc = previous.get("created_utc", created_utc)
    payload = {
        "workspace_version": "v02",
        "created_utc": created_utc,
        "updated_utc": utc_now(),
        "source_run": config.eda_run_dir.name,
        "source_manifest_sha256": sha256_file(config.eda_run_dir / "manifest.json"),
        "source_eda_version": eda_manifest["version"],
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "source_snapshots": {
            name: {"path": str(snapshot), "sha256": sha256_file(snapshot)}
            for name, snapshot in snapshots.items()
        },
        "master_membership": {
            name: {
                "path": str(config.audit_root / MASTER_FILENAMES[name]),
                "rows": len(master),
                "annotation_id_membership_sha256": membership_sha256(
                    master["annotation_id"]
                ),
            }
            for name, master in masters.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_release_record(
    config: AuditConfig,
    decisions: pd.DataFrame,
    completion: pd.DataFrame,
    agreement: list[dict[str, Any]],
    linguistic_state: str,
    sampling_features: list[str],
    linguistic_errors: list[str],
    masters: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], Path]:
    known_features = set(masters["linguistic_pattern_audit"]["feature"].astype(str))
    rules, machine_rule_errors = validate_machine_rules(decisions, known_features)
    required_audits_complete = bool(
        completion["status"].isin({"complete", "not_required"}).all()
    )
    required_double = {"template_family_audit", "register_audit"}
    if linguistic_state == "required":
        required_double.add("linguistic_pattern_audit")
    agreement_by_audit = {row["audit"]: row for row in agreement}
    double_complete = bool(
        all(
            name in agreement_by_audit and agreement_by_audit[name]["complete"]
            for name in required_double
        )
    )
    release_gate_passed = bool(
        required_audits_complete
        and double_complete
        and linguistic_state in {"not_required", "required"}
        and not linguistic_errors
        and not machine_rule_errors
    )
    record = {
        "record_version": "v02",
        "source_eda_version": "v05.1",
        "source_run": config.eda_run_dir.name,
        "source_manifest_sha256": sha256_file(config.eda_run_dir / "manifest.json"),
        "audit_workspace_manifest_sha256": sha256_file(
            config.audit_root / "audit_workspace_manifest.json"
        ),
        "created_utc": utc_now(),
        "audit_completion": completion.to_dict(orient="records"),
        "double_annotation_agreement": agreement,
        "audit_metrics": build_audit_metrics(masters),
        "decisions": decisions.to_dict(orient="records"),
        "accepted_rules": rules if not machine_rule_errors else {},
        "release_gate": {
            "required_audits_complete": required_audits_complete,
            "double_annotation_complete": double_complete,
            "linguistic_state": linguistic_state,
            "sampling_linguistic_features": sampling_features,
            "linguistic_decision_errors": linguistic_errors,
            "machine_rule_errors": machine_rule_errors,
            "machine_rules_valid": not machine_rule_errors,
            "passed": release_gate_passed,
        },
    }
    path = config.audit_root / "seed_v05_decision_record_v02.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record, path
