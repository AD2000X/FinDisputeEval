"""CFPB seed-source EDA v05 with separate population and enrichment frames.

This module contains the deterministic and reusable parts of the Colab notebook.
It never deletes source rows silently. Automated linguistic and privacy fields are
candidate instruments until their exported audit files have been reviewed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Complaint ID",
]

SOURCE_SPECS = {
    "zelle_fulltext": {
        "filename": "cfpb_zelle_fulltext.csv",
        "role": "enrichment",
        "requested_date_min": "2017-06-01",
        "requested_date_max": "2026-07-02",
        "expected_rows": 18_013,
        "expected_sha256": "7b685018b4c4415a5f7219aba57cdb6e76c92c539254fafeb3c0691c44aad6f8",
    },
    "inscope_recent": {
        "filename": "cfpb_inscope_recent.csv",
        "role": "population",
        "requested_date_min": "2025-01-01",
        "requested_date_max": "2026-07-02",
        "expected_rows": 197_489,
        "expected_sha256": "cb1cb7de32169e39725d35677319d8fda9d59f52ee08d60d78bc9624e3f1dc53",
    },
    "prepaid_alltime": {
        "filename": "cfpb_prepaid_alltime.csv",
        "role": "enrichment",
        "requested_date_min": "2011-12-01",
        "requested_date_max": "2026-07-02",
        "expected_rows": 11_269,
        "expected_sha256": "46ca4fb6240995e3b403f5d133db582fb61725643d3b3cf99f7388fcd3ffb54b",
    },
}


@dataclass
class EDAConfig:
    project_root: Path
    raw_dir: Path | None = None
    output_root: Path | None = None
    run_id: str | None = None
    random_seed: int = 20260713
    language_model_path: Path | None = None
    run_lid: bool = True
    run_presidio_sample: bool = True
    presidio_sample_size: int = 2_000
    run_minhash: bool = True
    minhash_threshold: float = 0.85
    minhash_num_perm: int = 128
    minhash_min_tokens: int = 20
    minhash_pair_cap_per_document: int = 20
    minhash_max_representatives: int | None = 50_000
    run_dependency_parser: bool = True
    spacy_model: str = "en_core_web_sm"
    tfidf_max_documents: int = 40_000
    audit_rows_per_stratum: int = 50
    family_min_residue_chars: int = 40
    family_min_tokens: int = 8

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        if self.raw_dir is None:
            self.raw_dir = (
                self.project_root
                / "dataset"
                / "external"
                / "cfpb"
                / "raw"
                / "seed_build_v03_cache_2026-07-04"
            )
        else:
            self.raw_dir = Path(self.raw_dir).resolve()
        if self.output_root is None:
            self.output_root = (
                self.project_root
                / "outputs"
                / "data_pipeline"
                / "cfpb_seed_source_eda"
                / "eda_v05"
            )
        else:
            self.output_root = Path(self.output_root).resolve()
        if self.run_id is None:
            self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if self.language_model_path is None:
            self.language_model_path = self.project_root / "temp" / "models" / "lid.176.ftz"
        else:
            self.language_model_path = Path(self.language_model_path).resolve()

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / f"run_{self.run_id}"


@dataclass
class PipelineState:
    config: EDAConfig
    sources: dict[str, pd.DataFrame] = field(default_factory=dict)
    source_inventory: pd.DataFrame | None = None
    raw_union: pd.DataFrame | None = None
    universe: pd.DataFrame | None = None
    source_conflicts: pd.DataFrame | None = None
    minhash_candidates: pd.DataFrame | None = None
    summaries: dict[str, pd.DataFrame] = field(default_factory=dict)
    audits: dict[str, pd.DataFrame] = field(default_factory=dict)
    feature_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    stage_log: dict[str, Any] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _join_unique(values: Iterable[str]) -> str:
    return "|".join(sorted({str(value) for value in values if str(value)}))


def validate_source_schema(frame: pd.DataFrame, source_name: str) -> None:
    """Validate the source contract with Pandera and explicit ID checks."""
    import pandera.pandas as pa

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name}: missing required columns: {missing}")
    schema = pa.DataFrameSchema(
        {column: pa.Column(str, nullable=True, required=True) for column in REQUIRED_COLUMNS},
        strict=False,
        coerce=True,
        name=f"cfpb_{source_name}",
    )
    schema.validate(frame, lazy=True)
    if frame["Complaint ID"].eq("").any():
        raise ValueError(f"{source_name}: blank Complaint ID values")
    if frame["Complaint ID"].duplicated().any():
        duplicate_count = int(frame["Complaint ID"].duplicated(keep=False).sum())
        raise ValueError(f"{source_name}: {duplicate_count} rows have duplicate Complaint IDs")


def load_sources(config: EDAConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load the three frozen source files and produce a source inventory."""
    sources: dict[str, pd.DataFrame] = {}
    inventory_rows: list[dict[str, Any]] = []
    for source_name, spec in SOURCE_SPECS.items():
        path = Path(config.raw_dir) / spec["filename"]
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen source: {path}")
        digest = sha256_file(path)
        if digest != spec["expected_sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {path.name}: expected {spec['expected_sha256']}, got {digest}"
            )
        frame = pd.read_csv(
            path,
            dtype="string",
            encoding="utf-8-sig",
            keep_default_na=False,
            low_memory=False,
        ).astype("string")
        validate_source_schema(frame, source_name)
        if len(frame) != spec["expected_rows"]:
            raise ValueError(
                f"Row-count mismatch for {path.name}: expected {spec['expected_rows']}, got {len(frame)}"
            )
        frame = frame.copy()
        frame["source_query"] = source_name
        frame["date_received_parsed"] = pd.to_datetime(
            frame["Date received"], errors="coerce", utc=True, format="mixed"
        )
        if frame["date_received_parsed"].isna().any():
            raise ValueError(
                f"{source_name}: {int(frame['date_received_parsed'].isna().sum())} unparseable dates"
            )
        inventory_rows.append(
            {
                "source_query": source_name,
                "role": spec["role"],
                "path": str(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "rows": len(frame),
                "unique_complaint_ids": frame["Complaint ID"].nunique(),
                "rows_with_narrative": int(
                    frame["Consumer complaint narrative"].str.strip().ne("").sum()
                ),
                "requested_date_min": spec["requested_date_min"],
                "requested_date_max": spec["requested_date_max"],
                "actual_date_min": frame["date_received_parsed"].min().date().isoformat(),
                "actual_date_max": frame["date_received_parsed"].max().date().isoformat(),
            }
        )
        sources[source_name] = frame
    return sources, pd.DataFrame(inventory_rows)


def build_dual_frames(
    sources: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one population frame and non-overlapping enrichment frames."""
    combined = pd.concat(sources.values(), ignore_index=True)
    memberships = (
        combined.groupby("Complaint ID", sort=False)["source_query"]
        .agg(_join_unique)
        .rename("source_membership")
    )
    overlap_counts = (
        combined.groupby("Complaint ID", sort=False)["source_query"]
        .nunique()
        .rename("source_count")
    )

    conflict_columns = [
        "Date received",
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "Consumer complaint narrative",
        "Company",
    ]
    overlaps = combined.loc[combined["Complaint ID"].duplicated(keep=False)].copy()
    if overlaps.empty:
        conflicts = pd.DataFrame(columns=["Complaint ID", "conflicting_columns"])
    else:
        unique_counts = overlaps.groupby("Complaint ID")[conflict_columns].nunique(dropna=False)
        conflict_rows = []
        for complaint_id, row in unique_counts.iterrows():
            differing = [column for column in conflict_columns if int(row[column]) > 1]
            if differing:
                conflict_rows.append(
                    {"Complaint ID": complaint_id, "conflicting_columns": "|".join(differing)}
                )
        conflicts = pd.DataFrame(conflict_rows)

    priority = {"inscope_recent": 0, "zelle_fulltext": 1, "prepaid_alltime": 2}
    combined["_source_priority"] = combined["source_query"].map(priority).astype(int)
    universe = (
        combined.sort_values(["Complaint ID", "_source_priority"])
        .drop_duplicates("Complaint ID", keep="first")
        .drop(columns="_source_priority")
        .copy()
    )
    universe = universe.join(memberships, on="Complaint ID")
    universe = universe.join(overlap_counts, on="Complaint ID")
    universe["source_inscope_recent"] = universe["source_membership"].str.contains(
        r"(?:^|\|)inscope_recent(?:\||$)", regex=True
    )
    universe["source_zelle_fulltext"] = universe["source_membership"].str.contains(
        r"(?:^|\|)zelle_fulltext(?:\||$)", regex=True
    )
    universe["source_prepaid_alltime"] = universe["source_membership"].str.contains(
        r"(?:^|\|)prepaid_alltime(?:\||$)", regex=True
    )
    universe["population_eligible"] = universe["source_inscope_recent"]
    date_year = universe["date_received_parsed"].dt.year
    universe["temporal_bucket"] = np.where(date_year >= 2025, "2025_plus", "pre_2025")

    def frame_name(row: pd.Series) -> str:
        if bool(row["source_inscope_recent"]):
            return "population_2025plus"
        if bool(row["source_zelle_fulltext"]) and bool(row["source_prepaid_alltime"]):
            return "zelle_prepaid_enrichment_overlap"
        if bool(row["source_zelle_fulltext"]):
            return (
                "zelle_historical_enrichment"
                if row["temporal_bucket"] == "pre_2025"
                else "zelle_outside_population_enrichment"
            )
        return "prepaid_historical_enrichment"

    universe["sampling_frame"] = universe.apply(frame_name, axis=1).astype("string")
    universe["enrichment_reason"] = np.select(
        [
            universe["sampling_frame"].eq("zelle_historical_enrichment"),
            universe["sampling_frame"].eq("zelle_outside_population_enrichment"),
            universe["sampling_frame"].eq("prepaid_historical_enrichment"),
            universe["sampling_frame"].eq("zelle_prepaid_enrichment_overlap"),
        ],
        [
            "historical_zelle_supply",
            "zelle_outside_main_product_frame",
            "historical_prepaid_supply",
            "zelle_and_prepaid_enrichment",
        ],
        default="",
    )
    universe["sampling_weight"] = np.where(universe["population_eligible"], 1.0, np.nan)
    if len(universe) != 205_589:
        raise ValueError(f"Expected 205,589 unique Complaint IDs, got {len(universe):,}")
    if int(universe["population_eligible"].sum()) != 197_489:
        raise ValueError("Population frame must contain exactly 197,489 Complaint IDs")
    return combined, universe.reset_index(drop=True), conflicts


AMOUNT_RE = re.compile(r"\{\s*\$\s*[\d,.]+\s*\}")
MASKED_DATE_RE = re.compile(r"X{2}/X{2}/(?:X{2,8}|\d{2,4}|year>)", re.I)
REDACTION_RE = re.compile(r"\bX{2,}\b")
CORE_SECTION_RE = re.compile(r"(?is)\bcomplaint narrative\s*:\s*(.+)")
PLACEHOLDER_RE = re.compile(r"\[(?:REDACTED|DATE|AMOUNT)\]")
WORD_RE = re.compile(r"\b[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?\b")
FORM_HEADER_RE = re.compile(
    r"(?im)^(?:company|account|consumer|email|address|desired resolution|"
    r"complaint narrative|key facts|timeline|disputed amount|what happened)\s*:"
)
LETTER_MARKER_RE = re.compile(
    r"(?im)^(?:subject|re|dear|to whom it may concern|sincerely|respectfully)\s*[: ,]"
)
LEGAL_TERM_RE = re.compile(
    r"\b(?:U\.?S\.?C\.?|CFR|FCRA|FDCPA|FCBA|EFTA|Regulation [EZ]|statute|"
    r"statutory|violation|legal action|attorney|lawsuit|pursuant to)\b",
    re.I,
)


def unwrap_byte_literal(text: str) -> tuple[str, str]:
    value = str(text)
    stripped = value.strip()
    if not re.match(r"^b(['\"]).*\1$", stripped, flags=re.S):
        return value, "not_applicable"
    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, bytes):
            return parsed.decode("utf-8"), "unwrapped_utf8"
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return value, "unwrap_failed"
    return value, "unwrap_failed"


def extract_core_narrative(text: str) -> tuple[str, bool]:
    match = CORE_SECTION_RE.search(str(text))
    if match:
        return match.group(1).strip(), True
    return str(text).strip(), False


def canonicalize_text(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = AMOUNT_RE.sub("[AMOUNT]", value)
    value = MASKED_DATE_RE.sub("[DATE]", value)
    value = REDACTION_RE.sub("[REDACTED]", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def matching_view(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    value = PLACEHOLDER_RE.sub(" [placeholder] ", value)
    value = re.sub(r"\d+", " [number] ", value)
    value = re.sub(r"[^a-z\[\] ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def lexical_view(text: str) -> str:
    return re.sub(r"\s+", " ", PLACEHOLDER_RE.sub(" ", str(text))).strip()


def add_text_views(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    unwrapped = result["Consumer complaint narrative"].map(unwrap_byte_literal)
    result["narrative_unwrapped"] = unwrapped.map(lambda item: item[0]).astype("string")
    result["byte_literal_status"] = unwrapped.map(lambda item: item[1]).astype("string")
    core = result["narrative_unwrapped"].map(extract_core_narrative)
    result["narrative_core"] = core.map(lambda item: item[0]).astype("string")
    result["core_extraction_applied"] = core.map(lambda item: item[1]).astype(bool)
    result["narrative_raw"] = result["Consumer complaint narrative"].astype("string")
    result["narrative_canonical"] = result["narrative_core"].map(canonicalize_text).astype("string")
    result["narrative_match"] = result["narrative_canonical"].map(matching_view).astype("string")
    result["narrative_lexical"] = result["narrative_canonical"].map(lexical_view).astype("string")
    result["unicode_or_whitespace_changed"] = result["narrative_core"].ne(
        result["narrative_canonical"]
    )
    return result


def _uppercase_ratio(text: str) -> float:
    cleaned = re.sub(r"\[(?:REDACTED|DATE|AMOUNT)\]", " ", str(text))
    letters = [character for character in cleaned if character.isalpha()]
    if not letters:
        return 0.0
    return sum(character.isupper() for character in letters) / len(letters)


def add_quality_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    canonical = result["narrative_canonical"].astype("string")
    lexical = result["narrative_lexical"].astype("string")
    result["char_count"] = canonical.str.len().astype(int)
    result["word_count"] = lexical.map(lambda value: len(WORD_RE.findall(str(value)))).astype(int)
    result["sentence_count_heuristic"] = canonical.map(
        lambda value: max(1, len(re.findall(r"[^.!?\n]+(?:[.!?]+|$)", str(value))))
    ).astype(int)
    result["redaction_count"] = canonical.str.count(r"\[REDACTED\]").astype(int)
    result["date_placeholder_count"] = canonical.str.count(r"\[DATE\]").astype(int)
    result["amount_placeholder_count"] = canonical.str.count(r"\[AMOUNT\]").astype(int)
    denominator = result["word_count"].replace(0, np.nan)
    result["redaction_density"] = (result["redaction_count"] / denominator).fillna(0.0)
    result["uppercase_ratio"] = canonical.map(_uppercase_ratio).astype(float)
    result["form_header_count"] = canonical.str.count(FORM_HEADER_RE).astype(int)
    result["letter_marker_count"] = canonical.str.count(LETTER_MARKER_RE).astype(int)
    result["legal_term_count"] = canonical.str.count(LEGAL_TERM_RE).astype(int)
    result["contains_url"] = canonical.str.contains(r"https?://|www\.", case=False, regex=True)
    result["repeated_punctuation"] = canonical.str.contains(r"[!?.,-]{4,}", regex=True)
    result["quality_candidate_very_short"] = result["char_count"].lt(100)
    result["quality_candidate_very_long"] = result["char_count"].gt(5_000)
    result["quality_candidate_mostly_uppercase"] = result["uppercase_ratio"].gt(0.50)
    result["quality_candidate_heavily_redacted"] = result["redaction_density"].gt(0.15)
    return result


PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "account_context": re.compile(r"\b(?:account|acct|card)\s*(?:number|no\.?|#)\s*\d{6,}\b", re.I),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def add_pii_regex_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    def detect(text: str) -> str:
        types = [name for name, pattern in PII_PATTERNS.items() if pattern.search(str(text))]
        return "|".join(types)

    result["pii_regex_types"] = result["narrative_canonical"].map(detect).astype("string")
    result["pii_regex_risk"] = result["pii_regex_types"].ne("")
    return result


def run_presidio_on_sample(
    frame: pd.DataFrame, sample_size: int, random_seed: int
) -> pd.DataFrame:
    """Run Presidio on bounded regex-risk and control samples."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    risks = frame.loc[frame["pii_regex_risk"]].copy()
    if len(risks) > sample_size:
        risks = risks.sample(n=sample_size, random_state=random_seed)
    controls = frame.loc[~frame["pii_regex_risk"]].sample(
        n=min(sample_size, int((~frame["pii_regex_risk"]).sum())), random_state=random_seed
    )
    sample = pd.concat([risks, controls], ignore_index=True).drop_duplicates("Complaint ID")
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(), supported_languages=["en"]
    )
    rows = []
    audit_columns = [
        "Complaint ID",
        "source_membership",
        "sampling_frame",
        "pii_regex_types",
        "narrative_canonical",
    ]
    for complaint_id, source_membership, sampling_frame, regex_types, text in sample[
        audit_columns
    ].itertuples(index=False, name=None):
        text = str(text)
        findings = analyzer.analyze(text=text, language="en")
        rows.append(
            {
                "Complaint ID": str(complaint_id),
                "source_membership": source_membership,
                "sampling_frame": sampling_frame,
                "pii_regex_types": regex_types,
                "presidio_entity_types": "|".join(sorted({finding.entity_type for finding in findings})),
                "presidio_max_score": max((finding.score for finding in findings), default=0.0),
                "manual_pii_present": "",
                "manual_pii_types": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def run_language_id(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    import fasttext

    model = fasttext.load_model(str(model_path))
    texts = frame["narrative_canonical"].astype(str).str.replace("\n", " ").tolist()
    labels: list[str] = []
    probabilities: list[float] = []
    batch_size = 10_000
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        predicted_labels, predicted_probabilities = model.predict(batch, k=1)
        labels.extend(label[0].removeprefix("__label__") for label in predicted_labels)
        probabilities.extend(float(probability[0]) for probability in predicted_probabilities)
    result = frame.copy()
    result["lid_label"] = pd.Series(labels, index=result.index, dtype="string")
    result["lid_confidence"] = pd.Series(probabilities, index=result.index, dtype=float)
    result["lid_status"] = np.select(
        [
            result["lid_label"].eq("en") & result["lid_confidence"].ge(0.80),
            result["lid_label"].eq("en"),
            result["lid_confidence"].ge(0.80),
        ],
        ["english_high_confidence", "english_low_confidence", "non_english_high_confidence"],
        default="mixed_or_uncertain",
    )
    return result


def family_signature(text: str, complaint_id: str, min_chars: int, min_tokens: int) -> tuple[str, bool]:
    residue = matching_view(text)
    residue = residue.replace("[placeholder]", " ").replace("[number]", " ")
    residue = re.sub(r"\s+", " ", residue).strip()
    eligible = len(residue) >= min_chars and len(residue.split()) >= min_tokens
    if not eligible:
        return f"ineligible:{complaint_id}", False
    return sha256_text(residue), True


def add_duplicate_features(frame: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    result = frame.copy()
    result["raw_text_sha256"] = result["narrative_raw"].map(sha256_text).astype("string")
    result["canonical_text_sha256"] = result["narrative_canonical"].map(sha256_text).astype("string")
    result["exact_raw_group_size"] = result.groupby("raw_text_sha256")["raw_text_sha256"].transform("size")
    result["exact_canonical_group_size"] = result.groupby("canonical_text_sha256")[
        "canonical_text_sha256"
    ].transform("size")
    signatures = result.apply(
        lambda row: family_signature(
            row["narrative_canonical"],
            row["Complaint ID"],
            config.family_min_residue_chars,
            config.family_min_tokens,
        ),
        axis=1,
    )
    result["family_signature"] = signatures.map(lambda item: item[0]).astype("string")
    result["family_signature_eligible"] = signatures.map(lambda item: item[1]).astype(bool)
    result["family_group_size"] = result.groupby("family_signature")["family_signature"].transform("size")
    result["family_equal_weight"] = 1.0 / result["family_group_size"].clip(lower=1)
    result["is_exact_duplicate_member"] = result["exact_raw_group_size"].gt(1)
    result["is_template_family_member"] = result["family_group_size"].gt(1)
    return result


def _word_shingles(text: str, width: int = 5) -> set[str]:
    tokens = matching_view(text).split()
    if len(tokens) < width:
        return set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def run_minhash_candidates(frame: pd.DataFrame, config: EDAConfig) -> pd.DataFrame:
    """Generate auditable fuzzy-duplicate candidates from family representatives."""
    from datasketch import MinHash, MinHashLSH
    from rapidfuzz.fuzz import token_set_ratio

    representatives = (
        frame.loc[frame["family_signature_eligible"]]
        .sort_values("Complaint ID")
        .drop_duplicates("family_signature", keep="first")
    )
    if (
        config.minhash_max_representatives is not None
        and len(representatives) > config.minhash_max_representatives
    ):
        enrichment = representatives.loc[~representatives["population_eligible"]]
        population = representatives.loc[representatives["population_eligible"]]
        population_budget = max(
            0, config.minhash_max_representatives - len(enrichment)
        )
        population = population.sample(
            n=min(population_budget, len(population)),
            random_state=config.random_seed,
        )
        representatives = pd.concat([enrichment, population], ignore_index=True).sort_values(
            "Complaint ID"
        )
    lsh = MinHashLSH(threshold=config.minhash_threshold, num_perm=config.minhash_num_perm)
    shingle_cache: dict[str, set[str]] = {}
    text_cache: dict[str, str] = {}
    candidate_rows: list[dict[str, Any]] = []
    for complaint_id, text in representatives[
        ["Complaint ID", "narrative_canonical"]
    ].itertuples(index=False, name=None):
        complaint_id = str(complaint_id)
        text = str(text)
        shingles = _word_shingles(text)
        if len(shingles) < max(1, config.minhash_min_tokens - 4):
            continue
        sketch = MinHash(num_perm=config.minhash_num_perm)
        for shingle in shingles:
            sketch.update(shingle.encode("utf-8"))
        matches = lsh.query(sketch)[: config.minhash_pair_cap_per_document]
        for other_id in matches:
            other_shingles = shingle_cache[other_id]
            union = shingles | other_shingles
            jaccard = len(shingles & other_shingles) / len(union) if union else 0.0
            candidate_rows.append(
                {
                    "complaint_id_a": other_id,
                    "complaint_id_b": complaint_id,
                    "shingle_jaccard": jaccard,
                    "rapidfuzz_token_set_ratio": token_set_ratio(text_cache[other_id], text) / 100.0,
                    "manual_same_template": "",
                    "notes": "",
                }
            )
        lsh.insert(complaint_id, sketch)
        shingle_cache[complaint_id] = shingles
        text_cache[complaint_id] = text
    if not candidate_rows:
        return pd.DataFrame(
            columns=[
                "complaint_id_a",
                "complaint_id_b",
                "shingle_jaccard",
                "rapidfuzz_token_set_ratio",
                "manual_same_template",
                "notes",
            ]
        )
    return pd.DataFrame(candidate_rows).drop_duplicates(
        ["complaint_id_a", "complaint_id_b"]
    )


LEXICAL_PATTERNS = {
    "lx_negation_candidate": re.compile(
        r"\b(?:no|not|never|neither|nor|without|cannot|can't|couldn't|didn't|doesn't|don't|"
        r"wasn't|weren't|isn't|aren't|won't|wouldn't|haven't|hasn't)\b",
        re.I,
    ),
    "lx_hedging_candidate": re.compile(
        r"\b(?:I think|I believe|I guess|maybe|might|may|possibly|probably|perhaps|"
        r"not sure|seems?|appears?|pretty sure|as far as I know)\b",
        re.I,
    ),
    "lx_authorization_candidate": re.compile(
        r"\b(?:authori[sz](?:e|ed|ation)|permission|consent|recognize|"
        r"made this (?:charge|purchase|transaction))\b",
        re.I,
    ),
    "lx_escalation_candidate": re.compile(
        r"\b(?:CFPB|regulator|attorney|lawyer|lawsuit|legal action|sue|"
        r"Better Business Bureau|BBB|news media|supervisor)\b",
        re.I,
    ),
    "lx_emotion_candidate": re.compile(
        r"\b(?:frustrat\w*|ridiculous|angry|upset|stress\w*|distress\w*|"
        r"unacceptable|furious|devastat\w*|scared|afraid)\b",
        re.I,
    ),
    "lx_temporal_candidate": re.compile(
        r"\b(?:yesterday|today|last (?:week|month|year|night)|this (?:week|month)|"
        r"ago|before|after|when|since|until|recently|on \[DATE\])\b",
        re.I,
    ),
    "lx_repair_candidate": re.compile(
        r"\b(?:I mean|I meant|actually|rather|correction|no,\s*I mean)\b|\bnot\b.{0,60}\bbut\b",
        re.I,
    ),
}


def _register_candidate(row: pd.Series) -> str:
    if int(row["family_group_size"]) >= 3:
        return "template_letter_family"
    if int(row["form_header_count"]) >= 2:
        return "template_form"
    if int(row["letter_marker_count"]) >= 1:
        return "pasted_correspondence"
    if int(row["legal_term_count"]) >= 2:
        return "legal_formal"
    return "consumer_narrative"


def add_lexical_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    denominator = result["word_count"].replace(0, np.nan)
    for feature, pattern in LEXICAL_PATTERNS.items():
        counts = result["narrative_canonical"].str.count(pattern).astype(int)
        result[feature] = counts.gt(0)
        result[f"{feature}_count"] = counts
        result[f"{feature}_per_100_words"] = (counts / denominator * 100).fillna(0.0)
    result["register_candidate"] = result.apply(_register_candidate, axis=1).astype("string")
    return result


def add_dependency_features(frame: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    import spacy

    nlp = spacy.load(model_name, disable=["ner"])
    if "parser" not in nlp.pipe_names:
        raise RuntimeError(f"{model_name} does not contain an active dependency parser")
    for placeholder in ("[DATE]", "[AMOUNT]", "[REDACTED]"):
        nlp.tokenizer.add_special_case(placeholder, [{"ORTH": placeholder}])
    rows = []
    texts = frame["narrative_canonical"].astype(str).tolist()
    for doc in nlp.pipe(texts, batch_size=128):
        negation_cues = [token.text for token in doc if token.dep_ == "neg"]
        passive_tokens = [
            token.text
            for token in doc
            if token.dep_ in {"nsubjpass", "auxpass"}
            or "Pass" in token.morph.get("Voice")
        ]
        rows.append(
            {
                "dep_negation_candidate": bool(negation_cues),
                "dep_negation_cue_count": len(negation_cues),
                "dep_passive_candidate": bool(passive_tokens),
                "dep_passive_marker_count": len(passive_tokens),
                "dep_agent_relation_present": any(token.dep_ == "agent" for token in doc),
                "dep_sentence_count": len(list(doc.sents)),
            }
        )
    features = pd.DataFrame(rows, index=frame.index)
    result = pd.concat([frame, features], axis=1)
    status = {
        "backend": model_name,
        "spacy_version": spacy.__version__,
        "pipeline_components": list(nlp.pipe_names),
        "dependency_parse_executed": True,
        "rows_processed": len(result),
    }
    return result, status


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    from statistics import NormalDist

    if total == 0:
        return math.nan, math.nan
    proportion = successes / total
    z_score = NormalDist().inv_cdf(1 - alpha / 2)
    denominator = 1 + (z_score**2 / total)
    centre = (proportion + z_score**2 / (2 * total)) / denominator
    margin = (
        z_score
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z_score**2 / (4 * total**2)
        )
        / denominator
    )
    return centre - margin, centre + margin


def _binary_summary(frame: pd.DataFrame, columns: list[str], group_name: str) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for column in columns:
        successes = int(frame[column].fillna(False).astype(bool).sum())
        low, high = wilson_interval(successes, total)
        rows.append(
            {
                "group": group_name,
                "feature": column,
                "count": successes,
                "n": total,
                "rate": successes / total if total else math.nan,
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    return pd.DataFrame(rows)


def _weighted_rate(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any() or float(weights[valid].sum()) == 0:
        return math.nan
    return float(np.average(values[valid].astype(float), weights=weights[valid].astype(float)))


def build_summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    quality_columns = [
        "quality_candidate_very_short",
        "quality_candidate_very_long",
        "quality_candidate_mostly_uppercase",
        "quality_candidate_heavily_redacted",
        "contains_url",
        "repeated_punctuation",
        "pii_regex_risk",
        "is_exact_duplicate_member",
        "is_template_family_member",
    ]
    lexical_columns = list(LEXICAL_PATTERNS)
    frame_summary = (
        frame.groupby("sampling_frame", dropna=False)
        .agg(
            rows=("Complaint ID", "size"),
            unique_ids=("Complaint ID", "nunique"),
            date_min=("date_received_parsed", "min"),
            date_max=("date_received_parsed", "max"),
            median_chars=("char_count", "median"),
            median_words=("word_count", "median"),
            median_redaction_density=("redaction_density", "median"),
        )
        .reset_index()
    )
    product_distribution = (
        frame.groupby(["sampling_frame", "Product"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    quality_summary = pd.concat(
        [
            _binary_summary(group, quality_columns, str(frame_name))
            for frame_name, group in frame.groupby("sampling_frame", dropna=False)
        ],
        ignore_index=True,
    )
    linguistic_summary = pd.concat(
        [
            _binary_summary(group, lexical_columns, str(frame_name))
            for frame_name, group in frame.groupby("sampling_frame", dropna=False)
        ],
        ignore_index=True,
    )

    population = frame.loc[frame["population_eligible"]].copy()
    sensitivity_rows = []
    exact_representatives = population.drop_duplicates("canonical_text_sha256")
    family_representatives = population.drop_duplicates("family_signature")
    for feature in lexical_columns + quality_columns:
        sensitivity_rows.extend(
            [
                {"feature": feature, "view": "raw_row_weighted", "rate": population[feature].mean()},
                {"feature": feature, "view": "exact_deduplicated", "rate": exact_representatives[feature].mean()},
                {"feature": feature, "view": "family_representative", "rate": family_representatives[feature].mean()},
                {
                    "feature": feature,
                    "view": "family_equal_weighted",
                    "rate": _weighted_rate(population[feature], population["family_equal_weight"]),
                },
            ]
        )
    prevalence_sensitivity = pd.DataFrame(sensitivity_rows)
    monthly_counts = (
        population.assign(month=population["date_received_parsed"].dt.to_period("M").astype(str))
        .groupby(["month", "Product"])
        .size()
        .rename("rows")
        .reset_index()
    )
    duplicate_summary = pd.DataFrame(
        [
            {
                "frame": frame_name,
                "rows": len(group),
                "unique_raw_texts": group["raw_text_sha256"].nunique(),
                "unique_canonical_texts": group["canonical_text_sha256"].nunique(),
                "unique_template_families": group["family_signature"].nunique(),
                "largest_family": int(group["family_group_size"].max()),
            }
            for frame_name, group in frame.groupby("sampling_frame", dropna=False)
        ]
    )
    return {
        "frame_summary": frame_summary,
        "product_distribution": product_distribution,
        "quality_summary": quality_summary,
        "linguistic_summary": linguistic_summary,
        "prevalence_sensitivity": prevalence_sensitivity,
        "monthly_counts": monthly_counts,
        "duplicate_summary": duplicate_summary,
    }


def run_lexical_statistics(frame: pd.DataFrame, config: EDAConfig) -> dict[str, pd.DataFrame]:
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer

    population = (
        frame.loc[frame["population_eligible"]]
        .sort_values("Complaint ID")
        .drop_duplicates("family_signature")
    )
    if len(population) > config.tfidf_max_documents:
        population = population.sample(
            n=config.tfidf_max_documents, random_state=config.random_seed
        ).sort_values("Complaint ID")
    semantic_function_words = {
        "no", "not", "never", "nor", "without", "cannot", "may", "might", "must",
        "should", "could", "would",
    }
    stop_words = sorted(set(ENGLISH_STOP_WORDS) - semantic_function_words)
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        max_features=10_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{1,}\b",
    )
    matrix = tfidf.fit_transform(population["narrative_lexical"])
    terms = np.asarray(tfidf.get_feature_names_out())
    tfidf_top = (
        pd.DataFrame(
            {
                "term": terms,
                "mean_tfidf": np.asarray(matrix.mean(axis=0)).ravel(),
                "document_frequency": np.asarray((matrix > 0).sum(axis=0)).ravel(),
            }
        )
        .sort_values("mean_tfidf", ascending=False)
        .head(100)
        .reset_index(drop=True)
    )

    counts = CountVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=(1, 1),
        min_df=5,
        max_features=20_000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{1,}\b",
    )
    count_matrix = counts.fit_transform(population["narrative_lexical"])
    count_terms = np.asarray(counts.get_feature_names_out())
    mask_a = population["register_candidate"].eq("consumer_narrative").to_numpy()
    mask_b = population["register_candidate"].eq("legal_formal").to_numpy()
    y_a = np.asarray(count_matrix[mask_a].sum(axis=0)).ravel().astype(float)
    y_b = np.asarray(count_matrix[mask_b].sum(axis=0)).ravel().astype(float)
    alpha = np.asarray(count_matrix.sum(axis=0)).ravel().astype(float)
    alpha = alpha * (1_000.0 / max(alpha.sum(), 1.0))
    n_a, n_b, a0 = y_a.sum(), y_b.sum(), alpha.sum()
    log_odds_a = np.log((y_a + alpha) / np.clip(n_a + a0 - y_a - alpha, 1e-12, None))
    log_odds_b = np.log((y_b + alpha) / np.clip(n_b + a0 - y_b - alpha, 1e-12, None))
    delta = log_odds_a - log_odds_b
    variance = 1.0 / np.clip(y_a + alpha, 1e-12, None) + 1.0 / np.clip(y_b + alpha, 1e-12, None)
    keyness = pd.DataFrame(
        {
            "term": count_terms,
            "z": delta / np.sqrt(variance),
            "count_consumer": y_a.astype(int),
            "count_legal": y_b.astype(int),
            "status": "exploratory_until_register_validation",
            "prior_strength": 1_000.0,
        }
    ).sort_values("z", ascending=False)

    ngrams = CountVectorizer(
        lowercase=True,
        stop_words=stop_words,
        ngram_range=(2, 3),
        min_df=5,
        max_features=30_000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{1,}\b",
    )
    ngram_matrix = ngrams.fit_transform(population["narrative_lexical"])
    frequent_ngrams = pd.DataFrame(
        {
            "ngram": ngrams.get_feature_names_out(),
            "term_count": np.asarray(ngram_matrix.sum(axis=0)).ravel().astype(int),
            "document_frequency": np.asarray((ngram_matrix > 0).sum(axis=0)).ravel().astype(int),
        }
    ).sort_values(["document_frequency", "term_count"], ascending=False)
    frequent_ngrams["status"] = "frequency_only_not_association_collocation"
    return {
        "tfidf_top_terms": tfidf_top,
        "register_keyness_exploratory": keyness,
        "frequent_ngrams": frequent_ngrams,
    }


def _sample_groups(
    frame: pd.DataFrame,
    group_columns: list[str],
    n_per_group: int,
    random_seed: int,
) -> pd.DataFrame:
    parts = []
    for offset, (_, group) in enumerate(frame.groupby(group_columns, dropna=False, sort=True)):
        parts.append(
            group.sample(n=min(n_per_group, len(group)), random_state=random_seed + offset)
        )
    if not parts:
        return frame.head(0).copy()
    return pd.concat(parts, ignore_index=True).drop_duplicates("Complaint ID")


def build_audit_samples(frame: pd.DataFrame, config: EDAConfig) -> dict[str, pd.DataFrame]:
    audits: dict[str, pd.DataFrame] = {}
    quality_flags = [
        "quality_candidate_very_short",
        "quality_candidate_very_long",
        "quality_candidate_mostly_uppercase",
        "quality_candidate_heavily_redacted",
        "pii_regex_risk",
    ]
    risk_key = frame[quality_flags].astype(int).astype(str).agg("".join, axis=1)
    quality_source = frame.assign(quality_risk_pattern=risk_key)
    quality_audit = _sample_groups(
        quality_source,
        ["sampling_frame", "quality_risk_pattern"],
        min(20, config.audit_rows_per_stratum),
        config.random_seed,
    )
    audits["quality_audit"] = quality_audit[
        [
            "Complaint ID", "sampling_frame", "Product", "Issue", "char_count", "word_count",
            "redaction_density", "quality_risk_pattern", "narrative_canonical",
        ]
    ].assign(manual_quality="", exclusion_reason="", notes="")

    register_audit = _sample_groups(
        frame,
        ["sampling_frame", "register_candidate"],
        config.audit_rows_per_stratum,
        config.random_seed + 1_000,
    )
    audits["register_audit"] = register_audit[
        [
            "Complaint ID", "sampling_frame", "Product", "Issue", "register_candidate",
            "family_group_size", "form_header_count", "letter_marker_count", "legal_term_count",
            "narrative_canonical",
        ]
    ].assign(manual_register="", manual_correct="", notes="")

    pattern_rows = []
    for offset, feature in enumerate(LEXICAL_PATTERNS):
        for expected, subset in (
            (True, frame.loc[frame[feature]]),
            (False, frame.loc[~frame[feature]]),
        ):
            if subset.empty:
                continue
            sampled = subset.sample(
                n=min(config.audit_rows_per_stratum, len(subset)),
                random_state=config.random_seed + 2_000 + offset + int(expected),
            )
            for complaint_id, sampling_frame, narrative in sampled[
                ["Complaint ID", "sampling_frame", "narrative_canonical"]
            ].itertuples(index=False, name=None):
                pattern_rows.append(
                    {
                        "Complaint ID": str(complaint_id),
                        "feature": feature,
                        "detector_output": expected,
                        "sampling_frame": sampling_frame,
                        "narrative_canonical": narrative,
                        "manual_present": "",
                        "notes": "",
                    }
                )
    pattern_audit = pd.DataFrame(pattern_rows)
    audits["linguistic_pattern_audit"] = pattern_audit

    family_members = frame.loc[frame["family_group_size"].gt(1)].copy()
    family_pairs = []
    for family_id, group in family_members.groupby("family_signature", sort=False):
        if len(family_pairs) >= 500:
            break
        group = group.sort_values("Complaint ID")
        first = group.iloc[0]
        second = group.iloc[1]
        family_pairs.append(
            {
                "family_signature": family_id,
                "family_size": len(group),
                "complaint_id_a": first["Complaint ID"],
                "complaint_id_b": second["Complaint ID"],
                "text_a": first["narrative_canonical"],
                "text_b": second["narrative_canonical"],
                "manual_same_template": "",
                "notes": "",
            }
        )
    audits["template_family_audit"] = pd.DataFrame(family_pairs)

    if "lid_status" in frame.columns:
        lid_frame = frame.copy()
        lid_frame["lid_confidence_bin"] = pd.cut(
            lid_frame["lid_confidence"],
            bins=[-np.inf, 0.50, 0.80, 0.95, np.inf],
            labels=["lt_0.50", "0.50_0.80", "0.80_0.95", "ge_0.95"],
        ).astype("string")
        lid_audit = _sample_groups(
            lid_frame,
            ["sampling_frame", "lid_status", "lid_confidence_bin"],
            config.audit_rows_per_stratum,
            config.random_seed + 4_000,
        )
        audits["lid_audit"] = lid_audit[
            [
                "Complaint ID", "sampling_frame", "lid_label", "lid_confidence",
                "lid_status", "lid_confidence_bin", "narrative_canonical",
            ]
        ].assign(manual_language="", manual_english="", notes="")
    return audits


def default_feature_registry(dependency_status: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    deterministic = {
        "char_count": ("document length", "document"),
        "word_count": ("lexical token count", "document"),
        "redaction_density": ("placeholder redaction density", "document"),
        "raw_text_sha256": ("exact raw duplicate identity", "document"),
        "family_signature": ("normalization-equivalent template identity", "document"),
        "sampling_frame": ("source sampling design", "document"),
    }
    for feature, (construct, unit) in deterministic.items():
        registry[feature] = {
            "construct": construct,
            "unit": unit,
            "tier": "deterministic_observation",
            "validation_status": "not_required",
            "allowed_uses": ["eda", "sampling", "reporting"],
        }
    for feature in LEXICAL_PATTERNS:
        registry[feature] = {
            "construct": feature.removeprefix("lx_").removesuffix("_candidate"),
            "unit": "document",
            "tier": "regex_heuristic",
            "validation_status": "pending_human_audit",
            "precision": None,
            "recall": None,
            "f1": None,
            "allowed_uses": ["sampling", "exploratory_eda"],
        }
    registry["register_candidate"] = {
        "construct": "complaint register",
        "unit": "document",
        "tier": "rule_based_candidate",
        "validation_status": "pending_double_annotation",
        "allowed_uses": ["sampling", "exploratory_eda"],
    }
    if dependency_status:
        for feature in (
            "dep_negation_candidate",
            "dep_passive_candidate",
            "dep_agent_relation_present",
        ):
            registry[feature] = {
                "construct": feature.removeprefix("dep_").removesuffix("_candidate"),
                "unit": "document",
                "tier": "parser_backed_candidate",
                "backend": dependency_status.get("backend"),
                "validation_status": "pending_human_audit",
                "precision": None,
                "recall": None,
                "f1": None,
                "allowed_uses": ["sampling", "exploratory_eda"],
            }
    return registry


def write_outputs(state: PipelineState) -> Path:
    if state.universe is None or state.source_inventory is None:
        raise ValueError("Pipeline state is incomplete")
    output_dir = state.config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "analysis_ready_corpus.parquet"
    safe_path = output_dir / "repo_safe_features.parquet"
    state.universe.to_parquet(full_path, index=False)
    text_columns = [
        "Consumer complaint narrative",
        "narrative_raw",
        "narrative_unwrapped",
        "narrative_core",
        "narrative_canonical",
        "narrative_match",
        "narrative_lexical",
        "Company",
        "ZIP code",
    ]
    repo_safe = state.universe.drop(
        columns=[column for column in text_columns if column in state.universe.columns]
    ).copy()
    repo_safe["source_id_sha256"] = repo_safe["Complaint ID"].map(sha256_text)
    repo_safe = repo_safe.drop(columns=["Complaint ID"])
    repo_safe.to_parquet(safe_path, index=False)

    state.source_inventory.to_csv(output_dir / "raw_inventory.csv", index=False, encoding="utf-8-sig")
    if state.source_conflicts is not None:
        state.source_conflicts.to_csv(
            output_dir / "source_conflicts.csv", index=False, encoding="utf-8-sig"
        )
    if state.minhash_candidates is not None:
        state.minhash_candidates.to_parquet(output_dir / "fuzzy_duplicate_candidates.parquet", index=False)
    for name, summary in state.summaries.items():
        summary.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    for name, audit in state.audits.items():
        audit.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (output_dir / "feature_registry.json").write_text(
        json.dumps(state.feature_registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "pipeline": "CFPB Seed Source EDA",
        "version": "v05",
        "created_utc": utc_now(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(state.config).items()},
        "source_design": {
            "population_frame": "inscope_recent only",
            "enrichment_rule": "only Complaint IDs absent from inscope_recent",
            "population_rows": int(state.universe["population_eligible"].sum()),
            "enrichment_rows": int((~state.universe["population_eligible"]).sum()),
            "universe_unique_ids": len(state.universe),
        },
        "stage_log": state.stage_log,
        "inputs": state.source_inventory.to_dict(orient="records"),
        "outputs": {},
        "claim_boundary": (
            "Automated register, linguistic, language, fuzzy-duplicate, and PII fields are candidates "
            "until their exported audit files have been completed. Population rates must use only "
            "population_eligible=True."
        ),
    }
    output_files = [path for path in output_dir.iterdir() if path.is_file()]
    manifest["outputs"] = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in output_files
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
