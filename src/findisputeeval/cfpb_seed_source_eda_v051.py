"""Corrected CFPB seed-source EDA v05.1.

This module preserves the frozen v05 implementation and overrides only the
behaviour changed by the v05.1 decision record: byte-literal failure handling,
bounded redaction measures, timezone-safe monthly grouping, versioned outputs,
and the corresponding audit/feature-registry fields.
"""

from __future__ import annotations

import ast
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import cfpb_seed_source_eda_v05 as _base
from .cfpb_seed_source_eda_v05 import *  # noqa: F401,F403


class EDAConfig(_base.EDAConfig):
    """v05 configuration with a distinct default output root."""

    def __post_init__(self) -> None:
        output_root_was_supplied = self.output_root is not None
        super().__post_init__()
        if not output_root_was_supplied:
            self.output_root = (
                self.project_root
                / "outputs"
                / "data_pipeline"
                / "cfpb_seed_source_eda"
                / "eda_v051"
            )


def unwrap_byte_literal(text: str) -> tuple[str, str]:
    """Strictly unwrap UTF-8 byte literals while retaining malformed text."""
    value = str(text)
    stripped = value.strip()
    if not re.match(r"^b(['\"]).*\1$", stripped, flags=re.S):
        return value, "not_applicable"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            warnings.simplefilter("error", DeprecationWarning)
            parsed = ast.literal_eval(stripped)
        if not isinstance(parsed, bytes):
            return value, "not_bytes_after_parse"
        return parsed.decode("utf-8"), "unwrapped_utf8"
    except (SyntaxWarning, DeprecationWarning):
        return value, "invalid_escape_sequence"
    except SyntaxError as error:
        message = str(error).lower()
        if "invalid escape" in message or "unicodeescape" in message:
            return value, "invalid_escape_sequence"
        return value, "invalid_byte_literal_syntax"
    except UnicodeDecodeError:
        return value, "invalid_utf8_bytes"
    except ValueError:
        return value, "invalid_byte_literal_value"


def add_text_views(frame: pd.DataFrame) -> pd.DataFrame:
    _base.unwrap_byte_literal = unwrap_byte_literal
    result = _base.add_text_views(frame)
    failed_statuses = {
        "invalid_escape_sequence",
        "invalid_byte_literal_syntax",
        "invalid_utf8_bytes",
        "invalid_byte_literal_value",
        "not_bytes_after_parse",
    }
    result["byte_literal_parse_failed"] = result["byte_literal_status"].isin(
        failed_statuses
    )
    result["byte_literal_invalid_escape"] = result["byte_literal_status"].eq(
        "invalid_escape_sequence"
    )
    return result


def add_quality_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    canonical = result["narrative_canonical"].astype("string")
    lexical = result["narrative_lexical"].astype("string")
    result["char_count"] = canonical.str.len().astype(int)
    result["word_count"] = lexical.map(
        lambda value: len(_base.WORD_RE.findall(str(value)))
    ).astype(int)
    result["sentence_count_heuristic"] = canonical.map(
        lambda value: max(1, len(re.findall(r"[^.!?\n]+(?:[.!?]+|$)", str(value))))
    ).astype(int)
    result["redaction_count"] = canonical.str.count(r"\[REDACTED\]").astype(int)
    result["date_placeholder_count"] = canonical.str.count(r"\[DATE\]").astype(int)
    result["amount_placeholder_count"] = canonical.str.count(r"\[AMOUNT\]").astype(int)
    lexical_denominator = result["word_count"].replace(0, np.nan)
    result["redactions_per_lexical_word"] = (
        result["redaction_count"] / lexical_denominator
    ).fillna(0.0)
    placeholder_count = (
        result["redaction_count"]
        + result["date_placeholder_count"]
        + result["amount_placeholder_count"]
    )
    bounded_denominator = result["word_count"] + placeholder_count
    result["redaction_placeholder_proportion"] = (
        result["redaction_count"] / bounded_denominator.replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    result["all_placeholder_proportion"] = (
        placeholder_count / bounded_denominator.replace(0, np.nan)
    ).fillna(0.0).clip(0.0, 1.0)
    result["uppercase_ratio"] = canonical.map(_base._uppercase_ratio).astype(float)
    result["form_header_count"] = canonical.str.count(_base.FORM_HEADER_RE).astype(int)
    result["letter_marker_count"] = canonical.str.count(_base.LETTER_MARKER_RE).astype(int)
    result["legal_term_count"] = canonical.str.count(_base.LEGAL_TERM_RE).astype(int)
    result["contains_url"] = canonical.str.contains(
        r"https?://|www\.", case=False, regex=True
    )
    result["repeated_punctuation"] = canonical.str.contains(r"[!?.,-]{4,}", regex=True)
    result["quality_candidate_very_short"] = result["char_count"].lt(100)
    result["quality_candidate_very_long"] = result["char_count"].gt(5_000)
    result["quality_candidate_mostly_uppercase"] = result["uppercase_ratio"].gt(0.50)
    result["quality_candidate_heavily_redacted"] = result[
        "redaction_placeholder_proportion"
    ].gt(0.15)
    return result


def build_summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    compatibility = frame.assign(
        redaction_density=frame["redaction_placeholder_proportion"]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Converting to PeriodArray/Index representation will drop timezone",
            category=UserWarning,
        )
        summaries = _base.build_summaries(compatibility)
    summaries["frame_summary"] = summaries["frame_summary"].rename(
        columns={
            "median_redaction_density": "median_redaction_placeholder_proportion"
        }
    )
    population = frame.loc[frame["population_eligible"]]
    summaries["monthly_counts"] = (
        population.assign(month=population["date_received_parsed"].dt.strftime("%Y-%m"))
        .groupby(["month", "Product"])
        .size()
        .rename("rows")
        .reset_index()
    )
    return summaries


def build_audit_samples(
    frame: pd.DataFrame, config: EDAConfig
) -> dict[str, pd.DataFrame]:
    compatibility = frame.assign(
        redaction_density=frame["redaction_placeholder_proportion"]
    )
    audits = _base.build_audit_samples(compatibility, config)
    quality = audits.get("quality_audit")
    if quality is not None:
        quality = quality.rename(
            columns={"redaction_density": "redaction_placeholder_proportion"}
        )
        audits["quality_audit"] = quality
    return audits


def default_feature_registry(
    dependency_status: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    registry = _base.default_feature_registry(dependency_status)
    registry.pop("redaction_density", None)
    registry["redactions_per_lexical_word"] = {
        "construct": "redaction placeholders per non-placeholder lexical word",
        "unit": "document",
        "tier": "deterministic_observation",
        "validation_status": "not_required",
        "allowed_uses": ["eda", "sampling", "reporting"],
    }
    registry["redaction_placeholder_proportion"] = {
        "construct": "bounded redaction-placeholder proportion",
        "unit": "document",
        "tier": "deterministic_observation",
        "validation_status": "not_required",
        "allowed_uses": ["eda", "sampling", "reporting"],
    }
    return registry


def write_outputs(state: PipelineState) -> Path:
    manifest_path = _base.write_outputs(state)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = "v05.1"
    manifest["corrections"] = {
        "byte_literal_failures_are_typed": True,
        "redaction_measure_is_bounded_for_quality_thresholds": True,
        "monthly_grouping_is_timezone_safe": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path
