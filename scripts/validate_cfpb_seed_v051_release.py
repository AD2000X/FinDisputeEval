"""Validate CFPB Seed v05.1 candidate artifacts and privacy-release gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


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
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "account": re.compile(
        r"\b(?:account|acct|card)\s*(?:number|no\.?|#)\s*\d{6,}\b", re.I
    ),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # A long run immediately after a decimal point is a fractional value, not
    # an identifier (for example ``0.000044235 %``).  The upstream CFPB PII
    # detector likewise does not classify that case as PII.
    "long_digit_run": re.compile(r"(?<![\d.])\d{9,}(?!\d)"),
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def read_output(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    raise ValueError(path)


def validate(
    release_dir: Path,
    decision_record: Path,
    eda_manifest: Path,
    require_ner: bool,
) -> dict[str, Any]:
    manifest_path = release_dir / "seed_v051_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_record.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    check(
        manifest.get("parent_eda_manifest_sha256") == sha256_file(eda_manifest),
        "Parent EDA manifest hash mismatch",
        errors,
    )
    check(
        manifest.get("parent_decision_record_v03_sha256")
        == sha256_file(decision_record),
        "Parent decision record v03 hash mismatch",
        errors,
    )
    check(
        decision.get("release_gate", {}).get("passed") is True,
        "Decision v03 candidate-build gate is not passed",
        errors,
    )

    frames: dict[str, pd.DataFrame] = {}
    output_checks: list[dict[str, Any]] = []
    for name, metadata in manifest["outputs"].items():
        relative = Path(metadata["path"])
        check(not relative.is_absolute(), f"Manifest path is not relative: {name}", errors)
        path = release_dir / relative
        exists = path.exists()
        size_ok = exists and path.stat().st_size == int(metadata["size_bytes"])
        hash_ok = exists and sha256_file(path) == metadata["sha256"]
        row_ok = False
        if exists:
            frame = read_output(path)
            frames[name] = frame
            row_ok = len(frame) == int(metadata["rows"])
        check(exists, f"Missing output: {name}", errors)
        check(size_ok, f"Output size mismatch: {name}", errors)
        check(hash_ok, f"Output hash mismatch: {name}", errors)
        check(row_ok, f"Output row mismatch: {name}", errors)
        output_checks.append(
            {
                "name": name,
                "path": str(relative),
                "exists": exists,
                "size_ok": size_ok,
                "hash_ok": hash_ok,
                "rows_ok": row_ok,
            }
        )

    seed = frames["seed_parquet"]
    seed_json = frames["seed_jsonl"]
    population = frames["population_parquet"]
    enrichment = frames["enrichment_parquet"]
    stress = frames["stress_parquet"]
    excluded = frames["excluded_parquet"]
    generation = frames["generation_parquet"]
    generation_json = frames["generation_jsonl"]

    seed_ids = set(seed["Complaint ID"].astype(str))
    stress_ids = set(stress["Complaint ID"].astype(str))
    excluded_ids = set(excluded["Complaint ID"].astype(str))
    check(len(seed) == len(population) + len(enrichment), "Seed split rows do not reconcile", errors)
    check(
        seed_ids
        == set(population["Complaint ID"].astype(str))
        | set(enrichment["Complaint ID"].astype(str)),
        "Seed split Complaint IDs do not reconcile",
        errors,
    )
    check(seed_ids.isdisjoint(stress_ids), "Seed and stress IDs overlap", errors)
    check(seed_ids.isdisjoint(excluded_ids), "Seed and excluded IDs overlap", errors)
    check(stress_ids.isdisjoint(excluded_ids), "Stress and excluded IDs overlap", errors)
    check(seed["seed_id"].is_unique, "Seed stable IDs are not unique", errors)
    check(stress["stress_id"].is_unique, "Stress stable IDs are not unique", errors)
    check("_selection_key" not in seed, "Internal _selection_key leaked to seed", errors)
    check("_selection_key" not in stress, "Internal _selection_key leaked to stress", errors)
    check(
        seed.groupby("canonical_text_sha256").size().max() <= 1,
        "Exact duplicate cap failed",
        errors,
    )
    check(
        seed.groupby("family_signature").size().max() <= 1,
        "Template-family cap failed",
        errors,
    )
    if "confirmed_fuzzy_cluster" in seed:
        check(
            seed.groupby("confirmed_fuzzy_cluster").size().max() <= 1,
            "Confirmed fuzzy-cluster cap failed",
            errors,
        )

    for name, frame in [("seed", seed), ("stress", stress)]:
        check("seed_text" in frame, f"{name} lacks seed_text", errors)
        check(
            not frame["seed_text"].fillna("").str.strip().eq("").any(),
            f"{name} contains empty seed_text",
            errors,
        )
        leaked = sorted(SOURCE_TEXT_COLUMNS & set(frame.columns))
        check(not leaked, f"{name} leaks source text columns: {leaked}", errors)
        uncleared = frame.loc[
            frame["pii_regex_risk"].fillna(False)
            & ~frame["pii_clearance_action"].eq("allow_after_regex_redaction")
        ]
        check(uncleared.empty, f"{name} contains uncleared regex-risk rows", errors)
        for pattern_name, pattern in PII_PATTERNS.items():
            count = int(frame["seed_text"].str.contains(pattern, na=False).sum())
            check(count == 0, f"{name} contains {count} residual {pattern_name} matches", errors)

    check("seed_text" not in excluded, "Excluded artifact contains release text", errors)
    override_ids = set(
        pd.read_csv(
            decision_record.parent / decision["rules"]["row_overrides"]["path"],
            dtype="string",
            keep_default_na=False,
            encoding="utf-8-sig",
        )["Complaint ID"].astype(str)
    )
    check(override_ids <= excluded_ids, "Not every row override is in excluded output", errors)

    expected_stress = manifest["rules"]["sampling_quotas"]["stress_quotas"]
    actual_stress = stress["stress_reason"].value_counts().to_dict()
    check(
        {key: int(actual_stress.get(key, 0)) for key in expected_stress}
        == {key: int(value) for key, value in expected_stress.items()},
        "Stress reason quotas do not match selected rows",
        errors,
    )
    for row in manifest["enrichment_accounting"]:
        check(row["shortfall_accepted"] is True, f"Unaccepted shortfall: {row['sampling_frame']}", errors)
        check(
            int(row["selected"]) >= int(row["minimum_acceptable"]),
            f"Enrichment below minimum: {row['sampling_frame']}",
            errors,
        )

    expected_generation_columns = manifest["rules"]["release_schema"][
        "generation_columns"
    ]
    check(
        list(generation.columns) == expected_generation_columns,
        "Generation input schema mismatch",
        errors,
    )
    check(
        set(generation["seed_id"].astype(str)) == set(seed["seed_id"].astype(str)),
        "Generation input IDs do not reconcile with seed",
        errors,
    )
    check(
        set(generation_json["seed_id"].astype(str)) == set(generation["seed_id"].astype(str)),
        "Generation JSONL and parquet IDs disagree",
        errors,
    )
    for flag in ["benchmark_eligible", "publication_eligible", "may_enter_final_dataset"]:
        check(not generation[flag].astype(bool).any(), f"Candidate generation flag is true: {flag}", errors)

    xxxx_counts = {
        "seed": int(seed["seed_text"].str.contains(r"XXXX", regex=True, na=False).sum()),
        "stress": int(stress["seed_text"].str.contains(r"XXXX", regex=True, na=False).sum()),
        "generation": int(
            generation["seed_narrative_excerpt"].str.contains(r"XXXX", regex=True, na=False).sum()
        ),
    }
    if any(xxxx_counts.values()):
        warnings.append(
            "Canonical XXXX tokens remain unchanged by policy and require generation-layer review."
        )

    presidio_available = importlib.util.find_spec("presidio_analyzer") is not None
    spacy_available = importlib.util.find_spec("spacy") is not None
    ner_scan_status = "not_run_dependency_unavailable"
    if presidio_available and spacy_available:
        ner_scan_status = "available_but_not_run_by_structural_validator"
        warnings.append(
            "Presidio/spaCy are available; run the dedicated full privacy scan and manual review."
        )
    elif require_ner:
        errors.append("NER privacy QA required but Presidio/spaCy is unavailable")

    structural_passed = not errors
    benchmark_gate_passed = structural_passed and ner_scan_status == "passed"  # fail closed
    report = {
        "report_version": "v01",
        "release": manifest["release"],
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(manifest_path),
        "decision_record_v03_sha256": sha256_file(decision_record),
        "checks": {
            "output_integrity": output_checks,
            "row_accounting": {
                "seed": len(seed),
                "population": len(population),
                "enrichment": len(enrichment),
                "stress": len(stress),
                "excluded": len(excluded),
                "generation": len(generation),
            },
            "stress_distribution": {
                str(key): int(value) for key, value in actual_stress.items()
            },
            "row_overrides_in_excluded": len(override_ids & excluded_ids),
            "xxxx_counts": xxxx_counts,
            "regex_privacy_scan": "passed" if not any("residual" in error for error in errors) else "failed",
            "ner_privacy_scan": ner_scan_status,
        },
        "errors": errors,
        "warnings": warnings,
        "structural_release_validator_passed": structural_passed,
        "benchmark_release_gate_passed": benchmark_gate_passed,
        "benchmark_blockers": (
            []
            if benchmark_gate_passed
            else [
                "full_seed_and_stress_presidio_or_equivalent_ner_scan",
                "manual_review_of_final_privacy_scan_positives",
            ]
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=root / "dataset/curated/seed_pools/cfpb_dispute/seed_v051",
    )
    parser.add_argument(
        "--decision-record",
        type=Path,
        default=(
            root
            / "dataset/curated/annotations/cfpb_seed_v05_audit"
            / "run_20260713T145423Z/seed_v051_decision_record_v03.json"
        ),
    )
    parser.add_argument(
        "--eda-manifest",
        type=Path,
        default=(
            root
            / "outputs/data_pipeline/cfpb_seed_source_eda/eda_v051"
            / "run_20260713T145423Z/manifest.json"
        ),
    )
    parser.add_argument("--require-ner", action="store_true")
    args = parser.parse_args()
    report = validate(
        args.release_dir.resolve(),
        args.decision_record.resolve(),
        args.eda_manifest.resolve(),
        args.require_ner,
    )
    output = args.release_dir / "seed_v051_release_qa.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["structural_release_validator_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
