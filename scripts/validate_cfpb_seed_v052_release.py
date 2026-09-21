"""Validate CFPB Seed v05.2 candidate artifacts and structural gates.

The validator deliberately cannot grant privacy clearance.  A separate full
Presidio/spaCy scan plus manual adjudication must supply that evidence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from findisputeeval.curation.cfpb_seed_common import sha256_file


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
    "long_digit_run": re.compile(r"(?<![\d.])\d{9,}(?!\d)"),
}


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def read_output(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    raise ValueError(f"Unsupported output: {path}")


def validate(
    release_dir: Path,
    decision_record: Path,
    eda_manifest: Path,
    privacy_clearance: Path | None = None,
) -> dict[str, Any]:
    manifest_path = release_dir / "seed_v052_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_record.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    check(manifest.get("release") == "CFPB Seed v05.2 candidate", "Wrong release", errors)
    check(
        manifest.get("parent_eda_manifest_sha256") == sha256_file(eda_manifest),
        "Parent EDA manifest hash mismatch",
        errors,
    )
    check(
        manifest.get("parent_decision_record_v04_sha256") == sha256_file(decision_record),
        "Parent decision v04 hash mismatch",
        errors,
    )
    check(decision.get("build_gate", {}).get("passed") is True, "Decision v04 build gate closed", errors)

    frames: dict[str, pd.DataFrame] = {}
    integrity: list[dict[str, Any]] = []
    for name, metadata in manifest["outputs"].items():
        relative = Path(metadata["path"])
        check(not relative.is_absolute(), f"Non-relative manifest path: {name}", errors)
        path = release_dir / relative
        exists = path.exists()
        size_ok = exists and path.stat().st_size == int(metadata["size_bytes"])
        hash_ok = exists and sha256_file(path) == metadata["sha256"]
        rows_ok = False
        if exists:
            frames[name] = read_output(path)
            rows_ok = len(frames[name]) == int(metadata["rows"])
        check(exists, f"Missing output: {name}", errors)
        check(size_ok, f"Size mismatch: {name}", errors)
        check(hash_ok, f"Hash mismatch: {name}", errors)
        check(rows_ok, f"Row mismatch: {name}", errors)
        integrity.append({"name": name, "exists": exists, "size_ok": size_ok, "hash_ok": hash_ok, "rows_ok": rows_ok})

    seed = frames["seed_parquet"]
    core = frames["population_core_parquet"]
    coverage = frames["coverage_supplement_parquet"]
    enrichment = frames["enrichment_parquet"]
    stress = frames["stress_parquet"]
    excluded = frames["excluded_parquet"]
    generation = frames["generation_parquet"]
    allocation = frames["primary_allocation_csv"]
    coverage_accounting = frames["coverage_accounting_csv"]

    ids = lambda frame: set(frame["Complaint ID"].astype(str))
    check(len(core) == 2_000, "Proportional core is not 2,000 rows", errors)
    check(len(seed) == len(core) + len(coverage) + len(enrichment), "Seed splits do not reconcile", errors)
    check(ids(seed) == ids(core) | ids(coverage) | ids(enrichment), "Seed IDs do not reconcile", errors)
    check(ids(core).isdisjoint(ids(coverage)), "Core overlaps coverage supplement", errors)
    check(ids(seed).isdisjoint(ids(stress)), "Seed overlaps stress", errors)
    check(ids(seed).isdisjoint(ids(excluded)), "Seed overlaps excluded", errors)
    check(seed["Complaint ID"].is_unique, "Complaint IDs are duplicated", errors)
    check(seed["seed_id"].is_unique, "Seed stable IDs are duplicated", errors)
    check(stress["stress_id"].is_unique, "Stress stable IDs are duplicated", errors)
    check(set(core["release_split"]) == {"population_core"}, "Core split label incorrect", errors)
    check(set(coverage["release_split"]) <= {"coverage_supplement"}, "Coverage split label incorrect", errors)
    check(set(enrichment["release_split"]) <= {"enrichment"}, "Enrichment split label incorrect", errors)
    check(not coverage["prevalence_eligible"].astype(bool).any(), "Coverage marked prevalence eligible", errors)
    check("_selection_key" not in seed.columns, "Internal selection key leaked", errors)
    check(seed.groupby("canonical_text_sha256").size().max() <= 1, "Exact cap failed", errors)
    check(seed.groupby("family_signature").size().max() <= 1, "Family cap failed", errors)
    if "confirmed_fuzzy_cluster" in seed:
        check(seed.groupby("confirmed_fuzzy_cluster").size().max() <= 1, "Audited fuzzy cap failed", errors)

    check(int(pd.to_numeric(allocation["allocated"]).sum()) == 2_000, "Core allocation does not total 2,000", errors)
    selected_coverage = pd.to_numeric(coverage_accounting["selected"])
    check(int(selected_coverage.sum()) == len(coverage), "Coverage accounting does not reconcile", errors)
    check((selected_coverage <= 1).all(), "More than one coverage row selected per stratum", errors)
    check(
        len(coverage) <= int(decision["rules"]["sampling_quotas"]["coverage_supplement"]["maximum_rows"]),
        "Coverage supplement exceeds approved maximum",
        errors,
    )

    for name, frame in (("seed", seed), ("stress", stress)):
        leaked = sorted(SOURCE_TEXT_COLUMNS & set(frame.columns))
        check(not leaked, f"{name} leaks source columns: {leaked}", errors)
        check("seed_text" in frame and not frame["seed_text"].fillna("").str.strip().eq("").any(), f"{name} has empty text", errors)
        for pattern_name, pattern in PII_PATTERNS.items():
            count = int(frame["seed_text"].str.contains(pattern, na=False).sum())
            check(count == 0, f"{name} has {count} residual {pattern_name} matches", errors)
    check("seed_text" not in excluded.columns, "Excluded output contains release text", errors)

    expected_columns = decision["rules"]["release_schema"]["generation_columns"]
    check(list(generation.columns) == expected_columns, "Generation schema mismatch", errors)
    check(set(generation["seed_id"].astype(str)) == set(seed["seed_id"].astype(str)), "Generation IDs do not reconcile", errors)
    for flag in ("benchmark_eligible", "publication_eligible", "may_enter_final_dataset"):
        check(not generation[flag].astype(bool).any(), f"Candidate flag is true: {flag}", errors)

    expected_stress = manifest["rules"]["sampling_quotas"]["stress_quotas"]
    actual_stress = stress["stress_reason"].value_counts().to_dict()
    check(
        {key: int(actual_stress.get(key, 0)) for key in expected_stress}
        == {key: int(value) for key, value in expected_stress.items()},
        "Stress quotas do not reconcile",
        errors,
    )
    for row in manifest["enrichment_accounting"]:
        check(row["shortfall_accepted"] is True, f"Unaccepted shortfall: {row['sampling_frame']}", errors)
        check(int(row["selected"]) >= int(row["minimum_acceptable"]), f"Below minimum: {row['sampling_frame']}", errors)

    privacy_status = "not_provided"
    privacy_hash = None
    if privacy_clearance is not None:
        clearance = json.loads(privacy_clearance.read_text(encoding="utf-8"))
        privacy_hash = sha256_file(privacy_clearance)
        privacy_status = "passed" if clearance.get("release_clearance_passed") is True else "failed"
        check(clearance.get("scope") == "full_v052", "Privacy clearance has wrong scope", errors)
        check(
            clearance.get("source_sha256") == sha256_file(manifest_path),
            "Privacy clearance is not bound to this manifest",
            errors,
        )
        check(privacy_status == "passed", "Full privacy clearance did not pass", errors)
    else:
        warnings.append("Full Presidio/spaCy NER clearance is still required.")

    structural_passed = not errors
    benchmark_passed = structural_passed and privacy_status == "passed"
    return {
        "report_version": "v01",
        "release": manifest["release"],
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(manifest_path),
        "decision_record_v04_sha256": sha256_file(decision_record),
        "privacy_clearance_sha256": privacy_hash,
        "checks": {
            "output_integrity": integrity,
            "rows": {"seed": len(seed), "core": len(core), "coverage": len(coverage), "enrichment": len(enrichment), "stress": len(stress), "excluded": len(excluded)},
            "privacy_clearance": privacy_status,
        },
        "errors": errors,
        "warnings": warnings,
        "structural_release_validator_passed": structural_passed,
        "benchmark_release_gate_passed": benchmark_passed,
        "benchmark_blockers": [] if benchmark_passed else ["full_v052_ner_scan_and_manual_adjudication"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=root / "dataset/curated/seed_pools/cfpb_dispute/seed_v052")
    parser.add_argument("--decision-record", type=Path, default=root / "dataset/curated/annotations/cfpb_seed_v05_audit/run_20260713T145423Z/seed_v052_decision_record_v04.json")
    parser.add_argument("--eda-manifest", type=Path, default=root / "outputs/data_pipeline/cfpb_seed_source_eda/eda_v051/run_20260713T145423Z/manifest.json")
    parser.add_argument("--privacy-clearance", type=Path)
    args = parser.parse_args()
    report = validate(args.release_dir.resolve(), args.decision_record.resolve(), args.eda_manifest.resolve(), args.privacy_clearance.resolve() if args.privacy_clearance else None)
    output = args.release_dir / "seed_v052_release_qa.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["structural_release_validator_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
