"""Validate Seed v05.2 plus the formal privacy-clearance v02 evidence chain."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from validate_cfpb_seed_v052_release import validate as validate_v01


FORMAL_MODELS = {"en_core_web_lg", "en_core_web_trf"}
REQUIRED_EVIDENCE = {
    "analyzer_config",
    "environment_lock",
    "findings_review",
    "negative_controls",
    "challenge_set",
    "challenge_results",
}


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_manual_review(frame: pd.DataFrame, label: str, errors: list[str]) -> None:
    required = {"manual_pii_present", "release_action", "reviewer"}
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"{label} review columns missing: {missing}")
        return
    labels = frame["manual_pii_present"].astype(str).str.lower()
    actions = frame["release_action"].astype(str).str.lower()
    check(labels.isin({"yes", "no"}).all(), f"{label} review is incomplete", errors)
    check(
        ~frame["reviewer"].astype(str).str.strip().eq("").any(),
        f"{label} review has missing reviewer",
        errors,
    )
    check(
        actions.loc[labels.eq("no")].eq("allow").all(),
        f"{label} negative actions are invalid",
        errors,
    )
    check(
        actions.loc[labels.eq("yes")].isin({"exclude", "redact_and_rescan"}).all(),
        f"{label} positive actions are invalid",
        errors,
    )


def validate(
    release_dir: Path,
    decision_record: Path,
    eda_manifest: Path,
    privacy_clearance: Path,
) -> dict[str, Any]:
    # V01 is the structural validator. Privacy is intentionally withheld here
    # so a failed clearance cannot pollute the structural status.
    base = validate_v01(
        release_dir,
        decision_record,
        eda_manifest,
        None,
    )
    structural_errors = list(base["errors"])
    privacy_errors: list[str] = []
    warnings = [
        warning
        for warning in base["warnings"]
        if warning != "Full Presidio/spaCy NER clearance is still required."
    ]
    checks: dict[str, Any] = dict(base["checks"])
    privacy_checks: dict[str, Any] = {}

    if not privacy_clearance.exists():
        privacy_errors.append(f"Formal privacy clearance v02 is missing: {privacy_clearance}")
        warnings.append("Full privacy QA v02 clearance is still required.")
        checks["privacy_clearance"] = "not_provided"
    else:
        clearance = json.loads(privacy_clearance.read_text(encoding="utf-8"))
        manifest = json.loads((release_dir / "seed_v052_manifest.json").read_text(encoding="utf-8"))
        check(clearance.get("clearance_version") == "v02", "Privacy clearance is not v02", privacy_errors)
        check(clearance.get("scope") == "full_v052", "Privacy scope is not full_v052", privacy_errors)
        check(
            clearance.get("source_sha256") == sha256_file(release_dir / "seed_v052_manifest.json"),
            "Privacy clearance is not bound to this manifest",
            privacy_errors,
        )
        expected_scanned = int(manifest["primary_rows"]) + int(manifest["coverage_supplement_rows"]) + int(manifest["enrichment_rows"]) + int(manifest["stress_rows"])
        check(
            int(clearance.get("scanned_records", -1)) == expected_scanned,
            f"Privacy scanned_records must equal {expected_scanned}",
            privacy_errors,
        )
        analyzer = clearance.get("analyzer", {})
        check(analyzer.get("model") in FORMAL_MODELS, "Formal privacy model is not approved", privacy_errors)
        check(analyzer.get("engine") == "presidio_spacy", "Privacy analyzer engine is unexpected", privacy_errors)

        evidence = clearance.get("evidence", {})
        check(set(evidence) == REQUIRED_EVIDENCE, "Privacy evidence inventory is incomplete or unexpected", privacy_errors)
        loaded: dict[str, pd.DataFrame] = {}
        loaded_json: dict[str, dict[str, Any]] = {}
        for name in sorted(REQUIRED_EVIDENCE & set(evidence)):
            item = evidence[name]
            path = privacy_clearance.parent / str(item.get("path", ""))
            check(path.exists(), f"Privacy evidence missing: {name}", privacy_errors)
            if not path.exists():
                continue
            check(sha256_file(path) == item.get("sha256"), f"Privacy evidence hash mismatch: {name}", privacy_errors)
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
                loaded[name] = frame
                check(len(frame) == int(item.get("rows", -1)), f"Privacy evidence row mismatch: {name}", privacy_errors)
            elif path.suffix.lower() == ".json":
                loaded_json[name] = json.loads(path.read_text(encoding="utf-8"))

        if "analyzer_config" in evidence:
            check(
                analyzer.get("config_sha256") == evidence["analyzer_config"].get("sha256"),
                "Analyzer config hash does not match evidence",
                privacy_errors,
            )
        analyzer_config = loaded_json.get("analyzer_config", {})
        environment_lock = loaded_json.get("environment_lock", {})
        if analyzer_config and environment_lock:
            check(
                analyzer_config.get("environment_lock_sha256")
                == evidence["environment_lock"].get("sha256"),
                "Analyzer config is not bound to the environment lock",
                privacy_errors,
            )
            check(
                environment_lock.get("model_name") == analyzer.get("model"),
                "Environment lock model differs from clearance",
                privacy_errors,
            )
            check(
                analyzer_config.get("model_version") == environment_lock.get("model_version"),
                "Analyzer model version differs from environment lock",
                privacy_errors,
            )
            check(
                analyzer_config.get("challenge_gate_policy")
                == "any_high_risk_detection_overlapping_expected_sensitive_span",
                "Challenge gate policy is missing or unexpected",
                privacy_errors,
            )
            helper_hashes = analyzer_config.get("privacy_helper_sha256", {})
            expected_helpers = {"cfpb_privacy_qa.py", "cfpb_privacy_qa_v02.py"}
            check(
                set(helper_hashes) == expected_helpers,
                "Privacy helper hash inventory is incomplete or unexpected",
                privacy_errors,
            )
            project_root = Path(__file__).resolve().parents[1]
            for helper_name in sorted(expected_helpers & set(helper_hashes)):
                helper_path = project_root / "src/findisputeeval/curation" / helper_name
                check(
                    helper_path.exists() and sha256_file(helper_path) == helper_hashes[helper_name],
                    f"Privacy helper hash mismatch: {helper_name}",
                    privacy_errors,
                )
        findings = loaded.get("findings_review")
        negatives = loaded.get("negative_controls")
        challenges = loaded.get("challenge_results")
        if findings is not None:
            validate_manual_review(findings, "Finding", privacy_errors)
            if "manual_pii_present" in findings:
                check(
                    not findings["manual_pii_present"].str.lower().eq("yes").any(),
                    "Confirmed PII remains in finding review",
                    privacy_errors,
                )
        if negatives is not None:
            validate_manual_review(negatives, "Negative-control", privacy_errors)
            check(len(negatives) >= 200, "Fewer than 200 negative controls were reviewed", privacy_errors)
            if "manual_pii_present" in negatives:
                check(
                    not negatives["manual_pii_present"].str.lower().eq("yes").any(),
                    "Missed PII was found in negative controls",
                    privacy_errors,
                )
            if "release_split" in negatives:
                check(
                    {"population_core", "coverage_supplement", "enrichment", "stress"}
                    <= set(negatives["release_split"]),
                    "Negative controls do not cover every release split",
                    privacy_errors,
                )
        if findings is not None and negatives is not None:
            check(
                "release_record_id" in findings and "release_record_id" in negatives,
                "Privacy reviews do not expose release_record_id",
                privacy_errors,
            )
        if findings is not None and negatives is not None and "release_record_id" in findings and "release_record_id" in negatives:
            check(
                set(findings["release_record_id"]).isdisjoint(set(negatives["release_record_id"])),
                "Negative controls include records with NER findings",
                privacy_errors,
            )
        if challenges is not None:
            required = challenges["required"].str.lower().isin({"true", "1"})
            passed = challenges["challenge_passed"].str.lower().isin({"true", "1"})
            check(required.any(), "No required privacy challenges exist", privacy_errors)
            check(passed.loc[required].all(), "At least one required privacy challenge failed", privacy_errors)

        summary_negative = clearance.get("negative_control_review", {})
        summary_challenge = clearance.get("challenge_validation", {})
        check(int(summary_negative.get("records_reviewed", -1)) >= 200, "Clearance summary has too few negative controls", privacy_errors)
        check(int(summary_negative.get("confirmed_pii_records", -1)) == 0, "Clearance summary reports missed PII", privacy_errors)
        check(summary_challenge.get("all_required_passed") is True, "Clearance summary reports failed challenges", privacy_errors)
        check(clearance.get("release_clearance_passed") is True, "Privacy clearance did not pass", privacy_errors)
        checks["privacy_clearance"] = (
            "passed" if not privacy_errors and clearance.get("release_clearance_passed") is True else "failed"
        )
        privacy_checks = {
            "clearance_version": clearance.get("clearance_version"),
            "model": analyzer.get("model"),
            "scanned_records": clearance.get("scanned_records"),
            "findings_reviewed": clearance.get("finding_review", {}).get("findings_reviewed"),
            "negative_controls_reviewed": summary_negative.get("records_reviewed"),
            "challenge_required": summary_challenge.get("required"),
            "challenge_failed": summary_challenge.get("failed"),
            "challenge_type_mismatches": summary_challenge.get("type_mismatches"),
            "evidence_hashes_verified": not any("evidence" in error.lower() or "config hash" in error.lower() for error in privacy_errors),
        }

    structural_passed = base["structural_release_validator_passed"]
    benchmark_passed = structural_passed and not privacy_errors
    errors = structural_errors + privacy_errors
    blockers = []
    if not structural_passed:
        blockers.append("seed_v052_structural_validation")
    if not benchmark_passed:
        blockers.append("full_v052_privacy_v02_evidence")
    checks["privacy_v02"] = privacy_checks
    return {
        **base,
        "report_version": "v02",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "privacy_clearance_sha256": sha256_file(privacy_clearance) if privacy_clearance.exists() else None,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "structural_release_validator_passed": structural_passed,
        "benchmark_release_gate_passed": benchmark_passed,
        "benchmark_blockers": blockers,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=root / "dataset/curated/seed_pools/cfpb_dispute/seed_v052")
    parser.add_argument("--decision-record", type=Path, default=root / "dataset/curated/annotations/cfpb_seed_v05_audit/run_20260713T145423Z/seed_v052_decision_record_v04.json")
    parser.add_argument("--eda-manifest", type=Path, default=root / "outputs/data_pipeline/cfpb_seed_source_eda/eda_v051/run_20260713T145423Z/manifest.json")
    parser.add_argument("--privacy-clearance", type=Path, default=root / "dataset/curated/annotations/cfpb_seed_v05_audit/run_20260713T145423Z/privacy_qa/full_v052_v02/privacy_clearance_v02.json")
    args = parser.parse_args()
    report = validate(
        args.release_dir.resolve(),
        args.decision_record.resolve(),
        args.eda_manifest.resolve(),
        args.privacy_clearance.resolve(),
    )
    output = args.release_dir / "seed_v052_release_qa_v02.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["benchmark_release_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
