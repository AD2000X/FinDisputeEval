"""Close CFPB Seed v05.2 privacy QA as an unreviewed pipeline-only override.

This utility is deliberately not a formal privacy clearance generator.  It
backs up the mutable review CSVs, marks every row with an assumed ``no`` PII
disposition, and uses a release action that the formal privacy finalizer does
not accept.  The resulting artifacts may unblock engineering smoke plumbing,
but they cannot make the benchmark release gate pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


FINDINGS_NAME = "presidio_spacy_findings_review_v02.csv"
NEGATIVES_NAME = "negative_control_review_v02.csv"
CHALLENGES_NAME = "challenge_results_v02.csv"
DISPOSITION_NAME = "pipeline_privacy_disposition_v01.json"
BACKUP_DIR_NAME = "pre_pipeline_override_v01"
ASSUMPTION = "resource_limited_unreviewed_no_pii_assumption"
PIPELINE_ACTION = "pipeline_smoke_allow_unreviewed"
PIPELINE_ACTOR = "UNREVIEWED_PIPELINE_OVERRIDE"
NOTE = (
    "PIPELINE-ONLY ASSUMPTION: no PII was assumed without manual verification "
    "because review resources were limited. This row is not formal privacy "
    "clearance and is prohibited from benchmark release use."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_review(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {
        "manual_pii_present",
        "release_action",
        "reviewer",
        "reviewed_utc",
        "notes",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return frame


def snapshot_counts(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "manual_pii_present": frame["manual_pii_present"].value_counts().to_dict(),
        "release_action": frame["release_action"].value_counts().to_dict(),
        "reviewers": frame["reviewer"].value_counts().to_dict(),
    }


def apply_override(frame: pd.DataFrame, *, applied_utc: str) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "manual_pii_present",
        "release_action",
        "reviewer",
        "reviewed_utc",
        "notes",
    ):
        backup_column = f"pre_override_{column}"
        if backup_column in result.columns:
            raise ValueError(f"Review CSV already contains {backup_column}; refusing a second override")
        result[backup_column] = result[column]

    result["manual_pii_present"] = "no"
    result["release_action"] = PIPELINE_ACTION
    result["reviewer"] = PIPELINE_ACTOR
    result["reviewed_utc"] = applied_utc
    result["notes"] = NOTE
    result["manual_review_performed"] = "false"
    result["review_basis"] = ASSUMPTION
    result["pipeline_smoke_eligible"] = "true"
    result["benchmark_eligible"] = "false"
    return result


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def apply_pipeline_override(qa_root: Path) -> dict[str, object]:
    qa_root = qa_root.resolve()
    findings_path = qa_root / FINDINGS_NAME
    negatives_path = qa_root / NEGATIVES_NAME
    challenges_path = qa_root / CHALLENGES_NAME
    disposition_path = qa_root / DISPOSITION_NAME
    backup_dir = qa_root / BACKUP_DIR_NAME

    if disposition_path.exists() or backup_dir.exists():
        raise FileExistsError(
            "Pipeline override or its backup already exists; refusing to overwrite provenance"
        )
    for path in (findings_path, negatives_path, challenges_path):
        if not path.exists():
            raise FileNotFoundError(path)

    findings = read_review(findings_path)
    negatives = read_review(negatives_path)
    challenges = pd.read_csv(
        challenges_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if len(findings) != 957 or len(negatives) != 200:
        raise ValueError(
            f"Unexpected review inventory: findings={len(findings)}, negatives={len(negatives)}"
        )
    required = challenges["required"].str.lower().isin({"true", "1"})
    passed = challenges["challenge_passed"].str.lower().isin({"true", "1"})
    if not required.any() or not passed.loc[required].all():
        raise ValueError("Required privacy challenge results are not all passing")

    applied_utc = datetime.now(timezone.utc).isoformat()
    before = {
        "findings": snapshot_counts(findings),
        "negative_controls": snapshot_counts(negatives),
    }

    backup_dir.mkdir(parents=False)
    backup_files: dict[str, dict[str, object]] = {}
    for path in (findings_path, negatives_path):
        target = backup_dir / path.name
        shutil.copy2(path, target)
        backup_files[path.name] = {
            "path": target.relative_to(qa_root).as_posix(),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    backup_manifest = {
        "backup_version": "v01",
        "created_utc": applied_utc,
        "purpose": "preserve_pre_pipeline_override_review_state",
        "files": backup_files,
        "review_state_before_override": before,
    }
    backup_manifest_path = backup_dir / "backup_manifest.json"
    backup_manifest_path.write_text(
        json.dumps(backup_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    overridden_findings = apply_override(findings, applied_utc=applied_utc)
    overridden_negatives = apply_override(negatives, applied_utc=applied_utc)
    write_csv_atomic(overridden_findings, findings_path)
    write_csv_atomic(overridden_negatives, negatives_path)

    disposition = {
        "disposition_version": "v01",
        "scope": "full_v052_pipeline_smoke_only",
        "applied_utc": applied_utc,
        "decision": "assume_no_pii_without_manual_verification",
        "reason": "privacy review resources were limited; close the check for pipeline plumbing only",
        "manual_review_performed": False,
        "assumed_no_pii": True,
        "pipeline_smoke_only_passed": True,
        "release_clearance_passed": False,
        "benchmark_eligible": False,
        "formal_privacy_check_status": "closed_without_verification",
        "formal_finalizer_compatible": False,
        "formal_finalizer_blocking_action": PIPELINE_ACTION,
        "rows_dispositioned": {
            "findings": len(overridden_findings),
            "negative_controls": len(overridden_negatives),
        },
        "challenge_results": {
            "required": int(required.sum()),
            "passed": int(passed.loc[required].sum()),
        },
        "prohibited_uses": [
            "formal_benchmark_generation",
            "benchmark_release",
            "privacy_safety_claims",
            "population_pii_prevalence_claims",
        ],
        "required_output_flags": [
            "provisional=true",
            "benchmark_eligible=false",
            "privacy_verified=false",
        ],
        "evidence": {
            FINDINGS_NAME: {
                "sha256": sha256_file(findings_path),
                "rows": len(overridden_findings),
            },
            NEGATIVES_NAME: {
                "sha256": sha256_file(negatives_path),
                "rows": len(overridden_negatives),
            },
            CHALLENGES_NAME: {
                "sha256": sha256_file(challenges_path),
                "rows": len(challenges),
            },
            "backup_manifest": {
                "path": backup_manifest_path.relative_to(qa_root).as_posix(),
                "sha256": sha256_file(backup_manifest_path),
            },
        },
    }
    disposition_path.write_text(
        json.dumps(disposition, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return disposition


def default_qa_root(project_root: Path) -> Path:
    return (
        project_root
        / "dataset/curated/annotations/cfpb_seed_v05_audit"
        / "run_20260713T145423Z/privacy_qa/full_v052_v02"
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", type=Path, default=default_qa_root(project_root))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = apply_pipeline_override(args.qa_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
