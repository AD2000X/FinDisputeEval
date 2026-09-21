"""Repair an Excel-exported tab-delimited adjudication file safely."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = (
    ROOT
    / "dataset/curated/annotations/cfpb_seed_v05_audit/run_20260713T145423Z"
)
TARGET = AUDIT_ROOT / "register_audit_adjudication.csv"
BACKUP = AUDIT_ROOT / "register_audit_adjudication_user_export.tsv"

EXPECTED_COLUMNS = [
    "annotation_id",
    "Complaint ID",
    "sampling_frame",
    "Product",
    "Issue",
    "register_candidate",
    "family_group_size",
    "form_header_count",
    "letter_marker_count",
    "legal_term_count",
    "narrative_canonical",
    "manual_correct",
    "double_annotation_required",
    "manual_register_A",
    "manual_register_B",
    "adjudicated_value",
    "adjudicator_id",
    "adjudication_notes",
]


def main() -> None:
    if BACKUP.exists():
        raise FileExistsError(f"Refusing to replace existing backup: {BACKUP}")
    frame = pd.read_csv(
        TARGET,
        sep="\t",
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if len(frame) != 150:
        raise ValueError(f"Expected 150 rows, found {len(frame)}")
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if frame["annotation_id"].duplicated().any():
        raise ValueError("Duplicate annotation IDs")

    shutil.copy2(TARGET, BACKUP)
    frame = frame[EXPECTED_COLUMNS]
    frame.to_csv(TARGET, index=False, encoding="utf-8-sig")

    check = pd.read_csv(
        TARGET,
        dtype="string",
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if not check.equals(frame):
        raise RuntimeError("Round-trip validation failed")
    print(f"Backup: {BACKUP}")
    print(f"Repaired CSV: {TARGET}")
    print(f"Rows: {len(check)}, columns: {len(check.columns)}")


if __name__ == "__main__":
    main()
