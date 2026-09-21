"""Prepare a deterministic CFPB Seed v05.2 pipeline-only smoke sample.

The source release has an explicit unreviewed privacy disposition.  This
preparer therefore refuses formal clearance, verifies the disposition and its
evidence, removes the original narrative excerpt, and sends only a
defense-in-depth redacted grounding excerpt to the generation provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


_PROJECT_ROOT_ENV = os.environ.get("FINDISPUTEEVAL_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_PROJECT_ROOT_ENV).resolve()
    if _PROJECT_ROOT_ENV
    else Path(__file__).resolve().parents[3]
)
SEED_ROOT = PROJECT_ROOT / "dataset/curated/seed_pools/cfpb_dispute/seed_v052"
PRIVACY_ROOT = (
    PROJECT_ROOT
    / "dataset/curated/annotations/cfpb_seed_v05_audit"
    / "run_20260713T145423Z/privacy_qa/full_v052_v02"
)
SMOKE_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override"
DEFAULT_SEED_INPUT = SEED_ROOT / "cfpb_seed_v052_generation_input.jsonl"
DEFAULT_SEED_MANIFEST = SEED_ROOT / "seed_v052_manifest.json"
DEFAULT_DISPOSITION = PRIVACY_ROOT / "pipeline_privacy_disposition_v01.json"
DEFAULT_OUTPUT_DIR = SMOKE_ROOT / "prepared_inputs"

PIPELINE_ACTION = "pipeline_smoke_allow_unreviewed"
REQUIRED_FLAGS = {
    "provisional=true",
    "benchmark_eligible=false",
    "privacy_verified=false",
}
REGEX_REDACTIONS = [
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"),
    re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I),
    re.compile(r"(?<!\d)\d{9,}(?!\d)"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_evidence_path(root: Path, name: str, item: dict[str, Any]) -> Path:
    relative = str(item.get("path", name)).replace("\\", "/")
    path = (root / relative).resolve()
    if root.resolve() not in (path, *path.parents):
        raise ValueError(f"Disposition evidence escapes its workspace: {relative}")
    return path


def verify_pipeline_disposition(
    disposition_path: Path,
    seed_manifest_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    expected = {
        "pipeline_smoke_only_passed": True,
        "manual_review_performed": False,
        "release_clearance_passed": False,
        "benchmark_eligible": False,
        "formal_privacy_check_status": "closed_without_verification",
    }
    mismatched = {
        key: (disposition.get(key), value)
        for key, value in expected.items()
        if disposition.get(key) != value
    }
    if mismatched:
        raise ValueError(f"Pipeline disposition policy mismatch: {mismatched}")
    if set(disposition.get("required_output_flags", [])) != REQUIRED_FLAGS:
        raise ValueError("Pipeline disposition required flags are missing or unexpected")

    privacy_root = disposition_path.parent
    evidence = disposition.get("evidence", {})
    required_evidence = {
        "presidio_spacy_findings_review_v02.csv",
        "negative_control_review_v02.csv",
        "challenge_results_v02.csv",
        "backup_manifest",
    }
    if set(evidence) != required_evidence:
        raise ValueError("Pipeline disposition evidence inventory is incomplete or unexpected")
    evidence_paths: dict[str, Path] = {}
    for name, item in evidence.items():
        path = resolve_evidence_path(privacy_root, name, item)
        if not path.exists():
            raise FileNotFoundError(path)
        if sha256_file(path) != item.get("sha256"):
            raise ValueError(f"Pipeline disposition evidence hash mismatch: {name}")
        evidence_paths[name] = path

    findings = pd.read_csv(
        evidence_paths["presidio_spacy_findings_review_v02.csv"],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    negatives = pd.read_csv(
        evidence_paths["negative_control_review_v02.csv"],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    for label, frame, expected_rows in (
        ("findings", findings, 957),
        ("negative controls", negatives, 200),
    ):
        if len(frame) != expected_rows:
            raise ValueError(f"Unexpected {label} row count: {len(frame)}")
        if not frame["manual_pii_present"].eq("no").all():
            raise ValueError(f"{label} do not all carry the assumed no disposition")
        if not frame["release_action"].eq(PIPELINE_ACTION).all():
            raise ValueError(f"{label} are not marked pipeline-only")
        if not frame["manual_review_performed"].str.lower().eq("false").all():
            raise ValueError(f"{label} incorrectly claim manual review")
        if not frame["benchmark_eligible"].str.lower().eq("false").all():
            raise ValueError(f"{label} incorrectly permit benchmark use")

    analyzer_config_path = privacy_root / "analyzer_config_v02.json"
    scan_inventory_path = privacy_root / "scan_inventory_v02.json"
    for path in (analyzer_config_path, scan_inventory_path):
        if not path.exists():
            raise FileNotFoundError(path)
    analyzer_config = json.loads(analyzer_config_path.read_text(encoding="utf-8"))
    inventory = json.loads(scan_inventory_path.read_text(encoding="utf-8"))
    seed_manifest_sha = sha256_file(seed_manifest_path)
    if analyzer_config.get("source_manifest_sha256") != seed_manifest_sha:
        raise ValueError("Privacy analyzer config is not bound to the Seed v05.2 manifest")
    if inventory.get("source_manifest_sha256") != seed_manifest_sha:
        raise ValueError("Privacy scan inventory is not bound to the Seed v05.2 manifest")
    if inventory.get("analyzer_config_sha256") != sha256_file(analyzer_config_path):
        raise ValueError("Privacy scan inventory analyzer hash mismatch")
    if int(inventory.get("scanned_records", -1)) != 3204:
        raise ValueError("Privacy scan inventory did not cover all 3,204 release records")
    return disposition, findings


def stable_key(random_seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"{random_seed}|{namespace}|{value}".encode("utf-8")
    ).hexdigest()


def stratified_sample(
    rows: list[dict[str, Any]],
    size: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    if not 10 <= size <= 20:
        raise ValueError("Pipeline smoke sample size must be between 10 and 20")
    if len(rows) < size:
        raise ValueError(f"Only {len(rows)} eligible rows are available")
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cell = (
            str(row.get("release_split", "")),
            str(row.get("product", "")),
            str(row.get("issue", "")),
        )
        by_cell[cell].append(row)
    cells = sorted(
        by_cell,
        key=lambda cell: stable_key(random_seed, "cell", "|".join(cell)),
    )
    for cell in cells:
        by_cell[cell].sort(
            key=lambda row: stable_key(random_seed, "record", str(row["seed_id"]))
        )
    selected: list[dict[str, Any]] = []
    rank = 0
    while len(selected) < size:
        progressed = False
        for cell in cells:
            if rank >= len(by_cell[cell]):
                continue
            selected.append(dict(by_cell[cell][rank]))
            progressed = True
            if len(selected) == size:
                break
        if not progressed:
            break
        rank += 1
    return sorted(
        selected,
        key=lambda row: stable_key(random_seed, "output", str(row["seed_id"])),
    )


def redact_grounding_excerpt(
    text: str,
    finding_rows: pd.DataFrame,
) -> tuple[str, int]:
    spans: list[tuple[int, int]] = []
    for row in finding_rows.to_dict(orient="records"):
        try:
            start, end = int(row["start"]), int(row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if start < len(text) and end > 0 and start < end:
            spans.append((max(0, start), min(len(text), end)))
    for pattern in REGEX_REDACTIONS:
        spans.extend((match.start(), match.end()) for match in pattern.finditer(text))
    if not spans:
        return text, 0
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    redacted = text
    for start, end in reversed(merged):
        redacted = redacted[:start] + "[PII_CANDIDATE]" + redacted[end:]
    return redacted, len(merged)


def prepare_smoke_inputs(
    *,
    seed_input: Path,
    seed_manifest: Path,
    disposition_path: Path,
    output_dir: Path,
    sample_size: int,
    random_seed: int,
) -> tuple[Path, Path, dict[str, Any]]:
    release = json.loads(seed_manifest.read_text(encoding="utf-8"))
    if release.get("release") != "CFPB Seed v05.2 candidate":
        raise ValueError("This preparer accepts only the Seed v05.2 candidate")
    if release.get("benchmark_eligible") is not False:
        raise ValueError("Seed v05.2 candidate must remain benchmark-ineligible")
    expected_input_hash = release["outputs"]["generation_jsonl"]["sha256"]
    actual_input_hash = sha256_file(seed_input)
    if actual_input_hash != expected_input_hash:
        raise ValueError("Generation input hash does not match Seed v05.2 manifest")

    disposition, findings = verify_pipeline_disposition(disposition_path, seed_manifest)
    rows = load_jsonl(seed_input)
    required = {
        "seed_id",
        "release_split",
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "seed_narrative_excerpt",
        "source_release",
        "source_release_status",
        "benchmark_eligible",
        "publication_eligible",
        "may_enter_final_dataset",
    }
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    if missing:
        raise ValueError(f"Generation input schema is missing: {missing}")
    eligible = [
        row
        for row in rows
        if row.get("source_release") == "CFPB Seed v05.2 candidate"
        and row.get("source_release_status") == "privacy_qa_pending"
        and row.get("benchmark_eligible") is False
        and row.get("publication_eligible") is False
        and row.get("may_enter_final_dataset") is False
    ]
    sample = stratified_sample(eligible, sample_size, random_seed)
    findings_by_id = {
        record_id: group
        for record_id, group in findings.groupby("release_record_id", sort=False)
    }
    disposition_sha = sha256_file(disposition_path)
    seed_manifest_sha = sha256_file(seed_manifest)
    prepared: list[dict[str, Any]] = []
    redaction_total = 0
    for index, source in enumerate(sample):
        seed_id = str(source["seed_id"])
        excerpt = str(source.pop("seed_narrative_excerpt", ""))
        redacted, count = redact_grounding_excerpt(
            excerpt,
            findings_by_id.get(seed_id, findings.iloc[0:0]),
        )
        redaction_total += count
        row = {
            key: source[key]
            for key in (
                "seed_id",
                "release_split",
                "product",
                "sub_product",
                "issue",
                "sub_issue",
                "sampling_frame",
                "enrichment_reason",
                "month",
                "register_candidate",
                "lid_status",
            )
            if key in source
        }
        row.update(
            {
                "smoke_sample_index": index,
                "generation_grounding_excerpt": redacted,
                "source_excerpt_sha256": sha256_text(excerpt),
                "grounding_redaction_count": count,
                "source_seed_manifest_sha256": seed_manifest_sha,
                "pipeline_privacy_disposition_sha256": disposition_sha,
                "pipeline_smoke_only": True,
                "provisional": True,
                "privacy_verified": False,
                "benchmark_eligible": False,
                "publication_eligible": False,
                "may_enter_final_dataset": False,
                "distribution_calibration_eligible": False,
            }
        )
        prepared.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"nemo_seed_v052_pipeline_smoke_{len(prepared)}.jsonl"
    manifest_path = output_dir / "pipeline_smoke_input_manifest.json"
    write_jsonl(sample_path, prepared)
    manifest: dict[str, Any] = {
        "manifest_version": "v01",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "cfpb_seed_v052_pipeline_smoke_only",
        "source_release": release["release"],
        "source_release_status": release["release_status"],
        "source_seed_manifest_sha256": seed_manifest_sha,
        "source_generation_input_sha256": actual_input_hash,
        "pipeline_privacy_disposition_sha256": disposition_sha,
        "privacy_status": disposition["formal_privacy_check_status"],
        "manual_review_performed": False,
        "random_seed": random_seed,
        "requested_rows": sample_size,
        "selected_rows": len(prepared),
        "selected_seed_ids": [row["seed_id"] for row in prepared],
        "grounding_redactions": redaction_total,
        "selection_note": "Coverage-oriented engineering sample; not a distribution estimate.",
        "product_distribution": dict(
            sorted(Counter(row["product"] for row in prepared).items())
        ),
        "release_split_distribution": dict(
            sorted(Counter(row["release_split"] for row in prepared).items())
        ),
        "policy": {
            "pipeline_smoke_only": True,
            "provisional": True,
            "privacy_verified": False,
            "benchmark_eligible": False,
            "publication_eligible": False,
            "may_enter_final_dataset": False,
            "distribution_calibration_eligible": False,
            "formal_pilot_allowed": False,
        },
        "output": {
            "path": sample_path.name,
            "rows": len(prepared),
            "sha256": sha256_file(sample_path),
            "size_bytes": sample_path.stat().st_size,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sample_path, manifest_path, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-input", type=Path, default=DEFAULT_SEED_INPUT)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    parser.add_argument("--privacy-disposition", type=Path, default=DEFAULT_DISPOSITION)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample, manifest_path, manifest = prepare_smoke_inputs(
        seed_input=args.seed_input,
        seed_manifest=args.seed_manifest,
        disposition_path=args.privacy_disposition,
        output_dir=args.out_dir,
        sample_size=args.sample_size,
        random_seed=args.random_seed,
    )
    print(
        json.dumps(
            {
                "sample": str(sample),
                "manifest": str(manifest_path),
                "rows": manifest["selected_rows"],
                "grounding_redactions": manifest["grounding_redactions"],
                **manifest["policy"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
