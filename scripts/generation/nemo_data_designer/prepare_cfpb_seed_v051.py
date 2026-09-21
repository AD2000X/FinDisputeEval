"""Prepare a strictly provisional NeMo smoke-test sample from Seed v05.1.

The output is engineering-only.  It is intentionally marked ineligible for
benchmark, publication, calibration, and the final dataset until the full
release privacy gate is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEED_INPUT = (
    PROJECT_ROOT
    / "dataset/curated/seed_pools/cfpb_dispute/seed_v051"
    / "cfpb_seed_v051_generation_input.jsonl"
)
DEFAULT_SEED_MANIFEST = DEFAULT_SEED_INPUT.parent / "seed_v051_manifest.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/generation/smoke_only/cfpb_seed_v051_candidate/prepared_inputs"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_sample(rows: list[dict], size: int, random_seed: int) -> list[dict]:
    if not 10 <= size <= 20:
        raise ValueError("Provisional smoke sample size must be between 10 and 20")
    if len(rows) < size:
        raise ValueError(f"Only {len(rows)} eligible rows are available")
    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        cell = (
            str(row.get("release_split", "")),
            str(row.get("product", "")),
            str(row.get("issue", "")),
        )
        by_cell[cell].append(row)
    rng = random.Random(random_seed)
    cells = sorted(by_cell)
    rng.shuffle(cells)
    selected: list[dict] = []
    used: set[str] = set()
    # Round-robin gives broad product/issue coverage without treating this
    # engineering sample as a benchmark-distribution estimate.
    while len(selected) < size:
        progressed = False
        for cell in cells:
            bucket = sorted(by_cell[cell], key=lambda row: str(row["seed_id"]))
            remaining = [row for row in bucket if str(row["seed_id"]) not in used]
            if not remaining:
                continue
            choice = remaining[rng.randrange(len(remaining))]
            selected.append(choice)
            used.add(str(choice["seed_id"]))
            progressed = True
            if len(selected) == size:
                break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-input", type=Path, default=DEFAULT_SEED_INPUT)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260715)
    args = parser.parse_args()

    release = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    if release.get("release") != "CFPB Seed v05.1 candidate":
        raise ValueError("This preparer accepts only the Seed v05.1 candidate")
    if release.get("benchmark_eligible") is not False:
        raise ValueError("Smoke input must remain benchmark-ineligible")
    expected = release["outputs"]["generation_jsonl"]["sha256"]
    actual = sha256_file(args.seed_input)
    if actual != expected:
        raise ValueError("Generation input hash does not match Seed v05.1 manifest")

    rows = load_jsonl(args.seed_input)
    required = {
        "seed_id", "release_split", "product", "issue", "seed_narrative_excerpt",
        "source_release_status", "benchmark_eligible", "publication_eligible",
        "may_enter_final_dataset",
    }
    missing = sorted(required - set(rows[0])) if rows else sorted(required)
    if missing:
        raise ValueError(f"Generation input schema is missing: {missing}")
    eligible = [
        row
        for row in rows
        if row.get("benchmark_eligible") is False
        and row.get("publication_eligible") is False
        and row.get("may_enter_final_dataset") is False
        and row.get("source_release_status") == "privacy_qa_pending"
    ]
    sample = stratified_sample(eligible, args.sample_size, args.random_seed)
    for row in sample:
        row["smoke_test_only"] = True
        row["benchmark_eligible"] = False
        row["publication_eligible"] = False
        row["may_enter_final_dataset"] = False
        row["distribution_calibration_eligible"] = False

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.out_dir / f"nemo_seed_v051_provisional_smoke_{len(sample)}.jsonl"
    manifest_path = args.out_dir / "provisional_smoke_input_manifest.json"
    write_jsonl(sample_path, sample)
    manifest = {
        "manifest_version": "v01",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "engineering_smoke_test_only",
        "source_release": release["release"],
        "source_release_status": release["release_status"],
        "source_seed_manifest_sha256": sha256_file(args.seed_manifest),
        "source_generation_input_sha256": actual,
        "random_seed": args.random_seed,
        "requested_rows": args.sample_size,
        "selected_rows": len(sample),
        "selection_note": "Coverage-oriented sample; never use for distribution calibration.",
        "product_distribution": dict(sorted(Counter(row["product"] for row in sample).items())),
        "release_split_distribution": dict(
            sorted(Counter(row["release_split"] for row in sample).items())
        ),
        "policy": {
            "benchmark_eligible": False,
            "publication_eligible": False,
            "may_enter_final_dataset": False,
            "distribution_calibration_eligible": False,
            "formal_pilot_minimum": 50,
            "formal_pilot_allowed": False,
        },
        "output": {
            "path": sample_path.name,
            "rows": len(sample),
            "sha256": sha256_file(sample_path),
            "size_bytes": sample_path.stat().st_size,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"sample": str(sample_path), "manifest": str(manifest_path), **manifest["policy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
