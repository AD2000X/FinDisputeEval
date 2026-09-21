"""Create an external, hash-bound inventory for a completed smoke run.

The inventory is intentionally stored outside the source run directory so the
completed run remains byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
        FrozenFile,
        FrozenManifest,
        row_count,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct execution from this script's directory.
    from cfpb_v052_pipeline_smoke_v02_common import (  # type: ignore[no-redef]
        FrozenFile,
        FrozenManifest,
        row_count,
        sha256_file,
    )


_PROJECT_ROOT_ENV = os.environ.get("FINDISPUTEEVAL_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_PROJECT_ROOT_ENV).resolve()
    if _PROJECT_ROOT_ENV
    else Path(__file__).resolve().parents[3]
)
SMOKE_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override"
DEFAULT_RUN = SMOKE_ROOT / "run_20260722T135306Z"
DEFAULT_OUTPUT = SMOKE_ROOT / "frozen_manifests/run_20260722T135306Z_files_v01.json"


def freeze_run(run_dir: Path, output_path: Path) -> FrozenManifest:
    run_dir = run_dir.resolve()
    smoke_root = SMOKE_ROOT.resolve()
    if smoke_root not in (run_dir, *run_dir.parents):
        raise ValueError("Run directory must be within the pipeline-smoke workspace")
    output_resolved = output_path.resolve()
    if run_dir in (output_resolved, *output_resolved.parents):
        raise ValueError("Frozen manifest must be stored outside the immutable run")

    run_manifest = run_dir / "pipeline_smoke_run_manifest.json"
    validation_report = run_dir / "pipeline_smoke_validation_report.json"
    if not run_manifest.is_file() or not validation_report.is_file():
        raise ValueError("Only a completed run with both manifests may be frozen")
    run_policy = json.loads(run_manifest.read_text(encoding="utf-8"))
    if run_policy.get("pipeline_plumbing_passed") is not True:
        raise ValueError("Run manifest does not record a completed pipeline")
    if run_policy.get("policy", {}).get("benchmark_eligible") is not False:
        raise ValueError("Smoke evidence must remain benchmark-ineligible")

    files = [path for path in sorted(run_dir.rglob("*")) if path.is_file()]
    entries = [
        FrozenFile(
            path=path.relative_to(run_dir).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
            rows=row_count(path),
        )
        for path in files
    ]
    manifest = FrozenManifest(
        source_run_id=str(run_policy.get("run_id", run_dir.name.removeprefix("run_"))),
        source_run_path=run_dir.relative_to(smoke_root).as_posix(),
        created_utc=datetime.now(timezone.utc).isoformat(),
        files=entries,
        file_count=len(entries),
        total_size_bytes=sum(entry.size_bytes for entry in entries),
        note=(
            "Tamper-evident inventory only; it does not change privacy_verified=false "
            "or benchmark_eligible=false. Revalidation artifacts belong outside this run."
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def verify_frozen_manifest(manifest_path: Path) -> dict[str, object]:
    manifest = FrozenManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    run_dir = (SMOKE_ROOT / manifest.source_run_path).resolve()
    expected = {entry.path: entry for entry in manifest.files}
    actual_paths = {
        path.relative_to(run_dir).as_posix(): path
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    missing = sorted(set(expected) - set(actual_paths))
    unexpected = sorted(set(actual_paths) - set(expected))
    changed = sorted(
        relative
        for relative in set(expected) & set(actual_paths)
        if actual_paths[relative].stat().st_size != expected[relative].size_bytes
        or sha256_file(actual_paths[relative]) != expected[relative].sha256
        or row_count(actual_paths[relative]) != expected[relative].rows
    )
    result: dict[str, object] = {
        "verified": not (missing or unexpected or changed),
        "source_run_id": manifest.source_run_id,
        "file_count": len(actual_paths),
        "missing": missing,
        "unexpected": unexpected,
        "changed": changed,
    }
    if not result["verified"]:
        raise ValueError(f"Frozen run verification failed: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify --output against the source run instead of rewriting it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_existing:
        print(
            json.dumps(
                verify_frozen_manifest(args.output), ensure_ascii=False, indent=2
            )
        )
        return
    manifest = freeze_run(args.run_dir, args.output)
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
