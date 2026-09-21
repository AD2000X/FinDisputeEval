"""Generate the self-contained Colab/VS Code CFPB Seed v05.1 builder notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "notebooks/20_seed_curation/cfpb_dispute/canonical"
    / "FinDisputeEval_CFPB_SeedPool_Build_colab_v051.ipynb"
)


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def payload(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), base64.b64encode(gzip.compress(raw)).decode("ascii")


def main() -> None:
    v05_hash, v05_blob = payload(ROOT / "src/findisputeeval/curation/cfpb_seed_v05.py")
    v051_hash, v051_blob = payload(ROOT / "src/findisputeeval/curation/cfpb_seed_v051.py")
    cells = [
        markdown(
            "# FinDisputeEval CFPB Seed Pool Build v05.1\n\n"
            "Builds a **candidate**, not a benchmark release. It reads the frozen EDA v05.1 run and hash-bound decision record v03, writes directly to Google Drive, and leaves EDA and Seed v05 unchanged. Formal generation remains blocked until full NER and manual privacy QA pass.\n"
        ),
        code(
            """from pathlib import Path
import hashlib
import json
import sys

IN_COLAB = "google.colab" in sys.modules
RUN_ID = "run_20260713T145423Z"
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/FinDisputeEval")
else:
    here = Path.cwd().resolve()
    ROOT = next((p for p in (here, *here.parents) if (p / "WORK_PROGRESS.md").exists()), None)
    if ROOT is None:
        raise FileNotFoundError("Open the FinDisputeEval repository in VS Code.")

EDA_RUN_DIR = ROOT / "outputs/data_pipeline/cfpb_seed_source_eda/eda_v051" / RUN_ID
AUDIT_ROOT = ROOT / "dataset/curated/annotations/cfpb_seed_v05_audit" / RUN_ID
DECISION_RECORD = AUDIT_ROOT / "seed_v051_decision_record_v03.json"
OUTPUT_DIR = ROOT / "dataset/curated/seed_pools/cfpb_dispute/seed_v051"
print(f"Runtime: {'Colab' if IN_COLAB else 'local VS Code'}")
print(f"Frozen EDA: {EDA_RUN_DIR}")
print(f"Governance: {DECISION_RECORD}")
print(f"Direct persistent output: {OUTPUT_DIR}")
"""
        ),
        markdown(
            "## Required Drive migration\n\n"
            "Before running, the EDA run must exist under `MyDrive/FinDisputeEval/outputs/...`, and the audit directory must contain decision v03, row overrides, PII clearance, and fuzzy adjudication. This cell fails closed on any missing or changed file.\n"
        ),
        code(
            """def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

required = [
    EDA_RUN_DIR / "manifest.json",
    EDA_RUN_DIR / "analysis_ready_corpus.parquet",
    DECISION_RECORD,
    AUDIT_ROOT / "seed_v051_row_overrides_v01.csv",
    AUDIT_ROOT / "pii_release_clearance_v01.csv",
    AUDIT_ROOT / "fuzzy_duplicate_audit_master.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError("Missing persistent inputs:\\n" + "\\n".join(missing))

decision = json.loads(DECISION_RECORD.read_text(encoding="utf-8"))
if decision.get("record_version") != "v03" or not decision.get("release_gate", {}).get("passed"):
    raise ValueError("Decision record v03 candidate-build gate is not passed")
eda_hash = sha256_file(EDA_RUN_DIR / "manifest.json")
if eda_hash != decision["source_manifest_sha256"]:
    raise ValueError("Frozen EDA manifest hash mismatch")
for filename, metadata in decision["governance_inputs"].items():
    path = AUDIT_ROOT / filename
    if not path.exists() or sha256_file(path) != metadata["sha256"]:
        raise ValueError(f"Governance hash mismatch: {filename}")
print("Pinned-input preflight: PASS")
print("Benchmark release gate:", decision["benchmark_release_gate"]["passed"])
"""
        ),
        code(
            """import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas>=2.2,<3", "numpy>=1.26,<3", "pyarrow>=16"])
"""
        ),
        code(
            f"""import base64
import gzip

RUNTIME_SRC = Path("/content/FinDisputeEval_runtime_src") if IN_COLAB else ROOT / ".runtime/seed_v051"
PACKAGE = RUNTIME_SRC / "findisputeeval/curation"
PACKAGE.mkdir(parents=True, exist_ok=True)
(RUNTIME_SRC / "findisputeeval/__init__.py").write_text("", encoding="utf-8")
(PACKAGE / "__init__.py").write_text("", encoding="utf-8")

embedded = {{
    "cfpb_seed_v05.py": ("{v05_hash}", "{v05_blob}"),
    "cfpb_seed_v051.py": ("{v051_hash}", "{v051_blob}"),
}}
for filename, (expected, blob) in embedded.items():
    raw = gzip.decompress(base64.b64decode(blob))
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError(f"Embedded source hash mismatch: {{filename}}")
    (PACKAGE / filename).write_bytes(raw)
sys.path.insert(0, str(RUNTIME_SRC))
print("Embedded builder source verified:", {{k: v[0] for k, v in embedded.items()}})
"""
        ),
        code(
            """from findisputeeval.curation.cfpb_seed_v051 import SeedV051Config, build_seed_v051

paths = build_seed_v051(SeedV051Config(
    eda_run_dir=EDA_RUN_DIR,
    decision_record_path=DECISION_RECORD,
    output_dir=OUTPUT_DIR,
    random_seed=20260713,
))
print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))
"""
        ),
        code(
            """manifest_path = OUTPUT_DIR / "seed_v051_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
errors = []
for name, metadata in manifest["outputs"].items():
    relative = Path(metadata["path"])
    if relative.is_absolute():
        errors.append(f"absolute manifest path: {name}")
        continue
    path = OUTPUT_DIR / relative
    if not path.exists() or sha256_file(path) != metadata["sha256"]:
        errors.append(f"output integrity failure: {name}")
if manifest.get("benchmark_eligible") is not False:
    errors.append("candidate was incorrectly marked benchmark eligible")
if errors:
    raise ValueError("\\n".join(errors))
print(json.dumps({
    "release": manifest["release"],
    "status": manifest["release_status"],
    "primary_rows": manifest["primary_rows"],
    "enrichment_rows": manifest["enrichment_rows"],
    "stress_rows": manifest["stress_rows"],
    "excluded_rows": manifest["excluded_rows"],
    "manifest_sha256": sha256_file(manifest_path),
    "output_integrity": "PASS",
    "benchmark_release_gate_passed": manifest["benchmark_release_gate"]["passed"],
}, indent=2))
"""
        ),
        markdown(
            "## Completion boundary\n\n"
            "This notebook completes the **candidate build** only. Do not run the formal 50–100 dialogue benchmark pilot until a full Seed + stress NER scan and manual review of all positives are recorded and the benchmark release gate is explicitly changed to passed.\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": OUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
