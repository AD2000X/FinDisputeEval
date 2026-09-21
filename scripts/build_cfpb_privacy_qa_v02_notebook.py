"""Build the Colab/VS Code formal CFPB privacy QA v02 notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/20_seed_curation/cfpb_dispute/canonical/FinDisputeEval_CFPB_Privacy_NER_QA_colab_v02.ipynb"


def source_lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


def embedded_module_cell() -> str:
    modules = {}
    for name in ("cfpb_privacy_qa.py", "cfpb_privacy_qa_v02.py"):
        raw = (ROOT / "src/findisputeeval/curation" / name).read_bytes()
        modules[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "payload": base64.b64encode(gzip.compress(raw)).decode("ascii"),
        }
    payload = json.dumps(modules, sort_keys=True)
    return f'''
RUNTIME = Path("/content/findisputeeval_privacy_v02") if IN_COLAB else ROOT / ".runtime/privacy_qa_v02"
PACKAGE = RUNTIME / "findisputeeval/curation"
PACKAGE.mkdir(parents=True, exist_ok=True)
(RUNTIME / "findisputeeval/__init__.py").write_text("", encoding="utf-8")
(PACKAGE / "__init__.py").write_text("", encoding="utf-8")
embedded = json.loads({json.dumps(payload)})
for name, item in embedded.items():
    raw = gzip.decompress(base64.b64decode(item["payload"]))
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("Embedded privacy module hash mismatch: " + name)
    (PACKAGE / name).write_bytes(raw)
sys.path.insert(0, str(RUNTIME))
for module_name in (
    "findisputeeval.curation.cfpb_privacy_qa_v02",
    "findisputeeval.curation.cfpb_privacy_qa",
):
    sys.modules.pop(module_name, None)
from findisputeeval.curation.cfpb_privacy_qa import HIGH_RISK_ENTITIES, build_presidio_analyzer, scan_frame, sha256_file
from findisputeeval.curation.cfpb_privacy_qa_v02 import (
    FORMAL_MODELS,
    build_challenge_set,
    build_negative_control_sample,
    evaluate_challenge_set,
    finalize_privacy_review_v02,
)
print({{name: item["sha256"] for name, item in embedded.items()}})
'''


cells = [
    markdown(
        """
# CFPB Seed v05.2 Formal Privacy NER QA v02

This notebook is the formal privacy release gate for **all 3,004 Seed v05.2
records plus all 200 stress records**. It does not replace the earlier smoke
review and writes to a new `full_v052_v02` workspace.

Release requires all three independent checks:

1. every Presidio/spaCy finding is manually adjudicated;
2. 200 deterministic, stratified zero-finding records are manually reviewed;
3. every required synthetic PII challenge passes.

The final clearance hash-binds the Seed v05.2 manifest, analyzer configuration,
exact package/model environment lock, finding review, negative-control review,
challenge set, and challenge results. The first session resolves and persists
exact versions; every later session recreates that same environment before
loading existing annotations.
`en_core_web_sm` is prohibited for this formal gate.
"""
    ),
    code(
        """
from pathlib import Path
import base64, gzip, hashlib, importlib.metadata, json, subprocess, sys

NER_MODEL = "en_core_web_trf"  # approved alternatives: en_core_web_trf or en_core_web_lg
MINIMUM_SCORE = 0.35
NEGATIVE_CONTROL_N = 200
RANDOM_SEED = 20260713
REVIEWER = "chang"

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/FinDisputeEval")
else:
    here = Path.cwd().resolve()
    ROOT = next((p for p in (here, *here.parents) if (p / "WORK_PROGRESS.md").exists()), None)
    if ROOT is None:
        raise FileNotFoundError("Open the FinDisputeEval repository in VS Code")
if NER_MODEL not in {"en_core_web_trf", "en_core_web_lg"}:
    raise ValueError("Formal privacy QA prohibits en_core_web_sm")
QA_ROOT = ROOT / "dataset/curated/annotations/cfpb_seed_v05_audit/run_20260713T145423Z/privacy_qa/full_v052_v02"
QA_ROOT.mkdir(parents=True, exist_ok=True)
ENV_LOCK = QA_ROOT / "environment_lock_v02.json"
"""
    ),
    code(
        """
MODEL_DISTRIBUTION = NER_MODEL.replace("_", "-")
BASE_RANGES = [
    "pandas>=2.2,<3", "pyarrow>=16", "spacy>=3.8,<4",
    "presidio-analyzer>=2.2,<3", "ipywidgets>=8,<9",
]

if ENV_LOCK.exists():
    environment_lock = json.loads(ENV_LOCK.read_text(encoding="utf-8"))
    if environment_lock.get("model_name") != NER_MODEL:
        raise ValueError("Existing environment lock uses a different spaCy model")
    exact_packages = [
        f"{name}=={version}"
        for name, version in environment_lock["packages"].items()
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *exact_packages])
    try:
        installed_model_version = importlib.metadata.version(MODEL_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        installed_model_version = None
    model_check = subprocess.run(
        [sys.executable, "-c", f"import spacy; spacy.load('{NER_MODEL}')"],
        check=False,
    )
    if installed_model_version != environment_lock["model_version"] or model_check.returncode != 0:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            environment_lock["model_wheel_url"],
        ])
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *BASE_RANGES])
    model_check = subprocess.run(
        [sys.executable, "-c", f"import spacy; spacy.load('{NER_MODEL}')"],
        check=False,
    )
    if model_check.returncode != 0:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", NER_MODEL])
    package_names = ["pandas", "pyarrow", "spacy", "presidio-analyzer", "ipywidgets"]
    for optional_name in ("spacy-transformers", "transformers"):
        try:
            importlib.metadata.version(optional_name)
            package_names.append(optional_name)
        except importlib.metadata.PackageNotFoundError:
            pass
    model_version = importlib.metadata.version(MODEL_DISTRIBUTION)
    environment_lock = {
        "lock_version": "v02",
        "model_name": NER_MODEL,
        "model_version": model_version,
        "model_wheel_url": (
            "https://github.com/explosion/spacy-models/releases/download/"
            f"{NER_MODEL}-{model_version}/{NER_MODEL}-{model_version}-py3-none-any.whl"
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in sorted(package_names)
        },
    }
    ENV_LOCK.write_text(
        json.dumps(environment_lock, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

# Verify that the recreated environment exactly matches the persistent lock.
environment_lock = json.loads(ENV_LOCK.read_text(encoding="utf-8"))
for name, expected_version in environment_lock["packages"].items():
    actual_version = importlib.metadata.version(name)
    if actual_version != expected_version:
        raise ValueError(f"Environment lock mismatch for {name}: {actual_version} != {expected_version}")
if importlib.metadata.version(MODEL_DISTRIBUTION) != environment_lock["model_version"]:
    raise ValueError("spaCy model version does not match environment lock")
import pandas as pd
import spacy
GPU_ENABLED = bool(spacy.prefer_gpu())
print({
    "model": NER_MODEL,
    "model_version": environment_lock["model_version"],
    "gpu_enabled": GPU_ENABLED,
    "environment_lock": str(ENV_LOCK),
})
"""
    ),
    code(embedded_module_cell()),
    code(
        """
RELEASE = ROOT / "dataset/curated/seed_pools/cfpb_dispute/seed_v052"
MANIFEST = RELEASE / "seed_v052_manifest.json"
SEED = RELEASE / "cfpb_seed_v052.parquet"
STRESS = RELEASE / "cfpb_seed_v052_stress_test.parquet"
for path in (MANIFEST, SEED, STRESS):
    if not path.exists():
        raise FileNotFoundError(path)

seed = pd.read_parquet(SEED)
stress = pd.read_parquet(STRESS).rename(columns={"stress_id": "seed_id"})
frame = pd.concat([seed, stress], ignore_index=True)
if len(frame) != 3204 or frame["seed_id"].nunique() != 3204:
    raise ValueError(f"Expected 3,204 unique release records; received {len(frame)}")

CONFIG = QA_ROOT / "analyzer_config_v02.json"
INVENTORY = QA_ROOT / "scan_inventory_v02.json"
FINDINGS = QA_ROOT / "presidio_spacy_findings_review_v02.csv"
NEGATIVES = QA_ROOT / "negative_control_review_v02.csv"
CHALLENGE_SET = QA_ROOT / "challenge_set_v02.csv"
CHALLENGE_RESULTS = QA_ROOT / "challenge_results_v02.csv"
CLEARANCE = QA_ROOT / "privacy_clearance_v02.json"

config = {
    "config_version": "v02",
    "source_manifest_sha256": sha256_file(MANIFEST),
    "scanned_records": len(frame),
    "model": NER_MODEL,
    "model_version": environment_lock["model_version"],
    "minimum_score": MINIMUM_SCORE,
    "entities": sorted(HIGH_RISK_ENTITIES),
    "negative_control_n": NEGATIVE_CONTROL_N,
    "random_seed": RANDOM_SEED,
    "environment_lock_sha256": sha256_file(ENV_LOCK),
    "locked_packages": environment_lock["packages"],
    "privacy_helper_sha256": {
        name: item["sha256"] for name, item in embedded.items()
    },
    "challenge_gate_policy": "any_high_risk_detection_overlapping_expected_sensitive_span",
    "spacy_version": importlib.metadata.version("spacy"),
    "presidio_analyzer_version": importlib.metadata.version("presidio-analyzer"),
}
config_text = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True)
if CONFIG.exists() and CONFIG.read_text(encoding="utf-8") != config_text:
    if FINDINGS.exists() or NEGATIVES.exists():
        raise ValueError("Existing annotated v02 workspace uses a different source or analyzer configuration")
    print("Replacing pre-scan analyzer config after a blocked challenge preflight")
CONFIG.write_text(config_text, encoding="utf-8")
CONFIG_SHA256 = sha256_file(CONFIG)
print({"rows": len(frame), "qa_root": str(QA_ROOT), "manifest_sha256": sha256_file(MANIFEST), "config_sha256": CONFIG_SHA256})
"""
    ),
    markdown(
        """
## A. Challenge set preflight

This runs before the corpus scan. Any failed required challenge blocks the scan
until the model/recognizer configuration is corrected. The values are synthetic
or reserved test values and are not intended to identify real people. A positive
challenge passes when any configured high-risk detector overlaps the expected
sensitive span; exact entity-type agreement is reported separately because a
mis-typed but surfaced span still reaches manual review.
"""
    ),
    code(
        """
analyzer = build_presidio_analyzer(NER_MODEL)
challenge_set = build_challenge_set()
if CHALLENGE_SET.exists():
    existing = pd.read_csv(CHALLENGE_SET, keep_default_na=False, encoding="utf-8-sig")
    if not existing.astype(str).equals(challenge_set.astype(str)):
        raise ValueError("Existing challenge set differs from the fixed v02 set")
else:
    challenge_set.to_csv(CHALLENGE_SET, index=False, encoding="utf-8-sig")
challenge_results = evaluate_challenge_set(challenge_set, analyzer, minimum_score=MINIMUM_SCORE)
challenge_results.to_csv(CHALLENGE_RESULTS, index=False, encoding="utf-8-sig")
display(challenge_results[[
    "challenge_id",
    "expected_entity_type",
    "detection_overlap_passed",
    "expected_type_matched",
    "challenge_passed",
    "detections_json",
]])
failed = challenge_results.loc[challenge_results["required"] & ~challenge_results["challenge_passed"]]
if not failed.empty:
    raise RuntimeError(f"BLOCKED: {len(failed)} required privacy challenges failed")
print("All required privacy challenges passed")
type_mismatches = challenge_results.loc[
    challenge_results["required"]
    & challenge_results["expected_entity_type"].ne("")
    & ~challenge_results["expected_type_matched"]
]
if not type_mismatches.empty:
    print(
        "Diagnostic only — sensitive spans surfaced under a different high-risk type:",
        type_mismatches["challenge_id"].tolist(),
    )
"""
    ),
    markdown(
        """
## B. Full release scan and stratified negative controls

The scan and initial samples are created once. Existing human annotations are
never overwritten. Reuse is allowed only when both the manifest and analyzer
configuration hashes match the inventory.
"""
    ),
    code(
        """
expected_inventory = {
    "inventory_version": "v02",
    "source_manifest_sha256": sha256_file(MANIFEST),
    "analyzer_config_sha256": CONFIG_SHA256,
    "scanned_records": len(frame),
}
if INVENTORY.exists():
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    for key, value in expected_inventory.items():
        if inventory.get(key) != value:
            raise ValueError(f"Existing scan inventory mismatch: {key}")

if FINDINGS.exists():
    findings_review = pd.read_csv(FINDINGS, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "release_record_id" not in findings_review and "record_id" in findings_review:
        findings_review = findings_review.rename(columns={"record_id": "release_record_id"})
    if "source_record_id" not in findings_review:
        source_ids = frame.set_index(frame["seed_id"].astype(str))["record_id"].astype(str)
        findings_review.insert(
            findings_review.columns.get_loc("release_record_id") + 1,
            "source_record_id",
            findings_review["release_record_id"].map(source_ids),
        )
    findings_review.to_csv(FINDINGS, index=False, encoding="utf-8-sig")
    print("Existing finding review loaded:", len(findings_review))
else:
    findings_review = scan_frame(
        frame,
        analyzer,
        id_column="seed_id",
        text_column="seed_text",
        split_column="release_split",
        minimum_score=MINIMUM_SCORE,
    )
    findings_review = findings_review.rename(columns={"record_id": "release_record_id"})
    source_ids = frame.set_index(frame["seed_id"].astype(str))["record_id"].astype(str)
    findings_review.insert(
        findings_review.columns.get_loc("release_record_id") + 1,
        "source_record_id",
        findings_review["release_record_id"].map(source_ids),
    )
    findings_review.to_csv(FINDINGS, index=False, encoding="utf-8-sig")
    print("NER findings created:", len(findings_review))

if NEGATIVES.exists():
    negative_review = pd.read_csv(NEGATIVES, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "release_record_id" not in negative_review and "record_id" in negative_review:
        negative_review = negative_review.rename(columns={"record_id": "release_record_id"})
    if "source_record_id" not in negative_review:
        source_ids = frame.set_index(frame["seed_id"].astype(str))["record_id"].astype(str)
        negative_review.insert(
            negative_review.columns.get_loc("release_record_id") + 1,
            "source_record_id",
            negative_review["release_record_id"].map(source_ids),
        )
    negative_review.to_csv(NEGATIVES, index=False, encoding="utf-8-sig")
    print("Existing negative controls loaded:", len(negative_review))
else:
    negative_review = build_negative_control_sample(
        frame,
        findings_review,
        id_column="seed_id",
        text_column="seed_text",
        split_column="release_split",
        sample_size=NEGATIVE_CONTROL_N,
        random_seed=RANDOM_SEED,
    )
    negative_review.to_csv(NEGATIVES, index=False, encoding="utf-8-sig")
    print("Negative controls created:", len(negative_review))

inventory = {
    **expected_inventory,
    "finding_rows_at_scan": len(findings_review),
    "negative_control_rows": len(negative_review),
    "challenge_rows": len(challenge_results),
}
INVENTORY.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
print({
    "finding_pending": int(findings_review.manual_pii_present.eq("pending").sum()),
    "negative_pending": int(negative_review.manual_pii_present.eq("pending").sum()),
    "negative_strata": int(negative_review.selection_stratum.nunique()),
    "negative_splits": negative_review.release_split.value_counts().to_dict(),
})
"""
    ),
    markdown(
        """
## C. Manual review dashboards

Review **every** pending finding and every selected zero-finding record. A `yes`
requires `exclude` or `redact_and_rescan`; a `no` is saved as `allow`.
Progress is persisted to Drive after each decision.
"""
    ),
    code(
        """
import ipywidgets as widgets
from IPython.display import display

def review_dashboard(frame, path, *, title, text_column, detail_columns):
    state = {"frame": frame, "position": 0}
    header = widgets.HTML()
    details = widgets.HTML()
    text = widgets.Textarea(disabled=True, layout=widgets.Layout(width="98%", height="320px"))
    decision = widgets.ToggleButtons(
        options=[("Choose…", ""), ("No — not PII", "no"), ("Yes — actual PII", "yes")],
        description="Decision",
    )
    action = widgets.Dropdown(
        options=[("Choose…", ""), ("Exclude", "exclude"), ("Redact and rescan", "redact_and_rescan")],
        description="If yes",
    )
    note = widgets.Textarea(description="Note", layout=widgets.Layout(width="90%", height="70px"))
    status = widgets.HTML()
    save = widgets.Button(description="Save & next", button_style="success", icon="save")
    reload_button = widgets.Button(description="Reload", icon="refresh")

    def pending_indices():
        return list(state["frame"].index[state["frame"].manual_pii_present.eq("pending")])

    def render():
        pending = pending_indices()
        header.value = f"<h4>{title}: {len(pending)} pending / {len(state['frame'])} total</h4>"
        if not pending:
            details.value = "<b>Complete.</b>"
            text.value = ""
            save.disabled = True
            return
        index = pending[0]
        row = state["frame"].loc[index]
        detail = " | ".join(f"{column}: {row.get(column, '')}" for column in detail_columns)
        details.value = f"<b>Row {index}</b> — {detail}"
        text.value = str(row[text_column])
        decision.value = ""
        action.value = ""
        note.value = ""
        save.disabled = False
        status.value = ""

    def save_next(_):
        pending = pending_indices()
        if not pending or decision.value not in {"yes", "no"}:
            status.value = "<span style='color:red'>Choose yes or no.</span>"
            return
        if decision.value == "yes" and action.value not in {"exclude", "redact_and_rescan"}:
            status.value = "<span style='color:red'>A positive requires an action.</span>"
            return
        index = pending[0]
        state["frame"].at[index, "manual_pii_present"] = decision.value
        state["frame"].at[index, "release_action"] = "allow" if decision.value == "no" else action.value
        state["frame"].at[index, "reviewer"] = REVIEWER
        state["frame"].at[index, "reviewed_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        state["frame"].at[index, "notes"] = note.value.strip()
        state["frame"].to_csv(path, index=False, encoding="utf-8-sig")
        render()

    def reload_review(_):
        state["frame"] = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        render()

    save.on_click(save_next)
    reload_button.on_click(reload_review)
    render()
    display(widgets.VBox([header, details, text, decision, action, note, widgets.HBox([save, reload_button]), status]))
    return state

finding_state = review_dashboard(
    findings_review,
    FINDINGS,
    title="NER finding review",
    text_column="context",
    detail_columns=["release_record_id", "source_record_id", "release_split", "entity_type", "score", "detected_text"],
)
"""
    ),
    code(
        """
negative_state = review_dashboard(
    negative_review,
    NEGATIVES,
    title="Zero-finding negative-control review",
    text_column="review_text",
    detail_columns=["release_record_id", "source_record_id", "release_split", "register_candidate", "lid_status", "length_bucket"],
)
"""
    ),
    markdown(
        """
## D. Finalize hash-bound clearance

Run only after both dashboards show zero pending rows. Any confirmed PII or
failed challenge produces `release_clearance_passed: false` and requires a
versioned exclusion/redaction, rebuild, and rescan.
"""
    ),
    code(
        """
findings_review = pd.read_csv(FINDINGS, dtype=str, keep_default_na=False, encoding="utf-8-sig")
negative_review = pd.read_csv(NEGATIVES, dtype=str, keep_default_na=False, encoding="utf-8-sig")
challenge_results = pd.read_csv(CHALLENGE_RESULTS, keep_default_na=False, encoding="utf-8-sig")
print({
    "finding_pending": int(findings_review.manual_pii_present.eq("pending").sum()),
    "negative_pending": int(negative_review.manual_pii_present.eq("pending").sum()),
})

result = finalize_privacy_review_v02(
    findings_review,
    negative_review,
    challenge_results,
    scope="full_v052",
    source_sha256=sha256_file(MANIFEST),
    analyzer_model=NER_MODEL,
    analyzer_config_sha256=CONFIG_SHA256,
    scanned_records=len(frame),
    evidence_paths={
        "analyzer_config": CONFIG,
        "environment_lock": ENV_LOCK,
        "findings_review": FINDINGS,
        "negative_controls": NEGATIVES,
        "challenge_set": CHALLENGE_SET,
        "challenge_results": CHALLENGE_RESULTS,
    },
    minimum_negative_controls=NEGATIVE_CONTROL_N,
)
CLEARANCE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["release_clearance_passed"]:
    print("BLOCKED: resolve PII/challenge failures, rebuild if release text changes, then rescan.")
else:
    print("Privacy v02 passed. Download the full full_v052_v02 directory and run validator v02 locally.")
"""
    ),
    markdown(
        """
## Local final gate

After downloading the complete `privacy_qa/full_v052_v02/` directory, run:

```powershell
python scripts/validate_cfpb_seed_v052_release_v02.py
```

The formal benchmark pilot may begin only when both
`structural_release_validator_passed` and `benchmark_release_gate_passed` are
`true` in `seed_v052_release_qa_v02.json`.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
