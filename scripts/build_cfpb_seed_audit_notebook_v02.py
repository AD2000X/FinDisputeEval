"""Build the self-contained Colab/VS Code CFPB Seed v05 audit notebook."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/findisputeeval/curation/cfpb_seed_audit_v02.py"
NOTEBOOK = ROOT / (
    "notebooks/20_seed_curation/cfpb_dispute/canonical/"
    "FinDisputeEval_CFPB_SeedSource_Audit_and_Decision_v01.ipynb"
)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


module_bytes = MODULE.read_bytes()
module_b64 = base64.b64encode(module_bytes).decode("ascii")
module_sha256 = hashlib.sha256(module_bytes).hexdigest()

cells = [
    markdown(
        """# FinDisputeEval CFPB Seed Source Audit and Decision v02

This notebook treats the EDA v05.1 run as an immutable parent. On the first run it copies each complete source audit CSV byte-for-byte into `source_snapshots/`, then creates separate annotation masters. Every later run reads and writes only those workspace masters, A/B assignments, adjudication tables, and decision records.

The historical filename is retained so existing links do not break; the workflow and emitted record are version `v02`.
"""
    ),
    markdown("## 0. Locate the frozen EDA run and persistent annotation workspace\n"),
    code(
        """from pathlib import Path
import hashlib
import json
import os
import sys

IN_COLAB = "google.colab" in sys.modules
RUN_ID = "run_20260713T145423Z"

# Optional explicit paths. Leave blank for automatic discovery.
EDA_RUN_OVERRIDE = ""
AUDIT_ROOT_OVERRIDE = ""

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_ROOT = Path("/content/FinDisputeEval")
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    DRIVE_ROOT = Path("/content/drive/MyDrive/FinDisputeEval")
else:
    here = Path.cwd().resolve()
    candidates = [here, *here.parents]
    PROJECT_ROOT = next(
        (candidate for candidate in candidates if (candidate / "WORK_PROGRESS.md").exists()),
        None,
    )
    if PROJECT_ROOT is None:
        raise FileNotFoundError("Open the FinDisputeEval repository in VS Code.")
    DRIVE_ROOT = None

def is_v051_run(path):
    manifest_path = Path(path) / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("version") == "v05.1"
        and manifest.get("source_design", {}).get("universe_unique_ids") == 205_589
    )


if EDA_RUN_OVERRIDE:
    EDA_RUN_DIR = Path(EDA_RUN_OVERRIDE).expanduser().resolve()
    if not is_v051_run(EDA_RUN_DIR):
        raise FileNotFoundError(
            f"EDA_RUN_OVERRIDE is not a valid v05.1 run directory: {EDA_RUN_DIR}"
        )
else:
    # Includes the current canonical path and the two legacy locations used by
    # earlier notebook versions. rglob also handles an extra ZIP extraction layer.
    search_roots = []
    if DRIVE_ROOT is not None:
        search_roots.extend(
            [
                DRIVE_ROOT / "outputs/data_pipeline/cfpb_seed_source_eda/eda_v051",
                DRIVE_ROOT / "outputs/cfpb_seed_source_eda/eda_v051",
                DRIVE_ROOT / "outputs/data_pipeline/cfpb_linguistic_eda/eda_v051",
                DRIVE_ROOT / "outputs/cfpb_linguistic_eda/eda_v051",
            ]
        )
    search_roots.extend(
        [
            PROJECT_ROOT / "outputs/data_pipeline/cfpb_seed_source_eda/eda_v051",
            PROJECT_ROOT / "outputs/cfpb_seed_source_eda/eda_v051",
            PROJECT_ROOT / "outputs/data_pipeline/cfpb_linguistic_eda/eda_v051",
        ]
    )

    discovered = []
    for root in search_roots:
        if not root.exists():
            continue
        for manifest_path in root.rglob("manifest.json"):
            candidate = manifest_path.parent
            if is_v051_run(candidate):
                discovered.append(candidate)

    # If the project folder was moved or renamed in Drive, search for the exact
    # run ID under MyDrive/Shared drives only after the fast paths fail.
    if not discovered and IN_COLAB:
        broad_roots = [
            Path("/content/drive/MyDrive"),
            Path("/content/drive/Shareddrives"),
        ]
        for root in broad_roots:
            if not root.exists():
                continue
            for candidate in root.glob(f"**/{RUN_ID}"):
                if candidate.is_dir() and is_v051_run(candidate):
                    discovered.append(candidate)

    discovered = sorted(set(discovered), key=lambda path: str(path))
    exact = [path for path in discovered if path.name == RUN_ID]
    if exact:
        EDA_RUN_DIR = exact[-1]
    elif discovered:
        EDA_RUN_DIR = discovered[-1]
    else:
        searched = "\\n  - ".join(str(path) for path in search_roots)
        raise FileNotFoundError(
            "No hash-verifiable EDA v05.1 run was found.\\n"
            f"Searched:\\n  - {searched}\\n"
            "If the folder is only under 'Shared with me', add a shortcut to My Drive, "
            "or set EDA_RUN_OVERRIDE to the directory that directly contains manifest.json."
        )

if AUDIT_ROOT_OVERRIDE:
    AUDIT_ROOT = Path(AUDIT_ROOT_OVERRIDE).expanduser().resolve()
elif DRIVE_ROOT is not None:
    AUDIT_ROOT = (
        DRIVE_ROOT
        / "dataset/curated/annotations/cfpb_seed_v05_audit"
        / EDA_RUN_DIR.name
    )
else:
    AUDIT_ROOT = (
        PROJECT_ROOT
        / "dataset/curated/annotations/cfpb_seed_v05_audit"
        / EDA_RUN_DIR.name
    )

AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Runtime: {'Colab' if IN_COLAB else 'local VS Code'}")
print(f"Frozen EDA run: {EDA_RUN_DIR}")
print(f"Annotation workspace: {AUDIT_ROOT}")
"""
    ),
    markdown("## 1. Install the byte-identical audit workflow helper\n"),
    code(
        f'''import base64

MODULE_SHA256 = "{module_sha256}"
MODULE_B64 = "{module_b64}"
module_path = PROJECT_ROOT / "src/findisputeeval/curation/cfpb_seed_audit_v02.py"
module_path.parent.mkdir(parents=True, exist_ok=True)
for init_path in [module_path.parent.parent / "__init__.py", module_path.parent / "__init__.py"]:
    init_path.touch(exist_ok=True)
payload = base64.b64decode(MODULE_B64)
if hashlib.sha256(payload).hexdigest() != MODULE_SHA256:
    raise RuntimeError("Embedded audit helper checksum mismatch")
if not module_path.exists() or hashlib.sha256(module_path.read_bytes()).hexdigest() != MODULE_SHA256:
    module_path.write_bytes(payload)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
print(f"Audit helper ready: {{module_path}}")
print(f"SHA-256: {{MODULE_SHA256}}")
'''
    ),
    code(
        """from findisputeeval.curation.cfpb_seed_audit_v02 import (
    AuditConfig,
    MASTER_FILENAMES,
    REGISTER_LABELS,
    YES_NO_UNCERTAIN,
    agreement_metrics,
    build_completion_table,
    build_release_record,
    create_double_assignment,
    initialize_decision_table,
    initialize_masters,
    linguistic_decision_state,
    merge_adjudication_into_master,
    read_audit_csv,
    snapshot_source_audits,
    sync_adjudication,
    verify_eda_run,
    write_workspace_manifest,
)

config = AuditConfig(eda_run_dir=EDA_RUN_DIR, audit_root=AUDIT_ROOT)
"""
    ),
    markdown("## 2. Verify EDA immutability, snapshot source audits, and initialize masters\n"),
    code(
        """# verify_eda_run checks the frozen manifest, every listed output size, and every SHA-256.
eda_manifest = verify_eda_run(config)
source_snapshots = snapshot_source_audits(config, eda_manifest)
masters = initialize_masters(config, source_snapshots)
decisions = initialize_decision_table(config)
workspace_manifest_path = write_workspace_manifest(
    config, eda_manifest, source_snapshots, masters
)

print("EDA verification: PASS (no EDA output was written)")
print(f"Workspace manifest: {workspace_manifest_path}")
display(
    __import__("pandas").DataFrame(
        [{"audit": name, "master_rows": len(frame), "file": MASTER_FILENAMES[name]}
         for name, frame in masters.items()]
    )
)
"""
    ),
    markdown("## 3. Inspect the redesigned template-family and PII workloads\n"),
    code(
        """import pandas as pd

template_summary = (
    masters["template_family_audit"]
    .groupby(["audit_selection", "family_size_band"], dropna=False)
    .size().rename("rows").reset_index()
)
pii_summary = (
    masters["pii_presidio_audit"]
    .groupby(["audit_selection", "primary_entity_type", "presidio_score_band"], dropna=False)
    .size().rename("rows").reset_index()
)
pii_source_rows = len(read_audit_csv(source_snapshots["pii_presidio_audit"]))
print(
    f"PII review workload: {len(masters['pii_presidio_audit']):,} stratified/census rows "
    f"instead of {pii_source_rows:,} source candidates."
)
display(template_summary)
display(pii_summary)
"""
    ),
    markdown(
        """## 4. Decision-dependent linguistic audit

Edit `seed_v05_decision_table.csv` in the annotation workspace. Linguistic audit has three explicit states:

- `pending_decision`: the decision is blank, invalid, or not accepted; release is blocked.
- `not_required`: an accepted machine rule explicitly contains `{"sampling_features": []}`.
- `required`: an accepted machine rule names one or more known features; only those features are release-required.

Every accepted decision also requires a non-empty rule/rationale, valid JSON, `decided_by`, and `decided_utc`.
"""
    ),
    code(
        """# Re-read because the table may have been edited outside the notebook.
decisions = initialize_decision_table(config)
known_features = set(masters["linguistic_pattern_audit"]["feature"].astype(str))
linguistic_state, sampling_features, linguistic_errors = linguistic_decision_state(
    decisions, known_features
)
print("linguistic_state:", linguistic_state)
print("sampling_features:", sampling_features)
print("errors:", linguistic_errors)
display(decisions)
"""
    ),
    markdown(
        """## 5. Create A/B assignments, adjudicate disagreements, then merge to masters

On first execution, deterministic A and B assignments are created. Annotators independently fill only their own assignment file. Re-run this section to create/update each adjudication table. Agreements merge automatically; disagreements merge only after `adjudicated_value`, `adjudicator_id`, and (when needed) `adjudication_notes` are completed.

Do not annotate double-required rows directly in the master. Annotate all remaining rows in the master files. Existing assignment membership is hash-stable and cannot silently change.
"""
    ),
    code(
        """double_specs = [
    {
        "audit": "template_family_audit",
        "label": "manual_same_template",
        "allowed": YES_NO_UNCERTAIN,
        "target": config.template_double_n,
        "strata": ["audit_selection", "family_size_band", "representative_sampling_frame"],
        "frame": masters["template_family_audit"],
    },
    {
        "audit": "register_audit",
        "label": "manual_register",
        "allowed": REGISTER_LABELS,
        "target": config.register_double_n,
        "strata": ["sampling_frame", "register_candidate"],
        "frame": masters["register_audit"],
    },
]

if linguistic_state == "required":
    linguistic_required = masters["linguistic_pattern_audit"].loc[
        masters["linguistic_pattern_audit"]["feature"].isin(sampling_features)
    ].copy()
    double_specs.append(
        {
            "audit": "linguistic_pattern_audit",
            "label": "manual_present",
            "allowed": YES_NO_UNCERTAIN,
            "target": min(config.linguistic_double_n, len(linguistic_required)),
            "strata": ["feature", "detector_output", "sampling_frame"],
            "frame": linguistic_required,
        }
    )

agreement = []
merge_results = []
for spec in double_specs:
    create_double_assignment(
        config,
        spec["audit"],
        spec["frame"],
        spec["label"],
        spec["target"],
        spec["strata"],
    )
    sync_adjudication(config, spec["audit"], spec["label"])
    merge_result = merge_adjudication_into_master(
        config, spec["audit"], spec["label"], spec["allowed"]
    )
    merge_results.append({"audit": spec["audit"], **merge_result})
    agreement.append(
        agreement_metrics(config, spec["audit"], spec["label"], spec["allowed"])
    )

display(pd.DataFrame(merge_results))
display(pd.DataFrame(agreement))
"""
    ),
    markdown(
        """## 6. Allowed-label validation and release gate

Allowed labels are enforced, not merely checked for non-blank values. `uncertain`, `unknown`, and `other` require notes. PII rows labeled `yes` require `manual_pii_types`. The release record remains blocked while any annotation, adjudication, decision, or machine rule is pending/invalid.
"""
    ),
    code(
        """# Reload masters after any consensus/adjudication merge.
masters = initialize_masters(config, source_snapshots)
completion = build_completion_table(masters, linguistic_state, sampling_features)
workspace_manifest_path = write_workspace_manifest(
    config, eda_manifest, source_snapshots, masters
)
record, decision_record_path = build_release_record(
    config=config,
    decisions=decisions,
    completion=completion,
    agreement=agreement,
    linguistic_state=linguistic_state,
    sampling_features=sampling_features,
    linguistic_errors=linguistic_errors,
    masters=masters,
)

display(completion)
print(f"Decision record: {decision_record_path}")
print("Release gate passed:", record["release_gate"]["passed"])
if not record["release_gate"]["passed"]:
    print("Gate detail:")
    print(json.dumps(record["release_gate"], ensure_ascii=False, indent=2))

# A final full verification proves the notebook did not alter the EDA parent.
verify_eda_run(config)
print("Final EDA hash verification: PASS")
"""
    ),
    markdown(
        """## Human workflow and file ownership

1. Keep `source_snapshots/*.csv` unchanged; they are byte-identical evidence copied from EDA.
2. Annotate non-double rows in `*_master.csv`.
3. Annotators A and B fill `*_double_annotator_A.csv` and `*_double_annotator_B.csv` independently.
4. Resolve disagreements in `*_adjudication.csv`, then re-run sections 5–6.
5. Populate and accept every row in `seed_v05_decision_table.csv` using measured audit evidence.
6. Generate Seed v05 only after `seed_v05_decision_record_v02.json` reports `release_gate.passed = true`.

The EDA run directory is read-only by design; all mutable human work belongs under the annotation workspace.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)
print(NOTEBOOK)
print(f"Embedded helper SHA-256: {module_sha256}")
