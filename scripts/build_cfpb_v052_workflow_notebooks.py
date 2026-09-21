"""Generate the interactive v05.2 fuzzy, privacy, and build notebooks."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks/20_seed_curation/cfpb_dispute/canonical"


def cell(kind: str, source: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


def notebook(name: str, cells: list[dict]) -> None:
    payload = {
        "cells": cells,
        "metadata": {
            "colab": {"name": name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOK_ROOT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def encoded(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), base64.b64encode(gzip.compress(raw)).decode("ascii")


def fuzzy_notebook() -> None:
    name = "FinDisputeEval_CFPB_FuzzyCluster_Audit_and_Decision_v04.ipynb"
    cells = [
        cell("markdown", "# CFPB Fuzzy Cluster C-lite Audit and Decision v04\n\nReview the 62 prioritized bridge edges first, then decide the six large components using representative texts and the completed bridge decisions. Progress is saved after every answer. `yes`, `no`, and `uncertain` are allowed; pending rows keep the v05.2 build gate closed. Components above 50 nodes require an explicit approval phrase and a written rationale.\n"),
        cell("code", '''from pathlib import Path
import hashlib, json, subprocess, sys
import pandas as pd
try:
    import ipywidgets as widgets
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ipywidgets>=8,<9"])
    import ipywidgets as widgets
from IPython.display import display

IN_COLAB = "google.colab" in sys.modules
RUN_ID = "run_20260713T145423Z"
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/FinDisputeEval")
else:
    here = Path.cwd().resolve()
    ROOT = next((p for p in (here, *here.parents) if (p / "WORK_PROGRESS.md").exists()), None)
    if ROOT is None: raise FileNotFoundError("Open the FinDisputeEval repository")
AUDIT_ROOT = ROOT / "dataset/curated/annotations/cfpb_seed_v05_audit" / RUN_ID
CLUSTERS = AUDIT_ROOT / "fuzzy_cluster_review_v04.csv"
BRIDGES = AUDIT_ROOT / "fuzzy_bridge_review_v04.csv"
EVIDENCE = AUDIT_ROOT / "fuzzy_cluster_evidence_v04.csv"
DRAFT = AUDIT_ROOT / "seed_v052_decision_record_v04_draft.json"
FINAL = AUDIT_ROOT / "seed_v052_decision_record_v04.json"
for path in [CLUSTERS, BRIDGES, EVIDENCE, DRAFT]:
    if not path.exists(): raise FileNotFoundError(path)
print(AUDIT_ROOT)
'''),
        cell("code", '''clusters = pd.read_csv(CLUSTERS, dtype=str, keep_default_na=False, encoding="utf-8-sig")
bridges = pd.read_csv(BRIDGES, dtype=str, keep_default_na=False, encoding="utf-8-sig")
evidence = pd.read_csv(EVIDENCE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
display(clusters.loc[clusters.review_required.str.lower().eq("true")])
print("Pending clusters:", clusters.manual_cluster_valid.eq("pending").sum())
print("Pending bridge edges:", bridges.manual_same_template.eq("pending").sum())
print("Representative evidence rows:", len(evidence))
'''),
        cell("code", '''# STEP 1 — Visible bridge-review dashboard (no blocking input popup).
bridges = pd.read_csv(BRIDGES, dtype=str, keep_default_na=False, encoding="utf-8-sig")
bridge_title = widgets.HTML()
bridge_text_a = widgets.Textarea(description="Text A", disabled=True, layout=widgets.Layout(width="49%", height="320px"))
bridge_text_b = widgets.Textarea(description="Text B", disabled=True, layout=widgets.Layout(width="49%", height="320px"))
bridge_label = widgets.ToggleButtons(options=[("Choose…", ""), ("Yes — same template", "yes"), ("No — different", "no"), ("Uncertain", "uncertain")], value="", description="Decision")
bridge_notes = widgets.Text(description="Note", placeholder="Optional rationale", layout=widgets.Layout(width="75%"))
bridge_save = widgets.Button(description="Save & next", button_style="success", icon="save")
bridge_reload = widgets.Button(description="Reload", icon="refresh")
bridge_status = widgets.HTML()
bridge_state = {"index": None}

def show_next_bridge(*_):
    global bridges
    bridges = pd.read_csv(BRIDGES, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    pending = list(bridges.index[bridges.manual_same_template.str.lower().eq("pending")])
    completed = len(bridges) - len(pending)
    if not pending:
        bridge_state["index"] = None
        bridge_title.value = f"<h3>Bridge review complete: {completed}/{len(bridges)}</h3>"
        bridge_text_a.value = ""
        bridge_text_b.value = ""
        bridge_save.disabled = True
        bridge_status.value = "<b>Proceed to STEP 2.</b>"
        return
    index = pending[0]
    row = bridges.loc[index]
    bridge_state["index"] = index
    bridge_title.value = (
        f"<h3>Bridge {completed + 1}/{len(bridges)}</h3>"
        f"<b>Component:</b> {row.component_id} &nbsp; <b>size:</b> {row.component_node_count} &nbsp; "
        f"<b>rank:</b> {row.bridge_review_rank}<br>"
        f"<b>Jaccard:</b> {row.shingle_jaccard} &nbsp; <b>token-set:</b> {row.rapidfuzz_token_set_ratio}<br>"
        "<i>yes only when both texts share a reusable wording/structure template; same topic alone is no.</i>"
    )
    bridge_text_a.value = str(row.text_a)
    bridge_text_b.value = str(row.text_b)
    bridge_label.value = ""
    bridge_notes.value = ""
    bridge_save.disabled = False
    bridge_status.value = f"Pending: {len(pending)}"

def save_bridge(_):
    index = bridge_state["index"]
    if index is None: return
    if bridge_label.value not in {"yes", "no", "uncertain"}:
        bridge_status.value = "<span style='color:#b00'><b>Select a decision first.</b></span>"
        return
    bridges.at[index, "manual_same_template"] = bridge_label.value
    bridges.at[index, "reviewer"] = "chang"
    bridges.at[index, "reviewed_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    bridges.at[index, "notes"] = bridge_notes.value.strip()
    bridges.to_csv(BRIDGES, index=False, encoding="utf-8-sig")
    show_next_bridge()

bridge_save.on_click(save_bridge)
bridge_reload.on_click(show_next_bridge)
display(bridge_title, widgets.HBox([bridge_text_a, bridge_text_b]), bridge_label, bridge_notes, widgets.HBox([bridge_save, bridge_reload]), bridge_status)
show_next_bridge()
'''),
        cell("code", '''# STEP 2 — Visible cluster-review dashboard. Run only after STEP 1 is complete.
bridges = pd.read_csv(BRIDGES, dtype=str, keep_default_na=False, encoding="utf-8-sig")
if bridges.manual_same_template.str.lower().eq("pending").any():
    raise ValueError(f"Complete bridge review first; pending={bridges.manual_same_template.str.lower().eq('pending').sum()}")
clusters = pd.read_csv(CLUSTERS, dtype=str, keep_default_na=False, encoding="utf-8-sig")
evidence = pd.read_csv(EVIDENCE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
cluster_title = widgets.HTML()
cluster_summary = widgets.HTML()
cluster_samples = widgets.Accordion(children=[])
cluster_label = widgets.ToggleButtons(options=[("Choose…", ""), ("Yes — approve cap", "yes"), ("No — bypass", "no"), ("Uncertain — bypass", "uncertain")], value="", description="Decision")
cluster_notes = widgets.Textarea(description="Rationale", placeholder="Required for oversized yes; recommended for every decision", layout=widgets.Layout(width="95%", height="90px"))
cluster_confirm = widgets.Text(description="Approval", placeholder="Shown phrase is required only for >50-node yes", layout=widgets.Layout(width="95%"))
cluster_save = widgets.Button(description="Save & next", button_style="success", icon="save")
cluster_reload = widgets.Button(description="Reload", icon="refresh")
cluster_status = widgets.HTML()
cluster_state = {"index": None, "phrase": ""}

def show_next_cluster(*_):
    global clusters
    clusters = pd.read_csv(CLUSTERS, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = clusters.review_required.str.lower().eq("true")
    pending = list(clusters.index[required & clusters.manual_cluster_valid.str.lower().eq("pending")])
    total = int(required.sum())
    if not pending:
        cluster_state["index"] = None
        cluster_title.value = f"<h3>Cluster review complete: {total}/{total}</h3>"
        cluster_summary.value = "<b>Proceed to finalization.</b>"
        cluster_samples.children = []
        cluster_save.disabled = True
        return
    index = pending[0]
    row = clusters.loc[index]
    cid, node_count = row.component_id, int(row.node_count)
    cluster_state["index"] = index
    cluster_state["phrase"] = f"APPROVE_{cid}_SIZE_{node_count}" if node_count > 50 else ""
    component_bridges = bridges.loc[bridges.component_id.eq(cid)]
    samples = evidence.loc[evidence.component_id.eq(cid)].copy()
    samples["rank_num"] = pd.to_numeric(samples.evidence_rank, errors="coerce")
    samples = samples.sort_values("rank_num", kind="mergesort")
    completed = total - len(pending)
    cluster_title.value = f"<h3>Cluster {completed + 1}/{total}: {cid}</h3><b>nodes:</b> {node_count} &nbsp; <b>edges:</b> {row.edge_count} &nbsp; <b>all graph bridges:</b> {row.bridge_count}"
    first = samples.iloc[0]
    phrase_text = f"<br><b>Oversized yes requires:</b> <code>{cluster_state['phrase']}</code> and a rationale." if cluster_state["phrase"] else ""
    cluster_summary.value = (
        f"<b>Reviewed bridge labels:</b> {component_bridges.manual_same_template.str.lower().value_counts().to_dict()} "
        f"({len(component_bridges)}/{row.bridge_count} graph bridges reviewed)<br>"
        f"<b>Products:</b> {first.component_product_counts_json}<br>"
        f"<b>Issues:</b> {first.component_issue_counts_json}<br>"
        f"<b>Registers:</b> {first.component_register_counts_json}<br>"
        "<i>yes caps rebuilt components after non-yes reviewed bridges are removed; no/uncertain bypasses this original component.</i>"
        + phrase_text
    )
    children = []
    for _, sample in samples.iterrows():
        value = (
            f"Reasons: {sample.selection_reasons}\\nDegree: {sample.graph_degree}; words: {sample.word_count}\\n"
            f"Metadata: {sample.Product} | {sample.Issue} | {sample.register_candidate}\\n\\n{sample.narrative_canonical}"
        )
        children.append(widgets.Textarea(value=value, disabled=True, layout=widgets.Layout(width="100%", height="330px")))
    cluster_samples.children = children
    for position, (_, sample) in enumerate(samples.iterrows()):
        cluster_samples.set_title(position, f"Rep {sample.evidence_rank}: {sample.complaint_id} [{sample.selection_reasons}]")
    cluster_label.value = ""
    cluster_notes.value = ""
    cluster_confirm.value = ""
    cluster_save.disabled = False
    cluster_status.value = f"Pending clusters: {len(pending)}"

def save_cluster(_):
    index = cluster_state["index"]
    if index is None: return
    label = cluster_label.value
    if label not in {"yes", "no", "uncertain"}:
        cluster_status.value = "<span style='color:#b00'><b>Select a decision first.</b></span>"
        return
    if cluster_state["phrase"] and label == "yes":
        if cluster_confirm.value.strip() != cluster_state["phrase"] or not cluster_notes.value.strip():
            cluster_status.value = "<span style='color:#b00'><b>Oversized yes requires the exact approval phrase and a rationale.</b></span>"
            return
    clusters.at[index, "manual_cluster_valid"] = label
    clusters.at[index, "reviewer"] = "chang"
    clusters.at[index, "reviewed_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    clusters.at[index, "notes"] = cluster_notes.value.strip()
    clusters.to_csv(CLUSTERS, index=False, encoding="utf-8-sig")
    show_next_cluster()

cluster_save.on_click(save_cluster)
cluster_reload.on_click(show_next_cluster)
display(cluster_title, cluster_summary, cluster_samples, cluster_label, cluster_notes, cluster_confirm, widgets.HBox([cluster_save, cluster_reload]), cluster_status)
show_next_cluster()
'''),
        cell("code", '''# Finalize decision v04 only when every required review is complete.
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
clusters = pd.read_csv(CLUSTERS, dtype=str, keep_default_na=False, encoding="utf-8-sig")
bridges = pd.read_csv(BRIDGES, dtype=str, keep_default_na=False, encoding="utf-8-sig")
required = clusters.loc[clusters.review_required.str.lower().eq("true")]
allowed = {"yes", "no", "uncertain"}
errors = []
errors += [f"pending cluster labels: {(~required.manual_cluster_valid.str.lower().isin(allowed)).sum()}"] if (~required.manual_cluster_valid.str.lower().isin(allowed)).any() else []
errors += [f"pending bridge labels: {(~bridges.manual_same_template.str.lower().isin(allowed)).sum()}"] if (~bridges.manual_same_template.str.lower().isin(allowed)).any() else []
errors += ["cluster reviewer missing"] if required.reviewer.str.strip().eq("").any() else []
errors += ["bridge reviewer missing"] if bridges.reviewer.str.strip().eq("").any() else []
if errors: raise ValueError("; ".join(errors))
record = json.loads(DRAFT.read_text(encoding="utf-8"))
record["record_version"] = "v04"
record["finalized_utc"] = pd.Timestamp.utcnow().isoformat()
record["fuzzy_decision"] = {
    "status": "accepted_with_c_lite_guards",
    "cluster_label_counts": required.manual_cluster_valid.str.lower().value_counts().to_dict(),
    "bridge_label_counts": bridges.manual_same_template.str.lower().value_counts().to_dict(),
    "review_order": "bridge_edges_then_cluster_decision",
    "cluster_evidence_rows": len(pd.read_csv(EVIDENCE)),
}
inputs = record.pop("parent_governance_inputs")
for filename in ["fuzzy_duplicate_audit_master.csv", CLUSTERS.name, BRIDGES.name, EVIDENCE.name, "seed_v052_coverage_analysis_v01.csv"]:
    path = AUDIT_ROOT / filename
    inputs[filename] = {"rows": len(pd.read_csv(path)), "sha256": sha(path)}
record["governance_inputs"] = inputs
record["build_gate"] = {"scope": "build_seed_v052_candidate", "passed": True, "blockers": []}
FINAL.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print("Decision v04:", FINAL)
print("SHA-256:", sha(FINAL))
'''),
    ]
    notebook(name, cells)


def privacy_notebook() -> None:
    name = "FinDisputeEval_CFPB_Privacy_NER_QA_colab_v01.ipynb"
    module_hash, module_blob = encoded(ROOT / "src/findisputeeval/curation/cfpb_privacy_qa.py")
    cells = [
        cell("markdown", "# CFPB Privacy NER QA\n\nDefault scope is the 20-row engineering smoke input. Change `SCAN_SCOPE` to `full_v052` only after Seed v05.2 exists. This uses Presidio with spaCy `en_core_web_sm`; every positive requires manual review.\n"),
        cell("code", '''from pathlib import Path
import base64, gzip, hashlib, json, subprocess, sys
import pandas as pd

SCAN_SCOPE = "smoke20"  # smoke20 or full_v052
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/FinDisputeEval")
else:
    here = Path.cwd().resolve()
    ROOT = next((p for p in (here, *here.parents) if (p / "WORK_PROGRESS.md").exists()), None)
    if ROOT is None: raise FileNotFoundError("Open the repository")
'''),
        cell("code", '''subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas>=2.2,<3", "pyarrow>=16", "spacy>=3.8,<4", "presidio-analyzer>=2.2,<3"])
check = subprocess.run([sys.executable, "-c", "import spacy; spacy.load('en_core_web_sm')"], check=False)
if check.returncode != 0:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
'''),
        cell("code", f'''RUNTIME = Path("/content/findisputeeval_privacy") if IN_COLAB else ROOT / ".runtime/privacy_qa"
PACKAGE = RUNTIME / "findisputeeval/curation"
PACKAGE.mkdir(parents=True, exist_ok=True)
(RUNTIME / "findisputeeval/__init__.py").write_text("", encoding="utf-8")
(PACKAGE / "__init__.py").write_text("", encoding="utf-8")
raw = gzip.decompress(base64.b64decode("{module_blob}"))
if hashlib.sha256(raw).hexdigest() != "{module_hash}": raise ValueError("Embedded privacy module hash mismatch")
(PACKAGE / "cfpb_privacy_qa.py").write_bytes(raw)
sys.path.insert(0, str(RUNTIME))
from findisputeeval.curation.cfpb_privacy_qa import build_presidio_analyzer, finalize_privacy_review, scan_frame, sha256_file
'''),
        cell("code", '''if SCAN_SCOPE == "smoke20":
    source = ROOT / "outputs/generation/smoke_only/cfpb_seed_v051_candidate/prepared_inputs/nemo_seed_v051_provisional_smoke_20.jsonl"
    frame = pd.read_json(source, lines=True)
    id_column, text_column, split_column = "seed_id", "seed_narrative_excerpt", "release_split"
elif SCAN_SCOPE == "full_v052":
    release = ROOT / "dataset/curated/seed_pools/cfpb_dispute/seed_v052"
    seed = pd.read_parquet(release / "cfpb_seed_v052.parquet")
    stress = pd.read_parquet(release / "cfpb_seed_v052_stress_test.parquet").rename(columns={"stress_id": "seed_id"})
    frame = pd.concat([seed, stress], ignore_index=True)
    source = release / "seed_v052_manifest.json"
    id_column, text_column, split_column = "seed_id", "seed_text", "release_split"
else:
    raise ValueError(SCAN_SCOPE)
if not source.exists(): raise FileNotFoundError(source)
QA_ROOT = ROOT / "dataset/curated/annotations/cfpb_seed_v05_audit/run_20260713T145423Z/privacy_qa" / SCAN_SCOPE
QA_ROOT.mkdir(parents=True, exist_ok=True)
REVIEW = QA_ROOT / "presidio_spacy_review.csv"
CLEARANCE = QA_ROOT / "privacy_clearance.json"
print({"scope": SCAN_SCOPE, "rows": len(frame), "source": str(source), "qa_root": str(QA_ROOT)})
'''),
        cell("code", '''# Scan once. Existing manual review is never overwritten.
if REVIEW.exists():
    review = pd.read_csv(REVIEW, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    print("Existing review loaded:", len(review))
else:
    analyzer = build_presidio_analyzer("en_core_web_sm")
    review = scan_frame(frame, analyzer, id_column=id_column, text_column=text_column, split_column=split_column, minimum_score=0.35)
    review.to_csv(REVIEW, index=False, encoding="utf-8-sig")
    print("NER findings:", len(review))
display(review.head(50))
'''),
        cell("code", '''# Interactive manual review; progress is saved after each answer.
for index in review.index[review.manual_pii_present.eq("pending")]:
    row = review.loc[index]
    print("\\n", row.record_id, row.entity_type, row.score)
    print(row.context)
    label = input("Actual PII? [yes/no/stop]: ").strip().lower()
    if label == "stop": break
    if label not in {"yes", "no"}: print("Invalid; unchanged"); continue
    review.at[index, "manual_pii_present"] = label
    review.at[index, "release_action"] = "allow" if label == "no" else input("Action [exclude/redact_and_rescan]: ").strip().lower()
    review.at[index, "reviewer"] = "chang"
    review.at[index, "reviewed_utc"] = pd.Timestamp.utcnow().isoformat()
    review.at[index, "notes"] = input("Optional note: ").strip()
    review.to_csv(REVIEW, index=False, encoding="utf-8-sig")
print("Pending:", review.manual_pii_present.eq("pending").sum())
'''),
        cell("code", '''result = finalize_privacy_review(review, scope=SCAN_SCOPE, source_sha256=sha256_file(source))
CLEARANCE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["release_clearance_passed"]:
    print("BLOCKED: apply versioned exclusions/redaction, rebuild, then rescan.")
'''),
    ]
    notebook(name, cells)


def build_notebook() -> None:
    name = "FinDisputeEval_CFPB_SeedPool_Build_colab_v052.ipynb"
    sources = {}
    for filename, path in {
        "cfpb_seed_v05.py": ROOT / "src/findisputeeval/curation/cfpb_seed_v05.py",
        "cfpb_seed_common.py": ROOT / "src/findisputeeval/curation/cfpb_seed_common.py",
        "cfpb_seed_v052.py": ROOT / "src/findisputeeval/curation/cfpb_seed_v052.py",
    }.items():
        sources[filename] = encoded(path)
    embedded = json.dumps(sources)
    cells = [
        cell("markdown", "# CFPB Seed Pool Build v05.2\n\nFail-closed builder for the 2,000-row proportional core plus coverage supplement. It cannot run until decision v04 has been finalized after C-lite fuzzy review. Output remains a privacy-QA-pending candidate.\n"),
        cell("code", '''from pathlib import Path
import base64, gzip, hashlib, json, subprocess, sys
IN_COLAB = "google.colab" in sys.modules
RUN_ID = "run_20260713T145423Z"
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/FinDisputeEval")
else:
    here = Path.cwd().resolve()
    ROOT = next((p for p in (here, *here.parents) if (p / "WORK_PROGRESS.md").exists()), None)
    if ROOT is None: raise FileNotFoundError("Open the repository")
EDA = ROOT / "outputs/data_pipeline/cfpb_seed_source_eda/eda_v051" / RUN_ID
AUDIT = ROOT / "dataset/curated/annotations/cfpb_seed_v05_audit" / RUN_ID
DECISION = AUDIT / "seed_v052_decision_record_v04.json"
OUTPUT = ROOT / "dataset/curated/seed_pools/cfpb_dispute/seed_v052"
if not DECISION.exists(): raise FileNotFoundError("Complete fuzzy v04 review first: " + str(DECISION))
record = json.loads(DECISION.read_text(encoding="utf-8"))
if record.get("record_version") != "v04" or not record.get("build_gate", {}).get("passed"):
    raise ValueError("Decision v04 build gate is closed")
print({"eda": str(EDA), "decision": str(DECISION), "output": str(OUTPUT)})
'''),
        cell("code", '''subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pandas>=2.2,<3", "numpy>=1.26,<3", "pyarrow>=16"])
'''),
        cell("code", f'''embedded = json.loads({json.dumps(embedded)})
RUNTIME = Path("/content/findisputeeval_v052") if IN_COLAB else ROOT / ".runtime/seed_v052"
PACKAGE = RUNTIME / "findisputeeval/curation"
PACKAGE.mkdir(parents=True, exist_ok=True)
(RUNTIME / "findisputeeval/__init__.py").write_text("", encoding="utf-8")
(PACKAGE / "__init__.py").write_text("", encoding="utf-8")
for filename, values in embedded.items():
    expected, blob = values
    raw = gzip.decompress(base64.b64decode(blob))
    if hashlib.sha256(raw).hexdigest() != expected: raise ValueError(filename)
    (PACKAGE / filename).write_bytes(raw)
sys.path.insert(0, str(RUNTIME))
from findisputeeval.curation.cfpb_seed_v052 import SeedV052Config, build_seed_v052
'''),
        cell("code", '''paths = build_seed_v052(SeedV052Config(eda_run_dir=EDA, decision_record_path=DECISION, output_dir=OUTPUT, random_seed=20260713))
manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
print(json.dumps({
    "release": manifest["release"],
    "primary_rows": manifest["primary_rows"],
    "coverage_supplement_rows": manifest["coverage_supplement_rows"],
    "enrichment_rows": manifest["enrichment_rows"],
    "stress_rows": manifest["stress_rows"],
    "benchmark_eligible": manifest["benchmark_eligible"],
    "manifest": str(paths["manifest"]),
}, indent=2))
'''),
    ]
    notebook(name, cells)


if __name__ == "__main__":
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    fuzzy_notebook()
    privacy_notebook()
    build_notebook()
    print("Generated v05.2 workflow notebooks")
