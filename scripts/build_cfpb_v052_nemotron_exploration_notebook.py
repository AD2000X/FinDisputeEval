"""Build the review-first notebook and a private Drive upload bundle (no API calls)."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
from textwrap import dedent
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = Path("notebooks/30_generation/nemo_data_designer/seeded_dialogue/canonical")
NOTEBOOK = CANONICAL / "FinDisputeEval_CFPB_v052_nemotron_exploration_20_v01_colab.ipynb"
GUIDE = Path("docs/generation/cfpb_v052_nemotron_exploration_20_v01_zh.md")
CONFIG = Path("configs/generation/cfpb_v052_nemotron_exploration_v01_inputs.json")
MODULE_ROOT = Path("scripts/generation/nemo_data_designer")
MODULE_NAMES = ["prepare_cfpb_seed_v052_pipeline_smoke_v02.py",
                "generate_multi_turn_dialogues_v052_pipeline_smoke_v02.py",
                "cfpb_v052_pipeline_smoke_v02_common.py",
                "validate_cfpb_v052_pipeline_smoke_v02.py",
                "cfpb_v052_nemotron_exploration_v01.py"]


def inputs():
    seed = Path("dataset/curated/seed_pools/cfpb_dispute/seed_v052")
    privacy = Path("dataset/curated/annotations/cfpb_seed_v05_audit/run_20260713T145423Z/privacy_qa/full_v052_v02")
    return [seed / name for name in ["cfpb_seed_v052.parquet", "cfpb_seed_v052_generation_input.jsonl", "seed_v052_manifest.json"]] + [
        privacy / name for name in ["pipeline_privacy_disposition_v01.json", "analyzer_config_v02.json",
            "scan_inventory_v02.json", "presidio_spacy_findings_review_v02.csv", "negative_control_review_v02.csv",
            "challenge_results_v02.csv", "pre_pipeline_override_v01/backup_manifest.json"]] + [Path(
        "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override/run_20260722T135306Z/"
        "prepared_inputs/nemo_seed_v052_pipeline_smoke_20.jsonl")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(kind, text):
    result = dict(cell_type=kind, metadata={}, source=dedent(text).strip() + "\n")
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook():
    inventory = {p.as_posix(): digest(ROOT / p) for p in inputs()}
    payload = {name: {"sha256": digest(ROOT / MODULE_ROOT / name),
                       "data": base64.b64encode(gzip.compress((ROOT / MODULE_ROOT / name).read_bytes(), mtime=0)).decode()}
               for name in MODULE_NAMES}
    cells = []
    md = lambda text: cells.append(cell("markdown", text))
    code = lambda text: cells.append(cell("code", text))
    md("""
    # CFPB v05.2 — Nemotron 20-row exploration v01

    **本輪只看生成品質：不呼叫 LLM judge、不 freeze、不正式發布。**
    使用既有 v02 prompt / privacy preparation，排除前一批 20 個 seed ID。
    新樣本仍是 engineering smoke，非正式 held-out、非隨機母體代表樣本。
    `privacy_verified=false`、`benchmark_eligible=false` 永遠保持。

    VS Code 選 Colab 的 **CPU runtime**，按以下順序逐格執行。
    第 5 步才允許呼叫 NVIDIA；預設開關為 False。無需 OpenRouter key。
    完整說明：`docs/generation/cfpb_v052_nemotron_exploration_20_v01_zh.md`。
    """)
    md("## 0. 連接 Drive，選擇獨立 run（重連時填 RUN_ID_OVERRIDE）")
    code('''
    from pathlib import Path
    from datetime import datetime, timezone
    import hashlib, json, os, re, sys

    RUN_ID_OVERRIDE = ""  # Resume: copy the complete run_... name printed below.
    IN_COLAB = "google.colab" in sys.modules
    if IN_COLAB:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").is_dir():
            try:
                drive.mount("/content/drive", timeout_ms=180000)
            except Exception as exc:
                raise RuntimeError("Drive authentication failed before any model call. Reconnect the VS Code Colab runtime and complete browser sign-in; do not change the API key to fix Drive.") from exc
        PROJECT_ROOT = Path("/content/drive/MyDrive/FinDisputeEval")
    else:
        here = Path.cwd().resolve()
        PROJECT_ROOT = next((p for p in (here, *here.parents) if (p / "WORK_PROGRESS.md").is_file()), None)
        if PROJECT_ROOT is None:
            raise FileNotFoundError("Open the FinDisputeEval project in VS Code")
    if not PROJECT_ROOT.is_dir():
        raise FileNotFoundError(PROJECT_ROOT)
    os.environ["FINDISPUTEEVAL_PROJECT_ROOT"] = str(PROJECT_ROOT)
    if RUN_ID_OVERRIDE:
        _FDE_EXPLORATION_RUN_ID = RUN_ID_OVERRIDE
    elif "_FDE_EXPLORATION_RUN_ID" not in globals():
        _FDE_EXPLORATION_RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    RUN_ID = _FDE_EXPLORATION_RUN_ID
    if not re.fullmatch(r"run_[A-Za-z0-9_-]+", RUN_ID):
        raise ValueError("RUN_ID must be a simple run_... directory name")
    RUN_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override/exploratory_nemotron_v01" / RUN_ID
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    print({"project_root": str(PROJECT_ROOT), "run_id": RUN_ID, "run_root": str(RUN_ROOT)})
    ''')
    md("## 1. 安裝 library（尚不呼叫模型）")
    code('''
    import importlib.metadata, subprocess
    PACKAGES = ["data-designer==0.7.0", "pydantic>=2.10,<3", "pandas>=2.2,<3",
                "pyarrow>=18,<23", "openpyxl>=3.1,<4"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
    VERSIONS = {name: importlib.metadata.version(name) for name in
                ["data-designer", "pydantic", "pandas", "pyarrow", "openpyxl"]}
    print(VERSIONS)
    ''')
    md("## 2. 載入內嵌程式快照與固定的輸入 SHA-256 清單")
    code(f'''
    import base64, gzip, importlib
    INPUT_INVENTORY = json.loads({json.dumps(inventory, sort_keys=True)!r})
    EMBEDDED = json.loads({json.dumps(payload, sort_keys=True)!r})
    SNAPSHOT = RUN_ROOT / "source_snapshot"
    SNAPSHOT.mkdir(exist_ok=True)
    for name, item in EMBEDDED.items():
        raw = gzip.decompress(base64.b64decode(item["data"]))
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Embedded module hash mismatch: " + name)
        target = SNAPSHOT / name
        if target.exists() and target.read_bytes() != raw:
            raise ValueError("Run has a different source snapshot; choose a new run")
        if not target.exists():
            target.write_bytes(raw)
    sys.path.insert(0, str(SNAPSHOT))
    for name in EMBEDDED:
        sys.modules.pop(Path(name).stem, None)
        sys.modules.pop("scripts.generation.nemo_data_designer." + Path(name).stem, None)
    # The existing validator supports both repo and flat imports. Bind its two
    # dependencies to this exact snapshot even when a full repo is on sys.path.
    for module_name in ("prepare_cfpb_seed_v052_pipeline_smoke_v02", "cfpb_v052_pipeline_smoke_v02_common"):
        loaded = importlib.import_module(module_name)
        sys.modules["scripts.generation.nemo_data_designer." + module_name] = loaded
    import cfpb_v052_nemotron_exploration_v01 as workflow
    import generate_multi_turn_dialogues_v052_pipeline_smoke_v02 as generation
    workflow.verify_inventory(PROJECT_ROOT, INPUT_INVENTORY)
    workflow.save_json(RUN_ROOT / "input_inventory.json", INPUT_INVENTORY)
    workflow.save_json(RUN_ROOT / "environment_versions.json", VERSIONS)
    MODULE_HASHES = {{name: item["sha256"] for name, item in EMBEDDED.items()}}
    print({{"inputs_verified": len(INPUT_INVENTORY), "modules_verified": len(MODULE_HASHES)}})
    ''')
    md("## 3. 準備 20 筆新 seed（offset、脫敏、稀疏度、舊 ID 排除；無 API）")
    code('''
    import pandas as pd
    RANDOM_SEED = 20260919
    PREPARED, INPUT_MANIFEST, prepared_info = workflow.prepare_run(
        PROJECT_ROOT, RUN_ROOT, INPUT_INVENTORY, random_seed=RANDOM_SEED)
    print({"selected_rows": prepared_info["selected_rows"],
           "prior_overlap": prepared_info["prior_sample_exclusion"]["selected_overlap"],
           "offset_invariant": prepared_info["offset_invariant"],
           "grounding_gate": prepared_info["grounding_gate"],
           "policy": workflow.POLICY})
    display(pd.DataFrame(workflow.load_records(PREPARED))[["seed_id", "product", "issue", "release_split"]])
    ''')
    md("## 4. 固定 Nemotron 與生成設定（仍不呼叫模型）")
    code('''
    import data_designer.config as dd
    MODEL = "nvidia/nemotron-3-nano-30b-a3b"
    MODEL_ALIAS = "nemotron-exploration-v01"
    # Same model, temperature/top_p and concurrency as the original smoke.
    # Explicit 120-second per-request timeout is new; no max_tokens/reasoning override.
    INFERENCE = {"temperature": 1.0, "top_p": 1.0, "max_parallel_requests": 4, "timeout": 120}
    builder = generation.build_config(str(PREPARED), MODEL_ALIAS)
    builder.add_model_config(dd.ModelConfig(alias=MODEL_ALIAS, model=MODEL,
                             provider="nvidia", inference_parameters=INFERENCE))
    for existing_model in list(builder.model_configs):
        if existing_model.alias != MODEL_ALIAS:
            builder.delete_model_config(existing_model.alias)
    # Pass only the explicitly pinned NVIDIA provider; never delete global cached settings.
    PROVIDER = dd.ModelProvider(name="nvidia", provider_type="openai",
                               endpoint="https://integrate.api.nvidia.com/v1", api_key="NVIDIA_API_KEY")
    workflow.save_json(RUN_ROOT / "builder_config.json", builder.build().model_dump(mode="json"))
    SPEC = {"purpose": "exploratory_generation_quality_not_heldout_benchmark",
            "model": MODEL, "inference_parameters": INFERENCE, "preview_records": 0,
            "provider": PROVIDER.model_dump(mode="json"), "random_seed": RANDOM_SEED,
            "prepared_sha256": workflow.sha256_file(PREPARED),
            "input_manifest_sha256": workflow.sha256_file(INPUT_MANIFEST),
            "input_inventory": INPUT_INVENTORY, "module_hashes": MODULE_HASHES,
            "environment_versions": VERSIONS, "policy": workflow.POLICY,
            "sampling_note": "Seed selection is deterministic; mood/length/model output are not guaranteed deterministic."}
    workflow.save_json(RUN_ROOT / "run_spec.json", SPEC)
    print({"model": MODEL, "rows": 20, "preview": 0, "judge": "disabled", "api_calls_so_far": 0})
    ''')
    md("""
    ## 5. 呼叫 NVIDIA 生成 20 筆（唯一付費/API 步驟）

    先確認第 3 步與第 4 步，再把 `RUN_GENERATION` 改為 `True`。
    Key 只輸入於密碼欄，不要貼在 notebook 原始碼。沿用既有 NVIDIA smoke 範圍；這不是 ZDR 或隱私安全保證。
    不額外 preview；Data Designer 健康檢查與內部重試仍可能增加請求數，並非恰好 20 次請求。
    已完成的 raw 會快取；中斷且不完整則停止，不自動重跑。
    """)
    code('''
    RUN_GENERATION = False  # Change to True only when ready to submit this new smoke batch.
    RAW = workflow.resolve_cached_raw(RUN_ROOT, PREPARED)
    if RAW is None:
        if not RUN_GENERATION:
            raise RuntimeError("Preparation complete. Set RUN_GENERATION=True to authorize this batch; no model call was made.")
        from getpass import getpass
        from data_designer.interface import DataDesigner
        if not os.environ.get("NVIDIA_API_KEY", "").strip():
            os.environ["NVIDIA_API_KEY"] = getpass("NVIDIA_API_KEY (not saved): ").strip()
        if not os.environ.get("NVIDIA_API_KEY"):
            raise RuntimeError("NVIDIA_API_KEY is required")
        # Recheck immutable inputs immediately before submitting data.
        workflow.verify_inventory(PROJECT_ROOT, INPUT_INVENTORY)
        if workflow.sha256_file(PREPARED) != SPEC["prepared_sha256"]:
            raise ValueError("Prepared input changed after configuration")
        workflow.save_json(RUN_ROOT / "generation_started.json", {"started_utc": workflow.utc_now(),
                           "run_spec_sha256": workflow.sha256_file(RUN_ROOT / "run_spec.json")})
        try:
            designer = DataDesigner(artifact_path=RUN_ROOT / "raw/data_designer", model_providers=[PROVIDER])
            designer.validate(builder)
            result = designer.create(builder, num_records=20)
            RAW = generation.resolve_final_dataset_file(Path(result.artifact_storage.final_dataset_path))
            workflow.record_raw(RUN_ROOT, RAW, PREPARED)
        except Exception as exc:
            workflow.save_json(RUN_ROOT / "generation_failure.json", {
                "failed_utc": workflow.utc_now(), "exception_type": type(exc).__name__,
                "note": "Inspect notebook/SDK artifacts. No automatic regeneration; credentials and response bodies not copied to this audit."})
            raise
    print({"raw": str(RAW), "raw_sha256": workflow.sha256_file(RAW), "rows": 20})
    ''')
    md("## 6. 產生並排 Excel、完整 HTML 與技術檢查報告（全部待審，無 judge）")
    code('''
    WORKBOOK = workflow.build_review_pack(RAW, PREPARED, RUN_ROOT / "review")
    # Immutable evidence files only; the editable workbook is versioned by summary hashes instead.
    files = [p for p in RUN_ROOT.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and p.suffix != ".pyc"
             and p.name != "evidence_manifest.json" and p.suffix != ".xlsx"
             and not p.name.startswith("summary_")]
    workflow.save_json(RUN_ROOT / "evidence_manifest.json", {
        "files": {p.relative_to(RUN_ROOT).as_posix(): workflow.sha256_file(p) for p in sorted(files)},
        "policy": workflow.POLICY, "note": "Evidence inventory, NOT a frozen or approved judge."})
    print({"excel": str(WORKBOOK), "html": str(RUN_ROOT / "review/review_readable.html"),
           "all_rows_pending_review": 20, "auto_accepted_rows": 0})
    ''')
    md("""
    ## 7. 初始／完成評閱後的摘要（可重跑；無 API）

    先下載或同步 Excel 與 HTML。本格初次只會顯示 20 筆 pending，**不會捏造品質結論**。
    填完 Excel 黃色欄位，儲存並同步回同一路徑後，只需重跑本格。
    重新連線時執行 0–4（填原 run ID），然後可直接執行本格，不必執行第 5 步。
    若先由 AI 初評，`reviewer_kind=llm`；真人自行讀過才填 `human`。
    """)
    code('''
    WORKBOOK = RUN_ROOT / "review/generation_review_20.xlsx"
    summary = workflow.summarize_review(WORKBOOK)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Sync the entire run directory back locally:", RUN_ROOT)
    ''')
    for i, c in enumerate(cells):
        c["id"] = f"exploration-{i:02d}"
    return dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}, "colab": {"name": NOTEBOOK.name, "provenance": []}},
                nbformat=4, nbformat_minor=5), inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="store_true", help="Package exact private inputs and code for manual Drive sync")
    args = parser.parse_args()
    notebook, inventory = build_notebook()
    destination = ROOT / NOTEBOOK
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if any(c.get("outputs") or c.get("execution_count") is not None for c in existing["cells"] if c["cell_type"] == "code"):
            raise ValueError("Refusing to overwrite an executed notebook")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (ROOT / CONFIG).write_text(json.dumps(inventory, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(destination)
    if args.bundle:
        out = ROOT / "outputs/handoff/cfpb_v052_nemotron_exploration_v01"
        out.mkdir(parents=True, exist_ok=True)
        archive = out / "FinDisputeEval_nemotron_exploration_v01_PRIVATE.zip"
        paths = inputs() + [NOTEBOOK, GUIDE, CONFIG] + [MODULE_ROOT / name for name in MODULE_NAMES]
        paths += [Path(__file__).relative_to(ROOT)]
        manifest = {p.as_posix(): digest(ROOT / p) for p in paths}
        with ZipFile(archive, "w", ZIP_DEFLATED) as z:
            for path in paths:
                z.write(ROOT / path, "FinDisputeEval/" + path.as_posix())
            z.writestr("FinDisputeEval/UPLOAD_MANIFEST_nemotron_exploration_v01.json", json.dumps(manifest, indent=2))
        (out / "upload_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(archive)
        print(f"Private bundle: {len(paths)} files, {archive.stat().st_size} bytes. Contains unverified source text; do not publish.")


if __name__ == "__main__":
    main()
