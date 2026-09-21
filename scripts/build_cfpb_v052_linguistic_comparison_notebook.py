"""Build a self-contained, offline comparison notebook. Does not modify source runs."""
from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluation/cfpb_v052_linguistic_comparison_v01.py"
CONTRACT = ROOT / "configs/evaluation/cfpb_v051_legacy_linguistic_contract_v01.json"
OUTPUT = ROOT / "notebooks/30_generation/nemo_data_designer/seeded_dialogue/canonical/FinDisputeEval_CFPB_v052_generation_linguistic_comparison_v01_colab.ipynb"


def legacy_contract():
    source = ROOT / "src/findisputeeval/cfpb_seed_source_eda_v05.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "LEXICAL_PATTERNS" for t in n.targets))
    # Evaluate only the existing literal regex dictionary, not the EDA program.
    patterns = eval(compile(ast.Expression(node.value), str(source), "eval"), {"re": re, "__builtins__": {}})
    return {"version": "v051_unchanged_lexical_candidates_v01", "source": source.relative_to(ROOT).as_posix(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "interpretation": "Unvalidated lexical candidates; reused only as a historical measurement bridge.",
            "patterns": {name: {"pattern": p.pattern, "flags": p.flags} for name, p in patterns.items()}}


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}
    result["id"] = hashlib.sha256(result["source"].encode()).hexdigest()[:12]
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build_notebook():
    contract = (json.dumps(legacy_contract(), ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    payload = {}
    for name, data in [(SCRIPT.name, SCRIPT.read_bytes()), (CONTRACT.name, contract)]:
        compressed = bytearray(gzip.compress(data, mtime=0))
        compressed[9] = 255  # Stable OS header across Python 3.11 and 3.13+.
        payload[name] = {"sha256": hashlib.sha256(data).hexdigest(),
                         "gzip_base64": base64.b64encode(compressed).decode()}
    cells = [cell("markdown", """
        # CFPB v05.2：生成對話 × Seed × EDA 語言比較 v01

        只讀取既有 Super 20-row run；不生成新對話，不呼叫 LLM/judge，不需要 API key。
        輸出為探索性 Excel / HTML / CSV / 中文摘要，不是品質判決或正式分布校準。
        全部 20 筆保留；完整 seed、excerpt、grounding、user、assistant 分開分析。

        使用 VS Code Colab 插件連接 CPU runtime，依序 Run All。安裝會下載程式庫與
        spaCy 英文模型，但不會將資料傳送到模型 API。報告仍含衍生文本，請留在私人 Drive。
        已有本地報告時可以直接閱讀，不必重跑。Colab Python 3.11–3.13；GPU 不需要。
        """), cell("code", '''
        from pathlib import Path
        from datetime import datetime, timezone
        import os, sys, json

        IN_COLAB = "google.colab" in sys.modules
        if IN_COLAB:
            from google.colab import drive
            if not Path("/content/drive/MyDrive").is_dir():
                try:
                    drive.mount("/content/drive", timeout_ms=180000)
                except Exception as exc:
                    raise RuntimeError("Drive authentication failed. Reconnect the Colab runtime and authorize the correct Google account, then rerun. No project input was read.") from exc
            PROJECT_ROOT = Path("/content/drive/MyDrive/FinDisputeEval")
        else:
            here = Path.cwd().resolve()
            PROJECT_ROOT = next((p for p in (here, *here.parents) if (p / "WORK_PROGRESS.md").is_file()), None)
            if PROJECT_ROOT is None:
                raise FileNotFoundError("Open the FinDisputeEval project first")

        ANALYSIS_ID_OVERRIDE = ""  # Set a NEW explicit ID if desired; never overwrite an existing report.
        ANALYSIS_ID = ANALYSIS_ID_OVERRIDE or datetime.now(timezone.utc).strftime("analysis_%Y%m%dT%H%M%S%fZ")
        if not ANALYSIS_ID or Path(ANALYSIS_ID).name != ANALYSIS_ID or "/" in ANALYSIS_ID or "\\\\" in ANALYSIS_ID or ANALYSIS_ID in {".", ".."}:
            raise ValueError("ANALYSIS_ID must be a single directory name")
        OUT = (PROJECT_ROOT / "outputs/analysis/smoke_only/cfpb_v052_linguistic_comparison"
               / "source_run_20260919T220844338002Z" / ANALYSIS_ID)
        print({"project_root": str(PROJECT_ROOT), "new_analysis_output": str(OUT), "api_calls": False})
        '''), cell("markdown", """
        ## 1. 安裝本地文字分析套件

        如安裝後提示已載入套件版本衝突，Restart Session 再從首格執行。
        解析器不可用時會停止，不會悄悄改用 regex 假裝已完成句法分析。
        """), cell("code", '''
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "spacy==3.8.7", "pandas>=2.2,<3", "numpy>=1.26,<3", "pyarrow>=17,<24",
            "scikit-learn>=1.5,<2", "openpyxl>=3.1,<4",
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"])
        '''), cell("markdown", "## 2. 載入內嵌、SHA-256 驗證的分析程式與歷史 detector contract"),
        cell("code", "import base64, gzip, hashlib, importlib.util, tempfile\n"
             + "EMBEDDED = " + repr(payload) + "\n" + textwrap.dedent('''
        RUNTIME_SOURCE = Path(tempfile.mkdtemp(prefix="fde_linguistic_v01_"))
        for name, item in EMBEDDED.items():
            data = gzip.decompress(base64.b64decode(item["gzip_base64"]))
            assert hashlib.sha256(data).hexdigest() == item["sha256"], name
            (RUNTIME_SOURCE / name).write_bytes(data)
        spec = importlib.util.spec_from_file_location("fde_linguistic_v01", RUNTIME_SOURCE / "cfpb_v052_linguistic_comparison_v01.py")
        analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analysis)
        CONTRACT = RUNTIME_SOURCE / "cfpb_v051_legacy_linguistic_contract_v01.json"
        print({"embedded_hashes_verified": True, "analysis_version": analysis.CONFIG["analysis_version"]})
        ''')), cell("markdown", """
        ## 3. 只讀 preflight：來源 hash 與 20 筆配對

        需要既有 Super run（包含 evidence manifest 所列全部檔案）、Seed v05.2 parquet / generation JSONL / manifest，
        以及 EDA v05.1 manifest 和五份小型背景統計表。缺檔先按操作說明同步，不必重跑 EDA 或 generation。
        """), cell("code", '''
        import pandas as pd
        from IPython.display import display, Markdown, FileLink
        inventory, raw, prepared = analysis.verify_inputs(PROJECT_ROOT)
        cases = analysis.assemble_cases(PROJECT_ROOT, raw, prepared)
        print({"verified_input_files": len(inventory), "cases": len(cases),
               "format_valid_cases": sum(c["format_valid"] for c in cases), "all_cases_retained": True})
        display(pd.DataFrame([{k: c[k] for k in ["seed_id", "release_split", "expected_messages", "actual_messages", "format_valid"]} for c in cases]))
        '''), cell("markdown", """
        ## 4. 完整離線分析與匯出

        將建立新 analysis 目錄；若重跑此格遇到既有結果，請直接看下一格，或回首格建立新 ANALYSIS_ID。
        不刪除／覆寫任何舊輸出。通常數分鐘內完成，CPU 即可。
        """), cell("code", '''
        report = analysis.run_analysis(PROJECT_ROOT, OUT, CONTRACT)
        print({k: report[k] for k in ["cases", "stage_rows", "format_valid_cases", "policy"]})
        '''), cell("markdown", "## 5. 閱讀結果：先 HTML，再 Excel 的逐筆配對，最後才對照 EDA 背景"),
        cell("code", '''
        report = analysis.read_json(OUT / "analysis_manifest.json")
        for name, expected in report["outputs"].items():
            assert analysis.sha(OUT / name) == expected, name
        display(Markdown((OUT / "comparison_summary_zh.md").read_text(encoding="utf-8")))
        for name in ["linguistic_comparison_20.html", "linguistic_comparison_20.xlsx", "analysis_manifest.json"]:
            print(str(OUT / name))
            display(FileLink(str(OUT / name)))
        # If links do not open in VS Code, download the files from this Drive output directory.
        '''), cell("markdown", """
        ## 解讀邊界

        - 來源截斷／脫敏與模型改寫分開；不要把 user+assistant 合起來跟投訴敘事比較。
        - MATTR 固定 25 tokens，短 turn 留空；否定、情態、候選情緒詞都不是語意真值。
        - 舊 EDA 的人口、enrichment、去重／family 權重视角分開；本批 20 筆不是比例樣本。
        - 詞彙 overlap / TF-IDF 不是 hallucination 或 correctness；候選修復詞不是已驗證的 conversational repair。
        - privacy_verified、benchmark_eligible、formal_pilot_allowed 仍為 False。
        """)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                            "language_info": {"name": "python"}, "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}, contract


if __name__ == "__main__":
    notebook, contract = build_notebook()
    CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT.write_bytes(contract)
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))
