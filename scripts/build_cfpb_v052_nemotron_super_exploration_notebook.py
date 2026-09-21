"""Build a separate Super exploration notebook; never rewrite Nano lineage."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from scripts import build_cfpb_v052_nemotron_exploration_notebook as base
except ImportError:
    import build_cfpb_v052_nemotron_exploration_notebook as base

OUTPUT = base.ROOT / base.CANONICAL / "FinDisputeEval_CFPB_v052_nemotron_super_exploration_20_v02_colab.ipynb"


def build_notebook():
    notebook, _ = base.build_notebook()
    old = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    old[0] = old[0].replace("_FDE_EXPLORATION_RUN_ID", "_FDE_SUPER_EXPLORATION_RUN_ID").replace(
        "exploratory_nemotron_v01", "exploratory_nemotron_super_v02")
    old[4] = old[4].replace('MODEL = "nvidia/nemotron-3-nano-30b-a3b"',
                            'MODEL = "nvidia/nemotron-3-super-120b-a12b"').replace(
        'MODEL_ALIAS = "nemotron-exploration-v01"', 'MODEL_ALIAS = "nemotron-super-exploration-v02"')
    old[4] = old[4].replace(
        '# Same model, temperature/top_p and concurrency as the original smoke.\n'
        '# Explicit 120-second per-request timeout is new; no max_tokens/reasoning override.\n'
        'INFERENCE = {"temperature": 1.0, "top_p": 1.0, "max_parallel_requests": 4, "timeout": 120}',
        '# Official Super sampling; explicit non-thinking and bounded output.\n'
        'INFERENCE = {"temperature": 1.0, "top_p": 0.95, "max_parallel_requests": 2,\n'
        '             "timeout": 120, "max_tokens": 4096,\n'
        '             "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}')
    old[4] += base.cell("code", '''
    def save_failure_audit(exc, path, stage):
        # Follow __context__ too: SDK errors raised "from None" hide HTTP 410.
        chain, seen = [], set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            message = str(exc)
            for secret_name in ("NVIDIA_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
                secret = os.environ.get(secret_name, "")
                if secret:
                    message = message.replace(secret, "[REDACTED_API_KEY]")
            message = re.sub(r"(?i)Bearer\\s+[^\\s'\\\"]+", "Bearer [REDACTED]", message)
            chain.append({"exception": type(exc).__name__, "status_code": getattr(exc, "status_code", None),
                          "message": message[:2000]})
            exc = exc.__cause__ or exc.__context__
        workflow.save_json(path, {"failed_utc": workflow.utc_now(), "stage": stage, "errors": chain,
                                  "model": MODEL, "retry_performed_by_notebook": False})
        print(json.dumps(chain, ensure_ascii=False, indent=2))

    PREFLIGHT_ROOT = RUN_ROOT / "preflight"
    PREFLIGHT_REPORT = PREFLIGHT_ROOT / "preflight_report.json"
    def require_passed_preflight():
        if not PREFLIGHT_REPORT.is_file():
            raise RuntimeError("Run step 5 first: no successful synthetic-only preflight")
        report = json.loads(PREFLIGHT_REPORT.read_text(encoding="utf-8"))
        if report.get("passed") is not True or report.get("run_spec_sha256") != workflow.sha256_file(RUN_ROOT / "run_spec.json"):
            raise ValueError("Preflight is failed or belongs to a different run configuration")
        response = PREFLIGHT_ROOT / "response.json"
        if workflow.sha256_file(response) != report.get("response_sha256"):
            raise ValueError("Preflight response changed")
        return report
    ''')["source"]
    preflight = '''
    RUN_PREFLIGHT = False  # Set True to test NVIDIA using invented text only.
    if PREFLIGHT_REPORT.exists():
        print("Verified preflight cache:", require_passed_preflight())
    else:
        if not RUN_PREFLIGHT:
            raise RuntimeError("Set RUN_PREFLIGHT=True to authorize a synthetic-only API test. No complaint data is submitted.")
        if (PREFLIGHT_ROOT / "started.json").exists():
            raise RuntimeError("A prior preflight attempt did not finish. Preserve its audit and use a NEW run ID after resolving the error.")
        from getpass import getpass
        from data_designer.interface import DataDesigner
        from validate_cfpb_v052_pipeline_smoke_v02 import Dialogue
        if not os.environ.get("NVIDIA_API_KEY", "").strip():
            os.environ["NVIDIA_API_KEY"] = getpass("NVIDIA_API_KEY (not saved): ").strip()
        if not os.environ.get("NVIDIA_API_KEY"):
            raise RuntimeError("NVIDIA_API_KEY is required")
        probe = dd.DataDesignerConfigBuilder(model_configs=list(builder.model_configs))
        probe.add_column(dd.LLMStructuredColumnConfig(
            name="dialogue", model_alias=MODEL_ALIAS, output_format=generation.PipelineSmokeConversation,
            system_prompt="Write fictional English dialogue. Return the required structured output only.",
            prompt="Create exactly four messages alternating user, assistant, user, assistant. "
                   "The fictional user sees an unfamiliar charge and wants to ask for clarification. "
                   "The assistant asks a neutral question and suggests the official support channel. "
                   "Do not add names, numbers, URLs, policies, promises, or legal claims. "
                   "Include synthetic_case_summary and privacy_notes. This is invented test data."))
        workflow.save_json(PREFLIGHT_ROOT / "builder_config.json", probe.build().model_dump(mode="json"))
        workflow.save_json(PREFLIGHT_ROOT / "started.json", {"started_utc": workflow.utc_now(),
                           "run_spec_sha256": workflow.sha256_file(RUN_ROOT / "run_spec.json")})
        try:
            # Same SDK, provider, model, schema and inference parameters as generation.
            # preview includes the SDK Hello health check, then one invented dialogue.
            probe_engine = DataDesigner(artifact_path=PREFLIGHT_ROOT / "data_designer", model_providers=[PROVIDER])
            probe_result = probe_engine.preview(probe, num_records=1)
            if probe_result.dataset is None or len(probe_result.dataset) != 1:
                raise ValueError("Preflight did not return one structured row")
            response = Dialogue.model_validate(workflow.parse_structured(probe_result.dataset.iloc[0]["dialogue"]))
            if len(response.conversation) != 4:
                raise ValueError("Preflight dialogue did not contain exactly four messages")
            workflow.save_json(PREFLIGHT_ROOT / "response.json", response.model_dump(mode="json"))
            workflow.save_json(PREFLIGHT_REPORT, {"passed": True, "completed_utc": workflow.utc_now(),
                "run_spec_sha256": workflow.sha256_file(RUN_ROOT / "run_spec.json"),
                "response_sha256": workflow.sha256_file(PREFLIGHT_ROOT / "response.json"),
                "model": MODEL, "complaint_data_submitted": False,
                "note": "Connectivity/schema check only, not generation quality or privacy clearance."})
        except Exception as exc:
            save_failure_audit(exc, PREFLIGHT_ROOT / "failure.json", "synthetic_only_preflight")
            raise
        print(require_passed_preflight())
    '''
    old[5] = old[5].replace('    from getpass import getpass',
                            '    require_passed_preflight()\n    from getpass import getpass')
    start = old[5].index('        workflow.save_json(RUN_ROOT / "generation_failure.json"')
    end = old[5].index('        raise', start)
    old[5] = (old[5][:start] + '        save_failure_audit(exc, RUN_ROOT / "generation_failure.json", "generation")\n'
              + old[5][end:])
    intro = '''
    # CFPB v05.2 — Nemotron Super 探索性生成 v02

    舊 Nano endpoint 回傳 HTTP 410：2026-09-01 退役。**不要再執行舊 v01 notebook。**
    本版使用 `nvidia/nemotron-3-super-120b-a12b`，全新輸出目錄，不覆寫舊 run。
    同一份 seed、既有生成 prompt v02、Excel/HTML 阅读流程；不是 judge notebook。

    官方 sampling：temperature=1.0、top_p=0.95；明確關閉 thinking。
    本實驗設定 max_tokens=4096、並行 2、每請求 timeout=120 秒；不是原 Nano 的同模型比較。
    官方依據：[Model card](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard)。
    官方網站列出模型不等於實際 endpoint 保證可用，因此先執行第 5 步。

    **步驟 5：無申訴資料的 API/結構化輸出測試 → 步驟 6：才提交 20 筆脫敏 seed。**
    兩個 API 開關預設 False。測試和 SDK health checks/retries 也會產生 API 請求。
    若出現 401/410 等錯誤，畫面與 `failure.json` 會保留底層 status_code。
    不自動換模型、不跳過 health check、不自動刪失敗 marker。

    原 11 個資料依賴不變；只需同步本 notebook。Colab **CPU** 即可，模型在 NVIDIA 雲端。
    `privacy_verified=false`、`benchmark_eligible=false`；本輪僅探索，沒有 LLM judge 或正式發布。
    '''
    sections = [
        ("0. 連接 Drive／建立新 Super run；第一次 RUN_ID_OVERRIDE 留空", old[0]),
        ("1. 安裝既有 library", old[1]),
        ("2. 載入 hash-bound snapshot 與來源驗證", old[2]),
        ("3. 準備 20 個新 seed ID，排除舊開發集", old[3]),
        ("4. 固定 Super 設定與失敗 audit（無 API）", old[4]),
        ("5. 無申訴資料的 API 測試：改 RUN_PREFLIGHT=True 後執行", preflight),
        ("6. 預檢通過後，改 RUN_GENERATION=True 生成 20 筆", old[5]),
        ("7. 產生 Excel、HTML 與技術報告（所有列仍待審）", old[6]),
        ("8. 初始／完成 Excel 評閱後的摘要（不呼叫 API）", old[7]),
    ]
    notebook["cells"] = [base.cell("markdown", intro)]
    for title, source in sections:
        notebook["cells"].extend([base.cell("markdown", "## " + title), base.cell("code", source)])
    notebook["cells"].append(base.cell("markdown", '''
    ## 恢復與交付

    輸出：`outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override/exploratory_nemotron_super_v02/run_<UTC>/`。
    成功後同步整個 run 與已執行 notebook 回本地；閱讀 `review/generation_review_20.xlsx` 和 `review_readable.html`。
    摘要初次是 pending，不會自動判定合格。填完 Excel 後只重跑第 8 步。
    重連時填原 Super run ID，執行 0–4 後可直接到 8；不需要重新呼叫 API。
    preflight 或 generation 若失敗，保留 run、讀 failure.json；解決根因後用全新 run ID。
    已完成的 raw／preflight 會驗證 hash 後重用。勿把 Nano run ID 當成 Super 執行結果。
    舊閱讀指南仍適用欄位說明，但本版執行順序以本 notebook 的 0–8 為準。
    '''))
    for i, c in enumerate(notebook["cells"]):
        c["id"] = f"super-exploration-{i:02d}"
    notebook["metadata"]["colab"]["name"] = OUTPUT.name
    return notebook


def main():
    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if any(c.get("outputs") or c.get("execution_count") is not None
               for c in previous["cells"] if c["cell_type"] == "code"):
            raise ValueError("Refusing to overwrite an executed notebook")
    notebook = build_notebook()
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            compile(c["source"], "notebook-cell", "exec")
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
