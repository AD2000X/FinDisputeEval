import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.build_cfpb_v052_nemotron_super_exploration_notebook import build_notebook
from scripts.generation.nemo_data_designer import cfpb_v052_nemotron_exploration_v01 as workflow
from scripts.generation.nemo_data_designer import validate_cfpb_v052_pipeline_smoke_v02 as validator


def codes():
    return [c["source"] for c in build_notebook()["cells"] if c["cell_type"] == "code"]


def context(tmp_path):
    ns = dict(RUN_ROOT=tmp_path, workflow=workflow, json=json, os=os, re=re,
              MODEL="nvidia/nemotron-3-super-120b-a12b")
    tree = ast.parse(codes()[4])
    # Load audit/gate helpers without loading Data Designer or calling any API.
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) or
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
                 t.id in {"PREFLIGHT_ROOT", "PREFLIGHT_REPORT"} for t in n.targets))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), "helpers", "exec"), ns)
    workflow.save_json(tmp_path / "run_spec.json", {"model": ns["MODEL"]})
    return ns


def test_notebook_pins_super_and_has_two_disabled_api_gates():
    cells = codes()
    assert len(cells) == 9
    for c in cells:
        ast.parse(c)
    assert 'MODEL = "nvidia/nemotron-3-super-120b-a12b"' in cells[4]
    assert '"enable_thinking": False' in cells[4]
    assert '"top_p": 0.95' in cells[4]
    assert '"max_tokens": 4096' in cells[4]
    assert "exploratory_nemotron_super_v02" in cells[0]
    assert "RUN_PREFLIGHT = False" in cells[5]
    assert "RUN_GENERATION = False" in cells[6]
    assert cells[6].index("require_passed_preflight()") < cells[6].index("generation_started.json")
    assert "skip_health_check=True" not in "".join(cells)
    assert "PREPARED" not in cells[5] and "generation_grounding_excerpt" not in cells[5]


def test_audit_recovers_http_410_and_masks_secret(tmp_path, monkeypatch):
    ns = context(tmp_path)
    monkeypatch.setenv("NVIDIA_API_KEY", "private-test-key")
    inner = RuntimeError("model retired Bearer private-test-key")
    inner.status_code = 410
    outer = RuntimeError("wrapped")
    outer.__context__ = inner
    path = tmp_path / "failure.json"
    ns["save_failure_audit"](outer, path, "preflight")
    report = json.loads(path.read_text())
    assert report["errors"][1]["status_code"] == 410
    assert "private-test-key" not in path.read_text()


def test_failed_preflight_cannot_unlock_generation(tmp_path):
    ns = context(tmp_path)
    with pytest.raises(RuntimeError):
        ns["require_passed_preflight"]()
    workflow.save_json(ns["PREFLIGHT_REPORT"], {"passed": False})
    with pytest.raises(ValueError):
        ns["require_passed_preflight"]()


def test_preflight_disabled_does_not_submit(tmp_path):
    ns = context(tmp_path)
    with pytest.raises(RuntimeError, match="RUN_PREFLIGHT=True"):
        exec(codes()[5], ns)
    assert not (ns["PREFLIGHT_ROOT"] / "started.json").exists()


def test_mock_sdk_preflight_cache_and_binding(tmp_path, monkeypatch):
    ns = context(tmp_path)
    calls = []
    class Builder:
        def __init__(self, **kwargs):
            pass
        def add_column(self, c):
            self.column = c
        def build(self):
            return SimpleNamespace(model_dump=lambda **kw: {"test": "invented text"})
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": "Fictional test content."} for i in range(4)]
    response = {"conversation": messages, "synthetic_case_summary": "Invented test", "privacy_notes": []}
    class Engine:
        def __init__(self, **kwargs):
            pass
        def preview(self, probe, num_records):
            calls.append(probe.column)
            return SimpleNamespace(dataset=pd.DataFrame([{"dialogue": response}]))
    monkeypatch.setitem(sys.modules, "data_designer.interface", SimpleNamespace(DataDesigner=Engine))
    monkeypatch.setitem(sys.modules, "validate_cfpb_v052_pipeline_smoke_v02", validator)
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-key-no-network")
    ns.update(dd=SimpleNamespace(DataDesignerConfigBuilder=Builder, LLMStructuredColumnConfig=lambda **kw: kw),
              generation=SimpleNamespace(PipelineSmokeConversation=validator.Dialogue),
              builder=SimpleNamespace(model_configs=[]), MODEL_ALIAS="test", PROVIDER="test")
    active = codes()[5].replace("RUN_PREFLIGHT = False", "RUN_PREFLIGHT = True")
    exec(active, ns)
    assert len(calls) == 1 and ns["require_passed_preflight"]()["complaint_data_submitted"] is False
    exec(active, ns)
    assert len(calls) == 1
    (tmp_path / "run_spec.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="different run"):
        ns["require_passed_preflight"]()
