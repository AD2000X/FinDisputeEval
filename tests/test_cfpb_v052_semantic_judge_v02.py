from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.generation.nemo_data_designer.run_cfpb_v052_blind_semantic_judge_v02 as judge
from scripts.generation.nemo_data_designer.run_cfpb_v052_blind_semantic_judge_v01 import (
    JudgeCase,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def example_case() -> JudgeCase:
    return JudgeCase(
        seed_id="seed_v02",
        labels={
            "product": "Money transfer",
            "sub_product": "Domestic transfer",
            "issue": "Other service problem",
            "sub_issue": "",
        },
        grounding=(
            "The consumer sent the transfer. The institution said it could not help."
        ),
        generated={
            "conversation": [
                {"role": "user", "content": "I sent the transfer."},
                {
                    "role": "assistant",
                    "content": "Contact the institution’s customer service.",
                },
            ],
            "synthetic_case_summary": "A transfer dispute.",
            "privacy_notes": ["No identifiers."],
        },
    )


def provider_response(content: str, response_id: str = "response_v02"):
    return SimpleNamespace(
        id=response_id,
        created=1,
        model="qwen/qwen3.5-35b-a3b",
        provider="deepinfra",
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ],
    )


def test_evidence_units_are_exact_and_stably_numbered() -> None:
    case = example_case()
    units = judge.build_evidence_units(case)
    by_id = {unit.ref_id: unit for unit in units}
    assert len(by_id) == len(units)
    assert by_id["L_ISSUE"].text == "issue: Other service problem"
    grounding = [unit for unit in units if unit.source == "grounding"]
    assert [unit.ref_id for unit in grounding] == ["G001", "G002"]
    for unit in grounding:
        assert case.grounding[unit.start : unit.end] == unit.text
    generated = [unit for unit in units if unit.source == "generated"]
    assert all(unit.text not in {"user", "assistant"} for unit in generated)
    assert any("institution’s" in unit.text for unit in generated)


def test_reference_resolution_preserves_source_unicode_exactly() -> None:
    case = example_case()
    target = next(
        unit
        for unit in judge.build_evidence_units(case)
        if "institution’s" in unit.text
    )
    response = judge.JudgeResponse(
        decision="reject",
        reasons=["unsupported_scenario_detail"],
        evidence_refs=[target.ref_id],
    )
    resolved = judge.validate_and_resolve_evidence(response, case)
    assert len(resolved) == 1
    assert resolved[0].source == "generated"
    assert resolved[0].text == "Contact the institution’s customer service."


def test_unknown_reference_fails_closed() -> None:
    response = judge.JudgeResponse(
        decision="reject",
        reasons=["unsupported_scenario_detail"],
        evidence_refs=["D999"],
    )
    with pytest.raises(ValueError, match="Unknown evidence reference IDs"):
        judge.validate_and_resolve_evidence(response, example_case())


def test_response_contract_has_no_free_text_evidence_field() -> None:
    schema = judge.JudgeResponse.model_json_schema()
    properties = schema["properties"]
    assert set(properties) == {"decision", "reasons", "evidence_refs"}
    with pytest.raises(ValueError, match="accept requires an empty evidence_refs"):
        judge.JudgeResponse(
            decision="accept", reasons=[], evidence_refs=["D001"]
        )


def test_provider_call_uses_reference_schema_and_units() -> None:
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    judge._provider_call(
        client=client,
        config=judge.RunnerConfig(),
        system_prompt="rubric",
        case=example_case(),
        correction=None,
    )
    schema = captured["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"decision", "reasons", "evidence_refs"}
    user_content = captured["messages"][-1]["content"]
    assert '"evidence_units"' in user_content
    assert '"ref_id": "G001"' in user_content
    assert "candidate_v02_reasons" not in user_content


def test_invalid_ref_gets_one_correction_then_resolves() -> None:
    good_ref = next(
        unit.ref_id
        for unit in judge.build_evidence_units(example_case())
        if "institution’s" in unit.text
    )
    contents = [
        json.dumps(
            {
                "decision": "reject",
                "reasons": ["unsupported_scenario_detail"],
                "evidence_refs": ["D999"],
            }
        ),
        json.dumps(
            {
                "decision": "reject",
                "reasons": ["unsupported_scenario_detail"],
                "evidence_refs": [good_ref],
            }
        ),
    ]
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return provider_response(contents[len(calls) - 1], f"response_{len(calls)}")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    failures: list[dict] = []
    judgment, audit = judge.judge_one(
        client=client,
        config=judge.RunnerConfig(),
        system_prompt="rubric",
        prompt_sha256="0" * 64,
        case=example_case(),
        on_failed_attempt=failures.append,
    )
    assert audit["attempt_number"] == 2
    assert judgment.evidence[0].text == (
        "Contact the institution’s customer service."
    )
    assert len(calls) == 2
    assert failures[0]["contract_retry_eligible"] is True
    assert failures[0]["retry_scheduled"] is True
    assert "Unknown evidence reference IDs" in calls[1]["messages"][-1]["content"]


def test_provider_error_is_not_retried() -> None:
    calls = 0

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider unavailable")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    failures: list[dict] = []
    with pytest.raises(RuntimeError, match="provider unavailable"):
        judge.judge_one(
            client=client,
            config=judge.RunnerConfig(),
            system_prompt="rubric",
            prompt_sha256="0" * 64,
            case=example_case(),
            on_failed_attempt=failures.append,
        )
    assert calls == 1
    assert failures[0]["contract_retry_eligible"] is False
    assert failures[0]["retry_scheduled"] is False


def test_generated_notebook_embedded_payload_is_hash_bound() -> None:
    notebook_path = (
        PROJECT_ROOT
        / "notebooks/30_generation/nemo_data_designer/seeded_dialogue/canonical"
        / "FinDisputeEval_CFPB_v052_blind_semantic_judge_v02_attempt01_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    materialize = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "embedded = json.loads" in "".join(cell["source"])
    )
    prefix = materialize.split("SNAPSHOT_ROOT.mkdir", maxsplit=1)[0]
    namespace: dict = {"json": json}
    exec(prefix, namespace)
    embedded = namespace["embedded"]
    assert len(embedded) == 8
    for item in embedded.values():
        raw = gzip.decompress(base64.b64decode(item["payload"]))
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
