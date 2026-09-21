from __future__ import annotations

import base64
import gzip
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.build_cfpb_v052_semantic_judge_v02_deepseek_attempt01_notebook as notebook_builder
from scripts.generation.nemo_data_designer import (
    run_cfpb_v052_blind_semantic_judge_v01 as v01,
)
from scripts.generation.nemo_data_designer import (
    run_cfpb_v052_blind_semantic_judge_v03 as judge,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def example_case() -> v01.JudgeCase:
    return v01.JudgeCase(
        seed_id="deepseek_case",
        labels={
            "product": "Money transfer",
            "sub_product": "Domestic transfer",
            "issue": "Other service problem",
            "sub_issue": "",
        },
        grounding="The consumer sent the transfer. The institution could not help.",
        generated={
            "conversation": [
                {"role": "user", "content": "I sent the transfer."},
                {"role": "assistant", "content": "Ask for a status update."},
            ],
            "synthetic_case_summary": "A transfer dispute.",
        },
    )


def endpoint(tag: str, *, uptime: float = 99.0) -> v01.PublicEndpoint:
    return v01.PublicEndpoint(
        name=f"Provider: {tag}",
        model_id=judge.DEFAULT_MODEL,
        provider_name=tag,
        tag=tag,
        status=0,
        supported_parameters=sorted(judge.REQUIRED_ENDPOINT_PARAMETERS),
        uptime_last_30m=uptime,
    )


def test_deepseek_profile_is_exact_and_qwen_fields_are_absent() -> None:
    config = judge.RunnerConfig()
    dumped = config.model_dump()
    assert config.model_name == "deepseek/deepseek-v4-flash-0731"
    assert config.temperature == 1.0
    assert config.top_p == 1.0
    assert not {
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "seed",
    }.intersection(dumped)
    with pytest.raises(ValueError, match="official chat sampling"):
        judge.RunnerConfig(temperature=0.0)


def test_provider_call_uses_nonthinking_deepseek_contract_only() -> None:
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    config = judge.RunnerConfig(provider_slug="deepseek")
    judge._provider_call(
        client=client,
        config=config,
        system_prompt="rubric",
        case=example_case(),
        correction=None,
    )
    assert captured["model"] == judge.DEFAULT_MODEL
    assert captured["temperature"] == 1.0
    assert captured["top_p"] == 1.0
    assert captured["extra_body"]["reasoning"] == {
        "enabled": False,
        "exclude": True,
    }
    assert captured["extra_body"]["provider"]["order"] == ["deepseek"]
    assert not {
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "seed",
    }.intersection(captured | captured["extra_body"])
    schema = captured["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"decision", "reasons", "evidence_refs"}


def test_preflight_selects_and_pins_one_exact_zdr_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Priority is intentionally stronger than the volatile uptime field.
    monkeypatch.setattr(
        judge,
        "_load_public_zdr_endpoints",
        lambda _timeout: [endpoint("deepinfra", uptime=100.0), endpoint("deepseek", uptime=90.0)],
    )
    output = PROJECT_ROOT / "temp/deepseek_judge_preflight_unit_test.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        resolved, snapshot = judge.resolve_zdr_endpoint_preflight(
            config=judge.RunnerConfig(provider_slug="auto"),
            output_path=output,
        )
        assert resolved.provider_slug == "deepseek"
        assert snapshot.provider_slug == "deepseek"
        assert snapshot.eligible_endpoint_count == 2
        assert output.is_file()
        validated = judge.validate_zdr_endpoint_preflight(output, config=resolved)
        assert validated.selected_endpoint.tag == "deepseek"
    finally:
        output.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def test_orchestrator_accepts_model_specific_preflight_validator() -> None:
    assert "validate_endpoint_preflight_fn" in inspect.signature(v01.run_judge).parameters


def test_deepseek_notebook_is_independent_hash_bound_and_nonfreezing() -> None:
    notebook = json.loads(notebook_builder.OUTPUT.read_text(encoding="utf-8"))
    all_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "deepseek/deepseek-v4-flash-0731" in all_source
    assert "openrouter_deepseek_v4_flash_0731_v02_attempt01" in all_source
    assert "deepseek_v4_flash_0731_official_chat" in all_source
    assert "temperature=1.0" in all_source
    assert 'provider_slug="auto"' in all_source
    assert "resolve_zdr_endpoint_preflight" in all_source
    assert "freeze_supported_by_this_notebook\": False" in all_source
    assert "freeze_judge(" not in all_source

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
    assert "run_cfpb_v052_blind_semantic_judge_v03.py" in embedded
    assert "cfpb_v052_semantic_judge_v02.md" in embedded
    for item in embedded.values():
        raw = gzip.decompress(base64.b64decode(item["payload"]))
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
