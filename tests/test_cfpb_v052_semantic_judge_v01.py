from __future__ import annotations

import json
import base64
import gzip
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.generation.nemo_data_designer.run_cfpb_v052_blind_semantic_judge_v01 as judge_runner

from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
    sha256_file,
)
from scripts.generation.nemo_data_designer.freeze_cfpb_v052_semantic_judge_v01 import (
    freeze_judge,
)
from scripts.generation.nemo_data_designer.run_cfpb_v052_blind_semantic_judge_v01 import (
    EndpointPreflightSnapshot,
    JudgeCase,
    JudgeResponse,
    PublicEndpoint,
    REQUIRED_ENDPOINT_PARAMETERS,
    RunnerConfig,
    _provider_call,
    fetch_zdr_endpoint_preflight,
    judge_one,
    load_blind_cases,
    render_case,
    render_label_lines,
    utc_now,
    validate_evidence,
    validate_zdr_endpoint_preflight,
)

TEST_TMP = Path(__file__).resolve().parents[1] / "temp/test_cfpb_v052_semantic_judge_v01"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def example_case() -> JudgeCase:
    return JudgeCase(
        seed_id="seed_1",
        labels={
            "product": "Money transfer",
            "sub_product": "Domestic transfer",
            "issue": "Transfer problem",
            "sub_issue": "Wrong amount",
        },
        grounding="The consumer sent a transfer and disputed the amount.",
        generated={
            "conversation": [
                {"role": "user", "content": "I disputed the transfer amount."},
                {"role": "assistant", "content": "Ask the institution to clarify it."},
            ],
            "synthetic_case_summary": "A transfer amount was disputed.",
            "privacy_notes": [],
        },
    )


def write_valid_preflight(path: Path, config: RunnerConfig) -> None:
    snapshot = EndpointPreflightSnapshot(
        source_url="https://openrouter.ai/api/v1/endpoints/zdr",
        retrieved_utc=utc_now(),
        model_id=config.model_name,
        provider_slug=config.provider_slug,
        required_parameters=sorted(REQUIRED_ENDPOINT_PARAMETERS),
        selected_endpoint=PublicEndpoint(
            name="DeepInfra | qwen/qwen3.5-35b-a3b-20260224",
            model_id=config.model_name,
            provider_name="DeepInfra",
            tag="deepinfra/fp8",
            status=0,
            quantization="fp8",
            supported_parameters=sorted(REQUIRED_ENDPOINT_PARAMETERS),
        ),
    )
    path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")


def test_blind_case_loader_exposes_no_oracle_or_candidate_fields() -> None:
    prepared = TEST_TMP / "prepared.jsonl"
    raw = TEST_TMP / "raw.jsonl"
    write_jsonl(
        prepared,
        [
            {
                "seed_id": "seed_1",
                "product": "Money transfer",
                "sub_product": "Domestic transfer",
                "issue": "Transfer problem",
                "sub_issue": "Wrong amount",
                "generation_grounding_excerpt": "The consumer disputed an amount.",
                "candidate_v02_reasons_json": '["claim_type_changed"]',
            }
        ],
    )
    write_jsonl(
        raw,
        [
            {
                "seed_id": "seed_1",
                "dialogue": {
                    "conversation": [],
                    "synthetic_case_summary": "Summary",
                    "privacy_notes": [],
                },
                "expected_decision": "reject",
            }
        ],
    )
    cases = load_blind_cases(raw, prepared)
    dumped = cases[0].model_dump()
    assert set(dumped) == {"seed_id", "labels", "grounding", "generated"}
    assert "candidate_v02_reasons_json" not in json.dumps(dumped)
    assert "expected_decision" not in json.dumps(dumped)


def test_response_contract_rejects_inconsistent_decision() -> None:
    with pytest.raises(ValueError, match="accept requires"):
        JudgeResponse(
            decision="accept",
            reasons=["claim_type_changed"],
            evidence=[{"source": "note", "text": "Conflict"}],
        )
    with pytest.raises(ValueError, match="reject requires"):
        JudgeResponse(decision="reject", reasons=[], evidence=[])


def test_evidence_must_be_exact_for_supplied_sections() -> None:
    case = example_case()
    valid = JudgeResponse(
        decision="reject",
        reasons=["unsupported_procedural_guidance"],
        evidence=[{"source": "generated", "text": "Ask the institution"}],
    )
    validate_evidence(valid, case)
    invalid = JudgeResponse(
        decision="reject",
        reasons=["unsupported_procedural_guidance"],
        evidence=[{"source": "generated", "text": "Call the institution"}],
    )
    with pytest.raises(ValueError, match="not one contiguous exact generated"):
        validate_evidence(invalid, case)


def test_label_evidence_uses_the_same_canonical_text_shown_to_judge() -> None:
    case = example_case()
    label_line = "issue: Transfer problem"
    assert label_line in render_label_lines(case.labels)
    assert label_line in render_case(case)
    valid = JudgeResponse(
        decision="reject",
        reasons=["label_grounding_conflict"],
        evidence=[{"source": "label", "text": label_line}],
    )
    validate_evidence(valid, case)

    compact_json_fragment = '\"issue\": \"Transfer problem\"'
    invalid = JudgeResponse(
        decision="reject",
        reasons=["label_grounding_conflict"],
        evidence=[{"source": "label", "text": compact_json_fragment}],
    )
    with pytest.raises(ValueError, match="not an exact canonical label line"):
        validate_evidence(invalid, case)


def test_judge_one_produces_valid_semantic_judgment() -> None:
    content = json.dumps(
        {"decision": "accept", "reasons": [], "evidence": []}
    )
    response = SimpleNamespace(
        id="response_1",
        created=1,
        model="qwen/qwen3.5-35b-a3b",
        system_fingerprint=None,
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ],
    )
    completions = SimpleNamespace(create=lambda **_: response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    prompt = TEST_TMP / "prompt.md"
    prompt.write_text("rubric", encoding="utf-8")
    judgment, audit = judge_one(
        client=client,
        config=RunnerConfig(max_attempts=1),
        system_prompt="rubric",
        prompt_sha256=sha256_file(prompt),
        case=example_case(),
    )
    assert judgment.decision == "accept"
    assert judgment.reasons == []
    assert judgment.judge_kind == "llm"
    assert audit["content"] == content
    assert audit["attempt_number"] == 1
    assert audit["reasoning"]["present"] is False


def test_nonempty_contract_failure_gets_one_correction_retry() -> None:
    bad_content = json.dumps(
        {
            "decision": "reject",
            "reasons": ["claim_type_changed"],
            "evidence": [
                {
                    "source": "grounding",
                    "text": "The consumer... disputed the amount.",
                }
            ],
        }
    )
    good_content = json.dumps(
        {
            "decision": "reject",
            "reasons": ["claim_type_changed"],
            "evidence": [
                {"source": "grounding", "text": "disputed the amount."}
            ],
        }
    )
    responses = [
        SimpleNamespace(
            id=f"response_{index}",
            created=index,
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
        for index, content in enumerate((bad_content, good_content), start=1)
    ]
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    failures: list[dict] = []
    judgment, audit = judge_one(
        client=client,
        config=RunnerConfig(),
        system_prompt="rubric",
        prompt_sha256="0" * 64,
        case=example_case(),
        on_failed_attempt=failures.append,
    )
    assert judgment.decision == "reject"
    assert audit["attempt_number"] == 2
    assert len(calls) == 2
    assert len(failures) == 1
    assert failures[0]["contract_retry_eligible"] is True
    assert failures[0]["retry_scheduled"] is True
    correction = calls[1]["messages"][-1]["content"]
    assert "one contiguous exact grounding substring" in correction


def test_provider_error_is_not_blindly_retried() -> None:
    calls = 0

    def create(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("transport down")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    failures: list[dict] = []
    with pytest.raises(RuntimeError, match="transport down"):
        judge_one(
            client=client,
            config=RunnerConfig(),
            system_prompt="rubric",
            prompt_sha256="0" * 64,
            case=example_case(),
            on_failed_attempt=failures.append,
        )
    assert calls == 1
    assert len(failures) == 1
    assert failures[0]["contract_retry_eligible"] is False
    assert failures[0]["retry_scheduled"] is False


def test_provider_call_enforces_openrouter_schema_and_zdr_policy() -> None:
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    _provider_call(
        client=client,
        config=RunnerConfig(),
        system_prompt="rubric",
        case=example_case(),
        correction=None,
    )
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["response_format"]["json_schema"]["schema"] == (
        JudgeResponse.model_json_schema()
    )
    provider = captured["extra_body"]["provider"]
    assert provider == {
        "order": ["deepinfra"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }
    assert captured["extra_body"]["reasoning"] == {
        "enabled": False,
        "exclude": True,
    }
    assert captured["extra_body"]["top_k"] == 20
    assert captured["extra_body"]["min_p"] == 0.0
    assert captured["extra_body"]["repetition_penalty"] == 1.0
    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.8
    assert captured["presence_penalty"] == 1.5


def test_default_provider_budget_is_bounded() -> None:
    config = RunnerConfig()
    assert config.request_timeout_seconds == 120.0
    assert config.max_attempts == 2
    assert config.max_tokens == 2048
    assert config.sampling_profile == "qwen3.5_official_instruct_general"
    assert config.temperature == 0.7
    assert config.top_p == 0.8
    assert config.top_k == 20
    assert config.min_p == 0.0
    assert config.presence_penalty == 1.5
    assert config.repetition_penalty == 1.0
    assert config.reasoning_enabled is False
    assert config.reasoning_exclude is True
    assert config.model_name == "qwen/qwen3.5-35b-a3b"
    assert config.base_url == "https://openrouter.ai/api/v1"
    assert config.provider_slug == "deepinfra"
    assert config.allow_fallbacks is False
    assert config.require_parameters is True
    assert config.data_collection == "deny"
    assert config.zdr is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.0),
        ("top_p", 0.9),
        ("top_k", 40),
        ("min_p", 0.1),
        ("presence_penalty", 0.0),
        ("repetition_penalty", 1.1),
        ("reasoning_enabled", True),
    ],
)
def test_runner_config_locks_attempt05_nonthinking_profile(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        RunnerConfig(**{field: value})


def test_failed_empty_response_is_audited_without_content_or_reasoning() -> None:
    response = SimpleNamespace(
        id="response_empty",
        created=2,
        model="qwen/qwen3.5-35b-a3b",
        provider="deepinfra",
        system_fingerprint=None,
        usage={"completion_tokens": 2048},
        choices=[
            SimpleNamespace(
                finish_reason="length",
                model_extra={"native_finish_reason": "MAX_TOKENS"},
                message=SimpleNamespace(
                    content="",
                    reasoning="hidden trace must not be retained",
                    reasoning_details=None,
                ),
            )
        ],
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response)
        )
    )
    failures: list[dict] = []
    with pytest.raises(RuntimeError, match="content is empty"):
        judge_one(
            client=client,
            config=RunnerConfig(),
            system_prompt="rubric",
            prompt_sha256="0" * 64,
            case=example_case(),
            on_failed_attempt=failures.append,
        )
    assert len(failures) == 1
    audit = failures[0]
    assert audit["finish_reason"] == "length"
    assert audit["native_finish_reason"] == "MAX_TOKENS"
    assert audit["usage"] == {"completion_tokens": 2048}
    assert audit["content_present"] is False
    assert "content" not in audit
    assert audit["response_content_retained"] is False
    assert audit["reasoning"]["present"] is True
    assert audit["reasoning"]["chars"] > 0
    assert "hidden trace" not in json.dumps(audit)
    assert audit["reasoning_content_retained"] is False
    assert audit["contract_retry_eligible"] is False
    assert audit["retry_scheduled"] is False


def test_zdr_preflight_selects_deepinfra_with_all_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = TEST_TMP / "zdr_endpoint_preflight.json"
    output.unlink(missing_ok=True)
    payload = {
        "data": [
            {
                "name": "Parasail | qwen/qwen3.5-35b-a3b-20260224",
                "model_id": "qwen/qwen3.5-35b-a3b",
                "provider_name": "Parasail",
                "tag": "parasail/fp8",
                "status": 0,
                "supported_parameters": sorted(
                    REQUIRED_ENDPOINT_PARAMETERS - {"min_p"}
                ),
            },
            {
                "name": "DeepInfra | qwen/qwen3.5-35b-a3b-20260224",
                "model_id": "qwen/qwen3.5-35b-a3b",
                "provider_name": "DeepInfra",
                "tag": "deepinfra/fp8",
                "status": 0,
                "supported_parameters": sorted(REQUIRED_ENDPOINT_PARAMETERS),
            },
        ]
    }
    monkeypatch.setattr(
        judge_runner.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(json.dumps(payload).encode("utf-8")),
    )
    config = RunnerConfig()
    snapshot = fetch_zdr_endpoint_preflight(config=config, output_path=output)
    assert snapshot.selected_endpoint.provider_name == "DeepInfra"
    assert snapshot.selected_endpoint.tag == "deepinfra/fp8"
    assert validate_zdr_endpoint_preflight(output, config=config) == snapshot


def test_zdr_preflight_rejects_parasail_missing_min_p(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "data": [
            {
                "name": "Parasail | qwen/qwen3.5-35b-a3b-20260224",
                "model_id": "qwen/qwen3.5-35b-a3b",
                "provider_name": "Parasail",
                "tag": "parasail/fp8",
                "status": 0,
                "supported_parameters": sorted(
                    REQUIRED_ENDPOINT_PARAMETERS - {"min_p"}
                ),
            }
        ]
    }
    monkeypatch.setattr(
        judge_runner.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(json.dumps(payload).encode("utf-8")),
    )
    with pytest.raises(RuntimeError, match="min_p"):
        fetch_zdr_endpoint_preflight(
            config=RunnerConfig(provider_slug="parasail"),
            output_path=TEST_TMP / "unroutable_parasail_preflight.json",
        )


def test_run_judge_persists_failed_attempt_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = TEST_TMP / "run_judge_failure_audit"
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in (
        "prepared.jsonl",
        "raw.jsonl",
        "prompt.md",
        "failed_attempts.jsonl",
        "semantic_judgments.jsonl",
        "raw_responses.jsonl",
        "judge_run_report.json",
    ):
        (tmp_path / name).unlink(missing_ok=True)
    prepared = tmp_path / "prepared.jsonl"
    raw = tmp_path / "raw.jsonl"
    prompt = tmp_path / "prompt.md"
    failures = tmp_path / "failed_attempts.jsonl"
    write_jsonl(
        prepared,
        [
            {
                "seed_id": "seed_1",
                "product": "Money transfer",
                "sub_product": "Domestic transfer",
                "issue": "Transfer problem",
                "sub_issue": "Wrong amount",
                "generation_grounding_excerpt": "The consumer disputed an amount.",
            }
        ],
    )
    write_jsonl(
        raw,
        [
            {
                "seed_id": "seed_1",
                "dialogue": {
                    "conversation": [],
                    "synthetic_case_summary": "Summary",
                    "privacy_notes": [],
                },
            }
        ],
    )
    prompt.write_text("rubric", encoding="utf-8")
    response = SimpleNamespace(
        id="response_empty",
        created=2,
        model="qwen/qwen3.5-35b-a3b",
        provider="deepinfra",
        system_fingerprint=None,
        usage={"completion_tokens": 2048},
        choices=[
            SimpleNamespace(
                finish_reason="length",
                model_extra={},
                message=SimpleNamespace(
                    content="", reasoning=None, reasoning_details=None
                ),
            )
        ],
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: response)
        )
    )
    monkeypatch.setattr(judge_runner, "make_client", lambda **_: client)
    preflight = tmp_path / "zdr_endpoint_preflight.json"
    config = RunnerConfig()
    write_valid_preflight(preflight, config)
    with pytest.raises(RuntimeError, match="content is empty"):
        judge_runner.run_judge(
            raw_path=raw,
            prepared_path=prepared,
            prompt_path=prompt,
            judgments_path=tmp_path / "semantic_judgments.jsonl",
            raw_responses_path=tmp_path / "raw_responses.jsonl",
            failed_attempts_path=failures,
            endpoint_preflight_path=preflight,
            report_path=tmp_path / "judge_run_report.json",
            config=config,
            api_key="test-only",
        )
    rows = [json.loads(line) for line in failures.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["seed_id"] == "seed_1"
    assert rows[0]["finish_reason"] == "length"
    assert rows[0]["attempt_status"] == "failed"
    assert rows[0]["response_content_retained"] is False
    assert rows[0]["contract_retry_eligible"] is False
    assert rows[0]["retry_scheduled"] is False


def test_run_judge_counts_contract_correction_as_a_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = TEST_TMP / "run_judge_contract_retry_count"
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in (
        "prepared.jsonl",
        "raw.jsonl",
        "prompt.md",
        "failed_attempts.jsonl",
        "semantic_judgments.jsonl",
        "raw_responses.jsonl",
        "judge_run_report.json",
        "zdr_endpoint_preflight.json",
    ):
        (tmp_path / name).unlink(missing_ok=True)
    prepared = tmp_path / "prepared.jsonl"
    raw = tmp_path / "raw.jsonl"
    prompt = tmp_path / "prompt.md"
    write_jsonl(
        prepared,
        [
            {
                "seed_id": "seed_1",
                "product": "Money transfer",
                "sub_product": "Domestic transfer",
                "issue": "Transfer problem",
                "sub_issue": "Wrong amount",
                "generation_grounding_excerpt": (
                    "The consumer sent a transfer and disputed the amount."
                ),
            }
        ],
    )
    write_jsonl(
        raw,
        [
            {
                "seed_id": "seed_1",
                "dialogue": {
                    "conversation": [],
                    "synthetic_case_summary": "Summary",
                    "privacy_notes": [],
                },
            }
        ],
    )
    prompt.write_text("rubric", encoding="utf-8")
    contents = [
        json.dumps(
            {
                "decision": "reject",
                "reasons": ["claim_type_changed"],
                "evidence": [
                    {
                        "source": "grounding",
                        "text": "The consumer... disputed the amount.",
                    }
                ],
            }
        ),
        json.dumps(
            {
                "decision": "reject",
                "reasons": ["claim_type_changed"],
                "evidence": [
                    {"source": "grounding", "text": "disputed the amount."}
                ],
            }
        ),
    ]
    calls = 0

    def create(**_kwargs):
        nonlocal calls
        content = contents[calls]
        calls += 1
        return SimpleNamespace(
            id=f"response_{calls}",
            created=calls,
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

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(judge_runner, "make_client", lambda **_: client)
    config = RunnerConfig()
    preflight = tmp_path / "zdr_endpoint_preflight.json"
    write_valid_preflight(preflight, config)
    report = judge_runner.run_judge(
        raw_path=raw,
        prepared_path=prepared,
        prompt_path=prompt,
        judgments_path=tmp_path / "semantic_judgments.jsonl",
        raw_responses_path=tmp_path / "raw_responses.jsonl",
        failed_attempts_path=tmp_path / "failed_attempts.jsonl",
        endpoint_preflight_path=preflight,
        report_path=tmp_path / "judge_run_report.json",
        config=config,
        api_key="test-only",
    )
    assert calls == 2
    assert report["new_provider_calls"] == 2
    assert report["judgment_rows"] == 1
    assert report["semantic_coverage_complete"] is True


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("allow_fallbacks", True),
        ("require_parameters", False),
        ("data_collection", "allow"),
        ("zdr", False),
    ],
)
def test_runner_config_rejects_unsafe_provider_policy(
    field: str, unsafe_value: object
) -> None:
    with pytest.raises(ValueError):
        RunnerConfig(**{field: unsafe_value})


def test_generated_notebook_embedded_payload_decodes_to_hash_bound_files() -> None:
    notebook_path = (
        PROJECT_ROOT
        / "notebooks/30_generation/nemo_data_designer/seeded_dialogue/canonical"
        / "FinDisputeEval_CFPB_v052_blind_semantic_judge_v01_attempt05_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    materialize_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "embedded = json.loads" in "".join(cell["source"])
    )
    prefix = materialize_cell.split("SNAPSHOT_ROOT.mkdir", maxsplit=1)[0]
    namespace: dict = {"json": json}
    exec(prefix, namespace)
    embedded = namespace["embedded"]
    assert isinstance(embedded, dict)
    assert len(embedded) == 7
    for item in embedded.values():
        raw = gzip.decompress(base64.b64decode(item["payload"]))
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]


def test_freeze_requires_explicit_approval() -> None:
    placeholder = TEST_TMP / "placeholder"
    with pytest.raises(ValueError, match="explicit approve_freeze"):
        freeze_judge(
            judge_report_path=placeholder,
            judgments_path=placeholder,
            evaluation_path=placeholder,
            metrics_path=placeholder,
            prompt_path=placeholder,
            oracle_path=placeholder,
            raw_path=placeholder,
            prepared_path=placeholder,
            runner_path=placeholder,
            common_path=placeholder,
            validator_path=placeholder,
            endpoint_preflight_path=placeholder,
            output_path=TEST_TMP / "frozen.json",
            approved_by="reviewer",
            approve_freeze=False,
        )
