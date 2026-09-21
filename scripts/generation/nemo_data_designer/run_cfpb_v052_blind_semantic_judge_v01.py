"""Run the blind CFPB v05.2 semantic judge through OpenRouter.

The runner intentionally has no oracle or candidate-judgment argument. It reads
only immutable raw generations, their prepared grounding/labels, and a frozen
rubric. Oracle scoring is a later, separate validator step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import uuid
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
try:
    from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
        EvidenceSpan,
        SEMANTIC_REASON_CODES,
        SemanticJudgment,
        load_records,
        load_semantic_judgments,
        sha256_file,
        write_jsonl,
    )
except ModuleNotFoundError:  # Direct execution or embedded Colab runtime.
    from cfpb_v052_pipeline_smoke_v02_common import (  # type: ignore[no-redef]
        EvidenceSpan,
        SEMANTIC_REASON_CODES,
        SemanticJudgment,
        load_records,
        load_semantic_judgments,
        sha256_file,
        write_jsonl,
    )


_PROJECT_ROOT_ENV = os.environ.get("FINDISPUTEEVAL_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_PROJECT_ROOT_ENV).resolve()
    if _PROJECT_ROOT_ENV
    else Path(__file__).resolve().parents[3]
)
SMOKE_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override"
RUN_ROOT = SMOKE_ROOT / "run_20260722T135306Z"
REVALIDATION_ROOT = (
    SMOKE_ROOT / "revalidations/validator_v02/source_run_20260722T135306Z"
)
DEFAULT_RAW = RUN_ROOT / "raw/data_designer/dataset/parquet-files/batch_00000.parquet"
DEFAULT_PREPARED = RUN_ROOT / "prepared_inputs/nemo_seed_v052_pipeline_smoke_20.jsonl"
DEFAULT_PROMPT = (
    PROJECT_ROOT / "configs/generation/judges/cfpb_v052_semantic_judge_v01.md"
)
DEFAULT_OUTPUT_ROOT = (
    REVALIDATION_ROOT
    / "judges/openrouter_qwen3_5_35b_a3b_v01_attempt05/calibration_20"
)
DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_ID = (
    "openrouter_qwen3_5_35b_a3b_deepinfra_zdr_nonthinking_blind_v01_attempt05"
)
DEFAULT_PROVIDER = "deepinfra"
DEFAULT_SAMPLING_PROFILE = "qwen3.5_official_instruct_general"
OPENROUTER_ZDR_ENDPOINTS_URL = "https://openrouter.ai/api/v1/endpoints/zdr"
REQUIRED_ENDPOINT_PARAMETERS = frozenset(
    {
        "reasoning",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "seed",
        "response_format",
        "structured_outputs",
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PublicEndpoint(BaseModel):
    """Validated subset of OpenRouter's public endpoint metadata."""

    model_config = ConfigDict(extra="ignore")

    name: str
    model_id: str
    provider_name: str
    tag: str
    status: int
    quantization: str | None = None
    supported_parameters: list[str]
    pricing: dict[str, Any] = Field(default_factory=dict)
    uptime_last_30m: float | None = None


class EndpointPreflightSnapshot(StrictModel):
    preflight_version: Literal["v01"] = "v01"
    source_url: str
    retrieved_utc: str
    model_id: str
    provider_slug: str
    required_parameters: list[str]
    selected_endpoint: PublicEndpoint
    eligible: Literal[True] = True


class JudgeEvidence(StrictModel):
    source: Literal["grounding", "generated", "label", "note"] = Field(
        description=(
            "Evidence section. For label evidence, copy one complete rendered "
            "'key: value' line from the labels array."
        )
    )
    text: str = Field(
        min_length=1,
        max_length=600,
        description=(
            "One contiguous exact quote from the selected supplied section. "
            "Never insert an ellipsis, omit intervening words, paraphrase, or "
            "join non-adjacent spans; use separate evidence items instead. "
            "Only source=note may contain a concise comparison."
        ),
    )


class JudgeResponse(StrictModel):
    decision: Literal["accept", "reject", "review"]
    reasons: list[str] = Field(default_factory=list)
    evidence: list[JudgeEvidence] = Field(default_factory=list)

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, reasons: list[str]) -> list[str]:
        unknown = sorted(set(reasons) - SEMANTIC_REASON_CODES)
        if unknown:
            raise ValueError(f"Unknown semantic reason codes: {unknown}")
        return sorted(set(reasons))

    @model_validator(mode="after")
    def validate_decision_contract(self) -> "JudgeResponse":
        if self.decision == "accept" and self.reasons:
            raise ValueError("accept requires an empty reasons list")
        if self.decision == "reject" and not self.reasons:
            raise ValueError("reject requires at least one reason")
        if self.decision == "reject" and not self.evidence:
            raise ValueError("reject requires at least one evidence item")
        return self


class JudgeCase(StrictModel):
    seed_id: str = Field(min_length=1)
    labels: dict[Literal["product", "sub_product", "issue", "sub_issue"], str]
    grounding: str
    generated: dict[str, Any]


class RunnerConfig(StrictModel):
    model_name: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    judge_id: str = DEFAULT_JUDGE_ID
    sampling_profile: Literal["qwen3.5_official_instruct_general"] = (
        DEFAULT_SAMPLING_PROFILE
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=1.5, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=256)
    provider_slug: str = DEFAULT_PROVIDER
    allow_fallbacks: Literal[False] = False
    require_parameters: Literal[True] = True
    data_collection: Literal["deny"] = "deny"
    zdr: Literal[True] = True
    reasoning_enabled: Literal[False] = False
    reasoning_exclude: Literal[True] = True
    seed: int = 20260722
    max_attempts: int = Field(default=2, ge=1, le=2)
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)

    @model_validator(mode="after")
    def validate_sampling_profile(self) -> "RunnerConfig":
        expected = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError(
                "qwen3.5_official_instruct_general requires "
                "temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, "
                "presence_penalty=1.5, and repetition_penalty=1.0"
            )
        return self


class JudgeAttemptError(RuntimeError):
    """Retryable provider-response or schema-contract failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _provider_name_slug(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def fetch_zdr_endpoint_preflight(
    *,
    config: RunnerConfig,
    output_path: Path,
    timeout_seconds: float = 30.0,
) -> EndpointPreflightSnapshot:
    """Fail closed unless a live ZDR endpoint supports the exact request."""

    request = urllib.request.Request(
        OPENROUTER_ZDR_ENDPOINTS_URL,
        headers={"User-Agent": "FinDisputeEval/semantic-judge-preflight-v01"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("OpenRouter ZDR endpoint response has an invalid schema")

    endpoints = [PublicEndpoint.model_validate(row) for row in payload["data"]]
    candidates = [
        endpoint
        for endpoint in endpoints
        if endpoint.model_id == config.model_name
        and _provider_name_slug(endpoint.provider_name) == config.provider_slug
    ]
    eligible = [
        endpoint
        for endpoint in candidates
        if endpoint.status == 0
        and REQUIRED_ENDPOINT_PARAMETERS.issubset(endpoint.supported_parameters)
    ]
    if not eligible:
        diagnostics = [
            {
                "name": endpoint.name,
                "status": endpoint.status,
                "missing_parameters": sorted(
                    REQUIRED_ENDPOINT_PARAMETERS - set(endpoint.supported_parameters)
                ),
            }
            for endpoint in candidates
        ]
        raise RuntimeError(
            "No live OpenRouter ZDR endpoint satisfies the pinned model, provider, "
            f"and request parameters: {diagnostics}"
        )

    selected = sorted(eligible, key=lambda endpoint: endpoint.tag)[0]
    snapshot = EndpointPreflightSnapshot(
        source_url=OPENROUTER_ZDR_ENDPOINTS_URL,
        retrieved_utc=utc_now(),
        model_id=config.model_name,
        provider_slug=config.provider_slug,
        required_parameters=sorted(REQUIRED_ENDPOINT_PARAMETERS),
        selected_endpoint=selected,
    )
    atomic_write_json(output_path, snapshot.model_dump(mode="json"))
    return snapshot


def validate_zdr_endpoint_preflight(
    path: Path,
    *,
    config: RunnerConfig,
    max_age_hours: float = 24.0,
) -> EndpointPreflightSnapshot:
    if not path.is_file():
        raise FileNotFoundError(path)
    snapshot = EndpointPreflightSnapshot.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    if snapshot.model_id != config.model_name:
        raise ValueError("Endpoint preflight model differs from current config")
    if snapshot.provider_slug != config.provider_slug:
        raise ValueError("Endpoint preflight provider differs from current config")
    endpoint = snapshot.selected_endpoint
    if endpoint.status != 0:
        raise ValueError("Endpoint preflight did not select a live endpoint")
    if not REQUIRED_ENDPOINT_PARAMETERS.issubset(endpoint.supported_parameters):
        raise ValueError("Endpoint preflight lacks required request parameters")
    retrieved = datetime.fromisoformat(snapshot.retrieved_utc)
    age_hours = (datetime.now(timezone.utc) - retrieved).total_seconds() / 3600
    if age_hours < 0 or age_hours > max_age_hours:
        raise ValueError(
            f"Endpoint preflight is stale ({age_hours:.2f} hours); refresh it"
        )
    return snapshot


def parse_dialogue(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Raw dialogue must be a JSON object")


def load_blind_cases(raw_path: Path, prepared_path: Path) -> list[JudgeCase]:
    """Create judge cases without loading oracle or candidate fields."""

    prepared = load_records(prepared_path)
    raw = load_records(raw_path)
    prepared_by_id = {str(row["seed_id"]): row for row in prepared}
    raw_by_id = {str(row["seed_id"]): row for row in raw}
    if len(prepared_by_id) != len(prepared) or len(raw_by_id) != len(raw):
        raise ValueError("Duplicate seed_id in judge inputs")
    if set(prepared_by_id) != set(raw_by_id):
        raise ValueError("Raw/prepared seed coverage mismatch")

    cases: list[JudgeCase] = []
    for seed_id in sorted(prepared_by_id):
        seed = prepared_by_id[seed_id]
        cases.append(
            JudgeCase(
                seed_id=seed_id,
                labels={
                    name: str(seed.get(name, ""))
                    for name in ("product", "sub_product", "issue", "sub_issue")
                },
                grounding=str(seed.get("generation_grounding_excerpt", "")),
                generated=parse_dialogue(raw_by_id[seed_id].get("dialogue")),
            )
        )
    return cases


LABEL_FIELDS = ("product", "sub_product", "issue", "sub_issue")


def render_label_lines(
    labels: dict[Literal["product", "sub_product", "issue", "sub_issue"], str],
) -> list[str]:
    """Return the single canonical label representation shown and validated."""

    lines: list[str] = []
    for name in LABEL_FIELDS:
        value = labels[name]
        if "\n" in value or "\r" in value:
            raise ValueError(f"Label value contains a line break: {name}")
        lines.append(f"{name}: {value}")
    return lines


def render_case(case: JudgeCase) -> str:
    payload = {
        "case_id": case.seed_id,
        "labels": render_label_lines(case.labels),
        "grounding_excerpt": case.grounding,
        "generated_dialogue": case.generated,
    }
    return (
        "Evaluate this single case under the system rubric. Return JSON only.\n"
        "For source=label evidence, copy one complete string exactly from the "
        "labels array, including its key prefix. For source=grounding or "
        "source=generated, text must be one contiguous exact substring: never "
        "insert an ellipsis, omit intervening words, paraphrase, or join "
        "non-adjacent spans. Use multiple evidence objects for multiple spans.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


def validate_evidence(response: JudgeResponse, case: JudgeCase) -> None:
    label_lines = render_label_lines(case.labels)
    sources = {
        "grounding": case.grounding,
        "generated": json.dumps(case.generated, ensure_ascii=False, sort_keys=True),
    }
    for item in response.evidence:
        if item.source == "note":
            continue
        if item.source == "label":
            if item.text not in label_lines:
                raise ValueError(
                    f"Evidence is not an exact canonical label line: {item.text!r}"
                )
            continue
        if item.text not in sources[item.source]:
            raise ValueError(
                f"Evidence is not one contiguous exact {item.source} substring; "
                "do not insert ellipses, omit words, paraphrase, or join "
                f"non-adjacent spans: {item.text!r}"
            )


def make_client(*, api_key: str, config: RunnerConfig) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Install openai before running the judge") from exc
    return OpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.request_timeout_seconds,
        # Do not multiply the explicit schema-attempt loop by hidden SDK retries.
        # A provider failure is surfaced within the configured timeout.
        max_retries=0,
    )


def _provider_call(
    *,
    client: Any,
    config: RunnerConfig,
    system_prompt: str,
    case: JudgeCase,
    correction: str | None,
) -> Any:
    user_content = render_case(case)
    if correction:
        user_content += (
            "\n\nYour previous response violated the JSON/schema contract. "
            f"Correct it without changing the case: {correction}"
        )
    extra_body: dict[str, Any] = {
        "provider": {
            "order": [config.provider_slug],
            "allow_fallbacks": config.allow_fallbacks,
            "require_parameters": config.require_parameters,
            "data_collection": config.data_collection,
            "zdr": config.zdr,
        },
        "reasoning": {
            "enabled": config.reasoning_enabled,
            "exclude": config.reasoning_exclude,
        },
        # Qwen's official instruct/non-thinking profile. These OpenRouter
        # parameters are sent explicitly so the run report can bind them.
        "top_k": config.top_k,
        "min_p": config.min_p,
        "repetition_penalty": config.repetition_penalty,
    }
    kwargs: dict[str, Any] = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "presence_penalty": config.presence_penalty,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "cfpb_semantic_judgment",
                "strict": True,
                "schema": JudgeResponse.model_json_schema(),
            },
        },
        "extra_body": extra_body,
    }
    return client.chat.completions.create(**kwargs)


def _dump_optional(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return str(value)


def _nested_value(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _fingerprint_hidden(value: Any) -> dict[str, Any]:
    """Record presence/size/hash without retaining hidden reasoning content."""

    if value is None:
        return {"present": False, "chars": 0, "sha256": None}
    serialized = (
        value
        if isinstance(value, str)
        else json.dumps(_dump_optional(value), ensure_ascii=False, sort_keys=True)
    )
    return {
        "present": bool(serialized),
        "chars": len(serialized),
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _response_audit(
    response: Any,
    *,
    seed_id: str,
    config: RunnerConfig,
    attempt_number: int,
) -> dict[str, Any]:
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    usage = getattr(response, "usage", None)
    response_extra = getattr(response, "model_extra", None) or {}
    choice_extra = getattr(choice, "model_extra", None) or {}
    return {
        "seed_id": seed_id,
        "attempt_number": attempt_number,
        "response_id": getattr(response, "id", None),
        "created": getattr(response, "created", None),
        "model": getattr(response, "model", None),
        "provider": getattr(response, "provider", None)
        or _nested_value(response_extra, "provider"),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "finish_reason": getattr(choice, "finish_reason", None),
        "native_finish_reason": getattr(choice, "native_finish_reason", None)
        or getattr(response, "native_finish_reason", None)
        or _nested_value(choice_extra, "native_finish_reason")
        or _nested_value(response_extra, "native_finish_reason"),
        "content": getattr(message, "content", None),
        "usage": _dump_optional(usage),
        "reasoning": _fingerprint_hidden(getattr(message, "reasoning", None)),
        "reasoning_details": _fingerprint_hidden(
            getattr(message, "reasoning_details", None)
        ),
        "structured_output": True,
        "requested_provider": config.provider_slug,
        "zdr_required": config.zdr,
    }


def _failure_audit(
    *,
    seed_id: str,
    attempt_number: int,
    config: RunnerConfig,
    error: Exception,
    response: Any | None,
    contract_retry_eligible: bool,
    retry_scheduled: bool,
) -> dict[str, Any]:
    if response is None:
        base: dict[str, Any] = {
            "seed_id": seed_id,
            "attempt_number": attempt_number,
            "response_id": None,
            "created": None,
            "model": None,
            "provider": None,
            "system_fingerprint": None,
            "finish_reason": None,
            "native_finish_reason": None,
            "usage": None,
            "reasoning": _fingerprint_hidden(None),
            "reasoning_details": _fingerprint_hidden(None),
            "structured_output": True,
            "requested_provider": config.provider_slug,
            "zdr_required": config.zdr,
        }
        content = None
    else:
        base = _response_audit(
            response,
            seed_id=seed_id,
            config=config,
            attempt_number=attempt_number,
        )
        content = base.pop("content", None)
    content_fingerprint = _fingerprint_hidden(content)
    return {
        **base,
        "audit_event_id": str(uuid.uuid4()),
        "attempted_utc": utc_now(),
        "attempt_status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error)[:800],
        "error_status_code": getattr(error, "status_code", None),
        "error_code": getattr(error, "code", None),
        "error_request_id": getattr(error, "request_id", None),
        "content_present": content_fingerprint["present"],
        "content_chars": content_fingerprint["chars"],
        "content_sha256": content_fingerprint["sha256"],
        "response_content_retained": False,
        "reasoning_content_retained": False,
        "contract_retry_eligible": contract_retry_eligible,
        "retry_scheduled": retry_scheduled,
        "request_config": config.model_dump(mode="json"),
    }


def _has_nonempty_response_content(response: Any | None) -> bool:
    """Return whether a provider response can support a contract correction."""

    if response is None:
        return False
    choices = getattr(response, "choices", None) or []
    if not choices:
        return False
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return isinstance(content, str) and bool(content.strip())


def judge_one(
    *,
    client: Any,
    config: RunnerConfig,
    system_prompt: str,
    prompt_sha256: str,
    case: JudgeCase,
    on_failed_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[SemanticJudgment, dict[str, Any]]:
    correction: str | None = None
    last_error: Exception | None = None
    for attempt_number in range(1, config.max_attempts + 1):
        print(
            f"  provider attempt {attempt_number}/{config.max_attempts} "
            f"(provider={config.provider_slug}, zdr={config.zdr}, "
            f"strict_json_schema=True, timeout={config.request_timeout_seconds}s, "
            f"max_tokens={config.max_tokens})",
            flush=True,
        )
        response = None
        try:
            response = _provider_call(
                client=client,
                config=config,
                system_prompt=system_prompt,
                case=case,
                correction=correction,
            )
            if not response.choices:
                raise ValueError("Provider response has no choices")
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Provider response content is empty")
            parsed = JudgeResponse.model_validate_json(content)
            validate_evidence(parsed, case)
        except Exception as exc:
            last_error = exc
            correction = str(exc)[:800]
            contract_retry_eligible = _has_nonempty_response_content(response)
            retry_scheduled = (
                attempt_number < config.max_attempts and contract_retry_eligible
            )
            failure = _failure_audit(
                seed_id=case.seed_id,
                attempt_number=attempt_number,
                config=config,
                error=exc,
                response=response,
                contract_retry_eligible=contract_retry_eligible,
                retry_scheduled=retry_scheduled,
            )
            if on_failed_attempt is not None:
                on_failed_attempt(failure)
            if retry_scheduled:
                continue
            raise JudgeAttemptError(correction) from exc

        judgment = SemanticJudgment(
            seed_id=case.seed_id,
            reasons=parsed.reasons,
            evidence=[EvidenceSpan(**item.model_dump()) for item in parsed.evidence],
            decision=parsed.decision,
            judge_kind="llm",
            judge_id=config.judge_id,
            prompt_or_guideline_sha256=prompt_sha256,
            model_name=config.model_name,
            reviewed_utc=utc_now(),
        )
        return judgment, _response_audit(
            response,
            seed_id=case.seed_id,
            config=config,
            attempt_number=attempt_number,
        )
    raise JudgeAttemptError(f"Judge failed without a result: {last_error}")


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def _load_audit_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_records(path)
    result = {str(row["seed_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate seed_id in raw-response audit")
    return result


def run_judge(
    *,
    raw_path: Path,
    prepared_path: Path,
    prompt_path: Path,
    judgments_path: Path,
    raw_responses_path: Path,
    failed_attempts_path: Path,
    endpoint_preflight_path: Path,
    report_path: Path,
    config: RunnerConfig,
    api_key: str,
    selected_seed_ids: list[str] | None = None,
    judge_one_fn: Callable[..., tuple[SemanticJudgment, dict[str, Any]]] | None = None,
    report_version: str = "v01",
    evidence_protocol: str = "free_text_exact_quote_v01",
    fields_sent: list[str] | None = None,
    validate_endpoint_preflight_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    endpoint_preflight = (
        validate_endpoint_preflight_fn or validate_zdr_endpoint_preflight
    )(
        endpoint_preflight_path,
        config=config,
    )
    system_prompt = prompt_path.read_text(encoding="utf-8")
    prompt_sha256 = sha256_file(prompt_path)
    cases = load_blind_cases(raw_path, prepared_path)
    if selected_seed_ids:
        wanted = set(selected_seed_ids)
        known = {case.seed_id for case in cases}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"Unknown selected seed IDs: {unknown}")
        cases = [case for case in cases if case.seed_id in wanted]

    existing = (
        load_semantic_judgments(judgments_path) if judgments_path.exists() else {}
    )
    expected_ids = {case.seed_id for case in cases}
    unexpected = sorted(set(existing) - expected_ids)
    if unexpected:
        raise ValueError(f"Cached judgments are outside this run: {unexpected}")
    for judgment in existing.values():
        if (
            judgment.judge_kind != "llm"
            or judgment.judge_id != config.judge_id
            or judgment.model_name != config.model_name
            or judgment.prompt_or_guideline_sha256 != prompt_sha256
        ):
            raise ValueError("Cached judgment provenance differs from current judge")

    audit_by_id = _load_audit_rows(raw_responses_path)
    unexpected_audit = sorted(set(audit_by_id) - expected_ids)
    if unexpected_audit:
        raise ValueError(f"Raw-response audit is outside this run: {unexpected_audit}")
    missing_cached_audit = sorted(set(existing) - set(audit_by_id))
    if missing_cached_audit:
        raise ValueError(
            f"Cached judgments lack raw-response audit rows: {missing_cached_audit}"
        )
    failed_attempt_rows = (
        load_records(failed_attempts_path) if failed_attempts_path.exists() else []
    )
    unexpected_failures = sorted(
        {str(row["seed_id"]) for row in failed_attempt_rows} - expected_ids
    )
    if unexpected_failures:
        raise ValueError(
            f"Failed-attempt audit is outside this run: {unexpected_failures}"
        )

    def persist_failed_attempt(row: dict[str, Any]) -> None:
        failed_attempt_rows.append(row)
        atomic_write_jsonl(failed_attempts_path, failed_attempt_rows)

    client = make_client(api_key=api_key, config=config)
    started = time.monotonic()
    new_calls = 0
    for index, case in enumerate(cases, start=1):
        if case.seed_id in existing:
            print(f"[{index}/{len(cases)}] cache hit {case.seed_id}")
            continue
        print(f"[{index}/{len(cases)}] judging {case.seed_id}")
        judgment, audit = (judge_one_fn or judge_one)(
            client=client,
            config=config,
            system_prompt=system_prompt,
            prompt_sha256=prompt_sha256,
            case=case,
            on_failed_attempt=persist_failed_attempt,
        )
        existing[case.seed_id] = judgment
        audit_by_id[case.seed_id] = audit
        atomic_write_jsonl(
            raw_responses_path,
            [audit_by_id[key] for key in sorted(audit_by_id)],
        )
        atomic_write_jsonl(
            judgments_path,
            [existing[key].model_dump(mode="json") for key in sorted(existing)],
        )
        # attempt_number is the actual number of provider calls made for this
        # newly completed case, including one eligible contract correction.
        new_calls += int(audit["attempt_number"])

    decisions: dict[str, int] = {"accept": 0, "reject": 0, "review": 0}
    for judgment in existing.values():
        decisions[judgment.decision] += 1
    report = {
        "report_version": report_version,
        "purpose": "blind_llm_judge_development_calibration",
        "created_utc": utc_now(),
        "blind_input_contract": {
            "loaded_oracle": False,
            "loaded_candidate_judgments": False,
            "fields_sent": fields_sent
            or ["case_id", "labels", "grounding_excerpt", "generated_dialogue"],
            "evidence_protocol": evidence_protocol,
        },
        "config": config.model_dump(mode="json"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("openai", "pydantic", "pandas", "pyarrow")
            },
        },
        "inputs": {
            "raw_path": str(raw_path),
            "raw_sha256": sha256_file(raw_path),
            "prepared_path": str(prepared_path),
            "prepared_sha256": sha256_file(prepared_path),
            "prompt_path": str(prompt_path),
            "prompt_sha256": prompt_sha256,
            "endpoint_preflight_path": str(endpoint_preflight_path),
            "endpoint_preflight_sha256": sha256_file(endpoint_preflight_path),
            "selected_endpoint": endpoint_preflight.selected_endpoint.model_dump(
                mode="json"
            ),
        },
        "selected_seed_ids": sorted(expected_ids),
        "expected_rows": len(cases),
        "judgment_rows": len(existing),
        "semantic_coverage_complete": set(existing) == expected_ids,
        "new_provider_calls": new_calls,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "decision_counts": decisions,
        "outputs": {
            "judgments": {
                "path": str(judgments_path),
                "sha256": sha256_file(judgments_path),
            },
            "raw_responses": {
                "path": str(raw_responses_path),
                "sha256": sha256_file(raw_responses_path),
            },
            "failed_attempts": {
                "path": str(failed_attempts_path),
                "rows": len(failed_attempt_rows),
                "sha256": (
                    sha256_file(failed_attempts_path)
                    if failed_attempts_path.exists()
                    else None
                ),
            },
        },
        "policy": {
            "development_calibration_only": True,
            "human_gold": False,
            "privacy_verified": False,
            "benchmark_eligible": False,
            "formal_pilot_allowed": False,
            "openrouter_zdr_required": True,
            "provider_fallbacks_allowed": False,
            "provider_data_collection_allowed": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge-id", default=DEFAULT_JUDGE_ID)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--seed-id", action="append", dest="seed_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    output_root = args.output_root
    config = RunnerConfig(
        model_name=args.model,
        base_url=args.base_url,
        judge_id=args.judge_id,
        max_attempts=args.max_attempts,
    )
    endpoint_preflight_path = output_root / "zdr_endpoint_preflight.json"
    fetch_zdr_endpoint_preflight(
        config=config,
        output_path=endpoint_preflight_path,
    )
    report = run_judge(
        raw_path=args.raw,
        prepared_path=args.prepared,
        prompt_path=args.prompt,
        judgments_path=output_root / "semantic_judgments.jsonl",
        raw_responses_path=output_root / "raw_responses.jsonl",
        failed_attempts_path=output_root / "failed_attempts.jsonl",
        endpoint_preflight_path=endpoint_preflight_path,
        report_path=output_root / "judge_run_report.json",
        config=config,
        api_key=api_key,
        selected_seed_ids=args.seed_ids,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
