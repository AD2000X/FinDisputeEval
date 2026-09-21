"""Run the evidence-reference judge with pinned DeepSeek V4 Flash 0731.

This module reuses the v01 orchestration/audit layer and v02 evidence-reference
protocol.  It deliberately owns the model-specific sampling and endpoint
contract so Qwen-only parameters are never sent to a DeepSeek endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

try:
    import run_cfpb_v052_blind_semantic_judge_v01 as v01
    import run_cfpb_v052_blind_semantic_judge_v02 as v02
    from cfpb_v052_pipeline_smoke_v02_common import SemanticJudgment, StrictModel
except ModuleNotFoundError:
    from scripts.generation.nemo_data_designer import (
        run_cfpb_v052_blind_semantic_judge_v01 as v01,
    )
    from scripts.generation.nemo_data_designer import (
        run_cfpb_v052_blind_semantic_judge_v02 as v02,
    )
    from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
        SemanticJudgment,
        StrictModel,
    )


PROJECT_ROOT = v01.PROJECT_ROOT
REVALIDATION_ROOT = v01.REVALIDATION_ROOT
DEFAULT_RAW = v01.DEFAULT_RAW
DEFAULT_PREPARED = v01.DEFAULT_PREPARED
DEFAULT_PROMPT = (
    PROJECT_ROOT / "configs/generation/judges/cfpb_v052_semantic_judge_v02.md"
)
DEFAULT_OUTPUT_ROOT = (
    REVALIDATION_ROOT
    / "judges/openrouter_deepseek_v4_flash_0731_v02_attempt01/calibration_20"
)
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_ID = (
    "openrouter_deepseek_v4_flash_0731_zdr_nonthinking_blind_v02_attempt01"
)
DEFAULT_SAMPLING_PROFILE = "deepseek_v4_flash_0731_official_chat"
OPENROUTER_ZDR_ENDPOINTS_URL = "https://openrouter.ai/api/v1/endpoints/zdr"

# Only parameters actually sent by _provider_call belong here.  In particular,
# Qwen-specific top_k, min_p, presence_penalty, and repetition_penalty are absent.
REQUIRED_ENDPOINT_PARAMETERS = frozenset(
    {
        "reasoning",
        "max_tokens",
        "temperature",
        "top_p",
        "response_format",
        "structured_outputs",
    }
)

# Selection occurs only within OpenRouter's public ZDR inventory.  The exact tag
# is then copied into RunnerConfig and persisted, so provider routing is pinned
# for every paid call in the attempt.
DEFAULT_PROVIDER_PRIORITY = (
    "deepseek",
    "atlascloud",
    "akashml",
    "weightsandbiases",
    "digitalocean",
    "alibabacloud",
    "deepinfra",
    "novita",
    "fireworks",
    "parasail",
)


class RunnerConfig(StrictModel):
    model_name: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    judge_id: str = DEFAULT_JUDGE_ID
    sampling_profile: Literal["deepseek_v4_flash_0731_official_chat"] = (
        DEFAULT_SAMPLING_PROFILE
    )
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=256)
    provider_slug: str = "auto"
    allow_fallbacks: Literal[False] = False
    require_parameters: Literal[True] = True
    data_collection: Literal["deny"] = "deny"
    zdr: Literal[True] = True
    reasoning_enabled: Literal[False] = False
    reasoning_exclude: Literal[True] = True
    max_attempts: int = Field(default=2, ge=1, le=2)
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)

    @model_validator(mode="after")
    def validate_model_profile(self) -> "RunnerConfig":
        if self.model_name != DEFAULT_MODEL:
            raise ValueError(f"This runner is pinned to {DEFAULT_MODEL}")
        if self.temperature != 1.0 or self.top_p != 1.0:
            raise ValueError(
                "DeepSeek V4 Flash 0731 official chat sampling requires "
                "temperature=1.0 and top_p=1.0"
            )
        if not self.provider_slug.strip():
            raise ValueError("provider_slug cannot be empty")
        return self


class EndpointPreflightSnapshot(StrictModel):
    preflight_version: Literal["v03"] = "v03"
    source_url: str
    retrieved_utc: str
    model_id: str
    provider_slug: str
    selection_policy: Literal["zdr_priority_then_uptime_v01"]
    provider_priority: list[str]
    required_parameters: list[str]
    eligible_endpoint_count: int = Field(ge=1)
    selected_endpoint: v01.PublicEndpoint
    eligible: Literal[True] = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_rank(tag: str, priority: tuple[str, ...]) -> int:
    base = tag.split("/", maxsplit=1)[0].lower()
    try:
        return priority.index(base)
    except ValueError:
        return len(priority)


def _load_public_zdr_endpoints(timeout_seconds: float) -> list[v01.PublicEndpoint]:
    request = urllib.request.Request(
        OPENROUTER_ZDR_ENDPOINTS_URL,
        headers={"User-Agent": "FinDisputeEval/deepseek-judge-preflight-v03"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("OpenRouter ZDR endpoint response has an invalid schema")
    return [v01.PublicEndpoint.model_validate(row) for row in payload["data"]]


def resolve_zdr_endpoint_preflight(
    *,
    config: RunnerConfig,
    output_path: Path,
    timeout_seconds: float = 30.0,
    provider_priority: tuple[str, ...] = DEFAULT_PROVIDER_PRIORITY,
) -> tuple[RunnerConfig, EndpointPreflightSnapshot]:
    """Select, persist, and return one exact live ZDR provider endpoint."""

    endpoints = _load_public_zdr_endpoints(timeout_seconds)
    candidates = [endpoint for endpoint in endpoints if endpoint.model_id == config.model_name]
    if config.provider_slug != "auto":
        requested = config.provider_slug.lower()
        candidates = [
            endpoint
            for endpoint in candidates
            if endpoint.tag.lower() == requested
            or endpoint.tag.lower().split("/", maxsplit=1)[0] == requested
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
                "tag": endpoint.tag,
                "status": endpoint.status,
                "missing_parameters": sorted(
                    REQUIRED_ENDPOINT_PARAMETERS - set(endpoint.supported_parameters)
                ),
            }
            for endpoint in candidates
        ]
        raise RuntimeError(
            "No live OpenRouter ZDR endpoint satisfies the exact DeepSeek model "
            f"and request parameters: {diagnostics}"
        )

    selected = sorted(
        eligible,
        key=lambda endpoint: (
            _provider_rank(endpoint.tag, provider_priority),
            -(endpoint.uptime_last_30m or 0.0),
            endpoint.tag,
        ),
    )[0]
    resolved_config = config.model_copy(update={"provider_slug": selected.tag})
    snapshot = EndpointPreflightSnapshot(
        source_url=OPENROUTER_ZDR_ENDPOINTS_URL,
        retrieved_utc=_utc_now(),
        model_id=config.model_name,
        provider_slug=selected.tag,
        selection_policy="zdr_priority_then_uptime_v01",
        provider_priority=list(provider_priority),
        required_parameters=sorted(REQUIRED_ENDPOINT_PARAMETERS),
        eligible_endpoint_count=len(eligible),
        selected_endpoint=selected,
    )
    v01.atomic_write_json(output_path, snapshot.model_dump(mode="json"))
    return resolved_config, snapshot


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


def _provider_call(
    *,
    client: Any,
    config: RunnerConfig,
    system_prompt: str,
    case: v01.JudgeCase,
    correction: str | None,
) -> Any:
    user_content = v02.render_case(case)
    if correction:
        user_content += (
            "\n\nYour previous response violated the JSON/reference contract. "
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
    }
    return client.chat.completions.create(
        model=config.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        stream=False,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cfpb_semantic_judgment_v02",
                "strict": True,
                "schema": v02.JudgeResponse.model_json_schema(),
            },
        },
        extra_body=extra_body,
    )


def judge_one(
    *,
    client: Any,
    config: RunnerConfig,
    system_prompt: str,
    prompt_sha256: str,
    case: v01.JudgeCase,
    on_failed_attempt=None,
) -> tuple[SemanticJudgment, dict[str, Any]]:
    correction: str | None = None
    last_error: Exception | None = None
    for attempt_number in range(1, config.max_attempts + 1):
        print(
            f"  provider attempt {attempt_number}/{config.max_attempts} "
            f"(provider={config.provider_slug}, zdr={config.zdr}, "
            f"strict_json_schema=True, evidence_refs=True, reasoning=False, "
            f"timeout={config.request_timeout_seconds}s, max_tokens={config.max_tokens})",
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
            parsed = v02.JudgeResponse.model_validate_json(content)
            evidence = v02.validate_and_resolve_evidence(parsed, case)
        except Exception as exc:
            last_error = exc
            correction = str(exc)[:800]
            eligible = v01._has_nonempty_response_content(response)
            scheduled = attempt_number < config.max_attempts and eligible
            failure = v01._failure_audit(
                seed_id=case.seed_id,
                attempt_number=attempt_number,
                config=config,
                error=exc,
                response=response,
                contract_retry_eligible=eligible,
                retry_scheduled=scheduled,
            )
            if on_failed_attempt is not None:
                on_failed_attempt(failure)
            if scheduled:
                continue
            raise v01.JudgeAttemptError(correction) from exc

        judgment = SemanticJudgment(
            seed_id=case.seed_id,
            reasons=parsed.reasons,
            evidence=evidence,
            decision=parsed.decision,
            judge_kind="llm",
            judge_id=config.judge_id,
            prompt_or_guideline_sha256=prompt_sha256,
            model_name=config.model_name,
            reviewed_utc=v01.utc_now(),
        )
        return judgment, v01._response_audit(
            response,
            seed_id=case.seed_id,
            config=config,
            attempt_number=attempt_number,
        )
    raise v01.JudgeAttemptError(f"Judge failed without a result: {last_error}")


def run_judge(**kwargs) -> dict[str, Any]:
    return v01.run_judge(
        **kwargs,
        judge_one_fn=judge_one,
        report_version="v03",
        evidence_protocol="stable_reference_ids_v01",
        fields_sent=["case_id", "evidence_units"],
        validate_endpoint_preflight_fn=validate_zdr_endpoint_preflight,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--seed-id", action="append", dest="seed_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    config, _snapshot = resolve_zdr_endpoint_preflight(
        config=RunnerConfig(provider_slug=args.provider),
        output_path=args.output_root.parent / "zdr_endpoint_preflight.json",
    )
    report = run_judge(
        raw_path=args.raw,
        prepared_path=args.prepared,
        prompt_path=args.prompt,
        judgments_path=args.output_root / "semantic_judgments.jsonl",
        raw_responses_path=args.output_root / "raw_responses.jsonl",
        failed_attempts_path=args.output_root / "failed_attempts.jsonl",
        endpoint_preflight_path=args.output_root.parent / "zdr_endpoint_preflight.json",
        report_path=args.output_root / "judge_run_report.json",
        config=config,
        api_key=api_key,
        selected_seed_ids=args.seed_ids,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
