"""Run CFPB v05.2 blind semantic judge v02 with evidence reference IDs.

The model never retypes evidence. The runner presents immutable, numbered source
units and resolves returned IDs back to exact source text before persisting the
standard SemanticJudgment contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

try:
    import run_cfpb_v052_blind_semantic_judge_v01 as v01
    from cfpb_v052_pipeline_smoke_v02_common import (
        EvidenceSpan,
        SEMANTIC_REASON_CODES,
        SemanticJudgment,
        StrictModel,
        sha256_file,
    )
except ModuleNotFoundError:
    from scripts.generation.nemo_data_designer import (
        run_cfpb_v052_blind_semantic_judge_v01 as v01,
    )
    from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
        EvidenceSpan,
        SEMANTIC_REASON_CODES,
        SemanticJudgment,
        StrictModel,
        sha256_file,
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
    / "judges/openrouter_qwen3_5_35b_a3b_v02_attempt01/calibration_20"
)
DEFAULT_JUDGE_ID = (
    "openrouter_qwen3_5_35b_a3b_deepinfra_zdr_nonthinking_blind_v02_attempt01"
)


class RunnerConfig(v01.RunnerConfig):
    judge_id: str = DEFAULT_JUDGE_ID


class EvidenceUnit(StrictModel):
    ref_id: str = Field(pattern=r"^(?:L_[A-Z_]+|G\d{3}|D\d{3})$")
    source: Literal["label", "grounding", "generated"]
    location: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidenceUnit":
        if (self.start is None) != (self.end is None):
            raise ValueError("Evidence-unit start/end must be both present or absent")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("Evidence-unit end must be greater than start")
        return self


class JudgeResponse(StrictModel):
    decision: Literal["accept", "reject", "review"]
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(
        default_factory=list,
        description=(
            "IDs copied exactly from supplied evidence_units. Use multiple IDs "
            "for comparisons; never return evidence text."
        ),
    )

    @field_validator("reasons")
    @classmethod
    def validate_reasons(cls, reasons: list[str]) -> list[str]:
        unknown = sorted(set(reasons) - SEMANTIC_REASON_CODES)
        if unknown:
            raise ValueError(f"Unknown semantic reason codes: {unknown}")
        return sorted(set(reasons))

    @field_validator("evidence_refs")
    @classmethod
    def deduplicate_refs(cls, refs: list[str]) -> list[str]:
        return list(dict.fromkeys(refs))

    @model_validator(mode="after")
    def validate_decision_contract(self) -> "JudgeResponse":
        if self.decision == "accept" and self.reasons:
            raise ValueError("accept requires an empty reasons list")
        if self.decision == "accept" and self.evidence_refs:
            raise ValueError("accept requires an empty evidence_refs list")
        if self.decision == "reject" and not self.reasons:
            raise ValueError("reject requires at least one reason")
        if self.decision == "reject" and not self.evidence_refs:
            raise ValueError("reject requires at least one evidence ref")
        return self


def _segment_grounding(text: str) -> list[str]:
    """Segment English text with PySBD while preserving exact source spans."""

    try:
        import pysbd
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Install pysbd==0.3.4 before running judge v02") from exc
    segments = pysbd.Segmenter(language="en", clean=False).segment(text)
    return [segment.strip() for segment in segments if segment.strip()]


def _walk_generated_strings(value: Any, path: str = "generated_dialogue"):
    if isinstance(value, str):
        if value:
            yield path, value
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_generated_strings(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key in sorted(value):
            if key == "role":
                continue
            yield from _walk_generated_strings(value[key], f"{path}.{key}")


def build_evidence_units(case: v01.JudgeCase) -> list[EvidenceUnit]:
    units: list[EvidenceUnit] = []
    for name in v01.LABEL_FIELDS:
        units.append(
            EvidenceUnit(
                ref_id=f"L_{name.upper()}",
                source="label",
                location=f"labels.{name}",
                text=f"{name}: {case.labels[name]}",
            )
        )

    cursor = 0
    for index, segment in enumerate(_segment_grounding(case.grounding), start=1):
        start = case.grounding.find(segment, cursor)
        if start < 0:
            raise ValueError("PySBD segment is not an exact grounding substring")
        end = start + len(segment)
        units.append(
            EvidenceUnit(
                ref_id=f"G{index:03d}",
                source="grounding",
                location="grounding_excerpt",
                text=segment,
                start=start,
                end=end,
            )
        )
        cursor = end

    for index, (location, text) in enumerate(
        _walk_generated_strings(case.generated), start=1
    ):
        units.append(
            EvidenceUnit(
                ref_id=f"D{index:03d}",
                source="generated",
                location=location,
                text=text,
            )
        )
    ids = [unit.ref_id for unit in units]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evidence reference ID")
    return units


def render_case(case: v01.JudgeCase) -> str:
    units = build_evidence_units(case)
    payload = {
        "case_id": case.seed_id,
        "evidence_units": [unit.model_dump(mode="json") for unit in units],
    }
    return (
        "Evaluate this single case under the system rubric. Return JSON only. "
        "Cite evidence only by copying ref_id values from evidence_units.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


def validate_and_resolve_evidence(
    response: JudgeResponse,
    case: v01.JudgeCase,
) -> list[EvidenceSpan]:
    by_id = {unit.ref_id: unit for unit in build_evidence_units(case)}
    unknown = sorted(set(response.evidence_refs) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown evidence reference IDs: {unknown}")
    return [
        EvidenceSpan(
            source=by_id[ref_id].source,
            text=by_id[ref_id].text,
            start=by_id[ref_id].start,
            end=by_id[ref_id].end,
        )
        for ref_id in response.evidence_refs
    ]


def _provider_call(
    *,
    client: Any,
    config: RunnerConfig,
    system_prompt: str,
    case: v01.JudgeCase,
    correction: str | None,
) -> Any:
    user_content = render_case(case)
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
        "top_k": config.top_k,
        "min_p": config.min_p,
        "repetition_penalty": config.repetition_penalty,
    }
    return client.chat.completions.create(
        model=config.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=config.temperature,
        top_p=config.top_p,
        presence_penalty=config.presence_penalty,
        max_tokens=config.max_tokens,
        seed=config.seed,
        stream=False,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cfpb_semantic_judgment_v02",
                "strict": True,
                "schema": JudgeResponse.model_json_schema(),
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
            f"strict_json_schema=True, evidence_refs=True, "
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
            parsed = JudgeResponse.model_validate_json(content)
            evidence = validate_and_resolve_evidence(parsed, case)
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
        report_version="v02",
        evidence_protocol="stable_reference_ids_v01",
        fields_sent=["case_id", "evidence_units"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed-id", action="append", dest="seed_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    config = RunnerConfig()
    preflight = args.output_root / "zdr_endpoint_preflight.json"
    v01.fetch_zdr_endpoint_preflight(config=config, output_path=preflight)
    report = run_judge(
        raw_path=args.raw,
        prepared_path=args.prepared,
        prompt_path=args.prompt,
        judgments_path=args.output_root / "semantic_judgments.jsonl",
        raw_responses_path=args.output_root / "raw_responses.jsonl",
        failed_attempts_path=args.output_root / "failed_attempts.jsonl",
        endpoint_preflight_path=preflight,
        report_path=args.output_root / "judge_run_report.json",
        config=config,
        api_key=api_key,
        selected_seed_ids=args.seed_ids,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
