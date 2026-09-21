"""Generate CFPB Seed v05.2 engineering-only smoke dialogues with NeMo.

The model generates dialogue content only.  Product and issue labels remain
seed provenance and are attached programmatically by the downstream validator.
Runs are restricted to 10--20 records beneath the smoke-only workspace.
"""

from __future__ import annotations

import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults


_PROJECT_ROOT_ENV = os.environ.get("FINDISPUTEEVAL_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_PROJECT_ROOT_ENV).resolve()
    if _PROJECT_ROOT_ENV
    else Path(__file__).resolve().parents[3]
)
SMOKE_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v052_pipeline_override"
DEFAULT_SEED_PATH = SMOKE_ROOT / "prepared_inputs/nemo_seed_v052_pipeline_smoke_20.jsonl"
DEFAULT_ARTIFACT_PATH = SMOKE_ROOT / "raw"


class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Speaker role")
    content: str = Field(..., description="Message content")


class PipelineSmokeConversation(BaseModel):
    conversation: list[Message] = Field(
        ..., description="Alternating user and assistant messages"
    )
    synthetic_case_summary: str = Field(
        ..., description="Brief summary using no identifying or contact details"
    )
    privacy_notes: list[str] = Field(
        ..., description="Brief checks confirming generic, non-identifying content"
    )


def resolve_final_dataset_file(path: Path) -> Path:
    """Resolve Data Designer's file-or-directory final dataset pointer.

    Data Designer 0.7.0 may return the ``parquet-files`` directory rather than
    its single smoke-batch Parquet file.  A 10--20 row smoke run must have one
    final Parquet batch; refusing ambiguity prevents accidental partial reads.
    """
    candidate = Path(path).resolve()
    if candidate.is_file():
        if candidate.suffix.lower() not in {".parquet", ".jsonl", ".ndjson"}:
            raise ValueError(f"Unsupported final dataset file: {candidate}")
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    if not candidate.is_dir():
        raise ValueError(f"Final dataset path is neither a file nor directory: {candidate}")
    parquet_files = sorted(item for item in candidate.rglob("*.parquet") if item.is_file())
    if len(parquet_files) != 1:
        raise ValueError(
            "Pipeline smoke expects exactly one final Parquet batch beneath "
            f"{candidate}; found {len(parquet_files)}"
        )
    return parquet_files[0]


SYSTEM_PROMPT = """You create synthetic engineering-test conversations for a financial-dispute intake system.
The grounding text is unverified and partially redacted. Treat it only as an abstract dispute pattern.
Classification labels are routing metadata, not factual evidence. If a label conflicts with the grounding,
preserve the grounding facts and keep the disputed point ambiguous.
Never reproduce identifying details. Never claim that you accessed an account or completed a real-world action.
All output must be English and must remain generic, fictional, and non-operational."""


PROMPT = """Create one synthetic financial-dispute support conversation.

Dataset classification metadata (routing aid only; do not repeat it as output fields):
- product: {{ product }}
- sub-product: {{ sub_product }}
- issue: {{ issue }}
- sub-issue: {{ sub_issue }}

Defense-in-depth redacted grounding excerpt:
{{ generation_grounding_excerpt }}

Conversation requirements:
- Produce exactly {{ conversation_length }} messages.
- Begin with the user and strictly alternate user and assistant roles.
- User tone: {{ customer_mood }}.
- Paraphrase the dispute pattern. Never copy eight consecutive words from the grounding excerpt.
- Use no person or company names, street addresses, cities tied to a person, phone numbers, emails, URLs,
  account/card numbers, SSNs, PINs, passwords, credentials, one-time codes, complaint IDs, IP addresses,
  exact dates, or other identifying strings.
- Do not invent contact details. If a detail is needed, say only "the merchant", "my account",
  "the transaction", "a recent date", or "the disputed amount".
- Do not infer whether a transaction was authorized, unauthorized, accidental, or scam-induced unless the
  grounding states it. Ask a neutral clarifying question when authorization is unclear.
- Do not speculate about fee causes, account-closure causes, fraud-review steps, product policy, or an
  institution process that is absent from the grounding.
- The assistant may explain generic options and suggest contacting the financial institution through its
  official channel. It must not claim to file, submit, open, investigate, monitor, freeze, reverse, refund,
  credit, escalate, or resolve anything.
- The assistant must not guarantee an outcome, promise a deadline, assert legal rights or conclusions, cite statutes,
  request sensitive credentials, or provide a URL.
- End with a cautious, actionable next step the user can perform.

The structured output contains only conversation, synthetic_case_summary, and privacy_notes.
Do not generate product or issue label fields; those are attached programmatically from the seed."""


def build_config(seed_path: str, model_alias: str) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder()
    builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=seed_path),
        sampling_strategy=dd.SamplingStrategy.ORDERED,
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="conversation_length",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=[4, 6, 8]),
        )
    )
    builder.add_column(
        dd.SamplerColumnConfig(
            name="customer_mood",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(
                values=["anxious", "frustrated", "confused", "firm but calm"]
            ),
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="dialogue",
            output_format=PipelineSmokeConversation,
            model_alias=model_alias,
            system_prompt=SYSTEM_PROMPT,
            prompt=PROMPT,
        )
    )
    return builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    artifact_path: Path,
    *,
    preview_records: int = 2,
) -> DatasetCreationResults:
    if not 10 <= num_records <= 20:
        raise ValueError("Pipeline smoke run must contain 10 to 20 records")
    resolved = artifact_path.resolve()
    if SMOKE_ROOT.resolve() not in (resolved, *resolved.parents):
        raise ValueError("Artifacts must remain under outputs/generation/smoke_only")
    artifact_path.mkdir(parents=True, exist_ok=True)
    designer = DataDesigner(artifact_path=artifact_path)
    designer.validate(config_builder)
    if preview_records:
        preview = designer.preview(
            config_builder=config_builder,
            num_records=min(preview_records, num_records),
        )
        preview.display_sample_record()
    return designer.create(config_builder, num_records=num_records)


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--model-alias", default="nvidia-text")
    parser.add_argument("--num-records", type=int, default=20)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--skip-preview", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.seed_path.exists():
        raise FileNotFoundError(
            "Run prepare_cfpb_seed_v052_pipeline_smoke.py first"
        )
    result = create_dataset(
        build_config(str(args.seed_path), args.model_alias),
        args.num_records,
        args.artifact_path,
        preview_records=0 if args.skip_preview else 2,
    )
    raw_output = resolve_final_dataset_file(
        Path(result.artifact_storage.final_dataset_path)
    )
    print(f"Pipeline-only raw artifact: {raw_output}")
    print(
        "POLICY: provisional=true; privacy_verified=false; "
        "benchmark_eligible=false; do not promote or calibrate."
    )


if __name__ == "__main__":
    main()
