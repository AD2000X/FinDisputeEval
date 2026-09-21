"""NeMo Data Designer recipe for the Seed v05.1 provisional smoke test.

This recipe refuses more than 20 records and writes only beneath smoke_only.
Its outputs cannot be promoted to a benchmark release.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SMOKE_ROOT = PROJECT_ROOT / "outputs/generation/smoke_only/cfpb_seed_v051_candidate"
DEFAULT_SEED_PATH = SMOKE_ROOT / "prepared_inputs/nemo_seed_v051_provisional_smoke_20.jsonl"
DEFAULT_ARTIFACT_PATH = SMOKE_ROOT / "raw"


class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Speaker role")
    content: str = Field(..., description="Message content")


class ProvisionalConversation(BaseModel):
    conversation: list[Message]
    synthetic_case_summary: str
    label_product: str
    label_issue: str
    label_sub_issue: str
    privacy_notes: list[str]


def build_config(seed_path: str, model_alias: str) -> dd.DataDesignerConfigBuilder:
    builder = dd.DataDesignerConfigBuilder()
    builder.with_seed_dataset(
        dd.LocalFileSeedSource(path=seed_path),
        sampling_strategy=dd.SamplingStrategy.SHUFFLE,
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
            params=dd.CategorySamplerParams(values=["anxious", "frustrated", "confused", "firm but calm"]),
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="conversation",
            output_format=ProvisionalConversation,
            model_alias=model_alias,
            prompt=(
                "Generate a synthetic financial-dispute support chat for an engineering-only smoke test.\n"
                "Use the source only as an abstract pattern. Do not copy phrases, real company names, "
                "complaint identifiers, names, addresses, phone numbers, emails, account/card numbers, "
                "or exact dates and amounts. Use obviously fictional details where needed.\n\n"
                "Labels: product={{ product }}; sub-product={{ sub_product }}; issue={{ issue }}; "
                "sub-issue={{ sub_issue }}.\n"
                "Source excerpt (grounding only): {{ seed_narrative_excerpt }}\n\n"
                "Produce exactly {{ conversation_length }} messages, beginning with user and alternating roles. "
                "Customer mood: {{ customer_mood }}. The assistant asks useful clarifying questions, explains "
                "procedural next steps, avoids legal conclusions and outcome promises, and ends with a next step. "
                "Copy the supplied product/issue labels into the structured label fields."
            ),
        )
    )
    return builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    artifact_path: Path,
) -> DatasetCreationResults:
    if not 10 <= num_records <= 20:
        raise ValueError("Provisional smoke run must contain 10 to 20 records")
    resolved = artifact_path.resolve()
    if SMOKE_ROOT.resolve() not in (resolved, *resolved.parents):
        raise ValueError("Artifacts must remain under outputs/generation/smoke_only")
    designer = DataDesigner(artifact_path=artifact_path)
    designer.validate(config_builder)
    preview = designer.preview(config_builder=config_builder, num_records=min(2, num_records))
    preview.display_sample_record()
    return designer.create(config_builder, num_records=num_records)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--model-alias", default="nvidia-text")
    parser.add_argument("--num-records", type=int, default=20)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args()
    if not args.seed_path.exists():
        raise FileNotFoundError("Run prepare_cfpb_seed_v051.py first")
    result = create_dataset(
        build_config(str(args.seed_path), args.model_alias),
        args.num_records,
        args.artifact_path,
    )
    print(f"Provisional raw artifacts: {result.artifact_storage.final_dataset_path}")
    print("POLICY: smoke-only; do not promote, publish, calibrate, or merge into a benchmark.")
