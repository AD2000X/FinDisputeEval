"""FinDisputeEval custom NeMo Data Designer multi-turn chat recipe.

Run after preparing a clean seed file with:
    python scripts/generation/nemo_data_designer/prepare_cfpb_seed_v04.py

Then in an environment with data-designer installed and a model provider configured:
    python scripts/generation/nemo_data_designer/generate_multi_turn_dialogues.py --num-records 20
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import data_designer.config as dd
from data_designer.interface import DataDesigner, DatasetCreationResults


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RECIPE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "generation"
    / "benchmark_v01"
    / "nemo_data_designer"
    / "seeded_dialogue"
    / "recipe_v01"
)
DEFAULT_SEED_PATH = RECIPE_ROOT / "prepared_inputs" / "nemo_seed_relabel_false_stratified_100.jsonl"
DEFAULT_ARTIFACT_PATH = RECIPE_ROOT / "work"


class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Speaker role.")
    content: str = Field(..., description="Message content.")


class FinDisputeConversation(BaseModel):
    conversation: list[Message] = Field(..., description="Alternating user/assistant messages.")
    synthetic_case_summary: str = Field(..., description="One paragraph summary of the synthetic dispute.")
    label_card_type: str = Field(..., description="Seed card_type copied from the seed row.")
    label_claim_type: str = Field(..., description="Seed claim_type copied from the seed row.")
    route_hint: str = Field(..., description="Seed route_hint copied from the seed row.")
    privacy_notes: list[str] = Field(..., description="Checks showing no real PII or complaint IDs were copied.")


GROUNDING_PROMPT = (
    "You are auditing a synthetic financial-dispute support conversation.\n"
    "Score whether the conversation stays faithful to the supplied seed labels and dispute pattern, "
    "without copying the original complaint wording or exposing real complaint identifiers.\n\n"
    "Seed labels: {{ card_type }} / {{ claim_type }} / {{ route_hint }}\n"
    "Seed excerpt:\n{{ seed_narrative_excerpt }}\n\n"
    "Synthetic conversation:\n{{ conversation }}\n"
)


grounding_score = dd.Score(
    name="Grounding",
    description="Faithfulness to seed labels and dispute pattern while avoiding verbatim copying.",
    options={
        4: "Strongly grounded; labels and facts align, no obvious copied phrasing.",
        3: "Mostly grounded; minor drift or mild over-generalization.",
        2: "Partially grounded; important label or scenario details are weak.",
        1: "Poorly grounded; wrong dispute type or route.",
        0: "Unusable; copies source wording, exposes identifiers, or contradicts labels.",
    },
)


def build_config(seed_path: str, model_alias: str) -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()
    seed_source = dd.LocalFileSeedSource(path=seed_path)
    config_builder.with_seed_dataset(seed_source, sampling_strategy=dd.SamplingStrategy.SHUFFLE)

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="conversation_length",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=[4, 6, 8]),
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="customer_mood",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["anxious", "frustrated", "confused", "firm but calm"]),
        )
    )
    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="support_stage",
            sampler_type=dd.SamplerType.CATEGORY,
            params=dd.CategorySamplerParams(values=["first contact", "follow-up", "post-denial escalation"]),
        )
    )

    config_builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="conversation",
            output_format=FinDisputeConversation,
            model_alias=model_alias,
            prompt=(
                "Generate a realistic multi-turn customer-support chat for FinDisputeEval.\n\n"
                "Use the seed row only as an abstract dispute pattern. Do not copy the complaint text. "
                "Do not mention the CFPB, complaint IDs, real people, addresses, phone numbers, emails, or exact company names. "
                "Use fictional names and plausible but synthetic amounts/dates if needed.\n\n"
                "Seed labels:\n"
                "- card_type: {{ card_type }}\n"
                "- claim_type: {{ claim_type }}\n"
                "- route_hint: {{ route_hint }}\n"
                "- zelle_mention: {{ zelle_mention }}\n"
                "- anchor_confidence: {{ anchor_confidence }}\n\n"
                "Seed narrative excerpt for grounding, not copying:\n"
                "{{ seed_narrative_excerpt }}\n\n"
                "Conversation requirements:\n"
                "- Exactly {{ conversation_length }} messages total.\n"
                "- Start with the user and alternate user/assistant roles.\n"
                "- The user mood is {{ customer_mood }}.\n"
                "- The support stage is {{ support_stage }}.\n"
                "- The assistant should ask targeted clarifying questions, explain next procedural steps, "
                "and avoid promising outcomes.\n"
                "- Preserve the seed claim_type and route_hint semantics.\n"
                "- For fraud/scam or unauthorized-transfer scenarios, keep authorization ambiguity explicit instead of resolving it too early.\n"
                "- End with a useful next step or escalation path.\n"
            ),
        )
    )

    config_builder.add_column(
        dd.LLMJudgeColumnConfig(
            name="grounding_evaluation",
            prompt=GROUNDING_PROMPT,
            scores=[grounding_score],
            model_alias=model_alias,
        )
    )
    return config_builder


def create_dataset(
    config_builder: dd.DataDesignerConfigBuilder,
    num_records: int,
    artifact_path: Path | str | None = None,
) -> DatasetCreationResults:
    data_designer = DataDesigner(artifact_path=artifact_path)
    data_designer.validate(config_builder)
    preview = data_designer.preview(config_builder=config_builder, num_records=min(2, num_records))
    preview.display_sample_record()
    return data_designer.create(config_builder, num_records=num_records)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--model-alias", type=str, default="nvidia-text")
    parser.add_argument("--num-records", type=int, default=20)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args()

    cfg = build_config(seed_path=str(args.seed_path), model_alias=args.model_alias)
    results = create_dataset(cfg, num_records=args.num_records, artifact_path=args.artifact_path)
    print(f"Dataset saved to: {results.artifact_storage.final_dataset_path}")
    results.load_analysis().to_report()
