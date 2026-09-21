"""Scaffold the 20-row human adjudication oracle for validator v02.

Candidate findings are desk-review prefill only.  The script never marks a row
reviewed and never copies candidate reasons into the expected-reasons column.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
        OracleRow,
        json_safe,
        load_records,
    )
except ModuleNotFoundError:  # Direct execution from this script's directory.
    from cfpb_v052_pipeline_smoke_v02_common import (  # type: ignore[no-redef]
        OracleRow,
        json_safe,
        load_records,
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
    SMOKE_ROOT
    / "revalidations/validator_v02/source_run_20260722T135306Z"
)
DEFAULT_RAW = RUN_ROOT / "raw/data_designer/dataset/parquet-files/batch_00000.parquet"
DEFAULT_PREPARED = RUN_ROOT / "prepared_inputs/nemo_seed_v052_pipeline_smoke_20.jsonl"
DEFAULT_OUTPUT = REVALIDATION_ROOT / "oracle/human_adjudication_20_template.csv"


# These are analysis leads, not ground truth.  A reviewer must confirm, remove,
# or replace them and supply exact expected reasons and provenance.
CANDIDATE_PREFILL: dict[str, dict[str, Any]] = {
    "cfpb_v052_72cbe577e54e5c87": {
        "reasons": [],
        "note": "v01 promise match appears benign; review as a likely false positive.",
    },
    "cfpb_v052_dff99a7913960475": {
        "reasons": ["insufficient_grounding", "unsupported_scenario_detail"],
        "note": "Very sparse post-redaction grounding; generated scenario adds material facts.",
    },
    "cfpb_v052_f7e62f357ab1814e": {
        "reasons": ["authorization_state_changed", "claim_type_changed"],
        "note": "Purchase/service dispute is reframed as a charge the customer did not approve.",
    },
    "cfpb_v052_2c70d920e4cfccd5": {
        "reasons": ["authorization_state_changed", "label_grounding_conflict"],
        "note": "Grounding says the consumer sent the payment; sub-issue says unauthorized.",
    },
    "cfpb_v052_0f6344daa27f2d71": {
        "reasons": ["factual_status_changed", "unsupported_procedural_guidance"],
        "note": "Grounding acknowledges accurate, resolved history; output alleges inaccuracy.",
    },
    "cfpb_v052_36e6ef4ba6ea4e5d": {
        "reasons": ["legal_or_rights_claim", "unsupported_procedural_guidance"],
        "note": "Assistant states a right and procedural conclusions without authority grounding.",
    },
    "cfpb_v052_3a638911520b2140": {
        "reasons": ["unsupported_product_policy"],
        "note": "Assistant speculates about region/category causes for a foreign fee.",
    },
    "cfpb_v052_6ca27b0ef108319b": {
        "reasons": ["label_grounding_conflict"],
        "note": "Review whether a clawback/negative balance was incorrectly reframed as fees.",
    },
    "cfpb_v052_8a4c6b351ee2bcc8": {
        "reasons": ["unsupported_procedural_guidance"],
        "note": "Review invented status-report or internal-process guidance.",
    },
    "cfpb_v052_e790b3057f417a69": {
        "reasons": ["unsupported_procedural_guidance"],
        "note": "Review unsupported fee-removal or goodwill guidance.",
    },
    "cfpb_v052_1d6f053e2e1c3e61": {
        "reasons": ["indirect_sensitive_information_guidance"],
        "note": "Review instruction to have account information ready.",
    },
}


def parse_dialogue(value: Any) -> dict[str, Any]:
    value = json_safe(value)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def generated_text(value: Any) -> str:
    dialogue = parse_dialogue(value)
    parts = [
        f"{message.get('role', '')}: {message.get('content', '')}"
        for message in dialogue.get("conversation", [])
        if isinstance(message, dict)
    ]
    if dialogue.get("synthetic_case_summary"):
        parts.append(f"summary: {dialogue['synthetic_case_summary']}")
    return "\n".join(parts)


def scaffold_oracle(raw_path: Path, prepared_path: Path, output_path: Path) -> pd.DataFrame:
    raw = pd.DataFrame(load_records(raw_path))
    prepared = pd.DataFrame(load_records(prepared_path))
    if raw["seed_id"].duplicated().any() or prepared["seed_id"].duplicated().any():
        raise ValueError("Raw or prepared input contains duplicate seed_id values")
    if set(raw["seed_id"].astype(str)) != set(prepared["seed_id"].astype(str)):
        raise ValueError("Raw and prepared seed coverage differ")
    merged = raw[["seed_id", "dialogue"]].merge(
        prepared[["seed_id", "generation_grounding_excerpt"]],
        on="seed_id",
        validate="one_to_one",
    )

    records: list[dict[str, str]] = []
    for row in merged.to_dict(orient="records"):
        seed_id = str(row["seed_id"])
        candidate = CANDIDATE_PREFILL.get(seed_id, {"reasons": [], "note": ""})
        oracle = OracleRow(
            seed_id=seed_id,
            candidate_v02_reasons=candidate["reasons"],
            grounding_evidence=str(row["generation_grounding_excerpt"]),
            generated_evidence=generated_text(row["dialogue"]),
            analysis_note=str(candidate["note"]),
        )
        record = oracle.model_dump(mode="json")
        record["expected_v02_reasons_json"] = json.dumps(
            record.pop("expected_v02_reasons"), ensure_ascii=False
        )
        record["candidate_v02_reasons_json"] = json.dumps(
            record.pop("candidate_v02_reasons"), ensure_ascii=False
        )
        records.append(record)
    frame = pd.DataFrame(records).sort_values("seed_id", kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = scaffold_oracle(args.raw, args.prepared, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(frame),
                "reviewed_rows": int(frame["review_status"].eq("reviewed").sum()),
                "candidate_prefill_rows": int(
                    frame["candidate_v02_reasons_json"].ne("[]").sum()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
