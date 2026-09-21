from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
    OracleRow,
    SemanticJudgment,
)
from scripts.generation.nemo_data_designer.prepare_cfpb_seed_v052_pipeline_smoke_v02 import (
    grounding_features,
    verify_excerpt_offset_invariant,
)
from scripts.generation.nemo_data_designer import (
    validate_cfpb_v052_pipeline_smoke_v02 as validator,
)


def seed(
    grounding: str = (
        "The customer noticed a disputed card transaction and asked the financial "
        "institution to explain the amount shown on a recent statement."
    ),
) -> dict:
    return {
        "seed_id": "seed_1",
        "product": "Card",
        "sub_product": "Credit card",
        "issue": "Billing dispute",
        "sub_issue": "Wrong amount",
        "generation_grounding_excerpt": grounding,
    }


def raw_with_assistant(text: str) -> dict:
    return {
        "seed_id": "seed_1",
        "product": "Card",
        "sub_product": "Credit card",
        "issue": "Billing dispute",
        "sub_issue": "Wrong amount",
        "conversation_length": 4,
        "dialogue": {
            "conversation": [
                {"role": "user", "content": "I noticed a disputed amount."},
                {"role": "assistant", "content": text},
                {"role": "user", "content": "What should I do next?"},
                {
                    "role": "assistant",
                    "content": "Use the institution's official channel and keep your records.",
                },
            ],
            "synthetic_case_summary": "A customer asks about a disputed amount.",
            "privacy_notes": ["No identifying details were included."],
        },
    }


def semantic_accept() -> SemanticJudgment:
    return SemanticJudgment(
        seed_id="seed_1",
        reasons=[],
        evidence=[],
        decision="accept",
        judge_kind="human",
        judge_id="reviewer-test",
        prompt_or_guideline_sha256="0" * 64,
        reviewed_utc="2026-07-22T00:00:00+00:00",
    )


def test_offset_invariant_verifies_against_full_text() -> None:
    full_text = "Contact Maria about the transfer. Additional full text."
    excerpt = "Contact Maria about the transfer."
    findings = pd.DataFrame(
        [{"start": 8, "end": 13, "detected_text": "Maria"}]
    )
    assert verify_excerpt_offset_invariant(
        seed_id="seed_1",
        excerpt=excerpt,
        full_text=full_text,
        finding_rows=findings,
    ) == 1


def test_offset_invariant_fails_closed_on_nonprefix_or_mismatch() -> None:
    findings = pd.DataFrame(
        [{"start": 8, "end": 13, "detected_text": "Maria"}]
    )
    with pytest.raises(ValueError, match="not a full-text prefix"):
        verify_excerpt_offset_invariant(
            seed_id="seed_1",
            excerpt="Maria appears in a window",
            full_text="Contact Maria about the transfer.",
            finding_rows=findings,
        )
    bad = findings.assign(detected_text="Marie")
    with pytest.raises(ValueError, match="text/offset mismatch"):
        verify_excerpt_offset_invariant(
            seed_id="seed_1",
            excerpt="Contact Maria about the transfer.",
            full_text="Contact Maria about the transfer.",
            finding_rows=bad,
        )


def test_grounding_features_ignore_privacy_placeholders() -> None:
    features = grounding_features(
        "[PII_CANDIDATE] sent [AMOUNT] through the payment service yesterday"
    )
    assert features["grounding_non_placeholder_tokens"] == 6
    assert features["grounding_placeholder_count"] == 2


@pytest.mark.parametrize(
    ("assistant_text", "should_flag"),
    [
        ("They will refund you within three days.", True),
        ("Your refund should arrive soon.", True),
        ("You could ask them when it will be completed.", False),
        (
            "Ask the support team to confirm when the deposit should be completed.",
            False,
        ),
    ],
)
def test_subject_aware_promise_detection(assistant_text: str, should_flag: bool) -> None:
    reasons, _, route = validator.validate_row(
        raw_with_assistant(assistant_text),
        {"seed_1": seed()},
        semantic_accept(),
    )
    assert ("assistant_outcome_or_deadline_promise" in reasons) is should_flag
    assert route == ("rejected" if should_flag else "validated")


def test_missing_semantic_judgment_routes_clean_row_to_review() -> None:
    reasons, routed, route = validator.validate_row(
        raw_with_assistant("What details do you remember about the amount?"),
        {"seed_1": seed()},
    )
    assert reasons == []
    assert route == "review"
    assert routed["pipeline_smoke_validation_v02"]["semantic_judgment_present"] is False
    assert routed["benchmark_eligible"] is False


def test_sparse_post_redaction_grounding_is_rejected() -> None:
    sparse = "[PII_CANDIDATE] cash I sent got lost [REDACTED]"
    reasons, _, route = validator.validate_row(
        raw_with_assistant("What happened to the transfer?"),
        {"seed_1": seed(sparse)},
        semantic_accept(),
    )
    assert "insufficient_grounding" in reasons
    assert route == "rejected"


def test_oracle_metrics_include_reason_and_decision_scores() -> None:
    oracle = {
        "seed_1": OracleRow(
            seed_id="seed_1",
            review_status="reviewed",
            expected_decision="reject",
            expected_v02_reasons=["authorization_state_changed"],
            reviewer_id="reviewer-test",
            reviewed_utc="2026-07-22T00:00:00+00:00",
        ),
        "seed_2": OracleRow(
            seed_id="seed_2",
            review_status="reviewed",
            expected_decision="accept",
            expected_v02_reasons=[],
            reviewer_id="reviewer-test",
            reviewed_utc="2026-07-22T00:00:00+00:00",
        ),
    }
    metrics = validator.evaluate_against_oracle(
        {
            "seed_1": {"authorization_state_changed"},
            "seed_2": set(),
        },
        oracle,
        predicted_routes={"seed_1": "rejected", "seed_2": "validated"},
    )
    assert metrics["micro_f1"] == 1.0
    assert metrics["decision_reject_f1"] == 1.0
    assert metrics["decision_accuracy"] == 1.0
