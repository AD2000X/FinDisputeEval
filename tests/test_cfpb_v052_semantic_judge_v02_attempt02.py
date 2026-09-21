from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

import scripts.build_cfpb_v052_semantic_judge_v02_attempt02_notebook as notebook_builder
from scripts.generation.nemo_data_designer import (
    build_cfpb_v052_judge_attempt02_review as review_builder,
)
from scripts.generation.nemo_data_designer.cfpb_v052_pipeline_smoke_v02_common import (
    load_oracle_csv,
    load_semantic_judgments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_second_pass_covers_exact_attempt01_disagreements() -> None:
    oracle = load_oracle_csv(review_builder.ORACLE_V01, require_complete=True)
    judgments = load_semantic_judgments(review_builder.ATTEMPT01_JUDGMENTS)
    disagreement_ids = {
        seed_id
        for seed_id, expected in oracle.items()
        if expected.expected_decision != judgments[seed_id].decision
        or set(expected.expected_v02_reasons) != set(judgments[seed_id].reasons)
    }
    assert len(disagreement_ids) == 12
    assert disagreement_ids == set(review_builder.SECOND_PASS)


def test_oracle_v02_is_versioned_complete_and_explicitly_not_human_gold() -> None:
    v01 = load_oracle_csv(review_builder.ORACLE_V01, require_complete=True)
    v02 = load_oracle_csv(review_builder.ORACLE_V02, require_complete=True)
    assert set(v01) == set(v02)
    changed = {
        seed_id
        for seed_id in v01
        if set(v01[seed_id].expected_v02_reasons)
        != set(v02[seed_id].expected_v02_reasons)
    }
    assert changed == {"cfpb_v052_dff99a7913960475"}
    assert "unsupported_product_policy" in v02[
        "cfpb_v052_dff99a7913960475"
    ].expected_v02_reasons
    report = json.loads(review_builder.REPORT.read_text(encoding="utf-8"))
    assert report["human_gold"] is False
    assert report["readjudication"]["rows"] == 12
    assert report["readjudication"]["confirmed_rows"] == 11
    assert report["readjudication"]["amended_rows"] == 1
    assert report["attempt01_judge_only_metrics_against_oracle_v02"][
        "micro_recall"
    ] == 0.32
    gate = report["attempt01_freeze_gate_against_oracle_v02"]
    assert gate["automatic_metrics_passed"] is False
    assert gate["human_disagreement_review_completed"] is False
    assert gate["frozen"] is False


def test_attempt02_rubric_requires_complete_reason_scan() -> None:
    prompt = (
        PROJECT_ROOT
        / "configs/generation/judges/cfpb_v052_semantic_judge_v02_attempt02.md"
    ).read_text(encoding="utf-8")
    assert "Mandatory internal checklist" in prompt
    assert "Do not stop after finding the" in prompt
    assert "first mismatch" in prompt
    assert "No reason is omitted merely because another reason" in prompt
    assert "synthetic case summary" in prompt
    assert "internal processing cycle" in prompt


def test_attempt02_notebook_is_independent_hash_bound_and_fail_closed() -> None:
    notebook = json.loads(notebook_builder.OUTPUT.read_text(encoding="utf-8"))
    all_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "openrouter_qwen3_5_35b_a3b_v02_attempt02" in all_source
    assert "cfpb_v052_semantic_judge_v02_attempt02.md" in all_source
    assert "gpt56sol_adjudication_20_v02.csv" in all_source
    assert "nonthinking_blind_v02_attempt02" in all_source
    assert "AUTO_FREEZE_METRICS_PASSED" in all_source
    assert "DEVELOPMENT_DISAGREEMENTS_REVIEWED_BY_HUMAN = False" in all_source
    assert "APPROVE_FREEZE = False" in all_source
    assert "cfpb_v052_semantic_judge_v02_attempt02.json" in all_source

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
    assert "cfpb_v052_semantic_judge_v02_attempt02.md" in embedded
    for item in embedded.values():
        raw = gzip.decompress(base64.b64decode(item["payload"]))
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
