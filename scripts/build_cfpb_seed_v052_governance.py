"""Prepare and finalize CFPB Seed v05.2 decision record v04.

Preparation creates a C-lite fuzzy component review pack and a v04 draft.
Finalization is fail-closed until the required cluster and bridge reviews are
complete. Existing review CSVs are never overwritten without ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from findisputeeval.curation.cfpb_seed_common import sha256_file


RUN_ID = "run_20260713T145423Z"
AUDIT_ROOT = ROOT / "dataset/curated/annotations/cfpb_seed_v05_audit" / RUN_ID
EDA_CORPUS = ROOT / "outputs/data_pipeline/cfpb_seed_source_eda/eda_v051" / RUN_ID / "analysis_ready_corpus.parquet"
PARENT = AUDIT_ROOT / "seed_v051_decision_record_v03.json"
PAIR_MASTER = AUDIT_ROOT / "fuzzy_duplicate_audit_master.csv"
CLUSTER_REVIEW = AUDIT_ROOT / "fuzzy_cluster_review_v04.csv"
BRIDGE_REVIEW = AUDIT_ROOT / "fuzzy_bridge_review_v04.csv"
CLUSTER_EVIDENCE = AUDIT_ROOT / "fuzzy_cluster_evidence_v04.csv"
COVERAGE_ANALYSIS = AUDIT_ROOT / "seed_v052_coverage_analysis_v01.csv"
DRAFT = AUDIT_ROOT / "seed_v052_decision_record_v04_draft.json"
FINAL = AUDIT_ROOT / "seed_v052_decision_record_v04.json"

REQUIRED_CLUSTER_SIZE = 10
MAX_CLUSTER_WITHOUT_EXPLICIT_APPROVAL = 50
MAX_BRIDGES_PER_REQUIRED_CLUSTER = 15
MAX_EVIDENCE_ROWS_PER_CLUSTER = 18
ALLOWED_REVIEW = {"yes", "no", "uncertain"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype="string", keep_default_na=False, encoding="utf-8-sig")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def bridge_edges(nodes: set[str], edges: list[tuple[str, str]]) -> set[tuple[str, str]]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    timer = 0
    discovered: dict[str, int] = {}
    low: dict[str, int] = {}
    bridges: set[tuple[str, str]] = set()

    def dfs(node: str, parent: str | None) -> None:
        nonlocal timer
        timer += 1
        discovered[node] = low[node] = timer
        for neighbor in sorted(adjacency[node]):
            if neighbor == parent:
                continue
            if neighbor not in discovered:
                dfs(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovered[node]:
                    bridges.add(tuple(sorted((node, neighbor))))
            else:
                low[node] = min(low[node], discovered[neighbor])

    for node in sorted(nodes):
        if node not in discovered:
            dfs(node, None)
    return bridges


def component_id(nodes: Iterable[str]) -> str:
    payload = "|".join(sorted(nodes)).encode("utf-8")
    return "fuzzy_v04_" + hashlib.sha256(payload).hexdigest()[:16]


def build_fuzzy_review() -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = read_csv(PAIR_MASTER)
    confirmed = pairs.loc[pairs["manual_same_template"].str.lower().isin(["yes", "true", "1"])].copy()
    uf = UnionFind()
    for left, right in confirmed[["complaint_id_a", "complaint_id_b"]].itertuples(index=False, name=None):
        uf.union(str(left), str(right))
    nodes_by_root: dict[str, set[str]] = defaultdict(set)
    for node in uf.parent:
        nodes_by_root[uf.find(node)].add(node)
    root_to_component = {root: component_id(nodes) for root, nodes in nodes_by_root.items()}
    node_to_component = {node: root_to_component[uf.find(node)] for node in uf.parent}
    confirmed["component_id"] = confirmed["complaint_id_a"].map(node_to_component)
    confirmed["shingle_jaccard_num"] = pd.to_numeric(confirmed["shingle_jaccard"], errors="coerce")
    confirmed["token_set_ratio_num"] = pd.to_numeric(confirmed["rapidfuzz_token_set_ratio"], errors="coerce")

    text_frame = pd.read_parquet(EDA_CORPUS, columns=["Complaint ID", "narrative_canonical"])
    text_by_id = dict(zip(text_frame["Complaint ID"].astype(str), text_frame["narrative_canonical"].astype(str)))
    cluster_rows: list[dict] = []
    bridge_rows: list[dict] = []
    for root, nodes in sorted(nodes_by_root.items(), key=lambda item: (-len(item[1]), item[0])):
        cid = root_to_component[root]
        component_pairs = confirmed.loc[confirmed["component_id"].eq(cid)].copy()
        edges = [tuple(sorted((str(left), str(right)))) for left, right in component_pairs[["complaint_id_a", "complaint_id_b"]].itertuples(index=False, name=None)]
        bridges = bridge_edges(nodes, edges)
        required = len(nodes) >= REQUIRED_CLUSTER_SIZE
        cluster_rows.append(
            {
                "component_id": cid,
                "node_count": len(nodes),
                "edge_count": len(component_pairs),
                "bridge_count": len(bridges),
                "min_shingle_jaccard": component_pairs["shingle_jaccard_num"].min(),
                "median_shingle_jaccard": component_pairs["shingle_jaccard_num"].median(),
                "min_token_set_ratio": component_pairs["token_set_ratio_num"].min(),
                "review_required": required,
                "review_reason": "size_ge_10" if required else "not_required_c_lite",
                "manual_cluster_valid": "pending" if required else "not_required",
                "reviewer": "",
                "reviewed_utc": "",
                "notes": "",
            }
        )
        if not required:
            continue
        component_pairs["edge_key"] = component_pairs.apply(
            lambda row: tuple(sorted((str(row["complaint_id_a"]), str(row["complaint_id_b"])))), axis=1
        )
        candidates = component_pairs.loc[component_pairs["edge_key"].isin(bridges)].copy()
        candidates = candidates.sort_values(
            ["shingle_jaccard_num", "token_set_ratio_num", "complaint_id_a", "complaint_id_b"],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).head(MAX_BRIDGES_PER_REQUIRED_CLUSTER)
        for rank, row in enumerate(candidates.to_dict(orient="records"), start=1):
            left, right = str(row["complaint_id_a"]), str(row["complaint_id_b"])
            bridge_rows.append(
                {
                    "component_id": cid,
                    "component_node_count": len(nodes),
                    "bridge_review_rank": rank,
                    "complaint_id_a": left,
                    "complaint_id_b": right,
                    "shingle_jaccard": row["shingle_jaccard"],
                    "rapidfuzz_token_set_ratio": row["rapidfuzz_token_set_ratio"],
                    "text_a": text_by_id.get(left, "")[:1200],
                    "text_b": text_by_id.get(right, "")[:1200],
                    "prior_llm_label": row["manual_same_template"],
                    "manual_same_template": "pending",
                    "reviewer": "",
                    "reviewed_utc": "",
                    "notes": "",
                }
            )
    return pd.DataFrame(cluster_rows), pd.DataFrame(bridge_rows)


def build_cluster_evidence(
    clusters: pd.DataFrame,
    bridges: pd.DataFrame,
) -> pd.DataFrame:
    """Create deterministic representative-text evidence for required clusters."""
    required_ids = set(
        clusters.loc[
            clusters["review_required"].astype(str).str.lower().eq("true"),
            "component_id",
        ].astype(str)
    )
    pairs = read_csv(PAIR_MASTER)
    confirmed = pairs.loc[
        pairs["manual_same_template"].str.lower().isin(["yes", "true", "1"])
    ].copy()
    uf = UnionFind()
    for left, right in confirmed[["complaint_id_a", "complaint_id_b"]].itertuples(
        index=False, name=None
    ):
        uf.union(str(left), str(right))
    nodes_by_root: dict[str, set[str]] = defaultdict(set)
    for node in uf.parent:
        nodes_by_root[uf.find(node)].add(node)
    root_to_component = {
        root: component_id(nodes) for root, nodes in nodes_by_root.items()
    }
    node_to_component = {
        node: root_to_component[uf.find(node)] for node in uf.parent
    }
    confirmed["component_id"] = confirmed["complaint_id_a"].map(node_to_component)
    confirmed["shingle_jaccard_num"] = pd.to_numeric(
        confirmed["shingle_jaccard"], errors="coerce"
    )

    relevant_nodes = {
        node
        for root, nodes in nodes_by_root.items()
        if root_to_component[root] in required_ids
        for node in nodes
    }
    corpus = pd.read_parquet(
        EDA_CORPUS,
        columns=[
            "Complaint ID",
            "Product",
            "Issue",
            "register_candidate",
            "word_count",
            "narrative_canonical",
        ],
    )
    corpus["Complaint ID"] = corpus["Complaint ID"].astype(str)
    corpus = corpus.loc[corpus["Complaint ID"].isin(relevant_nodes)].copy()
    corpus["word_count_num"] = pd.to_numeric(corpus["word_count"], errors="coerce").fillna(0)
    metadata = corpus.set_index("Complaint ID").to_dict(orient="index")

    rows: list[dict] = []
    for root, nodes in sorted(
        nodes_by_root.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        cid = root_to_component[root]
        if cid not in required_ids:
            continue
        component_pairs = confirmed.loc[confirmed["component_id"].eq(cid)].copy()
        degree: Counter[str] = Counter()
        for left, right in component_pairs[
            ["complaint_id_a", "complaint_id_b"]
        ].itertuples(index=False, name=None):
            degree[str(left)] += 1
            degree[str(right)] += 1

        selected: dict[str, set[str]] = {}

        def add(node: str, reason: str) -> None:
            node = str(node)
            if node in nodes:
                selected.setdefault(node, set()).add(reason)

        for node, _ in sorted(degree.items(), key=lambda item: (-item[1], item[0]))[:3]:
            add(node, "high_degree")
        weakest = component_pairs.sort_values(
            ["shingle_jaccard_num", "complaint_id_a", "complaint_id_b"],
            kind="mergesort",
        ).head(4)
        for rank, (left, right) in enumerate(
            weakest[["complaint_id_a", "complaint_id_b"]].itertuples(
                index=False, name=None
            ),
            start=1,
        ):
            add(str(left), f"weak_edge_endpoint_{rank}")
            add(str(right), f"weak_edge_endpoint_{rank}")
        component_bridges = bridges.loc[bridges["component_id"].eq(cid)].head(4)
        for left, right in component_bridges[
            ["complaint_id_a", "complaint_id_b"]
        ].itertuples(index=False, name=None):
            add(str(left), "reviewed_bridge_endpoint")
            add(str(right), "reviewed_bridge_endpoint")

        component_meta = corpus.loc[corpus["Complaint ID"].isin(nodes)].copy()
        if not component_meta.empty:
            ordered_length = component_meta.sort_values(
                ["word_count_num", "Complaint ID"], kind="mergesort"
            )
            add(str(ordered_length.iloc[0]["Complaint ID"]), "shortest_text")
            add(str(ordered_length.iloc[-1]["Complaint ID"]), "longest_text")
            diversity = component_meta.sort_values("Complaint ID", kind="mergesort").drop_duplicates(
                ["Product", "Issue", "register_candidate"]
            )
            for node in diversity["Complaint ID"].head(6):
                add(str(node), "metadata_diversity")

        ordered_nodes = sorted(
            selected,
            key=lambda node: (
                0 if any(reason.startswith("weak_edge") for reason in selected[node]) else 1,
                0 if "high_degree" in selected[node] else 1,
                -degree.get(node, 0),
                node,
            ),
        )[:MAX_EVIDENCE_ROWS_PER_CLUSTER]
        product_counts = component_meta["Product"].fillna("").value_counts().head(8).to_dict()
        issue_counts = component_meta["Issue"].fillna("").value_counts().head(8).to_dict()
        register_counts = (
            component_meta["register_candidate"].fillna("").value_counts().to_dict()
        )
        for rank, node in enumerate(ordered_nodes, start=1):
            item = metadata.get(node, {})
            rows.append(
                {
                    "component_id": cid,
                    "component_node_count": len(nodes),
                    "evidence_rank": rank,
                    "complaint_id": node,
                    "selection_reasons": "|".join(sorted(selected[node])),
                    "graph_degree": degree.get(node, 0),
                    "Product": item.get("Product", ""),
                    "Issue": item.get("Issue", ""),
                    "register_candidate": item.get("register_candidate", ""),
                    "word_count": item.get("word_count", ""),
                    "narrative_canonical": str(item.get("narrative_canonical", ""))[:2400],
                    "component_product_counts_json": json.dumps(
                        product_counts, ensure_ascii=False, sort_keys=True
                    ),
                    "component_issue_counts_json": json.dumps(
                        issue_counts, ensure_ascii=False, sort_keys=True
                    ),
                    "component_register_counts_json": json.dumps(
                        register_counts, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_coverage_analysis() -> pd.DataFrame:
    corpus = pd.read_parquet(EDA_CORPUS, columns=["population_eligible", "Product", "Issue", "date_received_parsed"])
    corpus["month"] = pd.to_datetime(corpus["date_received_parsed"], errors="coerce", utc=True).dt.strftime("%Y-%m")
    population = corpus.loc[corpus["population_eligible"]]
    core = pd.read_parquet(
        ROOT / "dataset/curated/seed_pools/cfpb_dispute/seed_v051/cfpb_seed_v051_population.parquet",
        columns=["Product", "Issue", "month"],
    )
    strata = ["Product", "Issue", "month"]
    population_cells = population.groupby(strata, dropna=False).size().rename("population_rows").reset_index()
    core_cells = core.groupby(strata, dropna=False).size().rename("core_rows").reset_index()
    analysis = population_cells.merge(core_cells, on=strata, how="left")
    analysis["core_rows"] = analysis["core_rows"].fillna(0).astype(int)
    analysis["covered_by_proportional_core"] = analysis["core_rows"].gt(0)
    analysis["coverage_supplement_target"] = analysis["core_rows"].eq(0).astype(int)
    analysis["prevalence_eligible"] = False
    return analysis.sort_values(strata, kind="mergesort").reset_index(drop=True)


def prepare(force: bool) -> None:
    for path in [PARENT, PAIR_MASTER, EDA_CORPUS]:
        if not path.exists():
            raise FileNotFoundError(path)
    protected = [CLUSTER_REVIEW, BRIDGE_REVIEW, CLUSTER_EVIDENCE, DRAFT]
    if not force and any(path.exists() for path in protected):
        raise FileExistsError("v04 review files already exist; use --force only before annotation begins")
    clusters, bridges = build_fuzzy_review()
    evidence = build_cluster_evidence(clusters, bridges)
    coverage = build_coverage_analysis()
    write_csv(clusters, CLUSTER_REVIEW)
    write_csv(bridges, BRIDGE_REVIEW)
    write_csv(evidence, CLUSTER_EVIDENCE)
    write_csv(coverage, COVERAGE_ANALYSIS)
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    rules = json.loads(json.dumps(parent["rules"]))
    rules["sampling_quotas"]["coverage_supplement"] = {
        "enabled": True,
        "strategy": "one_per_uncovered_eligible_stratum",
        "stratify_by": ["Product", "Issue", "month"],
        "core_size_unchanged": 2000,
        "maximum_rows": 500,
        "release_split": "coverage_supplement",
        "prevalence_eligible": False,
    }
    rules["fuzzy_duplicates"].update(
        {
            "validation_standard": "c_lite_cluster_and_bridge_review",
            "cluster_review_path": CLUSTER_REVIEW.name,
            "bridge_review_path": BRIDGE_REVIEW.name,
            "cluster_evidence_path": CLUSTER_EVIDENCE.name,
            "required_cluster_size": REQUIRED_CLUSTER_SIZE,
            "max_cluster_size_without_explicit_approval": MAX_CLUSTER_WITHOUT_EXPLICIT_APPROVAL,
            "unreviewed_or_uncertain_action": "bypass_component_cap",
            "rejected_bridge_action": "remove_edge_and_rebuild_components",
        }
    )
    rules["release_schema"]["record_id_prefix"] = "cfpb_v052_"
    rules["release_schema"]["generation_splits"] = [
        "population_core",
        "coverage_supplement",
        "enrichment",
    ]
    draft = {
        "record_version": "v04_draft",
        "release_target": "CFPB Seed v05.2 candidate",
        "created_utc": utc_now(),
        "source_eda_version": parent["source_eda_version"],
        "source_run": parent["source_run"],
        "source_manifest_sha256": parent["source_manifest_sha256"],
        "parent_decision_record_v03": PARENT.name,
        "parent_decision_record_v03_sha256": sha256_file(PARENT),
        "parent_governance_inputs": parent["governance_inputs"],
        "coverage_decision": {
            "status": "accepted",
            "approved_by": "chang",
            "basis": "User selected Option C two-stage sampling on 2026-07-16.",
            "raw_population_cells": len(coverage),
            "v051_core_covered_cells": int(coverage["covered_by_proportional_core"].sum()),
            "raw_uncovered_cells": int((~coverage["covered_by_proportional_core"]).sum()),
        },
        "fuzzy_decision": {
            "status": "pending_user_review",
            "standard": "C-lite",
            "required_cluster_reviews": int(clusters["review_required"].sum()),
            "required_bridge_reviews": len(bridges),
            "representative_evidence_rows": len(evidence),
            "review_order": "bridge_edges_then_cluster_decision",
        },
        "rules": rules,
        "build_gate": {"passed": False, "blockers": ["complete_fuzzy_cluster_review_v04", "finalize_decision_record_v04"]},
        "benchmark_release_gate": {
            "passed": False,
            "blocking_requirements": [
                "build_and_structurally_validate_seed_v052_candidate",
                "full_seed_and_stress_presidio_or_equivalent_ner_scan",
                "manual_review_of_final_privacy_scan_positives",
            ],
        },
    }
    DRAFT.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cluster_rows": len(clusters), "required_clusters": int(clusters["review_required"].sum()), "bridge_reviews": len(bridges), "cluster_evidence_rows": len(evidence), "coverage_cells": len(coverage), "raw_uncovered_cells": int((~coverage["covered_by_proportional_core"]).sum()), "draft": str(DRAFT)}, indent=2))


def refresh_evidence() -> None:
    """Add representative evidence without modifying any annotation labels."""
    for path in [PAIR_MASTER, EDA_CORPUS, CLUSTER_REVIEW, BRIDGE_REVIEW, DRAFT]:
        if not path.exists():
            raise FileNotFoundError(path)
    clusters = read_csv(CLUSTER_REVIEW)
    bridges = read_csv(BRIDGE_REVIEW)
    evidence = build_cluster_evidence(clusters, bridges)
    write_csv(evidence, CLUSTER_EVIDENCE)
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    draft["rules"]["fuzzy_duplicates"]["cluster_evidence_path"] = CLUSTER_EVIDENCE.name
    draft["fuzzy_decision"].update(
        {
            "representative_evidence_rows": len(evidence),
            "review_order": "bridge_edges_then_cluster_decision",
            "oversized_component_approval": "explicit_confirmation_and_required_note",
        }
    )
    draft["evidence_refreshed_utc"] = utc_now()
    DRAFT.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "cluster_evidence": str(CLUSTER_EVIDENCE),
                "rows": len(evidence),
                "components": int(evidence["component_id"].nunique()),
                "draft_updated": str(DRAFT),
                "annotation_labels_modified": False,
            },
            indent=2,
        )
    )


def finalize() -> None:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    if not CLUSTER_EVIDENCE.exists():
        raise FileNotFoundError(CLUSTER_EVIDENCE)
    clusters = read_csv(CLUSTER_REVIEW)
    bridges = read_csv(BRIDGE_REVIEW)
    required_clusters = clusters.loc[clusters["review_required"].str.lower().eq("true")]
    invalid_cluster_labels = required_clusters.loc[~required_clusters["manual_cluster_valid"].str.lower().isin(ALLOWED_REVIEW)]
    invalid_bridge_labels = bridges.loc[~bridges["manual_same_template"].str.lower().isin(ALLOWED_REVIEW)]
    missing_cluster_reviewer = required_clusters.loc[required_clusters["reviewer"].str.strip().eq("")]
    missing_bridge_reviewer = bridges.loc[bridges["reviewer"].str.strip().eq("")]
    blockers = []
    if not invalid_cluster_labels.empty:
        blockers.append(f"pending_cluster_labels:{len(invalid_cluster_labels)}")
    if not invalid_bridge_labels.empty:
        blockers.append(f"pending_bridge_labels:{len(invalid_bridge_labels)}")
    if not missing_cluster_reviewer.empty:
        blockers.append(f"missing_cluster_reviewer:{len(missing_cluster_reviewer)}")
    if not missing_bridge_reviewer.empty:
        blockers.append(f"missing_bridge_reviewer:{len(missing_bridge_reviewer)}")
    if blockers:
        raise ValueError("Decision v04 remains blocked: " + ", ".join(blockers))
    record = dict(draft)
    record["record_version"] = "v04"
    record["finalized_utc"] = utc_now()
    record["fuzzy_decision"] = {
        "status": "accepted_with_c_lite_guards",
        "cluster_label_counts": required_clusters["manual_cluster_valid"].str.lower().value_counts().to_dict(),
        "bridge_label_counts": bridges["manual_same_template"].str.lower().value_counts().to_dict(),
    }
    record["governance_inputs"] = {
        **record.pop("parent_governance_inputs"),
        PAIR_MASTER.name: {"rows": len(read_csv(PAIR_MASTER)), "sha256": sha256_file(PAIR_MASTER)},
        CLUSTER_REVIEW.name: {"rows": len(clusters), "sha256": sha256_file(CLUSTER_REVIEW)},
        BRIDGE_REVIEW.name: {"rows": len(bridges), "sha256": sha256_file(BRIDGE_REVIEW)},
        CLUSTER_EVIDENCE.name: {"rows": len(read_csv(CLUSTER_EVIDENCE)), "sha256": sha256_file(CLUSTER_EVIDENCE)},
        COVERAGE_ANALYSIS.name: {"rows": len(read_csv(COVERAGE_ANALYSIS)), "sha256": sha256_file(COVERAGE_ANALYSIS)},
    }
    record["build_gate"] = {"scope": "build_seed_v052_candidate", "passed": True, "blockers": []}
    FINAL.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"decision_record": str(FINAL), "sha256": sha256_file(FINAL), "build_gate_passed": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "evidence", "finalize"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.force)
    elif args.command == "evidence":
        refresh_evidence()
    else:
        finalize()


if __name__ == "__main__":
    main()
