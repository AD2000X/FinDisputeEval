"""Version-neutral deterministic sampling and duplicate utilities.

These functions contain mechanics only. Release policy remains in a versioned
decision record and builder. Multi-column strata are never flattened into a
delimiter-joined key, and every allocation can be exported for audit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def deterministic_key(seed: int, namespace: str, *values: object) -> str:
    payload = json.dumps(
        [int(seed), str(namespace), *[str(value) for value in values]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_stratum_values(frame: pd.DataFrame, strata: list[str]) -> pd.DataFrame:
    missing = sorted(set(strata) - set(frame.columns))
    if missing:
        raise ValueError(f"Sampling strata are missing from corpus: {missing}")
    normalized = frame[strata].copy()
    for column in strata:
        normalized[column] = normalized[column].astype("string").fillna("<NA>")
    return normalized


def proportional_allocation(
    frame: pd.DataFrame,
    n: int,
    strata: list[str],
) -> pd.DataFrame:
    """Allocate ``n`` rows by largest remainder with an explicit tie-break."""
    if n < 0:
        raise ValueError("Sample size cannot be negative")
    if not strata:
        raise ValueError("At least one stratum column is required")
    normalized = _normalized_stratum_values(frame, strata)
    if frame.empty:
        columns = [*strata, "available", "ideal", "allocated", "fractional_remainder", "inclusion_probability", "design_weight"]
        return pd.DataFrame(columns=columns)
    grouped = normalized.groupby(strata, sort=True, dropna=False).size().rename("available").reset_index()
    target = min(int(n), len(frame))
    grouped["ideal"] = grouped["available"] / len(frame) * target
    grouped["allocated"] = np.floor(grouped["ideal"]).astype(int)
    grouped["fractional_remainder"] = grouped["ideal"] - grouped["allocated"]
    grouped["_tie_key"] = grouped[strata].apply(
        lambda row: json.dumps(row.astype(str).tolist(), ensure_ascii=False, separators=(",", ":")),
        axis=1,
    )
    remainder = target - int(grouped["allocated"].sum())
    order = grouped.sort_values(
        ["fractional_remainder", "_tie_key"],
        ascending=[False, True],
        kind="mergesort",
    ).index
    for index in order[:remainder]:
        grouped.at[index, "allocated"] += 1
    grouped["inclusion_probability"] = grouped["allocated"] / grouped["available"]
    grouped["design_weight"] = 1.0 / grouped["inclusion_probability"].replace(0, np.nan)
    return grouped.drop(columns="_tie_key").sort_values(strata, kind="mergesort").reset_index(drop=True)


def proportional_stratified_sample_with_accounting(
    frame: pd.DataFrame,
    n: int,
    strata: list[str],
    random_seed: int,
    *,
    namespace: str = "proportional_sample",
    id_column: str = "Complaint ID",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if id_column not in frame.columns:
        raise ValueError(f"Sampling ID column is missing: {id_column}")
    allocation = proportional_allocation(frame, n, strata)
    if frame.empty or int(allocation["allocated"].sum()) == 0:
        return frame.head(0).copy(), allocation
    normalized = _normalized_stratum_values(frame, strata)
    working = frame.copy()
    for column in strata:
        working[f"__stratum_{column}"] = normalized[column]
    working["__selection_key"] = working[id_column].map(
        lambda value: deterministic_key(random_seed, namespace, value)
    )
    parts: list[pd.DataFrame] = []
    group_columns = [f"__stratum_{column}" for column in strata]
    allocation_by_key = {
        tuple(str(row[column]) for column in strata): row
        for row in allocation.to_dict(orient="records")
    }
    for key, group in working.groupby(group_columns, sort=True, dropna=False):
        key_tuple = (str(key),) if len(strata) == 1 else tuple(str(value) for value in key)
        row = allocation_by_key[key_tuple]
        take = int(row["allocated"])
        if take == 0:
            continue
        chosen = group.sort_values("__selection_key", kind="mergesort").head(take).copy()
        chosen["sampling_inclusion_probability"] = float(row["inclusion_probability"])
        chosen["sampling_design_weight"] = float(row["design_weight"])
        parts.append(chosen)
    result = pd.concat(parts, ignore_index=True)
    internal = [column for column in result if column.startswith("__")]
    result = result.drop(columns=internal)
    if len(result) != min(max(int(n), 0), len(frame)):
        raise ValueError("Proportional allocation did not reconcile to selected rows")
    return result, allocation


def proportional_stratified_sample(
    frame: pd.DataFrame,
    n: int,
    strata: list[str],
    random_seed: int,
) -> pd.DataFrame:
    selected, _ = proportional_stratified_sample_with_accounting(
        frame, n, strata, random_seed
    )
    return selected


def deterministic_group_cap(
    frame: pd.DataFrame,
    group_column: str,
    cap: int,
    random_seed: int,
    namespace: str,
    *,
    id_column: str = "Complaint ID",
) -> tuple[pd.DataFrame, dict[str, int]]:
    if cap < 1:
        raise ValueError("Duplicate cap must be positive")
    missing = sorted({group_column, id_column} - set(frame.columns))
    if missing:
        raise ValueError(f"Duplicate-cap columns are missing: {missing}")
    working = frame.copy()
    working["__selection_key"] = working[id_column].map(
        lambda value: deterministic_key(random_seed, namespace, value)
    )
    working = working.sort_values("__selection_key", kind="mergesort")
    retained = working.loc[working.groupby(group_column, dropna=False).cumcount().lt(cap)].copy()
    retained = retained.drop(columns="__selection_key")
    return retained, {
        "before": len(frame),
        "after": len(retained),
        "removed": len(frame) - len(retained),
        "groups": int(frame[group_column].nunique(dropna=False)),
        "cap": int(cap),
    }


def select_coverage_supplement(
    candidates: pd.DataFrame,
    proportional_core: pd.DataFrame,
    strata: list[str],
    random_seed: int,
    *,
    id_column: str = "Complaint ID",
    maximum_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one deterministic row for each eligible cell absent from the core."""
    candidate_keys = _normalized_stratum_values(candidates, strata)
    core_keys = _normalized_stratum_values(proportional_core, strata)
    candidate_tuples = pd.MultiIndex.from_frame(candidate_keys)
    core_tuples = set(pd.MultiIndex.from_frame(core_keys).tolist())
    uncovered = candidates.loc[[key not in core_tuples for key in candidate_tuples]].copy()
    normalized_uncovered = _normalized_stratum_values(uncovered, strata)
    for column in strata:
        uncovered[f"__stratum_{column}"] = normalized_uncovered[column]
    uncovered["__selection_key"] = uncovered[id_column].map(
        lambda value: deterministic_key(random_seed, "coverage_supplement", value)
    )
    group_columns = [f"__stratum_{column}" for column in strata]
    selected = (
        uncovered.sort_values("__selection_key", kind="mergesort")
        .drop_duplicates(group_columns, keep="first")
        .copy()
    )
    if maximum_rows is not None and len(selected) > int(maximum_rows):
        raise ValueError(
            f"Coverage supplement exceeds approved maximum: {len(selected)} > {maximum_rows}"
        )
    selected["sampling_inclusion_probability"] = np.nan
    selected["sampling_design_weight"] = np.nan
    accounting = selected[group_columns].copy()
    accounting.columns = strata
    accounting["available_candidates"] = (
        uncovered.groupby(group_columns, dropna=False)[id_column]
        .transform("size")
        .loc[selected.index]
        .to_numpy()
    )
    accounting["selected"] = 1
    accounting["prevalence_eligible"] = False
    internal = [column for column in selected if column.startswith("__")]
    return selected.drop(columns=internal).reset_index(drop=True), accounting.reset_index(drop=True)


class _UnionFind:
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


def fuzzy_component_id(nodes: Iterable[str]) -> str:
    payload = "|".join(sorted(str(node) for node in nodes)).encode("utf-8")
    return "fuzzy_v04_" + hashlib.sha256(payload).hexdigest()[:16]


def audited_fuzzy_group_cap(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    cluster_review: pd.DataFrame,
    bridge_review: pd.DataFrame,
    *,
    cap: int,
    random_seed: int,
    maximum_unapproved_component_size: int,
    id_column: str = "Complaint ID",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply C-lite fuzzy caps after cluster and selected-bridge review.

    A required cluster marked ``no`` or ``uncertain`` is bypassed entirely.
    A reviewed bridge marked anything other than ``yes`` is removed before
    components are rebuilt. Oversized components require explicit cluster yes.
    """
    confirmed = pairs.loc[
        pairs["manual_same_template"].astype(str).str.lower().isin(["yes", "true", "1"])
    ].copy()
    original = _UnionFind()
    edges: list[tuple[str, str]] = []
    for left, right in confirmed[["complaint_id_a", "complaint_id_b"]].itertuples(index=False, name=None):
        edge = tuple(sorted((str(left), str(right))))
        edges.append(edge)
        original.union(*edge)
    nodes_by_root: dict[str, set[str]] = {}
    for node in original.parent:
        nodes_by_root.setdefault(original.find(node), set()).add(node)
    node_to_original_component = {
        node: fuzzy_component_id(nodes_by_root[original.find(node)]) for node in original.parent
    }
    cluster_labels = {
        str(row["component_id"]): str(row["manual_cluster_valid"]).lower()
        for row in cluster_review.to_dict(orient="records")
    }
    bypass_components = {
        component for component, label in cluster_labels.items() if label in {"no", "uncertain"}
    }
    bridge_labels = {
        tuple(sorted((str(row["complaint_id_a"]), str(row["complaint_id_b"])))): str(row["manual_same_template"]).lower()
        for row in bridge_review.to_dict(orient="records")
    }
    effective = _UnionFind()
    removed_reviewed_edges = 0
    bypassed_edges = 0
    for edge in edges:
        component = node_to_original_component[edge[0]]
        if component in bypass_components:
            bypassed_edges += 1
            continue
        if edge in bridge_labels and bridge_labels[edge] != "yes":
            removed_reviewed_edges += 1
            continue
        effective.union(*edge)
    rebuilt_nodes: dict[str, set[str]] = {}
    for node in effective.parent:
        rebuilt_nodes.setdefault(effective.find(node), set()).add(node)
    for nodes in rebuilt_nodes.values():
        if len(nodes) <= int(maximum_unapproved_component_size):
            continue
        original_components = {node_to_original_component[node] for node in nodes}
        if any(cluster_labels.get(component) != "yes" for component in original_components):
            raise ValueError("Oversized fuzzy component lacks explicit manual approval")
    working = frame.copy()
    working["confirmed_fuzzy_cluster"] = working[id_column].map(
        lambda value: effective.find(str(value)) if str(value) in effective.parent else f"singleton:{value}"
    )
    retained, cap_accounting = deterministic_group_cap(
        working,
        "confirmed_fuzzy_cluster",
        cap,
        random_seed,
        "audited_fuzzy_cap",
        id_column=id_column,
    )
    return retained, {
        **cap_accounting,
        "confirmed_input_edges": len(edges),
        "reviewed_edges_removed": removed_reviewed_edges,
        "bypassed_component_edges": bypassed_edges,
        "bypassed_components": len(bypass_components),
        "rebuilt_components": len(rebuilt_nodes),
        "largest_rebuilt_component": max((len(nodes) for nodes in rebuilt_nodes.values()), default=1),
    }
