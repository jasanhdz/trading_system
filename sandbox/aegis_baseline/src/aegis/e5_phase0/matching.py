"""Synthetic C1/C2 filtering and deterministic augmenting-path matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .core import c1_seed, namespaced_seed, normalize_fold, normalize_horizon, normalize_symbol
from .errors import Phase0Error


@dataclass(frozen=True)
class C1FilterResult:
    experimental_symbol: str
    pre_filter_count: int
    self_match_exclusions: int
    candidates: tuple[str, ...]


def filter_c1_candidates(experimental_symbol: str, candidates: Iterable[str]) -> C1FilterResult:
    experimental = normalize_symbol(experimental_symbol)
    normalized = tuple(normalize_symbol(candidate) for candidate in candidates)
    retained = tuple(sorted(candidate for candidate in normalized if candidate != experimental))
    excluded = len(normalized) - len(retained)
    if not retained:
        raise Phase0Error("C1_NO_DISTINCT_SYMBOL_CONTROL", experimental)
    return C1FilterResult(experimental, len(normalized), excluded, retained)


def select_c1_control(experimental_symbol: str, candidates: Iterable[str], replicate_index: int) -> tuple[str, C1FilterResult]:
    filtered = filter_c1_candidates(experimental_symbol, candidates)
    generator = np.random.Generator(np.random.PCG64(c1_seed(replicate_index)))
    selected = filtered.candidates[int(generator.integers(0, len(filtered.candidates)))]
    validate_c1_assignment(filtered.experimental_symbol, selected)
    return selected, filtered


def validate_c1_assignment(experimental_symbol: str, control_symbol: str) -> None:
    if normalize_symbol(experimental_symbol) == normalize_symbol(control_symbol):
        raise Phase0Error("C1_SELF_MATCH_DETECTED", experimental_symbol)


@dataclass(frozen=True)
class C2Edge:
    left_id: str
    candidate_symbol: str
    candidate_cycle_id: str

    @property
    def right_id(self) -> str:
        return f"{normalize_symbol(self.candidate_symbol)}|{self.candidate_cycle_id}"


@dataclass(frozen=True)
class C2Left:
    left_id: str
    experimental_symbol: str
    experimental_cycle_id: str


@dataclass(frozen=True)
class C2FilterResult:
    pre_filter_count: int
    self_edge_exclusions: int
    edges: tuple[C2Edge, ...]


def filter_c2_self_edges(left_nodes: Mapping[str, C2Left], edges: Iterable[C2Edge]) -> C2FilterResult:
    normalized: list[C2Edge] = []
    excluded = 0
    total = 0
    for edge in edges:
        total += 1
        left = left_nodes.get(edge.left_id)
        if left is None:
            raise Phase0Error("C2_MATCHING_INFEASIBLE", f"unknown left node {edge.left_id}")
        candidate_symbol = normalize_symbol(edge.candidate_symbol)
        if candidate_symbol == normalize_symbol(left.experimental_symbol) and edge.candidate_cycle_id == left.experimental_cycle_id:
            excluded += 1
            continue
        normalized.append(C2Edge(edge.left_id, candidate_symbol, edge.candidate_cycle_id))
    retained = tuple(sorted(normalized, key=lambda edge: (edge.left_id, edge.candidate_symbol, edge.candidate_cycle_id)))
    if not retained:
        raise Phase0Error("C2_NO_DISTINCT_SYMBOL_CYCLE_CONTROL", "all C2 edges were exact self-edges")
    return C2FilterResult(total, excluded, retained)


def validate_c2_graph(left_nodes: Mapping[str, C2Left], edges: Sequence[C2Edge]) -> None:
    for edge in edges:
        left = left_nodes[edge.left_id]
        if normalize_symbol(edge.candidate_symbol) == normalize_symbol(left.experimental_symbol) and edge.candidate_cycle_id == left.experimental_cycle_id:
            raise Phase0Error("C2_SELF_EDGE_DETECTED", edge.left_id)


def validate_c2_assignment(left: C2Left, edge: C2Edge) -> None:
    if normalize_symbol(edge.candidate_symbol) == normalize_symbol(left.experimental_symbol) and edge.candidate_cycle_id == left.experimental_cycle_id:
        raise Phase0Error("C2_SELF_EDGE_DETECTED", left.left_id)


def randomized_augmenting_path_match(
    left_nodes: Sequence[C2Left],
    edges: Iterable[C2Edge],
    *,
    horizon: str,
    fold: int | str,
    month: str,
    replicate_index: int,
) -> dict[str, C2Edge]:
    canonical_left = tuple(sorted(left_nodes, key=lambda item: item.left_id))
    left_map = {item.left_id: item for item in canonical_left}
    if len(left_map) != len(canonical_left):
        raise Phase0Error("DUPLICATE_IDENTITY", "duplicate C2 left node")
    filtered = filter_c2_self_edges(left_map, edges)
    validate_c2_graph(left_map, filtered.edges)
    adjacency: dict[str, list[C2Edge]] = {left.left_id: [] for left in canonical_left}
    for edge in filtered.edges:
        adjacency[edge.left_id].append(edge)

    seed, _ = namespaced_seed("C2", normalize_horizon(horizon), normalize_fold(fold), month, replicate_index)
    generator = np.random.Generator(np.random.PCG64(seed))
    for left_id in sorted(adjacency):
        candidates = sorted(adjacency[left_id], key=lambda item: (item.candidate_symbol, item.candidate_cycle_id))
        order = generator.permutation(len(candidates)) if candidates else np.array([], dtype=int)
        adjacency[left_id] = [candidates[int(index)] for index in order]

    right_owner: dict[str, str] = {}
    assignment: dict[str, C2Edge] = {}

    def augment(left_id: str, visited: set[str]) -> bool:
        for edge in adjacency[left_id]:
            right_id = edge.right_id
            if right_id in visited:
                continue
            visited.add(right_id)
            previous = right_owner.get(right_id)
            if previous is None or augment(previous, visited):
                right_owner[right_id] = left_id
                assignment[left_id] = edge
                return True
        return False

    for left in canonical_left:
        if not augment(left.left_id, set()):
            raise Phase0Error("C2_MATCHING_INFEASIBLE", f"unmatched left node {left.left_id}")
    if len({edge.right_id for edge in assignment.values()}) != len(assignment):
        raise Phase0Error("C2_MATCHING_INFEASIBLE", "right-side reuse detected")
    for left_id, edge in assignment.items():
        validate_c2_assignment(left_map[left_id], edge)
    return {left_id: assignment[left_id] for left_id in sorted(assignment)}
