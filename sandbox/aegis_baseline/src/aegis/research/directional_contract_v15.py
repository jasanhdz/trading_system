"""Direction-specific feature contracts and fail-closed V15 selection."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .competing_barrier_v10 import BarrierResearchError
from .decomposed_entry_v9 import V9_FEATURE_NAMES
from .feature_information_v14 import feature_families


def contract_indices(config: Mapping[str, Any], side: str) -> tuple[int, ...]:
    if side not in {"BASELINE", "LONG", "SHORT"}:
        raise BarrierResearchError("invalid V15 contract side")
    if side == "BASELINE":
        return tuple(range(len(V9_FEATURE_NAMES)))
    contracts = config["contracts"]
    removed = {
        int(item["index"]) for item in contracts["deduplicated"]["remove_positions"]
    }
    if any(index < 0 or index >= len(V9_FEATURE_NAMES) for index in removed):
        raise BarrierResearchError("invalid V15 deduplication position")
    for item in contracts["deduplicated"]["remove_positions"]:
        index = int(item["index"])
        if V9_FEATURE_NAMES[index] != str(item["name"]):
            raise BarrierResearchError("V15 feature name/index authority mismatch")
    family_names = feature_families()
    for family in contracts["directional"][side]["remove_families"]:
        try:
            names = set(family_names[str(family)])
        except KeyError as exc:
            raise BarrierResearchError("unknown V15 feature family") from exc
        removed.update(
            index for index, name in enumerate(V9_FEATURE_NAMES) if name in names
        )
    indices = tuple(
        index for index in range(len(V9_FEATURE_NAMES)) if index not in removed
    )
    if not indices:
        raise BarrierResearchError("V15 contract removed every feature")
    return indices


def entry_quality_score(
    *,
    clean_probability: float,
    danger_probability: float,
    mae_q90: float,
    adverse: float,
) -> float:
    values = (clean_probability, danger_probability, mae_q90, adverse)
    if (
        not all(math.isfinite(float(value)) for value in values)
        or not 0.0 <= clean_probability <= 1.0
        or not 0.0 <= danger_probability <= 1.0
        or mae_q90 < 0.0
        or adverse <= 0.0
    ):
        raise BarrierResearchError("invalid V15 entry-quality score inputs")
    return float(clean_probability - danger_probability - mae_q90 / adverse)


def select_at_most_one_per_timestamp(
    rows: Sequence[Mapping[str, Any]], *, minimum_score: float
) -> tuple[bool, ...]:
    if not math.isfinite(minimum_score):
        raise BarrierResearchError("invalid V15 selection threshold")
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        score = float(row["score"])
        if not math.isfinite(score):
            raise BarrierResearchError("non-finite V15 score")
        if score >= minimum_score:
            grouped[str(row["timestamp"])].append((index, row))
    selected = [False] * len(rows)
    for candidates in grouped.values():
        candidates.sort(
            key=lambda item: (
                -float(item[1]["score"]),
                float(item[1]["danger_probability"]),
                float(item[1]["mae_q90"]),
                str(item[1]["symbol"]),
            )
        )
        selected[candidates[0][0]] = True
    return tuple(selected)
