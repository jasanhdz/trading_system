"""Pairwise trajectory preferences for V16 economic ranking research."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .competing_barrier_v10 import BarrierResearchError


def trajectory_tier(row: Mapping[str, Any]) -> int:
    utility = float(row["actual_utility"])
    if not math.isfinite(utility):
        raise BarrierResearchError("non-finite V16 utility")
    adverse = bool(row["danger"])
    clean = bool(row["clean"])
    if clean and utility > 0.0 and not adverse:
        return 3
    if utility > 0.0 and not adverse:
        return 2
    if not adverse:
        return 1
    return 0


def preference_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    mae = float(row["mae_fraction"])
    underwater = int(row["time_underwater_bars"])
    if not math.isfinite(mae) or mae < 0.0 or underwater < 0:
        raise BarrierResearchError("invalid V16 trajectory evidence")
    return (
        float(trajectory_tier(row)),
        float(row["actual_utility"]),
        -mae,
        -float(underwater),
    )


def pairwise_examples(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, Mapping[str, int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["timestamp"])].append(row)
    differences = []
    labels = []
    skipped_ties = 0
    unordered_pairs = 0
    for timestamp in sorted(grouped):
        candidates = sorted(grouped[timestamp], key=lambda row: str(row["symbol"]))
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                left_key = preference_key(left)
                right_key = preference_key(right)
                if left_key == right_key:
                    skipped_ties += 1
                    continue
                left_values = np.asarray(
                    [left["features"][index] for index in indices], dtype=np.float32
                )
                right_values = np.asarray(
                    [right["features"][index] for index in indices], dtype=np.float32
                )
                difference = left_values - right_values
                label = int(left_key > right_key)
                differences.extend((difference, -difference))
                labels.extend((label, 1 - label))
                unordered_pairs += 1
    if not differences or set(labels) != {0, 1}:
        raise BarrierResearchError("V16 has insufficient pairwise evidence")
    return (
        np.asarray(differences, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        {
            "timestamps": len(grouped),
            "unordered_pairs": unordered_pairs,
            "oriented_pairs": len(labels),
            "skipped_exact_ties": skipped_ties,
        },
    )


def pairwise_accuracy(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float]
) -> Mapping[str, float | int]:
    if len(rows) != len(scores):
        raise BarrierResearchError("V16 score width mismatch")
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        if not math.isfinite(float(score)):
            raise BarrierResearchError("non-finite V16 ranking score")
        grouped[str(row["timestamp"])].append((row, float(score)))
    correct = 0
    compared = 0
    predicted_ties = 0
    for candidates in grouped.values():
        candidates.sort(key=lambda item: str(item[0]["symbol"]))
        for left_index, (left, left_score) in enumerate(candidates):
            for right, right_score in candidates[left_index + 1 :]:
                left_key = preference_key(left)
                right_key = preference_key(right)
                if left_key == right_key:
                    continue
                compared += 1
                if math.isclose(left_score, right_score, abs_tol=1e-12):
                    predicted_ties += 1
                elif (left_score > right_score) == (left_key > right_key):
                    correct += 1
    if not compared:
        raise BarrierResearchError("V16 has no comparable scoring pairs")
    return {
        "compared_pairs": compared,
        "correct_pairs": correct,
        "predicted_ties": predicted_ties,
        "accuracy": correct / compared,
    }
