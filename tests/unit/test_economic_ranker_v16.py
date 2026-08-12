from __future__ import annotations

from aegis.research.economic_ranker_v16 import (
    pairwise_accuracy,
    pairwise_examples,
    preference_key,
    trajectory_tier,
)
from train_economic_ranker_v16_research import _rank_selected


def _row(
    symbol: str,
    *,
    utility: float,
    danger: bool = False,
    clean: bool = False,
    mae: float = 0.01,
    underwater: int = 2,
    feature: float = 0.0,
):
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "symbol": symbol,
        "actual_utility": utility,
        "danger": danger,
        "clean": clean,
        "mae_fraction": mae,
        "time_underwater_bars": underwater,
        "features": (feature,),
    }


def test_v16_trajectory_tiers_prioritize_canonical_path_quality() -> None:
    adverse = _row("A", utility=0.1, danger=True)
    non_adverse = _row("B", utility=-0.1)
    positive = _row("C", utility=0.1)
    clean = _row("D", utility=0.1, clean=True)
    assert [
        trajectory_tier(row) for row in (adverse, non_adverse, positive, clean)
    ] == [
        0,
        1,
        2,
        3,
    ]
    assert preference_key(clean) > preference_key(positive)


def test_v16_pairwise_examples_are_balanced_and_same_timestamp() -> None:
    rows = [
        _row("A", utility=-0.1, feature=-1.0),
        _row("B", utility=0.1, feature=1.0),
        _row("C", utility=0.2, clean=True, feature=2.0),
    ]
    matrix, labels, report = pairwise_examples(rows, (0,))
    assert matrix.shape == (6, 1)
    assert labels.tolist().count(0) == labels.tolist().count(1) == 3
    assert report["unordered_pairs"] == 3


def test_v16_pairwise_examples_never_cross_timestamps() -> None:
    first = [_row("A", utility=-0.1), _row("B", utility=0.1)]
    second = [
        {**_row("A", utility=0.2), "timestamp": "2026-01-01T01:00:00+00:00"},
        {**_row("B", utility=-0.2), "timestamp": "2026-01-01T01:00:00+00:00"},
    ]
    _, _, report = pairwise_examples([*first, *second], (0,))
    assert report["timestamps"] == 2
    assert report["unordered_pairs"] == 2


def test_v16_pairwise_accuracy_measures_cross_sectional_order() -> None:
    rows = [
        _row("A", utility=-0.1),
        _row("B", utility=0.1),
        _row("C", utility=0.2, clean=True),
    ]
    assert pairwise_accuracy(rows, [-1.0, 0.0, 1.0])["accuracy"] == 1.0
    assert pairwise_accuracy(rows, [1.0, 0.0, -1.0])["accuracy"] == 0.0


def test_v16_rank_selection_does_not_require_legacy_head_outputs() -> None:
    rows = [
        {**_row("A", utility=0.1), "score": 0.2},
        {**_row("B", utility=0.2), "score": 0.3},
    ]
    selected = _rank_selected(rows, 0.1)
    assert [row["symbol"] for row in selected] == ["B"]
