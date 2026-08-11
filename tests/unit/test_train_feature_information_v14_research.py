from __future__ import annotations

from datetime import datetime, timedelta, timezone

from train_feature_information_v14_research import _folds, _improvements, _split


def test_v14_folds_are_expanding_and_temporally_ordered() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    folds = _folds([start + timedelta(hours=index) for index in range(100)])
    assert len(folds) == 4
    assert all(train < test for train, test in folds)
    assert [train for train, _ in folds] == sorted(train for train, _ in folds)


def test_v14_split_enforces_embargo_and_independent_test_rows() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"timestamp": start, "independent": True},
        {"timestamp": start + timedelta(minutes=60), "independent": True},
        {"timestamp": start + timedelta(minutes=121), "independent": False},
        {"timestamp": start + timedelta(minutes=122), "independent": True},
    ]
    train, test = _split(
        rows,
        (start + timedelta(minutes=60), start + timedelta(minutes=180)),
        embargo_minutes=60,
    )
    assert len(train) == 2
    assert test == [rows[-1]]


def test_v14_improvement_counts_require_correct_metric_direction() -> None:
    baseline = [
        {"log_loss": 0.5, "average_precision": 0.4},
        {"log_loss": 0.5, "average_precision": 0.4},
    ]
    candidate = [
        {"log_loss": 0.4, "average_precision": 0.5},
        {"log_loss": 0.6, "average_precision": 0.3},
    ]
    assert _improvements(baseline, candidate) == {
        "log_loss": 1,
        "average_precision": 1,
    }
