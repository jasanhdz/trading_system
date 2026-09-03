from __future__ import annotations

from datetime import datetime, timedelta, timezone

from training.train_temporal_stability_v13_research import _recent_rows


def test_recent_window_prefers_time_bounded_rows() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [{"timestamp_value": start + timedelta(days=index)} for index in range(200)]
    result = _recent_rows(rows, days=30, minimum_rows=10)
    assert len(result) == 31
    assert result[0]["timestamp_value"] == start + timedelta(days=169)


def test_recent_window_falls_back_to_minimum_tail() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [{"timestamp_value": start + timedelta(days=index)} for index in range(20)]
    result = _recent_rows(rows, days=2, minimum_rows=10)
    assert len(result) == 10
    assert result[0]["timestamp_value"] == start + timedelta(days=10)
