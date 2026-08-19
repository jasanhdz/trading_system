from datetime import datetime, timedelta, timezone

import pytest

from aegis_strategy_router.audit.phase2 import (
    FreshStreamRecord,
    OutcomeFieldProhibited,
    assert_label_free_fields,
    audit_phase2_coverage,
)


def test_phase2_audit_rejects_outcomes_and_future_metrics() -> None:
    assert_label_free_fields(("timestamp", "symbol", "snapshot_id", "event_count"))
    with pytest.raises(OutcomeFieldProhibited):
        assert_label_free_fields(("symbol", "future_mfe_bps"))
    with pytest.raises(OutcomeFieldProhibited):
        assert_label_free_fields(("realized_pnl",))


def test_phase2_coverage_is_label_free_and_deterministic() -> None:
    start = datetime(2026, 8, 17, 21, 14, 26, 93_000, tzinfo=timezone.utc)
    rows = (
        FreshStreamRecord(start, "SUIUSDT", "BOOK", True, 0.0),
        FreshStreamRecord(start + timedelta(milliseconds=17), "SUIUSDT", "QUOTE", True, 17.0),
        FreshStreamRecord(start + timedelta(seconds=1), "ADAUSDT", "TRADE", False, 1000.0),
    )
    first = audit_phase2_coverage(rows)
    second = audit_phase2_coverage(reversed(rows))
    assert first == second
    assert first.stream_rows == 3
    assert first.valid_stream_rows == 2
    assert dict(first.counts_by_partition) == {"FRESH_TRAIN": 3}
    assert first.candidate_episode_count == 0
