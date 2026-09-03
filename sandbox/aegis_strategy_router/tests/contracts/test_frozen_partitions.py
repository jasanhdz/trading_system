from datetime import datetime, timezone

from aegis_strategy_router.audit.coverage import FRESH_TRAIN_START, audit_episode_ids, partition_at


def test_frozen_partition_boundaries_are_half_open() -> None:
    assert partition_at(datetime(2026, 8, 17, 21, 14, 26, 92_999, tzinfo=timezone.utc)) == "DISCOVERY_QUARANTINE"
    assert partition_at(FRESH_TRAIN_START) == "FRESH_TRAIN"
    assert partition_at(datetime(2026, 11, 1, tzinfo=timezone.utc)) == "FRESH_CALIBRATION"
    assert partition_at(datetime(2027, 1, 16, tzinfo=timezone.utc)) == "FINAL_SYSTEM_HOLDOUT"


def test_episode_audit_counts_ids_not_rows() -> None:
    audit = audit_episode_ids(["a", "a", "b", "c", "c"])
    assert audit.rows == 5
    assert audit.independent_episodes == 3
    assert audit.duplicate_rows == 2
