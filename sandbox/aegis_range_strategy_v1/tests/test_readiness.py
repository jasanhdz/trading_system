from __future__ import annotations

import gzip
import zipfile
from datetime import datetime, timezone

import pytest

from aegis_range_v1.readiness import PartitionAccessError, SealedPartitionGuard, _deterministic_gzip_csv, _funding_events


def test_partition_guards_default_closed_and_require_exact_opt_in():
    assert SealedPartitionGuard.access_flags({}) == {
        "TRAIN": False,
        "CALIBRATION": False,
        "VALIDATION": False,
        "HOLDOUT": False,
    }
    for name in ("TRAIN", "CALIBRATION", "VALIDATION", "HOLDOUT"):
        with pytest.raises(PartitionAccessError, match=f"{name}_SEALED"):
            SealedPartitionGuard.require(name, {})
        SealedPartitionGuard.require(name, {f"{name}_ACCESS": "true"})


def test_unknown_partition_fails_closed():
    with pytest.raises(PartitionAccessError, match="PARTITION_UNKNOWN"):
        SealedPartitionGuard.require("ALL", {"ALL_ACCESS": "true"})


def test_derived_gzip_is_byte_deterministic(tmp_path):
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    rows = [("BTCUSDT", datetime(2030, 1, 1, tzinfo=timezone.utc).isoformat(), "100.0")]
    first_result = _deterministic_gzip_csv(first, ("symbol", "open_time", "close"), rows)
    second_result = _deterministic_gzip_csv(second, ("symbol", "open_time", "close"), rows)
    assert first_result == second_result
    assert first.read_bytes() == second.read_bytes()
    assert gzip.decompress(first.read_bytes()).startswith(b"symbol,open_time,close\n")


def test_funding_calc_jitter_is_preserved_and_canonicalized(tmp_path):
    archive_path = tmp_path / "BTCUSDT-fundingRate-2024-01.zip"
    member = "BTCUSDT-fundingRate-2024-01.csv"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, "calc_time,funding_interval_hours,last_funding_rate\n1704096000007,8,0.0001\n")
    events = _funding_events({"file": str(archive_path), "zip_members": [member]})
    source_calc_time, funding_at, rate = events[0]
    assert source_calc_time.microsecond == 7000
    assert funding_at.microsecond == 0
    assert rate == "0.0001"
