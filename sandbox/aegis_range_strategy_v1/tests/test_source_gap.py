from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aegis_range_v1.source_gap import DailyAudit, DailyRequest, apply_precedence, audit_status


def test_daily_request_is_exact_official_scope():
    request = DailyRequest("BTCUSDT", "2026-06-29")
    assert request.official_path == "data/futures/um/daily/markPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2026-06-29.zip"
    assert request.url == "https://data.binance.vision/" + request.official_path
    assert request.member == "BTCUSDT-1m-2026-06-29.csv"


def test_monthly_primary_daily_gap_fill_and_conflict_recording():
    origin = datetime(2030, 1, 1, tzinfo=timezone.utc)
    first = ("100", "101", "99", "100.5")
    changed = ("100", "102", "99", "100.5")
    fill = ("101", "102", "100", "101.5")
    monthly = {origin: first}
    daily = {origin: changed, origin + timedelta(minutes=1): fill}
    merged, recovered, missing = apply_precedence(
        monthly,
        daily,
        (origin, origin + timedelta(minutes=1), origin + timedelta(minutes=2)),
    )
    assert merged[origin] == first
    assert merged[origin + timedelta(minutes=1)] == fill
    assert recovered == (origin + timedelta(minutes=1),)
    assert missing == (origin + timedelta(minutes=2),)


def test_incomplete_official_daily_blocks_gap_resolution():
    audit = DailyAudit(
        "BTCUSDT", "2024-08-12", True, "path", "/tmp/file", "0" * 64, 1,
        "member.csv", "00000000", True, 1438, 1438,
        "2024-08-12T00:00:00.000Z", "2024-08-12T23:59:00.000Z",
        0, 0, 1438, 1438, 0, 2, 0, 2,
    )
    assert audit_status([audit]) == "AEGIS_RANGE_R2_SOURCE_GAP_RESOLUTION_BLOCKED_BY_OFFICIAL_SOURCE_GAP"


def test_non_funding_gap_does_not_block_contractual_resolution():
    audits = []
    for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT"):
        audits.append(DailyAudit(symbol, "2024-08-12", True, "path", "/tmp/file", "0" * 64, 1, "member.csv", "00000000", True, 1438, 1438, "start", "end", 0, 0, 1438, 1438, 0, 2, 0, 2))
        audits.append(DailyAudit(symbol, "2026-06-29", True, "path", "/tmp/file", "0" * 64, 1, "member.csv", "00000000", True, 1440, 1440, "start", "end", 0, 0, 0, 0, 0, 1440, 1440, 0))
    assert audit_status(audits) == "DAILY_VALID_FOR_CONTRACTUAL_GAP_FILL"
