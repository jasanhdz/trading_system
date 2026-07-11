#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

import aegis_alpha.tools.gen2_d3_common as common
import aegis_alpha.tools.gen2_d3_snapshot as snap

BAR_MS = 5 * 60_000


def make_market(n_bars: int, start_ms: int = 1_700_000_000_000) -> dict[int, list]:
    bars = {}
    price = 100.0
    for i in range(n_bars):
        ot = start_ms + i * BAR_MS
        price += (1 if i % 3 else -1) * 0.1
        bars[ot] = [ot, price, price + 0.5, price - 0.5, price + 0.1, 10.0 + i % 7, ot + BAR_MS - 1, 1000.0, 42 + i, 5.0, 500.0]
    return bars


def fake_fetch_factory(market: dict[int, list], mutate_after: set[int] | None = None):
    calls = {"n": 0}

    def fake_fetch(symbol: str, start_ms: int, end_ms: int, limit: int = snap.PAGE_LIMIT, retries: int = 3):
        calls["n"] += 1
        rows = [list(v) for k, v in sorted(market.items()) if start_ms <= k <= end_ms][:limit]
        if mutate_after and calls["n"] > 1:
            for r in rows:
                if r[0] in mutate_after:
                    r[4] = float(r[4]) + 1.0  # mutated close on refetch
        return rows

    fake_fetch.calls = calls
    return fake_fetch


def test_fetch_pagination_and_closed_cutoff() -> None:
    market = make_market(3200)
    fetch = fake_fetch_factory(market)
    opens = sorted(market)
    cutoff = opens[-10] + BAR_MS  # last 10 bars not yet closed at cutoff
    df = snap.fetch_range("BTCUSDT", opens[0], opens[-1], cutoff, fetch, spacing=0)
    assert len(df) == 3200 - 9  # bars with close_time >= cutoff excluded
    assert df["open_time"].is_unique
    assert (df["open_time"].diff().dropna() == BAR_MS).all()
    assert fetch.calls["n"] >= 3  # paginated


def test_snapshot_immutability_and_manifest(monkey_root: Path | None = None) -> None:
    market = make_market(600)
    fetch = fake_fetch_factory(market)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "snaps"
        result = snap.build_snapshot(symbols=["BTCUSDT"], range_days=3, fetch_fn=fetch, snapshot_root=root, spacing=0, end_override_ms=max(market) + BAR_MS)
        d = Path(result["directory"])
        assert (d / "BTCUSDT_5m.csv").exists()
        assert (d / "snapshot_manifest.json").exists()
        ok, errors = common.verify_manifest_checksums(d, "snapshot_manifest")
        assert ok, errors
        # tamper detection
        csv = d / "BTCUSDT_5m.csv"
        csv.write_bytes(csv.read_bytes() + b"x")
        ok, errors = common.verify_manifest_checksums(d, "snapshot_manifest")
        assert not ok and any("BTCUSDT" in e for e in errors)


def _snapshot_for_g3(tmp: Path, mutate: set[int] | None):
    market = make_market(2000)
    fetch = fake_fetch_factory(market, mutate_after=mutate)
    root = tmp / "snaps"
    result = snap.build_snapshot(symbols=["BTCUSDT"], range_days=8, fetch_fn=fetch, snapshot_root=root, spacing=0, end_override_ms=max(market) + BAR_MS)
    d = Path(result["directory"])
    # backdate fetch_completed_at so the >=15 min wait passes
    manifest = json.loads((d / "snapshot_manifest.json").read_text())
    manifest["fetch_completed_at"] = str(pd.Timestamp(manifest["fetch_completed_at"]) - pd.Timedelta(minutes=30))
    (d / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2))
    return d, fetch, market


def test_g3_identical_refetch_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d, fetch, market = _snapshot_for_g3(Path(tmp), mutate=None)
        result = snap.refetch_check(d, fetch_fn=fetch, min_wait_minutes=15, spacing=0)
        assert result["passed"] is True
        assert result["bars_checked"] >= 500
        assert result["mismatch_count"] == 0


def test_g3_mutated_bar_fails() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        market_keys = sorted(make_market(2000))
        mutate = set(market_keys)  # every refetched closed bar differs
        d, fetch, market = _snapshot_for_g3(Path(tmp), mutate=mutate)
        result = snap.refetch_check(d, fetch_fn=fetch, min_wait_minutes=15, spacing=0)
        assert result["mismatch_count"] > 0
        assert result["passed"] is False


def test_g3_sample_is_deterministic() -> None:
    market = make_market(3000)
    df = snap.rows_to_frame([v for _, v in sorted(market.items())])
    cutoff = max(market) + BAR_MS
    a = snap.g3_targets(df, "snap-x", cutoff)
    b = snap.g3_targets(df, "snap-x", cutoff)
    c = snap.g3_targets(df, "snap-y", cutoff)
    assert a == b
    assert a != c
    assert len(a) >= 500


if __name__ == "__main__":
    # G3 reports go to the real reports root; redirect to tmp for the test run.
    with tempfile.TemporaryDirectory() as tmp:
        snap.REPORTS_ROOT = Path(tmp) / "reports"
        test_fetch_pagination_and_closed_cutoff()
        test_snapshot_immutability_and_manifest()
        test_g3_identical_refetch_passes()
        test_g3_mutated_bar_fails()
        test_g3_sample_is_deterministic()
    print("test_gen2_d3_snapshot: OK")
