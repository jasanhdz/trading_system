#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_phase_o_short_live_quality import (  # noqa: E402
    aggregate_symbols,
    classify_burst,
    classify_global,
    classify_symbol,
    compute_short_mae_mfe,
    hit_before_stop_short,
    pearson,
    roe_from_pnl,
    short_pnl,
    trade_quality_row,
    write_csv,
)


def candle(ts: str, high: float, low: float, close: float) -> dict:
    return {
        "timestamp": datetime.fromisoformat(ts.replace("Z", "+00:00")),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1.0,
    }


def trade() -> dict:
    return {
        "trade_id": "T1",
        "symbol": "ETHUSDT",
        "side": "SHORT",
        "opened_at": "2026-06-07T01:00:00Z",
        "entry_price": 100.0,
        "quantity": 2.0,
        "leverage": 20,
        "margin_estimated": 10.0,
        "notional_estimated": 200.0,
        "position_fraction": 0.5,
        "turbo_score": 0.75,
        "brackets_confirmed": True,
        "sl_price": 102.0,
        "tp_price": 97.5,
    }


def close_event(price: float = 98.0, pnl: float = 4.0) -> dict:
    return {
        "event": "TRADE_CLOSED",
        "timestamp": "2026-06-07T01:20:00Z",
        "price": price,
        "roe": 0.4,
        "reason": "SL/TP",
        "metadata": {"canonicalExitType": "TRAILING_STOP_EXIT", "pnl": pnl},
    }


def test_short_pnl_calculates_correctly() -> None:
    assert short_pnl(100.0, 98.0, 2.0) == 4.0
    assert short_pnl(100.0, 101.0, 2.0) == -2.0


def test_roe_uses_margin() -> None:
    assert roe_from_pnl(4.0, 10.0) == 0.4


def test_mfe_mae_short_correctly() -> None:
    candles = [
        candle("2026-06-07T01:00:00Z", 101.0, 99.0, 99.5),
        candle("2026-06-07T01:05:00Z", 100.5, 97.0, 98.0),
    ]
    out = compute_short_mae_mfe(100.0, 20.0, candles)
    assert round(out["mfe_price_move"], 4) == 0.03
    assert round(out["mae_price_move"], 4) == 0.01
    assert round(out["mfe_roe"], 4) == 0.6
    assert round(out["mae_roe"], 4) == 0.2


def test_mfe_mae_until_close_row() -> None:
    candles = [
        candle("2026-06-07T01:00:00Z", 100.5, 99.5, 99.8),
        candle("2026-06-07T01:05:00Z", 101.0, 97.0, 98.0),
        candle("2026-06-07T01:25:00Z", 105.0, 95.0, 96.0),
    ]
    row = trade_quality_row(trade(), [close_event()], candles, datetime(2026, 6, 7, 2, 0, tzinfo=timezone.utc))
    assert row["status"] == "CLOSED"
    assert row["gross_pnl"] == 4.0
    assert round(row["mfe_roe"], 4) == 0.6


def test_hit_before_stop_conservative_same_candle_stop_first() -> None:
    candles = [candle("2026-06-07T01:00:00Z", 101.0, 99.0, 100.0)]
    out = hit_before_stop_short(100.0, 20.0, candles, 0.10, 0.10)
    assert out["stopped"] is True
    assert out["ambiguous_hit_stop"] is True


def test_symbol_classification_keep_watch_disable() -> None:
    keep = {"closed_count": 2, "pnl_total": 1.0, "mfe_mae_ratio_avg": 2.0, "p90_mae_roe": 0.1}
    assert classify_symbol(keep) == "SYMBOL_KEEP_ACTIVE"
    disable = {"closed_count": 2, "pnl_total": -1.0, "mfe_roe_avg": 0.0, "p90_mae_roe": 0.1}
    assert classify_symbol(disable) == "SYMBOL_DISABLE_TEMPORARILY"
    insufficient = {"closed_count": 1, "pnl_total": 5.0}
    assert classify_symbol(insufficient) == "SYMBOL_INSUFFICIENT_DATA"


def test_score_correlation_small_sample_safe() -> None:
    assert pearson([1.0, 2.0], [2.0, 3.0]) is None


def test_burst_classifier_correlated_overexposure() -> None:
    mg = {"multisymbol_windows": [{"start": "2026-06-07T01:00:00Z", "end": "2026-06-07T01:01:00Z", "symbols": ["ETHUSDT", "BTCUSDT"]}]}
    rows = [
        {"symbol": "ETHUSDT", "opened_at": "2026-06-07T01:00:00+00:00", "net_pnl_estimated": -1.0, "mae_roe": 0.8},
        {"symbol": "BTCUSDT", "opened_at": "2026-06-07T01:00:30+00:00", "net_pnl_estimated": -1.0, "mae_roe": 0.8},
    ]
    assert classify_burst(mg, rows)[0]["classification"] == "BURST_CORRELATED_OVEREXPOSURE"


def test_global_too_early_and_promising() -> None:
    rows = [{"status": "CLOSED", "net_pnl_estimated": 1.0, "mfe_roe": 0.2, "mae_roe": 0.1} for _ in range(3)]
    assert classify_global(rows, []) == "PHASE_O_SHORT_LIVE_TOO_EARLY"
    rows = [{"status": "CLOSED", "net_pnl_estimated": 1.0, "mfe_roe": 0.3, "mae_roe": 0.1} for _ in range(10)]
    assert classify_global(rows, []) == "PHASE_O_SHORT_LIVE_STRONG_INITIAL"


def test_aggregate_symbols_serializable() -> None:
    rows = [
        {"symbol": "ETHUSDT", "status": "CLOSED", "winner": True, "net_pnl_estimated": 1.0, "roe": 0.1, "mfe_roe": 0.3, "mae_roe": 0.1, "mfe_mae_ratio": 3.0, "time_in_trade_seconds": 60, "closed_by": "trailing", "model_score": 0.7, "bucket": "premium"},
        {"symbol": "ETHUSDT", "status": "CLOSED", "winner": True, "net_pnl_estimated": 2.0, "roe": 0.2, "mfe_roe": 0.4, "mae_roe": 0.1, "mfe_mae_ratio": 4.0, "time_in_trade_seconds": 120, "closed_by": "trailing", "model_score": 0.8, "bucket": "premium"},
    ]
    out = aggregate_symbols(rows)
    assert out[0]["entry_quality_grade"] == "SYMBOL_KEEP_ACTIVE"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.csv"
        write_csv(path, out)
        assert "ETHUSDT" in path.read_text()
        json.dumps(out)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("audit_phase_o_short_live_quality tests passed")
