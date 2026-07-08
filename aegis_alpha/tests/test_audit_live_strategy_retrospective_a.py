#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.audit_live_strategy_retrospective_a import (  # noqa: E402
    assert_read_only_output_path,
    classify_signal,
    classify_trade_retrospective,
    learning_candidates,
    management_attribution,
    profitability_summary,
    safe_div,
    score_calibration,
    signal_funnel,
    symbol_diagnostics,
    v4_overlap,
    write_csv,
)
from aegis_alpha.tools.audit_phase_o_short_live_quality import compute_short_mae_mfe  # noqa: E402


def candle(ts: str, high: float, low: float, close: float) -> dict:
    return {
        "timestamp": datetime.fromisoformat(ts.replace("Z", "+00:00")),
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1.0,
    }


def trade(pnl: float = 1.0, mae: float = 0.05, mfe: float = 0.20, tid: str = "T1") -> dict:
    return {
        "trade_id": tid,
        "symbol": "ADAUSDT",
        "side": "SHORT",
        "status": "CLOSED",
        "opened_at": "2026-06-07T01:00:00+00:00",
        "closed_at": "2026-06-07T01:30:00+00:00",
        "entry_price": 100.0,
        "leverage": 20.0,
        "net_pnl_estimated": pnl,
        "gross_pnl": pnl + 0.1,
        "fee_estimate": 0.1,
        "roe": pnl / 10.0,
        "winner": pnl > 0,
        "mfe_roe": mfe,
        "mae_roe": mae,
        "mfe_mae_ratio": mfe / mae if mae else None,
        "model_score": 0.8 if pnl > 0 else 0.9,
        "bucket": "premium",
        "brackets_confirmed": True,
    }


def test_signal_classification_executed_and_blocked() -> None:
    assert classify_signal({"event": "ORDER_SUBMITTED", "symbol": "ADAUSDT"}) == "SIGNAL_EXECUTED_TRADE"
    assert classify_signal({"reason": "max_consecutive_losses_reached", "symbol": "ADAUSDT"}) == "SIGNAL_BLOCKED_BY_MAX_LOSSES"
    assert classify_signal({"symbol": "LINKUSDT", "reason": "phase_o_link_avoid_only_no_entry"}) == "SIGNAL_LINK_AVOID_ONLY"


def test_signal_funnel_counts() -> None:
    signals = [
        {"classification": "SIGNAL_EXECUTED_TRADE", "side": "SHORT"},
        {"classification": "SIGNAL_BLOCKED_BY_RISK", "side": "SHORT"},
        {"classification": "SIGNAL_HOLD", "side": ""},
    ]
    rows = signal_funnel(signals, [trade()])
    by_stage = {r["stage"]: r["count"] for r in rows}
    assert by_stage["total_signals"] == 3
    assert by_stage["executed_trades"] == 1
    assert by_stage["SIGNAL_BLOCKED_BY_RISK"] == 1


def test_profitability_and_expectancy() -> None:
    rows = [trade(1.0, tid="W"), trade(-0.5, tid="L")]
    out = profitability_summary(rows, [{"x": 1}, {"x": 2}], datetime(2026, 6, 7, tzinfo=timezone.utc), datetime(2026, 6, 8, tzinfo=timezone.utc))
    assert out["total_net_pnl"] == 0.5
    assert out["expectancy_per_trade"] == 0.25
    assert out["expectancy_per_signal"] == 0.25


def test_short_mfe_mae_uses_low_and_high() -> None:
    candles = [
        candle("2026-06-07T01:00:00Z", 101.0, 99.0, 99.5),
        candle("2026-06-07T01:05:00Z", 100.0, 97.0, 98.0),
    ]
    out = compute_short_mae_mfe(100.0, 20.0, candles)
    assert round(out["mfe_roe"], 4) == 0.6
    assert round(out["mae_roe"], 4) == 0.2


def test_big_loss_and_management_saved() -> None:
    assert classify_trade_retrospective(trade(-2.0, mae=0.30, mfe=0.05)) == "TRADE_BIG_LOSS"
    rows = [trade(0.1, mae=0.20, mfe=0.10)]
    candles = {"ADAUSDT": [candle("2026-06-07T01:00:00Z", 101, 99, 100), candle("2026-06-07T01:05:00Z", 102, 99.5, 101)]}
    assert management_attribution(rows, candles)[0]["classification"] == "MANAGEMENT_SAVED"


def test_score_calibration_weak() -> None:
    rows = [trade(1.0, tid="A"), trade(-1.0, tid="B"), trade(0.5, tid="C"), trade(-0.2, tid="D"), trade(0.1, tid="E")]
    out = score_calibration(rows)
    assert out[0]["status"] in {"SCORE_NOT_CALIBRATED", "SCORE_WEAKLY_CALIBRATED", "SCORE_INVERTED"}


def test_v4_overlap_bad_or_no_trade() -> None:
    rows = [trade(-1.0, mae=0.20, mfe=0.01)]
    candles = {
        "ADAUSDT": [
            candle("2026-06-07T00:55:00Z", 100.0, 100.0, 100.0),
            candle("2026-06-07T01:00:00Z", 100.0, 100.0, 100.0),
            candle("2026-06-07T01:05:00Z", 101.0, 99.9, 100.8),
            candle("2026-06-07T01:10:00Z", 101.5, 99.8, 101.0),
        ]
    }
    out = v4_overlap(rows, candles, horizon=2)[0]
    assert out["status"] == "OK"
    assert out["short_bad_entry_v4"] == 1 or out["short_no_trade_v4"] == 1


def test_symbol_recommendation_pause_live() -> None:
    rows = [trade(-1.0, mae=0.2, mfe=0.01, tid="A"), trade(-2.0, mae=0.3, mfe=0.02, tid="B")]
    for row in rows:
        row["trade_classification"] = classify_trade_retrospective(row)
        row["big_loss"] = row["trade_classification"] == "TRADE_BIG_LOSS"
    symbols = symbol_diagnostics(rows, [], [], [{"trade_id": "A", "short_bad_entry_v4": 1}, {"trade_id": "B", "short_bad_entry_v4": 1}], [])
    assert symbols[0]["recommendation"] == "PAUSE_LIVE"


def test_learning_candidates_positive_negative_and_serializes() -> None:
    rows = [trade(1.0, tid="P"), trade(-1.0, tid="N")]
    v4 = [{"trade_id": "P", "short_clean_entry_v4": 1}, {"trade_id": "N", "short_bad_entry_v4": 1}]
    out = learning_candidates(rows, [], v4, [])
    assert {r["recommended_use"] for r in out} == {"train_positive", "train_negative"}
    json.dumps(out)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.csv"
        write_csv(path, out)
        assert "train_positive" in path.read_text()


def test_no_active_output_paths() -> None:
    assert_read_only_output_path(Path("/tmp/reports"))
    try:
        assert_read_only_output_path(Path("/tmp/models/turbo/ADAUSDT/active/report.json"))
    except ValueError:
        return
    raise AssertionError("active path was not rejected")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("audit_live_strategy_retrospective_a tests passed")
