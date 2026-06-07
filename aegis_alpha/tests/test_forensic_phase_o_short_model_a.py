#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools.forensic_phase_o_short_model_a import (  # noqa: E402
    classify_live_trade,
    classify_random_global,
    global_diagnosis,
    management_comparison,
    percentile,
    proposed_model_design,
    random_baseline,
    score_calibration,
    symbol_diagnostics,
    write_csv,
    zscore,
)


def trade_row(symbol: str = "ETHUSDT", pnl: float = 1.0, mfe: float = 0.4, mae: float = 0.1) -> dict:
    return {
        "trade_id": f"T-{symbol}",
        "symbol": symbol,
        "opened_at": "2026-06-07T01:00:00+00:00",
        "entry_price": 100.0,
        "leverage": 20.0,
        "net_pnl_estimated": pnl,
        "mfe_roe": mfe,
        "mae_roe": mae,
        "mfe_mae_ratio": mfe / mae if mae else 10.0,
        "close_efficiency": 0.6,
        "model_score": 0.8,
        "roe": pnl / 10.0,
    }


def candle(ts: str, close: float, high: float | None = None, low: float | None = None) -> dict:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return {
        "timestamp": dt,
        "open": close,
        "high": high if high is not None else close + 1,
        "low": low if low is not None else close - 1,
        "close": close,
        "volume": 1.0,
    }


def candles() -> list[dict]:
    base = datetime.fromisoformat("2026-06-07T00:00:00+00:00")
    rows = []
    for i in range(80):
        ts = base.replace(minute=0) if i == 0 else rows[-1]["timestamp"]
        if i:
            from datetime import timedelta
            ts = ts + timedelta(minutes=5)
        close = 100 - (i % 7) * 0.1
        rows.append({"timestamp": ts, "open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1.0})
    return rows


def test_classifies_good_entry() -> None:
    assert classify_live_trade(trade_row(pnl=1, mfe=0.5, mae=0.1)) == "LIVE_TRADE_GOOD_ENTRY"


def test_classifies_saved_by_management() -> None:
    assert classify_live_trade(trade_row(pnl=0.1, mfe=0.1, mae=0.2)) == "LIVE_TRADE_SAVED_BY_MANAGEMENT"


def test_classifies_bad_and_big_loss() -> None:
    assert classify_live_trade(trade_row(pnl=-0.5, mfe=0.05, mae=0.2)) == "LIVE_TRADE_BAD_ENTRY"
    assert classify_live_trade(trade_row(pnl=-2.0, mfe=0.05, mae=0.4)) == "LIVE_TRADE_BIG_LOSS"


def test_feature_percentile_and_zscore() -> None:
    vals = [1, 2, 3, 4]
    assert percentile(vals, 3) == 0.75
    assert zscore(vals, 3) is not None


def test_score_calibration_weak_with_low_corr() -> None:
    rows = [trade_row(symbol=f"S{i}", pnl=(-1) ** i, mfe=0.1, mae=0.1) for i in range(6)]
    for i, row in enumerate(rows):
        row["model_score"] = 0.7
    assert score_calibration(rows)[0]["status"] == "SCORE_CALIBRATION_WEAK"


def test_random_baseline_not_better_than_random() -> None:
    rows = [trade_row("ETHUSDT", pnl=-1, mfe=0.01, mae=0.5)]
    out = random_baseline(rows, {"ETHUSDT": candles()}, n=20)
    assert out[0]["status"] in {"OK", "INSUFFICIENT_RANDOM_BASELINE"}
    if out[0]["status"] == "OK":
        assert out[0]["live_better_than_random_median"] is False


def test_random_global_detects_not_better() -> None:
    rows = [{"status": "OK", "live_quality_percentile": 0.2, "live_better_than_random_median": False} for _ in range(6)]
    assert classify_random_global(rows) == "MODEL_NOT_BETTER_THAN_RANDOM"


def test_management_masks_bad_entry() -> None:
    tr = trade_row(pnl=0.05, mfe=0.05, mae=0.2)
    tr["opened_at"] = "2026-06-07T01:00:00+00:00"
    tr["entry_price"] = 100.0
    out = management_comparison([tr], {"ETHUSDT": candles()})
    assert out[0]["classification"] in {"MANAGEMENT_MASKS_BAD_ENTRY", "ENTRY_DEPENDS_ON_MANAGEMENT", "UNKNOWN"}


def test_symbol_failure_mode_mae_danger_underpredicted() -> None:
    rows = [trade_row("ETHUSDT", pnl=-1, mfe=0.1, mae=0.3), trade_row("ETHUSDT", pnl=-1, mfe=0.1, mae=0.4)]
    sym = symbol_diagnostics(rows, [], [])
    assert sym[0]["failure_mode"] == "MAE_DANGER_UNDERPREDICTED"


def test_global_diagnosis_overtrading_no_edge() -> None:
    rows = [trade_row(symbol=f"S{i}", pnl=-1, mfe=0.1, mae=0.2) for i in range(10)]
    diag = global_diagnosis(rows, [{"status": "SCORE_CALIBRATION_WEAK"}], "MODEL_NOT_BETTER_THAN_RANDOM", [])
    assert diag == "MODEL_OVERTRADING_NO_EDGE"


def test_proposes_operable_short_quality_v4() -> None:
    design = proposed_model_design([{"symbol": "SUIUSDT", "recommendation": "KEEP_FOR_MODEL_REBUILD"}])
    assert design["name"] == "operable_short_quality_v4"
    assert "SUIUSDT" in design["initial_symbols"]


def test_json_csv_serializes() -> None:
    rows = [{"symbol": "ETHUSDT", "nested": {"x": 1}}]
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.csv"
        write_csv(p, rows)
        assert "ETHUSDT" in p.read_text()
        json.dumps(rows)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("forensic_phase_o_short_model_a tests passed")
