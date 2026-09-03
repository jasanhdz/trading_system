"""Diagnose causal LONG edge by fixed market regimes without model fitting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.training.run_state import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/experiments/aegis_long_candidate_l1.yaml"
ROUND_TRIP_COST = 0.001
HORIZONS = (12, 36, 72)


def _load_symbol(symbol: str, source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source / f"{symbol}_5m.csv", parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    close = frame["close"]
    for span in (6, 12, 24, 48):
        frame[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()
    for bars in (1, 3, 6, 12, 24):
        frame[f"ret_{bars}"] = close.pct_change(bars)
    frame["ema_slope_12"] = frame["ema_12"].pct_change(3)
    frame["ema_slope_24"] = frame["ema_24"].pct_change(6)
    frame["rolling_high_24"] = frame["high"].shift(1).rolling(24).max()
    frame["rolling_low_24"] = frame["low"].shift(1).rolling(24).min()
    frame["range_fraction"] = (frame["high"] - frame["low"]) / close
    frame["range_mean_24"] = frame["range_fraction"].rolling(24).mean()
    frame["volume_zscore_24"] = (
        (frame["volume"] - frame["volume"].rolling(24).mean())
        / frame["volume"].rolling(24).std(ddof=0).replace(0.0, np.nan)
    )
    frame["lower_wick"] = (
        np.minimum(frame["open"], close) - frame["low"]
    ) / close
    frame["upper_wick"] = (
        frame["high"] - np.maximum(frame["open"], close)
    ) / close
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14).mean()
    relative_strength = gain / loss.replace(0.0, np.nan)
    frame["rsi_14"] = 100.0 - 100.0 / (1.0 + relative_strength)
    return frame


def _hourly_hypotheses(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    market_returns = pd.concat(
        {symbol: frame["ret_6"] for symbol, frame in frames.items()},
        axis=1,
    )
    breadth = (market_returns > 0.0).mean(axis=1)
    market_direction = market_returns.mean(axis=1)
    rows = []
    for symbol, frame in frames.items():
        values = frame.copy()
        values["market_breadth_6"] = breadth
        values["market_direction_6"] = market_direction
        values["btc_ret_12"] = frames["BTCUSDT"]["ret_12"]
        anchors = values[
            (values.index.minute == 55)
            & (values.index.second == 0)
        ].copy()
        positions = frame.index.get_indexer(anchors.index)
        for anchor, position in zip(anchors.index, positions):
            if position < 48 or position + max(HORIZONS) >= len(frame):
                continue
            entry = float(frame.iloc[position + 1]["open"])
            row = anchors.loc[anchor].to_dict()
            row.update(
                {
                    "timestamp": anchor + pd.Timedelta(minutes=5),
                    "symbol": symbol,
                    "entry": entry,
                }
            )
            for horizon in HORIZONS:
                future = frame.iloc[position + 1 : position + horizon + 1]
                row[f"net_return_h{horizon}"] = (
                    float(future.iloc[-1]["close"] / entry - 1.0)
                    - ROUND_TRIP_COST
                )
                row[f"mae_h{horizon}"] = max(
                    0.0,
                    float((entry - future["low"].min()) / entry),
                )
                row[f"mfe_h{horizon}"] = max(
                    0.0,
                    float((future["high"].max() - entry) / entry),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _rules() -> dict[str, tuple[Callable[[pd.DataFrame], pd.Series], str]]:
    return {
        "UNCONDITIONAL_LONG": (
            lambda x: pd.Series(True, index=x.index),
            "ret_12",
        ),
        "STRICT_BULL_TREND": (
            lambda x: (
                (x["close"] > x["ema_6"])
                & (x["ema_6"] > x["ema_12"])
                & (x["ema_12"] > x["ema_24"])
                & (x["ema_24"] > x["ema_48"])
                & (x["ema_slope_12"] > 0.0)
                & (x["ema_slope_24"] > 0.0)
                & (x["ret_6"] > 0.0)
                & (x["ret_12"] > 0.0)
                & (x["market_breadth_6"] >= 0.60)
                & (x["market_direction_6"] > 0.0)
                & (x["btc_ret_12"] > 0.0)
            ),
            "ret_12",
        ),
        "BULL_TREND_PULLBACK": (
            lambda x: (
                (x["close"] > x["ema_24"])
                & (x["ema_24"] > x["ema_48"])
                & (x["ema_slope_24"] > 0.0)
                & (x["ret_1"] <= 0.0)
                & (x["close"] >= x["ema_12"])
                & (x["lower_wick"] > x["upper_wick"])
                & (x["market_breadth_6"] >= 0.55)
                & (x["btc_ret_12"] > 0.0)
            ),
            "lower_wick",
        ),
        "CONFIRMED_BREAKOUT": (
            lambda x: (
                (x["close"] > x["rolling_high_24"])
                & (x["ret_6"] > 0.0)
                & (x["ema_slope_12"] > 0.0)
                & (x["volume_zscore_24"] > 0.0)
                & (x["market_breadth_6"] >= 0.55)
            ),
            "volume_zscore_24",
        ),
        "OVERSOLD_REVERSAL": (
            lambda x: (
                (x["rsi_14"] < 35.0)
                & (x["close"] > x["open"])
                & (x["lower_wick"] > x["upper_wick"])
                & (x["close"] > x["rolling_low_24"])
                & (x["ret_1"] > 0.0)
            ),
            "lower_wick",
        ),
    }


def _metrics(rows: pd.DataFrame, horizon: int) -> dict[str, object]:
    return_field = f"net_return_h{horizon}"
    mae_field = f"mae_h{horizon}"
    mfe_field = f"mfe_h{horizon}"
    if rows.empty:
        return {
            "signals": 0,
            "mean_net_expectancy": 0.0,
            "win_rate": 0.0,
            "mean_mae": 0.0,
            "mean_mfe": 0.0,
            "profit_factor": 0.0,
            "symbol_concentration": 1.0,
        }
    gains = rows.loc[rows[return_field] > 0.0, return_field].sum()
    losses = -rows.loc[rows[return_field] < 0.0, return_field].sum()
    counts = rows["symbol"].value_counts()
    return {
        "signals": int(len(rows)),
        "mean_net_expectancy": float(rows[return_field].mean()),
        "win_rate": float((rows[return_field] > 0.0).mean()),
        "mean_mae": float(rows[mae_field].mean()),
        "mean_mfe": float(rows[mfe_field].mean()),
        "profit_factor": float(gains / losses) if losses > 0.0 else math.inf,
        "symbol_concentration": float(counts.max() / len(rows)),
        "symbol_counts": {
            str(symbol): int(count) for symbol, count in counts.sort_index().items()
        },
    }


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    source = Path(config["source"]["path"])
    frames = {symbol: _load_symbol(symbol, source) for symbol in CANONICAL_SYMBOLS}
    hypotheses = _hourly_hypotheses(frames)
    report: dict[str, object] = {
        "schema_id": "aegis-long-regime-edge-diagnostic-v1",
        "source": str(source),
        "side": "LONG",
        "entry": "NEXT_BAR_OPEN",
        "exit": "H12_CLOSE",
        "round_trip_cost_fraction": ROUND_TRIP_COST,
        "rules_defined_before_outcome_evaluation": True,
        "folds": {},
    }
    summaries: dict[str, dict[int, list[dict[str, object]]]] = {
        name: {horizon: [] for horizon in HORIZONS} for name in _rules()
    }
    for fold in config["fold_protocol"]["folds"]:
        start = pd.Timestamp(fold["scoring_start"])
        end = pd.Timestamp(fold["scoring_end"])
        block = hypotheses[
            (hypotheses["timestamp"] >= start)
            & (hypotheses["timestamp"] <= end)
        ]
        fold_result = {}
        for name, (predicate, ranking_field) in _rules().items():
            eligible = block[predicate(block)].copy()
            selected = (
                eligible.sort_values(
                    ["timestamp", ranking_field, "symbol"],
                    ascending=[True, False, True],
                )
                .groupby("timestamp", as_index=False)
                .first()
            )
            fold_result[name] = {}
            for horizon in HORIZONS:
                metrics = _metrics(selected, horizon)
                fold_result[name][f"H{horizon}"] = metrics
                summaries[name][horizon].append(metrics)
        report["folds"][str(fold["id"])] = fold_result
    report["summary"] = {
        name: {
            f"H{horizon}": {
                "signals": sum(int(row["signals"]) for row in rows),
                "positive_folds": sum(
                    float(row["mean_net_expectancy"]) > 0.0 for row in rows
                ),
                "mean_fold_expectancy": float(
                    np.mean([row["mean_net_expectancy"] for row in rows])
                ),
                "worst_fold_expectancy": float(
                    min(row["mean_net_expectancy"] for row in rows)
                ),
                "mean_fold_mae": float(
                    np.mean([row["mean_mae"] for row in rows])
                ),
            }
            for horizon, rows in horizon_rows.items()
        }
        for name, horizon_rows in summaries.items()
    }
    report["diagnostic_only"] = True
    report["automatic_live_activation"] = False
    output = (
        ROOT
        / "reports/experiments/aegis_long_regime_edge_multihorizon_diagnostic.json"
    )
    atomic_write_json(output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
