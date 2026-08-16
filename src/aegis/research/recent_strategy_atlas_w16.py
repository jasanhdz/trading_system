"""Walk-forward atlas of simple recent-regime trading strategies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any

import numpy as np
import pandas as pd

from aegis.research.recent_short_w14 import build_dataset


@dataclass(frozen=True)
class Variant:
    family: str
    parameters: tuple[tuple[str, float], ...]

    @property
    def identity(self) -> str:
        return self.family + "|" + "|".join(f"{key}={value:g}" for key, value in self.parameters)

    def value(self, name: str) -> float:
        return dict(self.parameters)[name]


def variants(config: dict[str, Any]) -> list[Variant]:
    result = []
    mean = config["families"]["mean_reversion"]
    for extension, rsi, slope, hold in product(mean["extension_atr"], mean["rsi_extremes"], mean["maximum_abs_ema25_slope_atr"], mean["maximum_hold_bars"]):
        result.append(Variant("mean_reversion", (("extension", extension), ("rsi_low", rsi[0]), ("rsi_high", rsi[1]), ("slope", slope), ("hold", hold))))
    trend = config["families"]["trend_pullback"]
    for slope, hold in product(trend["minimum_ema25_slope_atr"], trend["maximum_hold_bars"]):
        result.append(Variant("trend_pullback", (("slope", slope), ("hold", hold))))
    breakout = config["families"]["breakout"]
    for lookback, volume, hold in product(breakout["lookback_bars"], breakout["minimum_volume_ratio"], breakout["maximum_hold_bars"]):
        result.append(Variant("breakout", (("lookback", lookback), ("volume", volume), ("hold", hold))))
    return result


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in frame.groupby("symbol"):
        group = group.sort_index().copy()
        group["prior_high_12"] = group.high.shift(1).rolling(12).max()
        group["prior_low_12"] = group.low.shift(1).rolling(12).min()
        group["prior_high_24"] = group.high.shift(1).rolling(24).max()
        group["prior_low_24"] = group.low.shift(1).rolling(24).min()
        parts.append(group)
    return pd.concat(parts).sort_index()


def _signals(group: pd.DataFrame, variant: Variant) -> pd.Series:
    if variant.family == "mean_reversion":
        quiet = group.ema25_slope_atr.abs() <= variant.value("slope")
        long = quiet & group.ema25_distance_atr.le(-variant.value("extension")) & group.rsi6.le(variant.value("rsi_low"))
        short = quiet & group.ema25_distance_atr.ge(variant.value("extension")) & group.rsi6.ge(variant.value("rsi_high"))
    elif variant.family == "trend_pullback":
        ema25_above_ema99 = group.ema99_distance_atr > group.ema25_distance_atr
        candle_return = (group.close / group.open - 1) * 10_000
        long = ema25_above_ema99 & group.ema25_slope_atr.ge(variant.value("slope")) & group.ema25_distance_atr.gt(0) & candle_return.lt(0) & group.rsi6.between(35, 60)
        short = (~ema25_above_ema99) & group.ema25_slope_atr.le(-variant.value("slope")) & group.ema25_distance_atr.lt(0) & candle_return.gt(0) & group.rsi6.between(40, 65)
    else:
        lookback = int(variant.value("lookback"))
        long = group.close.gt(group[f"prior_high_{lookback}"]) & group.volume_ratio.ge(variant.value("volume")) & group.taker_imbalance.gt(0)
        short = group.close.lt(group[f"prior_low_{lookback}"]) & group.volume_ratio.ge(variant.value("volume")) & group.taker_imbalance.lt(0)
    return long.astype(int) - short.astype(int)


def _simulate(group: pd.DataFrame, signal_index: int, side: int, variant: Variant) -> tuple[float, int]:
    entry_index = signal_index + 1
    if entry_index >= len(group):
        return math.nan, entry_index
    entry = float(group.open.iloc[entry_index])
    atr = float(group.atr.iloc[signal_index])
    last = min(entry_index + int(variant.value("hold")) - 1, len(group) - 1)
    exit_price = float(group.close.iloc[last])
    exit_index = last
    for cursor in range(entry_index, last + 1):
        adverse = (entry - float(group.low.iloc[cursor])) if side > 0 else (float(group.high.iloc[cursor]) - entry)
        if adverse >= atr:
            exit_price = entry - side * atr
            exit_index = cursor
            break
        if variant.family == "mean_reversion" and side * float(group.ema25_distance_atr.iloc[cursor]) >= 0:
            exit_price = float(group.close.iloc[cursor])
            exit_index = cursor
            break
    return side * (exit_price / entry - 1) * 10_000, exit_index


def generate_trades(frame: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    rows = []
    for symbol, group in frame.groupby("symbol"):
        group = group.sort_index()
        signal = _signals(group, variant).to_numpy()
        next_free = 0
        for index in np.flatnonzero(signal):
            if index < next_free:
                continue
            gross, exit_index = _simulate(group, int(index), int(signal[index]), variant)
            if np.isfinite(gross):
                strength = abs(float(group.ema25_distance_atr.iloc[index])) + abs(float(group.taker_imbalance.iloc[index]))
                rows.append({"timestamp": group.index[index] + pd.Timedelta(minutes=5), "symbol": symbol,
                             "side": "LONG" if signal[index] > 0 else "SHORT", "gross_bps": gross,
                             "strength": strength, "variant": variant.identity, "family": variant.family})
                next_free = exit_index + 1
    if not rows:
        return pd.DataFrame(columns=["symbol", "side", "gross_bps", "strength", "variant", "family"])
    result = pd.DataFrame(rows)
    result.index = pd.DatetimeIndex(result.pop("timestamp"))
    return result.sort_index()


def _metrics(frame: pd.DataFrame, cost: float) -> dict[str, float | int]:
    net = frame.gross_bps.to_numpy() - cost if len(frame) else np.array([])
    loss = -net[net < 0].sum()
    return {"trades": len(frame), "gross_bps_per_trade": float(frame.gross_bps.mean()) if len(frame) else 0.0,
            "net_bps_per_trade": float(net.mean()) if len(net) else 0.0,
            "total_net_bps": float(net.sum()), "profit_factor": float(net[net > 0].sum() / loss) if loss else 0.0,
            "win_rate": float((net > 0).mean()) if len(net) else 0.0}


def walk_forward(root, config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = enrich(build_dataset(root, config["universe"]))
    catalog = variants(config)
    trade_cache = {variant.identity: generate_trades(frame, variant) for variant in catalog}
    test_days = pd.date_range(config["walk_forward"]["first_test_day"], config["walk_forward"]["last_test_day"], freq="D", tz="UTC")
    selected_rows = []
    fold_rows = []
    for day in test_days:
        train_start = day - pd.Timedelta(days=config["walk_forward"]["train_days"])
        candidates = []
        diagnostics = []
        for variant in catalog:
            trades = trade_cache[variant.identity]
            train = trades[(trades.index >= train_start) & (trades.index < day)]
            per_day = len(train) / config["walk_forward"]["train_days"]
            metrics = _metrics(train, 10.0)
            diagnostics.append((metrics["net_bps_per_trade"], per_day, variant, metrics))
            if metrics["trades"] >= config["walk_forward"]["minimum_training_trades"] and 3 <= per_day <= 8 and metrics["net_bps_per_trade"] > 0:
                candidates.append((metrics["total_net_bps"] / config["walk_forward"]["train_days"], variant, metrics))
        if not candidates:
            best_net, best_frequency, best_variant, best_metrics = max(diagnostics, key=lambda item: item[0])
            fold_rows.append({"day": str(day.date()), "variant": "NO_TRADE", "trades": 0, "net_bps": 0.0,
                              "best_failed_variant": best_variant.identity,
                              "best_failed_training_net_bps_per_trade": best_net,
                              "best_failed_training_trades_per_day": best_frequency,
                              "best_failed_training_trades": best_metrics["trades"]})
            continue
        _, winner, train_metrics = max(candidates, key=lambda item: item[0])
        trades = trade_cache[winner.identity]
        test = trades[(trades.index >= day) & (trades.index < day + pd.Timedelta(days=1))]
        test = test.sort_values("strength", ascending=False).head(config["walk_forward"]["maximum_trades_per_test_day"]).sort_index()
        selected_rows.append(test)
        metrics = _metrics(test, 10.0)
        fold_rows.append({"day": str(day.date()), "variant": winner.identity, "family": winner.family,
                          "training_net_bps_per_trade": train_metrics["net_bps_per_trade"],
                          "trades": metrics["trades"], "net_bps": metrics["total_net_bps"]})
    selected = pd.concat(selected_rows).sort_index() if selected_rows else pd.DataFrame(columns=["gross_bps"])
    folds = pd.DataFrame(fold_rows)
    per_family = {family: _metrics(group, 10.0) for family, group in selected.groupby("family")} if len(selected) else {}
    result = {"variant_count": len(catalog), "test_days": len(test_days), "days_traded": int((folds.trades > 0).sum()),
              "no_trade_days": int((folds.trades == 0).sum()), "baseline_10bps": _metrics(selected, 10.0),
              "stress_14bps": _metrics(selected, 14.0), "stress_20bps": _metrics(selected, 20.0),
              "positive_days": int((folds.net_bps > 0).sum()), "negative_days": int((folds.net_bps < 0).sum()),
              "per_family": per_family, "W16_RECENT_STRATEGY_EDGE_FOUND": False}
    enough = result["baseline_10bps"]["trades"] >= 45 and result["positive_days"] >= 8
    result["W16_RECENT_STRATEGY_EDGE_FOUND"] = bool(enough and result["stress_14bps"]["net_bps_per_trade"] > 2 and result["stress_20bps"]["net_bps_per_trade"] > 0)
    return result, folds
