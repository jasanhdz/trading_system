"""Causal multi-timeframe features for the historical Live-entry audit."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    relative = gain / loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + relative)
    return result.where(loss.ne(0.0), 100.0).where(gain.ne(0.0), 0.0)


def _signed_trend_age(close: pd.Series, ema: pd.Series) -> pd.Series:
    state = np.sign((close - ema).fillna(0.0).to_numpy(float))
    age = np.zeros(len(state), dtype=float)
    for index in range(1, len(state)):
        age[index] = age[index - 1] + state[index] if state[index] == state[index - 1] else state[index]
    return pd.Series(age, index=close.index)


def aggregate_klines(one_minute: pd.DataFrame, minutes: int) -> pd.DataFrame:
    frame = one_minute.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if minutes == 1:
        result = frame[["open_time", "open", "high", "low", "close", "volume", "taker_buy_volume"]].copy()
        result["bar_count"] = 1
    else:
        result = frame.set_index("open_time").resample(f"{minutes}min", label="left", closed="left").agg(
            open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum"), taker_buy_volume=("taker_buy_volume", "sum"), bar_count=("close", "count"),
        ).dropna().reset_index()
        result = result.loc[result["bar_count"].eq(minutes)].copy()
    result["close_time"] = result["open_time"] + pd.Timedelta(minutes=minutes)
    return result.reset_index(drop=True)


def indicator_frame(one_minute: pd.DataFrame, minutes: int) -> pd.DataFrame:
    frame = aggregate_klines(one_minute, minutes)
    close, high, low, volume = frame["close"], frame["high"], frame["low"], frame["volume"]
    previous = close.shift(1)
    true_range = pd.concat([(high - low), (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    ema7, ema25, ema99 = (close.ewm(span=value, adjust=False, min_periods=value).mean() for value in (7, 25, 99))
    median_volume = volume.shift(1).rolling(20, min_periods=10).median()
    log_volume = np.log1p(volume)
    volume_mean = log_volume.shift(1).rolling(50, min_periods=20).mean()
    volume_std = log_volume.shift(1).rolling(50, min_periods=20).std(ddof=0)
    prior_high = high.shift(1).rolling(48, min_periods=20).max()
    prior_low = low.shift(1).rolling(48, min_periods=20).min()
    path = close.diff().abs().rolling(6, min_periods=3).sum()
    net = (close - close.shift(6)).abs()
    prefix = f"tf{minutes}m__"
    output = pd.DataFrame({"close_time": frame["close_time"]})
    output[prefix + "return_1_bps"] = close.pct_change() * 10_000.0
    output[prefix + "return_3_bps"] = close.pct_change(3) * 10_000.0
    output[prefix + "return_6_bps"] = close.pct_change(6) * 10_000.0
    output[prefix + "atr_pct_bps"] = atr / close * 10_000.0
    output[prefix + "atr_percentile_96"] = (atr / close).rolling(96, min_periods=32).rank(pct=True)
    output[prefix + "rsi6"] = _rsi(close, 6)
    output[prefix + "rsi12"] = _rsi(close, 12)
    output[prefix + "rsi24"] = _rsi(close, 24)
    output[prefix + "ema7_extension_atr"] = (close - ema7) / atr
    output[prefix + "ema25_extension_atr"] = (close - ema25) / atr
    output[prefix + "ema99_extension_atr"] = (close - ema99) / atr
    output[prefix + "ema7_slope_atr"] = (ema7 - ema7.shift(3)) / atr
    output[prefix + "ema25_slope_atr"] = (ema25 - ema25.shift(3)) / atr
    output[prefix + "trend_age"] = _signed_trend_age(close, ema25)
    output[prefix + "prior_move_6_atr"] = (close - close.shift(6)) / atr
    output[prefix + "volume_ratio20"] = volume / median_volume
    output[prefix + "volume_z50"] = (log_volume - volume_mean) / volume_std.replace(0.0, np.nan)
    candle_range = (high - low).replace(0.0, np.nan)
    output[prefix + "body_ratio"] = (close - frame["open"]).abs() / candle_range
    output[prefix + "clv"] = (close - low) / candle_range
    output[prefix + "taker_imbalance"] = (2.0 * frame["taker_buy_volume"] - volume) / volume.replace(0.0, np.nan)
    output[prefix + "distance_recent_high_atr"] = (prior_high - close) / atr
    output[prefix + "distance_recent_low_atr"] = (close - prior_low) / atr
    output[prefix + "range_48_atr"] = (prior_high - prior_low) / atr
    output[prefix + "path_efficiency_6"] = net / path.replace(0.0, np.nan)
    output[prefix + "breakout_up"] = close.gt(prior_high).astype(float)
    output[prefix + "breakout_down"] = close.lt(prior_low).astype(float)
    return output


def attach_features(entries: pd.DataFrame, candles_by_symbol: dict[str, pd.DataFrame], timeframes: Iterable[int]) -> pd.DataFrame:
    result = entries.copy()
    result["entry_timestamp"] = pd.to_datetime(result["opened_at"], utc=True, format="mixed")
    pieces = []
    for symbol, group in result.groupby("symbol", sort=True):
        working = group.sort_values("entry_timestamp").copy()
        source = candles_by_symbol[symbol]
        for minutes in timeframes:
            indicators = indicator_frame(source, int(minutes)).sort_values("close_time")
            working = pd.merge_asof(
                working.sort_values("entry_timestamp"), indicators,
                left_on="entry_timestamp", right_on="close_time", direction="backward", allow_exact_matches=True,
            ).drop(columns="close_time")
        pieces.append(working)
    return pd.concat(pieces, ignore_index=True).sort_values("entry_timestamp").reset_index(drop=True)


def add_directional_context(frame: pd.DataFrame, timeframes: Iterable[int]) -> pd.DataFrame:
    result = frame.copy()
    direction = result["side"].map({"LONG": 1.0, "SHORT": -1.0}).astype(float)
    for minutes in timeframes:
        source = f"tf{minutes}m__"
        target = f"dir{minutes}m__"
        for name in [
            "return_1_bps", "return_3_bps", "return_6_bps", "ema7_extension_atr",
            "ema25_extension_atr", "ema99_extension_atr", "ema7_slope_atr",
            "ema25_slope_atr", "trend_age", "prior_move_6_atr", "taker_imbalance",
        ]:
            result[target + name] = direction * pd.to_numeric(result[source + name], errors="coerce")
        for period in (6, 12, 24):
            rsi = pd.to_numeric(result[source + f"rsi{period}"], errors="coerce")
            result[target + f"rsi{period}_extension"] = direction * (rsi - 50.0)
            result[target + f"rsi{period}_remaining_room"] = np.where(direction > 0, 100.0 - rsi, rsi)
        high_space = pd.to_numeric(result[source + "distance_recent_high_atr"], errors="coerce")
        low_space = pd.to_numeric(result[source + "distance_recent_low_atr"], errors="coerce")
        result[target + "favorable_space_atr"] = np.where(direction > 0, high_space, low_space)
        result[target + "adverse_space_atr"] = np.where(direction > 0, low_space, high_space)
        up = pd.to_numeric(result[source + "breakout_up"], errors="coerce")
        down = pd.to_numeric(result[source + "breakout_down"], errors="coerce")
        result[target + "aligned_breakout"] = np.where(direction > 0, up, down)
        result[target + "opposed_breakout"] = np.where(direction > 0, down, up)
        clv = pd.to_numeric(result[source + "clv"], errors="coerce")
        result[target + "directional_clv"] = direction * (2.0 * clv - 1.0)
    return result


def feature_comparison(frame: pd.DataFrame, features: list[str], split: str = "DISCOVERY") -> pd.DataFrame:
    rows = []
    discovery = frame.loc[frame["split"].eq(split)]
    for feature in features:
        good = pd.to_numeric(discovery.loc[discovery["good_entry"].eq(1), feature], errors="coerce").dropna()
        bad = pd.to_numeric(discovery.loc[discovery["bad_entry"].eq(1), feature], errors="coerce").dropna()
        pooled = pd.concat([good, bad])
        scale = float(pooled.std(ddof=0)) if len(pooled) else math.nan
        difference = float((bad.median() - good.median()) / scale) if scale and math.isfinite(scale) else math.nan
        rows.append({
            "feature": feature, "good_n": len(good), "bad_n": len(bad),
            "good_median": float(good.median()) if len(good) else math.nan,
            "bad_median": float(bad.median()) if len(bad) else math.nan,
            "standardized_median_difference_bad_minus_good": difference,
        })
    return pd.DataFrame(rows).sort_values(
        "standardized_median_difference_bad_minus_good", key=lambda values: values.abs(), ascending=False
    )
