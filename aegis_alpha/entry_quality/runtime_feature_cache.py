from __future__ import annotations

import bisect
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis_alpha.config import REPO_ROOT
from aegis_alpha.entry_quality.model_loader import normalize_symbol


DATABASE_PATH = REPO_ROOT / "data/binance_candles.db"
RUNTIME_CANDLE_LIMIT = 600
FEATURE_CACHE_TTL_SECONDS = 30.0

_CACHE: dict[str, tuple[float, str | None, "RuntimeFeatureResult"]] = {}
_LAST_STATUS_BY_SYMBOL: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class RuntimeFeatureResult:
    symbol: str
    values: dict[str, float]
    feature_timestamp: str | None
    source: str
    approximated_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


def _db_symbol(symbol: str) -> str:
    clean = normalize_symbol(symbol)
    return clean.replace("USDT", "/USDT") if "/" not in clean else clean


def _safe_div(numerator: pd.Series | np.ndarray | float, denominator: pd.Series | np.ndarray | float) -> Any:
    return numerator / pd.Series(denominator).replace(0, np.nan) if not isinstance(denominator, (int, float)) else numerator / (denominator or np.nan)


def _safe_float(value: Any) -> float:
    try:
        out = float(np.asarray(value).item())
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _latest_candle_timestamp(symbol: str, timeframe: str = "5m") -> str | None:
    if not DATABASE_PATH.exists():
        return None
    query = """
        SELECT timestamp
        FROM ohlcv_data
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            row = conn.execute(query, (_db_symbol(symbol), timeframe)).fetchone()
    except Exception:
        return None
    return str(row[0]) if row else None


def load_recent_candles(symbol: str, timeframe: str = "5m", limit: int = RUNTIME_CANDLE_LIMIT) -> pd.DataFrame:
    if not DATABASE_PATH.exists():
        return pd.DataFrame()
    query = """
        SELECT timestamp, open, high, low, close, volume, buy_volume
        FROM (
            SELECT timestamp, open, high, low, close, volume, buy_volume
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
        )
        ORDER BY timestamp ASC
    """
    with sqlite3.connect(DATABASE_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(_db_symbol(symbol), timeframe, int(limit)))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    df = df.set_index("timestamp", drop=True)
    for col in ("open", "high", "low", "close", "volume", "buy_volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    arr = values.astype(float).to_numpy()
    out = np.full(len(arr), np.nan, dtype=np.float32)
    sorted_window: list[float] = []
    queue: list[float] = []
    for idx, value in enumerate(arr):
        if np.isfinite(value):
            bisect.insort(sorted_window, float(value))
            queue.append(float(value))
        else:
            queue.append(float("nan"))
        if len(queue) > window:
            old = queue.pop(0)
            if np.isfinite(old):
                pos = bisect.bisect_left(sorted_window, old)
                if pos < len(sorted_window):
                    sorted_window.pop(pos)
        if np.isfinite(value) and sorted_window:
            out[idx] = bisect.bisect_right(sorted_window, float(value)) / len(sorted_window)
    return pd.Series(out, index=values.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.fillna(high - low)


def add_base_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    approximated: list[str] = []
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    for bars in (1, 2, 3, 6, 12):
        out[f"ret_{bars}"] = close / close.shift(bars) - 1.0
    out["momentum_3"] = out["ret_3"]
    out["momentum_6"] = out["ret_6"]
    out["momentum_12"] = out["ret_12"]
    out["candle_body_pct"] = (close - open_) / open_.replace(0, np.nan)
    candle_range = (high - low).replace(0, np.nan)
    out["upper_wick_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / candle_range
    out["lower_wick_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / candle_range
    out["green_candle_count_3"] = (close > open_).astype(float).rolling(3, min_periods=1).sum()
    out["red_candle_count_3"] = (close < open_).astype(float).rolling(3, min_periods=1).sum()

    out["ema_9"] = close.ewm(span=9, adjust=False).mean()
    out["ema_21"] = close.ewm(span=21, adjust=False).mean()
    out["ema_50"] = close.ewm(span=50, adjust=False).mean()
    out["price_to_ema_9"] = close / out["ema_9"].replace(0, np.nan) - 1.0
    out["price_to_ema_21"] = close / out["ema_21"].replace(0, np.nan) - 1.0
    out["ema_9_slope"] = out["ema_9"] / out["ema_9"].shift(3).replace(0, np.nan) - 1.0
    out["ema_21_slope"] = out["ema_21"] / out["ema_21"].shift(3).replace(0, np.nan) - 1.0

    tr = true_range(out)
    out["atr_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["atr_pct"] = out["atr_14"] / close.replace(0, np.nan)
    returns = close.pct_change()
    out["realized_vol_12"] = returns.rolling(12, min_periods=4).std()
    out["realized_vol_36"] = returns.rolling(36, min_periods=12).std()
    out["high_low_range_pct"] = (high - low) / close.replace(0, np.nan)
    out["atr_percentile_7d"] = rolling_percentile(out["atr_pct"], 7 * 24 * 12)

    vol_mean = volume.rolling(36, min_periods=12).mean()
    vol_std = volume.rolling(36, min_periods=12).std()
    out["volume_zscore_36"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    out["volume_ratio_12"] = volume / volume.rolling(12, min_periods=4).mean().replace(0, np.nan)
    if "quote_volume" in out.columns:
        out["quote_volume"] = pd.to_numeric(out["quote_volume"], errors="coerce")
    else:
        out["quote_volume"] = close * volume
        approximated.append("quote_volume")
    return out, approximated


def mtf_frame(df: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    resampled = pd.DataFrame(
        {
            "open": df["open"].resample(rule, label="right", closed="right").first(),
            "high": df["high"].resample(rule, label="right", closed="right").max(),
            "low": df["low"].resample(rule, label="right", closed="right").min(),
            "close": df["close"].resample(rule, label="right", closed="right").last(),
            "volume": df["volume"].resample(rule, label="right", closed="right").sum(),
        }
    ).dropna(subset=["open", "high", "low", "close"])
    if resampled.empty:
        return pd.DataFrame(index=df.index)
    close = resampled["close"].astype(float)
    resampled[f"{prefix}_ret_1"] = close / close.shift(1) - 1.0
    resampled[f"{prefix}_ret_2"] = close / close.shift(2) - 1.0
    ema_9 = close.ewm(span=9, adjust=False).mean()
    ema_21 = close.ewm(span=21, adjust=False).mean()
    resampled[f"{prefix}_ema_9_slope"] = ema_9 / ema_9.shift(2).replace(0, np.nan) - 1.0
    resampled[f"{prefix}_price_to_ema_21"] = close / ema_21.replace(0, np.nan) - 1.0
    atr = true_range(resampled).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    resampled[f"{prefix}_atr_pct"] = atr / close.replace(0, np.nan)
    trend = np.where(
        (resampled[f"{prefix}_ema_9_slope"] > 0) & (resampled[f"{prefix}_price_to_ema_21"] > 0),
        1,
        np.where((resampled[f"{prefix}_ema_9_slope"] < 0) & (resampled[f"{prefix}_price_to_ema_21"] < 0), -1, 0),
    )
    resampled[f"{prefix}_trend_direction"] = trend.astype(np.int8)
    cols = [
        f"{prefix}_ret_1",
        f"{prefix}_ret_2",
        f"{prefix}_ema_9_slope",
        f"{prefix}_price_to_ema_21",
        f"{prefix}_atr_pct",
        f"{prefix}_trend_direction",
    ]
    return resampled[cols].reindex(df.index, method="ffill")


def add_mtf_features(features: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out = out.join(mtf_frame(candles, "15min", "mtf_15m"))
    out = out.join(mtf_frame(candles, "1h", "mtf_1h"))
    out["long_mtf_agreement"] = (
        (out["momentum_3"] > 0)
        & (out["mtf_15m_trend_direction"] > 0)
        & (out["mtf_1h_trend_direction"] >= 0)
    ).astype(np.int8)
    out["short_mtf_agreement"] = (
        (out["momentum_3"] < 0)
        & (out["mtf_15m_trend_direction"] < 0)
        & (out["mtf_1h_trend_direction"] <= 0)
    ).astype(np.int8)
    five_min_side = np.sign(out["momentum_6"].fillna(0.0))
    mtf_side = np.sign(out["mtf_15m_trend_direction"].fillna(0.0) + out["mtf_1h_trend_direction"].fillna(0.0))
    out["mtf_conflict"] = ((five_min_side != 0) & (mtf_side != 0) & (five_min_side != mtf_side)).astype(np.int8)
    return out


def build_features_from_candles(symbol: str, candles: pd.DataFrame) -> RuntimeFeatureResult:
    start = time.perf_counter()
    normalized = normalize_symbol(symbol)
    warnings: list[str] = []
    if candles.empty:
        return RuntimeFeatureResult(normalized, {}, None, "candles", warnings=["no_recent_candles"], latency_ms=(time.perf_counter() - start) * 1000)
    if len(candles) < 64:
        warnings.append(f"limited_recent_candles:{len(candles)}")
    base, approximated = add_base_features(candles)
    full = add_mtf_features(base, candles)
    latest = full.iloc[-1]
    values = {str(col): _safe_float(latest[col]) for col in full.columns}
    return RuntimeFeatureResult(
        symbol=normalized,
        values=values,
        feature_timestamp=str(full.index[-1]) if len(full.index) else None,
        source="sqlite_recent_candles",
        approximated_features=approximated,
        warnings=warnings,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


def get_runtime_market_features(symbol: str) -> RuntimeFeatureResult:
    start = time.perf_counter()
    normalized = normalize_symbol(symbol)
    latest_ts = _latest_candle_timestamp(normalized)
    now = time.time()
    cached = _CACHE.get(normalized)
    if cached is not None and cached[1] == latest_ts and now - cached[0] <= FEATURE_CACHE_TTL_SECONDS:
        return cached[2]
    try:
        candles = load_recent_candles(normalized)
        result = build_features_from_candles(normalized, candles)
    except Exception as exc:
        result = RuntimeFeatureResult(
            normalized,
            {},
            latest_ts,
            "sqlite_recent_candles",
            warnings=[f"runtime_feature_build_error:{exc!r}"],
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    _CACHE[normalized] = (now, latest_ts, result)
    _LAST_STATUS_BY_SYMBOL[normalized] = {
        "feature_status": "ok" if result.values else "insufficient",
        "feature_timestamp": result.feature_timestamp,
        "source": result.source,
        "latency_ms": round(float(result.latency_ms), 3),
        "approximated_features": list(result.approximated_features),
        "warnings": list(result.warnings),
    }
    return result


def runtime_feature_cache_status() -> dict[str, Any]:
    return {
        "cache_size": len(_CACHE),
        "cache_ttl_seconds": FEATURE_CACHE_TTL_SECONDS,
        "last_market_feature_status_by_symbol": dict(_LAST_STATUS_BY_SYMBOL),
    }


def clear_runtime_feature_cache() -> None:
    _CACHE.clear()
    _LAST_STATUS_BY_SYMBOL.clear()
