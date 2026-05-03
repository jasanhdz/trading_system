from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from aegis_alpha.config import AegisConfig, load_config
from aegis_alpha.edge.common import build_edge_feature_matrix, edge_feature_names, safe_float
from aegis_alpha.features.feature_builder import FEATURE_COLUMNS, build_feature_frame
from aegis_alpha.features.regime_detector import detect_regime
from data.storage.database_manager import DatabaseManager


@dataclass(frozen=True)
class SignalMarket:
    cfg: AegisConfig
    features: np.ndarray
    signal_features: np.ndarray
    close: np.ndarray
    timestamps: np.ndarray
    regimes: np.ndarray
    feature_names: list[str]
    steps: np.ndarray


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) <= 1e-12:
        return float(default)
    return float(numerator / denominator)


def profit_factor(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float32)
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    if gross_loss <= 0.0:
        return 999.0 if gross_win > 0.0 else 0.0
    return gross_win / gross_loss


def win_rate(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=np.float32)
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0.0))


def return_stats(returns: np.ndarray) -> dict[str, float]:
    returns = np.asarray(returns, dtype=np.float32)
    if len(returns) == 0:
        return {
            "count": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "positive_rate": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
        }
    return {
        "count": float(len(returns)),
        "mean": safe_float(np.mean(returns)),
        "median": safe_float(np.median(returns)),
        "std": safe_float(np.std(returns)),
        "min": safe_float(np.min(returns)),
        "max": safe_float(np.max(returns)),
        "positive_rate": safe_float(np.mean(returns > 0.0)),
        "profit_factor": safe_float(profit_factor(returns)),
        "win_rate": safe_float(win_rate(returns)),
    }


def drawdown_stats(equity: np.ndarray) -> dict[str, float]:
    equity = np.asarray(equity, dtype=np.float32)
    if len(equity) == 0:
        return {"max_dd": 0.0, "p95_dd": 0.0, "avg_dd": 0.0}
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-10)
    return {
        "max_dd": safe_float(np.max(dd)),
        "p95_dd": safe_float(np.quantile(dd, 0.95)),
        "avg_dd": safe_float(np.mean(dd)),
    }


def percentile_threshold(scores: np.ndarray, pct: float, high: bool = True) -> float:
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0
    pct = min(max(float(pct), 0.0), 1.0)
    q = 1.0 - pct if high else pct
    return safe_float(np.quantile(scores, q))


def evaluate_ranked_slice(
    scores: np.ndarray,
    values: np.ndarray,
    regimes: np.ndarray,
    pct: float,
    high: bool = True,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    regimes = np.asarray(regimes)
    if len(scores) != len(values):
        raise ValueError("scores and values length mismatch")
    if len(scores) == 0:
        return {
            "count": 0,
            "threshold": 0.0,
            "avg_return_after_fees": 0.0,
            "median_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_mfe": 0.0,
            "avg_mae": 0.0,
            "regime_distribution": {},
            "dominance": 0.0,
        }

    threshold = percentile_threshold(scores, pct, high=high)
    mask = scores >= threshold if high else scores <= threshold
    selected = values[mask]
    selected_regimes = regimes[mask]
    return {
        "count": int(mask.sum()),
        "threshold": safe_float(threshold),
        "avg_return_after_fees": safe_float(np.mean(selected)) if len(selected) else 0.0,
        "median_return": safe_float(np.median(selected)) if len(selected) else 0.0,
        "win_rate": safe_float(np.mean(selected > 0.0)) if len(selected) else 0.0,
        "profit_factor": safe_float(profit_factor(selected)) if len(selected) else 0.0,
        "avg_mfe": 0.0,
        "avg_mae": 0.0,
        "regime_distribution": {str(k): int(v) for k, v in zip(*np.unique(selected_regimes, return_counts=True))} if len(selected_regimes) else {},
        "dominance": safe_float(max(np.mean(selected > 0.0), np.mean(selected <= 0.0))) if len(selected) else 0.0,
    }


def build_signal_feature_matrix(features: np.ndarray, window_size: int) -> np.ndarray:
    return build_edge_feature_matrix(features, window_size)


def load_signal_market(config_path: str | Path) -> SignalMarket:
    cfg = load_config(config_path)
    symbol = cfg.symbol if "/" in cfg.symbol else cfg.symbol.replace("USDT", "/USDT")
    db = DatabaseManager(cfg.database_url)
    df = db.get_ohlcv_data(symbol, cfg.timeframe)
    if df.empty and symbol != cfg.symbol:
        df = db.get_ohlcv_data(cfg.symbol, cfg.timeframe)
    if df.empty:
        raise RuntimeError(f"No candles found for {cfg.symbol} {cfg.timeframe}")

    frame = build_feature_frame(df)
    features = frame[FEATURE_COLUMNS].values.astype(np.float32)
    close = frame["close"].values.astype(np.float32)
    timestamps = frame.index.astype(str).values
    window_size = cfg.model.window_size
    regimes = np.empty((len(features),), dtype="U16")
    for idx in range(len(features)):
        start = max(0, idx - window_size + 1)
        regimes[idx] = detect_regime(features[start : idx + 1]).type
    signal_features = build_signal_feature_matrix(features, window_size)
    steps = np.arange(window_size, len(features), dtype=np.int64)
    feature_names = edge_feature_names(FEATURE_COLUMNS)
    return SignalMarket(
        cfg=cfg,
        features=features,
        signal_features=signal_features,
        close=close,
        timestamps=timestamps,
        regimes=regimes,
        feature_names=feature_names,
        steps=steps,
    )

