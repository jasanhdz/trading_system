"""Causal sequential dataset primitives for Aegis W10 research."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis.research.general_orderbook_direction_w9_1 import (
    _first_barrier,
    load_quotes,
    load_trades,
    sample_l2,
)


LOOKBACK_SECONDS = (5, 10, 20)
LATENCIES_MS = (0, 100, 250, 500, 1_000)
DECISION_SECONDS = 5
EPISODE_SECONDS = 120


def stable_momentum_episode_id(symbol: str, anchor_us: int) -> str:
    material = f"W10|{symbol}|{anchor_us}".encode()
    return "W10-" + hashlib.sha256(material).hexdigest()


def episode_anchors(day_start_us: int, day_end_us: int) -> np.ndarray:
    first = day_start_us + 300 * 1_000_000
    last = day_end_us - (EPISODE_SECONDS + 60) * 1_000_000
    return np.arange(first, last + 1, 300 * 1_000_000, dtype=np.int64)


def decision_grid(anchors: np.ndarray) -> np.ndarray:
    offsets = np.arange(0, EPISODE_SECONDS + 1, DECISION_SECONDS, dtype=np.int64) * 1_000_000
    return (anchors[:, None] + offsets[None, :]).ravel()


def required_book_times(decisions: np.ndarray) -> np.ndarray:
    offsets = np.array((0, -5_000_000, -10_000_000, -20_000_000), dtype=np.int64)
    return np.unique((decisions[:, None] + offsets[None, :]).ravel())


def _latest_indices(timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.searchsorted(timestamps, targets, side="right") - 1


def _window_trade_features(
    trades: pd.DataFrame,
    timestamps: np.ndarray,
    start_us: int,
    end_us: int,
) -> dict[str, float]:
    start = int(np.searchsorted(timestamps, start_us, side="right"))
    end = int(np.searchsorted(timestamps, end_us, side="right"))
    window = trades.iloc[start:end]
    if window.empty:
        return {
            "buy_notional": 0.0,
            "sell_notional": 0.0,
            "delta_ratio": 0.0,
            "trade_rate": 0.0,
            "mean_trade_notional": 0.0,
            "large_trade_fraction": 0.0,
        }
    buy = float(window.loc[window["side"].eq("buy"), "notional"].sum())
    sell = float(window.loc[window["side"].eq("sell"), "notional"].sum())
    total = buy + sell
    return {
        "buy_notional": buy,
        "sell_notional": sell,
        "delta_ratio": (buy - sell) / total if total else 0.0,
        "trade_rate": len(window) / max((end_us - start_us) / 1_000_000.0, 1e-9),
        "mean_trade_notional": float(window["notional"].mean()),
        "large_trade_fraction": float(window["notional"].max() / total) if total else 0.0,
    }


def _path_features(times: np.ndarray, mids: np.ndarray, now_us: int) -> dict[str, float]:
    targets = now_us - np.array((20, 15, 10, 5, 0), dtype=np.int64) * 1_000_000
    indices = _latest_indices(times, targets)
    if (indices < 0).any():
        raise ValueError("AEGIS_W10_QUOTE_WARMUP_MISSING")
    path = mids[indices]
    returns = np.diff(path) / path[:-1] * 10_000.0
    net = (path[-1] / path[0] - 1.0) * 10_000.0
    total_path = float(np.abs(returns).sum())
    return {
        "return_5s_bps": float((path[-1] / path[-2] - 1.0) * 10_000.0),
        "return_10s_bps": float((path[-1] / path[-3] - 1.0) * 10_000.0),
        "return_20s_bps": net,
        "velocity_5s_bps": float(returns[-1]),
        "acceleration_5s_bps": float(returns[-1] - returns[-2]),
        "persistence_20s": float(abs(np.sign(returns).sum()) / len(returns)),
        "path_efficiency_20s": abs(net) / total_path if total_path else 0.0,
        "up_excursion_20s_bps": float((path.max() / path[0] - 1.0) * 10_000.0),
        "down_excursion_20s_bps": float((path.min() / path[0] - 1.0) * 10_000.0),
    }


def _book_features(book: pd.DataFrame, now_us: int) -> dict[str, float]:
    current = book.loc[now_us]
    result = {
        "spread_bps": float(current["spread_bps"]),
        "microprice_minus_mid_bps": float(current["microprice_minus_mid_bps"]),
    }
    for level in (1, 5, 10, 20):
        result[f"obi_l{level}"] = float(current[f"obi_l{level}"])
    for distance in (1, 2, 5):
        result[f"bid_depth_{distance}bp"] = float(current[f"bid_depth_{distance}bp"])
        result[f"ask_depth_{distance}bp"] = float(current[f"ask_depth_{distance}bp"])
    for seconds in LOOKBACK_SECONDS:
        prior = book.loc[now_us - seconds * 1_000_000]
        scale = max(
            float(current["bid_depth_5bp"] + current["ask_depth_5bp"]),
            1e-12,
        )
        suffix = f"{seconds}s"
        for name in ("bid_depletion", "ask_depletion", "bid_replenishment", "ask_replenishment"):
            result[f"{name}_{suffix}"] = float(current[name] - prior[name]) / scale
        result[f"obi_l5_change_{suffix}"] = float(current["obi_l5"] - prior["obi_l5"])
        result[f"microprice_velocity_{suffix}"] = float(
            current["microprice_minus_mid_bps"] - prior["microprice_minus_mid_bps"]
        )
        result[f"spread_change_{suffix}"] = float(current["spread_bps"] - prior["spread_bps"])
    return result


def _latest_mid(quote_times: np.ndarray, mids: np.ndarray, timestamp_us: int) -> float:
    index = int(np.searchsorted(quote_times, timestamp_us, side="right") - 1)
    if index < 0:
        raise ValueError("AEGIS_W10_QUOTE_MISSING")
    return float(mids[index])


def _interval_extremes(
    quote_times: np.ndarray,
    mids: np.ndarray,
    start_us: int,
    end_us: int,
) -> tuple[float, float]:
    start = int(np.searchsorted(quote_times, start_us, side="right") - 1)
    end = int(np.searchsorted(quote_times, end_us, side="right"))
    path = mids[max(start, 0):end]
    if not len(path):
        value = _latest_mid(quote_times, mids, start_us)
        return value, value
    return float(path.max()), float(path.min())


def load_frozen_w7_scores(root: Path) -> dict[tuple[str, int], float]:
    path = root / "data/conditional_direction_w8/run_01/development_outcomes.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, columns=["timestamp", "symbol", "w7_opportunity_probability"])
    timestamps = pd.to_datetime(frame["timestamp"], utc=True).astype("int64") // 1_000
    return {
        (str(symbol), int(timestamp)): float(score)
        for symbol, timestamp, score in zip(frame["symbol"], timestamps, frame["w7_opportunity_probability"])
    }


def build_sequential_states(
    *,
    root: Path,
    symbol: str,
    date: str,
    l2_path: Path,
    quotes_path: Path,
    trades_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_us = int(pd.Timestamp(date, tz="UTC").timestamp() * 1_000_000)
    end_us = start_us + 86_400 * 1_000_000
    anchors = episode_anchors(start_us, end_us)
    decisions = decision_grid(anchors)
    book, book_audit = sample_l2(l2_path, required_book_times(decisions))
    if not book_audit["passes"]:
        raise RuntimeError("AEGIS_W10_L2_RECONSTRUCTION_FAILED")
    quotes = load_quotes(quotes_path)
    trades = load_trades(trades_path)
    quote_times = quotes["local_timestamp"].to_numpy(np.int64)
    mids = quotes["mid"].to_numpy(float)
    trade_times = trades["local_timestamp"].to_numpy(np.int64)
    w7_scores = load_frozen_w7_scores(root)
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        episode_id = stable_momentum_episode_id(symbol, int(anchor))
        w7_score = w7_scores.get((symbol, int(anchor)), math.nan)
        for step, now in enumerate(range(int(anchor), int(anchor) + EPISODE_SECONDS * 1_000_000 + 1, DECISION_SECONDS * 1_000_000)):
            price = _path_features(quote_times, mids, now)
            book_state = _book_features(book, now)
            flow: dict[str, float] = {}
            for seconds in LOOKBACK_SECONDS:
                values = _window_trade_features(trades, trade_times, now - seconds * 1_000_000, now)
                for name, value in values.items():
                    flow[f"{name}_{seconds}s"] = value
                response = price[f"return_{seconds}s_bps"] if seconds in (5, 10, 20) else 0.0
                flow[f"signed_flow_price_response_{seconds}s"] = values["delta_ratio"] * response
                flow[f"flow_without_response_{seconds}s"] = abs(values["delta_ratio"]) / (abs(response) + 1.0)
                flow[f"sell_absorption_{seconds}s"] = max(-values["delta_ratio"], 0.0) * max(book_state[f"bid_replenishment_{seconds}s"], 0.0) / (abs(response) + 1.0)
                flow[f"buy_absorption_{seconds}s"] = max(values["delta_ratio"], 0.0) * max(book_state[f"ask_replenishment_{seconds}s"], 0.0) / (abs(response) + 1.0)
            label, terminal, mfe, mae = _first_barrier(
                quote_times, mids, now, latency_ms=0, barrier_bps=20, horizon_seconds=60
            )
            interval_high, interval_low = _interval_extremes(
                quote_times, mids, now, min(now + DECISION_SECONDS * 1_000_000, int(anchor) + EPISODE_SECONDS * 1_000_000)
            )
            row: dict[str, Any] = {
                "momentum_episode_id": episode_id,
                "symbol": symbol,
                "date": date,
                "anchor_timestamp_us": int(anchor),
                "decision_timestamp_us": now,
                "step": step,
                "w7_opportunity_probability": w7_score,
                **{f"price__{name}": value for name, value in price.items()},
                **{f"book__{name}": value for name, value in book_state.items()},
                **{f"flow__{name}": value for name, value in flow.items()},
                "label__b20_h60": label,
                "target__terminal_bps": terminal,
                "target__mfe_bps": mfe,
                "target__mae_bps": mae,
                "path__interval_high": interval_high,
                "path__interval_low": interval_low,
            }
            for latency in LATENCIES_MS:
                row[f"execution_mid_l{latency}"] = _latest_mid(
                    quote_times, mids, now + latency * 1_000
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    feature_columns = [column for column in frame if column.startswith(("price__", "book__", "flow__"))]
    if not np.isfinite(frame[feature_columns].to_numpy(float)).all():
        raise RuntimeError("AEGIS_W10_NONFINITE_FEATURE")
    return frame, {
        "symbol": symbol,
        "date": date,
        "episodes": len(anchors),
        "states": len(frame),
        "feature_count": len(feature_columns),
        "w7_covered_episodes": int(frame.groupby("momentum_episode_id")["w7_opportunity_probability"].first().notna().sum()),
        "l2": book_audit,
        "quotes": len(quotes),
        "trades": len(trades),
        "passes": True,
    }
