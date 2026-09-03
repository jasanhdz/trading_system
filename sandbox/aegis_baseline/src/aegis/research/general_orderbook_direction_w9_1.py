"""Causal episode construction for W9.1 general order-book direction research."""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WINDOWS_MS = (100, 250, 500, 1_000, 2_000, 5_000)
LATENCIES_MS = (0, 100, 250, 500, 1_000)


@dataclass
class CausalBook:
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=lambda: {
        "bid_depletion": 0.0,
        "ask_depletion": 0.0,
        "bid_replenishment": 0.0,
        "ask_replenishment": 0.0,
    })
    ready: bool = False
    generation: int = 0
    best_bid: float | None = None
    best_ask: float | None = None

    def apply(self, rows: list[dict[str, str]]) -> None:
        snapshot = rows[0]["is_snapshot"].lower() == "true"
        if any((row["is_snapshot"].lower() == "true") != snapshot for row in rows):
            raise ValueError("AEGIS_W9_1_MIXED_MESSAGE")
        if snapshot:
            self.bids.clear()
            self.asks.clear()
            self.ready = False
            self.generation += 1
            self.best_bid = None
            self.best_ask = None
        elif not self.ready:
            raise ValueError("AEGIS_W9_1_UPDATE_BEFORE_SNAPSHOT")
        for row in rows:
            side = row["side"]
            levels = self.bids if side == "bid" else self.asks if side == "ask" else None
            if levels is None:
                raise ValueError("AEGIS_W9_1_SIDE_INVALID")
            price = float(row["price"])
            amount = float(row["amount"])
            if price <= 0.0 or amount < 0.0 or not math.isfinite(price + amount):
                raise ValueError("AEGIS_W9_1_LEVEL_INVALID")
            previous = levels.get(price, 0.0)
            if not snapshot:
                delta = amount - previous
                key = f"{side}_{'replenishment' if delta > 0 else 'depletion'}"
                self.counters[key] += abs(delta)
            if amount == 0.0:
                levels.pop(price, None)
            else:
                levels[price] = amount
        if snapshot:
            self.ready = True
            self.best_bid = max(self.bids) if self.bids else None
            self.best_ask = min(self.asks) if self.asks else None
        else:
            for row in rows:
                side = row["side"]
                price = float(row["price"])
                amount = float(row["amount"])
                if side == "bid":
                    if amount > 0.0 and (self.best_bid is None or price > self.best_bid):
                        self.best_bid = price
                    elif amount == 0.0 and price == self.best_bid:
                        self.best_bid = max(self.bids) if self.bids else None
                else:
                    if amount > 0.0 and (self.best_ask is None or price < self.best_ask):
                        self.best_ask = price
                    elif amount == 0.0 and price == self.best_ask:
                        self.best_ask = min(self.asks) if self.asks else None
        self.validate()

    def validate(self) -> None:
        if not self.ready or self.best_bid is None or self.best_ask is None:
            raise ValueError("AEGIS_W9_1_BOOK_INCOMPLETE")
        if self.best_bid >= self.best_ask:
            raise ValueError("AEGIS_W9_1_BOOK_CROSSED")

    def snapshot(self) -> dict[str, float]:
        self.validate()
        bids = sorted(self.bids.items(), reverse=True)
        asks = sorted(self.asks.items())
        best_bid, bid_amount = bids[0]
        best_ask, ask_amount = asks[0]
        mid = (best_bid + best_ask) / 2.0
        micro = (best_ask * bid_amount + best_bid * ask_amount) / (bid_amount + ask_amount)
        result: dict[str, float] = {
            "mid": mid,
            "spread_bps": (best_ask - best_bid) / mid * 10_000.0,
            "microprice_minus_mid_bps": (micro / mid - 1.0) * 10_000.0,
            "snapshot_generation": float(self.generation),
            **self.counters,
        }
        for levels in (1, 5, 10, 20):
            bid_depth = sum(amount for _, amount in bids[:levels])
            ask_depth = sum(amount for _, amount in asks[:levels])
            result[f"obi_l{levels}"] = _imbalance(bid_depth, ask_depth)
        for distance in (1, 2, 5):
            lower = mid * (1.0 - distance / 10_000.0)
            upper = mid * (1.0 + distance / 10_000.0)
            result[f"bid_depth_{distance}bp"] = sum(amount for price, amount in bids if price >= lower)
            result[f"ask_depth_{distance}bp"] = sum(amount for price, amount in asks if price <= upper)
        return result


def _imbalance(bid: float, ask: float) -> float:
    total = bid + ask
    return (bid - ask) / total if total > 0.0 else 0.0


def stable_episode_id(symbol: str, anchor_us: int) -> str:
    material = f"W9.1|{symbol}|{anchor_us}".encode()
    return "W9.1-" + hashlib.sha256(material).hexdigest()


def anchor_grid(day_start_us: int, day_end_us: int, *, warmup_seconds: int = 10, spacing_seconds: int = 120) -> np.ndarray:
    first = day_start_us + warmup_seconds * 1_000_000
    last = day_end_us - 60 * 1_000_000
    return np.arange(first, last + 1, spacing_seconds * 1_000_000, dtype=np.int64)


def requested_book_times(anchors: np.ndarray) -> np.ndarray:
    offsets = np.array((0,) + tuple(-window * 1_000 for window in WINDOWS_MS), dtype=np.int64)
    return np.unique((anchors[:, None] + offsets[None, :]).ravel())


def sample_l2(path: Path, sample_times: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    book = CausalBook()
    samples: list[dict[str, float]] = []
    sample_index = 0
    rows_count = messages = snapshots = 0
    current_local: int | None = None
    previous_local: int | None = None
    group: list[dict[str, str]] = []

    def capture_until(limit: int, inclusive: bool) -> None:
        nonlocal sample_index
        while sample_index < len(sample_times):
            timestamp = int(sample_times[sample_index])
            if timestamp > limit or (timestamp == limit and not inclusive):
                break
            if book.ready:
                samples.append({"sample_timestamp_us": timestamp, **book.snapshot()})
            sample_index += 1

    def consume(local: int, message_rows: list[dict[str, str]]) -> None:
        nonlocal messages, snapshots, previous_local
        if previous_local is not None and local < previous_local:
            raise ValueError("AEGIS_W9_1_LOCAL_TIMESTAMP_REORDERED")
        capture_until(local, inclusive=False)
        book.apply(message_rows)
        messages += 1
        snapshots += int(message_rows[0]["is_snapshot"].lower() == "true")
        capture_until(local, inclusive=True)
        previous_local = local

    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_count += 1
            local = int(row["local_timestamp"])
            if current_local is None:
                current_local = local
            if local != current_local:
                consume(current_local, group)
                group = []
                current_local = local
            group.append(row)
        if group and current_local is not None:
            consume(current_local, group)
    if previous_local is not None:
        capture_until(int(sample_times[-1]), inclusive=True)
    frame = pd.DataFrame(samples).set_index("sample_timestamp_us").sort_index()
    return frame, {
        "rows": rows_count,
        "messages": messages,
        "snapshots": snapshots,
        "requested_samples": len(sample_times),
        "captured_samples": len(frame),
        "passes": len(frame) == len(sample_times) and snapshots > 0,
    }


def load_quotes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "local_timestamp", "ask_price", "bid_price", "ask_amount", "bid_amount"])
    frame["mid"] = (frame["ask_price"] + frame["bid_price"]) / 2.0
    if not frame["local_timestamp"].is_monotonic_increasing:
        raise ValueError("AEGIS_W9_1_QUOTES_REORDERED")
    if (frame["bid_price"] >= frame["ask_price"]).any():
        raise ValueError("AEGIS_W9_1_QUOTES_CROSSED")
    return frame


def load_trades(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["timestamp", "local_timestamp", "id", "side", "price", "amount"])
    if not frame["local_timestamp"].is_monotonic_increasing:
        raise ValueError("AEGIS_W9_1_TRADES_REORDERED")
    frame["notional"] = frame["price"] * frame["amount"]
    frame["signed_notional"] = np.where(frame["side"].eq("buy"), frame["notional"], -frame["notional"])
    return frame


def _latest_index(timestamps: np.ndarray, target: int) -> int:
    return int(np.searchsorted(timestamps, target, side="right") - 1)


def _first_barrier(
    quote_times: np.ndarray,
    mids: np.ndarray,
    anchor_us: int,
    *,
    latency_ms: int,
    barrier_bps: int,
    horizon_seconds: int,
) -> tuple[str, float, float, float]:
    decision_us = anchor_us + latency_ms * 1_000
    start = _latest_index(quote_times, decision_us)
    if start < 0:
        return "MISSING", math.nan, math.nan, math.nan
    end = int(np.searchsorted(quote_times, decision_us + horizon_seconds * 1_000_000, side="right"))
    reference = mids[start]
    path = mids[start + 1 : end]
    if not len(path):
        return "MISSING", reference, math.nan, math.nan
    upper = reference * (1.0 + barrier_bps / 10_000.0)
    lower = reference * (1.0 - barrier_bps / 10_000.0)
    up = np.flatnonzero(path >= upper)
    down = np.flatnonzero(path <= lower)
    label = "NEITHER"
    if len(up) and (not len(down) or up[0] < down[0]):
        label = "UP_FIRST"
    elif len(down):
        label = "DOWN_FIRST"
    terminal = (path[-1] / reference - 1.0) * 10_000.0
    mfe = (path.max() / reference - 1.0) * 10_000.0
    mae = (path.min() / reference - 1.0) * 10_000.0
    return label, terminal, mfe, mae


def build_episodes(
    *,
    symbol: str,
    date: str,
    l2_path: Path,
    quotes_path: Path,
    trades_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    quotes = load_quotes(quotes_path)
    trades = load_trades(trades_path)
    quote_times = quotes["local_timestamp"].to_numpy(np.int64)
    mids = quotes["mid"].to_numpy(float)
    day_start = int(pd.Timestamp(date, tz="UTC").timestamp() * 1_000_000)
    day_end = day_start + 86_400 * 1_000_000
    anchors = anchor_grid(day_start, day_end)
    book_samples, l2_audit = sample_l2(l2_path, requested_book_times(anchors))
    trade_times = trades["local_timestamp"].to_numpy(np.int64)
    rows: list[dict[str, Any]] = []
    skipped_snapshot_resets = 0
    for anchor in anchors:
        if anchor not in book_samples.index:
            continue
        current = book_samples.loc[anchor]
        history = [book_samples.loc[anchor - window_ms * 1_000] for window_ms in WINDOWS_MS]
        if any(previous["snapshot_generation"] != current["snapshot_generation"] for previous in history):
            skipped_snapshot_resets += 1
            continue
        row: dict[str, Any] = {
            "orderbook_episode_id": stable_episode_id(symbol, int(anchor)),
            "symbol": symbol,
            "date": date,
            "anchor_timestamp_us": int(anchor),
        }
        quote_index = _latest_index(quote_times, int(anchor))
        if quote_index < 0:
            continue
        row["quality__l2_quote_mid_difference_bps"] = abs(float(current["mid"]) / mids[quote_index] - 1.0) * 10_000.0
        static_columns = [name for name in current.index if not name.endswith(("depletion", "replenishment")) and name not in {"mid", "snapshot_generation"}]
        for name in static_columns:
            row[f"static__{name}"] = float(current[name])
        for window_ms in WINDOWS_MS:
            previous = book_samples.loc[anchor - window_ms * 1_000]
            seconds = window_ms / 1_000.0
            depth_scale = max(float(current["bid_depth_5bp"] + current["ask_depth_5bp"]), 1e-12)
            for name in ("bid_depletion", "ask_depletion", "bid_replenishment", "ask_replenishment"):
                row[f"dynamics__{name}_{window_ms}ms"] = float(current[name] - previous[name]) / depth_scale
            row[f"dynamics__spread_change_{window_ms}ms"] = float(current["spread_bps"] - previous["spread_bps"])
            row[f"dynamics__microprice_velocity_{window_ms}ms"] = float(current["microprice_minus_mid_bps"] - previous["microprice_minus_mid_bps"]) / seconds
            row[f"dynamics__obi_l5_change_{window_ms}ms"] = float(current["obi_l5"] - previous["obi_l5"])
            left = int(np.searchsorted(trade_times, anchor - window_ms * 1_000, side="right"))
            right = int(np.searchsorted(trade_times, anchor, side="right"))
            window = trades.iloc[left:right]
            buy = float(window.loc[window["side"].eq("buy"), "notional"].sum())
            sell = float(window.loc[window["side"].eq("sell"), "notional"].sum())
            total = buy + sell
            delta = buy - sell
            count = len(window)
            large = float((window["notional"] >= 10_000.0).mean()) if count else 0.0
            row[f"flow__buy_notional_{window_ms}ms"] = buy
            row[f"flow__sell_notional_{window_ms}ms"] = sell
            row[f"flow__signed_delta_ratio_{window_ms}ms"] = delta / total if total else 0.0
            row[f"flow__trade_rate_{window_ms}ms"] = count / seconds
            row[f"flow__mean_trade_notional_{window_ms}ms"] = total / count if count else 0.0
            row[f"flow__large_trade_fraction_{window_ms}ms"] = large
            price_return = (float(current["mid"]) / float(previous["mid"]) - 1.0) * 10_000.0
            normalized_delta = delta / max(total, 1.0)
            row[f"response__price_return_{window_ms}ms"] = price_return
            row[f"response__signed_flow_price_response_{window_ms}ms"] = normalized_delta * price_return
            row[f"response__impact_per_10k_delta_{window_ms}ms"] = price_return / max(abs(delta) / 10_000.0, 1.0)
            row[f"response__flow_without_response_{window_ms}ms"] = abs(normalized_delta) / (abs(price_return) + 0.1)
            bid_replenishment = row[f"dynamics__bid_replenishment_{window_ms}ms"]
            ask_replenishment = row[f"dynamics__ask_replenishment_{window_ms}ms"]
            sell_share = sell / total if total else 0.0
            buy_share = buy / total if total else 0.0
            row[f"absorption__sell_pressure_bid_replenishment_{window_ms}ms"] = sell_share * bid_replenishment / (max(-price_return, 0.0) + 1.0)
            row[f"absorption__buy_pressure_ask_replenishment_{window_ms}ms"] = buy_share * ask_replenishment / (max(price_return, 0.0) + 1.0)
        for barrier, horizon in ((10, 30), (15, 60), (25, 60)):
            for latency in LATENCIES_MS:
                label, terminal, mfe, mae = _first_barrier(
                    quote_times, mids, int(anchor), latency_ms=latency,
                    barrier_bps=barrier, horizon_seconds=horizon,
                )
                prefix = f"target__b{barrier}_h{horizon}_l{latency}"
                row[f"{prefix}_label"] = label
                row[f"{prefix}_terminal_bps"] = terminal
                row[f"{prefix}_mfe_bps"] = mfe
                row[f"{prefix}_mae_bps"] = mae
        rows.append(row)
    episodes = pd.DataFrame(rows)
    if episodes["orderbook_episode_id"].duplicated().any():
        raise ValueError("AEGIS_W9_1_EPISODE_DUPLICATE")
    quote_p99 = float(episodes["quality__l2_quote_mid_difference_bps"].quantile(0.99))
    return episodes, {
        "symbol": symbol,
        "date": date,
        "episodes": len(episodes),
        "l2": l2_audit,
        "quote_rows": len(quotes),
        "trade_rows": len(trades),
        "skipped_snapshot_resets": skipped_snapshot_resets,
        "l2_quote_mid_difference_bps_p99": quote_p99,
        "passes": (
            len(episodes) + skipped_snapshot_resets == len(anchors)
            and l2_audit["passes"] and quote_p99 <= 5.0
        ),
    }
