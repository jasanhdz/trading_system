"""Columnar, causal batch evaluation for the preregistered M1A experiment."""

from __future__ import annotations

import hashlib
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .market_event_fast_track_m1a import (
    DirectionAxis,
    FastTrackContractError,
    PatternThresholds,
    RegimeThresholds,
)

KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
)
DISCOVERY_END_MS = int(pd.Timestamp("2025-04-01T00:00:00Z").timestamp() * 1000)
VALIDATION_END_MS = int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp() * 1000)
PSEUDO_END_MS = int(pd.Timestamp("2026-08-01T00:00:00Z").timestamp() * 1000)


def _read_month(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise FastTrackContractError("AEGIS_M1A_BATCH_ARCHIVE_INVALID")
        with archive.open(members[0]) as handle:
            frame = pd.read_csv(handle, header=None, names=KLINE_COLUMNS)
    numeric = pd.to_numeric(frame["open_time"], errors="coerce")
    frame = frame.loc[numeric.notna()].copy()
    timestamps = numeric.loc[numeric.notna()].astype("int64")
    timestamps = timestamps.where(timestamps < 100_000_000_000_000, timestamps // 1000)
    frame["open_time"] = timestamps
    for name in KLINE_COLUMNS[1:11]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    return frame


def load_symbol_frame(archive_root: Path, symbol: str) -> pd.DataFrame:
    futures_paths = sorted(
        (archive_root / "futures/um/monthly/klines" / symbol / "1m").glob("*.zip")
    )
    spot_paths = sorted(
        (archive_root / "spot/monthly/klines" / symbol / "1m").glob("*.zip")
    )
    if not futures_paths or len(futures_paths) != len(spot_paths):
        raise FastTrackContractError("AEGIS_M1A_BATCH_ARCHIVE_COVERAGE_INVALID")
    futures = pd.concat(
        (_read_month(path) for path in futures_paths), ignore_index=True
    )
    spot = pd.concat((_read_month(path) for path in spot_paths), ignore_index=True)
    if futures["open_time"].duplicated().any() or spot["open_time"].duplicated().any():
        raise FastTrackContractError("AEGIS_M1A_BATCH_DUPLICATE_TIMESTAMP")
    frame = futures.merge(
        spot[["open_time", "close"]].rename(columns={"close": "spot_close"}),
        on="open_time",
        how="inner",
        validate="one_to_one",
    ).sort_values("open_time", ignore_index=True)
    frame["symbol"] = symbol
    frame["flow_1"] = (
        2.0 * frame["taker_buy_quote"] / frame["quote_volume"].replace(0.0, np.nan)
        - 1.0
    ).fillna(0.0)
    return frame


def build_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    close = result["close"]
    for bars in (1, 3, 12, 60, 240):
        result[f"ret_{bars}"] = close / close.shift(bars) - 1.0
    result["flow_3"] = result["flow_1"].rolling(3, min_periods=3).mean()
    result["flow_12"] = result["flow_1"].rolling(12, min_periods=12).mean()
    result["volume_ratio"] = (
        result["quote_volume"]
        / result["quote_volume"].shift(1).rolling(24, min_periods=24).mean()
    )
    high_6 = result["high"].rolling(6, min_periods=6).max()
    low_6 = result["low"].rolling(6, min_periods=6).min()
    high_24 = result["high"].rolling(24, min_periods=24).max()
    low_24 = result["low"].rolling(24, min_periods=24).min()
    result["compression"] = (high_6 - low_6) / (high_24 - low_24).replace(0.0, np.nan)
    result["prior_high"] = result["high"].shift(1).rolling(20, min_periods=20).max()
    result["prior_low"] = result["low"].shift(1).rolling(20, min_periods=20).min()
    result["breakout_up"] = close / result["prior_high"] - 1.0
    result["breakout_down"] = result["prior_low"] / close - 1.0
    result["body"] = (close - result["open"]).abs() / result["open"]
    result["upper_wick"] = (
        result["high"] - result[["open", "close"]].max(axis=1)
    ) / result["open"]
    result["lower_wick"] = (
        result[["open", "close"]].min(axis=1) - result["low"]
    ) / result["open"]
    result["spot_ret_12"] = result["spot_close"] / result["spot_close"].shift(12) - 1.0
    result["basis"] = close / result["spot_close"] - 1.0
    result["prior_basis"] = result["basis"].shift(12)
    result["basis_convergence"] = result["prior_basis"].abs() - result["basis"].abs()
    result["price_response"] = result["ret_3"].abs()
    gaps = result["open_time"].diff().ne(60_000)
    segment = gaps.cumsum()
    segment_size = result.groupby(segment)["open_time"].transform("size")
    segment_index = result.groupby(segment).cumcount()
    result["causal_ready"] = (segment_size >= 301) & (segment_index >= 240)
    return result


def _quantile(series: pd.Series, probability: float) -> float:
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise FastTrackContractError("AEGIS_M1A_BATCH_QUANTILE_EMPTY")
    return float(values.quantile(probability, interpolation="linear"))


def fit_global_pattern_thresholds(frames: Sequence[pd.DataFrame]) -> PatternThresholds:
    required = [
        "flow_3",
        "volume_ratio",
        "compression",
        "breakout_up",
        "breakout_down",
        "upper_wick",
        "lower_wick",
        "body",
        "ret_60",
        "price_response",
        "prior_basis",
        "ret_3",
    ]
    train = pd.concat(
        [
            frame.loc[
                (frame["open_time"] < DISCOVERY_END_MS) & frame["causal_ready"],
                required,
            ]
            for frame in frames
        ],
        ignore_index=True,
    )
    if len(train) < 1000:
        raise FastTrackContractError("AEGIS_M1A_BATCH_TRAIN_INSUFFICIENT")
    wick_ratio = train[["upper_wick", "lower_wick"]].max(axis=1) / train["body"].clip(
        lower=1e-12
    )
    breakout = train[["breakout_up", "breakout_down"]].max(axis=1).clip(lower=0.0)
    return PatternThresholds(
        minimum_flow_imbalance=_quantile(train["flow_3"].abs(), 0.90),
        minimum_volume_ratio=_quantile(train["volume_ratio"], 0.80),
        maximum_compression_ratio=_quantile(train["compression"], 0.20),
        minimum_breakout_fraction=_quantile(breakout, 0.90),
        minimum_wick_body_ratio=_quantile(wick_ratio, 0.80),
        minimum_extension_fraction=_quantile(train["ret_60"].abs(), 0.90),
        maximum_absorption_response=_quantile(train["price_response"], 0.20),
        minimum_basis_divergence=_quantile(train["prior_basis"].abs(), 0.90),
        minimum_reclaim_fraction=_quantile(train["ret_3"].abs(), 0.60),
        minimum_session_move=_quantile(train["ret_60"].abs(), 0.85),
    )


def build_hourly_regime(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.copy()
    indexed.index = pd.to_datetime(indexed["open_time"], unit="ms", utc=True)
    indexed.index.name = "timestamp"
    hourly = indexed.resample("1h", closed="left", label="right").agg(
        close=("close", "last"),
        quote_volume=("quote_volume", "sum"),
        count=("close", "count"),
    )
    four = indexed.resample("4h", closed="left", label="right").agg(
        close=("close", "last"), count=("close", "count")
    )
    hourly = hourly.loc[hourly["count"] == 60].copy()
    four = four.loc[four["count"] == 240].copy()
    hourly["ret_1h"] = hourly["close"].pct_change()
    hourly["ema8"] = hourly["close"].ewm(span=8, adjust=False).mean()
    hourly["ema24"] = hourly["close"].ewm(span=24, adjust=False).mean()
    hourly["volatility"] = (
        hourly["close"].pct_change().pow(2).rolling(23, min_periods=23).mean().pow(0.5)
    )
    hourly["liquidity"] = (
        hourly["quote_volume"]
        / hourly["quote_volume"].rolling(168, min_periods=168).median()
    )
    four["ret_4h"] = four["close"].pct_change()
    hourly = pd.merge_asof(
        hourly.reset_index().sort_values("timestamp"),
        four.reset_index()[["timestamp", "ret_4h"]].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    hourly["direction_score"] = (
        0.35 * hourly["ret_1h"]
        + 0.45 * hourly["ret_4h"]
        + 0.20 * (hourly["ema8"] / hourly["ema24"] - 1.0)
    )
    hourly["timestamp_ms"] = (
        hourly["timestamp"].astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
    )
    return hourly.dropna(subset=["direction_score", "volatility", "liquidity"])


def fit_global_regime_thresholds(
    hourly_frames: Sequence[pd.DataFrame],
) -> RegimeThresholds:
    train = pd.concat(
        [
            frame.loc[frame["timestamp_ms"] < DISCOVERY_END_MS]
            for frame in hourly_frames
        ],
        ignore_index=True,
    )
    enter = _quantile(train["direction_score"].abs(), 0.70)
    return RegimeThresholds(
        enter,
        enter * 0.60,
        _quantile(train["volatility"], 0.20),
        _quantile(train["volatility"], 0.70),
        _quantile(train["volatility"], 0.95),
        _quantile(train["liquidity"], 0.20),
        _quantile(train["liquidity"], 0.80),
    )


def classify_hourly_regime(
    frame: pd.DataFrame, thresholds: RegimeThresholds
) -> pd.DataFrame:
    result = frame.copy()
    direction: list[str] = []
    current = DirectionAxis.TRANSITION.value
    pending = current
    pending_count = 0
    for score in result["direction_score"]:
        boundary = (
            thresholds.direction_exit
            if current in {"BULL", "BEAR"}
            else thresholds.direction_enter
        )
        candidate = (
            "BULL" if score >= boundary else "BEAR" if score <= -boundary else "RANGE"
        )
        if candidate == current:
            pending, pending_count = candidate, 0
        elif candidate == pending:
            pending_count += 1
            if pending_count >= 3:
                current, pending_count = candidate, 0
        else:
            pending, pending_count = candidate, 1
        direction.append(current)
    result["regime_direction"] = direction
    result["regime_volatility"] = np.select(
        [
            result["volatility"] <= thresholds.compressed_volatility,
            result["volatility"] >= thresholds.extreme_volatility,
            result["volatility"] >= thresholds.expanding_volatility,
        ],
        ["COMPRESSED", "EXTREME", "EXPANDING"],
        default="NORMAL",
    )
    return result[["timestamp_ms", "regime_direction", "regime_volatility"]]


def attach_regime(frame: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(
        frame.sort_values("open_time"),
        hourly.sort_values("timestamp_ms"),
        left_on="open_time",
        right_on="timestamp_ms",
        direction="backward",
    )


def detect_candidates(
    frame: pd.DataFrame, thresholds: PatternThresholds
) -> pd.DataFrame:
    f = frame
    records = []

    def add(pattern: str, side: str, mask: pd.Series) -> None:
        selected = f.loc[
            mask & f["causal_ready"] & f["regime_direction"].notna(),
            ["symbol", "open_time", "regime_direction", "regime_volatility"],
        ].copy()
        selected["pattern"] = pattern
        selected["side"] = side
        records.append(selected)

    bull = f["regime_direction"].eq("BULL")
    bear = f["regime_direction"].eq("BEAR")
    add(
        "TREND_PULLBACK_CONTINUATION",
        "LONG",
        bull
        & (f["ret_60"] > 0)
        & (f["ret_3"] > thresholds.minimum_reclaim_fraction)
        & (f["flow_3"] >= thresholds.minimum_flow_imbalance),
    )
    add(
        "TREND_PULLBACK_CONTINUATION",
        "SHORT",
        bear
        & (f["ret_60"] < 0)
        & (f["ret_3"] < -thresholds.minimum_reclaim_fraction)
        & (f["flow_3"] <= -thresholds.minimum_flow_imbalance),
    )
    compressed = (f["compression"] <= thresholds.maximum_compression_ratio) & (
        f["volume_ratio"] >= thresholds.minimum_volume_ratio
    )
    add(
        "COMPRESSION_BREAKOUT",
        "LONG",
        compressed
        & (f["breakout_up"] >= thresholds.minimum_breakout_fraction)
        & (f["flow_1"] >= thresholds.minimum_flow_imbalance),
    )
    add(
        "COMPRESSION_BREAKOUT",
        "SHORT",
        compressed
        & (f["breakout_down"] >= thresholds.minimum_breakout_fraction)
        & (f["flow_1"] <= -thresholds.minimum_flow_imbalance),
    )
    wick_base = f["body"].clip(lower=1e-12)
    add(
        "LIQUIDITY_SWEEP_REJECTION",
        "LONG",
        (f["low"] < f["prior_low"])
        & (f["lower_wick"] / wick_base >= thresholds.minimum_wick_body_ratio)
        & (f["close"] > f["prior_low"]),
    )
    add(
        "LIQUIDITY_SWEEP_REJECTION",
        "SHORT",
        (f["high"] > f["prior_high"])
        & (f["upper_wick"] / wick_base >= thresholds.minimum_wick_body_ratio)
        & (f["close"] < f["prior_high"]),
    )
    absorption = (
        (f["flow_3"].abs() >= thresholds.minimum_flow_imbalance)
        & (f["price_response"] <= thresholds.maximum_absorption_response)
        & (f["flow_1"] * f["flow_3"] < 0)
    )
    add("FLOW_PRICE_ABSORPTION_REVERSAL", "SHORT", absorption & (f["flow_3"] > 0))
    add("FLOW_PRICE_ABSORPTION_REVERSAL", "LONG", absorption & (f["flow_3"] < 0))
    exhaustion = (
        (f["ret_60"].abs() >= thresholds.minimum_extension_fraction)
        & (f["ret_3"].abs() < f["ret_12"].abs() / 4.0)
        & (f["flow_3"] * f["ret_60"] < 0)
    )
    add("EXHAUSTION_REVERSAL", "SHORT", exhaustion & (f["ret_60"] > 0))
    add("EXHAUSTION_REVERSAL", "LONG", exhaustion & (f["ret_60"] < 0))
    convergence = (f["prior_basis"].abs() >= thresholds.minimum_basis_divergence) & (
        f["basis_convergence"] >= thresholds.minimum_reclaim_fraction
    )
    add(
        "SPOT_FUTURES_DIVERGENCE_CONVERGENCE",
        "SHORT",
        convergence & (f["prior_basis"] > 0),
    )
    add(
        "SPOT_FUTURES_DIVERGENCE_CONVERGENCE",
        "LONG",
        convergence & (f["prior_basis"] < 0),
    )
    add(
        "MULTITIMEFRAME_RECLAIM",
        "LONG",
        bull
        & (f["ret_240"] > 0)
        & (f["ret_60"] < 0)
        & (f["ret_12"] >= thresholds.minimum_reclaim_fraction),
    )
    add(
        "MULTITIMEFRAME_RECLAIM",
        "SHORT",
        bear
        & (f["ret_240"] < 0)
        & (f["ret_60"] > 0)
        & (f["ret_12"] <= -thresholds.minimum_reclaim_fraction),
    )
    if not records:
        return pd.DataFrame()
    result = pd.concat(records, ignore_index=True).sort_values(
        ["open_time", "pattern", "side", "symbol"]
    )
    result["timestamp_ms"] = result["open_time"] + 60_000 - 1
    return result.drop(columns=["open_time"])


def partition_name(timestamp_ms: int) -> str:
    if timestamp_ms < DISCOVERY_END_MS:
        return "DISCOVERY"
    if timestamp_ms < VALIDATION_END_MS:
        return "VALIDATION"
    if timestamp_ms < PSEUDO_END_MS:
        return "PSEUDO_HOLDOUT"
    raise FastTrackContractError("AEGIS_M1A_BATCH_TIMESTAMP_OUTSIDE_PROTOCOL")


def collapse_events(events: pd.DataFrame) -> pd.DataFrame:
    accepted = []
    last_symbol: dict[tuple[str, str, str], int] = {}
    last_cluster: dict[tuple[str, str], int] = {}
    for row in events.sort_values(
        ["timestamp_ms", "pattern", "side", "symbol"]
    ).itertuples(index=False):
        symbol_key = (row.pattern, row.side, row.symbol)
        cluster_key = (row.pattern, row.side)
        if row.timestamp_ms - last_symbol.get(symbol_key, -(10**18)) <= 3_600_000:
            continue
        if row.timestamp_ms - last_cluster.get(cluster_key, -(10**18)) <= 900_000:
            continue
        accepted.append(row._asdict())
        last_symbol[symbol_key] = row.timestamp_ms
        last_cluster[cluster_key] = row.timestamp_ms
    return pd.DataFrame(accepted)


def evaluate_events(
    events: pd.DataFrame, frames: Mapping[str, pd.DataFrame], horizon: int = 60
) -> pd.DataFrame:
    outputs = []
    cost = 2.0 * (5.0 + 2.0) / 10_000.0 + 1.0 / 10_000.0 * horizon / 60.0
    for symbol, group in events.groupby("symbol"):
        frame = frames[symbol].sort_values("open_time", ignore_index=True)
        entry_times = group["timestamp_ms"].to_numpy(dtype=np.int64) + 1
        locations = pd.Index(frame["open_time"]).get_indexer(entry_times)
        valid = (locations >= 0) & (locations + horizon <= len(frame))
        end_locations = np.minimum(locations + horizon - 1, len(frame) - 1)
        contiguous = np.zeros(len(group), dtype=bool)
        contiguous[valid] = (
            frame["open_time"].to_numpy(dtype=np.int64)[end_locations[valid]]
            == entry_times[valid] + (horizon - 1) * 60_000
        )
        valid &= contiguous
        if not valid.any():
            continue
        selected = group.iloc[np.flatnonzero(valid)].copy().reset_index(drop=True)
        locations = locations[valid]
        entry_times = entry_times[valid]
        future_high = (
            frame["high"]
            .rolling(horizon, min_periods=horizon)
            .max()
            .shift(-(horizon - 1))
        ).to_numpy()[locations]
        future_low = (
            frame["low"]
            .rolling(horizon, min_periods=horizon)
            .min()
            .shift(-(horizon - 1))
        ).to_numpy()[locations]
        opens = frame["open"].to_numpy()[locations]
        exits = frame["close"].to_numpy()[locations + horizon - 1]
        signs = np.where(selected["side"].eq("LONG"), 1.0, -1.0)
        gross = signs * (exits - opens) / opens
        selected["partition"] = [
            partition_name(value) for value in selected["timestamp_ms"]
        ]
        selected["entry_timestamp_ms"] = entry_times
        selected["exit_timestamp_ms"] = entry_times + horizon * 60_000 - 1
        selected["gross_return_fraction"] = gross
        selected["cost_fraction"] = cost
        selected["net_return_fraction"] = gross - cost
        selected["mae_fraction"] = np.maximum(
            0.0,
            np.where(
                signs > 0, (opens - future_low) / opens, (future_high - opens) / opens
            ),
        )
        selected["mfe_fraction"] = np.maximum(
            0.0,
            np.where(
                signs > 0, (future_high - opens) / opens, (opens - future_low) / opens
            ),
        )
        outputs.append(selected)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def matched_random_control(
    events: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    horizon: int = 60,
) -> pd.DataFrame:
    outputs = []
    for symbol, symbol_events in events.groupby("symbol"):
        frame = frames[symbol]
        timestamp = frame["open_time"] + 59_999
        partitions = np.select(
            [
                timestamp < DISCOVERY_END_MS,
                timestamp < VALIDATION_END_MS,
                timestamp < PSEUDO_END_MS,
            ],
            ["DISCOVERY", "VALIDATION", "PSEUDO_HOLDOUT"],
            default="OUTSIDE_PROTOCOL",
        )
        last_entry = PSEUDO_END_MS - horizon * 60_000
        eligible = frame.loc[
            frame["open_time"].le(last_entry),
            ["open_time", "regime_direction", "regime_volatility"],
        ].copy()
        eligible["partition"] = partitions[frame["open_time"].le(last_entry)]
        pools = {
            key: group["open_time"].to_numpy(dtype=np.int64)
            for key, group in eligible.groupby(
                ["regime_direction", "regime_volatility", "partition"],
                dropna=True,
            )
        }
        for row in symbol_events.itertuples(index=False):
            pool = pools.get(
                (row.regime_direction, row.regime_volatility, row.partition)
            )
            if pool is None or len(pool) == 0:
                continue
            digest = hashlib.sha256(
                f"180101:{row.pattern}:{row.side}:{row.symbol}:{row.timestamp_ms}".encode()
            ).digest()
            start = int.from_bytes(digest[:8], "big") % len(pool)
            selected_time = None
            for offset in range(min(len(pool), 121)):
                candidate = int(pool[(start + offset) % len(pool)])
                if abs(candidate + 59_999 - row.timestamp_ms) > 3_600_000:
                    selected_time = candidate
                    break
            if selected_time is None:
                continue
            outputs.append(
                {
                    "pattern": row.pattern,
                    "side": row.side,
                    "symbol": row.symbol,
                    "timestamp_ms": selected_time + 59_999,
                    "regime_direction": row.regime_direction,
                    "regime_volatility": row.regime_volatility,
                }
            )
    return pd.DataFrame(outputs)


@dataclass(frozen=True)
class BatchSummary:
    events: int
    expectancy: float
    profit_factor: float
    win_rate: float
    mean_mae: float
    mean_mfe: float
    symbol_share_maximum: float


def summarize_batch(frame: pd.DataFrame) -> BatchSummary:
    values = frame["net_return_fraction"]
    gains = float(values.loc[values > 0].sum())
    losses = float(-values.loc[values < 0].sum())
    return BatchSummary(
        events=len(frame),
        expectancy=float(values.mean()),
        profit_factor=gains / losses if losses > 0 else math.inf if gains > 0 else 0.0,
        win_rate=float((values > 0).mean()),
        mean_mae=float(frame["mae_fraction"].mean()),
        mean_mfe=float(frame["mfe_fraction"].mean()),
        symbol_share_maximum=float(frame["symbol"].value_counts(normalize=True).max()),
    )


def summaries(frame: pd.DataFrame) -> Mapping[str, Mapping[str, object]]:
    result = {}
    for keys, group in frame.groupby(["partition", "pattern", "side"]):
        result[":".join(keys)] = asdict(summarize_batch(group))
    return result


def bootstrap_expectancy(
    frame: pd.DataFrame,
    *,
    seed: int = 180101,
    repetitions: int = 1_000,
) -> Mapping[str, float]:
    """Bootstrap complete UTC days to preserve intraday dependence."""
    if frame.empty:
        raise FastTrackContractError("AEGIS_M1A_BATCH_BOOTSTRAP_EMPTY")
    values = frame.copy()
    values["utc_day"] = pd.to_datetime(
        values["timestamp_ms"], unit="ms", utc=True
    ).dt.floor("1D")
    daily = [
        group["net_return_fraction"].to_numpy()
        for _, group in values.groupby("utc_day")
    ]
    random = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    profit_factors = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = random.integers(0, len(daily), size=len(daily))
        sample = np.concatenate([daily[item] for item in sampled])
        estimates[index] = float(sample.mean())
        gains = float(sample[sample > 0].sum())
        losses = float(-sample[sample < 0].sum())
        profit_factors[index] = gains / losses if losses > 0 else math.inf
    return {
        "expectancy_lower_95": float(np.quantile(estimates, 0.025)),
        "expectancy_median": float(np.quantile(estimates, 0.5)),
        "expectancy_upper_95": float(np.quantile(estimates, 0.975)),
        "profit_factor_lower_95": float(
            np.quantile(profit_factors, 0.025, method="inverted_cdf")
        ),
        "profit_factor_median": float(
            np.quantile(profit_factors, 0.5, method="inverted_cdf")
        ),
        "profit_factor_upper_95": float(
            np.quantile(profit_factors, 0.975, method="inverted_cdf")
        ),
    }
