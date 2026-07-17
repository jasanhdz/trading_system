"""Causal market validation and the shared training/inference feature pipeline."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Mapping, Protocol, Sequence

from .config import CANONICAL_SYMBOLS, UniverseConfig
from .domain import (
    Candle,
    FeatureBatch,
    FeatureQuality,
    FeatureRow,
    MarketSnapshot,
    ValidationStatus,
)
from .utils import ordered_name_hash


FEATURE_SCHEMA_VERSION = "aegis-features-v1"
FEATURE_NAMES = (
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "log_ret_1",
    "close_to_open_return", "candle_range_fraction", "candle_body_fraction",
    "upper_wick_fraction", "lower_wick_fraction", "body_to_range", "close_position_in_range",
    "volume_return_1", "volume_zscore_24", "volume_ratio_6_24",
    "range_mean_6", "range_mean_24", "atr_12", "atr_24", "volatility_ratio_6_24",
    "ema_gap_6_12", "ema_gap_12_24", "ema_slope_12",
    "momentum_acceleration_3_12", "return_zscore_24", "persistence_6",
    "chop_12", "trend_strength_12", "range_expansion",
    "relative_return_6", "relative_return_12", "cross_rank_return_6",
    "cross_dispersion_return_6", "market_breadth_6", "market_direction_6",
    "market_concentration_6", "btc_divergence_6", "eth_divergence_6",
)
FEATURE_HASH = ordered_name_hash(FEATURE_NAMES)


class SnapshotValidationError(ValueError):
    def __init__(self, status: ValidationStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class FeaturePipeline(Protocol):
    schema_version: str
    feature_names: tuple[str, ...]
    feature_hash: str

    def transform(self, snapshot: MarketSnapshot) -> FeatureBatch: ...


@dataclass(frozen=True)
class FrozenNormalizer:
    """Normalizer parameters published with a model bundle."""

    means: Mapping[str, float] = field(default_factory=dict)
    scales: Mapping[str, float] = field(default_factory=dict)
    clip_absolute: float = 12.0

    def normalize(self, name: str, value: float) -> tuple[float, bool]:
        mean = float(self.means.get(name, 0.0))
        scale = float(self.scales.get(name, 1.0))
        if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid frozen normalizer for {name}")
        result = (value - mean) / scale
        clipped = abs(result) > self.clip_absolute
        result = max(-self.clip_absolute, min(self.clip_absolute, result))
        return result, clipped


@dataclass(frozen=True)
class MarketSnapshotValidator:
    universe: UniverseConfig

    def validate(self, snapshot: MarketSnapshot, now: datetime) -> None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise SnapshotValidationError(ValidationStatus.ERROR_CONTRACT, "validation clock must be timezone-aware")
        if snapshot.timeframe != self.universe.timeframe:
            raise SnapshotValidationError(ValidationStatus.ERROR_CONTRACT, "snapshot timeframe mismatch")
        if snapshot.symbol_set_hash != self.universe.symbol_set_hash:
            raise SnapshotValidationError(ValidationStatus.NO_TRADE_UNIVERSE_MISMATCH, "snapshot universe hash mismatch")
        symbols = tuple(series.symbol for series in snapshot.series)
        if len(symbols) != len(set(symbols)):
            raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, "duplicate symbol series")
        if set(symbols) != set(self.universe.symbols):
            raise SnapshotValidationError(ValidationStatus.NO_TRADE_UNIVERSE_MISMATCH, "snapshot symbols mismatch")
        age = (now - snapshot.closed_at).total_seconds()
        if age < 0:
            raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, "snapshot is in the future")
        if age > self.universe.maximum_snapshot_age_seconds:
            raise SnapshotValidationError(ValidationStatus.NO_TRADE_DATA_STALE, "snapshot is stale")

        interval = _timeframe_delta(snapshot.timeframe)
        for series in snapshot.series:
            if len(series.candles) < self.universe.minimum_history_bars:
                raise SnapshotValidationError(ValidationStatus.NO_TRADE_DATA_INSUFFICIENT, f"insufficient history for {series.symbol}")
            if series.feed_quality.duplicate_bars:
                raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, f"feed reports duplicate bars for {series.symbol}")
            if series.feed_quality.missing_bars > self.universe.maximum_gap_bars:
                raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, f"feed reports gaps for {series.symbol}")
            previous: Candle | None = None
            for candle in series.candles:
                if not candle.is_closed:
                    raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, f"partial candle for {series.symbol}")
                if candle.close_time > snapshot.closed_at or candle.close_time > now:
                    raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, f"future candle for {series.symbol}")
                if candle.high < max(candle.open, candle.close) or candle.low > min(candle.open, candle.close) or candle.high < candle.low:
                    raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, f"incoherent OHLC for {series.symbol}")
                if previous is not None:
                    if candle.open_time <= previous.open_time:
                        raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, f"unordered candles for {series.symbol}")
                    gap = candle.open_time - previous.open_time
                    if gap != interval:
                        raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, f"candle gap for {series.symbol}")
                previous = candle
            if series.candles[-1].close_time != series.last_confirmed_close:
                raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, f"last close mismatch for {series.symbol}")
            if series.last_confirmed_close != snapshot.closed_at:
                raise SnapshotValidationError(ValidationStatus.ERROR_DATA_INVALID, f"uncoordinated close for {series.symbol}")


def _timeframe_delta(timeframe: str) -> timedelta:
    if timeframe != "5m":
        raise SnapshotValidationError(ValidationStatus.ERROR_CONTRACT, f"unsupported timeframe: {timeframe}")
    return timedelta(minutes=5)


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-15:
        return 0.0
    value = numerator / denominator
    return value if math.isfinite(value) else 0.0


def _return(values: Sequence[float], bars: int) -> float:
    return _safe_div(values[-1], values[-1 - bars]) - 1.0


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _ema(values: Sequence[float], span: int) -> float:
    alpha = 2.0 / (span + 1.0)
    result = float(values[0])
    for value in values[1:]:
        result = alpha * float(value) + (1.0 - alpha) * result
    return result


def _rank_fraction(value: float, values: Sequence[float]) -> float:
    lower = sum(candidate < value for candidate in values)
    equal = sum(candidate == value for candidate in values)
    average_rank = lower + (equal - 1) / 2.0
    return average_rank / max(1, len(values) - 1)


@dataclass(frozen=True)
class DeterministicFeaturePipeline:
    """One causal implementation used by both training and inference."""

    normalizer: FrozenNormalizer = field(default_factory=FrozenNormalizer)
    schema_version: str = FEATURE_SCHEMA_VERSION
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_hash: str = FEATURE_HASH

    def transform(self, snapshot: MarketSnapshot) -> FeatureBatch:
        by_symbol = {series.symbol: series for series in snapshot.series}
        if set(by_symbol) != set(CANONICAL_SYMBOLS):
            raise ValueError("feature pipeline requires the canonical eleven-symbol universe")
        local = {symbol: self._local_features(by_symbol[symbol].candles) for symbol in CANONICAL_SYMBOLS}
        returns_6 = {symbol: values["ret_6"] for symbol, values in local.items()}
        returns_12 = {symbol: values["ret_12"] for symbol, values in local.items()}
        market_6 = tuple(returns_6.values())
        dispersion = _std(market_6)
        breadth = _mean([1.0 if value > 0 else 0.0 for value in market_6])
        direction = _mean(market_6)
        concentration = max((abs(value) for value in market_6), default=0.0) / max(math.fsum(abs(value) for value in market_6), 1e-15)
        btc = returns_6.get("BTCUSDT", direction)
        eth = returns_6.get("ETHUSDT", direction)

        rows: list[FeatureRow] = []
        for symbol in CANONICAL_SYMBOLS:
            values = local[symbol]
            values.update({
                "relative_return_6": returns_6[symbol] - direction,
                "relative_return_12": returns_12[symbol] - _mean(tuple(returns_12.values())),
                "cross_rank_return_6": _rank_fraction(returns_6[symbol], market_6),
                "cross_dispersion_return_6": dispersion,
                "market_breadth_6": breadth,
                "market_direction_6": direction,
                "market_concentration_6": concentration,
                "btc_divergence_6": returns_6[symbol] - btc,
                "eth_divergence_6": returns_6[symbol] - eth,
            })
            raw = tuple(float(values[name]) for name in self.feature_names)
            if not all(math.isfinite(value) for value in raw):
                raise ValueError(f"non-finite raw feature for {symbol}")
            normalized_values: list[float] = []
            clipped = 0
            for name, value in zip(self.feature_names, raw):
                normalized, was_clipped = self.normalizer.normalize(name, value)
                normalized_values.append(normalized)
                clipped += int(was_clipped)
            rows.append(FeatureRow(
                symbol=symbol,
                raw_values=raw,
                normalized_values=tuple(normalized_values),
                quality=FeatureQuality(missing_values=0, clipped_values=clipped, finite=True, history_rows=len(by_symbol[symbol].candles)),
            ))
        return FeatureBatch(self.schema_version, self.feature_names, self.feature_hash, tuple(rows))

    def _local_features(self, candles: Sequence[Candle]) -> dict[str, float]:
        if len(candles) < 48:
            raise ValueError("at least 48 closed candles are required")
        closes = [c.close for c in candles]
        opens = [c.open for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        ranges = [_safe_div(high - low, close) for high, low, close in zip(highs, lows, closes)]
        true_ranges = []
        for index, candle in enumerate(candles):
            previous_close = closes[index - 1] if index else candle.open
            true_ranges.append(max(candle.high - candle.low, abs(candle.high - previous_close), abs(candle.low - previous_close)) / previous_close)
        returns = [_safe_div(closes[index], closes[index - 1]) - 1.0 for index in range(1, len(closes))]
        current = candles[-1]
        current_range = current.high - current.low
        ema6, ema12, ema24 = (_ema(closes[-48:], span) for span in (6, 12, 24))
        previous_ema12 = _ema(closes[-25:-1], 12)
        volume_std = _std(volumes[-24:])
        ret_std = _std(returns[-24:])
        absolute_path = math.fsum(abs(value) for value in returns[-12:])
        return {
            "ret_1": _return(closes, 1), "ret_3": _return(closes, 3), "ret_6": _return(closes, 6),
            "ret_12": _return(closes, 12), "ret_24": _return(closes, 24),
            "log_ret_1": math.log(closes[-1] / closes[-2]),
            "close_to_open_return": _safe_div(current.close, current.open) - 1.0,
            "candle_range_fraction": _safe_div(current_range, current.close),
            "candle_body_fraction": _safe_div(abs(current.close - current.open), current.close),
            "upper_wick_fraction": _safe_div(current.high - max(current.open, current.close), current_range),
            "lower_wick_fraction": _safe_div(min(current.open, current.close) - current.low, current_range),
            "body_to_range": _safe_div(abs(current.close - current.open), current_range),
            "close_position_in_range": _safe_div(current.close - current.low, current_range),
            "volume_return_1": _safe_div(volumes[-1], volumes[-2]) - 1.0,
            "volume_zscore_24": _safe_div(volumes[-1] - _mean(volumes[-24:]), volume_std),
            "volume_ratio_6_24": _safe_div(_mean(volumes[-6:]), _mean(volumes[-24:])),
            "range_mean_6": _mean(ranges[-6:]), "range_mean_24": _mean(ranges[-24:]),
            "atr_12": _mean(true_ranges[-12:]), "atr_24": _mean(true_ranges[-24:]),
            "volatility_ratio_6_24": _safe_div(_std(returns[-6:]), ret_std),
            "ema_gap_6_12": _safe_div(ema6, ema12) - 1.0, "ema_gap_12_24": _safe_div(ema12, ema24) - 1.0,
            "ema_slope_12": _safe_div(ema12, previous_ema12) - 1.0,
            "momentum_acceleration_3_12": _return(closes, 3) - _return(closes, 12) / 4.0,
            "return_zscore_24": _safe_div(returns[-1] - _mean(returns[-24:]), ret_std),
            "persistence_6": _mean([1.0 if value > 0 else -1.0 if value < 0 else 0.0 for value in returns[-6:]]),
            "chop_12": 1.0 - min(1.0, _safe_div(abs(closes[-1] - closes[-13]), absolute_path)),
            "trend_strength_12": _safe_div(abs(_return(closes, 12)), _mean(true_ranges[-12:])),
            "range_expansion": _safe_div(_mean(ranges[-6:]), _mean(ranges[-24:])) - 1.0,
        }
