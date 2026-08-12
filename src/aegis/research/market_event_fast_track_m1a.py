"""Causal multi-timeframe regime and micro-pattern engine for M1A research."""

from __future__ import annotations

import csv
import math
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..domain import TradeSide
from ..utils import Sha256HashProvider


class FastTrackContractError(ValueError):
    pass


class DirectionAxis(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    RANGE = "RANGE"
    TRANSITION = "TRANSITION"


class VolatilityAxis(str, Enum):
    COMPRESSED = "COMPRESSED"
    NORMAL = "NORMAL"
    EXPANDING = "EXPANDING"
    EXTREME = "EXTREME"


class LiquidityAxis(str, Enum):
    THIN = "THIN"
    NORMAL = "NORMAL"
    DEEP = "DEEP"


class MicroPattern(str, Enum):
    TREND_PULLBACK_CONTINUATION = "TREND_PULLBACK_CONTINUATION"
    COMPRESSION_BREAKOUT = "COMPRESSION_BREAKOUT"
    LIQUIDITY_SWEEP_REJECTION = "LIQUIDITY_SWEEP_REJECTION"
    FLOW_PRICE_ABSORPTION_REVERSAL = "FLOW_PRICE_ABSORPTION_REVERSAL"
    EXHAUSTION_REVERSAL = "EXHAUSTION_REVERSAL"
    SPOT_FUTURES_DIVERGENCE_CONVERGENCE = "SPOT_FUTURES_DIVERGENCE_CONVERGENCE"
    MULTITIMEFRAME_RECLAIM = "MULTITIMEFRAME_RECLAIM"
    SESSION_FUNDING_DISLOCATION = "SESSION_FUNDING_DISLOCATION"


def normalize_timestamp_ms(raw: str | int) -> int:
    value = int(raw)
    if value >= 100_000_000_000_000:
        value //= 1000
    if not 1_000_000_000_000 <= value < 10_000_000_000_000:
        raise FastTrackContractError("AEGIS_M1A_TIMESTAMP_INVALID")
    return value


@dataclass(frozen=True)
class MinuteBar:
    symbol: str
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    base_volume: float
    quote_volume: float
    trade_count: int
    taker_buy_quote: float
    interval_minutes: int = 1

    def __post_init__(self) -> None:
        prices = (self.open, self.high, self.low, self.close)
        values = (*prices, self.base_volume, self.quote_volume, self.taker_buy_quote)
        if (
            not self.symbol
            or self.open_time_ms <= 0
            or self.interval_minutes <= 0
            or self.trade_count < 0
            or not all(math.isfinite(value) for value in values)
            or min(prices) <= 0.0
            or self.low > min(self.open, self.close)
            or self.high < max(self.open, self.close)
            or min(self.base_volume, self.quote_volume, self.taker_buy_quote) < 0.0
            or self.taker_buy_quote > self.quote_volume + 1e-9
        ):
            raise FastTrackContractError("AEGIS_M1A_BAR_INVALID")

    @property
    def close_time_ms(self) -> int:
        return self.open_time_ms + self.interval_minutes * 60_000 - 1


@dataclass(frozen=True)
class FlowBucket:
    symbol: str
    open_time_ms: int
    aggressive_buy_quote: float
    aggressive_sell_quote: float
    trade_count: int

    @property
    def imbalance(self) -> float:
        total = self.aggressive_buy_quote + self.aggressive_sell_quote
        return (
            (self.aggressive_buy_quote - self.aggressive_sell_quote) / total
            if total > 0.0
            else 0.0
        )


def _csv_rows(path: Path) -> Iterable[list[str]]:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise FastTrackContractError("AEGIS_M1A_ARCHIVE_MEMBER_COUNT_INVALID")
        with archive.open(members[0]) as binary:
            text = (line.decode("utf-8") for line in binary)
            yield from csv.reader(text)


def read_kline_archive(path: Path, symbol: str) -> tuple[MinuteBar, ...]:
    rows = []
    for raw in _csv_rows(path):
        if not raw:
            continue
        try:
            timestamp = normalize_timestamp_ms(raw[0])
        except (ValueError, FastTrackContractError):
            if raw[0].lower().replace("_", " ").startswith("open time"):
                continue
            raise
        if len(raw) < 11:
            raise FastTrackContractError("AEGIS_M1A_KLINE_ROW_INVALID")
        rows.append(
            MinuteBar(
                symbol=symbol,
                open_time_ms=timestamp,
                open=float(raw[1]),
                high=float(raw[2]),
                low=float(raw[3]),
                close=float(raw[4]),
                base_volume=float(raw[5]),
                quote_volume=float(raw[7]),
                trade_count=int(raw[8]),
                taker_buy_quote=float(raw[10]),
            )
        )
    ordered = tuple(sorted(rows, key=lambda item: item.open_time_ms))
    if len({row.open_time_ms for row in ordered}) != len(ordered):
        raise FastTrackContractError("AEGIS_M1A_DUPLICATE_KLINE")
    return ordered


def read_agg_trade_archive(path: Path, symbol: str) -> tuple[FlowBucket, ...]:
    buckets: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for raw in _csv_rows(path):
        if not raw:
            continue
        try:
            timestamp = normalize_timestamp_ms(raw[5])
        except (ValueError, FastTrackContractError):
            if len(raw) > 5 and "time" in raw[5].lower():
                continue
            raise
        if len(raw) < 7:
            raise FastTrackContractError("AEGIS_M1A_AGG_TRADE_ROW_INVALID")
        price, quantity = float(raw[1]), float(raw[2])
        if not all(math.isfinite(value) and value > 0.0 for value in (price, quantity)):
            raise FastTrackContractError("AEGIS_M1A_AGG_TRADE_VALUE_INVALID")
        buyer_maker = raw[6].strip().lower()
        if buyer_maker not in {"true", "false"}:
            raise FastTrackContractError("AEGIS_M1A_AGGRESSOR_FLAG_INVALID")
        bucket = timestamp // 60_000 * 60_000
        notional = price * quantity
        if buyer_maker == "false":
            buckets[bucket][0] += notional
        else:
            buckets[bucket][1] += notional
        buckets[bucket][2] += 1
    return tuple(
        FlowBucket(symbol, timestamp, values[0], values[1], int(values[2]))
        for timestamp, values in sorted(buckets.items())
    )


def resample_closed_bars(
    rows: Sequence[MinuteBar], interval_minutes: int
) -> tuple[MinuteBar, ...]:
    if interval_minutes <= 0:
        raise FastTrackContractError("AEGIS_M1A_RESAMPLE_INTERVAL_INVALID")
    groups: dict[int, list[MinuteBar]] = defaultdict(list)
    interval_ms = interval_minutes * 60_000
    for row in rows:
        if row.interval_minutes != 1:
            raise FastTrackContractError("AEGIS_M1A_RESAMPLE_REQUIRES_1M")
        groups[row.open_time_ms // interval_ms * interval_ms].append(row)
    result = []
    for timestamp, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda item: item.open_time_ms)
        expected = tuple(timestamp + offset * 60_000 for offset in range(interval_minutes))
        if tuple(row.open_time_ms for row in ordered) != expected:
            continue
        result.append(
            MinuteBar(
                symbol=ordered[0].symbol,
                open_time_ms=timestamp,
                open=ordered[0].open,
                high=max(row.high for row in ordered),
                low=min(row.low for row in ordered),
                close=ordered[-1].close,
                base_volume=sum(row.base_volume for row in ordered),
                quote_volume=sum(row.quote_volume for row in ordered),
                trade_count=sum(row.trade_count for row in ordered),
                taker_buy_quote=sum(row.taker_buy_quote for row in ordered),
                interval_minutes=interval_minutes,
            )
        )
    return tuple(result)


def _return(rows: Sequence[MinuteBar], bars: int) -> float:
    if len(rows) <= bars:
        raise FastTrackContractError("AEGIS_M1A_HISTORY_INSUFFICIENT")
    return rows[-1].close / rows[-1 - bars].close - 1.0


def _ema(values: Sequence[float], period: int) -> float:
    if len(values) < period:
        raise FastTrackContractError("AEGIS_M1A_EMA_HISTORY_INSUFFICIENT")
    alpha = 2.0 / (period + 1.0)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


@dataclass(frozen=True)
class RegimeThresholds:
    direction_enter: float
    direction_exit: float
    compressed_volatility: float
    expanding_volatility: float
    extreme_volatility: float
    thin_liquidity: float
    deep_liquidity: float

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise FastTrackContractError("AEGIS_M1A_REGIME_THRESHOLD_INVALID")
        if self.direction_exit >= self.direction_enter:
            raise FastTrackContractError("AEGIS_M1A_REGIME_HYSTERESIS_INVALID")
        if not self.compressed_volatility < self.expanding_volatility < self.extreme_volatility:
            raise FastTrackContractError("AEGIS_M1A_VOLATILITY_BOUNDARIES_INVALID")
        if self.thin_liquidity >= self.deep_liquidity:
            raise FastTrackContractError("AEGIS_M1A_LIQUIDITY_BOUNDARIES_INVALID")


@dataclass(frozen=True)
class RegimeObservation:
    timestamp_ms: int
    direction_score: float
    realized_volatility: float
    median_quote_volume: float
    direction: DirectionAxis
    volatility: VolatilityAxis
    liquidity: LiquidityAxis
    content_hash: str


class CausalRegimeClassifier:
    def __init__(self, thresholds: RegimeThresholds, minimum_state_bars: int = 3) -> None:
        if minimum_state_bars < 1:
            raise FastTrackContractError("AEGIS_M1A_REGIME_DURATION_INVALID")
        self.thresholds = thresholds
        self.minimum_state_bars = minimum_state_bars
        self._current = DirectionAxis.TRANSITION
        self._pending = DirectionAxis.TRANSITION
        self._pending_count = 0
        self._last_timestamp = 0

    def observe(self, hourly: Sequence[MinuteBar], four_hourly: Sequence[MinuteBar]) -> RegimeObservation:
        if len(hourly) < 25 or len(four_hourly) < 7:
            raise FastTrackContractError("AEGIS_M1A_REGIME_HISTORY_INSUFFICIENT")
        timestamp = min(hourly[-1].close_time_ms, four_hourly[-1].close_time_ms)
        if timestamp <= self._last_timestamp:
            raise FastTrackContractError("AEGIS_M1A_REGIME_NON_CHRONOLOGICAL")
        self._last_timestamp = timestamp
        direction_score, volatility, liquidity = _regime_raw_metrics(hourly, four_hourly)
        boundary = (
            self.thresholds.direction_exit
            if self._current in {DirectionAxis.BULL, DirectionAxis.BEAR}
            else self.thresholds.direction_enter
        )
        candidate = (
            DirectionAxis.BULL
            if direction_score >= boundary
            else DirectionAxis.BEAR
            if direction_score <= -boundary
            else DirectionAxis.RANGE
        )
        if candidate == self._current:
            self._pending, self._pending_count = candidate, 0
        elif candidate == self._pending:
            self._pending_count += 1
            if self._pending_count >= self.minimum_state_bars:
                self._current, self._pending_count = candidate, 0
        else:
            self._pending, self._pending_count = candidate, 1
            if self._pending_count >= self.minimum_state_bars:
                self._current, self._pending_count = candidate, 0
        volatility_axis = (
            VolatilityAxis.COMPRESSED
            if volatility <= self.thresholds.compressed_volatility
            else VolatilityAxis.EXTREME
            if volatility >= self.thresholds.extreme_volatility
            else VolatilityAxis.EXPANDING
            if volatility >= self.thresholds.expanding_volatility
            else VolatilityAxis.NORMAL
        )
        liquidity_axis = (
            LiquidityAxis.THIN
            if liquidity <= self.thresholds.thin_liquidity
            else LiquidityAxis.DEEP
            if liquidity >= self.thresholds.deep_liquidity
            else LiquidityAxis.NORMAL
        )
        payload = {
            "timestamp_ms": timestamp,
            "direction_score": direction_score,
            "realized_volatility": volatility,
            "median_quote_volume": liquidity,
            "direction": self._current.value,
            "volatility": volatility_axis.value,
            "liquidity": liquidity_axis.value,
        }
        return RegimeObservation(
            timestamp_ms=timestamp,
            direction_score=direction_score,
            realized_volatility=volatility,
            median_quote_volume=liquidity,
            direction=self._current,
            volatility=volatility_axis,
            liquidity=liquidity_axis,
            content_hash=Sha256HashProvider().digest_value(payload),
        )


def _regime_raw_metrics(
    hourly: Sequence[MinuteBar], four_hourly: Sequence[MinuteBar]
) -> tuple[float, float, float]:
    if len(hourly) < 25 or len(four_hourly) < 7:
        raise FastTrackContractError("AEGIS_M1A_REGIME_HISTORY_INSUFFICIENT")
    ret_1h = _return(hourly, 1)
    ret_4h = _return(four_hourly, 1)
    closes = [row.close for row in hourly[-25:]]
    direction_score = (
        0.35 * ret_1h
        + 0.45 * ret_4h
        + 0.20 * (_ema(closes, 8) / _ema(closes, 24) - 1.0)
    )
    returns = [
        hourly[index].close / hourly[index - 1].close - 1.0
        for index in range(len(hourly) - 23, len(hourly))
    ]
    volatility = math.sqrt(sum(value * value for value in returns) / len(returns))
    liquidity = sorted(row.quote_volume for row in hourly[-24:])[12]
    return direction_score, volatility, liquidity


def fit_regime_thresholds_from_train(
    samples: Sequence[tuple[Sequence[MinuteBar], Sequence[MinuteBar]]],
) -> RegimeThresholds:
    """Fit global regime boundaries from chronological TRAIN histories."""

    if len(samples) < 100:
        raise FastTrackContractError("AEGIS_M1A_REGIME_TRAIN_INSUFFICIENT")
    metrics = [_regime_raw_metrics(hourly, four_hourly) for hourly, four_hourly in samples]
    direction = [abs(item[0]) for item in metrics]
    volatility = [item[1] for item in metrics]
    liquidity = [item[2] for item in metrics]
    enter = _quantile(direction, 0.70)
    return RegimeThresholds(
        direction_enter=enter,
        direction_exit=enter * 0.60,
        compressed_volatility=_quantile(volatility, 0.20),
        expanding_volatility=_quantile(volatility, 0.70),
        extreme_volatility=_quantile(volatility, 0.95),
        thin_liquidity=_quantile(liquidity, 0.20),
        deep_liquidity=_quantile(liquidity, 0.80),
    )


@dataclass(frozen=True)
class PatternThresholds:
    minimum_flow_imbalance: float
    minimum_volume_ratio: float
    maximum_compression_ratio: float
    minimum_breakout_fraction: float
    minimum_wick_body_ratio: float
    minimum_extension_fraction: float
    maximum_absorption_response: float
    minimum_basis_divergence: float
    minimum_reclaim_fraction: float
    minimum_session_move: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) and value >= 0.0 for value in self.__dict__.values()):
            raise FastTrackContractError("AEGIS_M1A_PATTERN_THRESHOLD_INVALID")


@dataclass(frozen=True)
class PatternCandidate:
    pattern: MicroPattern
    side: TradeSide
    symbol: str
    timestamp_ms: int
    regime_direction: DirectionAxis
    regime_volatility: VolatilityAxis
    feature_hash: str
    reason_codes: tuple[str, ...]


def extract_pattern_features(
    *,
    futures: Sequence[MinuteBar],
    spot: Sequence[MinuteBar],
    flow: Sequence[FlowBucket],
    funding_rate: float | None,
) -> Mapping[str, float | int | None]:
    """Build the complete M1A feature snapshot at one closed minute."""

    if len(futures) < 241 or len(spot) < 61 or len(flow) < 13:
        raise FastTrackContractError("AEGIS_M1A_PATTERN_HISTORY_INSUFFICIENT")
    recent_futures = futures[-241:]
    if any(
        current.open_time_ms - previous.open_time_ms != 60_000
        for previous, current in zip(recent_futures, recent_futures[1:])
    ):
        raise FastTrackContractError("AEGIS_M1A_PATTERN_HISTORY_GAP")
    timestamp = futures[-1].close_time_ms
    if spot[-1].close_time_ms > timestamp or flow[-1].open_time_ms > futures[-1].open_time_ms:
        raise FastTrackContractError("AEGIS_M1A_PATTERN_CAUSALITY_VIOLATION")
    if futures[-1].open_time_ms != spot[-1].open_time_ms or futures[-1].open_time_ms != flow[-1].open_time_ms:
        raise FastTrackContractError("AEGIS_M1A_PATTERN_CLOCK_MISMATCH")
    current = futures[-1]
    body = abs(current.close - current.open) / current.open
    upper_wick = (current.high - max(current.open, current.close)) / current.open
    lower_wick = (min(current.open, current.close) - current.low) / current.open
    ret_1 = _return(futures, 1)
    ret_3 = _return(futures, 3)
    ret_12 = _return(futures, 12)
    ret_60 = _return(futures, 60)
    ret_240 = _return(futures, 240)
    flow_1 = flow[-1].imbalance
    flow_3 = sum(item.imbalance for item in flow[-3:]) / 3.0
    flow_12 = sum(item.imbalance for item in flow[-12:]) / 12.0
    volume_ratio = current.quote_volume / max(
        sum(row.quote_volume for row in futures[-25:-1]) / 24.0, 1e-12
    )
    short_range = max(row.high for row in futures[-6:]) - min(row.low for row in futures[-6:])
    long_range = max(row.high for row in futures[-24:]) - min(row.low for row in futures[-24:])
    compression = short_range / max(long_range, 1e-12)
    prior_high = max(row.high for row in futures[-21:-1])
    prior_low = min(row.low for row in futures[-21:-1])
    basis = futures[-1].close / spot[-1].close - 1.0
    prior_basis = futures[-13].close / spot[-13].close - 1.0
    return {
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_12": ret_12,
        "ret_60": ret_60,
        "ret_240": ret_240,
        "flow_1": flow_1,
        "flow_3": flow_3,
        "flow_12": flow_12,
        "volume_ratio": volume_ratio,
        "compression": compression,
        "breakout_up": current.close / prior_high - 1.0,
        "breakout_down": prior_low / current.close - 1.0,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "body": body,
        "spot_ret_12": _return(spot, 12),
        "basis": basis,
        "prior_basis": prior_basis,
        "basis_convergence": abs(prior_basis) - abs(basis),
        "price_response": abs(ret_3),
        "funding_rate": funding_rate,
        "hour_utc": (timestamp // 3_600_000) % 24,
        "weekday_utc": (timestamp // 86_400_000 + 3) % 7,
        "prior_high": prior_high,
        "prior_low": prior_low,
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise FastTrackContractError("AEGIS_M1A_QUANTILE_INPUT_INVALID")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def fit_pattern_thresholds_from_train(
    rows: Sequence[Mapping[str, float | int | None]],
) -> PatternThresholds:
    """Fit the preregistered first-pass quantiles from TRAIN snapshots only."""

    required = {
        "flow_3", "volume_ratio", "compression", "breakout_up", "breakout_down",
        "upper_wick", "lower_wick", "body", "ret_60", "price_response",
        "prior_basis", "basis_convergence", "ret_3",
    }
    if len(rows) < 1000 or any(not required <= set(row) for row in rows):
        raise FastTrackContractError("AEGIS_M1A_TRAIN_ROWS_INSUFFICIENT")

    def finite(name: str, transform=abs) -> list[float]:
        values = []
        for row in rows:
            raw = row[name]
            if raw is None or not math.isfinite(float(raw)):
                raise FastTrackContractError("AEGIS_M1A_TRAIN_VALUE_INVALID")
            values.append(transform(float(raw)))
        return values

    wick_ratios = [
        max(float(row["upper_wick"]), float(row["lower_wick"]))
        / max(float(row["body"]), 1e-12)
        for row in rows
    ]
    breakout = [
        max(float(row["breakout_up"]), float(row["breakout_down"]), 0.0)
        for row in rows
    ]
    return PatternThresholds(
        minimum_flow_imbalance=_quantile(finite("flow_3"), 0.90),
        minimum_volume_ratio=_quantile(finite("volume_ratio", lambda value: value), 0.80),
        maximum_compression_ratio=_quantile(finite("compression", lambda value: value), 0.20),
        minimum_breakout_fraction=_quantile(breakout, 0.90),
        minimum_wick_body_ratio=_quantile(wick_ratios, 0.80),
        minimum_extension_fraction=_quantile(finite("ret_60"), 0.90),
        maximum_absorption_response=_quantile(finite("price_response"), 0.20),
        minimum_basis_divergence=_quantile(finite("prior_basis"), 0.90),
        minimum_reclaim_fraction=_quantile(finite("ret_3"), 0.60),
        minimum_session_move=_quantile(finite("ret_60"), 0.85),
    )


def detect_micro_patterns(
    *,
    symbol: str,
    futures: Sequence[MinuteBar],
    spot: Sequence[MinuteBar],
    flow: Sequence[FlowBucket],
    regime: RegimeObservation,
    thresholds: PatternThresholds,
    funding_rate: float | None,
) -> tuple[PatternCandidate, ...]:
    """Detect frozen event families from data available at the current close."""

    features = extract_pattern_features(
        futures=futures, spot=spot, flow=flow, funding_rate=funding_rate
    )
    timestamp = futures[-1].close_time_ms
    current = futures[-1]
    feature_hash = Sha256HashProvider().digest_value(features)
    ret_3, ret_12, ret_60, ret_240 = (float(features[name]) for name in ("ret_3", "ret_12", "ret_60", "ret_240"))
    flow_1, flow_3 = float(features["flow_1"]), float(features["flow_3"])
    volume_ratio, compression = float(features["volume_ratio"]), float(features["compression"])
    breakout_up, breakout_down = float(features["breakout_up"]), float(features["breakout_down"])
    upper_wick, lower_wick, body = float(features["upper_wick"]), float(features["lower_wick"]), float(features["body"])
    prior_high, prior_low = float(features["prior_high"]), float(features["prior_low"])
    prior_basis, basis_convergence = float(features["prior_basis"]), float(features["basis_convergence"])
    price_response = float(features["price_response"])
    candidates: list[PatternCandidate] = []

    def emit(pattern: MicroPattern, side: TradeSide, *reasons: str) -> None:
        candidates.append(
            PatternCandidate(
                pattern,
                side,
                symbol,
                timestamp,
                regime.direction,
                regime.volatility,
                feature_hash,
                tuple(reasons),
            )
        )

    aligned_side = TradeSide.LONG if regime.direction is DirectionAxis.BULL else TradeSide.SHORT
    aligned_sign = 1.0 if aligned_side is TradeSide.LONG else -1.0
    if regime.direction in {DirectionAxis.BULL, DirectionAxis.BEAR} and (
        aligned_sign * ret_60 > 0
        and aligned_sign * ret_3 > thresholds.minimum_reclaim_fraction
        and aligned_sign * flow_3 >= thresholds.minimum_flow_imbalance
    ):
        emit(MicroPattern.TREND_PULLBACK_CONTINUATION, aligned_side, "DIRECTIONAL_REGIME", "FLOW_ALIGNED_RECLAIM")

    if compression <= thresholds.maximum_compression_ratio and volume_ratio >= thresholds.minimum_volume_ratio:
        if breakout_up >= thresholds.minimum_breakout_fraction and flow_1 >= thresholds.minimum_flow_imbalance:
            emit(MicroPattern.COMPRESSION_BREAKOUT, TradeSide.LONG, "COMPRESSION", "UP_BREAK_FLOW")
        if breakout_down >= thresholds.minimum_breakout_fraction and flow_1 <= -thresholds.minimum_flow_imbalance:
            emit(MicroPattern.COMPRESSION_BREAKOUT, TradeSide.SHORT, "COMPRESSION", "DOWN_BREAK_FLOW")

    wick_denominator = max(body, 1e-12)
    if current.low < prior_low and lower_wick / wick_denominator >= thresholds.minimum_wick_body_ratio and current.close > prior_low:
        emit(MicroPattern.LIQUIDITY_SWEEP_REJECTION, TradeSide.LONG, "LOW_SWEEP", "CLOSE_RECLAIM")
    if current.high > prior_high and upper_wick / wick_denominator >= thresholds.minimum_wick_body_ratio and current.close < prior_high:
        emit(MicroPattern.LIQUIDITY_SWEEP_REJECTION, TradeSide.SHORT, "HIGH_SWEEP", "CLOSE_REJECT")

    if abs(flow_3) >= thresholds.minimum_flow_imbalance and price_response <= thresholds.maximum_absorption_response and flow_1 * flow_3 < 0:
        emit(
            MicroPattern.FLOW_PRICE_ABSORPTION_REVERSAL,
            TradeSide.SHORT if flow_3 > 0 else TradeSide.LONG,
            "EXTREME_FLOW",
            "LOW_PRICE_RESPONSE",
            "FLOW_REVERSAL",
        )

    if abs(ret_60) >= thresholds.minimum_extension_fraction and abs(ret_3) < abs(ret_12) / 4.0 and flow_3 * ret_60 < 0:
        emit(
            MicroPattern.EXHAUSTION_REVERSAL,
            TradeSide.SHORT if ret_60 > 0 else TradeSide.LONG,
            "EXTENDED_MOVE",
            "DECELERATION",
            "FLOW_DIVERGENCE",
        )

    if abs(prior_basis) >= thresholds.minimum_basis_divergence and basis_convergence >= thresholds.minimum_reclaim_fraction:
        side = TradeSide.SHORT if prior_basis > 0 else TradeSide.LONG
        emit(MicroPattern.SPOT_FUTURES_DIVERGENCE_CONVERGENCE, side, "BASIS_EXTREME", "CONVERGENCE")

    if regime.direction in {DirectionAxis.BULL, DirectionAxis.BEAR} and aligned_sign * ret_240 > 0 and aligned_sign * ret_60 < 0 and aligned_sign * ret_12 >= thresholds.minimum_reclaim_fraction:
        emit(MicroPattern.MULTITIMEFRAME_RECLAIM, aligned_side, "4H_DIRECTION", "1H_PULLBACK", "RECLAIM")

    if funding_rate is not None and math.isfinite(funding_rate) and abs(ret_60) >= thresholds.minimum_session_move and abs(flow_3) >= thresholds.minimum_flow_imbalance:
        side = TradeSide.LONG if ret_60 > 0 and flow_3 > 0 else TradeSide.SHORT if ret_60 < 0 and flow_3 < 0 else TradeSide.NO_TRADE
        if side is not TradeSide.NO_TRADE:
            emit(MicroPattern.SESSION_FUNDING_DISLOCATION, side, "SESSION_MOVE", "FUNDING_PRESENT", "FLOW_CONFIRMED")

    return tuple(candidates)
