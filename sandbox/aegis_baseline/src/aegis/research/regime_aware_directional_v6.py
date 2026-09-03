"""Causal labels and lifecycle replay for regime-aware LONG/SHORT research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from ..data import CanonicalBar
from ..domain import Candle
from ..training.hybrid_directional import DirectionalSide
from .hybrid_ts_protection_replay import (
    IntrabarPath,
    TsProtectionConfig,
    replay_ts_price_protection,
)
from .long_entry_v21_shadow import LONG_V21_FEATURE_NAMES


class RegimePhase(str, Enum):
    CONTINUATION = "CONTINUATION"
    PULLBACK = "PULLBACK"
    EXHAUSTION = "EXHAUSTION"
    TRANSITION = "TRANSITION"


class DirectionalRole(str, Enum):
    PRIMARY_TREND = "PRIMARY_TREND"
    TACTICAL_COUNTERTREND = "TACTICAL_COUNTERTREND"
    SELECTIVE = "SELECTIVE"


REGIME_DIRECTIONS = ("BULLISH", "NEUTRAL", "BEARISH")
REGIME_VOLATILITIES = ("LOW", "NORMAL", "HIGH")
REGIME_STRUCTURES = ("TREND", "RANGE", "TRANSITION")
REGIME_PHASES = tuple(value.value for value in RegimePhase)
DIRECTIONAL_INTERACTION_NAMES = (
    "side_ret_1",
    "side_ret_3",
    "side_ret_12",
    "side_market_direction_6",
    "side_market_breadth_6",
    "volume_ratio_6_24",
    "side_close_position_in_range",
    "favorable_wick_fraction",
    "adverse_wick_fraction",
    "side_trend_stack",
)
REGIME_AWARE_V6_FEATURE_NAMES = (
    *LONG_V21_FEATURE_NAMES,
    *DIRECTIONAL_INTERACTION_NAMES,
    *(f"regime_direction_{value}" for value in REGIME_DIRECTIONS),
    *(f"regime_volatility_{value}" for value in REGIME_VOLATILITIES),
    *(f"regime_structure_{value}" for value in REGIME_STRUCTURES),
    *(f"regime_phase_{value}" for value in REGIME_PHASES),
    *(f"directional_role_{value.value}" for value in DirectionalRole),
)
REGIME_ROUTER_FEATURE_NAMES = (
    "btc_ret_1",
    "btc_ret_3",
    "btc_ret_12",
    "btc_atr_12",
    "btc_volume_ratio_6_24",
    "btc_trend_strength_12",
    "cross_ret_1_mean",
    "cross_ret_3_mean",
    "cross_ret_12_mean",
    "cross_positive_breadth_1",
    "cross_positive_breadth_3",
    "cross_positive_breadth_12",
    "cross_atr_mean",
    "cross_volume_ratio_mean",
    "cross_high_vol_fraction",
    "cross_low_vol_fraction",
)


class RegimeAwareV6Error(ValueError):
    pass


@dataclass(frozen=True)
class DirectionalPathContract:
    leverage: float
    roe_checkpoints: tuple[float, ...]
    primary_protectable_roe: float
    favorable_atr_multiple: float
    adverse_atr_multiple: float
    favorable_floor_fraction: float
    adverse_floor_fraction: float
    favorable_ceiling_fraction: float
    adverse_ceiling_fraction: float
    fast_success_bars: int
    early_reversal_bars: int
    round_trip_cost_fraction: float

    def __post_init__(self) -> None:
        finite = (
            self.leverage,
            self.primary_protectable_roe,
            self.favorable_atr_multiple,
            self.adverse_atr_multiple,
            self.favorable_floor_fraction,
            self.adverse_floor_fraction,
            self.favorable_ceiling_fraction,
            self.adverse_ceiling_fraction,
            self.round_trip_cost_fraction,
            *self.roe_checkpoints,
        )
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in finite)
            or self.leverage <= 0.0
            or not self.roe_checkpoints
            or tuple(sorted(set(self.roe_checkpoints))) != self.roe_checkpoints
            or self.primary_protectable_roe not in self.roe_checkpoints
            or self.favorable_floor_fraction > self.favorable_ceiling_fraction
            or self.adverse_floor_fraction > self.adverse_ceiling_fraction
            or self.fast_success_bars <= 0
            or self.early_reversal_bars <= 0
            or self.round_trip_cost_fraction >= 1.0
        ):
            raise RegimeAwareV6Error("invalid directional v6 path contract")


@dataclass(frozen=True)
class CommitteeObservation:
    action: str
    long_votes: int
    short_votes: int
    neutral_votes: int
    available: bool = True

    def __post_init__(self) -> None:
        if (
            self.action not in {"LONG", "SHORT", "HOLD", "PASS", "CLOSE"}
            or min(self.long_votes, self.short_votes, self.neutral_votes) < 0
        ):
            raise RegimeAwareV6Error("invalid committee observation")


@dataclass(frozen=True)
class ExitEyeReplayConfig:
    enabled: bool
    min_roe_to_protect: float
    min_peak_roe_to_protect: float
    min_giveback_from_peak_roe: float
    neutral_votes_to_protect: int
    opposite_votes_to_close: int
    min_roe_to_close_on_opposite: float
    min_peak_roe_to_close_on_opposite: float
    close_on_neutral_decay: bool
    neutral_close_votes: int
    min_roe_to_close_on_neutral: float
    min_peak_roe_to_close_on_neutral: float
    min_giveback_to_close_on_neutral: float
    require_consecutive_neutral_close: int
    require_consecutive_neutral: int
    require_consecutive_opposite: int
    min_minutes_in_trade: float

    def __post_init__(self) -> None:
        values = (
            self.min_roe_to_protect,
            self.min_peak_roe_to_protect,
            self.min_giveback_from_peak_roe,
            self.min_roe_to_close_on_opposite,
            self.min_peak_roe_to_close_on_opposite,
            self.min_roe_to_close_on_neutral,
            self.min_peak_roe_to_close_on_neutral,
            self.min_giveback_to_close_on_neutral,
            self.min_minutes_in_trade,
        )
        counts = (
            self.neutral_votes_to_protect,
            self.opposite_votes_to_close,
            self.neutral_close_votes,
            self.require_consecutive_neutral_close,
            self.require_consecutive_neutral,
            self.require_consecutive_opposite,
        )
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in values)
            or min(counts) < 0
        ):
            raise RegimeAwareV6Error("invalid ExitEye replay contract")


def _fraction(side: DirectionalSide, entry: float, price: float) -> float:
    return ((price - entry) / entry) * side.sign


def _first_at_least(values: Sequence[float], threshold: float) -> int | None:
    return next(
        (index for index, value in enumerate(values, start=1) if value >= threshold),
        None,
    )


def directional_path_outcome(
    *,
    signal: Candle,
    future: Sequence[Candle],
    atr_fraction: float,
    side: DirectionalSide,
    contract: DirectionalPathContract,
) -> Mapping[str, Any]:
    """Label one side using only the future path, never as an input feature."""

    if (
        not future
        or not math.isfinite(atr_fraction)
        or atr_fraction <= 0.0
        or signal.close_time != future[0].open_time
    ):
        raise RegimeAwareV6Error("directional path is invalid")
    entry = future[0].open
    if not math.isfinite(entry) or entry <= 0.0:
        raise RegimeAwareV6Error("directional entry price is invalid")
    favorable_barrier = min(
        contract.favorable_ceiling_fraction,
        max(
            contract.favorable_floor_fraction,
            atr_fraction * contract.favorable_atr_multiple,
        ),
    )
    adverse_barrier = min(
        contract.adverse_ceiling_fraction,
        max(
            contract.adverse_floor_fraction,
            atr_fraction * contract.adverse_atr_multiple,
        ),
    )
    if side is DirectionalSide.LONG:
        favorable_path = [max(0.0, candle.high / entry - 1.0) for candle in future]
        adverse_path = [max(0.0, 1.0 - candle.low / entry) for candle in future]
    else:
        favorable_path = [max(0.0, 1.0 - candle.low / entry) for candle in future]
        adverse_path = [max(0.0, candle.high / entry - 1.0) for candle in future]
    favorable_bar = _first_at_least(favorable_path, favorable_barrier)
    adverse_bar = _first_at_least(adverse_path, adverse_barrier)
    ambiguous = favorable_bar is not None and favorable_bar == adverse_bar
    if ambiguous:
        order = "SAME_BAR_AMBIGUOUS"
    elif favorable_bar is not None and (
        adverse_bar is None or favorable_bar < adverse_bar
    ):
        order = "FAVORABLE_FIRST"
    elif adverse_bar is not None:
        order = "ADVERSE_FIRST"
    else:
        order = "NEITHER_REACHED"
    checkpoint_bars = {
        f"roe_{int(round(roe * 100))}_first_bar": _first_at_least(
            favorable_path, roe / contract.leverage
        )
        for roe in contract.roe_checkpoints
    }
    primary_bar = checkpoint_bars[
        f"roe_{int(round(contract.primary_protectable_roe * 100))}_first_bar"
    ]
    protectable_before_adverse = primary_bar is not None and (
        adverse_bar is None or primary_bar < adverse_bar
    )
    close_returns = [_fraction(side, entry, candle.close) for candle in future]
    positive_after_cost_bar = next(
        (
            index
            for index, value in enumerate(close_returns, start=1)
            if value > contract.round_trip_cost_fraction
        ),
        None,
    )
    return {
        "side": side.value,
        "entry_price": entry,
        "atr_fraction": atr_fraction,
        "favorable_barrier_fraction": favorable_barrier,
        "adverse_barrier_fraction": adverse_barrier,
        "first_favorable_bar": favorable_bar,
        "first_adverse_bar": adverse_bar,
        "barrier_order": order,
        "same_bar_ambiguity": ambiguous,
        "target_before_stop": order == "FAVORABLE_FIRST",
        "clean_fast_success": order == "FAVORABLE_FIRST"
        and favorable_bar is not None
        and favorable_bar <= contract.fast_success_bars,
        "protectable_advantage": protectable_before_adverse,
        "early_reversal": adverse_bar is not None
        and adverse_bar <= contract.early_reversal_bars
        and not protectable_before_adverse,
        "mfe_fraction": max(favorable_path),
        "mae_fraction": max(adverse_path),
        "time_underwater_bars": sum(value <= 0.0 for value in close_returns),
        "first_positive_after_cost_bar": positive_after_cost_bar,
        "time_to_protectable_fraction": (
            primary_bar / len(future) if primary_bar is not None else 1.0
        ),
        "terminal_return_after_costs": close_returns[-1]
        - contract.round_trip_cost_fraction,
        **checkpoint_bars,
    }


def classify_regime_axes(
    base_features: Mapping[str, Any],
    context: Mapping[str, float],
) -> Mapping[str, str]:
    """Factor global/local context without using any future observation."""

    required = (
        "market_breadth_6",
        "high_vol_regime_proxy",
        "low_vol_regime_proxy",
        "ret_1",
        "ret_3",
        "ret_12",
        "lower_wick_fraction",
        "upper_wick_fraction",
        "volume_ratio_6_24",
    )
    try:
        base = {name: float(base_features[name]) for name in required}
        hourly_return = float(context["1h_ret_3"])
        hourly_long = float(context["1h_trend_stack_long"]) > 0.5
        hourly_short = float(context["1h_trend_stack_short"]) > 0.5
        vol_ratio = float(context["15m_volatility_ratio_6_24"])
        hourly_chop = float(context["1h_chop_12"])
        hourly_strength = float(context["1h_trend_strength_12"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RegimeAwareV6Error("regime inputs are incomplete") from exc
    if not all(
        math.isfinite(value)
        for value in (
            *base.values(),
            hourly_return,
            vol_ratio,
            hourly_chop,
            hourly_strength,
        )
    ):
        raise RegimeAwareV6Error("regime inputs are non-finite")
    breadth = base["market_breadth_6"]
    if hourly_long and hourly_return > 0.0 and breadth >= 0.50:
        direction = "BULLISH"
    elif hourly_short and hourly_return < 0.0 and breadth <= 0.50:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
    if base["high_vol_regime_proxy"] > 0.5 or vol_ratio >= 1.25:
        volatility = "HIGH"
    elif base["low_vol_regime_proxy"] > 0.5 and vol_ratio <= 0.80:
        volatility = "LOW"
    else:
        volatility = "NORMAL"
    if (hourly_long or hourly_short) and hourly_strength >= 1.5 and hourly_chop <= 0.60:
        structure = "TREND"
    elif hourly_chop >= 0.70:
        structure = "RANGE"
    else:
        structure = "TRANSITION"
    aligned = (direction == "BULLISH" and base["ret_3"] > 0.0) or (
        direction == "BEARISH" and base["ret_3"] < 0.0
    )
    against = (direction == "BULLISH" and base["ret_1"] < 0.0) or (
        direction == "BEARISH" and base["ret_1"] > 0.0
    )
    exhaustion = (
        abs(base["ret_12"]) > 2.0 * max(abs(base["ret_3"]), 1e-8)
        and base["volume_ratio_6_24"] >= 1.0
        and (base["lower_wick_fraction"] >= 0.45 or base["upper_wick_fraction"] >= 0.45)
    )
    if exhaustion:
        phase = RegimePhase.EXHAUSTION.value
    elif against:
        phase = RegimePhase.PULLBACK.value
    elif aligned and structure == "TREND":
        phase = RegimePhase.CONTINUATION.value
    else:
        phase = RegimePhase.TRANSITION.value
    return {
        "direction": direction,
        "volatility": volatility,
        "structure": structure,
        "phase": phase,
        "identity": f"{direction}::{volatility}::{structure}::{phase}",
    }


def regime_router_feature_vector(
    features_by_symbol: Mapping[str, Mapping[str, Any]],
    *,
    btc_symbol: str = "BTCUSDT",
) -> tuple[float, ...]:
    """Build causal BTC and cross-sectional context for the regime router."""

    if btc_symbol not in features_by_symbol or not features_by_symbol:
        raise RegimeAwareV6Error("regime router requires BTC and symbol features")
    required = (
        "ret_1",
        "ret_3",
        "ret_12",
        "atr_12",
        "volume_ratio_6_24",
        "trend_strength_12",
        "high_vol_regime_proxy",
        "low_vol_regime_proxy",
    )
    try:
        normalized = {
            symbol: {name: float(values[name]) for name in required}
            for symbol, values in features_by_symbol.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RegimeAwareV6Error("regime router features are incomplete") from exc
    if not all(
        math.isfinite(value)
        for values in normalized.values()
        for value in values.values()
    ):
        raise RegimeAwareV6Error("regime router features are non-finite")
    btc = normalized[btc_symbol]
    rows = tuple(normalized.values())
    result = (
        btc["ret_1"],
        btc["ret_3"],
        btc["ret_12"],
        btc["atr_12"],
        btc["volume_ratio_6_24"],
        btc["trend_strength_12"],
        *(
            sum(row[name] for row in rows) / len(rows)
            for name in ("ret_1", "ret_3", "ret_12")
        ),
        *(
            sum(row[name] > 0.0 for row in rows) / len(rows)
            for name in ("ret_1", "ret_3", "ret_12")
        ),
        sum(row["atr_12"] for row in rows) / len(rows),
        sum(row["volume_ratio_6_24"] for row in rows) / len(rows),
        sum(row["high_vol_regime_proxy"] > 0.5 for row in rows) / len(rows),
        sum(row["low_vol_regime_proxy"] > 0.5 for row in rows) / len(rows),
    )
    if len(result) != len(REGIME_ROUTER_FEATURE_NAMES):
        raise RegimeAwareV6Error("regime router feature length is invalid")
    return tuple(float(value) for value in result)


def realized_global_regime(
    future_return_by_symbol: Mapping[str, float],
    *,
    btc_direction_threshold_fraction: float,
    cross_section_breadth_threshold: float,
    btc_symbol: str = "BTCUSDT",
) -> str:
    """Label the future market path for supervised, fold-contained routing."""

    if (
        btc_symbol not in future_return_by_symbol
        or not future_return_by_symbol
        or not math.isfinite(btc_direction_threshold_fraction)
        or btc_direction_threshold_fraction <= 0.0
        or not math.isfinite(cross_section_breadth_threshold)
        or not 0.5 < cross_section_breadth_threshold < 1.0
    ):
        raise RegimeAwareV6Error("realized regime contract is invalid")
    values = tuple(float(value) for value in future_return_by_symbol.values())
    if not all(math.isfinite(value) for value in values):
        raise RegimeAwareV6Error("realized regime returns are non-finite")
    btc_return = float(future_return_by_symbol[btc_symbol])
    breadth = sum(value > 0.0 for value in values) / len(values)
    if (
        btc_return >= btc_direction_threshold_fraction
        and breadth >= cross_section_breadth_threshold
    ):
        return "BULLISH"
    if (
        btc_return <= -btc_direction_threshold_fraction
        and breadth <= 1.0 - cross_section_breadth_threshold
    ):
        return "BEARISH"
    return "NEUTRAL"


def directional_role(side: DirectionalSide, direction: str) -> DirectionalRole:
    if direction == "BEARISH":
        return (
            DirectionalRole.PRIMARY_TREND
            if side is DirectionalSide.SHORT
            else DirectionalRole.TACTICAL_COUNTERTREND
        )
    if direction == "BULLISH":
        return (
            DirectionalRole.PRIMARY_TREND
            if side is DirectionalSide.LONG
            else DirectionalRole.TACTICAL_COUNTERTREND
        )
    if direction == "NEUTRAL":
        return DirectionalRole.SELECTIVE
    raise RegimeAwareV6Error("unknown regime direction")


def directional_interactions(
    base_features: Mapping[str, Any], side: DirectionalSide
) -> tuple[float, ...]:
    """Mirror causal market evidence so specialists see side-relative values."""

    names = (
        "ret_1",
        "ret_3",
        "ret_12",
        "market_direction_6",
        "market_breadth_6",
        "volume_ratio_6_24",
        "close_position_in_range",
        "lower_wick_fraction",
        "upper_wick_fraction",
        "trend_stack_long",
        "trend_stack_short",
    )
    try:
        values = {name: float(base_features[name]) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        raise RegimeAwareV6Error("directional features are incomplete") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise RegimeAwareV6Error("directional features are non-finite")
    favorable_wick = (
        values["lower_wick_fraction"]
        if side is DirectionalSide.LONG
        else values["upper_wick_fraction"]
    )
    adverse_wick = (
        values["upper_wick_fraction"]
        if side is DirectionalSide.LONG
        else values["lower_wick_fraction"]
    )
    trend = (
        values["trend_stack_long"]
        if side is DirectionalSide.LONG
        else values["trend_stack_short"]
    )
    breadth = (
        values["market_breadth_6"]
        if side is DirectionalSide.LONG
        else 1.0 - values["market_breadth_6"]
    )
    return (
        values["ret_1"] * side.sign,
        values["ret_3"] * side.sign,
        values["ret_12"] * side.sign,
        values["market_direction_6"] * side.sign,
        breadth,
        values["volume_ratio_6_24"],
        (
            values["close_position_in_range"]
            if side is DirectionalSide.LONG
            else 1.0 - values["close_position_in_range"]
        ),
        favorable_wick,
        adverse_wick,
        trend,
    )


def regime_aware_feature_vector(
    *,
    multitimeframe_features: Sequence[float],
    base_features: Mapping[str, Any],
    side: DirectionalSide,
    regime: Mapping[str, str],
) -> tuple[float, ...]:
    """Build the shared schema while keeping LONG and SHORT fits separate."""

    if len(multitimeframe_features) != len(LONG_V21_FEATURE_NAMES):
        raise RegimeAwareV6Error("multitimeframe feature length is invalid")
    values = tuple(float(value) for value in multitimeframe_features)
    if not all(math.isfinite(value) for value in values):
        raise RegimeAwareV6Error("multitimeframe features are non-finite")
    direction = str(regime.get("direction"))
    volatility = str(regime.get("volatility"))
    structure = str(regime.get("structure"))
    phase = str(regime.get("phase"))
    if (
        direction not in REGIME_DIRECTIONS
        or volatility not in REGIME_VOLATILITIES
        or structure not in REGIME_STRUCTURES
        or phase not in REGIME_PHASES
    ):
        raise RegimeAwareV6Error("regime axes are invalid")
    role = directional_role(side, direction)
    result = (
        *values,
        *directional_interactions(base_features, side),
        *(1.0 if direction == value else 0.0 for value in REGIME_DIRECTIONS),
        *(1.0 if volatility == value else 0.0 for value in REGIME_VOLATILITIES),
        *(1.0 if structure == value else 0.0 for value in REGIME_STRUCTURES),
        *(1.0 if phase == value else 0.0 for value in REGIME_PHASES),
        *(1.0 if role is value else 0.0 for value in DirectionalRole),
    )
    if len(result) != len(REGIME_AWARE_V6_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in result
    ):
        raise RegimeAwareV6Error("regime-aware feature vector is invalid")
    return result


def _canonical(candles: Sequence[Candle]) -> tuple[CanonicalBar, ...]:
    return tuple(
        CanonicalBar(
            candle.open_time,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        )
        for candle in candles
    )


def _exit_eye_close(
    *,
    side: DirectionalSide,
    entry: float,
    future: Sequence[Candle],
    observations: Sequence[CommitteeObservation],
    leverage: float,
    config: ExitEyeReplayConfig,
) -> Mapping[str, Any]:
    if len(observations) != len(future):
        raise RegimeAwareV6Error("ExitEye observations do not align with future")
    peak_roe = 0.0
    neutral_count = 0
    neutral_close_count = 0
    opposite_count = 0
    protect_events = 0
    unavailable = 0
    for bar_index, (candle, observation) in enumerate(
        zip(future, observations), start=1
    ):
        favorable_price = candle.high if side is DirectionalSide.LONG else candle.low
        peak_roe = max(peak_roe, _fraction(side, entry, favorable_price) * leverage)
        current_roe = _fraction(side, entry, candle.close) * leverage
        if not observation.available:
            unavailable += 1
            neutral_count = neutral_close_count = opposite_count = 0
            continue
        supported = side.value
        opposite = (
            DirectionalSide.SHORT.value
            if side is DirectionalSide.LONG
            else DirectionalSide.LONG.value
        )
        opposite_votes = (
            observation.short_votes
            if side is DirectionalSide.LONG
            else observation.long_votes
        )
        neutral_condition = (
            observation.action != supported
            and observation.neutral_votes >= config.neutral_votes_to_protect
        )
        opposite_condition = (
            observation.action == opposite
            and opposite_votes >= config.opposite_votes_to_close
        )
        neutral_count = neutral_count + 1 if neutral_condition else 0
        neutral_close_count = neutral_close_count + 1 if neutral_condition else 0
        opposite_count = opposite_count + 1 if opposite_condition else 0
        if (
            not config.enabled
            or current_roe <= 0.0
            or bar_index * 5.0 < config.min_minutes_in_trade
        ):
            continue
        giveback = max(0.0, peak_roe - current_roe)
        if (
            current_roe >= config.min_roe_to_close_on_opposite
            and peak_roe >= config.min_peak_roe_to_close_on_opposite
            and opposite_condition
            and opposite_count >= config.require_consecutive_opposite
        ):
            return {
                "bar": bar_index,
                "price": candle.close,
                "reason": "EXIT_EYE_OPPOSITE_SIGNAL",
                "peak_roe": peak_roe,
                "unavailable_observations": unavailable,
                "protect_events": protect_events,
            }
        neutral_decay = (
            current_roe >= config.min_roe_to_protect
            and peak_roe >= config.min_peak_roe_to_protect
            and giveback >= config.min_giveback_from_peak_roe
            and neutral_condition
            and neutral_count >= config.require_consecutive_neutral
        )
        if neutral_decay:
            protect_events += 1
        if (
            neutral_decay
            and config.close_on_neutral_decay
            and current_roe >= config.min_roe_to_close_on_neutral
            and peak_roe >= config.min_peak_roe_to_close_on_neutral
            and giveback >= config.min_giveback_to_close_on_neutral
            and observation.neutral_votes >= config.neutral_close_votes
            and neutral_close_count >= config.require_consecutive_neutral_close
        ):
            return {
                "bar": bar_index,
                "price": candle.close,
                "reason": "EXIT_EYE_NEUTRAL_DECAY",
                "peak_roe": peak_roe,
                "unavailable_observations": unavailable,
                "protect_events": protect_events,
            }
    return {
        "bar": None,
        "price": None,
        "reason": None,
        "peak_roe": peak_roe,
        "unavailable_observations": unavailable,
        "protect_events": protect_events,
    }


def replay_full_lifecycle(
    *,
    side: DirectionalSide,
    history: Sequence[Candle],
    future: Sequence[Candle],
    observations: Sequence[CommitteeObservation],
    protection: TsProtectionConfig,
    exit_eye: ExitEyeReplayConfig,
) -> Mapping[str, Any]:
    """Overlay causal close-time ExitEye exits on pessimistic price protection."""

    if not future:
        raise RegimeAwareV6Error("full lifecycle requires future bars")
    eye = _exit_eye_close(
        side=side,
        entry=future[0].open,
        future=future,
        observations=observations,
        leverage=protection.leverage,
        config=exit_eye,
    )
    path_results: dict[str, Mapping[str, Any]] = {}
    for path in IntrabarPath:
        price = replay_ts_price_protection(
            side=side,
            history=_canonical(history),
            future=_canonical(future),
            path=path,
            config=protection,
        )
        eye_bar = eye["bar"]
        eye_net = (
            _fraction(side, future[0].open, float(eye["price"]))
            - protection.round_trip_cost_fraction
            if eye_bar is not None
            else None
        )
        use_eye = eye_bar is not None and int(eye_bar) < price.bars_held
        same_bar = eye_bar is not None and int(eye_bar) == price.bars_held
        if use_eye:
            net = float(eye_net)
            reason = str(eye["reason"])
            bars = int(eye_bar)
        elif (
            same_bar and eye_net is not None and eye_net < price.net_return_after_costs
        ):
            net = float(eye_net)
            reason = f"{eye['reason']}_SAME_BAR_PESSIMISTIC"
            bars = int(eye_bar)
        else:
            net = price.net_return_after_costs
            reason = price.exit_reason.value
            bars = price.bars_held
        path_results[path.value] = {
            "net_return_after_costs": net,
            "exit_reason": reason,
            "bars_held": bars,
            "price_protection_net": price.net_return_after_costs,
            "price_protection_reason": price.exit_reason.value,
            "break_even_armed": price.break_even_armed,
            "trailing_armed": price.trailing_armed,
        }
    values = [float(row["net_return_after_costs"]) for row in path_results.values()]
    price_values = [float(row["price_protection_net"]) for row in path_results.values()]
    worst_path = min(
        path_results, key=lambda key: path_results[key]["net_return_after_costs"]
    )
    return {
        "full_lifecycle_worst_net_return": min(values),
        "price_protection_worst_net_return": min(price_values),
        "full_lifecycle_path_spread": max(values) - min(values),
        "full_lifecycle_worst_exit_reason": path_results[worst_path]["exit_reason"],
        "full_lifecycle_worst_bars_held": path_results[worst_path]["bars_held"],
        "exit_eye_close_bar": eye["bar"],
        "exit_eye_close_reason": eye["reason"],
        "exit_eye_protect_events": eye["protect_events"],
        "committee_observations_complete": eye["unavailable_observations"] == 0,
        "committee_unavailable_observations": eye["unavailable_observations"],
        "path_results": path_results,
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }
