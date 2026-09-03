"""LONG v3 causal microstructure features, candidates, and hard negatives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from ..domain import Candle
from .long_entry_v22_shadow import LONG_V22_FEATURE_NAMES


MICROSTRUCTURE_FEATURE_NAMES = (
    "taker_buy_ratio_1",
    "taker_buy_ratio_3",
    "taker_buy_ratio_12",
    "taker_imbalance_acceleration_3_12",
    "trade_intensity_ratio_3_24",
    "quote_volume_ratio_3_24",
    "average_trade_size_ratio_3_24",
    "signed_taker_price_pressure",
    "funding_rate_last",
    "funding_rate_change",
)
LONG_V3_FEATURE_NAMES = (*LONG_V22_FEATURE_NAMES, *MICROSTRUCTURE_FEATURE_NAMES)


class LongCandidateFamily(str, Enum):
    BREAKOUT_EXPANSION = "BREAKOUT_EXPANSION"
    TREND_PULLBACK_RECLAIM = "TREND_PULLBACK_RECLAIM"
    CONFIRMED_REVERSAL = "CONFIRMED_REVERSAL"
    BREAKOUT_RETEST = "BREAKOUT_RETEST"
    NONE = "NONE"


class HardNegativeType(str, Enum):
    WRONG_DIRECTION = "WRONG_DIRECTION"
    LATE_RECOVERY = "LATE_RECOVERY"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    FALLING_KNIFE = "FALLING_KNIFE"
    CHOP_NO_RESOLUTION = "CHOP_NO_RESOLUTION"
    NOT_HARD_NEGATIVE = "NOT_HARD_NEGATIVE"


class LongV3ShadowError(ValueError):
    pass


@dataclass(frozen=True)
class MicrostructureBar:
    quote_volume: float
    trade_count: int
    taker_buy_base: float
    base_volume: float

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    self.quote_volume,
                    self.taker_buy_base,
                    self.base_volume,
                )
            )
            or self.trade_count < 0
        ):
            raise LongV3ShadowError("invalid LONG v3 microstructure bar")


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    return numerator / denominator if denominator > 0.0 else fallback


def microstructure_feature_vector(
    bars: Sequence[MicrostructureBar],
    *,
    return_1: float,
    atr_fraction: float,
    funding_rate_last: float,
    funding_rate_previous: float,
) -> tuple[float, ...]:
    """Build causal rolling pressure features ending at the signal candle."""

    if len(bars) < 24 or not all(
        math.isfinite(value)
        for value in (
            return_1,
            atr_fraction,
            funding_rate_last,
            funding_rate_previous,
        )
    ):
        raise LongV3ShadowError("insufficient LONG v3 microstructure history")
    if atr_fraction <= 0.0:
        raise LongV3ShadowError("LONG v3 ATR must be positive")
    taker_ratios = [
        min(1.0, max(0.0, _ratio(bar.taker_buy_base, bar.base_volume, 0.5)))
        for bar in bars[-24:]
    ]
    trades = [float(bar.trade_count) for bar in bars[-24:]]
    quote = [bar.quote_volume for bar in bars[-24:]]
    average_trade = [
        _ratio(bar.quote_volume, float(bar.trade_count), 0.0) for bar in bars[-24:]
    ]
    taker_1 = taker_ratios[-1]
    taker_3 = _mean(taker_ratios[-3:])
    taker_12 = _mean(taker_ratios[-12:])
    features = (
        taker_1,
        taker_3,
        taker_12,
        taker_3 - taker_12,
        _ratio(_mean(trades[-3:]), _mean(trades), 1.0),
        _ratio(_mean(quote[-3:]), _mean(quote), 1.0),
        _ratio(_mean(average_trade[-3:]), _mean(average_trade), 1.0),
        (2.0 * taker_3 - 1.0) * math.tanh(return_1 / atr_fraction),
        funding_rate_last,
        funding_rate_last - funding_rate_previous,
    )
    if not all(math.isfinite(value) for value in features):
        raise LongV3ShadowError("non-finite LONG v3 microstructure feature")
    return features


def long_v3_feature_vector(
    v22_vector: Sequence[float], microstructure: Sequence[float]
) -> tuple[float, ...]:
    if len(v22_vector) != len(LONG_V22_FEATURE_NAMES) or len(
        microstructure
    ) != len(MICROSTRUCTURE_FEATURE_NAMES):
        raise LongV3ShadowError("LONG v3 feature input length is invalid")
    result = (*v22_vector, *microstructure)
    if not all(math.isfinite(float(value)) for value in result):
        raise LongV3ShadowError("LONG v3 feature vector is non-finite")
    return result


def _number(values: Mapping[str, Any], name: str) -> float:
    try:
        value = float(values[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise LongV3ShadowError(f"missing LONG v3 candidate input: {name}") from exc
    if not math.isfinite(value):
        raise LongV3ShadowError(f"non-finite LONG v3 candidate input: {name}")
    return value


def classify_long_v3_candidate(
    *,
    base: Mapping[str, Any],
    context: Mapping[str, float],
    micro: Mapping[str, float],
    history: Sequence[Candle],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Recognize selective LONG events from information known at signal close."""

    if len(history) < 24:
        raise LongV3ShadowError("insufficient candidate history")
    atr = max(_number(base, "atr_12"), 1e-8)
    breakout_config = config["breakout_expansion"]
    breakout = {
        "at_prior_high": _number(base, "distance_to_rolling_high_12")
        <= float(breakout_config["maximum_distance_to_prior_high"]),
        "positive_return": _number(base, "ret_3")
        > float(breakout_config["minimum_return_3"]),
        "close_conviction": _number(base, "close_position_in_range")
        >= float(breakout_config["minimum_close_location"]),
        "body_conviction": _number(base, "body_to_range")
        >= float(breakout_config["minimum_body_to_range"]),
        "volume_expansion": _number(base, "volume_ratio_6_24")
        >= float(breakout_config["minimum_volume_ratio"]),
        "taker_support": _number(micro, "taker_buy_ratio_3")
        >= float(breakout_config["minimum_taker_buy_ratio"]),
        "trade_intensity": _number(micro, "trade_intensity_ratio_3_24")
        >= float(breakout_config["minimum_trade_intensity_ratio"]),
    }

    pullback_config = config["trend_pullback_reclaim"]
    close_ema_atr = _number(base, "close_vs_ema_12") / atr
    pullback = {
        "long_stack": _number(base, "trend_stack_long") > 0.5,
        "hourly_long_stack": _number(context, "1h_trend_stack_long") > 0.5,
        "prior_direction": _number(base, "ret_12")
        > float(pullback_config["minimum_return_12"]),
        "turning_up": _number(base, "ret_1")
        > float(pullback_config["minimum_return_1"]),
        "near_ema": float(pullback_config["minimum_close_vs_ema12_atr"])
        <= close_ema_atr
        <= float(pullback_config["maximum_close_vs_ema12_atr"]),
        "taker_support": _number(micro, "taker_buy_ratio_3")
        >= float(pullback_config["minimum_taker_buy_ratio"]),
    }

    reversal_config = config["confirmed_reversal"]
    reversal = {
        "prior_down": _number(base, "ret_12")
        < float(reversal_config["maximum_prior_return_12"]),
        "one_bar_turn": _number(base, "ret_1")
        > float(reversal_config["minimum_return_1"]),
        "three_bar_turn": _number(base, "ret_3")
        > float(reversal_config["minimum_return_3"]),
        "fast_ema_reclaim": _number(base, "close_vs_ema_6") > 0.0,
        "lower_wick_dominance": _number(base, "lower_wick_fraction")
        > _number(base, "upper_wick_fraction"),
        "taker_support": _number(micro, "taker_buy_ratio_3")
        >= float(reversal_config["minimum_taker_buy_ratio"]),
        "taker_acceleration": _number(micro, "taker_imbalance_acceleration_3_12")
        > float(reversal_config["minimum_taker_acceleration"]),
        "fifteen_minute_turn": _number(context, "15m_ret_3")
        > float(reversal_config["minimum_15m_return_3"]),
    }

    retest_config = config["breakout_retest"]
    lookback = int(retest_config["breakout_lookback_bars"])
    prior = history[-(lookback + 13) : -lookback]
    recent = history[-lookback:-1]
    prior_high = max(candle.high for candle in prior)
    breakout_seen = any(candle.close > prior_high for candle in recent)
    current = history[-1]
    retest_distance_atr = abs(current.close / prior_high - 1.0) / atr
    retest = {
        "breakout_seen": breakout_seen,
        "level_held": current.close >= prior_high * (1.0 - 0.5 * atr),
        "near_breakout_level": retest_distance_atr
        <= float(retest_config["maximum_retest_distance_atr"]),
        "turning_up": _number(base, "ret_1")
        > float(retest_config["minimum_return_1"]),
        "taker_support": _number(micro, "taker_buy_ratio_3")
        >= float(retest_config["minimum_taker_buy_ratio"]),
        "long_stack": _number(base, "trend_stack_long") > 0.5,
    }

    if all(retest.values()):
        family = LongCandidateFamily.BREAKOUT_RETEST
    elif all(breakout.values()):
        family = LongCandidateFamily.BREAKOUT_EXPANSION
    elif all(pullback.values()):
        family = LongCandidateFamily.TREND_PULLBACK_RECLAIM
    elif all(reversal.values()):
        family = LongCandidateFamily.CONFIRMED_REVERSAL
    else:
        family = LongCandidateFamily.NONE
    return {
        "schema_id": "aegis-long-v3-candidate-shadow-v1",
        "family": family.value,
        "is_candidate": family is not LongCandidateFamily.NONE,
        "evidence": {
            LongCandidateFamily.BREAKOUT_EXPANSION.value: breakout,
            LongCandidateFamily.TREND_PULLBACK_RECLAIM.value: pullback,
            LongCandidateFamily.CONFIRMED_REVERSAL.value: reversal,
            LongCandidateFamily.BREAKOUT_RETEST.value: retest,
        },
        "selection_effect": "NONE",
        "exchange_authority": False,
        "exchange_mutations": 0,
    }


def classify_hard_negative(
    candidate_family: str, outcome: Mapping[str, Any]
) -> HardNegativeType:
    order = str(outcome["barrier_order"])
    mae = float(outcome["mae_fraction"])
    atr = max(float(outcome["atr_fraction"]), 1e-12)
    if mae >= 1.5 * atr and order != "FAVORABLE_FIRST":
        return HardNegativeType.FALLING_KNIFE
    if candidate_family in {
        LongCandidateFamily.BREAKOUT_EXPANSION.value,
        LongCandidateFamily.BREAKOUT_RETEST.value,
    } and order in {"ADVERSE_FIRST", "SAME_BAR_AMBIGUOUS"}:
        return HardNegativeType.FALSE_BREAKOUT
    if bool(outcome["target_before_stop"]) and not bool(outcome["clean_fast_success"]):
        return HardNegativeType.LATE_RECOVERY
    if order == "NEITHER_REACHED":
        return HardNegativeType.CHOP_NO_RESOLUTION
    if not bool(outcome["target_before_stop"]):
        return HardNegativeType.WRONG_DIRECTION
    return HardNegativeType.NOT_HARD_NEGATIVE
