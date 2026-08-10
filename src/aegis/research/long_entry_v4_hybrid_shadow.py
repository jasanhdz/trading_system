"""LONG v4 technical setup, entry-quality, and exit-observation contracts."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping, Sequence

from ..domain import Candle


class LongTechnicalSetup(str, Enum):
    TREND_CONTINUATION = "TREND_CONTINUATION"
    BREAKOUT_RETEST = "BREAKOUT_RETEST"
    CAPITULATION_REVERSAL = "CAPITULATION_REVERSAL"


SETUP_SOURCE = {
    "TREND_PULLBACK_RECLAIM": LongTechnicalSetup.TREND_CONTINUATION,
    "BREAKOUT_RETEST": LongTechnicalSetup.BREAKOUT_RETEST,
    "CONFIRMED_REVERSAL": LongTechnicalSetup.CAPITULATION_REVERSAL,
}

EXIT_STATE_FEATURE_NAMES = (
    "elapsed_fraction",
    "current_net_return",
    "mfe_to_checkpoint",
    "mae_to_checkpoint",
    "drawdown_from_peak",
    "underwater_fraction",
    "return_last_1",
    "return_last_3",
    "favorable_distance_atr",
    "adverse_distance_atr",
)


class LongV4ShadowError(ValueError):
    pass


def technical_setup(candidate_family: str) -> LongTechnicalSetup | None:
    """Map only preregistered causal candidates into economic setup families."""

    return SETUP_SOURCE.get(candidate_family)


def clean_entry_label(row: Mapping[str, Any], *, horizon_bars: int) -> bool:
    """Require correct direction, bounded MAE, and prompt resolution."""

    if horizon_bars <= 0:
        raise LongV4ShadowError("LONG v4 horizon must be positive")
    required = (
        "target_before_stop",
        "clean_fast_success",
        "mae_fraction",
        "adverse_barrier_fraction",
        "time_underwater_bars",
    )
    if any(name not in row for name in required):
        raise LongV4ShadowError("LONG v4 entry label inputs are incomplete")
    return bool(
        row["target_before_stop"]
        and row["clean_fast_success"]
        and float(row["mae_fraction"]) <= float(row["adverse_barrier_fraction"])
        and int(row["time_underwater_bars"]) <= horizon_bars / 2.0
    )


def exit_state_feature_vector(
    *,
    entry_price: float,
    observed: Sequence[Candle],
    horizon_bars: int,
    atr_fraction: float,
    round_trip_cost_fraction: float,
) -> tuple[float, ...]:
    """Describe an open LONG using only bars closed by the checkpoint."""

    if (
        entry_price <= 0.0
        or not observed
        or horizon_bars <= 0
        or not math.isfinite(atr_fraction)
        or atr_fraction <= 0.0
        or not 0.0 <= round_trip_cost_fraction < 1.0
    ):
        raise LongV4ShadowError("LONG v4 exit state is invalid")
    closes = [bar.close for bar in observed]
    current_return = closes[-1] / entry_price - 1.0
    mfe = max(bar.high / entry_price - 1.0 for bar in observed)
    mae = max(1.0 - bar.low / entry_price for bar in observed)
    peak = max(bar.high for bar in observed) / entry_price - 1.0
    return_1 = closes[-1] / (observed[-1].open or entry_price) - 1.0
    return_3 = closes[-1] / (closes[-4] if len(closes) >= 4 else entry_price) - 1.0
    features = (
        len(observed) / horizon_bars,
        current_return - round_trip_cost_fraction,
        mfe,
        mae,
        max(0.0, peak - current_return),
        sum(close < entry_price for close in closes) / len(closes),
        return_1,
        return_3,
        max(0.0, mfe - current_return) / atr_fraction,
        max(0.0, current_return + mae) / atr_fraction,
    )
    if len(features) != len(EXIT_STATE_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in features
    ):
        raise LongV4ShadowError("LONG v4 exit feature vector is invalid")
    return features


def exit_now_preferred_label(
    *, current_net_return: float, continue_worst_protected_net: float
) -> bool:
    if not all(
        math.isfinite(value)
        for value in (current_net_return, continue_worst_protected_net)
    ):
        raise LongV4ShadowError("LONG v4 exit label is non-finite")
    return current_net_return > continue_worst_protected_net


def hybrid_score(
    opportunity_probability: float,
    clean_entry_probability: float,
    falling_knife_probability: float,
    path_risk_probability: float,
) -> float:
    values = (
        opportunity_probability,
        clean_entry_probability,
        falling_knife_probability,
        path_risk_probability,
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise LongV4ShadowError("LONG v4 probability is invalid")
    return (
        opportunity_probability
        * clean_entry_probability
        * (1.0 - falling_knife_probability)
        * (1.0 - path_risk_probability)
    )
