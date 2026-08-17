"""Deterministic, causal technical entry guard used only for offline audit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


REASON_PRIORITY = (
    "SKIP_OPPOSED",
    "SKIP_EXHAUSTED",
    "SKIP_NO_SPACE",
    "SKIP_VOLATILITY_SHOCK",
)


def _value(row: Mapping[str, Any], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _any_le(values: list[float], threshold: float) -> bool:
    return any(np.isfinite(value) and value <= threshold for value in values)


def _any_ge(values: list[float], threshold: float) -> bool:
    return any(np.isfinite(value) and value >= threshold for value in values)


def assess_technical_entry(row: Mapping[str, Any], guard: Mapping[str, Any]) -> dict[str, Any]:
    """Return one deterministic action from causal features available at entry."""
    rsi_room = [_value(row, f"dir{tf}m__rsi6_remaining_room") for tf in (5, 15)]
    extension = [_value(row, f"dir{tf}m__ema25_extension_atr") for tf in (5, 15)]
    prior_move = [_value(row, f"dir{tf}m__prior_move_6_atr") for tf in (5, 15)]
    space = [_value(row, f"dir{tf}m__favorable_space_atr") for tf in (5, 15)]

    rsi_exhausted = _any_le(rsi_room, float(guard["rsi_remaining_room_critical"]))
    ema_extended = _any_ge(extension, float(guard["ema25_extension_atr_critical"]))
    mature_move = _any_ge(prior_move, float(guard["prior_move_atr_critical"]))
    exhausted = rsi_exhausted and (ema_extended or mature_move)

    opposition_checks = {
        "ema_5m": _value(row, "dir5m__ema25_extension_atr") <= float(guard["ema25_opposition_atr"]),
        "ema_15m": _value(row, "dir15m__ema25_extension_atr") <= float(guard["ema25_opposition_atr"]),
        "return_1m": _value(row, "dir1m__return_3_bps") <= float(guard["directional_return_1m_bps_opposed"]),
        "return_5m": _value(row, "dir5m__return_1_bps") <= float(guard["directional_return_5m_bps_opposed"]),
        "taker_1m": _value(row, "dir1m__taker_imbalance") <= float(guard["taker_imbalance_opposed"]),
        "taker_5m": _value(row, "dir5m__taker_imbalance") <= float(guard["taker_imbalance_opposed"]),
    }
    opposition_votes = sum(opposition_checks.values())
    price_opposed = opposition_checks["return_1m"] or opposition_checks["return_5m"]
    context_opposed = opposition_checks["ema_5m"] or opposition_checks["ema_15m"]
    flow_opposed = opposition_checks["taker_1m"] or opposition_checks["taker_5m"]
    opposed = (
        opposition_votes >= int(guard["minimum_opposition_votes"])
        and price_opposed
        and (context_opposed or flow_opposed)
    )

    no_space = (
        _any_le(space, float(guard["favorable_space_atr_critical"]))
        and (ema_extended or mature_move)
    )
    volatility_shock = (
        _any_ge(
            [_value(row, "tf5m__atr_percentile_96"), _value(row, "tf15m__atr_percentile_96")],
            float(guard["atr_percentile_shock"]),
        )
        and _any_ge(
            [_value(row, "tf5m__volume_ratio20"), _value(row, "tf15m__volume_ratio20")],
            float(guard["volume_ratio_shock"]),
        )
        and _any_le(
            [_value(row, "tf5m__path_efficiency_6"), _value(row, "tf15m__path_efficiency_6")],
            float(guard["path_efficiency_disorder"]),
        )
    )

    risks = {
        "SKIP_EXHAUSTED": exhausted,
        "SKIP_OPPOSED": opposed,
        "SKIP_NO_SPACE": no_space,
        "SKIP_VOLATILITY_SHOCK": volatility_shock,
    }
    reason = next((name for name in REASON_PRIORITY if risks[name]), "ENTER")
    return {
        "action": "ENTER" if reason == "ENTER" else "SKIP",
        "reason": reason,
        "risk_count": int(sum(risks.values())),
        "rsi_exhausted": bool(rsi_exhausted),
        "ema_extended": bool(ema_extended),
        "mature_move": bool(mature_move),
        "opposition_votes": int(opposition_votes),
        **{name.lower(): bool(value) for name, value in risks.items()},
    }


def apply_guard(frame: pd.DataFrame, guard: Mapping[str, Any]) -> pd.DataFrame:
    assessments = pd.DataFrame(
        [assess_technical_entry(row, guard) for row in frame.to_dict(orient="records")],
        index=frame.index,
    )
    return pd.concat([frame.copy(), assessments], axis=1)

