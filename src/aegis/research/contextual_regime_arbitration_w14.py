"""Causal context taxonomy and frozen ENTER/WAIT/SKIP policy for W14."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


STATES = (
    "SHOCK_OR_UNCERTAIN",
    "MATURE_OR_EXHAUSTED",
    "BREAKOUT_OR_RETEST",
    "TREND_CONTINUATION",
    "PULLBACK_WITHIN_TREND",
    "RANGE_MEAN_REVERSION",
    "REGIME_TRANSITION",
    "UNCERTAIN",
)


def _number(row: Mapping[str, Any], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _vote(value: float, threshold: float = 0.0) -> int:
    if not np.isfinite(value):
        return 0
    return 1 if value > threshold else -1 if value < -threshold else 0


def context_evidence(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    local_values = [
        _number(row, "dir1m__return_3_bps"),
        _number(row, "dir5m__return_1_bps"),
        _number(row, "dir5m__taker_imbalance"),
    ]
    local_votes = [
        _vote(local_values[0], float(thresholds["local_return_bps"])),
        _vote(local_values[1], float(thresholds["local_return_bps"])),
        _vote(local_values[2], float(thresholds["local_taker_imbalance"])),
    ]
    higher_values = [
        _number(row, "dir60m__ema25_slope_atr"),
        _number(row, "dir240m__ema25_slope_atr"),
        _number(row, "dir240m__ema25_extension_atr"),
        _number(row, "dir1440m__return_1_bps"),
        _number(row, "btc_dir60m__return_3_bps"),
        _number(row, "btc_dir240m__return_1_bps"),
    ]
    higher_votes = [_vote(value, float(thresholds["higher_ema_slope_atr"])) for value in higher_values]
    higher_available = sum(np.isfinite(value) for value in higher_values)
    minimum_higher = min(int(thresholds["minimum_higher_alignment_votes"]), higher_available)
    higher_score = sum(higher_votes)
    local_score = sum(local_votes)
    higher_aligned = minimum_higher >= 2 and higher_score >= minimum_higher
    higher_opposed = minimum_higher >= 2 and higher_score <= -minimum_higher
    local_aligned = local_score >= int(thresholds["minimum_local_alignment_votes"])
    local_opposed = local_score <= -int(thresholds["minimum_local_alignment_votes"])

    rsi_exhausted = min(
        _number(row, "dir5m__rsi6_remaining_room"), _number(row, "dir15m__rsi6_remaining_room")
    ) <= float(thresholds["rsi_remaining_room_exhausted"])
    extended = max(
        _number(row, "dir5m__ema25_extension_atr"), _number(row, "dir15m__ema25_extension_atr")
    ) >= float(thresholds["ema_extension_atr_exhausted"])
    mature = max(
        _number(row, "dir5m__prior_move_6_atr"), _number(row, "dir15m__prior_move_6_atr")
    ) >= float(thresholds["prior_move_atr_mature"])
    exhausted = rsi_exhausted and (extended or mature)
    no_space = min(
        _number(row, "dir5m__favorable_space_atr"), _number(row, "dir15m__favorable_space_atr")
    ) <= float(thresholds["favorable_space_atr_minimum"])
    breakout = max(
        _number(row, "dir5m__aligned_breakout"), _number(row, "dir15m__aligned_breakout")
    ) >= 1.0
    invalidated = max(
        _number(row, "dir5m__opposed_breakout"), _number(row, "dir15m__opposed_breakout")
    ) >= 1.0
    range_state = (
        _number(row, "tf60m__path_efficiency_6") <= float(thresholds["range_path_efficiency"])
        and abs(_number(row, "tf60m__ema25_slope_atr")) <= float(thresholds["range_ema_slope_abs_atr"])
    )
    mean_reversion_inward = _number(row, "dir15m__prior_move_6_atr") <= float(
        thresholds["mean_reversion_prior_move_atr"]
    )
    shock = (
        max(_number(row, "tf5m__atr_percentile_96"), _number(row, "tf15m__atr_percentile_96"))
        >= float(thresholds["shock_atr_percentile"])
        and max(_number(row, "tf5m__volume_ratio20"), _number(row, "tf15m__volume_ratio20"))
        >= float(thresholds["shock_volume_ratio"])
    )
    return {
        "local_score": int(local_score),
        "higher_score": int(higher_score),
        "higher_votes_available": int(higher_available),
        "local_aligned": bool(local_aligned),
        "local_opposed": bool(local_opposed),
        "higher_aligned": bool(higher_aligned),
        "higher_opposed": bool(higher_opposed),
        "rsi_exhausted": bool(rsi_exhausted),
        "extended": bool(extended),
        "mature": bool(mature),
        "exhausted": bool(exhausted),
        "no_space": bool(no_space),
        "breakout": bool(breakout),
        "invalidated": bool(invalidated),
        "range_state": bool(range_state),
        "mean_reversion_inward": bool(mean_reversion_inward),
        "shock": bool(shock),
    }


def classify_context(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    evidence = context_evidence(row, thresholds)
    if evidence["shock"] and not evidence["local_aligned"]:
        state = "SHOCK_OR_UNCERTAIN"
    elif evidence["exhausted"] or (evidence["no_space"] and (evidence["extended"] or evidence["mature"])):
        state = "MATURE_OR_EXHAUSTED"
    elif evidence["breakout"] and evidence["local_aligned"] and not evidence["higher_opposed"]:
        state = "BREAKOUT_OR_RETEST"
    elif evidence["higher_aligned"] and evidence["local_aligned"] and not evidence["invalidated"]:
        state = "TREND_CONTINUATION"
    elif evidence["higher_aligned"] and evidence["local_opposed"] and not evidence["invalidated"]:
        state = "PULLBACK_WITHIN_TREND"
    elif evidence["range_state"] and evidence["mean_reversion_inward"] and not evidence["no_space"]:
        state = "RANGE_MEAN_REVERSION"
    elif evidence["higher_opposed"] or evidence["invalidated"]:
        state = "REGIME_TRANSITION"
    else:
        state = "UNCERTAIN"
    return {"context_state": state, **evidence}


def choose_episode_decision(
    episode: pd.DataFrame, thresholds: Mapping[str, float], policy: Mapping[str, Any]
) -> tuple[pd.Series, bool, str, str]:
    ordered = episode.sort_values("delay_minutes")
    initial = ordered.loc[ordered["delay_minutes"].eq(0)].iloc[0]
    initial_state = classify_context(initial, thresholds)["context_state"]
    if initial_state in set(policy["enter_states"]):
        return initial, True, "ENTER_NOW", initial_state
    if initial_state in set(policy["wait_states"]):
        maximum = int(policy["maximum_wait_minutes"])
        for _, candidate in ordered.loc[ordered["delay_minutes"].between(1, maximum)].iterrows():
            state = classify_context(candidate, thresholds)["context_state"]
            evidence = context_evidence(candidate, thresholds)
            if state in {"TREND_CONTINUATION", "BREAKOUT_OR_RETEST"} and not evidence["no_space"]:
                return candidate, True, f"ENTER_AFTER_{int(candidate['delay_minutes'])}M", initial_state
        return initial, False, "SKIP_PULLBACK_NOT_REALIGNED", initial_state
    return initial, False, f"SKIP_{initial_state}", initial_state


def add_context_columns(frame: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    context = pd.DataFrame(
        [classify_context(row, thresholds) for row in frame.to_dict(orient="records")], index=frame.index
    )
    return pd.concat([frame.copy(), context], axis=1)

