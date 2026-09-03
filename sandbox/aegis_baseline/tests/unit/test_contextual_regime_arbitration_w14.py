import pandas as pd

from aegis.research.contextual_regime_arbitration_w14 import classify_context, choose_episode_decision


T = {
    "local_return_bps": 0.0,
    "local_taker_imbalance": 0.0,
    "higher_ema_slope_atr": 0.0,
    "higher_ema_extension_atr": 0.0,
    "minimum_higher_alignment_votes": 2,
    "minimum_local_alignment_votes": 2,
    "rsi_remaining_room_exhausted": 30.0,
    "ema_extension_atr_exhausted": 1.0,
    "prior_move_atr_mature": 1.5,
    "favorable_space_atr_minimum": 0.75,
    "range_path_efficiency": 0.35,
    "range_ema_slope_abs_atr": 0.20,
    "shock_atr_percentile": 0.90,
    "shock_volume_ratio": 2.50,
    "mean_reversion_prior_move_atr": -0.50,
}


def aligned_row() -> dict[str, float]:
    return {
        "dir1m__return_3_bps": 3.0,
        "dir5m__return_1_bps": 4.0,
        "dir5m__taker_imbalance": 0.2,
        "dir60m__ema25_slope_atr": 0.2,
        "dir240m__ema25_slope_atr": 0.2,
        "dir240m__ema25_extension_atr": 0.5,
        "dir5m__rsi6_remaining_room": 50.0,
        "dir15m__rsi6_remaining_room": 50.0,
        "dir5m__ema25_extension_atr": 0.4,
        "dir15m__ema25_extension_atr": 0.4,
        "dir5m__prior_move_6_atr": 0.5,
        "dir15m__prior_move_6_atr": 0.5,
        "dir5m__favorable_space_atr": 2.0,
        "dir15m__favorable_space_atr": 2.0,
    }


def test_aligned_local_and_higher_context_is_continuation() -> None:
    assert classify_context(aligned_row(), T)["context_state"] == "TREND_CONTINUATION"


def test_exhaustion_has_priority_over_continuation() -> None:
    row = aligned_row()
    row["dir5m__rsi6_remaining_room"] = 20.0
    row["dir5m__ema25_extension_atr"] = 1.2
    assert classify_context(row, T)["context_state"] == "MATURE_OR_EXHAUSTED"


def test_pullback_can_wait_for_realignment() -> None:
    first = aligned_row()
    first.update({"delay_minutes": 0, "dir1m__return_3_bps": -3.0, "dir5m__return_1_bps": -4.0, "dir5m__taker_imbalance": -0.2})
    second = aligned_row()
    second["delay_minutes"] = 1
    selected, execute, action, state = choose_episode_decision(
        pd.DataFrame([first, second]), T,
        {"maximum_wait_minutes": 3, "enter_states": ["TREND_CONTINUATION"], "wait_states": ["PULLBACK_WITHIN_TREND"]},
    )
    assert execute
    assert action == "ENTER_AFTER_1M"
    assert state == "PULLBACK_WITHIN_TREND"
    assert selected["delay_minutes"] == 1

