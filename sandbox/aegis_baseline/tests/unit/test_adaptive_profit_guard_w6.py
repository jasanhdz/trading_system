import pytest

from aegis.research.adaptive_profit_guard_w6 import (
    choose_best_action,
    simulate_guard,
    simulate_simple_baseline,
    trailing_activation_atr,
    validate_feature_contract,
)


def episode(side="LONG", highs=None, lows=None, closes=None, atrs=None):
    closes = closes or [100.5, 101.0, 100.8]
    return {
        "side": side,
        "simulated_entry": 100.0,
        "entry_atr": float((atrs or [0.1])[0]),
        "path_high": highs or [100.6, 101.2, 101.0],
        "path_low": lows or [100.1, 100.7, 100.4],
        "path_close": closes,
        "path_atr": atrs or [0.1] * len(closes),
    }


def test_feature_contract_rejects_future_information():
    validate_feature_contract(["peak_mfe_atr", "volume_ratio_20"])
    with pytest.raises(ValueError, match="FUTURE_FEATURE_PROHIBITED"):
        validate_feature_contract(["peak_mfe_atr", "target_future_giveback_atr"])


def test_activation_is_mapped_from_roe_without_leverage_in_objective():
    assert trailing_activation_atr(100.0, 1.0, 0.15, 20.0) == pytest.approx(0.75)


def test_new_trailing_stop_is_only_effective_on_next_bar():
    result = simulate_guard(episode(), atr_multiplier=1.5, cost_bps=14)
    assert result.exit_reason == "PROFIT_GUARD"
    assert result.exit_bar == 3


def test_short_is_symmetric_and_guard_protects_profit():
    result = simulate_guard(
        episode("SHORT", highs=[99.9, 99.3, 99.6], lows=[99.4, 98.8, 99.0], closes=[99.5, 99.0, 99.4]),
        atr_multiplier=1.5,
        cost_bps=14,
    )
    assert result.exit_reason == "PROFIT_GUARD"
    assert result.gross_return > 0


def test_hard_stop_is_unchanged():
    result = simulate_guard(
        episode(highs=[100.1], lows=[97.5], closes=[98.0], atrs=[1.0]),
        atr_multiplier=2.25,
        cost_bps=14,
    )
    assert result.exit_reason == "COMMON_HARD_STOP"
    assert result.gross_return == pytest.approx(-0.02)


def test_best_action_prefers_normal_on_exact_tie():
    normal = simulate_guard(episode(highs=[100.1], lows=[99.9], closes=[100.0]), atr_multiplier=1.5, cost_bps=14)
    assert choose_best_action({"DEFENSIVE": normal, "NORMAL": normal, "EXPANSION": normal}) == "NORMAL"


def test_simple_baseline_does_not_activate_before_profit_gate():
    result = simulate_simple_baseline(
        episode(highs=[100.1, 100.2], lows=[99.9, 99.8], closes=[100.0, 100.1], atrs=[1.0, 1.0]),
        policy="TIME_EXIT",
        parameter=1,
        gate_atr=0.25,
        cost_bps=14,
    )
    assert result.exit_reason == "BOUNDED_HOLD"
