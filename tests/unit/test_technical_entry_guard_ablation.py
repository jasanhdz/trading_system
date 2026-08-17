from aegis.research.technical_entry_guard_ablation import factor_state, policy_skips


THRESHOLDS = {
    "rsi_remaining_room": 30.0,
    "ema25_opposed_atr": 0.0,
    "ema25_extended_atr": 1.0,
    "prior_move_atr": 1.5,
    "favorable_space_atr": 0.75,
    "price_return_1m_bps": -5.0,
    "price_return_5m_bps": -5.0,
    "taker_imbalance": -0.10,
    "atr_percentile": 0.90,
    "volume_ratio": 2.50,
    "path_efficiency": 0.35,
}


def test_rsi_and_structure_policy_requires_both_factors() -> None:
    row = {"dir5m__rsi6_remaining_room": 20.0, "dir5m__favorable_space_atr": 0.5}
    policies = policy_skips(factor_state(row, THRESHOLDS))
    assert policies["rsi_only"]
    assert policies["structure_only"]
    assert policies["rsi_and_structure"]


def test_ema_only_requires_both_timeframes_opposed() -> None:
    mixed = factor_state(
        {"dir5m__ema25_extension_atr": -0.2, "dir15m__ema25_extension_atr": 0.1}, THRESHOLDS
    )
    opposed = factor_state(
        {"dir5m__ema25_extension_atr": -0.2, "dir15m__ema25_extension_atr": -0.1}, THRESHOLDS
    )
    assert not mixed["ema_opposed"]
    assert opposed["ema_opposed"]


def test_price_and_flow_is_compound() -> None:
    row = {
        "dir1m__return_3_bps": -8.0,
        "dir5m__return_1_bps": -6.0,
        "dir1m__taker_imbalance": -0.2,
        "dir5m__taker_imbalance": -0.3,
    }
    policies = policy_skips(factor_state(row, THRESHOLDS))
    assert policies["price_and_flow"]

