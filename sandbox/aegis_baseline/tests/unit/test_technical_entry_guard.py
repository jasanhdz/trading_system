from aegis.research.technical_entry_guard import assess_technical_entry


BASE = {
    "rsi_remaining_room_critical": 30.0,
    "ema25_extension_atr_critical": 1.0,
    "prior_move_atr_critical": 1.5,
    "ema25_opposition_atr": -0.75,
    "directional_return_1m_bps_opposed": -5.0,
    "directional_return_5m_bps_opposed": -5.0,
    "taker_imbalance_opposed": -0.10,
    "favorable_space_atr_critical": 0.75,
    "atr_percentile_shock": 0.90,
    "volume_ratio_shock": 2.50,
    "path_efficiency_disorder": 0.35,
    "minimum_opposition_votes": 3,
}


def test_single_rsi_extreme_does_not_block() -> None:
    result = assess_technical_entry({"dir5m__rsi6_remaining_room": 20.0}, BASE)
    assert result["action"] == "ENTER"


def test_rsi_plus_extension_blocks_as_exhausted() -> None:
    result = assess_technical_entry(
        {"dir5m__rsi6_remaining_room": 20.0, "dir5m__ema25_extension_atr": 1.2}, BASE
    )
    assert result["action"] == "SKIP"
    assert result["reason"] == "SKIP_EXHAUSTED"


def test_opposition_requires_compound_evidence() -> None:
    row = {
        "dir5m__ema25_extension_atr": -1.0,
        "dir1m__return_3_bps": -8.0,
        "dir1m__taker_imbalance": -0.2,
    }
    result = assess_technical_entry(row, BASE)
    assert result["action"] == "SKIP"
    assert result["reason"] == "SKIP_OPPOSED"


def test_no_space_requires_extension_or_maturity() -> None:
    clean = assess_technical_entry({"dir5m__favorable_space_atr": 0.5}, BASE)
    blocked = assess_technical_entry(
        {"dir5m__favorable_space_atr": 0.5, "dir5m__prior_move_6_atr": 1.8}, BASE
    )
    assert clean["action"] == "ENTER"
    assert blocked["reason"] == "SKIP_NO_SPACE"

