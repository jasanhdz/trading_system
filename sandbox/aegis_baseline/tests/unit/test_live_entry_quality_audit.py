from aegis.research.live_entry_quality_audit import classify_entry, stable_trade_hash


CONFIG = {
    "good_min_mfe_bps": 28.0,
    "good_max_mae_bps": 75.0,
    "good_min_mfe_mae_ratio": 1.25,
    "bad_low_mfe_bps": 14.0,
    "bad_high_mae_bps": 100.0,
    "bad_extreme_mae_bps": 200.0,
    "bad_max_mfe_mae_ratio": 0.50,
    "bad_extreme_max_mfe_mae_ratio": 0.75,
}


def test_classifies_clean_good_entry() -> None:
    assert classify_entry(pnl_usdt=1.0, mfe_bps=100.0, mae_bps=30.0, config=CONFIG) == "GOOD_CLEAN_ENTRY"


def test_classifies_losing_adverse_entry() -> None:
    assert classify_entry(pnl_usdt=-1.0, mfe_bps=5.0, mae_bps=150.0, config=CONFIG) == "BAD_ENTRY"


def test_profitable_but_high_mae_path_is_not_clean() -> None:
    assert classify_entry(pnl_usdt=1.0, mfe_bps=80.0, mae_bps=120.0, config=CONFIG) == "MIXED_OR_EXIT_DEPENDENT"


def test_trade_hash_is_stable_and_non_plaintext() -> None:
    value = stable_trade_hash("trade-1")
    assert value == stable_trade_hash("trade-1")
    assert value != "trade-1"
    assert len(value) == 64
