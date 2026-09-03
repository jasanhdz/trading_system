import pytest

from aegis.research.opportunity_direction_w7 import (
    opportunity_path_outcomes,
    stable_signal_id,
    validate_opportunity_features,
)


def test_opportunity_magnitude_is_direction_neutral():
    common = dict(entry=100.0, highs=[101.0, 100.5], lows=[99.5, 98.0], closes=[100.0, 99.0], cost_bps=14)
    long = opportunity_path_outcomes(**common, frozen_direction="LONG")
    short = opportunity_path_outcomes(**common, frozen_direction="SHORT")
    assert long["opportunity_magnitude_bps"] == pytest.approx(200.0)
    assert short["opportunity_magnitude_bps"] == pytest.approx(200.0)
    assert long["directional_net_return_bps"] == pytest.approx(-114.0)
    assert short["directional_net_return_bps"] == pytest.approx(86.0)


def test_feature_contract_prohibits_side_and_future_information():
    validate_opportunity_features(["atr_12", "volume_ratio_6_24"])
    with pytest.raises(ValueError, match="PROHIBITED"):
        validate_opportunity_features(["atr_12", "side_ret_3"])
    with pytest.raises(ValueError, match="PROHIBITED"):
        validate_opportunity_features(["atr_12", "future_mfe"])


def test_signal_identity_includes_frozen_direction():
    left = stable_signal_id("BTCUSDT", "2026-01-01T00:00:00Z", "SHORT")
    right = stable_signal_id("BTCUSDT", "2026-01-01T00:00:00Z", "LONG")
    assert left != right


def test_invalid_path_fails_closed():
    with pytest.raises(ValueError, match="PATH_INPUT_INVALID"):
        opportunity_path_outcomes(entry=0, highs=[1], lows=[1], closes=[1], frozen_direction="SHORT", cost_bps=14)
