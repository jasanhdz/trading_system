from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aegis.research.intrabar_wave_w3 import (
    assert_probability_contract,
    barrier_return,
    directional_clv,
    directional_excursions,
    episode_weights,
    future_giveback_before_new_extreme,
    profit_capture_ratio,
    stable_wave_episode_id,
)


def test_wave_identity_is_stable_and_side_specific() -> None:
    first = stable_wave_episode_id("ADAUSDT", "LONG", 123)
    assert first == stable_wave_episode_id("ADAUSDT", "LONG", 123)
    assert first != stable_wave_episode_id("ADAUSDT", "SHORT", 123)


def test_directional_clv_is_symmetric() -> None:
    assert directional_clv(10, 12, 8, 11, 1) == pytest.approx(0.75)
    assert directional_clv(10, 12, 8, 11, -1) == pytest.approx(0.25)


def test_excursions_are_symmetric() -> None:
    long = directional_excursions(100, [102], [99], 1)
    short = directional_excursions(100, [101], [98], -1)
    assert long[0][0] == pytest.approx(short[0][0])
    assert long[1][0] == pytest.approx(short[1][0])


def test_barrier_resolution_is_adverse_first() -> None:
    gross, outcome, mfe, mae = barrier_return(
        100, 1, [101], [99], [100], 1, 0.5, 0.25
    )
    assert outcome == -1
    assert gross == pytest.approx(-0.0025)
    assert mfe == pytest.approx(0.01)
    assert mae == pytest.approx(0.01)


def test_giveback_target_checks_adverse_before_new_extreme() -> None:
    result = future_giveback_before_new_extreme(
        peak_favorable=0.02,
        current_favorable=0.018,
        future_favorable_highs=[0.023],
        future_favorable_lows=[0.016],
        atr_fraction=0.01,
    )
    assert result == 1


def test_episode_weights_sum_to_episode_count() -> None:
    frame = pd.DataFrame({"wave_episode_id": ["a", "a", "b"]})
    assert episode_weights(frame).sum() == pytest.approx(2.0)


def test_profit_capture_ratio_is_bounded() -> None:
    assert profit_capture_ratio(0.5, 1.0) == pytest.approx(0.5)
    assert profit_capture_ratio(2.0, 1.0) == pytest.approx(1.0)
    assert profit_capture_ratio(-1.0, 1.0) == 0.0


def test_probability_contract_rejects_invalid_values() -> None:
    assert_probability_contract(np.array([0.0, 0.5, 1.0]))
    with pytest.raises(ValueError, match="AEGIS_W3_PROBABILITY_CONTRACT_INVALID"):
        assert_probability_contract(np.array([1.1]))
