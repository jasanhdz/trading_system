from __future__ import annotations

from aegis.domain import TradeSide
from scripts.build_regime_aware_directional_v6_dataset import committee_observation


def test_committee_observation_preserves_real_votes_without_fabrication() -> None:
    batch = {
        "results": {
            "ADAUSDT": {
                "selected": True,
                "candidate": {"side": "SHORT"},
                "predictions": [
                    {"side": "SHORT"},
                    {"side": "SHORT"},
                    {"side": "NO_TRADE"},
                ],
            }
        }
    }
    observation = committee_observation(batch, "ADAUSDT")
    assert observation.action == "SHORT"
    assert observation.short_votes == 2
    assert observation.neutral_votes == 1
    assert observation.long_votes == 0


def test_unselected_candidate_is_hold_but_keeps_votes() -> None:
    batch = {
        "results": {
            "ADAUSDT": {
                "selected": False,
                "candidate": {"side": "LONG"},
                "predictions": [{"side": "LONG"}, {"side": "NO_TRADE"}],
            }
        }
    }
    observation = committee_observation(batch, "ADAUSDT")
    assert observation.action == "HOLD"
    assert observation.long_votes == 1
    assert observation.neutral_votes == 1


def test_committee_observation_normalizes_real_trade_side_enums() -> None:
    batch = {
        "results": {
            "BTCUSDT": {
                "selected": True,
                "candidate": {"side": TradeSide.SHORT},
                "predictions": [{"side": TradeSide.SHORT}],
            }
        }
    }
    observation = committee_observation(batch, "BTCUSDT")
    assert observation.action == "SHORT"
    assert observation.short_votes == 1
    assert observation.long_votes == 0
