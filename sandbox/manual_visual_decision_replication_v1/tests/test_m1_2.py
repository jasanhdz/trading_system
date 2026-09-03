from __future__ import annotations

import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sandbox/manual_visual_decision_replication_v1/src"))

from mvdr_v1.m1 import Bar, _aggregate
from mvdr_v1.m1_2 import (
    ProspectiveCorridorShadowRecorder,
    btc_context,
    causal_zone_width_bps,
    compact_corridor_frame,
    corridor_clarity,
    decide_compact_action,
    level_respect_score,
    make_zone,
    movement_score,
    nearest_relevant_zones,
    reaction_score,
    volatility_shock,
)


def bars(start_price: float = 1.0, slope: float = 0.00005, count: int = 260) -> list[Bar]:
    start = datetime(2026, 8, 24, 20, tzinfo=timezone.utc)
    result = []
    for index in range(count):
        open_price = start_price + slope * index
        close = open_price + slope * 0.4
        result.append(Bar(start + timedelta(minutes=index), open_price, max(open_price, close) + 0.0008, min(open_price, close) - 0.0008, close, 1000 + index, 500 + index / 2))
    return result


def test_support_zone_is_causal_and_around_center() -> None:
    source = bars()
    zones = nearest_relevant_zones(source, source[-1].close)
    zone = zones["support"]["zone"]
    assert zone["lower_bound"] < zone["center"] < zone["upper_bound"]


def test_resistance_zone_is_causal_and_around_center() -> None:
    source = bars()
    zones = nearest_relevant_zones(source, source[-1].close)
    zone = zones["resistance"]["zone"]
    assert zone["lower_bound"] < zone["center"] < zone["upper_bound"]


def test_zone_width_uses_past_volatility() -> None:
    source = bars()
    assert causal_zone_width_bps(source) > 0
    assert causal_zone_width_bps(source[:-1]) > 0


def test_respect_score_is_bounded() -> None:
    source = bars()
    score = level_respect_score(source, make_zone(source[-20].low, 20), "support")
    assert 0 <= score["score"] <= 1


def test_touch_counting_records_contacts() -> None:
    source = bars(slope=0)
    result = level_respect_score(source, make_zone(1.0, 20), "support")
    assert result["touch_count"] > 0


def test_break_penalty_reduces_integrity() -> None:
    source = bars(slope=0)
    zone = make_zone(1.01, 10)
    result = level_respect_score(source, zone, "support")
    assert result["break_count"] > 0
    assert result["components"]["break_integrity_score"] < 1


def test_corridor_clarity_is_bounded() -> None:
    source = bars()
    zones = nearest_relevant_zones(source, source[-1].close)
    result = corridor_clarity(source, zones, source[-1].close)
    assert 0 <= result["score"] <= 1


def test_no_clear_corridor_when_level_respect_is_low() -> None:
    source = bars()
    zones = nearest_relevant_zones(source, source[-1].close)
    zones["support"]["respect"]["score"] = 0.1
    assert not corridor_clarity(source, zones, source[-1].close)["clear"]


def test_room_to_support_and_resistance() -> None:
    source = bars()
    zones = nearest_relevant_zones(source, source[-1].close)
    corridor = corridor_clarity(source, zones, source[-1].close)
    assert corridor["room_to_support_bps"] >= 0
    assert corridor["room_to_resistance_bps"] >= 0


def test_travel_to_support() -> None:
    source = bars(slope=-0.0001)
    result = movement_score(source, source[-1].close - 0.01, True, 12)
    assert result["score"] > 0.5


def test_travel_to_resistance() -> None:
    source = bars(slope=0.0001)
    result = movement_score(source, source[-1].close + 0.01, False, 12)
    assert result["score"] > 0.5


def test_support_reaction_score_is_bounded() -> None:
    source = bars()
    result = reaction_score(source, make_zone(source[-1].close, 20), "support")
    assert 0 <= result["score"] <= 1


def test_resistance_rejection_score_is_bounded() -> None:
    source = bars()
    result = reaction_score(source, make_zone(source[-1].close, 20), "resistance")
    assert 0 <= result["score"] <= 1


def test_short_to_support_decision() -> None:
    assert decide_compact_action(True, False, False, 0.1, 0.1, 0.8, 0.2)[0] == "SHORT_TO_SUPPORT"


def test_long_from_support_decision() -> None:
    assert decide_compact_action(True, True, False, 0.8, 0.1, 0.2, 0.2)[0] == "LONG_FROM_SUPPORT"


def test_long_to_resistance_decision() -> None:
    assert decide_compact_action(True, False, False, 0.1, 0.1, 0.2, 0.8)[0] == "LONG_TO_RESISTANCE"


def test_short_from_resistance_decision() -> None:
    assert decide_compact_action(True, False, True, 0.1, 0.8, 0.2, 0.2)[0] == "SHORT_FROM_RESISTANCE"


def test_btc_strongly_opposing_veto_state() -> None:
    source = bars(100.0, 0.0)
    row = source[-1]
    source[-1] = Bar(row.open_at, 120.0, 120.1, 119.9, 120.0, row.volume, row.buy_volume)
    assert btc_context(source, "SHORT_TO_SUPPORT")["state"] == "STRONGLY_OPPOSING"


def test_volatility_shock_veto_state() -> None:
    source = bars(slope=0)
    last = source[-1]
    source[-1] = Bar(last.open_at, 1.0, 1.2, 0.8, 1.0, last.volume, last.buy_volume)
    assert volatility_shock(source)["shock"]


def test_no_trade_abstention() -> None:
    assert decide_compact_action(False, False, False, 0.9, 0.9, 0.9, 0.1)[0] == "NO_TRADE"
    assert decide_compact_action(True, False, False, 0.1, 0.1, 0.55, 0.52)[0] == "NO_TRADE"


def test_no_future_candles() -> None:
    sui, btc = bars(), bars(100.0, 0.005)
    decision = sui[-2].open_at + timedelta(minutes=1)
    expected = compact_corridor_frame(decision, sui, btc)
    mutated = copy.deepcopy(sui)
    future = mutated[-1]
    mutated[-1] = Bar(future.open_at, 999, 1000, 998, 999, 1e9, 1e9)
    assert compact_corridor_frame(decision, mutated, btc) == expected


def test_partial_mtf_is_causal() -> None:
    source = bars()[-7:]
    assert _aggregate(source, 5)[-1].close == source[-1].close


def test_frame_is_deterministic() -> None:
    decision = bars()[-1].open_at + timedelta(minutes=1)
    assert compact_corridor_frame(decision, bars(), bars(100.0, 0.005)) == compact_corridor_frame(decision, bars(), bars(100.0, 0.005))


def test_shadow_recorder_is_append_only(tmp_path: Path) -> None:
    frame = compact_corridor_frame(bars()[-1].open_at + timedelta(minutes=1), bars(), bars(100.0, 0.005))
    recorder = ProspectiveCorridorShadowRecorder(tmp_path)
    record = recorder.append(frame)
    assert record["orders_enabled"] is False
    day = frame["decision_at_utc"][:10]
    assert (tmp_path / f"{day}.jsonl").exists()
    assert json.loads((tmp_path / f"{day}.jsonl").read_text().splitlines()[0])["outcome_known"] is False
    with pytest.raises(RuntimeError, match="APPEND_ONLY"):
        recorder.append(frame)
