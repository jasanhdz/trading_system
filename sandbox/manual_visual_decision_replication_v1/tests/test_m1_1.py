from __future__ import annotations

import copy
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sandbox/manual_visual_decision_replication_v1/src"))

from mvdr_v1.m1 import Bar, _aggregate
from mvdr_v1.m1_1 import (
    STATE_ACTION,
    _distance_bin,
    _shuffle_labels,
    _shuffle_states,
    _training_rows,
    approach_components,
    apply_entry_labels,
    build_matched_controls,
    causal_levels,
    choose_state,
    deceleration_components,
    level_state_frame,
    manual_state_annotations,
    reaction_components,
)


def market(start_price: float = 1.0, slope: float = 0.0001, count: int = 180) -> list[Bar]:
    start = datetime(2026, 8, 25, tzinfo=timezone.utc) - timedelta(hours=3)
    rows = []
    for index in range(count):
        open_price = start_price + index * slope
        close = open_price + slope * 0.5
        rows.append(Bar(start + timedelta(minutes=index), open_price, max(open_price, close) + 0.001, min(open_price, close) - 0.001, close, 1000 + index, 500 + index / 2))
    return rows


def family_scores(winner: str) -> dict[str, float]:
    result = {name: 0.2 for name in ("SHORT_TO_SUPPORT", "LONG_FROM_SUPPORT", "LONG_TO_RESISTANCE", "SHORT_FROM_RESISTANCE")}
    result[winner] = 0.8
    return result


def test_causal_support_is_not_above_price() -> None:
    bars = market()
    levels = causal_levels(bars, bars[-1].close)
    assert levels["support_price"] <= bars[-1].close


def test_causal_resistance_is_not_below_price() -> None:
    bars = market()
    levels = causal_levels(bars, bars[-1].close)
    assert levels["resistance_price"] >= bars[-1].close
    assert levels["support_price"] < levels["resistance_price"]


def test_distance_calculations_are_bps() -> None:
    frame = level_state_frame(market()[-1].open_at + timedelta(minutes=1), market(), market(100.0, 0.01))
    expected = (frame["current_price"] - frame["support_price"]) / frame["current_price"] * 10_000
    assert frame["distance_to_support_bps"] == expected


def test_position_between_levels_raw() -> None:
    frame = level_state_frame(market()[-1].open_at + timedelta(minutes=1), market(), market(100.0, 0.01))
    expected = (frame["current_price"] - frame["support_price"]) / (frame["resistance_price"] - frame["support_price"])
    assert frame["position_between_levels_raw"] == expected


def test_approaching_support_scores_down_path_higher() -> None:
    bars = market(slope=-0.0002)
    down = approach_components(bars, bars[-1].close - 0.01, True, 15)
    assert down["score"] > 0.5 and down["velocity_bps_per_min"] > 0


def test_approaching_resistance_scores_up_path_higher() -> None:
    bars = market(slope=0.0002)
    up = approach_components(bars, bars[-1].close + 0.01, False, 15)
    assert up["score"] > 0.5 and up["velocity_bps_per_min"] > 0


def test_deceleration_support_detects_contracting_decline() -> None:
    bars = market(slope=-0.0002)
    result = deceleration_components(bars, True)
    assert 0 <= result["score"] <= 1


def test_deceleration_resistance_detects_contracting_rise() -> None:
    bars = market(slope=0.0002)
    result = deceleration_components(bars, False)
    assert 0 <= result["score"] <= 1


def test_reaction_from_support_is_causal() -> None:
    bars = market()
    support = bars[-1].close - 0.0005
    result = reaction_components(bars, support, bars[-1].close + 0.02)
    assert result["reaction_support_score"] >= 0


def test_reaction_from_resistance_is_causal() -> None:
    bars = market()
    result = reaction_components(bars, bars[-1].close - 0.02, bars[-1].close + 0.0005)
    assert result["reaction_resistance_score"] >= 0


def test_short_to_support_state() -> None:
    assert choose_state(family_scores("SHORT_TO_SUPPORT"))[0] == "TOWARD_SUPPORT"
    assert STATE_ACTION["TOWARD_SUPPORT"] == "SHORT"


def test_long_from_support_state() -> None:
    assert choose_state(family_scores("LONG_FROM_SUPPORT"))[0] == "AT_SUPPORT"
    assert STATE_ACTION["AT_SUPPORT"] == "LONG"


def test_long_to_resistance_state() -> None:
    assert choose_state(family_scores("LONG_TO_RESISTANCE"))[0] == "TOWARD_RESISTANCE"
    assert STATE_ACTION["TOWARD_RESISTANCE"] == "LONG"


def test_short_from_resistance_state() -> None:
    assert choose_state(family_scores("SHORT_FROM_RESISTANCE"))[0] == "AT_RESISTANCE"
    assert STATE_ACTION["AT_RESISTANCE"] == "SHORT"


def test_no_future_candles() -> None:
    sui, btc = market(), market(100.0, 0.01)
    decision = sui[-2].open_at + timedelta(minutes=1)
    baseline = level_state_frame(decision, sui, btc)
    mutated = copy.deepcopy(sui)
    mutated[-1] = Bar(mutated[-1].open_at, 999, 1000, 998, 999, 1e9, 1e9)
    assert level_state_frame(decision, mutated, btc) == baseline


def test_partial_mtf_uses_only_supplied_completed_candles() -> None:
    bars = market()[-7:]
    partial = _aggregate(bars, 5)[-1]
    assert partial.close == bars[-1].close


def test_matched_negatives_are_not_bad_trade_labels() -> None:
    frames = [level_state_frame(market()[index].open_at + timedelta(minutes=1), market()[: index + 1], market(100.0, 0.01)[: index + 1]) for index in range(120, 150)]
    manifest = [{"manual_trade_id": "T1", "entry_at_utc": frames[0]["decision_at_utc"], "side": frames[0]["potential_action"]}]
    rows = apply_entry_labels(frames, manifest, [])
    anchor = rows[0]
    for row in rows[7:]:
        row["potential_action"] = anchor["potential_action"]
        row["model_inferred_family"] = anchor["model_inferred_family"]
        row["distance_support_bin"] = anchor["distance_support_bin"]
        row["distance_resistance_bin"] = anchor["distance_resistance_bin"]
        row["volatility_safety_score"] = anchor["volatility_safety_score"]
    controls = build_matched_controls(rows, manifest)
    assert controls and all(row["label"] == "NOT_SELECTED_BY_TRADER" and not row["selected"] for row in controls)


def test_loto_isolation_excludes_heldout_positive() -> None:
    candidates = [{"manual_trade_id": "A", "selected": True, "session": "s1", "features": {"level.x": 1}}, {"manual_trade_id": "B", "selected": True, "session": "s2", "features": {"level.x": 2}}]
    controls = [{"matched_to_trade_id": "B", "selected": False, "session": "s2", "candidate_id": "c", "features": {"level.x": 0}}]
    train = _training_rows(candidates, controls, "A", False)
    assert all(row.get("manual_trade_id") != "A" for row in train)


def test_session_isolation_excludes_same_session() -> None:
    candidates = [{"manual_trade_id": "A", "selected": True, "session": "s1", "features": {"level.x": 1}}, {"manual_trade_id": "B", "selected": True, "session": "s2", "features": {"level.x": 2}}]
    controls = [{"matched_to_trade_id": "B", "selected": False, "session": "s2", "candidate_id": "c", "features": {"level.x": 0}}]
    train = _training_rows(candidates, controls, "A", True)
    assert all(row["session"] != "s1" for row in train)


def test_label_shuffle_changes_labels_and_requires_new_manifest() -> None:
    frames = [level_state_frame(market()[index].open_at + timedelta(minutes=1), market()[: index + 1], market(100.0, 0.01)[: index + 1]) for index in range(120, 140)]
    for frame in frames:
        frame["session"] = "2026-08-25"
        frame["potential_action"] = "LONG"
        frame["candidate_id"] = frame["frame_id"]
        frame["selected"] = False
        frame["manual_trade_id"] = None
        frame["manual_side"] = None
    manifest = [{"manual_trade_id": "A", "entry_at_utc": "2026-08-25T00:00:00Z", "side": "LONG"}]
    shuffled, new_manifest = _shuffle_labels(frames, manifest, random.Random(1), 0)
    assert len(new_manifest) == 1 and sum(row["selected"] for row in shuffled) == 1


def test_state_shuffle_changes_state_payload() -> None:
    rows = [{"session": "s", "potential_action": "LONG", "model_inferred_state": "AT_SUPPORT", "model_inferred_family": "LONG_FROM_SUPPORT", "features": {"state.short_to_support": i, "state.long_from_support": i + 1, "state.long_to_resistance": i + 2, "state.short_from_resistance": i + 3, "state.confidence": i + 4}} for i in range(10)]
    shuffled = _shuffle_states(rows, random.Random(2))
    assert [row["features"]["state.confidence"] for row in shuffled] != [row["features"]["state.confidence"] for row in rows]


def test_determinism_and_distance_bins() -> None:
    decision = market()[-1].open_at + timedelta(minutes=1)
    assert level_state_frame(decision, market(), market(100.0, 0.01)) == level_state_frame(decision, market(), market(100.0, 0.01))
    assert [_distance_bin(value) for value in (0, 10, 25, 50, 100)] == ["0_10", "10_25", "25_50", "50_100", "100_PLUS"]


def test_manual_annotation_does_not_label_other_eight() -> None:
    manifest = [{"manual_trade_id": f"MVDR-M1-{index:02d}"} for index in range(1, 10)]
    result = manual_state_annotations(manifest)
    assert result["manual_annotation_count"] == 1
    assert sum(row["annotation_status"] == "UNANNOTATED" for row in result["annotations"]) == 8
