from __future__ import annotations

import gzip
from datetime import datetime, timedelta, timezone

import pytest

from aegis_range_v1.candidates import candidate_grid
from aegis_range_v1.models import Candle5m, FillEvent, LevelSnapshot, PendingEntry, RangePair
from aegis_range_v1.numeric import iso_utc_millis
from aegis_range_v1.sweep_reclaim_discovery import (
    DiscoveryLifecycle,
    FLAGS,
    FrozenRange,
    SweepReclaimMachine,
    aggregate_economics,
    assign_opportunity_weights,
    canonical_sweep_opportunity_id,
    classify_edge,
    conservative_path,
    deterministic_gzip_jsonl,
    first_passage,
    generate_candidate_mappings,
    match_v1,
    reclaim_matches,
    reproducible_sample,
    scenario_economics,
    stop_recovery,
    sweep_depth,
    synchronized_block_bootstrap,
    verify_authority,
    _canonical_contract_means,
    _decorate_entries,
    _diagnostics,
)
from aegis_range_v1.lifecycle import RangeLifecycleV1

ORIGIN = datetime(2024, 6, 1, tzinfo=timezone.utc)


def bar(index: int, *, open_: float = 101, high: float = 104, low: float = 100, close: float = 101, segment: int = 0) -> Candle5m:
    opened = ORIGIN + timedelta(minutes=5 * index)
    return Candle5m("BTCUSDT", opened, opened + timedelta(minutes=5), open_, high, low, close, 1, segment)


def snapshot(*, support: float = 100, resistance: float = 110, midpoint: float = 105, atr: float = 2, episode: str = "ep") -> LevelSnapshot:
    pair = RangePair("support", "resistance", support, resistance, midpoint, resistance - support, 2, 2, ORIGIN, ORIGIN)
    return LevelSnapshot(ORIGIN, pair, atr, episode, "range")


def frozen(side: str = "LONG", **kwargs) -> FrozenRange:
    snap = snapshot(**kwargs)
    return FrozenRange.from_snapshot("BTCUSDT", side, snap, ORIGIN - timedelta(hours=1))


def process(machine: SweepReclaimMachine, candle: Candle5m, snap: LevelSnapshot | None = None, post: str | None = "ep", **kwargs):
    snap = snapshot() if snap is None else snap
    return machine.process_close(candle, snap, ORIGIN - timedelta(hours=1), post, **kwargs)


def test_candidate_mapping_proves_384_ids_and_multiplicity():
    mapping = generate_candidate_mappings()
    assert len(mapping) == 24
    assert {len(value) for value in mapping.values()} == {16}
    assert {item for values in mapping.values() for item in values} == {f"C{i:03d}" for i in range(len(candidate_grid()))}


@pytest.mark.parametrize(
    ("side", "candle"),
    [
        ("LONG", bar(0, low=99.7, close=101)),
        ("SHORT", bar(0, open_=109, high=110.3, low=106, close=109)),
    ],
)
def test_long_short_s1(side, candle):
    rows = process(SweepReclaimMachine("s"), candle)
    selected = [row for row in rows if row["side"] == side]
    assert len(selected) == 1
    assert selected[0]["status"] == "RECLAIMED" and selected[0]["reclaim_type"] == "S1"


@pytest.mark.parametrize(
    ("side", "sweep", "reclaim"),
    [
        ("LONG", bar(0, low=99.7, close=99), bar(1, low=100, close=101)),
        ("SHORT", bar(0, high=110.3, low=106, close=111), bar(1, open_=109, high=110, low=106, close=109)),
    ],
)
def test_long_short_s2_first_matching_close(side, sweep, reclaim):
    machine = SweepReclaimMachine("s")
    assert process(machine, sweep) == []
    rows = process(machine, reclaim)
    selected = [row for row in rows if row["side"] == side]
    assert selected[0]["reclaim_type"] == "S2" and selected[0]["s2_delay_bars"] == 1


def test_no_reclaim_expires_on_exact_third_subsequent_close():
    machine = SweepReclaimMachine("s")
    process(machine, bar(0, low=99.7, close=99))
    process(machine, bar(1, low=99, close=99))
    process(machine, bar(2, low=99, close=99))
    rows = process(machine, bar(3, low=99, close=99))
    assert rows[0]["status"] == "NO_RECLAIM" and rows[0]["terminal_reason"] == "WINDOW_EXPIRED"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"episode_event": "CONFIRMED_BREAKOUT"}, "CONFIRMED_BREAKOUT"),
        ({"episode_event": "EXPIRED_48H"}, "EPISODE_EXPIRED_48H"),
        ({"episode_event": "PAIR_REPLACED"}, "EPISODE_REPLACED"),
        ({"episode_event": "AMPLITUDE"}, "AMPLITUDE_INVALIDATED"),
        ({"episode_event": "STRUCTURE_LOST"}, "STRUCTURE_LOSS"),
        ({"same_split": False}, "SPLIT_BOUNDARY"),
        ({"contiguous": False}, "DATA_GAP"),
    ],
)
def test_pending_cancellation_types(kwargs, reason):
    machine = SweepReclaimMachine("s")
    process(machine, bar(0, low=99.7, close=99))
    rows = process(machine, bar(1, low=100, close=101), **kwargs)
    assert rows[0]["terminal_reason"] == reason


def test_episode_end_cancels_and_midpoint_touch_is_conservative():
    machine = SweepReclaimMachine("s")
    process(machine, bar(0, low=99.7, close=99))
    assert process(machine, bar(1), post=None)[0]["terminal_reason"] == "EPISODE_ENDED"
    sweep_touch = process(SweepReclaimMachine("s"), bar(0, high=105, low=99.7, close=101))[0]
    assert sweep_touch["terminal_reason"] == "MIDPOINT_TOUCHED"


def test_no_duplicate_pending_and_causal_rearm_requires_later_inside_close():
    machine = SweepReclaimMachine("s")
    assert process(machine, bar(0, low=99.7, close=99)) == []
    assert process(machine, bar(1, low=99.6, close=99)) == []
    terminal = process(machine, bar(2, low=100, close=101))[0]
    assert terminal["reclaim_type"] == "S2"
    assert process(machine, bar(3, low=99.7, close=99)) == []
    process(machine, bar(4, low=100.1, close=101))
    assert process(machine, bar(5, low=99.7, close=101))[0]["reclaim_type"] == "S1"


def test_prior_snapshot_and_post_episode_are_both_required_no_lookahead():
    machine = SweepReclaimMachine("s")
    assert machine.process_close(bar(0, low=99.7), None, None, "ep") == []
    assert process(machine, bar(1, low=99.7), post="replacement") == []
    rows = process(machine, bar(2, low=99.7))
    assert rows[0]["support"] == 100 and rows[0]["causality_snapshot_is_prior_close"] is False


def test_future_extension_compares_independent_run_prefixes():
    prefix = [bar(0, low=99.7, close=99), bar(1, low=100.1, close=101)]
    future = [bar(index, high=200, low=1, close=50) for index in range(2, 10)]

    def run(candles):
        machine = SweepReclaimMachine("s")
        events = []
        for candle in candles:
            events.extend(process(machine, candle))
        return [{key: value for key, value in row.items() if not key.startswith("_")} for row in events if row["terminal_decision_at"] <= iso_utc_millis(prefix[-1].available_at)]

    assert run(prefix) == run(prefix + future)


def test_sweep_depth_fields_are_nonnegative_and_bins_frozen():
    total_atr, total_bps, bin_name, past_atr, past_bps = sweep_depth("LONG", bar(0, low=99.7), frozen())
    assert total_atr == pytest.approx(0.15) and total_bps == pytest.approx(30)
    assert bin_name == "0.10_TO_0.25" and past_atr == pytest.approx(0.05) and past_bps == pytest.approx(10)
    depth = sweep_depth("SHORT", bar(0, high=111.1), frozen("SHORT"))
    assert depth[0] >= 0 and depth[1] >= 0 and depth[2] == "GE_0.50"
    zero = frozen(atr=0)
    assert sweep_depth("LONG", bar(0, low=99), zero)[0] == 0


def test_reclaim_boundaries_are_strict_at_range_edge_inclusive_at_midpoint():
    value = frozen()
    assert not reclaim_matches("LONG", 100, value)
    assert reclaim_matches("LONG", 105, value)
    short = frozen("SHORT")
    assert reclaim_matches("SHORT", 105, short) and not reclaim_matches("SHORT", 110, short)


def test_canonical_id_excludes_outcomes_candidate_and_future():
    row = process(SweepReclaimMachine("s"), bar(0, low=99.7, close=101))[0]
    changed = {**row, "status": "NO_RECLAIM", "candidate_id": "C999", "future": 123}
    assert canonical_sweep_opportunity_id(row) == canonical_sweep_opportunity_id(changed)
    weighted = assign_opportunity_weights([row, changed])
    assert [item["unique_weight"] for item in weighted] == [0.5, 0.5]


def _pending(side="LONG", support=96, resistance=110, midpoint=105, atr=2):
    return PendingEntry("BTCUSDT", side, ORIGIN, ORIGIN, "ep", "range", ORIGIN - timedelta(hours=1), support, resistance, midpoint, atr, "RANGE", 1, None)


def test_raw_open_42bps_rr_and_same_split_are_delegated_to_frozen_lifecycle():
    candidate = candidate_grid()[0]
    lifecycle = RangeLifecycleV1(candidate)
    lifecycle.schedule_entry(_pending(midpoint=100.4))
    assert lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True) is None
    assert lifecycle.last_entry_cancel_reason == "TARGET_DISTANCE_LT_42_BPS"
    lifecycle.schedule_entry(_pending(support=80, midpoint=105))
    assert lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True) is None
    assert lifecycle.last_entry_cancel_reason == "REWARD_RISK_LT_1"
    lifecycle.schedule_entry(_pending())
    assert lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=False, episode_active=True) is None
    assert lifecycle.last_entry_cancel_reason == "OUTSIDE_SPLIT"


@pytest.mark.parametrize(
    ("consume", "reason"),
    [
        ({"open_at": ORIGIN, "raw_open": None, "same_split": True, "episode_active": True}, "MISSING_NEXT_BAR"),
        ({"open_at": ORIGIN, "raw_open": float("nan"), "same_split": True, "episode_active": True}, "INVALID_OPEN"),
        ({"open_at": ORIGIN + timedelta(minutes=5), "raw_open": 100, "same_split": True, "episode_active": True}, "NOT_NEXT_BAR_OPEN"),
        ({"open_at": ORIGIN, "raw_open": 100, "same_split": True, "episode_active": False}, "EPISODE_ENDED"),
        ({"open_at": ORIGIN, "raw_open": 95, "same_split": True, "episode_active": True}, "OPEN_OUTSIDE_RANGE"),
    ],
)
def test_entry_cancellation_types_are_fail_closed(consume, reason):
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    lifecycle.schedule_entry(_pending())
    assert lifecycle.consume_pending_entry(**consume) is None
    assert lifecycle.last_entry_cancel_reason == reason


def test_entry_expiry_and_target_not_favorable_cancellations():
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    expired = PendingEntry("BTCUSDT", "LONG", ORIGIN, ORIGIN, "ep", "range", ORIGIN - timedelta(hours=48), 96, 110, 105, 2, "RANGE", 1, None)
    lifecycle.schedule_entry(expired)
    assert lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True) is None
    assert lifecycle.last_entry_cancel_reason == "EPISODE_EXPIRED"
    lifecycle.schedule_entry(_pending(midpoint=99))
    assert lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True) is None
    assert lifecycle.last_entry_cancel_reason == "TARGET_NOT_FAVORABLE"


def test_lifecycle_is_adverse_first_when_stop_and_target_touch():
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    lifecycle.schedule_entry(_pending())
    position = lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True)
    event = lifecycle.process_position_open_and_intrabar(bar(0, high=110, low=90), include_open_gaps=False)
    assert position is not None and event.reason == "STOP"


def test_first_passage_progress_and_same_bar_adverse_first():
    result = first_passage(100, "LONG", 104, [bar(0, high=104, low=99.5)])
    assert result["midpoint_progress_first_bar"] == {"25": 0, "50": 0, "100": 0}
    assert result["favorable_adverse_matrix"]["F10_A10"] == "ADVERSE_FIRST"
    assert result["progress_adverse_matrix"]["P25_A10"] == "ADVERSE_FIRST"


def test_conservative_path_floors_excursions_and_terminal_stop():
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    lifecycle.schedule_entry(_pending())
    position = lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True)
    assert position is not None
    event = FillEvent("BTCUSDT", "LONG", ORIGIN, position.stop_at_entry, "STOP")
    path, passage = conservative_path(position, event, [bar(0, open_=100, high=120, low=90)])
    assert path["mfe"] == 0 and path["mae"] >= 0 and path["MAE_before_MFE"]
    assert passage["same_bar_policy"] == "ADVERSE_FIRST"
    assert all(value is None for value in passage["favorable_first_bar"].values())
    assert all(value is None for value in passage["midpoint_progress_first_bar"].values())


def test_target_gap_terminal_uses_target_price_not_gap_open():
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    lifecycle.schedule_entry(_pending())
    position = lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True)
    assert position is not None
    event = FillEvent("BTCUSDT", "LONG", ORIGIN, position.target_at_entry, "TARGET_GAP")
    gap_open_below = bar(0, open_=99, high=200, low=1)
    path, passage = conservative_path(position, event, [gap_open_below])
    direction = 1.0
    expected_mfe = max(0.0, direction * (position.target_at_entry - position.entry_fill) / position.entry_fill)
    assert path["mfe"] == pytest.approx(expected_mfe)
    assert path["mfe_bar_index"] == 0
    assert path["bars_to_MFE"] == 1
    assert path["time_to_MFE_minutes"] == 5


def test_stop_gap_terminal_uses_gap_open():
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    lifecycle.schedule_entry(_pending())
    position = lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True)
    assert position is not None
    event = FillEvent("BTCUSDT", "LONG", ORIGIN, 90.0, "STOP_GAP")
    path, passage = conservative_path(position, event, [bar(0, open_=90, high=120, low=89)])
    assert path["mae_bar_index"] == 0
    assert path["bars_to_MAE"] == 1
    assert path["time_to_MAE_minutes"] == 5


def test_short_target_gap_terminal_uses_target_price():
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    lifecycle.schedule_entry(_pending(side="SHORT", support=90, resistance=110, midpoint=105, atr=2))
    position = lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=108, same_split=True, episode_active=True)
    assert position is not None
    event = FillEvent("BTCUSDT", "SHORT", ORIGIN, position.target_at_entry, "TARGET_GAP")
    path, passage = conservative_path(position, event, [bar(0, open_=110, high=111, low=80)])
    direction = -1.0
    expected_mfe = max(0.0, direction * (position.target_at_entry - position.entry_fill) / position.entry_fill)
    assert path["mfe"] == pytest.approx(expected_mfe)


def test_short_stop_gap_terminal_uses_gap_open():
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    lifecycle.schedule_entry(_pending(side="SHORT", support=90, resistance=110, midpoint=105, atr=2))
    position = lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=108, same_split=True, episode_active=True)
    assert position is not None
    event = FillEvent("BTCUSDT", "SHORT", ORIGIN, 110.0, "STOP_GAP")
    path, passage = conservative_path(position, event, [bar(0, open_=110, high=111, low=80)])
    assert path["mae_bar_index"] == 0


def test_timing_elapsed_bars_and_minutes_are_one_based():
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    lifecycle.schedule_entry(_pending())
    position = lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True)
    assert position is not None
    event = FillEvent("BTCUSDT", "LONG", ORIGIN + timedelta(minutes=10), position.target_at_entry, "TARGET")
    bars = [bar(0, open_=100, high=102, low=99), bar(1, open_=101, high=103, low=100), bar(2, open_=102, high=104, low=101)]
    path, passage = conservative_path(position, event, bars)
    assert path["bars_to_MFE"] == 3
    assert path["time_to_MFE_minutes"] == 15
    assert path["mfe_bar_index"] == 2


def test_timing_entry_bar_is_first_completed_bar():
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    lifecycle.schedule_entry(_pending())
    position = lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True)
    assert position is not None
    event = FillEvent("BTCUSDT", "LONG", ORIGIN + timedelta(minutes=5), position.target_at_entry, "TARGET")
    path, passage = conservative_path(position, event, [bar(0, open_=100, high=106, low=99)])
    assert path["bars_to_MFE"] == 1
    assert path["time_to_MFE_minutes"] == 5


def test_timing_tie_records_first_occurrence():
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    lifecycle.schedule_entry(_pending())
    position = lifecycle.consume_pending_entry(open_at=ORIGIN, raw_open=100, same_split=True, episode_active=True)
    assert position is not None
    event = FillEvent("BTCUSDT", "LONG", ORIGIN + timedelta(minutes=10), position.target_at_entry, "TARGET")
    bars = [bar(0, open_=100, high=103, low=97), bar(1, open_=101, high=103, low=97)]
    path, passage = conservative_path(position, event, bars)
    assert path["bars_to_MAE"] == 1
    assert path["time_to_MAE_minutes"] == 5
    assert path["MAE_before_MFE"] is True


def test_stop_recovery_uses_first_following_bars_without_changing_stop():
    result = stop_recovery(100, 105, 95, "LONG", [bar(i, high=106 if i == 3 else 99, low=94) for i in range(24)])
    assert result["15"]["entry_recovered"] is False
    assert result["120"]["midpoint_recovered"] is True
    assert result["start_policy"] == "FIRST_COMPLETE_FOLLOWING_BAR"


def test_funding_interval_and_unit_notional_economics():
    events = ((0.001, 101.0),)
    long = scenario_economics("LONG", 100, 101, events)
    short = scenario_economics("SHORT", 100, 99, events)
    assert long["BASELINE"]["funding_return"] < 0 < short["BASELINE"]["funding_return"]
    assert set(long) == {"BASELINE", "STRESS_20", "STRESS_30"}


def economic_row(identifier, day, gross, net, *, side="LONG", reclaim="S1"):
    scenarios = {name: {"gross_return": gross, "net_return": net, "funding_return": 0.0, "entry_fill": 100, "exit_fill": 101, "fees": gross - net} for name in ("BASELINE", "STRESS_20", "STRESS_30")}
    return {"canonical_sweep_opportunity_id": identifier, "sweep_decision_at": iso_utc_millis(ORIGIN + timedelta(days=day)), "unique_weight": 1, "row_weight": 1, "side": side, "reclaim_type": reclaim, "exit_reason": "TARGET", "scenarios": scenarios}


def test_economic_classification_and_aggregation():
    assert classify_edge(0, 1, 2) == "NO_GROSS_EDGE"
    assert classify_edge(0.1, 0, 2) == "COST_LIMITED_SIGNAL"
    assert classify_edge(0.1, 0.05, 2) == "POTENTIAL_NET_EDGE"
    with pytest.raises(ValueError, match="PROFIT_FACTOR"):
        classify_edge(0.1, 0.05, 0.5)
    summary = aggregate_economics([economic_row("a", 0, 0.01, 0.005)])
    assert summary["BASELINE"]["decision_label"] == "POTENTIAL_NET_EDGE"
    assert summary["BASELINE"]["whitelist"] is False


def test_bootstrap_is_synchronized_deterministic_nearest_rank():
    rows = [economic_row(str(index), index * 7, 0.01 if index % 2 else -0.005, 0.005 if index % 2 else -0.01) for index in range(4)]
    first = synchronized_block_bootstrap(rows, draws=100, seed=20260825)
    assert first == synchronized_block_bootstrap(rows, draws=100, seed=20260825)
    assert first["percentile"] == "NEAREST_RANK" and first["synthetic_days"] == 365


def test_bootstrap_daily_overlapping_empty_blocks_weighted_contracts_and_order_independence():
    first_contract = economic_row("same", 0, 0.10, 0.08)
    second_contract = {**economic_row("same", 0, -0.10, -0.12), "unique_weight": 3}
    rows = [first_contract, second_contract]
    canonical = _canonical_contract_means(rows, ("gross_return", "net_return"), scenario="BASELINE")
    assert canonical[0]["gross_return"] == pytest.approx(-0.05)
    first = synchronized_block_bootstrap(rows, draws=50)
    second = synchronized_block_bootstrap(list(reversed(rows)), draws=50)
    assert first == second
    assert first["eligible_daily_starts"] == 360
    assert first["anchor_memberships"] == 7
    assert first["nonempty_blocks"] == 7 and first["empty_blocks"] == 353


def test_candidate_contract_weights_sum_one_and_rejections_stay_in_abstention():
    base = {"canonical_sweep_opportunity_id": "o", "candidate_id": "C000", "symbol": "BTCUSDT", "range_episode_id": "ep", "side": "LONG", "sweep_decision_at": iso_utc_millis(ORIGIN), "entry_status": "FILLED"}
    entries = [{**base, "candidate_id": f"C{index:03d}", "entry_status": "FILLED" if index < 2 else "REJECTED"} for index in range(4)]
    decorated = _decorate_entries(entries, [])
    assert sum(row["unique_weight"] for row in decorated) == 1
    assert {row["group_multiplicity"] for row in decorated} == {4}
    paths = [{**economic_row("o", 0, 0.01, 0.005), **{key: decorated[index][key] for key in ("candidate_id", "unique_weight", "row_weight")}} for index in range(2)]
    metrics = aggregate_economics(paths, "unique_weight", decorated)["BASELINE"]
    assert metrics["effective_eligible_opportunity_weight"] == 1
    assert metrics["effective_retained_opportunity_weight"] == 0.5
    assert metrics["abstention_rate"] == 0.5


def test_sidecar_train_boundary_disposes_pending_and_open_without_exit():
    pending_sidecar = DiscoveryLifecycle(candidate_grid()[0])
    opportunity = {"canonical_sweep_opportunity_id": "o", "reclaim_decision_at": iso_utc_millis(ORIGIN), "sweep_decision_at": iso_utc_millis(ORIGIN), "reclaim_type": "S1"}
    pending_sidecar.schedule(opportunity, frozen())
    pending, opened = pending_sidecar.finalize()
    assert pending["entry_status"] == "PURGED" and pending["entry_cancel_reason"] == "TRAIN_BOUNDARY" and opened is None

    open_sidecar = DiscoveryLifecycle(candidate_grid()[0])
    open_sidecar.schedule(opportunity, frozen())
    entry, closed, boundary = open_sidecar.process_bar(bar(0, open_=101, high=102, low=100.5), preceding_episode_active=True)
    assert entry["entry_status"] == "FILLED" and closed is None and boundary is None
    pending, opened = open_sidecar.finalize()
    assert pending is None and opened["outcome_status"] == "PURGED_OPEN_POSITION" and opened["censored"] is True


def test_sidecar_split_purges_open_position_without_fabricated_exit():
    sidecar = DiscoveryLifecycle(candidate_grid()[0])
    opportunity = {"canonical_sweep_opportunity_id": "o", "reclaim_decision_at": iso_utc_millis(ORIGIN), "sweep_decision_at": iso_utc_millis(ORIGIN), "reclaim_type": "S1"}
    sidecar.schedule(opportunity, frozen())
    entry, closed, boundary = sidecar.process_bar(bar(0, open_=101, high=102, low=100.5), preceding_episode_active=True)
    assert entry["entry_status"] == "FILLED" and closed is None and boundary is None
    pending, closed, boundary = sidecar.process_bar(bar(1, segment=1), preceding_episode_active=False, same_split=False)
    assert pending is None and closed is None
    assert boundary["purge_reason"] == "SPLIT_BOUNDARY" and boundary["censored"] is True


def test_sidecar_path_bars_are_not_duplicated_and_chronology_is_strict():
    sidecar = DiscoveryLifecycle(candidate_grid()[0])
    opportunity = {"canonical_sweep_opportunity_id": "o", "reclaim_decision_at": iso_utc_millis(ORIGIN), "sweep_decision_at": iso_utc_millis(ORIGIN), "reclaim_type": "S1"}
    sidecar.schedule(opportunity, frozen())
    entry, closed, _ = sidecar.process_bar(bar(0, open_=101, high=102, low=100.5), preceding_episode_active=True)
    assert entry["entry_at"] == iso_utc_millis(ORIGIN) and closed is None
    _, closed, _ = sidecar.process_bar(bar(1, open_=102, high=106, low=101), preceding_episode_active=True)
    position, event, _, held, _ = closed
    assert [item.open_time for item in held] == [ORIGIN, ORIGIN + timedelta(minutes=5)]
    assert position.entry_at < event.fill_at and len({item.open_time for item in held}) == len(held)


def test_required_diagnostics_expose_candidate_unique_retention_and_passage():
    entry = {"canonical_sweep_opportunity_id": "o", "candidate_id": "C000", "unique_weight": 1, "row_weight": 1}
    path = {**economic_row("o", 0, 0.01, 0.005), "candidate_id": "C000", "MFE_before_MAE": True, "MAE_before_MFE": False}
    passage = {**entry, "favorable_adverse_matrix": {"F20_A20": "FAVORABLE_FIRST", "F30_A20": "ADVERSE_FIRST"}, "midpoint_progress_first_bar": {"25": 0, "50": 1, "100": None}}
    diagnostics = _diagnostics([entry], [path], [passage])
    assert diagnostics["sample_sizes"]["unique_opportunities"] == 1
    assert diagnostics["primary_unique_view"]["economics"]["BASELINE"]["retention_rate"] == 1
    assert diagnostics["primary_unique_view"]["first_passage"]["F20_before_A20"] == 1


def test_v1_matching_has_no_outcome_condition_and_quota_guard():
    entry = {"candidate_id": "C001", "symbol": "BTCUSDT", "range_episode_id": "ep", "side": "LONG", "sweep_decision_at": iso_utc_millis(ORIGIN)}
    trade = {**entry, "entry_at": iso_utc_millis(ORIGIN - timedelta(minutes=5)), "exit_reason": "STOP"}
    assert match_v1(entry, [trade])["matched_view"] == "MATCHED"
    assert match_v1(entry, [{**trade, "side": "SHORT"}])["matched_view"] == "UNMATCHED"
    with pytest.raises(RuntimeError, match="QUOTA"):
        match_v1(entry, [trade, trade])


def test_deterministic_sample_and_gzip(tmp_path):
    assert reproducible_sample() == reproducible_sample()
    first, second = tmp_path / "a.gz", tmp_path / "b.gz"
    rows = [{"z": 1, "a": 2}]
    assert deterministic_gzip_jsonl(first, rows) == deterministic_gzip_jsonl(second, rows)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="ascii") as handle:
        assert handle.read() == '{"a":2,"z":1}\n'


def test_partition_and_immutable_output_guard_fail_closed(tmp_path, monkeypatch):
    with pytest.raises(PermissionError, match="PARTITION"):
        verify_authority(tmp_path, tmp_path / "out", {"TRAIN_ACCESS": "true", "CALIBRATION_ACCESS": "true", "VALIDATION_ACCESS": "false", "HOLDOUT_ACCESS": "false"})
    monkeypatch.setattr("aegis_range_v1.sweep_reclaim_discovery.SealedPartitionGuard.access_flags", lambda environment: FLAGS)
    monkeypatch.setattr("aegis_range_v1.sweep_reclaim_discovery.subprocess.run", lambda *args, **kwargs: type("Result", (), {"stdout": "5a85018a6363006a44f683b22f2dad8cbdb49f49\n"})())
    repo = tmp_path / "repo"
    immutable = repo / "sandbox/aegis_range_strategy_v1/artifacts/r2_train/run_a"
    immutable.mkdir(parents=True)
    with pytest.raises(PermissionError, match="IMMUTABLE"):
        verify_authority(repo, immutable / "child")
