from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aegis.data import CanonicalBar
from aegis.training.econ import CostScenario
from aegis.utils import sha256_file
from scripts.diagnostics.e4a_mechanical_exit.experiment import (
    DataCoverageError,
    PolicyParameters,
    TradeTrajectory,
    _net_value,
    atomic_parquet,
    break_even_price,
    build_trajectories,
    callback_trigger_roe,
    initial_atr,
    initial_stop_price,
    protection_target_roe,
    short_roe,
    simulate_policy,
    take_profit_price,
    true_range,
    update_atr,
)
from scripts.diagnostics.exit_excursion_d1a.experiment import FrozenEntry


ROOT = Path(__file__).resolve().parents[2]
UTC = timezone.utc


def _bar(timestamp: datetime, *, open_: float = 100.0, high: float = 100.1, low: float = 99.9, close: float = 100.0) -> CanonicalBar:
    return CanonicalBar(timestamp, open_, high, low, close, 1.0)


def _parameters() -> PolicyParameters:
    return PolicyParameters(
        leverage=20.0, stop_roe=-0.40, take_profit_roe=0.50,
        break_even_activation_roe=0.08, break_even_offset_fraction=0.003,
        trailing_activation_roe=0.15, atr_multiplier=1.5, callback_fraction=0.08,
        protection_min_peak_roe=0.08, protection_giveback_roe=0.05,
        protection_min_roe=0.01, immediate_buffer_fraction=0.001, timeout_bars=96,
    )


def _trajectory(path: tuple[CanonicalBar, ...] | None = None) -> TradeTrajectory:
    signal = datetime(2025, 1, 1, tzinfo=UTC)
    history = tuple(_bar(signal - timedelta(minutes=5 * offset)) for offset in range(14, -1, -1))
    bars = path or tuple(_bar(signal + timedelta(minutes=5 * index)) for index in range(1, 97))
    expected_exit = bars[11]
    entry = FrozenEntry(
        trade_id="trade", symbol="BTCUSDT", fold=1, signal_timestamp=signal,
        entry_timestamp=bars[0].timestamp, exit_timestamp=expected_exit.timestamp,
        entry_price=100.0, exit_price=expected_exit.close,
        expected_gross=(100.0 - expected_exit.close) / 100.0,
        expected_mfe=max((100.0 - bar.low) / 100.0 for bar in bars[:12]),
        expected_mae=max((bar.high - 100.0) / 100.0 for bar in bars[:12]),
        expected_cost=0.0015, expected_net=(100.0 - expected_exit.close) / 100.0 - 0.0015,
    )
    return TradeTrajectory(entry, history, bars, initial_atr(history))


def test_short_roe_and_historical_price_conversions() -> None:
    assert short_roe(100.0, 98.0, 20.0) == pytest.approx(0.40)
    assert initial_stop_price(100.0, -0.40, 20.0) == pytest.approx(102.0)
    assert take_profit_price(100.0, 0.50, 20.0) == pytest.approx(97.5)
    assert break_even_price(100.0, 0.003) == pytest.approx(99.7)


def test_preregistration_freezes_leverage_and_owner_parameters() -> None:
    import yaml

    path = ROOT / "reports/experiments/e4a_mechanical_exit/preregistration.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    binding = json.loads((ROOT / "reports/governance/e4a_mechanical_exit/preregistration_binding.json").read_text())
    assert sha256_file(path) == binding["preregistration"]["physical_sha256"]
    assert config["fixed_leverage"]["value"] == 20.0
    assert config["fixed_leverage"]["sizing_effect"] == "FORBIDDEN"
    assert config["policy_parameters"]["stop_roe"] == -0.40
    assert config["policy_parameters"]["take_profit_roe"] == 0.50


def test_initial_atr_uses_fourteen_completed_true_ranges() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    history = tuple(_bar(start + timedelta(minutes=5 * index), high=101.0, low=99.0) for index in range(15))
    assert true_range(history[1], history[0].close) == 2.0
    assert initial_atr(history) == 2.0
    assert update_atr(2.0, 3.4) == pytest.approx(2.1)


def test_initial_atr_has_no_future_lookahead() -> None:
    trajectory = _trajectory()
    future_changed = list(trajectory.path)
    future_changed[0] = _bar(future_changed[0].timestamp, high=120.0, low=80.0)
    assert initial_atr(trajectory.pre_entry) == trajectory.initial_atr
    assert initial_atr(TradeTrajectory(trajectory.entry, trajectory.pre_entry, tuple(future_changed), trajectory.initial_atr).pre_entry) == trajectory.initial_atr


def test_callback_and_profit_protection_use_observed_peak_only() -> None:
    assert callback_trigger_roe(0.25, 0.08) == pytest.approx(0.23)
    assert protection_target_roe(0.08, 0.05, 0.01) == pytest.approx(0.03)
    assert protection_target_roe(0.20, 0.05, 0.01) == pytest.approx(0.15)


def test_conservative_intrabar_precedence_uses_stop_over_take_profit() -> None:
    trajectory = _trajectory(tuple(
        [_bar(datetime(2025, 1, 1, 0, 5, tzinfo=UTC), high=103.0, low=97.0)]
        + [_bar(datetime(2025, 1, 1, 0, 5, tzinfo=UTC) + timedelta(minutes=5 * index)) for index in range(1, 96)]
    ))
    conservative, _ = simulate_policy(trajectory, "P2", "CONSERVATIVE", _parameters())
    optimistic, _ = simulate_policy(trajectory, "P2", "OPTIMISTIC", _parameters())
    assert conservative["exit_reason"] == "CLOSED_STOP"
    assert conservative["exit_price"] == pytest.approx(102.0)
    assert conservative["intrabar_ambiguous"] is True
    assert optimistic["exit_reason"] == "CLOSED_TAKE_PROFIT"
    assert optimistic["exit_price"] == pytest.approx(97.5)


def test_break_even_is_effective_only_after_activation_candle_conservatively() -> None:
    start = datetime(2025, 1, 1, 0, 5, tzinfo=UTC)
    bars = [_bar(start, high=100.0, low=99.5, close=99.6)]
    bars.append(_bar(start + timedelta(minutes=5), high=100.0, low=99.5, close=99.8))
    bars.extend(_bar(start + timedelta(minutes=5 * index)) for index in range(2, 96))
    trade, events = simulate_policy(_trajectory(tuple(bars)), "P3", "CONSERVATIVE", _parameters())
    assert trade["exit_bar"] == 2
    assert trade["exit_reason"] == "CLOSED_BREAK_EVEN"
    assert trade["exit_price"] == pytest.approx(99.7)
    assert any(event["event_type"] == "BREAK_EVEN_ARMED" for event in events)


def test_trailing_stop_updates_are_monotonic_for_short() -> None:
    start = datetime(2025, 1, 1, 0, 5, tzinfo=UTC)
    bars = []
    for index in range(96):
        center = 99.0 - index * 0.02
        bars.append(_bar(start + timedelta(minutes=5 * index), open_=center, high=center + 0.10, low=center - 0.10, close=center))
    parameters = replace(_parameters(), protection_min_peak_roe=10.0)
    _, events = simulate_policy(_trajectory(tuple(bars)), "P4", "CONSERVATIVE", parameters)
    stops = [json.loads(event["details_json"])["stop"] for event in events if event["event_type"] == "TRAILING_STOP_UPDATED"]
    assert stops
    assert all(right <= left for left, right in zip(stops, stops[1:]))


def test_p1_times_out_every_open_trade_at_bar_96() -> None:
    trade, _ = simulate_policy(_trajectory(), "P1", "CONSERVATIVE", _parameters())
    assert trade["exit_bar"] == 96
    assert trade["exit_reason"] == "CLOSED_TIMEOUT"


def test_p0_reproduces_h12_close_exactly() -> None:
    trajectory = _trajectory()
    trade, _ = simulate_policy(trajectory, "P0", "CONSERVATIVE", _parameters())
    assert trade["exit_bar"] == 12
    assert trade["exit_price"] == trajectory.entry.exit_price
    assert trade["gross_return"] == trajectory.entry.expected_gross


def test_costs_use_actual_holding_duration() -> None:
    scenario = CostScenario("B_BASE", 5.0, 2.0, 1.0)
    row = {"gross_return": 0.01, "exit_bar": 12}
    assert _net_value(row, scenario) == pytest.approx(0.0085)
    long_row = {"gross_return": 0.01, "exit_bar": 96}
    assert _net_value(long_row, scenario) == pytest.approx(0.0078)


def test_trajectory_builder_enforces_boundary_and_coverage() -> None:
    trajectory = _trajectory()
    all_bars = trajectory.pre_entry + trajectory.path
    built, rows, atr_rows, quality = build_trajectories(
        (trajectory.entry,), {"BTCUSDT": all_bars}, horizon_bars=96, pre_entry_bars=15,
        dev_boundary=trajectory.path[-1].timestamp,
    )
    assert len(built) == 1
    assert len(rows) == 96
    assert len(atr_rows) == 96
    assert quality["complete_fraction"] == 1.0
    with pytest.raises(DataCoverageError):
        build_trajectories(
            (trajectory.entry,), {"BTCUSDT": all_bars[:-1]}, horizon_bars=96, pre_entry_bars=15,
            dev_boundary=trajectory.path[-1].timestamp,
        )


def test_parquet_and_simulation_are_deterministic(tmp_path: Path) -> None:
    trajectory = _trajectory()
    first_trade, first_events = simulate_policy(trajectory, "P4", "CONSERVATIVE", _parameters())
    second_trade, second_events = simulate_policy(trajectory, "P4", "CONSERVATIVE", _parameters())
    assert first_trade == second_trade
    assert first_events == second_events
    first = tmp_path / "one.parquet"; second = tmp_path / "two.parquet"
    atomic_parquet(first, [first_trade], tuple(first_trade))
    atomic_parquet(second, [second_trade], tuple(second_trade))
    assert first.read_bytes() == second.read_bytes()


def test_e3_d1a_provenance_and_lockbox_are_unchanged() -> None:
    assert sha256_file(ROOT / "config/experiments/aegis_short_candidate_e3.yaml") == "281e27f93f8be0f9c4fbe78673d55f1a4f3391aa421c3fcaf90823113dc81e62"
    assert sha256_file(ROOT / "reports/experiments/exit_excursion_d1a/scientific_aggregate.json") == "fc6edaddb2621cb1b66aebc01935ff69d52d5c5f4b336fb84e89a4ab4b07b13d"
    assert sha256_file(ROOT / "reports/governance/exit_policy_provenance_p1/provenance_manifest.json") == "eb7186418a8f6dfb50b93deeb380d92be4fe09eded333601693daaaa6a30531b"
    lockbox = json.loads((ROOT / "reports/experiments/lockbox_semi_blind_20260427_20260711.json").read_text())
    assert lockbox["status"] == "NOT_CONSUMED"
    assert lockbox["consumed_queries"] == []
    assert lockbox["maximum_queries_total"] == 1
