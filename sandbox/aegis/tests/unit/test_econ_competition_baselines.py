import inspect
from datetime import datetime, timedelta, timezone

import numpy as np

import aegis.training.econ as econ_module
from aegis.data import CanonicalBar
from aegis.domain import Regime
from aegis.training.econ import (
    BASELINE_STRATEGIES, EconomicBaselineCycle, raw_baseline_indicators,
    replay_economics, select_competition_baselines,
)


def _history(*, drift: float = 0.0, wide: bool = False, gap: bool = False) -> tuple[CanonicalBar, ...]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(288):
        timestamp = start + timedelta(minutes=5 * (index + (1 if gap and index >= 200 else 0)))
        close = 100.0 + drift * index
        spread = 3.0 if wide else 0.25
        rows.append(CanonicalBar(timestamp, close - drift * 0.5, close + spread, close - spread, close, 1000.0))
    return tuple(rows)


def _cycle(timestamp: datetime, fold: int = 1) -> EconomicBaselineCycle:
    histories = {
        "AAA": _history(drift=-0.05),
        "BBB": _history(drift=0.05),
        "CCC": _history(drift=0.0, wide=True),
    }
    return EconomicBaselineCycle(
        timestamp, fold, histories, {symbol: Regime.BEAR_TREND for symbol in histories},
        {"AAA": 0.1, "BBB": 0.3, "CCC": 0.2},
        {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2},
        ("AAA", "CCC"), ("AAA", "BBB", "CCC"),
    )


def test_raw_baseline_formulae_match_golden_calculations() -> None:
    rows = _history(drift=-0.05)
    result = raw_baseline_indicators(rows)
    assert result is not None
    closes = np.asarray([row.close for row in rows])
    expected_ema = closes[0]
    for close in closes[1:]:
        expected_ema = 2.0 / 25.0 * close + 23.0 / 25.0 * expected_ema
    expected_tr = [
        max(rows[index].high - rows[index].low, abs(rows[index].high - rows[index - 1].close), abs(rows[index].low - rows[index - 1].close)) / rows[index].close
        for index in range(264, 288)
    ]
    assert result.ret_12 == closes[-1] / closes[-13] - 1.0
    assert result.close_to_ema24 == closes[-1] / expected_ema - 1.0
    assert result.volatility_24 == float(np.mean(expected_tr))


def test_raw_baselines_reject_incomplete_or_noncontiguous_windows_without_imputation() -> None:
    assert raw_baseline_indicators(_history()[:-1]) is None
    assert raw_baseline_indicators(_history(gap=True)) is None
    source = inspect.getsource(econ_module)
    assert "aegis.features" not in source and "FeaturePipeline" not in source


def test_rules_ties_random_seed_and_one_selection_per_cycle_are_deterministic() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    cycles = (_cycle(start), _cycle(start + timedelta(hours=1)))
    first = select_competition_baselines(cycles, {1: 2})
    second = select_competition_baselines(cycles, {1: 2})
    assert first == second and first.content_hash == second.content_hash
    assert {row.symbol for row in first.signals["momentum_rule"]} == {"AAA"}
    assert {row.symbol for row in first.signals["mean_reversion_rule"]} == {"BBB"}
    assert {row.symbol for row in first.signals["volatility_rule"]} == {"CCC"}
    assert all(len({row.timestamp for row in values}) == len(values) for values in first.signals.values())
    rng = np.random.Generator(np.random.PCG64(20260718))
    expected_first = sorted([(float(rng.random()), symbol) for symbol in ("AAA", "BBB", "CCC")], key=lambda item: (-item[0], item[1]))[0][1]
    assert first.signals["random_directional_with_gates"][0].symbol == expected_first
    assert first.signals["eqm_only"][0].symbol == "AAA"
    assert first.signals["trrm_only"][0].symbol == "AAA"


def test_fold_budget_caps_without_fill_or_redistribution_and_marks_zero_fold() -> None:
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    cycles = (_cycle(start, 1), _cycle(start + timedelta(hours=1), 1), _cycle(start + timedelta(hours=2), 2))
    result = select_competition_baselines(cycles, {1: 5, 2: 0})
    assert all(result.selected_counts[name][1] <= 2 for name in BASELINE_STRATEGIES if name != "no_trade")
    assert all(result.fold_status[name][1] == "ELIGIBILITY_DEFICIT" for name in BASELINE_STRATEGIES if name != "no_trade")
    assert all(result.selected_counts[name][2] == 0 for name in BASELINE_STRATEGIES)
    assert all(result.fold_status[name][2] == "NO_TRADE_FOLD" for name in BASELINE_STRATEGIES)
    assert result.signals["no_trade"] == ()


def test_no_trade_has_null_profit_factor_and_no_synthetic_trade() -> None:
    report = replay_economics((), {}, bootstrap_repetitions=10, strategy_ids=("no_trade",))
    metrics = report.metrics["no_trade"]["B_BASE"]
    assert metrics.trades == 0 and metrics.expectancy == 0.0
    assert metrics.profit_factor is None and metrics.maximum_drawdown == 0.0
    assert report.trades == ()
