from datetime import datetime, timedelta, timezone

import pytest

from aegis.data import CanonicalBar
from aegis.domain import Regime, TradeSide
from aegis.training.econ import COST_SCENARIOS, EconomicSignal, equal_budget, replay_economics


def _prices(direction: float = -1.0) -> tuple[CanonicalBar, ...]:
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    rows = []
    for index in range(15):
        open_price = 100.0 + direction * index
        rows.append(CanonicalBar(
            start + timedelta(minutes=5 * index), open_price,
            open_price + 0.5, open_price - 0.5, open_price + direction * 0.25, 1000.0,
        ))
    return tuple(rows)


def _signal(strategy: str = "full") -> EconomicSignal:
    return EconomicSignal(
        _prices()[0].timestamp, "ADAUSDT", TradeSide.SHORT, 0.8, strategy, 0, Regime.BEAR_TREND,
    )


def test_econ_uses_next_bar_open_h12_prices_and_registered_costs() -> None:
    report = replay_economics((_signal(),), {"ADAUSDT": _prices()}, bootstrap_repetitions=20)
    assert len(report.trades) == 3
    base = next(trade for trade in report.trades if trade.scenario_id == "B_BASE")
    assert base.entry_price == 99.0
    assert base.exit_price == pytest.approx(87.75)
    assert base.gross_return_fraction == pytest.approx((99.0 - 87.75) / 99.0)
    assert base.cost_fraction == pytest.approx(COST_SCENARIOS[1].cost_fraction(12))
    assert base.net_return_fraction == pytest.approx(base.gross_return_fraction - base.cost_fraction)
    assert base.mfe_fraction > 0.0
    assert base.mae_fraction == pytest.approx(0.5 / 99.0)


def test_econ_is_deterministic_short_only_and_reports_each_scenario() -> None:
    kwargs = {"signals": (_signal(),), "prices": {"ADAUSDT": _prices()}, "bootstrap_repetitions": 20}
    first = replay_economics(**kwargs)
    second = replay_economics(**kwargs)
    assert first == second
    assert set(first.metrics["full"]) == {item.scenario_id for item in COST_SCENARIOS}
    assert first.metrics["full"]["B_BASE"].profit_factor > 1.0
    with pytest.raises(ValueError, match="SHORT"):
        replay_economics(
            (EconomicSignal(_prices()[0].timestamp, "ADAUSDT", TradeSide.LONG, 0.8, "bad", 0, Regime.BULL_TREND),),
            {"ADAUSDT": _prices()},
        )


def test_equal_budget_does_not_use_outcomes() -> None:
    signals = {
        "full": (_signal("full"), _signal("full")),
        "baseline": (_signal("baseline"),),
    }
    budgeted = equal_budget(signals)
    assert {name: len(rows) for name, rows in budgeted.items()} == {"baseline": 1, "full": 1}
