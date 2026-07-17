from dataclasses import replace
from datetime import timedelta

import pytest

from aegis.domain import Candle, DecisionRequest, DecisionStatus, EvidenceMode, TradeSide
from aegis.shadow import (
    ShadowOutcomeError,
    ShadowReplayCase,
    ShadowReplayEngine,
    resolve_shadow_outcome,
)
from aegis.utils import MutableUtcClock


def _request(runtime, snapshot, suffix: str) -> DecisionRequest:
    return DecisionRequest(
        f"request-{suffix}", f"cycle-{suffix}", "aegis-decision-request-v1",
        runtime.config.contract_version, runtime.config.config_version, snapshot,
    )


def _future(start, *, rising: bool, bars: int = 12, adverse_fraction: float = 0.001):
    candles = []
    price = 100.0
    for index in range(bars):
        open_time = start + timedelta(minutes=5 * index)
        close_time = open_time + timedelta(minutes=5)
        close = price * (1.002 if rising else 0.998)
        high = max(price, close) * (1.0 + adverse_fraction)
        low = min(price, close) * (1.0 - adverse_fraction)
        candles.append(Candle(open_time, close_time, price, high, low, close, 1000.0, True, "SHADOW_FIXTURE"))
        price = close
    return tuple(candles)


@pytest.mark.parametrize("side,rising", [("LONG", True), ("SHORT", False)])
def test_shadow_resolves_long_and_short_without_execution(
    side, rising, snapshot_factory, scenario_runtime_factory,
) -> None:
    snapshot = snapshot_factory()
    runtime = scenario_runtime_factory(side, snapshot)
    response = runtime.evaluate(_request(runtime, snapshot, side.lower()))
    selected = response.selected[0]
    selected = replace(selected, risk_intent=replace(selected.risk_intent, stop_distance_fraction=0.50))
    response = replace(response, selected=(selected,))
    outcome = resolve_shadow_outcome(
        response, {selected.symbol: _future(response.generated_at, rising=rising)}, friction_fraction=0.0014,
    )
    assert outcome is not None
    assert outcome.execution_mode is EvidenceMode.SHADOW
    assert outcome.executed is False
    assert outcome.accepted is False
    assert outcome.fill.status.value == "NOT_EXECUTED"
    assert outcome.hypothetical_details["net_return_fraction"] > 0
    assert outcome.hypothetical_details["invalidation_triggered"] is False


def test_shadow_scientific_invalidation_is_applied_without_an_order(
    snapshot_factory, scenario_runtime_factory,
) -> None:
    snapshot = snapshot_factory()
    runtime = scenario_runtime_factory("LONG", snapshot)
    response = runtime.evaluate(_request(runtime, snapshot, "invalidated"))
    selected = response.selected[0]
    stop = selected.risk_intent.stop_distance_fraction
    assert stop is not None
    outcome = resolve_shadow_outcome(
        response,
        {selected.symbol: _future(response.generated_at, rising=True, adverse_fraction=stop + 0.01)},
        friction_fraction=0.0014,
    )
    assert outcome is not None
    assert outcome.close_reason == "SCIENTIFIC_INVALIDATION"
    assert outcome.hypothetical_details["invalidation_triggered"] is True
    assert outcome.hypothetical_details["gross_return_fraction"] == pytest.approx(-stop)


def test_no_trade_has_no_hypothetical_outcome(snapshot_factory, scenario_runtime_factory) -> None:
    snapshot = snapshot_factory()
    runtime = scenario_runtime_factory("NEUTRAL", snapshot)
    response = runtime.evaluate(_request(runtime, snapshot, "neutral-shadow"))
    assert response.status is DecisionStatus.NO_TRADE
    assert resolve_shadow_outcome(response, {}, friction_fraction=0.0014) is None


def test_shadow_rejects_immature_partial_and_gapped_candles(snapshot_factory, scenario_runtime_factory) -> None:
    snapshot = snapshot_factory()
    runtime = scenario_runtime_factory("SHORT", snapshot)
    response = runtime.evaluate(_request(runtime, snapshot, "bad-forward"))
    symbol = response.selected[0].symbol
    with pytest.raises(ShadowOutcomeError, match="not mature"):
        resolve_shadow_outcome(response, {symbol: _future(response.generated_at, rising=False, bars=11)}, friction_fraction=0.0)
    partial = list(_future(response.generated_at, rising=False))
    partial[-1] = replace(partial[-1], is_closed=False)
    with pytest.raises(ShadowOutcomeError, match="final"):
        resolve_shadow_outcome(response, {symbol: partial}, friction_fraction=0.0)
    gapped = list(_future(response.generated_at, rising=False))
    gapped[5] = replace(
        gapped[5], open_time=gapped[5].open_time + timedelta(minutes=1),
        close_time=gapped[5].close_time + timedelta(minutes=1),
    )
    with pytest.raises(ShadowOutcomeError, match="gap"):
        resolve_shadow_outcome(response, {symbol: gapped}, friction_fraction=0.0)


def test_shadow_replay_is_chronological_deterministic_and_records_once(
    snapshot_factory, scenario_runtime_factory,
) -> None:
    first_snapshot = snapshot_factory(closed_at=snapshot_factory().closed_at)
    second_snapshot = snapshot_factory(closed_at=first_snapshot.closed_at + timedelta(hours=1))
    runtime = scenario_runtime_factory("SHORT", first_snapshot)
    runtime.clock = MutableUtcClock(first_snapshot.closed_at)
    first = _request(runtime, first_snapshot, "one")
    second = _request(runtime, second_snapshot, "two")
    symbol = runtime.evaluate(first).selected[0].symbol
    # Use a fresh runtime so the replay itself owns all decision and outcome evidence.
    runtime = scenario_runtime_factory("SHORT", first_snapshot)
    runtime.clock = MutableUtcClock(first_snapshot.closed_at)
    replay = ShadowReplayEngine(runtime, friction_fraction=0.0014)
    result = replay.run((
        ShadowReplayCase(second, {symbol: _future(second_snapshot.closed_at, rising=False)}),
        ShadowReplayCase(first, {symbol: _future(first_snapshot.closed_at, rising=False)}),
    ))
    assert [item.decision_cycle_id for item in result.decisions] == ["cycle-one", "cycle-two"]
    assert len(result.outcomes) == 2
    assert all(item.execution_mode is EvidenceMode.REPLAY for item in result.outcomes)
    assert all(not item.executed for item in result.outcomes)
    assert len(runtime.evidence.events) == 4  # type: ignore[attr-defined]
    for outcome in result.outcomes:
        runtime.evidence.record_outcome(outcome)
    assert len(runtime.evidence.events) == 4  # type: ignore[attr-defined]


def test_shadow_replay_rejects_duplicate_cycles(snapshot_factory, scenario_runtime_factory) -> None:
    snapshot = snapshot_factory()
    runtime = scenario_runtime_factory("LONG", snapshot)
    runtime.clock = MutableUtcClock(snapshot.closed_at)
    request = _request(runtime, snapshot, "duplicate")
    case = ShadowReplayCase(request, {})
    with pytest.raises(ShadowOutcomeError, match="duplicate"):
        ShadowReplayEngine(runtime, friction_fraction=0.0).run((case, case))
