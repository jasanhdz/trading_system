from dataclasses import replace

import pytest

from aegis.domain import DecisionRequest, DecisionStatus, TradeSide


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_positive_direction_scenarios_freeze_and_record(side, snapshot_factory, scenario_runtime_factory) -> None:
    snapshot = snapshot_factory()
    runtime = scenario_runtime_factory(side, snapshot)
    request = DecisionRequest(f"request-{side}", f"cycle-{side}", "aegis-decision-request-v1",
                              runtime.config.contract_version, runtime.config.config_version, snapshot)
    response = runtime.evaluate(request)
    assert response.status is DecisionStatus.SELECTED
    assert len(response.selected) == 1
    assert response.selected[0].side is TradeSide(side)
    assert len(response.ranking) == 11
    assert len(runtime.evidence.events) == 1  # type: ignore[attr-defined]
    assert runtime.evaluate(request) == response
    assert len(runtime.evidence.events) == 1  # type: ignore[attr-defined]


def test_neutral_scenario_is_first_class_no_trade(snapshot_factory, scenario_runtime_factory) -> None:
    snapshot = snapshot_factory()
    runtime = scenario_runtime_factory("NEUTRAL", snapshot)
    request = DecisionRequest("request-neutral", "cycle-neutral", "aegis-decision-request-v1",
                              runtime.config.contract_version, runtime.config.config_version, snapshot)
    response = runtime.evaluate(request)
    assert response.status is DecisionStatus.NO_TRADE
    assert not response.selected
    assert all(not row.eligible for row in response.ranking)


def test_reordered_snapshot_has_identical_science(snapshot_factory, scenario_runtime_factory) -> None:
    snapshot = snapshot_factory()
    runtime_a = scenario_runtime_factory("LONG", snapshot)
    reordered = replace(snapshot, series=tuple(reversed(snapshot.series)))
    runtime_b = scenario_runtime_factory("LONG", reordered)
    request_a = DecisionRequest("same", "cycle", "aegis-decision-request-v1", runtime_a.config.contract_version, runtime_a.config.config_version, snapshot)
    request_b = replace(request_a, snapshot=reordered)
    response_a = runtime_a.evaluate(request_a)
    response_b = runtime_b.evaluate(request_b)
    assert response_a.selected == response_b.selected
    assert response_a.ranking == response_b.ranking
    assert response_a.decision_id == response_b.decision_id
