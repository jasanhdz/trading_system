from datetime import timedelta
from pathlib import Path

from aegis.api import BrainApi, create_app
from aegis.domain import DecisionStatus
from aegis.runtime import build_runtime
from aegis.utils import FixedUtcClock


def test_end_to_end_scientific_pipeline_is_deterministic_and_global(decision_request) -> None:
    runtime = build_runtime(Path(__file__).parents[2] / "config", clock=FixedUtcClock(decision_request.snapshot.closed_at + timedelta(minutes=1)))
    first = runtime.evaluate(decision_request)
    second = runtime.evaluate(decision_request)
    assert first == second
    assert first.status in (DecisionStatus.SELECTED, DecisionStatus.NO_TRADE)
    assert len(first.ranking) == 11
    assert len(first.selected) <= 1
    assert len(runtime.evidence.events) == 2  # type: ignore[attr-defined]
    metrics = runtime.metrics.snapshot()
    assert metrics["counters"]["requests"] == 2
    assert {"validation", "features", "models", "layers", "candidates", "selection", "total"} <= set(metrics["mean_latency_seconds"])


def test_api_health_manifest_and_routes(decision_request) -> None:
    runtime = build_runtime(Path(__file__).parents[2] / "config", clock=FixedUtcClock(decision_request.snapshot.closed_at + timedelta(minutes=1)))
    api = BrainApi(runtime)
    assert api.health() == {"status": "alive"}
    assert api.manifest().ready is True and len(api.manifest().symbols) == 11
    assert api.evaluate(decision_request).decision_cycle_id == "cycle-1"
    paths = {route.path for route in create_app(api).routes}
    assert {"/health", "/ready", "/manifest", "/v1/decisions/evaluate", "/v1/evidence/outcome"} <= paths
    assert not ({"/order", "/positions", "/binance", "/close", "/brackets", "/leverage"} & paths)
