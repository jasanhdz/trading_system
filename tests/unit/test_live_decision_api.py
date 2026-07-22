from __future__ import annotations

import inspect
import math
from datetime import datetime, timedelta, timezone

import pytest
import httpx

from aegis.config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from aegis.domain import Candle, FeedQuality, MarketSnapshot, PortfolioContext, SymbolSeries
from aegis.live_api import create_app
from aegis.live_decision import (
    FEATURE_COUNT,
    FEATURE_SCHEMA,
    MODEL_ARTIFACT_SHA256,
    MODEL_BUNDLE_SHA256,
    CurrentBrainDecisionService,
    CurrentBrainEngine,
    CurrentBrainError,
    compatibility_response,
)


class StaticProvider:
    def __init__(self, snapshot) -> None:
        self.value = snapshot
        self.calls = 0

    def snapshot(self):
        self.calls += 1
        return self.value


def recent_close() -> datetime:
    now = datetime.now(timezone.utc)
    aligned = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 5)
    return aligned - timedelta(minutes=5)


@pytest.fixture(scope="module")
def real_engine() -> CurrentBrainEngine:
    engine = CurrentBrainEngine()
    engine.initialize()
    return engine


@pytest.fixture(scope="module")
def current_snapshot():
    end = recent_close()
    series = []
    bars = 96
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        candles = []
        base = 10.0 + symbol_index * 3.0
        for index in range(bars):
            close_time = end - timedelta(minutes=5 * (bars - 1 - index))
            open_time = close_time - timedelta(minutes=5)
            drift = (symbol_index - 5) * 0.00008
            open_price = base * (1.0 + drift * index)
            close = open_price * (1.0 + drift + ((index % 5) - 2) * 0.00003)
            high = max(open_price, close) * 1.001
            low = min(open_price, close) * 0.999
            candles.append(Candle(
                open_time,
                close_time,
                open_price,
                high,
                low,
                close,
                1000.0 + symbol_index * 10 + index,
                True,
                "CURRENT_BRAIN_HTTP_FIXTURE",
                str(index),
            ))
        series.append(SymbolSeries(symbol, tuple(candles), end, FeedQuality()))
    return MarketSnapshot(
        end,
        "5m",
        CANONICAL_SYMBOL_SET_HASH,
        tuple(reversed(series)),
        PortfolioContext(available_slots=1, operational_time=end),
    )


@pytest.fixture(scope="module")
def canonical_batch(real_engine, current_snapshot):
    return real_engine.evaluate(current_snapshot)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_health_not_ready_before_initialization() -> None:
    engine = CurrentBrainEngine()
    service = CurrentBrainDecisionService(engine, StaticProvider(None))
    health = service.health()
    assert health["ready"] is False
    assert health["status"] == "not_ready"


def test_health_ready_only_after_real_model_self_check(real_engine) -> None:
    health = CurrentBrainDecisionService(real_engine, StaticProvider(None)).health()
    assert health["ready"] is True
    assert health["model_sha256"] == MODEL_ARTIFACT_SHA256
    assert health["bundle_sha256"] == MODEL_BUNDLE_SHA256
    assert health["feature_schema"] == FEATURE_SCHEMA
    assert health["feature_count"] == FEATURE_COUNT


@pytest.mark.anyio
async def test_real_pipeline_and_http_adapter_are_lossless(real_engine, current_snapshot, canonical_batch) -> None:
    provider = StaticProvider(current_snapshot)
    service = CurrentBrainDecisionService(real_engine, provider, cache_seconds=60)
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(20):
            for symbol in CANONICAL_SYMBOLS:
                response = await client.post("/ml-v2/predict", json={"symbol": symbol})
                assert response.status_code == 200
                expected = compatibility_response(
                    canonical_batch,
                    symbol,
                    response.json()["metadata"]["trace_id"],
                )
                assert response.json() == expected
    assert provider.calls == 1


@pytest.mark.parametrize("symbol", CANONICAL_SYMBOLS)
def test_all_configured_symbols_have_valid_current_outputs(canonical_batch, symbol) -> None:
    response = compatibility_response(canonical_batch, symbol, "trace")
    probabilities = (response["long_prob"], response["short_prob"], response["neutral_prob"])
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)
    assert response["features"]["fallback"] is False
    assert response["features"]["schema"] == "aegis-features-v2"
    assert response["features"]["count"] == 83
    assert response["aegis"]["decision_brain"]["decision"] in {"ENTER_NOW", "DO_NOT_ENTER"}


def test_single_estimator_is_not_inflated_into_legacy_consensus(canonical_batch) -> None:
    for symbol in CANONICAL_SYMBOLS:
        response = compatibility_response(canonical_batch, symbol, "trace")
        votes = response["aegis"]["turbo"]["raw"]["votes"]
        assert sum(votes.values()) == 1
        assert response["metadata"]["directional_estimator_count"] == 1
        assert response["metadata"]["leverage_recommendation"] == "NOT_PRESENT"
        assert response["metadata"]["position_fraction"] == "NOT_PRESENT"


def test_would_execute_and_action_match_canonical_selection(canonical_batch) -> None:
    for symbol, canonical in canonical_batch["results"].items():
        response = compatibility_response(canonical_batch, symbol, "trace")
        raw = response["aegis"]["turbo"]["raw"]
        assert raw["would_execute"] is canonical["selected"]
        assert (raw["action"] in {"LONG", "SHORT"}) is canonical["selected"]
        assert raw["turbo_score"] == canonical["candidate"]["raw_score"]
        assert response["metadata"]["canonical_calibrated_score"] == canonical["candidate"]["calibrated_score"]


@pytest.mark.anyio
async def test_unknown_symbol_and_malformed_request_fail_safely(real_engine, current_snapshot) -> None:
    service = CurrentBrainDecisionService(real_engine, StaticProvider(current_snapshot))
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/ml-v2/predict", json={"symbol": "UNKNOWN"})).status_code == 422
        assert (await client.post("/ml-v2/predict", json={"symbol": "ETHUSDT", "extra": True})).status_code == 422
        assert (await client.post("/ml-v2/predict", json={})).status_code == 422


@pytest.mark.anyio
async def test_model_failure_never_masquerades_as_successful_hold(real_engine) -> None:
    class FailingProvider:
        def snapshot(self):
            raise CurrentBrainError("TEST_MODEL_UNAVAILABLE")

    service = CurrentBrainDecisionService(real_engine, FailingProvider())
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ml-v2/predict", json={"symbol": "ETHUSDT"})
        assert response.status_code == 503
        assert "long_prob" not in response.json()


@pytest.mark.anyio
async def test_exit_signal_fails_closed_because_current_brain_has_no_exit_contract(
    real_engine,
    current_snapshot,
) -> None:
    service = CurrentBrainDecisionService(real_engine, StaticProvider(current_snapshot))
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ml-v2/exit_signal", json={"symbol": "ETHUSDT"})
        assert response.status_code == 501
        assert response.json()["detail"] == "AEGIS_CURRENT_BRAIN_EXIT_SIGNAL_NOT_PRESENT"


def test_live_http_modules_have_no_binance_mutation_surface() -> None:
    import aegis.live_api as api_module
    import aegis.live_decision as decision_module

    source = (inspect.getsource(api_module) + inspect.getsource(decision_module)).lower()
    forbidden = (
        "create_order",
        "cancel_order",
        "cancel_all",
        "change_leverage",
        "change_margin",
        "positionside/dual",
        "/fapi/v1/order",
        "api_secret",
        "api_key",
    )
    assert all(value not in source for value in forbidden)
    assert "https://fapi.binance.com/fapi/v1/klines" in source
