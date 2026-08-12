from __future__ import annotations

import copy
import inspect
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import httpx

from aegis.config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from aegis.domain import (
    Candle,
    FeedQuality,
    MarketSnapshot,
    PortfolioContext,
    SymbolSeries,
)
from aegis.live_api import create_app
from aegis.hybrid_live_experiment import (
    AUTHORITY as HYBRID_LIVE_AUTHORITY,
    CONFIGURATION_SHA256 as HYBRID_LIVE_CONFIGURATION_SHA256,
    CONTRACT_VERSION as HYBRID_LIVE_CONTRACT_VERSION,
    MODEL_IDENTIFIER as HYBRID_LIVE_MODEL_IDENTIFIER,
    MODEL_SHA256 as HYBRID_LIVE_MODEL_SHA256,
    HybridLiveExperimentConfig,
    HybridLiveExperimentSelector,
    load_hybrid_live_experiment_config,
)
from aegis.directional_confirmation import DirectionalConfirmationPolicy
from aegis.live_decision import (
    CANONICAL_DECISION_CONTRACT,
    CANONICAL_LIVE_AUTHORITY,
    CONFIGURATION_SHA256,
    FEATURE_COUNT,
    FEATURE_SCHEMA,
    MODEL_ARTIFACT_SHA256,
    MODEL_BUNDLE_SHA256,
    CurrentBrainDecisionService,
    CurrentBrainEngine,
    CurrentBrainError,
    PublicKlineSnapshotProvider,
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


def synthetic_snapshot(variant: int = 0):
    end = recent_close()
    series = []
    bars = 96
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        candles = []
        base = 10.0 + symbol_index * 3.0
        for index in range(bars):
            close_time = end - timedelta(minutes=5 * (bars - 1 - index))
            open_time = close_time - timedelta(minutes=5)
            drift = (symbol_index - 5) * 0.00008 + variant * 0.00001
            open_price = base * (1.0 + drift * index)
            close = open_price * (1.0 + drift + (((index + variant) % 5) - 2) * 0.00003)
            high = max(open_price, close) * 1.001
            low = min(open_price, close) * 0.999
            candles.append(
                Candle(
                    open_time,
                    close_time,
                    open_price,
                    high,
                    low,
                    close,
                    1000.0 + symbol_index * 10 + index + variant,
                    True,
                    "CURRENT_BRAIN_HTTP_FIXTURE",
                    f"{variant}-{index}",
                )
            )
        series.append(SymbolSeries(symbol, tuple(candles), end, FeedQuality()))
    return MarketSnapshot(
        end,
        "5m",
        CANONICAL_SYMBOL_SET_HASH,
        tuple(reversed(series)),
        PortfolioContext(available_slots=1, operational_time=end),
    )


@pytest.fixture(scope="module")
def current_snapshot():
    return synthetic_snapshot()


@pytest.fixture(scope="module")
def canonical_batch(real_engine, current_snapshot):
    return real_engine.evaluate(current_snapshot)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def hybrid_overlay(*, selected_symbol: str, selected_side: str) -> dict:
    overlay = {}
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        predictions = {}
        for side in ("LONG", "SHORT"):
            selected = symbol == selected_symbol and side == selected_side
            predictions[side] = {
                "side": side,
                "opportunity_probability": 0.8 if selected else 0.3,
                "danger_probability": 0.1 if selected else 0.7,
                "mae_q50": 0.005,
                "mae_q90": 0.005 + symbol_index * 0.0001,
                "mfe_q50": 0.02 if selected else 0.004,
                "net_return_mean": 0.01 if selected else -0.001,
                "shadow_rank_score": 0.9 if selected else 0.01 - symbol_index * 0.0001,
                "selection_effect": "NONE",
                "exchange_authority": False,
            }
        overlay[symbol] = {
            "hybrid_directional_shadow": {
                "mode": "SHADOW",
                "status": "OFFLINE_VALIDATION_FAILED_OBSERVATION_ONLY",
                "predictions": predictions,
            }
        }
    return overlay


def hybrid_selector(tmp_path: Path) -> HybridLiveExperimentSelector:
    config = HybridLiveExperimentConfig(
        path=tmp_path / "config.yaml",
        artifact_path=tmp_path / "artifact.json",
        readiness_path=tmp_path / "readiness.json",
        decision_journal=tmp_path / "decisions.jsonl",
        maximum_selected_per_cycle=len(CANONICAL_SYMBOLS),
        maximum_selected_per_symbol=1,
        confirmation_policy=DirectionalConfirmationPolicy(
            round_trip_cost_fraction=0.001,
            minimum_opportunity_probability_long=0.44,
            minimum_opportunity_probability_short=0.39,
            maximum_danger_probability=0.63,
            minimum_net_return_fraction=-0.0015,
            minimum_opportunity_percentile=0.60,
            minimum_danger_quality_percentile=0.40,
            minimum_net_return_percentile=0.60,
            minimum_path_efficiency_percentile=0.40,
            minimum_confirmation_components=3,
            minimum_close_location=0.55,
            minimum_volume_zscore=0.0,
        ),
    )
    return HybridLiveExperimentSelector(config)


def confirm_direction(batch: dict, symbol: str, side: str) -> None:
    features = batch["results"][symbol]["research_features"]
    sign = 1.0 if side == "LONG" else -1.0
    features.update(
        {
            "atr_12": 0.01,
            "close_vs_ema_6": sign * 0.01,
            "close_vs_ema_12": sign * 0.008,
            "close_vs_ema_24": sign * 0.006,
            "close_to_open_return": sign * 0.004,
            "close_position_in_range": 0.8 if side == "LONG" else 0.2,
            "ret_1": sign * 0.004,
            "ret_3": sign * 0.008,
            "ret_6": sign * 0.01,
            "ret_12": sign * 0.012,
            "ema_slope_6": sign * 0.002,
            "trend_stack_long": float(side == "LONG"),
            "trend_stack_short": float(side == "SHORT"),
            "volume_zscore_24": 1.0,
            "volume_return_1": 0.2,
        }
    )


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


def test_hybrid_live_config_freezes_multi_symbol_quality_selection() -> None:
    root = Path(__file__).parents[2]
    config = load_hybrid_live_experiment_config(
        root / "config/hybrid_directional_live_experiment.yaml",
        repo_root=root,
    )
    assert config.maximum_selected_per_cycle == len(CANONICAL_SYMBOLS)
    assert config.maximum_selected_per_symbol == 1
    assert config.confirmation_policy.minimum_opportunity_probability_long == 0.44
    assert config.confirmation_policy.minimum_opportunity_probability_short == 0.39
    assert config.confirmation_policy.minimum_confirmation_components == 3


@pytest.mark.parametrize(
    ("selected_symbol", "selected_side"),
    (("ADAUSDT", "LONG"), ("AVAXUSDT", "SHORT")),
)
def test_hybrid_live_selects_real_long_and_short_without_fabricated_votes(
    tmp_path: Path,
    canonical_batch,
    selected_symbol: str,
    selected_side: str,
) -> None:
    selector = hybrid_selector(tmp_path)
    batch = {
        **copy.deepcopy(canonical_batch),
        "decision_cycle_id": f"hybrid-{selected_symbol}-{selected_side}",
        "market_timestamp": "2026-08-02T20:00:00Z",
        "_entry_quality_v2": hybrid_overlay(
            selected_symbol=selected_symbol,
            selected_side=selected_side,
        ),
    }
    confirm_direction(batch, selected_symbol, selected_side)
    live = selector.apply(batch)
    assert live["candidate_count"] == 22
    assert live["selected_symbol"] == selected_symbol
    assert live["selected_side"] == selected_side
    assert sum(int(value["selected"]) for value in live["by_symbol"].values()) == 1
    selected = live["by_symbol"][selected_symbol]
    assert selected["fabricated_votes"] == 0
    assert selected["selected_prediction"]["opportunity_probability"] == 0.8
    archetype_observation = selected["selected_prediction"][
        "long_entry_archetype_v2_shadow"
    ]
    assert archetype_observation["mode"] == "SHADOW"
    assert archetype_observation["selection_effect"] == "NONE"
    assert archetype_observation["exchange_authority"] is False

    response = compatibility_response(
        {**batch, "_hybrid_directional_live": live}, selected_symbol, "trace"
    )
    brain = response["aegis"]["decision_brain"]
    assert brain["contract_version"] == HYBRID_LIVE_CONTRACT_VERSION
    assert brain["authority"] == HYBRID_LIVE_AUTHORITY
    assert brain["model_version"] == HYBRID_LIVE_MODEL_IDENTIFIER
    assert brain["model_sha256"] == HYBRID_LIVE_MODEL_SHA256
    assert brain["bundle_sha256"] == HYBRID_LIVE_MODEL_SHA256
    assert brain["configuration_sha256"] == HYBRID_LIVE_CONFIGURATION_SHA256
    assert brain["side"] == selected_side
    assert brain["selected"] is True
    assert response["aegis"]["directional_evidence"]["fabricated_votes"] == 0
    assert "votes" not in response["aegis"]["turbo"]["raw"]
    assert selector.health()["decision_records"] == 22
    assert selector.health()["long_entry_archetype_v2_shadow"]["records"] == 22
    selector.apply(batch)
    assert selector.health()["decision_records"] == 22


def test_hybrid_live_evaluates_and_selects_on_every_closed_five_minute_bar(
    tmp_path: Path, canonical_batch
) -> None:
    selector = hybrid_selector(tmp_path)
    batch = {
        **copy.deepcopy(canonical_batch),
        "decision_cycle_id": "hybrid-non-anchor",
        "market_timestamp": "2026-08-02T20:05:00Z",
        "_entry_quality_v2": hybrid_overlay(
            selected_symbol="ADAUSDT", selected_side="LONG"
        ),
    }
    confirm_direction(batch, "ADAUSDT", "LONG")
    live = selector.apply(batch)
    assert live["closed_bar_evaluation"] is True
    assert live["selected_symbols"] == ["ADAUSDT"]
    assert live["by_symbol"]["ADAUSDT"]["selected"] is True
    assert selector.health()["decision_records"] == 22


def test_cycle_identity_is_stable_when_same_closed_bar_is_revised(real_engine) -> None:
    first = real_engine.evaluate(synthetic_snapshot(variant=1))
    revised = real_engine.evaluate(synthetic_snapshot(variant=2))

    assert first["market_timestamp"] == revised["market_timestamp"]
    assert first["decision_cycle_id"] == revised["decision_cycle_id"]
    assert (
        first["results"]["ETHUSDT"]["feature_vector_hash"]
        != revised["results"]["ETHUSDT"]["feature_vector_hash"]
    )


def test_public_provider_waits_for_kline_finalization_delay() -> None:
    final_close = datetime(2026, 8, 3, 3, 0, tzinfo=timezone.utc)
    first_open = final_close - timedelta(minutes=5 * 96)
    rows = []
    for index in range(96):
        open_time = first_open + timedelta(minutes=5 * index)
        rows.append(
            [
                int(open_time.timestamp() * 1000),
                "100",
                "101",
                "99",
                "100.5",
                "1000",
            ]
        )

    class Response:
        is_redirect = False
        is_permanent_redirect = False

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return rows

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    provider = PublicKlineSnapshotProvider(
        session=Session(),
        now=lambda: final_close + timedelta(seconds=1),
    )
    candles = provider._candles("ETHUSDT")

    assert candles[-1].close_time == final_close - timedelta(minutes=5)


def test_service_never_redecides_an_already_processed_closed_bar(real_engine) -> None:
    class RevisingProvider:
        def __init__(self) -> None:
            self.snapshots = [
                synthetic_snapshot(variant=3),
                synthetic_snapshot(variant=4),
            ]
            self.calls = 0

        def snapshot(self):
            snapshot = self.snapshots[min(self.calls, len(self.snapshots) - 1)]
            self.calls += 1
            return snapshot

    provider = RevisingProvider()
    service = CurrentBrainDecisionService(real_engine, provider, cache_seconds=-1)
    first = service.predict("ETHUSDT", "first")
    revised = service.predict("ETHUSDT", "revised")

    assert provider.calls == 2
    assert first["features"]["vector_hash"] == revised["features"]["vector_hash"]
    assert (
        first["metadata"]["decision_cycle_id"]
        == revised["metadata"]["decision_cycle_id"]
    )


def test_hybrid_live_selects_multiple_quality_symbols_in_same_cycle(
    tmp_path: Path, canonical_batch
) -> None:
    selector = hybrid_selector(tmp_path)
    batch = {
        **copy.deepcopy(canonical_batch),
        "decision_cycle_id": "hybrid-multiple",
        "market_timestamp": "2026-08-02T20:10:00Z",
        "_entry_quality_v2": hybrid_overlay(
            selected_symbol="ADAUSDT", selected_side="LONG"
        ),
    }
    overlay = batch["_entry_quality_v2"]
    second = overlay["BTCUSDT"]["hybrid_directional_shadow"]["predictions"]["SHORT"]
    second.update(
        {
            "opportunity_probability": 0.75,
            "danger_probability": 0.2,
            "mae_q90": 0.004,
            "mfe_q50": 0.018,
            "net_return_mean": 0.008,
            "shadow_rank_score": 0.8,
        }
    )
    confirm_direction(batch, "ADAUSDT", "LONG")
    confirm_direction(batch, "BTCUSDT", "SHORT")

    live = selector.apply(batch)

    assert live["selected_count"] == 2
    assert set(live["selected_symbols"]) == {"ADAUSDT", "BTCUSDT"}
    assert live["selected_sides"] == {"ADAUSDT": "LONG", "BTCUSDT": "SHORT"}


def test_hybrid_live_abstains_when_best_rank_lacks_calibrated_quality(
    tmp_path: Path, canonical_batch
) -> None:
    selector = hybrid_selector(tmp_path)
    batch = {
        **copy.deepcopy(canonical_batch),
        "decision_cycle_id": "hybrid-weak-best",
        "market_timestamp": "2026-08-02T20:15:00Z",
        "_entry_quality_v2": hybrid_overlay(
            selected_symbol="XRPUSDT", selected_side="LONG"
        ),
    }
    weak = batch["_entry_quality_v2"]["XRPUSDT"]["hybrid_directional_shadow"][
        "predictions"
    ]["LONG"]
    weak.update(
        {
            "opportunity_probability": 0.4218,
            "danger_probability": 0.5870,
            "net_return_mean": -0.000992,
            "shadow_rank_score": 0.9,
        }
    )
    confirm_direction(batch, "XRPUSDT", "LONG")

    live = selector.apply(batch)

    assert live["selected_count"] == 0
    assessment = live["by_symbol"]["XRPUSDT"]["confirmation"]["LONG"]
    assert assessment["state"] == "ABSTAIN_WEAK_QUALITY"
    assert (
        "OPPORTUNITY_PROBABILITY_BELOW_CALIBRATED_MINIMUM" in assessment["reason_codes"]
    )


def test_hybrid_live_uses_lower_opportunity_minimum_only_for_short(
    tmp_path: Path, canonical_batch
) -> None:
    selector = hybrid_selector(tmp_path)
    batch = {
        **copy.deepcopy(canonical_batch),
        "decision_cycle_id": "hybrid-directional-opportunity-minimum",
        "market_timestamp": "2026-08-02T20:20:00Z",
        "_entry_quality_v2": hybrid_overlay(
            selected_symbol="ADAUSDT", selected_side="LONG"
        ),
    }
    long_prediction = batch["_entry_quality_v2"]["ADAUSDT"][
        "hybrid_directional_shadow"
    ]["predictions"]["LONG"]
    short_prediction = batch["_entry_quality_v2"]["BTCUSDT"][
        "hybrid_directional_shadow"
    ]["predictions"]["SHORT"]
    for prediction in (long_prediction, short_prediction):
        prediction.update(
            {
                "opportunity_probability": 0.42,
                "danger_probability": 0.2,
                "mae_q90": 0.004,
                "mfe_q50": 0.018,
                "net_return_mean": 0.008,
                "shadow_rank_score": 0.9,
            }
        )
    confirm_direction(batch, "ADAUSDT", "LONG")
    confirm_direction(batch, "BTCUSDT", "SHORT")

    live = selector.apply(batch)

    assert live["selected_symbols"] == ["BTCUSDT"]
    long_confirmation = live["by_symbol"]["ADAUSDT"]["confirmation"]["LONG"]
    short_confirmation = live["by_symbol"]["BTCUSDT"]["confirmation"]["SHORT"]
    assert long_confirmation["minimum_opportunity_probability"] == 0.44
    assert short_confirmation["minimum_opportunity_probability"] == 0.39
    assert long_confirmation["state"] == "ABSTAIN_WEAK_QUALITY"
    assert short_confirmation["state"] == "CONFIRMED"


@pytest.mark.anyio
async def test_real_pipeline_and_http_adapter_are_lossless(
    real_engine, current_snapshot, canonical_batch
) -> None:
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


@pytest.mark.anyio
async def test_twenty_distinct_snapshots_preserve_http_parity_and_feature_driven_selection(
    real_engine,
) -> None:
    feature_hashes = {symbol: set() for symbol in CANONICAL_SYMBOLS}
    calibrated_scores = {symbol: set() for symbol in CANONICAL_SYMBOLS}
    probability_vectors = {symbol: set() for symbol in CANONICAL_SYMBOLS}
    decisions = {symbol: set() for symbol in CANONICAL_SYMBOLS}

    for variant in range(20):
        snapshot = synthetic_snapshot(variant)
        canonical = real_engine.evaluate(snapshot)
        service = CurrentBrainDecisionService(
            real_engine, StaticProvider(snapshot), cache_seconds=60
        )
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            for symbol in CANONICAL_SYMBOLS:
                response = await client.post("/ml-v2/predict", json={"symbol": symbol})
                assert response.status_code == 200
                payload = response.json()
                expected = compatibility_response(
                    canonical, symbol, payload["metadata"]["trace_id"]
                )
                assert payload == expected

                result = canonical["results"][symbol]
                feature_hashes[symbol].add(result["feature_vector_hash"])
                calibrated_scores[symbol].add(result["candidate"]["calibrated_score"])
                probability_vectors[symbol].add(
                    (
                        payload["long_prob"],
                        payload["short_prob"],
                        payload["neutral_prob"],
                    )
                )
                decisions[symbol].add(payload["aegis"]["decision_brain"]["decision"])

    assert all(len(values) == 20 for values in feature_hashes.values())
    assert all(len(values) == 20 for values in calibrated_scores.values())
    # The qualified artifact has one deliberately SHORT-only directional estimator.
    # Entry variability comes from the feature-driven layers and global selection.
    assert all(len(values) == 1 for values in probability_vectors.values())
    assert any("ENTER_NOW" in values for values in decisions.values())


@pytest.mark.parametrize("symbol", CANONICAL_SYMBOLS)
def test_all_configured_symbols_have_valid_current_outputs(
    canonical_batch, symbol
) -> None:
    response = compatibility_response(canonical_batch, symbol, "trace")
    probabilities = (
        response["long_prob"],
        response["short_prob"],
        response["neutral_prob"],
    )
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)
    assert response["features"]["fallback"] is False
    assert response["features"]["schema"] == "aegis-features-v2"
    assert response["features"]["count"] == 83
    brain = response["aegis"]["decision_brain"]
    assert brain["decision"] in {"ENTER_NOW", "DO_NOT_ENTER"}
    assert brain["contract_version"] == CANONICAL_DECISION_CONTRACT
    assert brain["authority"] == CANONICAL_LIVE_AUTHORITY
    assert brain["selected"] is brain["execute"]
    assert brain["model_sha256"] == MODEL_ARTIFACT_SHA256
    assert brain["bundle_sha256"] == MODEL_BUNDLE_SHA256
    assert brain["configuration_sha256"] == CONFIGURATION_SHA256
    assert brain["feature_schema"] == FEATURE_SCHEMA
    assert brain["feature_count"] == FEATURE_COUNT
    assert brain["fallback"] is False


def test_single_estimator_is_not_inflated_into_legacy_consensus(
    canonical_batch,
) -> None:
    for symbol in CANONICAL_SYMBOLS:
        response = compatibility_response(canonical_batch, symbol, "trace")
        raw = response["aegis"]["turbo"]["raw"]
        votes = raw["votes"]
        evidence = response["aegis"]["directional_evidence"]
        assert sum(votes.values()) == 1
        assert raw["vote_semantics"] == "SINGLE_DIRECTIONAL_ESTIMATOR_OUTPUT"
        assert raw["directional_member_count"] == 1
        assert raw["independent_directional_votes"] == "NOT_APPLICABLE"
        assert raw["directional_consensus"] == "NOT_APPLICABLE_SINGLE_ESTIMATOR"
        assert evidence == {
            "schema_id": "aegis-directional-evidence-v1",
            "semantics": "SINGLE_DIRECTIONAL_ESTIMATOR_OUTPUT",
            "eligible_directional_members": 1,
            "independent_directional_votes": "NOT_APPLICABLE",
            "directional_consensus": "NOT_APPLICABLE_SINGLE_ESTIMATOR",
            "fabricated_votes": 0,
            "selection_authority": "CANONICAL_PYTHON_SELECTED",
        }
        assert response["metadata"]["directional_estimator_count"] == 1
        assert response["metadata"]["fabricated_votes"] == 0
        assert response["metadata"]["leverage_recommendation"] == "NOT_PRESENT"
        assert response["metadata"]["position_fraction"] == "NOT_PRESENT"
        assert response["aegis"]["candidate_uncertainty"] == {
            "schema_id": "aegis-candidate-uncertainty-v1",
            "value": None,
            "confidence": None,
            "semantics": "NOT_APPLICABLE_SINGLE_ESTIMATOR",
            "selection_effect": "NONE",
        }


def test_canonical_batch_exposes_complete_ranking_without_changing_selection(
    canonical_batch,
) -> None:
    assert len(canonical_batch["ranking"]) == len(CANONICAL_SYMBOLS)
    assert [row["rank"] for row in canonical_batch["ranking"]] == list(
        range(1, len(CANONICAL_SYMBOLS) + 1)
    )
    selected = {
        symbol
        for symbol, result in canonical_batch["results"].items()
        if result["selected"]
    }
    ranked_selected = {
        row["symbol"] for row in canonical_batch["ranking"] if row["eligible"]
    }
    assert selected <= ranked_selected


def test_specialized_committees_remain_non_promotable_shadow_observers() -> None:
    import yaml

    root = Path(__file__).parents[2]
    v2 = yaml.safe_load((root / "config/committee_v2_shadow.yaml").read_text())
    v21 = yaml.safe_load((root / "config/committee_v21_shadow.yaml").read_text())

    assert v2["mode"] == "SHADOW"
    assert v2["meta_selector"]["fabricated_votes_prohibited"] is True
    assert v2["meta_selector"]["majority_vote_prohibited"] is True
    assert v2["promotion"]["automatic_promotion"] is False
    assert v2["promotion"]["live_authority"] is False
    assert v21["mode"] == "SHADOW"
    assert v21["runtime_authority"] == "OBSERVATIONAL_ONLY"
    assert v21["authority"]["automatic_promotion"] is False
    assert v21["counterfactual"]["fabricated_votes_prohibited"] is True


def test_would_execute_and_action_match_canonical_selection(canonical_batch) -> None:
    for symbol, canonical in canonical_batch["results"].items():
        response = compatibility_response(canonical_batch, symbol, "trace")
        raw = response["aegis"]["turbo"]["raw"]
        assert raw["would_execute"] is canonical["selected"]
        assert (raw["action"] in {"LONG", "SHORT"}) is canonical["selected"]
        assert raw["turbo_score"] == canonical["candidate"]["raw_score"]
        assert (
            response["metadata"]["canonical_calibrated_score"]
            == canonical["candidate"]["calibrated_score"]
        )


@pytest.mark.anyio
async def test_unknown_symbol_and_malformed_request_fail_safely(
    real_engine, current_snapshot
) -> None:
    service = CurrentBrainDecisionService(real_engine, StaticProvider(current_snapshot))
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (
            await client.post("/ml-v2/predict", json={"symbol": "UNKNOWN"})
        ).status_code == 422
        assert (
            await client.post(
                "/ml-v2/predict", json={"symbol": "ETHUSDT", "extra": True}
            )
        ).status_code == 422
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
        assert (
            response.json()["detail"] == "AEGIS_CURRENT_BRAIN_EXIT_SIGNAL_NOT_PRESENT"
        )


def test_live_http_modules_have_no_binance_mutation_surface() -> None:
    import aegis.live_api as api_module
    import aegis.live_decision as decision_module
    import aegis.hybrid_live_experiment as hybrid_module

    source = (
        inspect.getsource(api_module)
        + inspect.getsource(decision_module)
        + inspect.getsource(hybrid_module)
    ).lower()
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


def test_pm2_definition_uses_local_module_and_no_credentials() -> None:
    text = (Path(__file__).parents[2] / "ecosystem.config.js").read_text(
        encoding="utf-8"
    )
    assert "-m aegis.live_api --host 127.0.0.1 --port 8001" in text
    python_block = text.split('name: "02-Aegis-API"', 1)[1]
    assert "BINANCE_API" not in python_block
    assert '"PYTHONPATH": "/home/jasan/Develop/trading_system/src"' in python_block


def test_live_api_uses_single_lazy_application_factory() -> None:
    source = (Path(__file__).parents[2] / "src/aegis/live_api.py").read_text(
        encoding="utf-8"
    )
    assert "app = create_app()" not in source
    assert '"aegis.live_api:create_app"' in source
    assert "factory=True" in source
