from datetime import datetime, timedelta, timezone

import pytest

from aegis.config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH
from aegis.domain import Candle, DecisionRequest, FeedQuality, MarketSnapshot, PortfolioContext, SymbolSeries
from aegis.decision import GlobalSelectionPolicy
from aegis.models import (
    BundleMetadata, CalibrationBlock, CalibrationMethod, CalibratorSpec,
    DeterministicModelRuntime, EstimatorSpec, LinearHead, ModelBundle, QuantileHeadSpec,
)
from aegis.features import FEATURE_HASH, FEATURE_SCHEMA_VERSION, FrozenNormalizer


@pytest.fixture
def snapshot_factory():
    def build(*, bars: int = 60, closed_at: datetime | None = None, available_slots: int = 1) -> MarketSnapshot:
        end = closed_at or datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
        series = []
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
                candles.append(Candle(open_time, close_time, open_price, high, low, close,
                                      1000.0 + symbol_index * 10 + index, True, "OFFLINE_FIXTURE", str(index)))
            series.append(SymbolSeries(symbol, tuple(candles), end, FeedQuality()))
        return MarketSnapshot(end, "5m", CANONICAL_SYMBOL_SET_HASH, tuple(reversed(series)),
                              PortfolioContext(available_slots=available_slots, operational_time=end))
    return build


@pytest.fixture
def decision_request(snapshot_factory):
    return DecisionRequest("request-1", "cycle-1", "aegis-decision-request-v1",
                           "aegis-clean-rebuild-v1", "aegis-scientific-config-v1", snapshot_factory())


@pytest.fixture
def scenario_bundle_factory():
    def build(side: str) -> ModelBundle:
        if side not in {"LONG", "SHORT", "NEUTRAL"}:
            raise ValueError(side)
        directional = 9.0 if side != "NEUTRAL" else -9.0
        other = -9.0
        long_bias = directional if side == "LONG" else other
        short_bias = directional if side == "SHORT" else other
        neutral_bias = 9.0 if side == "NEUTRAL" else -9.0
        empty = {}
        estimator = EstimatorSpec(
            model_id=f"fixture-{side.lower()}-h12", horizon_bars=12,
            long=LinearHead(long_bias, empty), short=LinearHead(short_bias, empty), neutral=LinearHead(neutral_bias, empty),
            expected_return=LinearHead(0.03 if side == "LONG" else -0.03 if side == "SHORT" else 0.0, empty),
            tail_risk=LinearHead(-9.0, empty), qmae_mean=LinearHead(0.0, empty), quality=LinearHead(9.0, empty),
            qmae_quantiles=QuantileHeadSpec(
                LinearHead(0.0, empty), LinearHead(0.0, empty), 0.0, 0.90,
            ),
        )
        identity = CalibratorSpec(CalibrationMethod.IDENTITY, 0.01, 0.01, 100)
        return ModelBundle(
            bundle_id=f"fixture-{side.lower()}-bundle", schema_version="aegis-model-bundle-v1",
            feature_schema_version=FEATURE_SCHEMA_VERSION, feature_hash=FEATURE_HASH,
            universe_id="aegis-operational-eleven-v1", symbol_set_hash=CANONICAL_SYMBOL_SET_HASH,
            timeframe="5m", approved=True, content_hash="f" * 64,
            normalizer=FrozenNormalizer(), estimators=(estimator,),
            metadata=BundleMetadata(
                "TEST_FIXTURE", True, None, None, None, 0, "fixture", "1", "tests", "FIXED_FIXTURE", 39,
                {
                    "direction": 0.50,
                    "selection": 0.45,
                    "trrm_max_tail_probability": 0.70,
                    "qmae_max_fraction": 0.03,
                    "eqm_min_score": 0.0,
                },
            ),
            calibration=CalibrationBlock(
                "aegis-calibration-v1", True, identity, identity, identity, identity, identity,
            ),
        )
    return build


@pytest.fixture
def scenario_runtime_factory(scenario_bundle_factory):
    from pathlib import Path
    from aegis.runtime import build_runtime
    from aegis.utils import FixedUtcClock

    def build(side: str, snapshot: MarketSnapshot):
        runtime = build_runtime(Path(__file__).parents[1] / "config", clock=FixedUtcClock(snapshot.closed_at))
        runtime.models = DeterministicModelRuntime(scenario_bundle_factory(side), 0.50)
        runtime.config = __import__("dataclasses").replace(
            runtime.config,
            models=__import__("dataclasses").replace(runtime.config.models, model_bundle_id=f"fixture-{side.lower()}-bundle"),
        )
        runtime.selection_policy = GlobalSelectionPolicy(0.45)
        return runtime
    return build
