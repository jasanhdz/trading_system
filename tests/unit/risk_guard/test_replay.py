"""Deterministic replay test — validates E4 Tail Risk Guard over historical signals."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from aegis.risk_guard.domain import RiskGuardConfig, RiskDecision, RiskGuardVerdict, FROZEN_TAIL_RISK_THRESHOLD
from aegis.risk_guard.e4_tail_risk_guard import E4TailRiskGuard
from aegis.risk_guard.replay import DeterministicReplay


REPO_ROOT = Path(__file__).resolve().parents[3]
E4_MODELS_PATH = REPO_ROOT / "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/artifacts/run_01/development_models.joblib"
E4_SCHEMA_PATH = REPO_ROOT / "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/artifacts/dataset_v1/feature_schema.json"
SCORED_SIGNALS_PATH = REPO_ROOT / "sandbox/aegis_strategy_router/experiments/aegis_e4_risk_guard_validation_v1/artifacts/dataset_v1/causal_scored_signals.parquet"
THRESHOLDS_PATH = REPO_ROOT / "sandbox/aegis_strategy_router/experiments/aegis_e4_risk_guard_validation_v1/config/thresholds_frozen_v1.json"
DEVELOPMENT_LABELED_PATH = REPO_ROOT / "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/artifacts/dataset_v1/development_labeled.parquet"
SCHEMA_SHA256 = "9f86bf95bd78508698a5a1eac9147becaae48565aca6f3fcc1a8e0597d5ba1f2"


def _artifacts_exist() -> bool:
    return all(p.exists() for p in [E4_MODELS_PATH, E4_SCHEMA_PATH, SCORED_SIGNALS_PATH, THRESHOLDS_PATH])


def _dev_labeled_exists() -> bool:
    return DEVELOPMENT_LABELED_PATH.exists()


@pytest.mark.skipif(not _artifacts_exist(), reason="E4 artifacts not available")
class TestDeterministicReplay:
    """Replay historical signals through the frozen E4 Tail Risk Guard."""

    def _load_config(self) -> RiskGuardConfig:
        thresholds = json.loads(THRESHOLDS_PATH.read_text())
        tail_threshold = thresholds["thresholds"]["tail_risk_guard"]["threshold"]
        models_hash = E4TailRiskGuard._sha256_file(E4_MODELS_PATH)

        return RiskGuardConfig(
            enabled=True,
            mode="enforce",
            tail_risk_threshold=tail_threshold,
            models_joblib_path=str(E4_MODELS_PATH),
            models_joblib_sha256=models_hash,
            feature_schema_path=str(E4_SCHEMA_PATH),
            feature_schema_sha256=SCHEMA_SHA256,
            fail_closed=True,
        )

    def test_guard_loads_successfully(self):
        config = self._load_config()
        guard = E4TailRiskGuard(config)
        guard.load()
        assert guard.is_available()

    def test_guard_rejects_wrong_hash(self):
        config = self._load_config()
        config = RiskGuardConfig(
            enabled=True,
            mode="enforce",
            tail_risk_threshold=config.tail_risk_threshold,
            models_joblib_path=config.models_joblib_path,
            models_joblib_sha256="wrong_hash",
            fail_closed=True,
        )
        guard = E4TailRiskGuard(config)
        with pytest.raises(RuntimeError, match="HASH_MISMATCH"):
            guard.load()

    def test_guard_rejects_wrong_schema_hash(self):
        config = self._load_config()
        config = RiskGuardConfig(
            enabled=True,
            mode="enforce",
            tail_risk_threshold=config.tail_risk_threshold,
            models_joblib_path=config.models_joblib_path,
            models_joblib_sha256=config.models_joblib_sha256,
            feature_schema_path=config.feature_schema_path,
            feature_schema_sha256="wrong_schema_hash",
            fail_closed=True,
        )
        guard = E4TailRiskGuard(config)
        with pytest.raises(RuntimeError, match="SCHEMA_HASH_MISMATCH"):
            guard.load()

    def test_replay_validation_signals(self):
        config = self._load_config()
        replay = DeterministicReplay(config)
        replay.initialize()

        df = pd.read_parquet(SCORED_SIGNALS_PATH)
        val = df[df["split"] == "VALIDATION"].copy()

        assert len(val) == 108, f"Expected 108 VALIDATION signals, got {len(val)}"
        assert (val["side"] == "SHORT").all(), "VALIDATION should be 100% SHORT"

        results = replay.replay_signals(val, pre_computed_scores=val["e4_tail_risk_score"])

        assert len(results) == 108
        assert "tail_risk_score" in results.columns
        assert "risk_decision" in results.columns
        assert "verdict" in results.columns

        allow_count = (results["risk_decision"] == "ALLOW").sum()
        block_count = (results["risk_decision"] == "BLOCK").sum()
        assert allow_count + block_count == 108

    def test_replay_matches_prior_experiment(self):
        config = self._load_config()
        replay = DeterministicReplay(config)
        replay.initialize()

        df = pd.read_parquet(SCORED_SIGNALS_PATH)
        val = df[df["split"] == "VALIDATION"].copy()

        results = replay.replay_signals(val, pre_computed_scores=val["e4_tail_risk_score"])

        prior_threshold = config.tail_risk_threshold
        prior_blocks = (val["e4_tail_risk_score"] >= prior_threshold).sum()
        new_blocks = (results["risk_decision"] == "BLOCK").sum()

        assert new_blocks == prior_blocks, (
            f"Replay block count {new_blocks} != prior experiment {prior_blocks}"
        )

    def test_fail_closed_on_missing_features(self):
        config = self._load_config()
        replay = DeterministicReplay(config)
        replay.initialize()

        df = pd.read_parquet(SCORED_SIGNALS_PATH)
        val = df[df["split"] == "VALIDATION"].head(5).copy()

        for col in [c for c in val.columns if c.startswith("feature__")]:
            val[col] = float("nan")

        results = replay.replay_signals(val)

        assert (results["risk_decision"] == "BLOCK").all()

    def test_metrics_after_replay(self):
        config = self._load_config()
        replay = DeterministicReplay(config)
        replay.initialize()

        df = pd.read_parquet(SCORED_SIGNALS_PATH)
        val = df[df["split"] == "VALIDATION"].copy()

        replay.replay_signals(val, pre_computed_scores=val["e4_tail_risk_score"])
        summary = replay.metrics_summary()

        assert summary["total_evaluated"] == 108
        assert summary["total_blocked"] > 0
        assert summary["block_rate_pct"] > 0
        assert summary["block_rate_pct"] < 100


@pytest.mark.skipif(not _artifacts_exist(), reason="E4 artifacts not available")
class TestScoreValidation:
    """Test score validation edge cases."""

    def _load_config(self) -> RiskGuardConfig:
        thresholds = json.loads(THRESHOLDS_PATH.read_text())
        tail_threshold = thresholds["thresholds"]["tail_risk_guard"]["threshold"]
        models_hash = E4TailRiskGuard._sha256_file(E4_MODELS_PATH)
        return RiskGuardConfig(
            enabled=True,
            mode="enforce",
            tail_risk_threshold=tail_threshold,
            models_joblib_path=str(E4_MODELS_PATH),
            models_joblib_sha256=models_hash,
            feature_schema_path=str(E4_SCHEMA_PATH),
            feature_schema_sha256=SCHEMA_SHA256,
            fail_closed=True,
        )

    def test_score_validation_rejects_nan(self):
        """Score validation must reject NaN scores."""
        from datetime import datetime, timezone
        from aegis.risk_guard.domain import Signal, Direction

        config = self._load_config()
        guard = E4TailRiskGuard(config)
        guard.load()

        signal = Signal(
            signal_id="test-nan",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            side=Direction.SHORT,
            direction_source="TEST",
            direction_model_version="TEST",
        )

        result = guard.evaluate(signal, {"features": {"invalid": "data"}})
        assert result.decision == RiskDecision.BLOCK

    def test_score_validation_rejects_out_of_range(self):
        """Score validation must reject scores outside [0, 1]."""
        assert FROZEN_TAIL_RISK_THRESHOLD >= 0.0
        assert FROZEN_TAIL_RISK_THRESHOLD <= 1.0

        assert E4TailRiskGuard._validate_score(0.5) is True
        assert E4TailRiskGuard._validate_score(0.0) is True
        assert E4TailRiskGuard._validate_score(1.0) is True
        assert E4TailRiskGuard._validate_score(-0.1) is False
        assert E4TailRiskGuard._validate_score(1.1) is False
        assert E4TailRiskGuard._validate_score(float("nan")) is False
        assert E4TailRiskGuard._validate_score(float("inf")) is False


@pytest.mark.skipif(not _artifacts_exist(), reason="E4 artifacts not available")
@pytest.mark.skipif(not _dev_labeled_exists(), reason="development_labeled.parquet not available")
class TestRealE4Inference:
    """End-to-end E4 inference parity test.

    Feeds the 146 E4 features from development_labeled.parquet directly
    into the frozen E4 model and verifies:
    1. All scores are finite and in [0, 1]
    2. Scores produce deterministic ALLOW/BLOCK decisions
    3. Score distribution is consistent (not all same value)
    """

    def _load_config(self) -> RiskGuardConfig:
        thresholds = json.loads(THRESHOLDS_PATH.read_text())
        tail_threshold = thresholds["thresholds"]["tail_risk_guard"]["threshold"]
        models_hash = E4TailRiskGuard._sha256_file(E4_MODELS_PATH)
        return RiskGuardConfig(
            enabled=True,
            mode="enforce",
            tail_risk_threshold=tail_threshold,
            models_joblib_path=str(E4_MODELS_PATH),
            models_joblib_sha256=models_hash,
            feature_schema_path=str(E4_SCHEMA_PATH),
            feature_schema_sha256=SCHEMA_SHA256,
            fail_closed=True,
        )

    def test_feature_bridge_roundtrip(self):
        """FeatureBridge.from_dataframe_row → E4 model → valid score."""
        from datetime import datetime, timezone
        from aegis.risk_guard.domain import Signal, Direction
        from aegis.risk_guard.feature_bridge import FeatureBridge

        config = self._load_config()
        guard = E4TailRiskGuard(config)
        guard.load()

        bridge = FeatureBridge(guard._tail_bundle["features"])

        dev = pd.read_parquet(DEVELOPMENT_LABELED_PATH)
        train = dev[dev["split"] == "TRAIN"].head(50)

        for _, row in train.iterrows():
            feature_row = bridge.from_dataframe_row(
                row,
                symbol=str(row["symbol"]),
                side=str(row["side"]),
            )

            signal = Signal(
                signal_id="parity-test",
                timestamp=datetime.now(timezone.utc),
                symbol=feature_row.symbol,
                side=Direction.SHORT,
                direction_source="TEST",
                direction_model_version="TEST",
            )

            result = guard.evaluate(signal, {"features": feature_row.features})

            assert result.decision in (RiskDecision.ALLOW, RiskDecision.BLOCK)
            assert E4TailRiskGuard._validate_score(result.score), (
                f"Invalid score: {result.score}"
            )

    def test_inference_scores_finite_and_valid(self):
        """All 146 features → E4 model → finite scores in [0, 1]."""
        import numpy as np
        from datetime import datetime, timezone
        from aegis.risk_guard.domain import Signal, Direction

        config = self._load_config()
        guard = E4TailRiskGuard(config)
        guard.load()

        dev = pd.read_parquet(DEVELOPMENT_LABELED_PATH)
        tail_features = guard._tail_bundle["features"]

        sample = pd.concat([
            dev[dev["split"] == "TRAIN"].head(100),
            dev[dev["split"] == "VALIDATION"].head(100),
        ]).reset_index(drop=True)

        scores = []
        for _, row in sample.iterrows():
            feat_dict = {name: float(row[name]) for name in tail_features if name in row.index}
            feat_df = pd.DataFrame([feat_dict])
            raw = guard._tail_bundle["model"].decision_function(feat_df).reshape(-1, 1)
            score = float(guard._tail_bundle["calibrator"].predict_proba(raw)[:, 1][0])
            scores.append(score)

        assert len(scores) == 200
        assert all(np.isfinite(s) for s in scores), "Some scores are not finite"
        assert all(0.0 <= s <= 1.0 for s in scores), "Some scores out of [0,1] range"

        assert np.std(scores) > 0.01, f"Scores too uniform (std={np.std(scores):.4f})"

        threshold = config.tail_risk_threshold
        blocks = sum(1 for s in scores if s >= threshold)
        assert blocks > 0, f"Expected at least some blocks, got {blocks}/{len(scores)}"

    def test_score_determinism(self):
        """Same features → same score (deterministic inference)."""
        from datetime import datetime, timezone
        from aegis.risk_guard.domain import Signal, Direction

        config = self._load_config()
        guard = E4TailRiskGuard(config)
        guard.load()

        dev = pd.read_parquet(DEVELOPMENT_LABELED_PATH)
        tail_features = guard._tail_bundle["features"]

        row = dev[dev["split"] == "TRAIN"].iloc[0]
        feat_dict = {name: float(row[name]) for name in tail_features if name in row.index}
        feat_df = pd.DataFrame([feat_dict])

        scores = []
        for _ in range(5):
            raw = guard._tail_bundle["model"].decision_function(feat_df).reshape(-1, 1)
            score = float(guard._tail_bundle["calibrator"].predict_proba(raw)[:, 1][0])
            scores.append(score)

        assert len(set(scores)) == 1, f"Scores not deterministic: {scores}"


class TestFeatureBridgeValidation:
    """Test FeatureBridge input validation contracts."""

    def test_requires_feature_names(self):
        from aegis.risk_guard.feature_bridge import FeatureBridge
        with pytest.raises(ValueError, match="feature_names must be a non-empty list"):
            FeatureBridge([])

    def test_rejects_nan_features(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        bridge = FeatureBridge(["f1", "f2", "f3"])
        with pytest.raises(ValueError, match="Non-finite"):
            bridge.from_feature_dict(
                {"f1": 1.0, "f2": float("nan"), "f3": 3.0},
                symbol="BTCUSDT", side="LONG",
                timestamp=datetime.now(timezone.utc),
            )

    def test_rejects_inf_features(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        bridge = FeatureBridge(["f1", "f2", "f3"])
        with pytest.raises(ValueError, match="Non-finite"):
            bridge.from_feature_dict(
                {"f1": 1.0, "f2": float("inf"), "f3": 3.0},
                symbol="BTCUSDT", side="LONG",
                timestamp=datetime.now(timezone.utc),
            )

    def test_rejects_missing_features(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        bridge = FeatureBridge(["f1", "f2", "f3"])
        with pytest.raises(ValueError, match="Missing"):
            bridge.from_feature_dict(
                {"f1": 1.0},
                symbol="BTCUSDT", side="LONG",
                timestamp=datetime.now(timezone.utc),
            )

    def test_rejects_empty_symbol(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        bridge = FeatureBridge(["f1", "f2", "f3"])
        with pytest.raises(ValueError, match="symbol is required"):
            bridge.from_feature_dict(
                {"f1": 0.5, "f2": -1.0, "f3": 3.14},
                symbol="", side="LONG",
                timestamp=datetime.now(timezone.utc),
            )

    def test_rejects_invalid_side(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        bridge = FeatureBridge(["f1", "f2", "f3"])
        with pytest.raises(ValueError, match="side must be 'LONG' or 'SHORT'"):
            bridge.from_feature_dict(
                {"f1": 0.5, "f2": -1.0, "f3": 3.14},
                symbol="BTCUSDT", side="MEDIUM",
                timestamp=datetime.now(timezone.utc),
            )

    def test_accepts_valid_features(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        bridge = FeatureBridge(["f1", "f2", "f3"])
        row = bridge.from_feature_dict(
            {"f1": 0.5, "f2": -1.0, "f3": 3.14},
            symbol="BTCUSDT", side="SHORT",
            timestamp=datetime.now(timezone.utc),
        )
        assert len(row.features) == 3
        assert row.feature_hash != ""
        assert row.symbol == "BTCUSDT"
        assert row.side == "SHORT"


class TestMarketCandlesValidation:
    """Test from_market_candles() contracts."""

    def test_rejects_invalid_side(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        bridge = FeatureBridge(["f1"])
        with pytest.raises(ValueError, match="side must be 'LONG' or 'SHORT'"):
            bridge.from_market_candles(
                candles_by_symbol={"BTCUSDT": pd.DataFrame()},
                target_symbol="BTCUSDT",
                side="MEDIUM",
                decision_at=datetime.now(timezone.utc),
            )

    def test_rejects_out_of_universe_symbol(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        bridge = FeatureBridge(["f1"])
        with pytest.raises(ValueError, match="not in E4 universe"):
            bridge.from_market_candles(
                candles_by_symbol={"BTCUSDT": pd.DataFrame()},
                target_symbol="FAKEUSDT",
                side="LONG",
                decision_at=datetime.now(timezone.utc),
            )

    def test_rejects_missing_symbols(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        bridge = FeatureBridge(["f1"])
        with pytest.raises(ValueError, match="MISSING_SYMBOL"):
            bridge.from_market_candles(
                candles_by_symbol={"BTCUSDT": pd.DataFrame()},
                target_symbol="BTCUSDT",
                side="LONG",
                decision_at=datetime.now(timezone.utc),
            )

    def test_rejects_duplicate_minutes(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        import numpy as np
        bridge = FeatureBridge(["f1"])

        bad_candle = pd.DataFrame({
            "open_time_ms": [1000, 1000],
            "open": [1.0, 1.0], "high": [1.0, 1.0],
            "low": [1.0, 1.0], "close": [1.0, 1.0],
            "volume": [1.0, 1.0], "taker_buy_volume": [0.5, 0.5],
        })
        candles = {s: bad_candle.copy() for s in [
            "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
            "ETHUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT",
        ]}
        with pytest.raises(ValueError, match="CANDLE_DUPLICATE_MINUTE"):
            bridge.from_market_candles(
                candles_by_symbol=candles,
                target_symbol="BTCUSDT",
                side="LONG",
                decision_at=datetime.now(timezone.utc),
            )

    def test_rejects_minute_gap(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import FeatureBridge
        bridge = FeatureBridge(["f1"])

        gap_candle = pd.DataFrame({
            "open_time_ms": [1000, 120_000],
            "open": [1.0, 1.0], "high": [1.0, 1.0],
            "low": [1.0, 1.0], "close": [1.0, 1.0],
            "volume": [1.0, 1.0], "taker_buy_volume": [0.5, 0.5],
        })
        candles = {s: gap_candle.copy() for s in [
            "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
            "ETHUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT",
        ]}
        with pytest.raises(ValueError, match="CANDLE_MINUTE_GAP"):
            bridge.from_market_candles(
                candles_by_symbol=candles,
                target_symbol="BTCUSDT",
                side="LONG",
                decision_at=datetime.now(timezone.utc),
            )

    def test_build_anchors_generates_window(self):
        from datetime import datetime, timezone
        from aegis.risk_guard.feature_bridge import build_anchors, ANCHOR_COUNT, ANCHOR_CADENCE_MINUTES
        import pandas as pd

        decision_at = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
        anchors = build_anchors(decision_at)

        assert len(anchors) == ANCHOR_COUNT + 1
        expected_ts = pd.Timestamp(decision_at).tz_convert("UTC")
        assert anchors[-1] == expected_ts
        expected_start = expected_ts - pd.Timedelta(minutes=ANCHOR_COUNT * ANCHOR_CADENCE_MINUTES)
        assert anchors[0] == expected_start

    def test_frozen_universe_has_11_symbols(self):
        from aegis.risk_guard.feature_bridge import FROZEN_E4_UNIVERSE
        assert len(FROZEN_E4_UNIVERSE) == 11
        assert "BTCUSDT" in FROZEN_E4_UNIVERSE
        assert "ETHUSDT" in FROZEN_E4_UNIVERSE

    def test_frozen_timeframes(self):
        from aegis.risk_guard.feature_bridge import FROZEN_E4_TIMEFROZEN
        assert FROZEN_E4_TIMEFROZEN == [5, 15, 60, 240]
