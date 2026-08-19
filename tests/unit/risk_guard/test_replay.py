"""Deterministic replay test — validates E4 Tail Risk Guard over historical signals."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from aegis.risk_guard.domain import RiskGuardConfig, RiskDecision, RiskGuardVerdict
from aegis.risk_guard.e4_tail_risk_guard import E4TailRiskGuard
from aegis.risk_guard.replay import DeterministicReplay


REPO_ROOT = Path(__file__).resolve().parents[3]
E4_MODELS_PATH = REPO_ROOT / "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/artifacts/run_01/development_models.joblib"
E4_SCHEMA_PATH = REPO_ROOT / "sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/artifacts/dataset_v1/feature_schema.json"
SCORED_SIGNALS_PATH = REPO_ROOT / "sandbox/aegis_strategy_router/experiments/aegis_e4_risk_guard_validation_v1/artifacts/dataset_v1/causal_scored_signals.parquet"
THRESHOLDS_PATH = REPO_ROOT / "sandbox/aegis_strategy_router/experiments/aegis_e4_risk_guard_validation_v1/config/thresholds_frozen_v1.json"
SCHEMA_SHA256 = "9f86bf95bd78508698a5a1eac9147becaae48565aca6f3fcc1a8e0597d5ba1f2"


def _artifacts_exist() -> bool:
    return all(p.exists() for p in [E4_MODELS_PATH, E4_SCHEMA_PATH, SCORED_SIGNALS_PATH, THRESHOLDS_PATH])


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
class TestRealE4Inference:
    """Test real E4 model inference from feature rows — NOT pre-computed scores.

    This verifies that the frozen model produces the same scores as the
    prior experiment when given the same feature rows.
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

    def test_inference_matches_prior_scores(self):
        """Run model inference from features and compare to pre-computed scores.

        Uses the side_panel feature rows that the prior experiment used.
        The scores must match within floating-point tolerance.
        """
        from datetime import datetime, timezone
        from aegis.risk_guard.domain import Signal, Direction

        config = self._load_config()
        guard = E4TailRiskGuard(config)
        guard.load()

        df = pd.read_parquet(SCORED_SIGNALS_PATH)
        val = df[df["split"] == "VALIDATION"].copy()

        # Check which feature columns are available
        feature_cols = [c for c in val.columns if c.startswith("feature__")]
        if not feature_cols:
            pytest.skip("No feature__ columns in parquet — cannot test real inference")

        threshold = config.tail_risk_threshold
        prior_blocks = (val["e4_tail_risk_score"] >= threshold).sum()
        real_blocks = 0
        mismatches = 0

        for _, row in val.iterrows():
            signal = Signal(
                signal_id=str(row.get("signal_id", "test")),
                timestamp=row.get("signal_timestamp", datetime.now(timezone.utc)),
                symbol=str(row["symbol"]),
                side=Direction.SHORT,
                direction_source="TEST",
                direction_model_version="TEST",
            )

            # Build feature dict from feature__ columns
            features = {}
            for col in feature_cols:
                features[col] = row[col]

            result = guard.evaluate(signal, {"features": features})

            if result.decision == RiskDecision.BLOCK:
                real_blocks += 1

            # Compare to pre-computed score
            prior_score = float(row["e4_tail_risk_score"])
            if abs(result.score - prior_score) > 0.01:
                mismatches += 1

        assert real_blocks == prior_blocks, (
            f"Real inference blocks {real_blocks} != prior {prior_blocks}"
        )
        assert mismatches == 0, (
            f"{mismatches} signals had score mismatch > 0.01"
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

        # NaN score should fail-closed
        result = guard.evaluate(signal, {"features": {"invalid": "data"}})
        assert result.decision == RiskDecision.BLOCK

    def test_score_validation_rejects_out_of_range(self):
        """Score validation must reject scores outside [0, 1]."""
        from aegis.risk_guard.flags import FROZEN_TAIL_RISK_THRESHOLD

        # A valid score outside [0,1] should be caught
        assert FROZEN_TAIL_RISK_THRESHOLD >= 0.0
        assert FROZEN_TAIL_RISK_THRESHOLD <= 1.0

        # Validate the static method directly
        assert E4TailRiskGuard._validate_score(0.5) is True
        assert E4TailRiskGuard._validate_score(0.0) is True
        assert E4TailRiskGuard._validate_score(1.0) is True
        assert E4TailRiskGuard._validate_score(-0.1) is False
        assert E4TailRiskGuard._validate_score(1.1) is False
        assert E4TailRiskGuard._validate_score(float("nan")) is False
        assert E4TailRiskGuard._validate_score(float("inf")) is False
