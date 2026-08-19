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
