"""Tests for E4 precompute, market snapshot, evidence, and API integration."""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import pandas as pd
import numpy as np

from aegis.risk_guard.domain import (
    FROZEN_TAIL_RISK_THRESHOLD,
    RiskDecision,
    RiskGuardConfig,
)
from aegis.risk_guard.precompute import (
    E4PrecomputeService,
    PrecomputedScore,
    PrecomputeCycleResult,
    _cache_key,
)
from aegis.risk_guard.market_snapshot import (
    MarketSnapshot,
    _compute_snapshot_hash,
    _frozen_universe,
)
from aegis.risk_guard.observability import E4EvidenceRecorder
from aegis.risk_guard.feature_bridge import FROZEN_E4_UNIVERSE


# ═══════════════════════════════════════════════════════════════════
# Market Snapshot Tests
# ═══════════════════════════════════════════════════════════════════

class TestFrozenUniverse:
    def test_universe_has_exactly_11_symbols(self):
        universe = _frozen_universe()
        assert len(universe) == 11

    def test_universe_matches_feature_bridge(self):
        universe = _frozen_universe()
        assert set(universe) == set(FROZEN_E4_UNIVERSE)

    def test_universe_is_sorted(self):
        universe = _frozen_universe()
        assert universe == sorted(universe)

    def test_universe_contains_required_symbols(self):
        universe = _frozen_universe()
        required = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                     "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT"}
        assert set(universe) == required


class TestSnapshotHash:
    def test_hash_deterministic(self):
        candles = {
            "BTCUSDT": pd.DataFrame({
                "open_time_ms": [1000, 2000],
                "open": [100, 101],
                "high": [102, 103],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [1000, 1100],
                "taker_buy_volume": [500, 550],
                "quote_volume": [100000, 110000],
                "close_time": pd.to_datetime([2000, 3000], unit="ms", utc=True),
            })
        }
        dt = datetime(2023, 11, 7, 12, 0, tzinfo=timezone.utc)
        h1 = _compute_snapshot_hash(candles, dt)
        h2 = _compute_snapshot_hash(candles, dt)
        assert h1 == h2

    def test_hash_changes_with_different_symbol_count(self):
        candles1 = {
            "BTCUSDT": pd.DataFrame({
                "open_time_ms": [1000], "open": [100], "high": [102],
                "low": [99], "close": [101], "volume": [1000],
                "taker_buy_volume": [500], "quote_volume": [100000],
                "close_time": pd.to_datetime([2000], unit="ms", utc=True),
            })
        }
        candles2 = {
            "BTCUSDT": pd.DataFrame({
                "open_time_ms": [1000], "open": [100], "high": [102],
                "low": [99], "close": [101], "volume": [1000],
                "taker_buy_volume": [500], "quote_volume": [100000],
                "close_time": pd.to_datetime([2000], unit="ms", utc=True),
            }),
            "ETHUSDT": pd.DataFrame({
                "open_time_ms": [1000], "open": [200], "high": [202],
                "low": [199], "close": [201], "volume": [2000],
                "taker_buy_volume": [1000], "quote_volume": [200000],
                "close_time": pd.to_datetime([2000], unit="ms", utc=True),
            })
        }
        dt = datetime(2023, 11, 7, 12, 0, tzinfo=timezone.utc)
        assert _compute_snapshot_hash(candles1, dt) != _compute_snapshot_hash(candles2, dt)

    def test_hash_changes_with_decision_at(self):
        candles = {
            "BTCUSDT": pd.DataFrame({
                "open_time_ms": [1000], "open": [100], "high": [102],
                "low": [99], "close": [101], "volume": [1000],
                "taker_buy_volume": [500], "quote_volume": [100000],
                "close_time": pd.to_datetime([2000], unit="ms", utc=True),
            })
        }
        dt1 = datetime(2023, 11, 7, 12, 0, tzinfo=timezone.utc)
        dt2 = datetime(2023, 11, 7, 12, 5, tzinfo=timezone.utc)
        assert _compute_snapshot_hash(candles, dt1) != _compute_snapshot_hash(candles, dt2)


class TestMarketSnapshot:
    def test_snapshot_frozen_fields(self):
        candles = {s: pd.DataFrame() for s in FROZEN_E4_UNIVERSE}
        snap = MarketSnapshot(
            snapshot_id="test",
            decision_at=datetime.now(timezone.utc),
            captured_at=datetime.now(timezone.utc),
            candles_by_symbol=candles,
            snapshot_hash="abc123",
            source_timestamps={s: datetime.now(timezone.utc) for s in FROZEN_E4_UNIVERSE},
            source_feed_lag_ms={s: 0.0 for s in FROZEN_E4_UNIVERSE},
        )
        assert snap.snapshot_id == "test"
        assert len(snap.candles_by_symbol) == 11


# ═══════════════════════════════════════════════════════════════════
# Precompute Tests
# ═══════════════════════════════════════════════════════════════════

class TestPrecomputedScore:
    def test_frozen_fields(self):
        score = PrecomputedScore(
            symbol="BTCUSDT",
            side="LONG",
            decision_at=datetime(2023, 11, 7, 12, 0, tzinfo=timezone.utc),
            score=0.22,
            threshold=FROZEN_TAIL_RISK_THRESHOLD,
            risk_decision="ALLOW",
            reason="test",
            model_version="E4_V1_FROZEN",
            feature_snapshot_hash="abc",
            feature_available_at=datetime.now(timezone.utc),
            source_feed_lag_ms={"tf5m": 0.0},
            computed_at=datetime.now(timezone.utc),
            snapshot_id="snap_test",
        )
        assert score.threshold == FROZEN_TAIL_RISK_THRESHOLD
        assert score.risk_decision == "ALLOW"

    def test_score_freezes_threshold(self):
        score = PrecomputedScore(
            symbol="BTCUSDT", side="LONG",
            decision_at=datetime.now(timezone.utc),
            score=0.22, threshold=FROZEN_TAIL_RISK_THRESHOLD,
            risk_decision="ALLOW", reason="test",
            model_version="E4_V1_FROZEN", feature_snapshot_hash="abc",
            feature_available_at=None, source_feed_lag_ms=None,
            computed_at=datetime.now(timezone.utc), snapshot_id="snap",
        )
        assert score.threshold == FROZEN_TAIL_RISK_THRESHOLD


class TestCacheKey:
    def test_key_format(self):
        dt = datetime(2023, 11, 7, 12, 0, tzinfo=timezone.utc)
        key = _cache_key(dt, "BTCUSDT", "LONG")
        assert "BTCUSDT" in key
        assert "LONG" in key
        assert "2023-11-07" in key

    def test_keys_unique_per_symbol_side(self):
        dt = datetime(2023, 11, 7, 12, 0, tzinfo=timezone.utc)
        k1 = _cache_key(dt, "BTCUSDT", "LONG")
        k2 = _cache_key(dt, "BTCUSDT", "SHORT")
        k3 = _cache_key(dt, "ETHUSDT", "LONG")
        assert k1 != k2
        assert k1 != k3
        assert k2 != k3


class TestE4PrecomputeService:
    def test_health_when_not_initialized(self):
        config = RiskGuardConfig(enabled=True, mode="observe_only")
        service = E4PrecomputeService(config)
        health = service.health()
        assert health["available"] is False
        assert health["cycle_count"] == 0

    def test_lookup_returns_none_when_empty(self):
        config = RiskGuardConfig(enabled=True, mode="observe_only")
        service = E4PrecomputeService(config)
        dt = datetime.now(timezone.utc)
        result = service.lookup("BTCUSDT", "LONG", dt)
        assert result is None

    def test_cycle_count_starts_at_zero(self):
        config = RiskGuardConfig(enabled=True, mode="observe_only")
        service = E4PrecomputeService(config)
        assert service.cycle_count == 0


# ═══════════════════════════════════════════════════════════════════
# Evidence Recorder Tests
# ═══════════════════════════════════════════════════════════════════

class TestE4EvidenceRecorder:
    def test_record_evaluation(self, tmp_path):
        jsonl_path = tmp_path / "e4_evidence.jsonl"
        recorder = E4EvidenceRecorder(jsonl_path)

        recorder.record_evaluation(
            signal_id="sig_001",
            decision_id="dec_001",
            decision_cycle_id="cycle_001",
            decision_at=datetime(2023, 11, 7, 12, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            side="LONG",
            direction_source="AEGIS",
            e4_score=0.22,
            e4_threshold=FROZEN_TAIL_RISK_THRESHOLD,
            e4_decision="ALLOW",
            e4_reason="score below threshold",
            e4_model_version="E4_V1_FROZEN",
            feature_snapshot_hash="abc123",
            feature_available_at=datetime.now(timezone.utc),
            source_feed_lag_ms={"tf5m": 0.0},
            snapshot_id="snap_001",
            cache_age_ms=0.0,
            python_latency_ms=15.0,
        )

        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "e4_tail_risk_evaluation"
        assert entry["e4_score"] == 0.22
        assert entry["e4_decision"] == "ALLOW"

    def test_record_blocked_outcome(self, tmp_path):
        jsonl_path = tmp_path / "e4_evidence.jsonl"
        recorder = E4EvidenceRecorder(jsonl_path)

        recorder.record_blocked_outcome(
            signal_id="sig_002",
            symbol="ETHUSDT",
            side="SHORT",
            blocked_at=datetime.now(timezone.utc),
            actual_outcome="would_have_been_profit",
            actual_pnl_bps=15.0,
            blocked_correctly=False,
        )

        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "e4_blocked_outcome"
        assert entry["blocked_correctly"] is False

    def test_record_without_file(self):
        recorder = E4EvidenceRecorder(None)
        recorder.record({"test": True})


# ═══════════════════════════════════════════════════════════════════
# API Integration Tests
# ═══════════════════════════════════════════════════════════════════

class TestE4APIIntegration:
    def test_e4_unavailable_response_structure(self):
        from aegis.live_api import _build_e4_unavailable_response
        resp = _build_e4_unavailable_response("BTCUSDT", "LONG", "", "TEST")
        assert resp["available"] is False
        assert resp["decision"] == "BLOCK"
        assert resp["threshold"] == FROZEN_TAIL_RISK_THRESHOLD
        assert resp["score"] is None

    def test_e4_config_frozen_threshold(self):
        from aegis.live_api import _build_e4_config
        config = _build_e4_config()
        assert config.tail_risk_threshold == FROZEN_TAIL_RISK_THRESHOLD
        assert config.fail_closed is True


# ═══════════════════════════════════════════════════════════════════
# Frozen Invariant Tests
# ═══════════════════════════════════════════════════════════════════

class TestFrozenInvariants:
    def test_threshold_exact(self):
        assert FROZEN_TAIL_RISK_THRESHOLD == 0.4522452210875323

    def test_fail_closed_true(self):
        config = RiskGuardConfig(enabled=True, mode="observe_only", fail_closed=True)
        assert config.fail_closed is True

    def test_fail_closed_rejects_false(self):
        with pytest.raises(ValueError, match="fail_closed must be True"):
            RiskGuardConfig(enabled=True, mode="observe_only", fail_closed=False)

    def test_threshold_rejects_different(self):
        with pytest.raises(ValueError, match="tail_risk_threshold must be FROZEN"):
            RiskGuardConfig(
                enabled=True, mode="observe_only",
                tail_risk_threshold=0.5
            )


# ═══════════════════════════════════════════════════════════════════
# Feature Parity Preservation Tests
# ═══════════════════════════════════════════════════════════════════

class TestFeatureParity:
    def test_feature_count_146(self):
        schema_path = Path("sandbox/aegis_strategy_router/experiments/aegis_e4_robust_training/artifacts/dataset_v1/feature_schema.json")
        if schema_path.exists():
            schema = json.loads(schema_path.read_text())
            features = schema.get("features", [])
            assert len(features) == 146

    def test_frozen_universe_exact(self):
        expected = {
            "ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
            "ETHUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "SUIUSDT", "XRPUSDT"
        }
        assert set(FROZEN_E4_UNIVERSE) == expected
