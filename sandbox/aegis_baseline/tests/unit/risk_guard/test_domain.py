"""Unit tests for Risk Guard domain types."""

from __future__ import annotations

from datetime import datetime, timezone

from aegis.risk_guard.domain import (
    Direction,
    EntryDecision,
    RiskDecision,
    RiskGuardConfig,
    RiskGuardResult,
    RiskGuardVerdict,
    Signal,
)


def _make_signal(side: Direction = Direction.SHORT) -> Signal:
    return Signal(
        signal_id="TEST-001",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        symbol="BTCUSDT",
        side=side,
        direction_source="TEST",
        direction_model_version="v1",
    )


def _make_risk_result(decision: RiskDecision = RiskDecision.ALLOW) -> RiskGuardResult:
    return RiskGuardResult(
        decision=decision,
        score=0.3,
        threshold=0.45,
        model_version="E4_V1",
        feature_snapshot_hash="abc123",
        reason="TEST",
    )


class TestSignal:
    def test_signal_creation(self):
        s = _make_signal()
        assert s.signal_id == "TEST-001"
        assert s.side == Direction.SHORT
        assert s.symbol == "BTCUSDT"

    def test_signal_skip(self):
        s = _make_signal(Direction.SKIP)
        assert s.side == Direction.SKIP

    def test_signal_timestamp_iso(self):
        s = _make_signal()
        assert "2026-01-01" in s.timestamp_iso


class TestRiskGuardResult:
    def test_allow_result(self):
        r = _make_risk_result(RiskDecision.ALLOW)
        assert r.decision == RiskDecision.ALLOW

    def test_block_result(self):
        r = _make_risk_result(RiskDecision.BLOCK)
        assert r.decision == RiskDecision.BLOCK


class TestEntryDecision:
    def test_allow_verdict(self):
        d = EntryDecision(
            signal=_make_signal(),
            risk_result=_make_risk_result(RiskDecision.ALLOW),
            verdict=RiskGuardVerdict.ALLOW,
            enforced=False,
            observe_only=False,
        )
        assert not d.would_block
        assert not d.is_effective_block

    def test_enforced_block(self):
        d = EntryDecision(
            signal=_make_signal(),
            risk_result=_make_risk_result(RiskDecision.BLOCK),
            verdict=RiskGuardVerdict.BLOCK,
            enforced=True,
            observe_only=False,
        )
        assert d.would_block
        assert d.is_effective_block

    def test_observed_block(self):
        d = EntryDecision(
            signal=_make_signal(),
            risk_result=_make_risk_result(RiskDecision.BLOCK),
            verdict=RiskGuardVerdict.OBSERVED_BLOCK,
            enforced=False,
            observe_only=True,
        )
        assert d.would_block
        assert not d.is_effective_block

    def test_to_dict(self):
        d = EntryDecision(
            signal=_make_signal(),
            risk_result=_make_risk_result(RiskDecision.ALLOW),
            verdict=RiskGuardVerdict.ALLOW,
            enforced=False,
            observe_only=False,
        )
        result = d.to_dict()
        assert result["signal_id"] == "TEST-001"
        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "SHORT"
        assert result["risk_decision"] == "ALLOW"
        assert result["verdict"] == "ALLOW"


class TestRiskGuardConfig:
    def test_default_config(self):
        c = RiskGuardConfig()
        assert not c.enabled
        assert c.mode == "observe_only"
        assert c.fail_closed

    def test_enforce_mode(self):
        c = RiskGuardConfig(enabled=True, mode="enforce")
        assert c.enforce
        assert not c.observe_only

    def test_observe_only_mode(self):
        c = RiskGuardConfig(enabled=True, mode="observe_only")
        assert not c.enforce
        assert c.observe_only

    def test_disabled_mode(self):
        c = RiskGuardConfig(enabled=False, mode="enforce")
        assert not c.enforce
        assert not c.observe_only

    def test_threshold_frozen(self):
        """Threshold must be exactly the frozen V1 value."""
        c = RiskGuardConfig()
        assert c.tail_risk_threshold == 0.4522452210875323

    def test_threshold_rejects_different_value(self):
        """Cannot construct RiskGuardConfig with non-frozen threshold."""
        import pytest
        with pytest.raises(ValueError, match="must be FROZEN"):
            RiskGuardConfig(tail_risk_threshold=0.9)

    def test_threshold_rejects_zero(self):
        import pytest
        with pytest.raises(ValueError, match="must be FROZEN"):
            RiskGuardConfig(tail_risk_threshold=0.0)

    def test_mode_valid_values(self):
        for mode in ("disabled", "observe_only", "enforce"):
            c = RiskGuardConfig(mode=mode)
            assert c.mode == mode

    def test_mode_rejects_typo(self):
        import pytest
        with pytest.raises(ValueError, match="Invalid mode"):
            RiskGuardConfig(mode="enfroce")

    def test_mode_rejects_arbitrary_string(self):
        import pytest
        with pytest.raises(ValueError, match="Invalid mode"):
            RiskGuardConfig(mode="whatever")

    def test_enabled_with_disabled_mode_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="Contradictory"):
            RiskGuardConfig(enabled=True, mode="disabled")
