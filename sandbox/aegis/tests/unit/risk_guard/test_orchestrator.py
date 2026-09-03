"""Unit tests for EntryDecisionOrchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from aegis.risk_guard.domain import (
    Direction,
    EntryDecision,
    RiskDecision,
    RiskGuardConfig,
    RiskGuardResult,
    RiskGuardVerdict,
    Signal,
)
from aegis.risk_guard.direction_provider import DirectionProvider
from aegis.risk_guard.entry_decision import EntryDecisionOrchestrator
from aegis.risk_guard.risk_guard import RiskGuard


class MockDirectionProvider(DirectionProvider):
    """Test double for DirectionProvider."""

    def __init__(self, side: Direction = Direction.SHORT) -> None:
        self._side = side

    def evaluate(self, symbol: str, context: dict[str, Any] | None = None) -> Signal:
        return Signal(
            signal_id=f"MOCK-{symbol}",
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=self._side,
            direction_source="MOCK",
            direction_model_version="mock_v1",
        )

    def name(self) -> str:
        return "MOCK"

    def version(self) -> str:
        return "mock_v1"


class MockRiskGuard(RiskGuard):
    """Test double for RiskGuard."""

    def __init__(self, decision: RiskDecision = RiskDecision.ALLOW, score: float = 0.3) -> None:
        self._decision = decision
        self._score = score
        self._available = True

    def evaluate(self, signal: Signal, context: dict[str, Any] | None = None) -> RiskGuardResult:
        return RiskGuardResult(
            decision=self._decision,
            score=self._score,
            threshold=0.45,
            model_version="MOCK_V1",
            feature_snapshot_hash="mock_hash",
            reason=f"MOCK_{self._decision.value}",
        )

    def name(self) -> str:
        return "MOCK_GUARD"

    def version(self) -> str:
        return "mock_v1"

    def is_available(self) -> bool:
        return self._available


class TestEntryDecisionOrchestrator:
    def test_skip_direction_bypasses_guard(self):
        provider = MockDirectionProvider(Direction.SKIP)
        guard = MockRiskGuard(RiskDecision.BLOCK)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT")

        assert decision.signal.side == Direction.SKIP
        assert decision.verdict == RiskGuardVerdict.ALLOW
        assert not decision.would_block

    def test_guard_disabled_allows_all(self):
        provider = MockDirectionProvider(Direction.SHORT)
        guard = MockRiskGuard(RiskDecision.BLOCK, score=0.9)
        config = RiskGuardConfig(enabled=False)
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT")

        assert decision.verdict == RiskGuardVerdict.ALLOW
        assert decision.risk_result.reason == "RISK_GUARD_DISABLED"

    def test_guard_allow_passes_through(self):
        provider = MockDirectionProvider(Direction.SHORT)
        guard = MockRiskGuard(RiskDecision.ALLOW, score=0.2)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT")

        assert decision.verdict == RiskGuardVerdict.ALLOW
        assert not decision.would_block

    def test_guard_block_enforced(self):
        provider = MockDirectionProvider(Direction.SHORT)
        guard = MockRiskGuard(RiskDecision.BLOCK, score=0.8)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT")

        assert decision.verdict == RiskGuardVerdict.BLOCK
        assert decision.is_effective_block
        assert decision.enforced

    def test_guard_block_observed_only(self):
        provider = MockDirectionProvider(Direction.SHORT)
        guard = MockRiskGuard(RiskDecision.BLOCK, score=0.8)
        config = RiskGuardConfig(enabled=True, mode="observe_only")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT")

        assert decision.verdict == RiskGuardVerdict.OBSERVED_BLOCK
        assert decision.would_block
        assert not decision.is_effective_block
        assert decision.observe_only

    def test_no_guard_available(self):
        provider = MockDirectionProvider(Direction.SHORT)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, None, config)

        decision = orch.evaluate("BTCUSDT")

        assert decision.verdict == RiskGuardVerdict.ALLOW
        assert decision.risk_result.reason == "RISK_GUARD_DISABLED"

    def test_long_direction(self):
        provider = MockDirectionProvider(Direction.LONG)
        guard = MockRiskGuard(RiskDecision.BLOCK, score=0.9)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("ETHUSDT")

        assert decision.signal.side == Direction.LONG
        assert decision.verdict == RiskGuardVerdict.BLOCK

    def test_decision_to_dict_roundtrip(self):
        provider = MockDirectionProvider(Direction.SHORT)
        guard = MockRiskGuard(RiskDecision.ALLOW, score=0.3)
        config = RiskGuardConfig(enabled=True, mode="observe_only")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT")
        d = decision.to_dict()

        assert d["symbol"] == "BTCUSDT"
        assert d["side"] == "SHORT"
        assert d["tail_risk_score"] == 0.3
        assert d["risk_decision"] == "ALLOW"
        assert d["observe_only"] is True
