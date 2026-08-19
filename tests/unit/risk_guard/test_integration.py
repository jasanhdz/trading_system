"""Integration tests for the Risk Guard architecture."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
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
from aegis.risk_guard.flags import RiskGuardFlags, RiskGuardMode
from aegis.risk_guard.observability import RiskGuardMetrics, RiskGuardObserver
from aegis.risk_guard.position_manager import (
    AllowOnlyPositionManager,
    PositionManagerContract,
    PositionManagerResult,
)
from aegis.risk_guard.risk_guard import RiskGuard


class StubDirectionProvider(DirectionProvider):
    """Deterministic direction provider for testing."""

    def __init__(self, responses: dict[str, Direction] | None = None) -> None:
        self._responses = responses or {}
        self._call_count = 0

    def evaluate(self, symbol: str, context: dict[str, Any] | None = None) -> Signal:
        self._call_count += 1
        side = self._responses.get(symbol, Direction.SHORT)
        return Signal(
            signal_id=f"STUB-{symbol}-{self._call_count:04d}",
            timestamp=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
            symbol=symbol,
            side=side,
            direction_source="STUB",
            direction_model_version="stub_v1",
        )

    def name(self) -> str:
        return "STUB"

    def version(self) -> str:
        return "stub_v1"


class StubRiskGuard(RiskGuard):
    """Deterministic risk guard for testing."""

    def __init__(self, block_above: float = 0.45) -> None:
        self._block_above = block_above
        self._evaluations: list[tuple[str, float]] = []

    def evaluate(self, signal: Signal, context: dict[str, Any] | None = None) -> RiskGuardResult:
        score = (context or {}).get("score", 0.3)
        self._evaluations.append((signal.symbol, score))

        if score >= self._block_above:
            return RiskGuardResult(
                decision=RiskDecision.BLOCK,
                score=score,
                threshold=self._block_above,
                model_version="STUB_V1",
                feature_snapshot_hash="stub_hash",
                reason=f"STUB_BLOCK:{score:.4f} >= {self._block_above}",
            )
        return RiskGuardResult(
            decision=RiskDecision.ALLOW,
            score=score,
            threshold=self._block_above,
            model_version="STUB_V1",
            feature_snapshot_hash="stub_hash",
            reason=f"STUB_ALLOW:{score:.4f} < {self._block_above}",
        )

    def name(self) -> str:
        return "STUB_GUARD"

    def version(self) -> str:
        return "stub_v1"

    def is_available(self) -> bool:
        return True


class TestFullFlow:
    """Integration tests for the complete DirectionProvider → RiskGuard → EntryDecision flow."""

    def test_full_flow_allow(self):
        provider = StubDirectionProvider({"BTCUSDT": Direction.SHORT})
        guard = StubRiskGuard(block_above=0.45)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT", {"score": 0.3})

        assert decision.signal.symbol == "BTCUSDT"
        assert decision.signal.side == Direction.SHORT
        assert decision.risk_result.decision == RiskDecision.ALLOW
        assert decision.verdict == RiskGuardVerdict.ALLOW
        assert not decision.would_block
        assert not decision.is_effective_block

    def test_full_flow_block_enforced(self):
        provider = StubDirectionProvider({"BTCUSDT": Direction.SHORT})
        guard = StubRiskGuard(block_above=0.45)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT", {"score": 0.8})

        assert decision.risk_result.decision == RiskDecision.BLOCK
        assert decision.verdict == RiskGuardVerdict.BLOCK
        assert decision.is_effective_block
        assert decision.enforced

    def test_full_flow_block_observed(self):
        provider = StubDirectionProvider({"BTCUSDT": Direction.SHORT})
        guard = StubRiskGuard(block_above=0.45)
        config = RiskGuardConfig(enabled=True, mode="observe_only")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT", {"score": 0.8})

        assert decision.risk_result.decision == RiskDecision.BLOCK
        assert decision.verdict == RiskGuardVerdict.OBSERVED_BLOCK
        assert decision.would_block
        assert not decision.is_effective_block
        assert decision.observe_only

    def test_multiple_symbols(self):
        provider = StubDirectionProvider({
            "BTCUSDT": Direction.SHORT,
            "ETHUSDT": Direction.LONG,
            "SOLUSDT": Direction.SHORT,
        })
        guard = StubRiskGuard(block_above=0.45)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        d1 = orch.evaluate("BTCUSDT", {"score": 0.2})
        d2 = orch.evaluate("ETHUSDT", {"score": 0.8})
        d3 = orch.evaluate("SOLUSDT", {"score": 0.5})

        assert d1.verdict == RiskGuardVerdict.ALLOW
        assert d2.verdict == RiskGuardVerdict.BLOCK
        assert d3.verdict == RiskGuardVerdict.BLOCK

    def test_metrics_integration(self):
        provider = StubDirectionProvider()
        guard = StubRiskGuard(block_above=0.45)
        config = RiskGuardConfig(enabled=True, mode="observe_only")
        orch = EntryDecisionOrchestrator(provider, guard, config)
        metrics = RiskGuardMetrics()

        for score in [0.1, 0.3, 0.5, 0.7, 0.9]:
            d = orch.evaluate("BTCUSDT", {"score": score})
            metrics.record(d)

        summary = metrics.summary()
        assert summary["total_evaluated"] == 5
        assert summary["total_blocked"] == 3
        assert summary["total_observed_blocks"] == 3

    def test_observer_integration(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            provider = StubDirectionProvider()
            guard = StubRiskGuard(block_above=0.45)
            config = RiskGuardConfig(enabled=True, mode="enforce")
            orch = EntryDecisionOrchestrator(provider, guard, config)
            observer = RiskGuardObserver(jsonl_path=path)

            for score in [0.1, 0.6]:
                d = orch.evaluate("BTCUSDT", {"score": score})
                observer.record(d)

            with open(path) as f:
                lines = f.readlines()

            assert len(lines) == 2
            e1 = json.loads(lines[0])
            e2 = json.loads(lines[1])
            assert e1["risk_decision"] == "ALLOW"
            assert e2["risk_decision"] == "BLOCK"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_side_never_changes(self):
        """Critical invariant: the risk guard NEVER changes the side."""
        provider = StubDirectionProvider({"BTCUSDT": Direction.LONG})
        guard = StubRiskGuard(block_above=0.0)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT", {"score": 0.99})

        assert decision.signal.side == Direction.LONG
        assert decision.verdict == RiskGuardVerdict.BLOCK

    def test_no_autonomous_trade_creation(self):
        """Critical invariant: the system never creates a trade on its own."""
        provider = StubDirectionProvider({"BTCUSDT": Direction.SKIP})
        guard = StubRiskGuard(block_above=0.0)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)

        decision = orch.evaluate("BTCUSDT", {"score": 0.99})

        assert decision.signal.side == Direction.SKIP
        assert decision.verdict == RiskGuardVerdict.ALLOW


class TestPositionManagerContract:
    """Integration tests for the PositionManager contract handoff."""

    def test_allow_passes_to_position_manager(self):
        """ALLOW decisions should be accepted by the PositionManager."""
        provider = StubDirectionProvider({"BTCUSDT": Direction.SHORT})
        guard = StubRiskGuard(block_above=0.45)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)
        pm = AllowOnlyPositionManager()

        decision = orch.evaluate("BTCUSDT", {"score": 0.3})

        assert pm.can_execute(decision)
        result = pm.execute(decision)
        assert result.accepted
        assert result.reason == "ALLOW"

    def test_block_rejected_by_position_manager(self):
        """BLOCK decisions should be rejected by the PositionManager."""
        provider = StubDirectionProvider({"BTCUSDT": Direction.SHORT})
        guard = StubRiskGuard(block_above=0.45)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)
        pm = AllowOnlyPositionManager()

        decision = orch.evaluate("BTCUSDT", {"score": 0.8})

        assert not pm.can_execute(decision)
        result = pm.execute(decision)
        assert not result.accepted
        assert "BLOCK" in result.reason

    def test_observed_block_rejected_by_position_manager(self):
        """OBSERVED_BLOCK decisions should be rejected (not executed)."""
        provider = StubDirectionProvider({"BTCUSDT": Direction.SHORT})
        guard = StubRiskGuard(block_above=0.45)
        config = RiskGuardConfig(enabled=True, mode="observe_only")
        orch = EntryDecisionOrchestrator(provider, guard, config)
        pm = AllowOnlyPositionManager()

        decision = orch.evaluate("BTCUSDT", {"score": 0.8})

        assert not pm.can_execute(decision)
        result = pm.execute(decision)
        assert not result.accepted
        assert "OBSERVED_BLOCK" in result.reason

    def test_full_chain_allow(self):
        """Full chain: Provider → Guard → Orchestrator → PositionManager (ALLOW)."""
        provider = StubDirectionProvider({"BTCUSDT": Direction.SHORT})
        guard = StubRiskGuard(block_above=0.45)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)
        pm = AllowOnlyPositionManager()

        decision = orch.evaluate("BTCUSDT", {"score": 0.3})

        assert decision.signal.side == Direction.SHORT
        assert decision.risk_result.decision == RiskDecision.ALLOW
        assert decision.verdict == RiskGuardVerdict.ALLOW
        assert pm.can_execute(decision)
        result = pm.execute(decision)
        assert result.accepted

    def test_full_chain_block(self):
        """Full chain: Provider → Guard → Orchestrator → PositionManager (BLOCK)."""
        provider = StubDirectionProvider({"BTCUSDT": Direction.SHORT})
        guard = StubRiskGuard(block_above=0.45)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)
        pm = AllowOnlyPositionManager()

        decision = orch.evaluate("BTCUSDT", {"score": 0.8})

        assert decision.signal.side == Direction.SHORT
        assert decision.risk_result.decision == RiskDecision.BLOCK
        assert decision.verdict == RiskGuardVerdict.BLOCK
        assert not pm.can_execute(decision)
        result = pm.execute(decision)
        assert not result.accepted

    def test_skip_never_reaches_position_manager(self):
        """SKIP decisions never reach the PositionManager (orchestrator returns ALLOW)."""
        provider = StubDirectionProvider({"BTCUSDT": Direction.SKIP})
        guard = StubRiskGuard(block_above=0.0)
        config = RiskGuardConfig(enabled=True, mode="enforce")
        orch = EntryDecisionOrchestrator(provider, guard, config)
        pm = AllowOnlyPositionManager()

        decision = orch.evaluate("BTCUSDT", {"score": 0.99})

        assert decision.signal.side == Direction.SKIP
        assert decision.verdict == RiskGuardVerdict.ALLOW
        assert pm.can_execute(decision)
        result = pm.execute(decision)
        assert result.accepted
        assert result.reason == "ALLOW"
