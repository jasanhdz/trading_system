"""Unit tests for Risk Guard observability."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aegis.risk_guard.domain import (
    Direction,
    EntryDecision,
    RiskDecision,
    RiskGuardResult,
    RiskGuardVerdict,
    Signal,
)
from aegis.risk_guard.observability import RiskGuardMetrics, RiskGuardObserver


def _make_decision(
    side: Direction = Direction.SHORT,
    risk_decision: RiskDecision = RiskDecision.ALLOW,
    verdict: RiskGuardVerdict = RiskGuardVerdict.ALLOW,
    score: float = 0.3,
    symbol: str = "BTCUSDT",
) -> EntryDecision:
    return EntryDecision(
        signal=Signal(
            signal_id=f"TEST-{symbol}",
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=side,
            direction_source="TEST",
            direction_model_version="v1",
        ),
        risk_result=RiskGuardResult(
            decision=risk_decision,
            score=score,
            threshold=0.45,
            model_version="E4_V1",
            feature_snapshot_hash="hash123",
            reason=f"TEST_{risk_decision.value}",
        ),
        verdict=verdict,
        enforced=verdict == RiskGuardVerdict.BLOCK,
        observe_only=verdict == RiskGuardVerdict.OBSERVED_BLOCK,
    )


class TestRiskGuardObserver:
    def test_record_writes_jsonl(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            observer = RiskGuardObserver(jsonl_path=path)
            decision = _make_decision()
            observer.record(decision)

            with open(path) as f:
                lines = f.readlines()

            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["event"] == "risk_guard_decision"
            assert entry["symbol"] == "BTCUSDT"
            assert entry["side"] == "SHORT"
            assert entry["risk_decision"] == "ALLOW"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_record_multiple(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            observer = RiskGuardObserver(jsonl_path=path)
            for i in range(5):
                observer.record(_make_decision(score=0.1 * i))

            with open(path) as f:
                lines = f.readlines()

            assert len(lines) == 5
        finally:
            Path(path).unlink(missing_ok=True)

    def test_record_without_jsonl(self):
        observer = RiskGuardObserver()
        observer.record(_make_decision())


class TestRiskGuardMetrics:
    def test_empty_metrics(self):
        m = RiskGuardMetrics()
        s = m.summary()
        assert s["total_evaluated"] == 0
        assert s["total_blocked"] == 0

    def test_count_allow(self):
        m = RiskGuardMetrics()
        m.record(_make_decision(risk_decision=RiskDecision.ALLOW, verdict=RiskGuardVerdict.ALLOW))
        m.record(_make_decision(risk_decision=RiskDecision.ALLOW, verdict=RiskGuardVerdict.ALLOW))
        s = m.summary()
        assert s["total_evaluated"] == 2
        assert s["total_allowed"] == 2
        assert s["total_blocked"] == 0

    def test_count_enforced_block(self):
        m = RiskGuardMetrics()
        m.record(_make_decision(
            risk_decision=RiskDecision.BLOCK,
            verdict=RiskGuardVerdict.BLOCK,
            score=0.8,
        ))
        s = m.summary()
        assert s["total_enforced_blocks"] == 1
        assert s["total_blocked"] == 1

    def test_count_observed_block(self):
        m = RiskGuardMetrics()
        m.record(_make_decision(
            risk_decision=RiskDecision.BLOCK,
            verdict=RiskGuardVerdict.OBSERVED_BLOCK,
            score=0.8,
        ))
        s = m.summary()
        assert s["total_observed_blocks"] == 1
        assert s["total_blocked"] == 1

    def test_skip_not_counted(self):
        m = RiskGuardMetrics()
        m.record(_make_decision(side=Direction.SKIP))
        s = m.summary()
        assert s["total_skip"] == 1
        assert s["total_evaluated"] == 1

    def test_per_symbol_breakdown(self):
        m = RiskGuardMetrics()
        m.record(_make_decision(symbol="BTCUSDT", verdict=RiskGuardVerdict.ALLOW))
        m.record(_make_decision(symbol="BTCUSDT", verdict=RiskGuardVerdict.BLOCK,
                                risk_decision=RiskDecision.BLOCK))
        m.record(_make_decision(symbol="ETHUSDT", verdict=RiskGuardVerdict.ALLOW))
        s = m.summary()
        assert s["per_symbol"]["BTCUSDT"]["evaluated"] == 2
        assert s["per_symbol"]["BTCUSDT"]["blocked"] == 1
        assert s["per_symbol"]["ETHUSDT"]["evaluated"] == 1

    def test_per_side_breakdown(self):
        m = RiskGuardMetrics()
        m.record(_make_decision(side=Direction.SHORT, verdict=RiskGuardVerdict.ALLOW))
        m.record(_make_decision(side=Direction.LONG, verdict=RiskGuardVerdict.BLOCK,
                                risk_decision=RiskDecision.BLOCK))
        s = m.summary()
        assert s["per_side"]["SHORT"]["allowed"] == 1
        assert s["per_side"]["LONG"]["blocked"] == 1

    def test_score_statistics(self):
        m = RiskGuardMetrics()
        for score in [0.1, 0.3, 0.5, 0.7, 0.9]:
            m.record(_make_decision(score=score, verdict=RiskGuardVerdict.ALLOW))
        s = m.summary()
        assert s["score_mean"] == pytest.approx(0.5, abs=0.01)
        assert s["score_p90"] >= 0.7

    def test_reset(self):
        m = RiskGuardMetrics()
        m.record(_make_decision())
        m.reset()
        s = m.summary()
        assert s["total_evaluated"] == 0
