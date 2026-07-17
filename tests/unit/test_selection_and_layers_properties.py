from dataclasses import replace
from datetime import timedelta

from aegis.decision import GlobalSelectionPolicy
from aegis.domain import Candidate, CandidateSet, ModelPrediction, ModelPredictions, PortfolioContext, ReasonCode, Regime, RiskIntent, ScientificContext, TradeSide
from aegis.features import DeterministicFeaturePipeline
from aegis.layers import LayerSettings, OrderedScientificLayers


def _candidate(symbol: str, score: float) -> Candidate:
    return Candidate(f"candidate-{symbol}", symbol, TradeSide.LONG, score, score, 0.8, 0.2,
                     Regime.BULL_TREND, 0.8, 0.02, 12, RiskIntent(maximum_holding_bars=12),
                     (ReasonCode.ELIGIBLE,), (), "bundle", "f" * 64, symbol.lower() * 4, True)


def test_global_ranking_ties_and_portfolio_fallback_are_deterministic() -> None:
    now = __import__("datetime").datetime(2026, 7, 17, tzinfo=__import__("datetime").timezone.utc)
    candidates = CandidateSet("cycle", (_candidate("ETHUSDT", 0.8), _candidate("BTCUSDT", 0.8), _candidate("ADAUSDT", 0.7)))
    policy = GlobalSelectionPolicy(0.5)
    assert policy.select(candidates, PortfolioContext(available_slots=1), now).selected[0].symbol == "BTCUSDT"
    blocked = policy.select(candidates, PortfolioContext(blocked_symbols=("BTCUSDT",), available_slots=1), now)
    assert blocked.selected[0].symbol == "ETHUSDT"
    cooldown = policy.select(candidates, PortfolioContext(active_cooldowns={"BTCUSDT": now + timedelta(hours=1)}, available_slots=1), now)
    assert cooldown.selected[0].symbol == "ETHUSDT"
    no_slots = policy.select(candidates, PortfolioContext(available_slots=0), now)
    assert not no_slots.selected and no_slots.reason_codes == (ReasonCode.NO_AVAILABLE_SLOT,)


def _predictions(features, *, tail: float, qmae: float, expected: float = 0.03, disagreement: bool = False):
    rows = []
    for row in features.rows:
        rows.append(ModelPrediction("a", row.symbol, 12, TradeSide.LONG, 0.9, 0.05, 0.05, expected, tail, qmae, 0.9, 0.1))
        if disagreement:
            rows.append(ModelPrediction("b", row.symbol, 12, TradeSide.SHORT, 0.1, 0.85, 0.05, -expected, tail, qmae, 0.9, 0.1))
    return ModelPredictions("bundle", features.feature_hash, tuple(rows))


def _apply(snapshot, features, predictions, *, cost=0.0014):
    layers = OrderedScientificLayers(LayerSettings(0.7, 0.03, 0.0, cost, 0.5))
    return layers.apply(predictions, ScientificContext("r", "c", snapshot.closed_at, "5m", snapshot.portfolio, features))


def test_trrm_qmae_eqm_and_econ_monotonic_properties(snapshot_factory) -> None:
    snapshot = snapshot_factory()
    features = DeterministicFeaturePipeline().transform(snapshot)
    low_tail = _apply(snapshot, features, _predictions(features, tail=0.1, qmae=0.005))
    high_tail = _apply(snapshot, features, _predictions(features, tail=0.8, qmae=0.005))
    assert high_tail.results[0].trrm_compatibility < low_tail.results[0].trrm_compatibility
    assert ReasonCode.TRRM_TAIL_RISK_VETO in high_tail.results[0].reason_codes
    high_qmae = _apply(snapshot, features, _predictions(features, tail=0.1, qmae=0.04))
    assert high_qmae.results[0].qmae_quality < low_tail.results[0].qmae_quality
    assert ReasonCode.QMAE_ADVERSE_EXCURSION_HIGH in high_qmae.results[0].reason_codes
    expensive = _apply(snapshot, features, _predictions(features, tail=0.1, qmae=0.005), cost=0.05)
    assert expensive.results[0].econ_edge < low_tail.results[0].econ_edge
    assert ReasonCode.ECON1_EDGE_BELOW_COST in expensive.results[0].reason_codes
    disagreement = _apply(snapshot, features, _predictions(features, tail=0.1, qmae=0.005, disagreement=True))
    assert disagreement.results[0].calibrated_score <= low_tail.results[0].calibrated_score
