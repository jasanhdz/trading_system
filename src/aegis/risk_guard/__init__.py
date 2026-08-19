"""Aegis Risk Guard architecture.

Separates direction provision from risk evaluation:

    DirectionProvider → RiskGuard → EntryDecision → PositionManager

The RiskGuard evaluates whether an Aegis signal should be ALLOWed or BLOCKed
based on frozen model artifacts. It never changes the side or creates trades.
"""

from .domain import (
    Direction,
    EntryDecision,
    RiskDecision,
    RiskGuardConfig,
    RiskGuardResult,
    RiskGuardVerdict,
    Signal,
)
from .direction_provider import DirectionProvider
from .risk_guard import RiskGuard
from .e4_tail_risk_guard import E4TailRiskGuard
from .entry_decision import EntryDecisionOrchestrator
from .flags import RiskGuardMode, RiskGuardFlags

__all__ = [
    "Direction",
    "DirectionProvider",
    "EntryDecision",
    "EntryDecisionOrchestrator",
    "E4TailRiskGuard",
    "RiskDecision",
    "RiskGuard",
    "RiskGuardConfig",
    "RiskGuardFlags",
    "RiskGuardMode",
    "RiskGuardResult",
    "RiskGuardVerdict",
    "Signal",
]
