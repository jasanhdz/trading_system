"""Aegis Risk Guard architecture.

Separates direction provision from risk evaluation:

    DirectionProvider → RiskGuard → EntryDecision → PositionManager

The RiskGuard evaluates whether an Aegis signal should be ALLOWed or BLOCKed
based on frozen model artifacts. It never changes the side or creates trades.

Frozen invariants (V1):
    - tail_risk_threshold = 0.4522452210875323 (immutable)
    - feature_schema_sha256 = verified at load time
    - models_joblib_sha256 = verified at load time
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
from .flags import RiskGuardMode, RiskGuardFlags, FROZEN_TAIL_RISK_THRESHOLD
from .position_manager import (
    PositionManagerContract,
    PositionManagerResult,
    AllowOnlyPositionManager,
)

__all__ = [
    "Direction",
    "DirectionProvider",
    "EntryDecision",
    "EntryDecisionOrchestrator",
    "E4TailRiskGuard",
    "FROZEN_TAIL_RISK_THRESHOLD",
    "PositionManagerContract",
    "PositionManagerResult",
    "AllowOnlyPositionManager",
    "RiskDecision",
    "RiskGuard",
    "RiskGuardConfig",
    "RiskGuardFlags",
    "RiskGuardMode",
    "RiskGuardResult",
    "RiskGuardVerdict",
    "Signal",
]
