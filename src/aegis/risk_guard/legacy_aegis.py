"""Legacy Aegis DirectionProvider — wraps the current Aegis brain as a DirectionProvider."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .direction_provider import DirectionProvider
from .domain import Direction, Signal

logger = logging.getLogger(__name__)


class LegacyAegisDirectionProvider(DirectionProvider):
    """Wraps the current Aegis brain as a DirectionProvider.

    This adapter allows the existing Aegis decision system to be used
    as the direction source in the new architecture. It preserves all
    existing behavior and adds no risk evaluation.

    In the future, this can be replaced with a new DirectionProvider
    without modifying the RiskGuard or EntryDecisionOrchestrator.
    """

    def __init__(self, brain_engine: Any | None = None) -> None:
        """Initialize with an optional brain engine.

        If brain_engine is None, evaluate() will return SKIP (safe default).
        """
        self._brain = brain_engine

    def name(self) -> str:
        return "LEGACY_AEGIS"

    def version(self) -> str:
        return "AEGIS_CURRENT"

    def evaluate(self, symbol: str, context: dict[str, Any] | None = None) -> Signal:
        """Evaluate direction using the legacy Aegis brain.

        Returns a Signal with the direction determined by Aegis.
        If the brain is unavailable, returns SKIP.
        """
        if self._brain is None:
            logger.warning("LegacyAegisDirectionProvider: brain not available, returning SKIP")
            return self._skip(symbol, "BRAIN_UNAVAILABLE")

        try:
            result = self._evaluate_brain(symbol, context)
            return result
        except Exception as exc:
            logger.error("LegacyAegisDirectionProvider error for %s: %s", symbol, exc)
            return self._skip(symbol, f"BRAIN_ERROR:{exc}")

    def _evaluate_brain(self, symbol: str, context: dict[str, Any] | None) -> Signal:
        """Run the brain and extract direction."""
        trace_id = (context or {}).get("trace_id", "")

        if hasattr(self._brain, "predict"):
            response = self._brain.predict(symbol, trace_id)
        elif hasattr(self._brain, "evaluate"):
            response = self._brain.evaluate(symbol, context)
        else:
            return self._skip(symbol, "BRAIN_NO_EVALUATE_METHOD")

        side_str = self._extract_side(response)
        if side_str is None:
            return self._skip(symbol, "NO_DIRECTION")

        if side_str == "LONG":
            side = Direction.LONG
        elif side_str == "SHORT":
            side = Direction.SHORT
        else:
            return self._skip(symbol, f"UNKNOWN_SIDE:{side_str}")

        turbo_score = self._extract_turbo_score(response)
        model_version = self._extract_model_version(response)

        return Signal(
            signal_id=f"AEGIS-{symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{id(self) % 1000:03d}",
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=side,
            direction_source=self.name(),
            direction_model_version=model_version,
            turbo_score=turbo_score,
            metadata={"raw_response_keys": list(response.keys()) if isinstance(response, dict) else []},
        )

    def _extract_side(self, response: Any) -> str | None:
        """Extract the side from a brain response."""
        if isinstance(response, dict):
            prod = response.get("aegis", {}).get("prod", {})
            side = prod.get("side")
            if side:
                return str(side).upper()

            turbo = response.get("aegis", {}).get("turbo", {})
            side = turbo.get("side")
            if side:
                return str(side).upper()

            decision = response.get("decision_brain", {})
            side = decision.get("side")
            if side:
                return str(side).upper()

        return None

    def _extract_turbo_score(self, response: Any) -> float:
        """Extract turbo score from brain response."""
        if isinstance(response, dict):
            turbo = response.get("aegis", {}).get("turbo", {})
            score = turbo.get("turbo_score") or turbo.get("score")
            if score is not None:
                return float(score)
        return 0.0

    def _extract_model_version(self, response: Any) -> str:
        """Extract model version from brain response."""
        if isinstance(response, dict):
            decision = response.get("decision_brain", {})
            version = decision.get("model_version") or decision.get("authority", "")
            if version:
                return str(version)
        return "UNKNOWN"

    def _skip(self, symbol: str, reason: str) -> Signal:
        return Signal(
            signal_id=f"SKIP-{symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=Direction.SKIP,
            direction_source=self.name(),
            direction_model_version=self.version(),
            metadata={"skip_reason": reason},
        )
