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

    Compatible with CurrentBrainDecisionService.predict(symbol, trace_id)
    which returns a dict with:
        - aegis.decision_brain.side ("LONG" | "SHORT" | "HOLD")
        - aegis.decision_brain.decision
        - aegis.prod.action
        - aegis.turbo.turbo_score
        - aegis.candidate.model_identifier

    Preserves temporal identity from Aegis:
        - decision_id → Signal.signal_id
        - generated_at → Signal.timestamp
        - decision_cycle_id → Signal.metadata["decision_cycle_id"]
        - model_bundle_id → Signal.direction_model_version
    """

    def __init__(self, brain_engine: Any | None = None) -> None:
        """Initialize with an optional brain engine.

        If brain_engine is None, evaluate() will return SKIP (safe default).
        brain_engine should be a CurrentBrainDecisionService with .predict(symbol, trace_id).
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
        """Run the brain and extract direction.

        Uses CurrentBrainDecisionService.predict(symbol, trace_id) interface.
        """
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
        decision_id = self._extract_decision_id(response)
        generated_at = self._extract_generated_at(response)
        decision_cycle_id = self._extract_decision_cycle_id(response)

        return Signal(
            signal_id=decision_id,
            timestamp=generated_at,
            symbol=symbol,
            side=side,
            direction_source=self.name(),
            direction_model_version=model_version,
            turbo_score=turbo_score,
            metadata={
                "decision_cycle_id": decision_cycle_id,
                "raw_response_keys": list(response.keys()) if isinstance(response, dict) else [],
            },
        )

    def _extract_side(self, response: Any) -> str | None:
        """Extract the side from a brain response.

        Priority order matches compatibility_response() output:
        1. aegis.decision_brain.side (primary)
        2. aegis.prod.action (fallback)
        3. top-level action (fallback)
        """
        if not isinstance(response, dict):
            return None

        # Primary: aegis.decision_brain.side
        decision_brain = response.get("aegis", {}).get("decision_brain", {})
        side = decision_brain.get("side")
        if side and str(side).upper() in {"LONG", "SHORT"}:
            return str(side).upper()

        # Fallback: aegis.prod.action
        prod = response.get("aegis", {}).get("prod", {})
        action = prod.get("action")
        if action and str(action).upper() in {"LONG", "SHORT"}:
            return str(action).upper()

        # Fallback: top-level action
        action = response.get("action")
        if action and str(action).upper() in {"LONG", "SHORT"}:
            return str(action).upper()

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
            # Try aegis.candidate.model_identifier first
            candidate = response.get("aegis", {}).get("candidate", {})
            if isinstance(candidate, str):
                return candidate
            if isinstance(candidate, dict):
                ident = candidate.get("model_identifier") or candidate.get("identity", "")
                if ident:
                    return str(ident)

            # Fallback to metadata
            metadata = response.get("metadata", {})
            bundle_id = metadata.get("model_bundle_id") or metadata.get("bundle_sha256", "")
            if bundle_id:
                return str(bundle_id)

        return "UNKNOWN"

    def _extract_decision_id(self, response: Any) -> str:
        """Extract decision_id from brain response, preserving Aegis temporal identity."""
        if isinstance(response, dict):
            # From metadata (compatibility_response output)
            metadata = response.get("metadata", {})
            decision_id = metadata.get("decision_id", "")
            if decision_id:
                return str(decision_id)

            # From aegis subtree
            aegis = response.get("aegis", {})
            decision_id = aegis.get("decision_id", "")
            if decision_id:
                return str(decision_id)

        # Fallback: generate deterministic ID
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        return f"AEGIS-{ts}"

    def _extract_generated_at(self, response: Any) -> datetime:
        """Extract generated_at from brain response, preserving Aegis timestamp."""
        if isinstance(response, dict):
            metadata = response.get("metadata", {})
            ts_str = metadata.get("generated_at") or metadata.get("market_timestamp", "")
            if ts_str:
                try:
                    return datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            aegis = response.get("aegis", {})
            ts_str = aegis.get("generated_at", "")
            if ts_str:
                try:
                    return datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

        return datetime.now(timezone.utc)

    def _extract_decision_cycle_id(self, response: Any) -> str:
        """Extract decision_cycle_id from brain response."""
        if isinstance(response, dict):
            metadata = response.get("metadata", {})
            cycle_id = metadata.get("decision_cycle_id", "")
            if cycle_id:
                return str(cycle_id)

            aegis = response.get("aegis", {})
            cycle_id = aegis.get("decision_cycle_id", "")
            if cycle_id:
                return str(cycle_id)

        return ""

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
