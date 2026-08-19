"""E4 Tail Risk Guard — frozen model implementation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .domain import RiskDecision, RiskGuardConfig, RiskGuardResult, Signal
from .risk_guard import RiskGuard

logger = logging.getLogger(__name__)


class E4TailRiskGuard(RiskGuard):
    """Tail Risk Guard using frozen E4 model artifacts.

    Uses only:
    - development_models.joblib (frozen, hash-verified)
    - feature_schema.json (frozen, hash-verified)
    - thresholds_frozen_v1.json (frozen threshold)

    Never retrains, never adjusts features, never modifies the model.
    """

    def __init__(self, config: RiskGuardConfig) -> None:
        self._config = config
        self._models: dict[str, Any] | None = None
        self._schema: dict[str, Any] | None = None
        self._tail_bundle: dict[str, Any] | None = None
        self._threshold = config.tail_risk_threshold
        self._loaded = False

    def name(self) -> str:
        return "E4_TAIL_RISK_GUARD"

    def version(self) -> str:
        return "E4_TAIL_RISK_GUARD_V1"

    def is_available(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load frozen model artifacts with hash verification."""
        models_path = Path(self._config.models_joblib_path)
        if not models_path.exists():
            raise FileNotFoundError(f"E4 models not found: {models_path}")

        actual_hash = self._sha256_file(models_path)
        if actual_hash != self._config.models_joblib_sha256:
            raise RuntimeError(
                f"E4_MODELS_HASH_MISMATCH: expected {self._config.models_joblib_sha256}, "
                f"got {actual_hash}"
            )

        self._models = joblib.load(models_path)
        self._tail_bundle = self._models.get("target__tail_risk")
        if self._tail_bundle is None:
            raise KeyError("target__tail_risk head not found in E4 models")

        if self._config.feature_schema_path:
            schema_path = Path(self._config.feature_schema_path)
            if schema_path.exists():
                self._schema = json.loads(schema_path.read_text())

        self._loaded = True
        logger.info(
            "E4 Tail Risk Guard loaded: threshold=%.6f, features=%d",
            self._threshold,
            len(self._tail_bundle.get("features", [])),
        )

    def evaluate(self, signal: Signal, context: dict[str, Any] | None = None) -> RiskGuardResult:
        """Evaluate tail risk for a signal.

        Args:
            signal: The Aegis signal to evaluate.
            context: Optional dict with:
                - "features": pre-computed feature row (dict or pd.Series)
                - "candle_data": 1m candle DataFrame for feature construction

        Returns:
            RiskGuardResult with ALLOW or BLOCK.
        """
        if not self._loaded:
            return self._fail_closed(signal, "E4_MODELS_NOT_LOADED")

        if context and "pre_computed_tail_risk_score" in context:
            score = context["pre_computed_tail_risk_score"]
            if score >= self._threshold:
                decision = RiskDecision.BLOCK
                reason = f"TAIL_RISK_SCORE={score:.6f} >= THRESHOLD={self._threshold:.6f}"
            else:
                decision = RiskDecision.ALLOW
                reason = f"TAIL_RISK_SCORE={score:.6f} < THRESHOLD={self._threshold:.6f}"
            return RiskGuardResult(
                decision=decision,
                score=score,
                threshold=self._threshold,
                model_version=self.version(),
                feature_snapshot_hash="PRE_COMPUTED",
                reason=reason,
            )

        features = self._extract_features(signal, context)
        if features is None:
            return self._fail_closed(signal, "FEATURES_UNAVAILABLE")

        start = time.monotonic()
        try:
            score = self._score_tail_risk(features)
            elapsed_ms = (time.monotonic() - start) * 1000

            if score >= self._threshold:
                decision = RiskDecision.BLOCK
                reason = f"TAIL_RISK_SCORE={score:.6f} >= THRESHOLD={self._threshold:.6f}"
            else:
                decision = RiskDecision.ALLOW
                reason = f"TAIL_RISK_SCORE={score:.6f} < THRESHOLD={self._threshold:.6f}"

            feature_hash = self._hash_features(features)

            return RiskGuardResult(
                decision=decision,
                score=score,
                threshold=self._threshold,
                model_version=self.version(),
                feature_snapshot_hash=feature_hash,
                reason=reason,
                evaluation_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.error("E4 scoring error for %s: %s", signal.signal_id, exc)
            if self._config.fail_closed:
                return RiskGuardResult(
                    decision=RiskDecision.BLOCK,
                    score=float("nan"),
                    threshold=self._threshold,
                    model_version=self.version(),
                    feature_snapshot_hash="",
                    reason=f"E4_SCORING_ERROR:{exc}",
                    evaluation_time_ms=elapsed_ms,
                )
            return RiskGuardResult(
                decision=RiskDecision.ALLOW,
                score=float("nan"),
                threshold=self._threshold,
                model_version=self.version(),
                feature_snapshot_hash="",
                reason=f"E4_SCORING_ERROR_ALLOW_OPEN:{exc}",
                evaluation_time_ms=elapsed_ms,
            )

    def _extract_features(
        self, signal: Signal, context: dict[str, Any] | None
    ) -> pd.DataFrame | None:
        """Extract E4 features for the tail risk head.

        Expects context["features"] to be a dict or Series with feature__* keys,
        or context["feature_row"] as a pre-computed DataFrame row.
        """
        if context is None:
            return None

        feature_row = context.get("feature_row")
        if feature_row is not None:
            if isinstance(feature_row, pd.DataFrame):
                return feature_row
            if isinstance(feature_row, dict):
                return pd.DataFrame([feature_row])

        features_dict = context.get("features")
        if features_dict is not None:
            if isinstance(features_dict, dict):
                return pd.DataFrame([features_dict])
            if isinstance(features_dict, pd.Series):
                return features_dict.to_frame().T

        return None

    def _score_tail_risk(self, features: pd.DataFrame) -> float:
        """Run the frozen tail risk model on features."""
        tail_feats = features[self._tail_bundle["features"]]
        raw = self._tail_bundle["model"].decision_function(tail_feats).reshape(-1, 1)
        score = float(self._tail_bundle["calibrator"].predict_proba(raw)[:, 1][0])
        return score

    def _fail_closed(self, signal: Signal, reason: str) -> RiskGuardResult:
        """Return a fail-closed BLOCK result."""
        if self._config.fail_closed:
            return RiskGuardResult(
                decision=RiskDecision.BLOCK,
                score=float("nan"),
                threshold=self._threshold,
                model_version=self.version(),
                feature_snapshot_hash="",
                reason=f"FAIL_CLOSED:{reason}",
            )
        return RiskGuardResult(
            decision=RiskDecision.ALLOW,
            score=float("nan"),
            threshold=self._threshold,
            model_version=self.version(),
            feature_snapshot_hash="",
            reason=f"FAIL_OPEN:{reason}",
        )

    @staticmethod
    def _hash_features(features: pd.DataFrame) -> str:
        """Deterministic hash of the feature row for auditability."""
        raw = features.to_json(sort_keys=True, default=str).encode()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as h:
            for block in iter(lambda: h.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
