"""E4 Tail Risk Guard — frozen model implementation with FeatureBridge integration."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .domain import RiskDecision, RiskGuardConfig, RiskGuardResult, Signal, FROZEN_TAIL_RISK_THRESHOLD
from .feature_bridge import FeatureBridge
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
        self._bridge: FeatureBridge | None = None

    def name(self) -> str:
        return "E4_TAIL_RISK_GUARD"

    def version(self) -> str:
        return "E4_TAIL_RISK_GUARD_V1"

    def is_available(self) -> bool:
        return self._loaded

    @staticmethod
    def _patch_sklearn_compat() -> None:
        """Patch sklearn compatibility for models trained with 1.8.0+.

        The frozen models were pickled with sklearn 1.8.0 which removed the
        LogisticRegression.multi_class attribute. If we're running on 1.7.x,
        the attribute is missing and calibrator.predict_proba fails.
        """
        import sklearn.linear_model
        if not hasattr(sklearn.linear_model.LogisticRegression, "multi_class"):
            sklearn.linear_model.LogisticRegression.multi_class = "auto"

    def load(self) -> None:
        """Load frozen model artifacts with hash verification.

        Requires:
            - models_joblib_path + models_joblib_sha256 (both non-empty)
            - feature_schema_path + feature_schema_sha256 (both non-empty)

        Refuses to load if any required artifact or hash is missing.
        """
        self._patch_sklearn_compat()

        if not self._config.models_joblib_path or not self._config.models_joblib_sha256:
            raise ValueError(
                "E4 models_joblib_path and models_joblib_sha256 are both required"
            )

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

        if not self._config.feature_schema_path or not self._config.feature_schema_sha256:
            raise ValueError(
                "E4 feature_schema_path and feature_schema_sha256 are both required"
            )

        schema_path = Path(self._config.feature_schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"E4 feature schema not found: {schema_path}")

        schema_raw = schema_path.read_text()
        self._schema = json.loads(schema_raw)
        actual_schema_hash = hashlib.sha256(schema_raw.encode()).hexdigest()
        if actual_schema_hash != self._config.feature_schema_sha256:
            raise RuntimeError(
                f"E4_SCHEMA_HASH_MISMATCH: expected {self._config.feature_schema_sha256}, "
                f"got {actual_schema_hash}"
            )

        self._loaded = True
        self._bridge = FeatureBridge(self._tail_bundle["features"])
        logger.info(
            "E4 Tail Risk Guard loaded: threshold=%.6f, features=%d",
            self._threshold,
            len(self._tail_bundle.get("features", [])),
        )

    def evaluate(self, signal: Signal, context: dict[str, Any] | None = None) -> RiskGuardResult:
        """Evaluate tail risk for a signal.

        Dispatches based on context contents:
            - context["candles_by_symbol"] + context["decision_at"] → live path
            - context["features"] → pre-computed features path
            - context["_replay_pre_computed_score"] → replay path
            - otherwise → FEATURES_UNAVAILABLE
        """
        if not self._loaded:
            return self._fail_closed(signal, "E4_MODELS_NOT_LOADED")

        if context and "_replay_pre_computed_score" in context:
            score = context["_replay_pre_computed_score"]
            if not self._validate_score(score):
                return self._fail_closed(signal, f"INVALID_REPLAY_SCORE:{score}")
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
                feature_snapshot_hash="REPLAY_PRE_COMPUTED",
                reason=reason,
            )

        if context and "candles_by_symbol" in context and "decision_at" in context:
            return self.evaluate_from_candles(
                signal,
                context["candles_by_symbol"],
                context["decision_at"],
            )

        features = self._extract_features(signal, context)
        if features is None:
            return self._fail_closed(signal, "FEATURES_UNAVAILABLE")

        start = time.monotonic()
        try:
            score = self._score_tail_risk(features)
            elapsed_ms = (time.monotonic() - start) * 1000

            if not self._validate_score(score):
                return self._fail_closed(signal, f"INVALID_SCORE:{score}")

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

    def evaluate_from_candles(
        self,
        signal: Signal,
        candles_by_symbol: dict[str, pd.DataFrame],
        decision_at: datetime,
    ) -> RiskGuardResult:
        """Evaluate tail risk by building features from raw candles.

        This is the live-path entry point. It:
            1. Validates candle availability and freshness
            2. Builds features via FeatureBridge
            3. Scores with frozen E4 model
            4. Returns ALLOW/BLOCK with full telemetry

        Feature availability states:
            - FEATURES_UNAVAILABLE: no candles provided
            - STALE_DATA: candles too old (decision_at not aligned to 5m)
            - FEATURE_BUILD_ERROR: exception during feature construction
        """
        if not self._loaded:
            return self._fail_closed(signal, "E4_MODELS_NOT_LOADED")

        if not candles_by_symbol:
            return RiskGuardResult(
                decision=RiskDecision.FEATURES_UNAVAILABLE,
                score=float("nan"),
                threshold=self._threshold,
                model_version=self.version(),
                feature_snapshot_hash="",
                reason="FEATURES_UNAVAILABLE:no_candles_provided",
            )

        try:
            feature_build_start = time.monotonic()
            feature_row = self._bridge.from_market_candles(
                candles_by_symbol=candles_by_symbol,
                target_symbol=signal.symbol,
                side=signal.side.value,
                decision_at=decision_at,
            )
            feature_build_ms = (time.monotonic() - feature_build_start) * 1000

        except ValueError as exc:
            exc_str = str(exc)
            if "UNIVERSE_MISMATCH" in exc_str:
                return RiskGuardResult(
                    decision=RiskDecision.FEATURES_UNAVAILABLE,
                    score=float("nan"),
                    threshold=self._threshold,
                    model_version=self.version(),
                    feature_snapshot_hash="",
                    reason=f"FEATURES_UNAVAILABLE:{exc}",
                )
            if "aligned to 5-minute" in exc_str:
                return RiskGuardResult(
                    decision=RiskDecision.STALE_DATA,
                    score=float("nan"),
                    threshold=self._threshold,
                    model_version=self.version(),
                    feature_snapshot_hash="",
                    reason=f"STALE_DATA:{exc}",
                )
            if "NON_CAUSAL_DATA" in exc_str:
                return RiskGuardResult(
                    decision=RiskDecision.NON_CAUSAL_DATA,
                    score=float("nan"),
                    threshold=self._threshold,
                    model_version=self.version(),
                    feature_snapshot_hash="",
                    reason=f"NON_CAUSAL_DATA:{exc}",
                )
            if "CANDLE_DUPLICATE_MINUTE" in exc_str or "CANDLE_MINUTE_GAP" in exc_str:
                return RiskGuardResult(
                    decision=RiskDecision.STALE_DATA,
                    score=float("nan"),
                    threshold=self._threshold,
                    model_version=self.version(),
                    feature_snapshot_hash="",
                    reason=f"STALE_DATA:{exc}",
                )
            return RiskGuardResult(
                decision=RiskDecision.FEATURE_BUILD_ERROR,
                score=float("nan"),
                threshold=self._threshold,
                model_version=self.version(),
                feature_snapshot_hash="",
                reason=f"FEATURE_BUILD_ERROR:{exc}",
            )
        except Exception as exc:
            return RiskGuardResult(
                decision=RiskDecision.FEATURE_BUILD_ERROR,
                score=float("nan"),
                threshold=self._threshold,
                model_version=self.version(),
                feature_snapshot_hash="",
                reason=f"FEATURE_BUILD_ERROR:{exc}",
            )

        features_df = feature_row.to_dataframe()
        score_start = time.monotonic()
        try:
            score = self._score_tail_risk(features_df)
            score_ms = (time.monotonic() - score_start) * 1000
        except Exception as exc:
            score_ms = (time.monotonic() - score_start) * 1000
            logger.error("E4 scoring error for %s: %s", signal.signal_id, exc)
            return RiskGuardResult(
                decision=RiskDecision.FEATURE_BUILD_ERROR,
                score=float("nan"),
                threshold=self._threshold,
                model_version=self.version(),
                feature_snapshot_hash=feature_row.feature_hash,
                reason=f"E4_SCORING_ERROR:{exc}",
                evaluation_time_ms=feature_build_ms + score_ms,
                feature_available_at=feature_row.timestamp,
                feature_build_latency_ms=feature_build_ms,
            )

        if not self._validate_score(score):
            return self._fail_closed(signal, f"INVALID_SCORE:{score}")

        if score >= self._threshold:
            decision = RiskDecision.BLOCK
            reason = f"TAIL_RISK_SCORE={score:.6f} >= THRESHOLD={self._threshold:.6f}"
        else:
            decision = RiskDecision.ALLOW
            reason = f"TAIL_RISK_SCORE={score:.6f} < THRESHOLD={self._threshold:.6f}"

        feed_lag = feature_row.source_feed_lag_ms or {}
        valid_lags = [v for v in feed_lag.values() if v != float("inf")]
        max_feed_lag_ms = max(valid_lags) if valid_lags else 0.0

        return RiskGuardResult(
            decision=decision,
            score=score,
            threshold=self._threshold,
            model_version=self.version(),
            feature_snapshot_hash=feature_row.feature_hash,
            reason=reason,
            evaluation_time_ms=feature_build_ms + score_ms,
            feature_available_at=feature_row.max_available_at,
            feature_build_latency_ms=feature_build_ms,
            feature_staleness_ms=max_feed_lag_ms,
            source_feed_lag_ms=feed_lag,
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
    def _validate_score(score: float) -> bool:
        """Validate that a score is finite and in [0, 1] range."""
        import math
        if not math.isfinite(score):
            return False
        if score < 0.0 or score > 1.0:
            return False
        return True

    @staticmethod
    def _hash_features(features: pd.DataFrame) -> str:
        """Deterministic hash of the feature row for auditability."""
        d = features.to_dict(orient="records")[0]
        items = sorted(d.items())
        raw = str(items).encode()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as h:
            for block in iter(lambda: h.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
