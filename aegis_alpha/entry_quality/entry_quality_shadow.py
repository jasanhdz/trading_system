from __future__ import annotations

import time
from typing import Any

import numpy as np

from aegis_alpha.entry_quality.feature_builder import build_entry_quality_features, entry_quality_feature_cache_status
from aegis_alpha.entry_quality.model_loader import (
    entry_quality_model_status,
    get_model_pair,
    load_entry_quality_models,
    normalize_symbol,
)
from aegis_alpha.entry_quality.schema import MODEL_VERSION, QUALITY_MIN, TAIL_MAX, shadow_result


def _predict_score(bundle: Any, x: np.ndarray) -> float | None:
    if bundle is None:
        return None
    estimator = bundle.get("estimator") if isinstance(bundle, dict) else bundle
    preprocessor = bundle.get("preprocessor") if isinstance(bundle, dict) else None
    if estimator is None:
        return None
    x_in = preprocessor.transform(x) if preprocessor is not None else x
    if hasattr(estimator, "predict_proba"):
        score = estimator.predict_proba(x_in)[:, 1][0]
    else:
        score = estimator.predict(x_in)[0]
    score = float(np.asarray(score).item())
    return float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else None


def _recommendation(quality: float | None, tail: float | None) -> tuple[str, str]:
    if quality is None or tail is None:
        return "INSUFFICIENT_DATA", "insufficient_entry_quality_scores"
    quality_low = quality < QUALITY_MIN
    tail_high = tail > TAIL_MAX
    if quality_low and tail_high:
        return "BLOCK_SHADOW", "quality_low_and_tail_high"
    if quality_low:
        return "BLOCK_SHADOW", "entry_quality_below_threshold"
    if tail_high:
        return "BLOCK_SHADOW", "tail_risk_above_threshold"
    return "ALLOW_SHADOW", "quality_above_threshold_tail_ok"


def evaluate_entry_quality_shadow(symbol: str, turbo_context: dict[str, Any] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    normalized = normalize_symbol(symbol)
    try:
        cache = load_entry_quality_models()
        if not cache.feature_columns:
            return shadow_result(
                symbol=normalized,
                recommendation="INSUFFICIENT_DATA",
                reason="entry_quality_feature_columns_missing",
                feature_status="insufficient",
                model_scope="none",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        pair = get_model_pair(normalized)
        if pair.scope == "none" or pair.entry_quality is None or pair.tail_risk is None:
            return shadow_result(
                symbol=normalized,
                recommendation="INSUFFICIENT_DATA",
                reason="entry_quality_models_missing",
                feature_status="insufficient",
                model_scope="none",
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        features = build_entry_quality_features(normalized, turbo_context=turbo_context)
        if features.x is None or features.feature_status == "insufficient":
            return shadow_result(
                symbol=normalized,
                recommendation="INSUFFICIENT_DATA",
                reason=features.reason,
                feature_status="insufficient",
                missing_features=features.missing_features,
                model_scope=pair.scope,  # type: ignore[arg-type]
                latency_ms=(time.perf_counter() - start) * 1000,
                model_version=MODEL_VERSION,
            )
        quality_score = _predict_score(pair.entry_quality, features.x)
        tail_score = _predict_score(pair.tail_risk, features.x)
        rec, reason = _recommendation(quality_score, tail_score)
        return shadow_result(
            symbol=normalized,
            entry_quality_score=quality_score,
            tail_risk_score=tail_score,
            recommendation=rec,  # type: ignore[arg-type]
            reason=reason,
            feature_status=features.feature_status,  # type: ignore[arg-type]
            missing_features=features.missing_features,
            model_scope=pair.scope,  # type: ignore[arg-type]
            latency_ms=(time.perf_counter() - start) * 1000,
            model_version=MODEL_VERSION,
        )
    except Exception as exc:
        return shadow_result(
            symbol=normalized,
            recommendation="MODEL_ERROR",
            reason=f"entry_quality_model_error:{exc!r}",
            feature_status="insufficient",
            model_scope="none",
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def entry_quality_runtime_status(load_if_needed: bool = False) -> dict[str, Any]:
    status = entry_quality_model_status(load_if_needed=load_if_needed)
    status.update(entry_quality_feature_cache_status())
    return status
