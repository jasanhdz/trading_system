from __future__ import annotations

import time
from typing import Any

import numpy as np

from aegis_alpha.decision_brain.feature_builder import build_decision_brain_features, load_latest_news_sentiment, normalize_symbol
from aegis_alpha.decision_brain.model_loader import DEFAULT_MODEL_VERSION, decision_brain_loader_status, load_decision_brain_artifacts
from aegis_alpha.decision_brain.schema import DecisionBrainShadowOutput, base_shadow_output, model_dump


LABEL_PROB_FIELDS = {
    "ENTER_NOW": "enter_now_prob",
    "WAIT_CONFIRMATION": "wait_confirmation_prob",
    "MANUAL_ONLY": "manual_only_prob",
    "DO_NOT_ENTER": "do_not_enter_prob",
}
RECOMMENDATION_BY_DECISION = {
    "ENTER_NOW": "ENTER_NOW_SHADOW",
    "WAIT_CONFIRMATION": "WAIT_CONFIRMATION_SHADOW",
    "MANUAL_ONLY": "MANUAL_ONLY_SHADOW",
    "DO_NOT_ENTER": "DO_NOT_ENTER_SHADOW",
}
REASON_BY_DECISION = {
    "ENTER_NOW": "decision_brain_enter_now_candidate",
    "WAIT_CONFIRMATION": "decision_brain_wait_confirmation",
    "MANUAL_ONLY": "decision_brain_manual_only_context_risk",
    "DO_NOT_ENTER": "decision_brain_do_not_enter_high_risk",
}

LAST_DECISION_BY_SYMBOL: dict[str, str] = {}
LAST_FEATURE_STATUS_BY_SYMBOL: dict[str, str] = {}
LAST_FEATURE_PARITY_BY_SYMBOL: dict[str, float] = {}
LAST_MISSING_COUNT_BY_SYMBOL: dict[str, int] = {}
LAST_CRITICAL_MISSING_BY_SYMBOL: dict[str, list[str]] = {}
LAST_LATENCY_BY_SYMBOL: dict[str, float] = {}
LAST_ERRORS: list[str] = []


def _remember_error(message: str) -> None:
    LAST_ERRORS.append(message)
    del LAST_ERRORS[:-10]


def _probabilities(model: Any, transformed: np.ndarray, label_names: list[str]) -> dict[str, float]:
    out = {field: 0.0 for field in LABEL_PROB_FIELDS.values()}
    if not hasattr(model, "predict_proba"):
        pred = str(label_names[int(model.predict(transformed)[0])])
        out[LABEL_PROB_FIELDS.get(pred, "do_not_enter_prob")] = 1.0
        return out
    raw_proba = model.predict_proba(transformed)
    classes = getattr(model, "classes_", np.arange(len(label_names)))
    for idx, klass in enumerate(classes):
        label = label_names[int(klass)]
        field = LABEL_PROB_FIELDS.get(label)
        if field:
            out[field] = round(float(raw_proba[0, idx]), 6)
    total = sum(out.values())
    if total > 0:
        out = {key: round(float(value / total), 6) for key, value in out.items()}
    return out


def evaluate_decision_brain_shadow(
    symbol: str,
    side: str | None = None,
    turbo_context: dict[str, Any] | None = None,
    entry_quality_model: dict[str, Any] | None = None,
    event_risk_auto: dict[str, Any] | None = None,
    news_sentiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    normalized_symbol = normalize_symbol(symbol)
    output = base_shadow_output(normalized_symbol, side, DEFAULT_MODEL_VERSION)
    artifacts = load_decision_brain_artifacts()
    if artifacts is None:
        output.recommendation = "INSUFFICIENT_DATA"
        output.reason = "decision_brain_artifacts_missing"
        output.total_latency_ms = round((time.perf_counter() - start) * 1000, 3)
        output.latency_ms = output.total_latency_ms
        return model_dump(output)

    output.model_version = artifacts.model_version
    try:
        feature_vector, feature_meta = build_decision_brain_features(
            symbol=normalized_symbol,
            side=side,
            turbo_context=turbo_context,
            entry_quality_model=entry_quality_model,
            event_risk_auto=event_risk_auto,
            news_sentiment=news_sentiment if news_sentiment is not None else load_latest_news_sentiment(),
            feature_columns=artifacts.feature_columns,
        )
        output.side = str(feature_meta.get("side") or side or "UNKNOWN")
        output.feature_status = feature_meta["feature_status"]
        output.feature_parity_pct = feature_meta["feature_parity_pct"]
        output.missing_features_count = feature_meta["missing_features_count"]
        output.missing_features = list(feature_meta["missing_features"])
        output.critical_missing_groups = list(feature_meta["critical_missing_groups"])
        output.available_feature_groups = list(feature_meta.get("available_feature_groups") or [])
        output.approximated_features = list(feature_meta.get("approximated_features") or [])
        output.missing_features_by_group = dict(feature_meta.get("missing_features_by_group") or {})
        output.feature_group_coverage_pct = dict(feature_meta.get("feature_group_coverage_pct") or {})
        output.feature_warnings = list(feature_meta.get("feature_warnings") or [])
        output.feature_build_latency_ms = feature_meta["feature_build_latency_ms"]
        if feature_vector.shape[1] != len(artifacts.feature_columns):
            output.recommendation = "INSUFFICIENT_DATA"
            output.reason = "decision_brain_feature_alignment_error"
            output.feature_status = "insufficient"
            return model_dump(output)

        model_start = time.perf_counter()
        transformed = artifacts.preprocessor.transform(feature_vector)
        model = artifacts.model_package.get("main_model")
        label_names = [str(item) for item in artifacts.model_package.get("label_names", ["ENTER_NOW", "WAIT_CONFIRMATION", "MANUAL_ONLY", "DO_NOT_ENTER"])]
        if model is None:
            output.recommendation = "MODEL_ERROR"
            output.reason = "decision_brain_model_missing_in_package"
            output.feature_status = "insufficient"
            return model_dump(output)
        probs = _probabilities(model, transformed, label_names)
        for key, value in probs.items():
            setattr(output, key, value)
        decision = max(probs.items(), key=lambda item: item[1])[0]
        reverse = {value: key for key, value in LABEL_PROB_FIELDS.items()}
        label = reverse.get(decision, "UNKNOWN")
        output.decision = label  # type: ignore[assignment]
        output.recommendation = RECOMMENDATION_BY_DECISION.get(label, "INSUFFICIENT_DATA")  # type: ignore[assignment]
        output.reason = REASON_BY_DECISION.get(label, "decision_brain_insufficient_features")
        output.model_latency_ms = round((time.perf_counter() - model_start) * 1000, 3)
    except Exception as exc:  # pragma: no cover - endpoint safety
        _remember_error(f"{normalized_symbol}:{exc!r}")
        output.decision = "UNKNOWN"
        output.recommendation = "MODEL_ERROR"
        output.reason = f"decision_brain_model_error:{exc!r}"
        output.feature_status = "insufficient"
    finally:
        output.execute = False
        output.production_allowed = False
        output.mode = "SHADOW"
        output.status = "RESEARCH_CANDIDATE_NOT_LIVE"
        output.total_latency_ms = round((time.perf_counter() - start) * 1000, 3)
        output.latency_ms = output.total_latency_ms
        LAST_DECISION_BY_SYMBOL[normalized_symbol] = output.decision
        LAST_FEATURE_STATUS_BY_SYMBOL[normalized_symbol] = output.feature_status
        LAST_FEATURE_PARITY_BY_SYMBOL[normalized_symbol] = output.feature_parity_pct
        LAST_MISSING_COUNT_BY_SYMBOL[normalized_symbol] = output.missing_features_count
        LAST_CRITICAL_MISSING_BY_SYMBOL[normalized_symbol] = list(output.critical_missing_groups)
        LAST_LATENCY_BY_SYMBOL[normalized_symbol] = output.total_latency_ms
    return model_dump(output)


def decision_brain_runtime_status() -> dict[str, Any]:
    artifacts = load_decision_brain_artifacts()
    loader_status = decision_brain_loader_status()
    return {
        "enabled": True,
        "mode": "SHADOW",
        "model_version": DEFAULT_MODEL_VERSION,
        "manifest_exists": bool(loader_status.get("manifest_exists")),
        "model_loaded": artifacts is not None,
        "feature_columns_count": len(artifacts.feature_columns) if artifacts is not None else 0,
        "last_decision_by_symbol": dict(LAST_DECISION_BY_SYMBOL),
        "last_feature_status_by_symbol": dict(LAST_FEATURE_STATUS_BY_SYMBOL),
        "last_feature_parity_pct_by_symbol": dict(LAST_FEATURE_PARITY_BY_SYMBOL),
        "last_missing_features_count_by_symbol": dict(LAST_MISSING_COUNT_BY_SYMBOL),
        "last_critical_missing_groups_by_symbol": dict(LAST_CRITICAL_MISSING_BY_SYMBOL),
        "last_latency_by_symbol": dict(LAST_LATENCY_BY_SYMBOL),
        "cache_size": 1 if artifacts is not None else 0,
        "last_errors": list(LAST_ERRORS) + list(loader_status.get("last_errors") or []),
    }
