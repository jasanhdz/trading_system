from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from aegis_alpha.config import REPO_ROOT
from aegis_alpha.edge.common import load_model_bundle
from aegis_alpha.inference.shadow_schema import build_shadow_signal
from aegis_alpha.signals.combination_utils import RuleCondition, predict_scores, threshold_for_rule
from aegis_alpha.signals.common import load_signal_market


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = REPO_ROOT / "aegis_alpha" / "configs" / "base.yaml"
CANDIDATE_V060 = REPO_ROOT / "aegis_alpha" / "models" / "strategy_candidates" / "aegis_h12_tail_risk_candidate_v060.json"
CANDIDATE_V052 = REPO_ROOT / "aegis_alpha" / "models" / "strategy_candidates" / "aegis_h12_tail_risk_candidate_v052.json"
ALLOWED_REGIMES = {"mixed", "chop", "high_vol"}
SHADOW_HEAVY_EVAL_ENABLED = os.environ.get("AEGIS_SHADOW_HEAVY_EVAL", "0") == "1"
SHADOW_CACHE_TTL_SECONDS = int(os.environ.get("AEGIS_SHADOW_CACHE_TTL_SECONDS", "60"))


def _fallback_shadow(symbol: str, reason: str, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    return build_shadow_signal(
        candidate=str((candidate or {}).get("config_id", "no_candidate_loaded")),
        candidate_status=str((candidate or {}).get("status", "NOT_LOADED")),
        live_enabled=False,
        strategy_name=str((candidate or {}).get("config_id", "aegis_shadow_candidate")),
        strategy_version="v060-shadow",
        symbol=symbol,
        timestamp="",
        action="HOLD",
        would_execute=False,
        reason=reason,
        size_mode="blocked",
        position_fraction=0.0,
        edge_score_h12=0.0,
        tail_risk_score=0.0,
        regime="unknown",
        risk_tier="blocked",
        model_status=str((candidate or {}).get("status", "NOT_LOADED")),
        not_live_reason=(candidate or {}).get("reason_not_live", ["candidate_or_model_load_failed"]),
    )


def _candidate_path() -> Path | None:
    if CANDIDATE_V060.exists():
        return CANDIDATE_V060
    if CANDIDATE_V052.exists():
        return CANDIDATE_V052
    return None


def _load_estimator(path: Path) -> tuple[Any, dict[str, Any]]:
    bundle = load_model_bundle(path)
    estimator = bundle.get("estimator") or bundle.get("classifier") or bundle.get("regressor")
    if estimator is None:
        raise RuntimeError(f"missing estimator in {path}")
    metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
    return estimator, metadata


@lru_cache(maxsize=2)
def _load_candidate_bundle(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    candidate = json.loads(path.read_text(encoding="utf-8"))
    model_paths = candidate.get("model_paths", {})
    edge_path = REPO_ROOT / model_paths["long_edge_h12"]
    tail_path = REPO_ROOT / model_paths["long_tail_risk_h12"]
    edge_model, edge_meta = _load_estimator(edge_path)
    tail_model, tail_meta = _load_estimator(tail_path)
    return {
        "candidate": candidate,
        "candidate_path": str(path),
        "models": {
            "long_edge_h12": edge_model,
            "long_tail_risk_h12": tail_model,
        },
        "model_paths": {
            "long_edge_h12": str(model_paths["long_edge_h12"]),
            "long_tail_risk_h12": str(model_paths["long_tail_risk_h12"]),
        },
        "model_versions": {
            "long_edge_h12": str(edge_meta.get("sklearn_version", "unknown")),
            "long_tail_risk_h12": str(tail_meta.get("sklearn_version", candidate.get("sklearn_version", "unknown"))),
        },
    }


def _cache_bucket(ttl_seconds: int = SHADOW_CACHE_TTL_SECONDS) -> int:
    return int(time.time() // max(int(ttl_seconds), 1))


@lru_cache(maxsize=2)
def _load_signal_market_cached(bucket: int):
    return load_signal_market(DEFAULT_CONFIG)


@lru_cache(maxsize=2)
def _predict_scores_cached(path_text: str, bucket: int) -> tuple[Any, dict[str, np.ndarray]]:
    bundle = _load_candidate_bundle(path_text)
    market = _load_signal_market_cached(bucket)
    preds = predict_scores(market, bundle["models"])
    return market, preds


def runtime_debug() -> dict[str, Any]:
    return {
        "heavy_eval_enabled": SHADOW_HEAVY_EVAL_ENABLED,
        "candidate_path": str(_candidate_path()) if _candidate_path() else None,
        "candidate_cache": _load_candidate_bundle.cache_info()._asdict(),
        "market_cache": _load_signal_market_cached.cache_info()._asdict(),
        "score_cache": _predict_scores_cached.cache_info()._asdict(),
    }


def _band_thresholds(candidate: dict[str, Any]) -> list[float]:
    bands = candidate.get("sizing_config", {}).get("bands") or candidate.get("oos_metrics", {}).get("bands") or []
    return sorted({float(item["max_pct"]) for item in bands})


def _size_for_tail(candidate: dict[str, Any], tail_score: float, thresholds: dict[float, float]) -> tuple[str, float, str]:
    bands = candidate.get("sizing_config", {}).get("bands") or candidate.get("oos_metrics", {}).get("bands") or []
    for band in bands:
        max_pct = float(band["max_pct"])
        if tail_score <= thresholds[max_pct]:
            return str(band["label"]), float(band["fraction"]), f"h12_top3_tail{int(max_pct * 100)}_{band['label']}"
    return "blocked", 0.0, "tail_risk_above_50"


def evaluate_shadow_candidate(symbol: str, payload: dict | None = None) -> dict[str, Any]:
    path = _candidate_path()
    if path is None:
        return {
            "candidate": "no_candidate_loaded",
            "candidate_status": "NOT_LOADED",
            "live_enabled": False,
            "model_paths": {},
            "model_versions": {},
            "price": None,
            "shadow": _fallback_shadow(symbol, "no_candidate_loaded"),
        }

    try:
        bundle = _load_candidate_bundle(str(path))
        candidate = bundle["candidate"]
        if not SHADOW_HEAVY_EVAL_ENABLED:
            return {
                "candidate": path.stem,
                "candidate_status": str(candidate.get("status", "UNKNOWN")),
                "live_enabled": False,
                "model_paths": bundle["model_paths"],
                "model_versions": bundle["model_versions"],
                "price": None,
                "shadow": _fallback_shadow(symbol, "shadow_heavy_eval_disabled_for_api_stability", candidate),
            }
        market, preds = _predict_scores_cached(str(path), _cache_bucket())
        if len(market.signal_features) == 0:
            shadow = _fallback_shadow(symbol, "insufficient_data", candidate)
        else:
            rel_idx = len(preds["long_edge_h12"]) - 1
            step = market.cfg.model.window_size + rel_idx
            if rel_idx < 0 or step >= len(market.close):
                shadow = _fallback_shadow(symbol, "insufficient_data", candidate)
            else:
                entry = candidate.get("entry_rule", {"mode": "top_pct", "value": 0.03})
                edge_threshold = threshold_for_rule(
                    preds["long_edge_h12"],
                    RuleCondition("long_edge_h12", str(entry.get("mode", "top_pct")), float(entry.get("value", 0.03))),
                )
                tail_thresholds = {
                    pct: threshold_for_rule(preds["long_tail_risk_h12"], RuleCondition("long_tail_risk_h12", "bottom_pct", pct))
                    for pct in _band_thresholds(candidate)
                }
                edge_score = float(preds["long_edge_h12"][rel_idx])
                tail_score = float(preds["long_tail_risk_h12"][rel_idx])
                regime = str(market.regimes[step])
                action = "HOLD"
                would_execute = False
                size_mode = "blocked"
                position_fraction = 0.0
                risk_tier = "blocked"
                allowed_regimes = set(candidate.get("allowed_regimes", sorted(ALLOWED_REGIMES)))
                if regime not in allowed_regimes:
                    reason = "regime_block"
                elif edge_score < edge_threshold:
                    reason = "edge_below_h12_top3"
                else:
                    size_mode, position_fraction, reason = _size_for_tail(candidate, tail_score, tail_thresholds)
                    if position_fraction > 0.0:
                        action = "LONG"
                        would_execute = True
                        risk_tier = "research_candidate"
                    else:
                        reason = "tail_risk_above_50"
                shadow = build_shadow_signal(
                    candidate=path.stem,
                    candidate_status=str(candidate.get("status", "UNKNOWN")),
                    live_enabled=False,
                    strategy_name=str(candidate.get("config_id", path.stem)),
                    strategy_version="v060-shadow",
                    symbol=symbol,
                    timestamp=str(market.timestamps[step]),
                    action=action,
                    would_execute=would_execute,
                    reason=reason,
                    size_mode=size_mode,
                    position_fraction=position_fraction,
                    edge_score_h12=edge_score,
                    tail_risk_score=tail_score,
                    regime=regime,
                    risk_tier=risk_tier,
                    model_status=str(candidate.get("status", "UNKNOWN")),
                    not_live_reason=candidate.get("reason_not_live", []),
                )
                return {
                    "candidate": path.stem,
                    "candidate_status": str(candidate.get("status", "UNKNOWN")),
                    "live_enabled": False,
                    "model_paths": bundle["model_paths"],
                    "model_versions": bundle["model_versions"],
                    "price": float(market.close[step]),
                    "thresholds": {
                        "long_edge_h12_top3": float(edge_threshold),
                        **{f"long_tail_risk_h12_bottom{int(pct * 100)}": float(value) for pct, value in tail_thresholds.items()},
                    },
                    "shadow": shadow,
                }
        return {
            "candidate": path.stem,
            "candidate_status": str(candidate.get("status", "UNKNOWN")),
            "live_enabled": False,
            "model_paths": bundle["model_paths"],
            "model_versions": bundle["model_versions"],
            "price": None,
            "shadow": shadow,
        }
    except Exception as exc:  # pragma: no cover - endpoint safety
        LOGGER.exception("shadow candidate evaluation failed")
        return {
            "candidate": path.stem,
            "candidate_status": "LOAD_FAILED",
            "live_enabled": False,
            "model_paths": {},
            "model_versions": {},
            "price": None,
            "shadow": _fallback_shadow(symbol, "candidate_or_model_load_failed"),
            "error": repr(exc),
        }
