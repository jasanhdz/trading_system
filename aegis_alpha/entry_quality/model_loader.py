from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from aegis_alpha.config import REPO_ROOT
from aegis_alpha.entry_quality.schema import MODEL_VERSION, STATUS


MODEL_DIR = REPO_ROOT / "aegis_alpha/models/entry_quality/v020"
MANIFEST_PATH = MODEL_DIR / "model_manifest.json"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"

_LOCK = threading.Lock()
_CACHE: "EntryQualityModelCache | None" = None
_LAST_ERRORS: list[str] = []


@dataclass
class ModelPair:
    entry_quality: Any | None
    tail_risk: Any | None
    scope: str


@dataclass
class EntryQualityModelCache:
    manifest: dict[str, Any]
    feature_columns: list[str]
    symbol_encoding: dict[str, int]
    global_entry_quality: Any | None
    global_tail_risk: Any | None
    symbol_models: dict[str, ModelPair]
    loaded_at: float


def normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("/", "").replace("-", "")


def _record_error(message: str) -> None:
    _LAST_ERRORS.append(message)
    del _LAST_ERRORS[:-20]


def _load_bundle(path: Path) -> Any | None:
    if not path.exists():
        _record_error(f"missing_model:{path}")
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        _record_error(f"load_failed:{path}:{exc!r}")
        return None


def _load_feature_columns() -> tuple[list[str], dict[str, int]]:
    if not FEATURE_COLUMNS_PATH.exists():
        _record_error(f"feature_columns_missing:{FEATURE_COLUMNS_PATH}")
        return [], {}
    try:
        raw = json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))
        return list(raw.get("feature_columns") or []), {str(k): int(v) for k, v in (raw.get("symbol_encoding") or {}).items()}
    except Exception as exc:
        _record_error(f"feature_columns_load_failed:{exc!r}")
        return [], {}


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        _record_error(f"manifest_missing:{MANIFEST_PATH}")
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        _record_error(f"manifest_load_failed:{exc!r}")
        return {}


def _load_symbol_pair(symbol: str) -> ModelPair | None:
    symbol_dir = MODEL_DIR / normalize_symbol(symbol)
    entry = _load_bundle(symbol_dir / "entry_quality_model.joblib")
    tail = _load_bundle(symbol_dir / "tail_risk_model.joblib")
    if entry is None or tail is None:
        return None
    return ModelPair(entry_quality=entry, tail_risk=tail, scope="symbol")


def load_entry_quality_models(force_reload: bool = False) -> EntryQualityModelCache:
    global _CACHE
    with _LOCK:
        if _CACHE is not None and not force_reload:
            return _CACHE
        _LAST_ERRORS.clear()
        manifest = _load_manifest()
        feature_columns, symbol_encoding = _load_feature_columns()
        global_entry = _load_bundle(MODEL_DIR / "global_entry_quality_model.joblib")
        global_tail = _load_bundle(MODEL_DIR / "global_tail_risk_model.joblib")
        symbol_models: dict[str, ModelPair] = {}
        for symbol in symbol_encoding:
            pair = _load_symbol_pair(symbol)
            if pair is not None:
                symbol_models[normalize_symbol(symbol)] = pair
        _CACHE = EntryQualityModelCache(
            manifest=manifest,
            feature_columns=feature_columns,
            symbol_encoding=symbol_encoding,
            global_entry_quality=global_entry,
            global_tail_risk=global_tail,
            symbol_models=symbol_models,
            loaded_at=time.time(),
        )
        return _CACHE


def get_model_pair(symbol: str) -> ModelPair:
    cache = load_entry_quality_models()
    normalized = normalize_symbol(symbol)
    pair = cache.symbol_models.get(normalized)
    if pair is not None:
        return pair
    if cache.global_entry_quality is not None and cache.global_tail_risk is not None:
        return ModelPair(cache.global_entry_quality, cache.global_tail_risk, "global")
    return ModelPair(None, None, "none")


def entry_quality_model_status(load_if_needed: bool = False) -> dict[str, Any]:
    cache = _CACHE if _CACHE is not None else (load_entry_quality_models() if load_if_needed else None)
    feature_count = len(cache.feature_columns) if cache else 0
    global_loaded = bool(cache and cache.global_entry_quality is not None and cache.global_tail_risk is not None)
    return {
        "enabled": True,
        "mode": "SHADOW",
        "model_version": MODEL_VERSION,
        "status": STATUS,
        "manifest_exists": MANIFEST_PATH.exists(),
        "global_models_loaded": global_loaded,
        "symbol_models_available": len(cache.symbol_models) if cache else sum(1 for p in MODEL_DIR.glob("*/entry_quality_model.joblib")),
        "feature_columns_count": feature_count if feature_count else (len(json.loads(FEATURE_COLUMNS_PATH.read_text()).get("feature_columns", [])) if FEATURE_COLUMNS_PATH.exists() else 0),
        "last_errors": list(_LAST_ERRORS),
        "cache_size": 0 if cache is None else 1 + len(cache.symbol_models),
        "loaded": cache is not None,
    }


def clear_entry_quality_model_cache() -> None:
    global _CACHE
    with _LOCK:
        _CACHE = None
        _LAST_ERRORS.clear()
