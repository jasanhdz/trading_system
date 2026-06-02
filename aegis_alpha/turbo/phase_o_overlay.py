#!/usr/bin/env python3
"""Persistent Phase O SHORT overlay for turbo active manifests.

The scheduled retrain owns the base active models. Phase O owns a SHORT-only
experimental overlay. This module reapplies the overlay after a base promotion
without changing LONG model paths.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE_O_POINTER_NAME = "phase_o_short_manifest.json"
PHASE_O_SCHEMA = "aegis_phase_o_overlay_v1"
ENTRY_SYMBOLS = (
    "LTCUSDT", "AVAXUSDT", "ETHUSDT", "SUIUSDT", "ADAUSDT",
    "DOGEUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
)
AVOID_ONLY_SYMBOLS = ("LINKUSDT",)
PHASE_O_SYMBOLS = ENTRY_SYMBOLS + AVOID_ONLY_SYMBOLS


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phase_o_pointer(base_model_dir: str | Path) -> dict[str, Any]:
    """Load the Phase O pointer manifest."""
    path = Path(base_model_dir) / PHASE_O_POINTER_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Phase O pointer missing: {path}")
    pointer = _read_json(path)
    pointer["_path"] = str(path)
    return pointer


def load_phase_o_global_manifest(base_model_dir: str | Path) -> dict[str, Any]:
    """Load the latest global Phase O manifest, tolerating moved artifact dirs."""
    base = Path(base_model_dir)
    pointer = load_phase_o_pointer(base)
    raw = pointer.get("latest_manifest")
    if not raw:
        raise ValueError("Phase O pointer has no latest_manifest")
    candidate = Path(str(raw))
    if not candidate.is_file():
        candidate = base / candidate.name
    if not candidate.is_file():
        raise FileNotFoundError(f"Phase O global manifest missing: {raw}")
    manifest = _read_json(candidate)
    manifest["_path"] = str(candidate)
    return manifest


def _artifact_stamp(global_manifest: dict[str, Any]) -> str:
    stamp = str(global_manifest.get("artifact_stamp") or global_manifest.get("latest_artifact_stamp") or "")
    if stamp.startswith("phase_o_"):
        return stamp[len("phase_o_"):]
    return stamp


def _find_symbol_artifact_dir(symbol: str, base: Path, stamp: str) -> Path:
    artifact_name = f"phase_o_{stamp}"
    candidates = [base / symbol / "active" / artifact_name]
    candidates.extend(sorted((base / symbol / "backups").glob(f"*/{artifact_name}"), reverse=True))
    for candidate in candidates:
        if (candidate / "symbol_shadow_manifest.json").is_file():
            return candidate
    raise FileNotFoundError(f"Phase O artifact dir missing for {symbol}: {artifact_name}")


def _entry_model_path(artifact_dir: Path, artifact_manifest: dict[str, Any]) -> Path:
    files = [Path(str(value)).name for value in artifact_manifest.get("model_files", [])]
    candidates = [artifact_dir / name for name in files]
    candidates.extend(sorted(artifact_dir.glob("turbo_short_edge_*phase_o_*.joblib")))
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix == ".joblib" and candidate.name.startswith("turbo_short_edge_"):
            return candidate.resolve()
    raise FileNotFoundError(f"Phase O entry model missing under {artifact_dir}")


def _avoid_model_paths(artifact_dir: Path, artifact_manifest: dict[str, Any]) -> list[Path]:
    expected = ("micro_hit_classifier.joblib", "micro_quality_regressor.joblib", "micro_danger_classifier.joblib")
    discovered = [artifact_dir / name for name in expected]
    if not all(path.is_file() for path in discovered):
        files = [artifact_dir / Path(str(value)).name for value in artifact_manifest.get("model_files", [])]
        discovered = [path for path in files if path.is_file()]
    if not discovered or not all(path.is_file() for path in discovered):
        raise FileNotFoundError(f"Phase O LINK avoid-only models missing under {artifact_dir}")
    return [path.resolve() for path in discovered]


def resolve_phase_o_symbol_overlay(symbol: str, base_model_dir: str | Path) -> dict[str, Any] | None:
    """Resolve the live-compatible overlay for one Phase O symbol."""
    symbol = symbol.upper()
    if symbol not in PHASE_O_SYMBOLS:
        return None
    base = Path(base_model_dir)
    pointer = load_phase_o_pointer(base)
    global_manifest = load_phase_o_global_manifest(base)
    stamp = _artifact_stamp(global_manifest) or str(pointer.get("latest_artifact_stamp", ""))
    if not stamp:
        raise ValueError("Phase O artifact stamp missing")
    artifact_dir = _find_symbol_artifact_dir(symbol, base, stamp)
    artifact_manifest_path = artifact_dir / "symbol_shadow_manifest.json"
    artifact_manifest = _read_json(artifact_manifest_path)
    if str(artifact_manifest.get("symbol", "")).upper() != symbol:
        raise ValueError(f"symbol artifact mismatch for {symbol}: {artifact_manifest_path}")
    lookback_days = int(artifact_manifest.get("lookback_days", 0) or 0)
    if lookback_days <= 0:
        raise ValueError(f"lookback_days missing for {symbol}")
    result: dict[str, Any] = {
        "schema_version": PHASE_O_SCHEMA,
        "symbol": symbol,
        "artifact_stamp": stamp,
        "artifact_dir": str(artifact_dir.resolve()),
        "artifact_manifest_path": str(artifact_manifest_path.resolve()),
        "artifact_manifest_sha256": _sha256(artifact_manifest_path),
        "lookback_days": lookback_days,
        "short_window_key": f"short_{lookback_days}d",
        "shadow_type": artifact_manifest.get("shadow_type"),
        "entry_enabled": symbol in ENTRY_SYMBOLS,
        "avoid_only": symbol in AVOID_ONLY_SYMBOLS,
        "global_manifest_path": global_manifest["_path"],
        "pointer_manifest_path": pointer["_path"],
    }
    if symbol in ENTRY_SYMBOLS:
        model_path = _entry_model_path(artifact_dir, artifact_manifest)
        result["phase_o_model_path"] = str(model_path)
        result["model_files"] = [str(model_path)]
    else:
        avoid_paths = _avoid_model_paths(artifact_dir, artifact_manifest)
        result["phase_o_avoid_artifacts"] = [str(path) for path in avoid_paths]
        result["model_files"] = [str(path) for path in avoid_paths]
    return result


def preserve_phase_o_fields(old_manifest: dict[str, Any] | None, new_manifest: dict[str, Any]) -> dict[str, Any]:
    """Carry audit metadata while leaving newly-trained base paths authoritative."""
    result = copy.deepcopy(new_manifest)
    if not old_manifest:
        return result
    for key, value in old_manifest.items():
        if key.startswith("phase_o_") and key not in result:
            result[key] = copy.deepcopy(value)
    if "pre_phase_o_live_model_paths" in old_manifest and "pre_phase_o_live_model_paths" not in result:
        result["pre_phase_o_live_model_paths"] = copy.deepcopy(old_manifest["pre_phase_o_live_model_paths"])
    return result


def apply_phase_o_overlay_to_active_manifest(symbol: str, active_manifest: dict[str, Any], base_model_dir: str | Path) -> dict[str, Any]:
    """Apply the Phase O SHORT overlay, leaving LONG model paths untouched."""
    symbol = symbol.upper()
    overlay = resolve_phase_o_symbol_overlay(symbol, base_model_dir)
    result = copy.deepcopy(active_manifest)
    if overlay is None:
        return result
    model_paths = copy.deepcopy(result.get("model_paths") or {})
    if not isinstance(model_paths, dict):
        raise ValueError(f"model_paths must be an object for {symbol}")
    base_paths = copy.deepcopy(model_paths)
    result.setdefault("pre_phase_o_live_model_paths", base_paths)
    result.update({
        "phase_o_overlay_schema": PHASE_O_SCHEMA,
        "phase_o_overlay_persistence_enabled": True,
        "phase_o_overlay_applied_at": _utc_now(),
        "phase_o_overlay_source_manifest": overlay["artifact_manifest_path"],
        "phase_o_live_artifact_stamp": overlay["artifact_stamp"],
        "phase_o_live_entry_symbols": list(ENTRY_SYMBOLS),
        "phase_o_live_avoid_only_symbols": list(AVOID_ONLY_SYMBOLS),
        "phase_o_live_requires_experimental_yaml": True,
        "phase_o_live_capital_profile": "test_capital",
        "phase_o_live_kill_switch_supported": True,
    })
    if symbol in ENTRY_SYMBOLS:
        model_paths[overlay["short_window_key"]] = overlay["phase_o_model_path"]
        result.update({
            "phase_o_live_enabled": True,
            "phase_o_live_mode": "experimental_short_only",
            "phase_o_live_entry_enabled": True,
            "phase_o_avoid_only": False,
            "phase_o_link_entry_enabled": False,
            "phase_o_model_path": overlay["phase_o_model_path"],
            "phase_o_model_window": overlay["short_window_key"],
        })
    else:
        result.update({
            "phase_o_live_enabled": False,
            "phase_o_live_mode": "avoid_only",
            "phase_o_live_entry_enabled": False,
            "phase_o_avoid_only": True,
            "phase_o_link_avoid_only_enabled": True,
            "phase_o_link_entry_enabled": False,
            "phase_o_avoid_artifacts": overlay["phase_o_avoid_artifacts"],
        })
    result["model_paths"] = model_paths
    return result


def validate_phase_o_overlay(symbol: str, manifest: dict[str, Any], base_model_dir: str | Path) -> list[str]:
    """Return validation errors for one Phase O manifest overlay."""
    symbol = symbol.upper()
    overlay = resolve_phase_o_symbol_overlay(symbol, base_model_dir)
    if overlay is None:
        return []
    errors: list[str] = []
    model_paths = manifest.get("model_paths") or {}
    if not isinstance(model_paths, dict):
        return ["model_paths_not_object"]
    if not manifest.get("phase_o_overlay_persistence_enabled"):
        errors.append("phase_o_overlay_persistence_disabled")
    if symbol in ENTRY_SYMBOLS:
        if not manifest.get("phase_o_live_enabled"):
            errors.append("phase_o_live_enabled_missing")
        actual = str(model_paths.get(overlay["short_window_key"], ""))
        expected = str(overlay["phase_o_model_path"])
        if actual != expected:
            errors.append(f"phase_o_short_path_mismatch:{overlay['short_window_key']}")
        if not Path(expected).is_file():
            errors.append("phase_o_entry_model_missing")
    else:
        if manifest.get("phase_o_link_entry_enabled") or manifest.get("phase_o_live_entry_enabled"):
            errors.append("link_entry_enabled")
        if not manifest.get("phase_o_avoid_only"):
            errors.append("link_avoid_only_missing")
        phase_paths = [str(value) for value in model_paths.values() if "/phase_o_" in str(value)]
        if phase_paths:
            errors.append("link_has_phase_o_entry_model_path")
        avoid_paths = manifest.get("phase_o_avoid_artifacts") or []
        if not avoid_paths or not all(Path(str(path)).is_file() for path in avoid_paths):
            errors.append("link_avoid_artifacts_missing")
    previous = manifest.get("pre_phase_o_live_model_paths") or {}
    if isinstance(previous, dict):
        for key, previous_path in previous.items():
            if str(key).startswith("long_") and str(model_paths.get(key, "")) != str(previous_path):
                errors.append(f"long_path_changed:{key}")
    else:
        errors.append("pre_phase_o_live_model_paths_not_object")
    return errors
