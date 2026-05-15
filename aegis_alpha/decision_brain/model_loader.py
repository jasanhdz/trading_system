from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


DEFAULT_MODEL_VERSION = "v010"
DEFAULT_MODEL_DIR = Path("aegis_alpha/models/decision_brain/v010")


@dataclass
class DecisionBrainArtifacts:
    model_version: str
    model_dir: Path
    manifest: dict[str, Any]
    model_package: dict[str, Any]
    preprocessor: Any
    feature_columns: list[str]


_LOCK = threading.Lock()
_CACHE: DecisionBrainArtifacts | None = None
_LAST_ERRORS: list[str] = []


def _remember_error(message: str) -> None:
    _LAST_ERRORS.append(message)
    del _LAST_ERRORS[:-10]


def _paths(model_dir: Path) -> dict[str, Path]:
    return {
        "manifest": model_dir / "model_manifest.json",
        "model": model_dir / "decision_brain_model.joblib",
        "preprocessor": model_dir / "preprocessor.joblib",
        "feature_columns": model_dir / "feature_columns.json",
    }


def load_decision_brain_artifacts(
    model_version: str = DEFAULT_MODEL_VERSION,
    model_dir: Path = DEFAULT_MODEL_DIR,
    *,
    force_reload: bool = False,
) -> DecisionBrainArtifacts | None:
    global _CACHE
    with _LOCK:
        if _CACHE is not None and not force_reload:
            return _CACHE
        paths = _paths(model_dir)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            _remember_error(f"decision_brain_artifacts_missing:{','.join(missing)}")
            return None
        try:
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            model_package = joblib.load(paths["model"])
            preprocessor = joblib.load(paths["preprocessor"])
            feature_columns = json.loads(paths["feature_columns"].read_text(encoding="utf-8"))
            _CACHE = DecisionBrainArtifacts(
                model_version=model_version,
                model_dir=model_dir,
                manifest=manifest,
                model_package=model_package,
                preprocessor=preprocessor,
                feature_columns=[str(item) for item in feature_columns],
            )
            return _CACHE
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            _remember_error(f"decision_brain_artifact_load_error:{exc!r}")
            return None


def decision_brain_loader_status() -> dict[str, Any]:
    paths = _paths(DEFAULT_MODEL_DIR)
    artifacts = load_decision_brain_artifacts()
    return {
        "enabled": True,
        "mode": "SHADOW",
        "model_version": DEFAULT_MODEL_VERSION,
        "manifest_exists": paths["manifest"].exists(),
        "model_loaded": artifacts is not None,
        "feature_columns_count": len(artifacts.feature_columns) if artifacts is not None else 0,
        "model_dir": str(DEFAULT_MODEL_DIR),
        "last_errors": list(_LAST_ERRORS),
    }


def reset_decision_brain_cache() -> None:
    global _CACHE
    with _LOCK:
        _CACHE = None
