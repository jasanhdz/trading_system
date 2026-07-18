"""Hash-gated, allowlisted loader for frozen historical Gen2 pickles."""

from __future__ import annotations

import importlib
import json
import pickle
import platform
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn._loss._loss as runtime_loss

# Importing the Cython extension registers its declared short module name. The replay
# retains only the exact class object and removes that process-global compatibility leak.
sys.modules.pop("_loss", None)

from .manifests import sha256_file


CUSTOM_GLOBALS = {
    ("__main__", "MedianImputer"),
    ("aegis_alpha.tools.gen2_rv2_train", "MedianImputer"),
}

# Exact globals observed by pickletools in the two frozen artifacts.
ALLOWED_GLOBALS = CUSTOM_GLOBALS | {
    ("_loss", "CyHalfBinomialLoss"), ("_loss", "CyPinballLoss"),
    ("builtins", "slice"),
    ("numpy", "dtype"), ("numpy", "ndarray"),
    ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("numpy.random._pcg64", "PCG64"),
    ("numpy.random._pickle", "__bit_generator_ctor"), ("numpy.random._pickle", "__generator_ctor"),
    ("numpy.random.bit_generator", "SeedSequence"),
    ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence"),
    ("pandas", "Index"), ("pandas", "Series"), ("pandas", "StringDtype"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"),
    ("pandas.arrays", "StringArray"), ("pandas.arrays", "StringDtype"),
    ("pandas.core.indexes.base", "Index"), ("pandas.core.indexes.base", "_new_Index"),
    ("pandas.core.internals.managers", "SingleBlockManager"),
    ("sklearn._loss.link", "IdentityLink"), ("sklearn._loss.link", "Interval"),
    ("sklearn._loss.link", "LogitLink"),
    ("sklearn._loss.loss", "HalfBinomialLoss"), ("sklearn._loss.loss", "PinballLoss"),
    ("sklearn._loss._loss", "CyHalfBinomialLoss"),
    ("sklearn.ensemble._forest", "ExtraTreesRegressor"),
    ("sklearn.ensemble._forest", "RandomForestClassifier"),
    ("sklearn.ensemble._hist_gradient_boosting.binning", "_BinMapper"),
    ("sklearn.ensemble._hist_gradient_boosting.gradient_boosting", "HistGradientBoostingClassifier"),
    ("sklearn.ensemble._hist_gradient_boosting.gradient_boosting", "HistGradientBoostingRegressor"),
    ("sklearn.ensemble._hist_gradient_boosting.predictor", "TreePredictor"),
    ("sklearn.isotonic", "IsotonicRegression"),
    ("sklearn.preprocessing._label", "LabelEncoder"),
    ("sklearn.tree._classes", "DecisionTreeClassifier"),
    ("sklearn.tree._classes", "ExtraTreeRegressor"),
    ("sklearn.tree._tree", "Tree"),
}

FROZEN_PICKLES = {
    "rv2": (Path("/home/jasan/Develop/aegis_gen2/rv2/20260711T171832Z/rv2_candidate.pkl"), "69c03e12ea8aad36b05410ee716ffbf5ae3a80d3d274be8d6d1b87f95ff9e4e4"),
    "eqm1": (Path("/home/jasan/Develop/aegis_gen2/eqm1/20260711T201456Z/eqm1_candidate.pkl"), "77887b7c33c5a472f51fe168564f4972d48a8c5b094b01080261d6896b1cf3fa"),
}


class HistoricalCompatibilityError(RuntimeError):
    pass


def resolve_legacy_serialization_global(module: str, name: str) -> Any | None:
    """Resolve only explicitly approved legacy Cython serialization paths."""
    approved = {
        ("_loss", "CyPinballLoss"): runtime_loss.CyPinballLoss,
        ("_loss", "CyHalfBinomialLoss"): runtime_loss.CyHalfBinomialLoss,
    }
    resolved = approved.get((module, name))
    if resolved is None:
        return None
    if (
        resolved.__name__ != name
        or resolved.__module__ != "_loss"
        or resolved is not getattr(sklearn._loss._loss, name)
    ):
        raise HistoricalCompatibilityError(f"LEGACY_GLOBAL_IDENTITY_MISMATCH: {module}.{name}")
    return resolved


class AllowlistedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_GLOBALS:
            raise HistoricalCompatibilityError(f"UNINVENTORIED_PICKLE_GLOBAL: {module}.{name}")
        legacy = resolve_legacy_serialization_global(module, name)
        if legacy is not None:
            return legacy
        resolved = super().find_class(module, name)
        public_identities = {
            ("pandas", "Index"): pd.Index,
            ("pandas", "StringDtype"): pd.StringDtype,
        }
        expected = public_identities.get((module, name))
        if expected is not None and resolved is not expected:
            raise HistoricalCompatibilityError(f"PUBLIC_GLOBAL_IDENTITY_MISMATCH: {module}.{name}")
        return resolved


@contextmanager
def historical_namespace() -> Iterator[type]:
    root = Path(__file__).with_name("historical_namespace").resolve()
    before_path = tuple(sys.path); before_modules = set(sys.modules)
    main = sys.modules["__main__"]; had_alias = hasattr(main, "MedianImputer")
    prior_alias = getattr(main, "MedianImputer", None)
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("aegis_alpha.tools.gen2_rv2_train")
        main.MedianImputer = module.MedianImputer
        yield module.MedianImputer
    finally:
        sys.path[:] = before_path
        if had_alias:
            main.MedianImputer = prior_alias
        elif hasattr(main, "MedianImputer"):
            delattr(main, "MedianImputer")
        for name in tuple(set(sys.modules) - before_modules):
            if name == "aegis_alpha" or name.startswith("aegis_alpha."):
                sys.modules.pop(name, None)


@dataclass(frozen=True)
class LoadedHistoricalPickle:
    artifact_id: str
    path: Path
    sha256: str
    payload: Mapping[str, Any]
    warnings: tuple[str, ...]
    summary: Mapping[str, Any]


def environment_versions() -> Mapping[str, str | None]:
    return {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "joblib": joblib.__version__, "cloudpickle": None,
    }


def _component_summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    result: dict[str, Any] = {"root_class": f"{type(payload).__module__}.{type(payload).__name__}", "components": {}}
    for key, value in payload.items():
        if hasattr(value, "get_params") or key in {"imputer", "calibrator"}:
            entry: dict[str, Any] = {"class": f"{type(value).__module__}.{type(value).__name__}"}
            for attribute in ("n_features_in_", "feature_names_in_", "classes_"):
                if hasattr(value, attribute):
                    raw = getattr(value, attribute)
                    entry[attribute] = raw.tolist() if hasattr(raw, "tolist") else raw
            if hasattr(value, "get_params"):
                entry["parameters"] = value.get_params(deep=False)
            result["components"][key] = entry
    result["feature_count"] = len(payload.get("features", ()))
    return result


def load_frozen_pickle(artifact_id: str) -> LoadedHistoricalPickle:
    if artifact_id not in FROZEN_PICKLES:
        raise HistoricalCompatibilityError(f"unknown frozen pickle: {artifact_id}")
    path, expected = FROZEN_PICKLES[artifact_id]
    actual = sha256_file(path)
    if actual != expected:
        raise HistoricalCompatibilityError(f"historical pickle hash mismatch: {artifact_id}")
    with historical_namespace(), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with path.open("rb") as handle:
            payload = AllowlistedUnpickler(handle).load()
    semantic = [str(item.message) for item in captured if "InconsistentVersionWarning" in type(item.message).__name__]
    if semantic:
        raise HistoricalCompatibilityError(f"sklearn semantic compatibility warning: {semantic}")
    if not isinstance(payload, Mapping):
        raise HistoricalCompatibilityError("historical pickle root is not a mapping")
    return LoadedHistoricalPickle(artifact_id, path, actual, payload, tuple(str(item.message) for item in captured), _component_summary(payload))


def write_load_report(path: Path) -> Mapping[str, Any]:
    artifacts = [load_frozen_pickle(name) for name in ("rv2", "eqm1")]
    report = {
        "schema_version": "aegis-historical-pickle-load-v1", "environment": environment_versions(),
        "artifacts": [{"artifact_id": item.artifact_id, "path": str(item.path), "sha256": item.sha256, "warnings": item.warnings, "summary": item.summary} for item in artifacts],
        "namespace_persistent": any(name == "aegis_alpha" or name.startswith("aegis_alpha.") for name in sys.modules),
    }
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return report
