"""Immutable scientific configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .domain import ScientificLayerName
from .utils import Sha256HashProvider, ordered_name_hash


EXPECTED_UNIVERSE_SIZE = 11
CANONICAL_SYMBOLS = (
    "ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "LTCUSDT",
)
CANONICAL_SYMBOL_SET_HASH = ordered_name_hash(CANONICAL_SYMBOLS)
EXPECTED_LAYERS = tuple(layer.value for layer in ScientificLayerName)


class ConfigurationError(ValueError):
    """Raised when frozen scientific configuration is inconsistent."""


@dataclass(frozen=True)
class UniverseConfig:
    schema_version: str
    universe_id: str
    symbols: tuple[str, ...]
    timeframe: str
    symbol_set_hash: str
    minimum_history_bars: int
    maximum_snapshot_age_seconds: int
    maximum_gap_bars: int


@dataclass(frozen=True)
class ModelConfig:
    schema_version: str
    model_bundle_id: str
    feature_schema_version: str
    artifact_registry: Path
    ordered_layers: tuple[str, ...]
    direction_threshold: float
    selection_threshold: float
    trrm_max_tail_probability: float
    qmae_max_fraction: float
    eqm_min_score: float
    maximum_decision_age_seconds: int


@dataclass(frozen=True)
class BrainConfig:
    schema_version: str
    config_version: str
    contract_version: str
    build_id: str
    universe: UniverseConfig
    models: ModelConfig
    evidence_path: Path
    persistence_enabled: bool
    config_hash: str


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping")
    return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"configuration file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(payload, str(path))


def _fraction(value: Any, name: str, *, allow_zero: bool = True) -> float:
    result = float(value)
    lower_ok = result >= 0 if allow_zero else result > 0
    if not lower_ok or result > 1:
        raise ConfigurationError(f"{name} must be within {'[0, 1]' if allow_zero else '(0, 1]'}")
    return result


def load_brain_config(config_dir: Path) -> BrainConfig:
    """Load all scientific configuration once and verify frozen handshakes."""

    root = config_dir.resolve()
    brain = _load_yaml(root / "brain.yaml")
    universe_data = _load_yaml(root / str(brain.get("runtime", {}).get("universe_config", "universe.yaml")))
    models_data = _load_yaml(root / str(brain.get("runtime", {}).get("models_config", "models.yaml")))

    symbols = tuple(str(symbol) for symbol in universe_data.get("symbols", ()))
    if len(symbols) != EXPECTED_UNIVERSE_SIZE or len(set(symbols)) != EXPECTED_UNIVERSE_SIZE:
        raise ConfigurationError("universe must contain exactly eleven unique symbols")
    if symbols != CANONICAL_SYMBOLS:
        raise ConfigurationError("universe symbols or canonical order do not match the operational contract")
    actual_symbol_hash = ordered_name_hash(symbols)
    if universe_data.get("symbol_set_hash") != actual_symbol_hash:
        raise ConfigurationError("symbol_set_hash does not match the ordered universe")
    timeframe = str(universe_data.get("timeframe", ""))
    if timeframe != "5m":
        raise ConfigurationError("clean rebuild currently supports only the frozen 5m timeframe")

    layers = tuple(str(value) for value in models_data.get("layers", ()))
    if layers != EXPECTED_LAYERS:
        raise ConfigurationError("scientific layer order must be REGIME/RV2/TRRM/QMAE/EQM")
    thresholds = _mapping(models_data.get("thresholds", {}), "models.thresholds")
    model_bundle_id = str(models_data.get("model_bundle_id", ""))
    if not model_bundle_id or model_bundle_id.startswith("TODO"):
        raise ConfigurationError("an explicit approved model_bundle_id is required")

    universe = UniverseConfig(
        schema_version=str(universe_data["schema_version"]),
        universe_id=str(universe_data["universe_id"]),
        symbols=symbols,
        timeframe=timeframe,
        symbol_set_hash=actual_symbol_hash,
        minimum_history_bars=int(universe_data.get("minimum_history_bars", 48)),
        maximum_snapshot_age_seconds=int(universe_data.get("maximum_snapshot_age_seconds", 600)),
        maximum_gap_bars=int(universe_data.get("maximum_gap_bars", 0)),
    )
    if universe.minimum_history_bars < 48 or universe.maximum_snapshot_age_seconds <= 0:
        raise ConfigurationError("universe history and freshness limits are invalid")

    models = ModelConfig(
        schema_version=str(models_data["schema_version"]),
        model_bundle_id=model_bundle_id,
        feature_schema_version=str(models_data["feature_schema_version"]),
        artifact_registry=(root / str(models_data["artifact_registry"])).resolve(),
        ordered_layers=layers,
        direction_threshold=_fraction(thresholds.get("direction_probability", 0.5), "direction_probability"),
        selection_threshold=_fraction(thresholds.get("selection_score", 0.5), "selection_score"),
        trrm_max_tail_probability=_fraction(thresholds.get("trrm_max_tail_probability", 0.70), "trrm_max_tail_probability"),
        qmae_max_fraction=_fraction(thresholds.get("qmae_max_fraction", 0.03), "qmae_max_fraction"),
        eqm_min_score=_fraction(thresholds.get("eqm_min_score", 0.0), "eqm_min_score"),
        maximum_decision_age_seconds=int(models_data.get("maximum_decision_age_seconds", 30)),
    )
    if models.maximum_decision_age_seconds <= 0:
        raise ConfigurationError("maximum_decision_age_seconds must be positive")

    normalized = {"brain": brain, "universe": universe_data, "models": models_data}
    config_hash = Sha256HashProvider().digest_value(normalized)
    evidence = _mapping(brain.get("evidence", {}), "brain.evidence")
    return BrainConfig(
        schema_version=str(brain["schema_version"]),
        config_version=str(brain["config_version"]),
        contract_version=str(brain["contract_version"]),
        build_id=str(brain["build_id"]),
        universe=universe,
        models=models,
        evidence_path=(root / str(evidence.get("path", "../data/scientific_evidence.jsonl"))).resolve(),
        persistence_enabled=bool(evidence.get("persistence_enabled", False)),
        config_hash=config_hash,
    )
