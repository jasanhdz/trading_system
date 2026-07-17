"""Scientific configuration contracts; parsing and validation remain TODO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EXPECTED_UNIVERSE_SIZE = 11
CANONICAL_SYMBOLS = (
    "ETHUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "LTCUSDT",
)


@dataclass(frozen=True)
class UniverseConfig:
    universe_id: str
    symbols: tuple[str, ...]
    timeframe: str
    symbol_set_hash: str


@dataclass(frozen=True)
class ModelConfig:
    model_bundle_id: str
    feature_schema_version: str
    ordered_layers: tuple[str, ...]


@dataclass(frozen=True)
class BrainConfig:
    contract_version: str
    universe: UniverseConfig
    models: ModelConfig


def load_brain_config(config_dir: Path) -> BrainConfig:
    """TODO: load YAML, validate versions, and verify the symbol handshake."""
    raise NotImplementedError("Scientific configuration loading is not implemented")
