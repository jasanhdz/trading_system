from pathlib import Path

import pytest
import yaml

from aegis.config import CANONICAL_SYMBOLS, CANONICAL_SYMBOL_SET_HASH, ConfigurationError, load_brain_config


def test_universe_configuration_is_exact_versioned_and_stable() -> None:
    config = load_brain_config(Path(__file__).parents[2] / "config")
    assert config.universe.symbols == CANONICAL_SYMBOLS
    assert len(set(config.universe.symbols)) == 11
    assert config.universe.symbol_set_hash == CANONICAL_SYMBOL_SET_HASH
    assert config.universe.timeframe == "5m"


def test_python_and_typescript_universes_match_operational_source() -> None:
    root = Path(__file__).parents[2]
    ts_root = root / "binance-futures-bot-ts"
    live = yaml.safe_load((ts_root / "regime_config.live.yaml").read_text(encoding="utf-8"))
    integration = yaml.safe_load((ts_root / "config" / "regimen.config.yaml").read_text(encoding="utf-8"))
    assert tuple(live["symbols"]["entries"]) == CANONICAL_SYMBOLS == tuple(integration["universe"]["symbols"])
    assert integration["execution"]["enabledByConfig"] is False


def test_configuration_rejects_tampered_universe(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "config"
    for name in ("brain.yaml", "models.yaml", "universe.yaml"):
        (tmp_path / name).write_text((source / name).read_text(), encoding="utf-8")
    payload = yaml.safe_load((tmp_path / "universe.yaml").read_text())
    payload["symbols"][-1] = "BTCUSDT"
    (tmp_path / "universe.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_brain_config(tmp_path)
