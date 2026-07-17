from pathlib import Path

import yaml

from aegis.config import CANONICAL_SYMBOLS, EXPECTED_UNIVERSE_SIZE


EXPECTED_SYMBOLS = (
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


def test_scaffold_universe_is_exact_and_stable() -> None:
    assert CANONICAL_SYMBOLS == EXPECTED_SYMBOLS
    assert len(CANONICAL_SYMBOLS) == EXPECTED_UNIVERSE_SIZE
    assert len(set(CANONICAL_SYMBOLS)) == EXPECTED_UNIVERSE_SIZE


def test_universe_yaml_matches_the_python_contract() -> None:
    path = Path(__file__).parents[2] / "config" / "universe.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert tuple(payload["symbols"]) == CANONICAL_SYMBOLS
    assert payload["timeframe"] == "5m"
    assert payload["symbol_set_hash"] == "TODO_COMPUTE_AND_HANDSHAKE"


def test_python_and_typescript_scaffolds_match_the_operational_source() -> None:
    root = Path(__file__).parents[2]
    ts_root = root / "binance-futures-bot-ts"
    live = yaml.safe_load((ts_root / "regime_config.live.yaml").read_text(encoding="utf-8"))
    scaffold = yaml.safe_load(
        (ts_root / "config" / "regimen.config.yaml").read_text(encoding="utf-8")
    )

    operational_symbols = tuple(live["symbols"]["entries"])
    scaffold_symbols = tuple(scaffold["universe"]["symbols"])
    assert operational_symbols == CANONICAL_SYMBOLS == scaffold_symbols
    assert live["aegis"]["regime_context"]["timeframe"] == "5m"
    assert scaffold["universe"]["timeframe"] == "5m"
    assert scaffold["execution"]["enabledByConfig"] is False
