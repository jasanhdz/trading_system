from pathlib import Path

import pytest

from aegis_range_v1 import replay_benchmark


def test_replay_symbol_counts_discards_rows_and_sums_counts(monkeypatch):
    monkeypatch.setattr(replay_benchmark, "load_train_candles", lambda repo, symbol: ["candle"])
    monkeypatch.setattr(replay_benchmark, "load_train_funding", lambda repo, symbol: ["funding"])
    monkeypatch.setattr(replay_benchmark, "load_regime_cache", lambda path, count: ["snapshot"])
    monkeypatch.setattr(replay_benchmark, "structural_candidates", lambda: ("first", "second"))
    monkeypatch.setattr(
        replay_benchmark,
        "replay_structure",
        lambda symbol, candidate, candles, snapshots, funding: ([1], [1, 2], [1, 2, 3], [1, 2, 3, 4]),
    )

    result = replay_benchmark.replay_symbol_counts(Path("repo"), Path("run-a"), "BTCUSDT")

    assert result["symbol"] == "BTCUSDT"
    assert {key: result[key] for key in ("opportunities", "entries", "paths", "passages")} == {
        "opportunities": 2,
        "entries": 4,
        "paths": 6,
        "passages": 8,
    }


def test_parallel_replay_counts_validates_inputs():
    with pytest.raises(ValueError, match="at least 1"):
        replay_benchmark.parallel_replay_counts(Path("repo"), Path("run-a"), workers=0)
    with pytest.raises(ValueError, match="outside frozen universe"):
        replay_benchmark.parallel_replay_counts(
            Path("repo"),
            Path("run-a"),
            workers=1,
            symbols=("UNKNOWN",),
        )
