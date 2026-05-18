from __future__ import annotations

import numpy as np

from aegis_alpha.turbo import turbo_signal


def test_score_models_cached_reuses_same_symbol_timestamp(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_score_models(_x, _symbol):
        calls["count"] += 1
        return {"long_7d": 0.001, "short_7d": -0.001}, {"long": 1, "short": 0, "neutral": 0}

    monkeypatch.setattr(turbo_signal, "_manifest_signature", lambda symbol: ("test", symbol, 1))
    monkeypatch.setattr(turbo_signal, "_score_models", fake_score_models)
    turbo_signal._SCORE_CACHE.clear()

    x = np.zeros((1, 4), dtype=np.float32)
    first_scores, first_votes = turbo_signal._score_models_cached(x, "ETHUSDT", "2026-05-17T20:00:00+00:00")
    second_scores, second_votes = turbo_signal._score_models_cached(x, "ETHUSDT", "2026-05-17T20:00:00+00:00")

    assert calls["count"] == 1
    assert second_scores == first_scores
    assert second_votes == first_votes


def test_score_models_cached_separates_symbols_and_timestamps(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_score_models(_x, symbol):
        calls["count"] += 1
        return {"long_7d": float(calls["count"]), "short_7d": None}, {"long": 1, "short": 0, "neutral": 0}

    monkeypatch.setattr(turbo_signal, "_manifest_signature", lambda symbol: ("test", symbol, 1))
    monkeypatch.setattr(turbo_signal, "_score_models", fake_score_models)
    turbo_signal._SCORE_CACHE.clear()

    x = np.zeros((1, 4), dtype=np.float32)
    turbo_signal._score_models_cached(x, "ETHUSDT", "2026-05-17T20:00:00+00:00")
    turbo_signal._score_models_cached(x, "LINKUSDT", "2026-05-17T20:00:00+00:00")
    turbo_signal._score_models_cached(x, "ETHUSDT", "2026-05-17T20:01:00+00:00")

    assert calls["count"] == 3
