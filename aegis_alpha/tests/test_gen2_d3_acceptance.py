#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import aegis_alpha.tools.gen2_d3_acceptance as acc


def make_series(tmp: Path, n: int = 400, spike_every: int | None = 25) -> Path:
    ts = pd.date_range("2026-06-01", periods=n, freq="5min")
    close = np.full(n, 100.0)
    high = close + 0.1
    if spike_every:
        high[::spike_every] = close[::spike_every] * 1.02  # 2% adverse move = 0.40 ROE at 20x
    df = pd.DataFrame({
        "timestamp": ts.astype(str), "open": close, "high": high, "low": close - 0.1,
        "close": close, "volume": 10.0, "buy_volume": 5.0,
    })
    d = tmp / "series"
    d.mkdir(exist_ok=True)
    df.to_csv(d / "BTCUSDT_5m.csv", index=False)
    return d


def make_gen1(tmp: Path, series_dir: Path, corrupt_labels: bool) -> Path:
    candles = pd.read_csv(series_dir / "BTCUSDT_5m.csv")
    rows = []
    for idx in range(64, len(candles) - 30, 3):
        ts = candles["timestamp"].iloc[idx]
        window_high = candles["high"].iloc[idx + 1: idx + 13].max()
        entry = candles["close"].iloc[idx]
        mae = max(0.0, (window_high - entry) / entry) * 20.0
        label = int(mae >= 0.30)
        if corrupt_labels:
            label = 0  # Gen1 sensor missed every tail
            mae = 0.01
        rows.append({
            "id.symbol": "BTCUSDT", "id.timestamp": ts, "id.horizon": 12,
            "target.tail_risk_roe_030": label, "future_eval.future_mae_roe_proxy": mae,
        })
    path = tmp / "gen1_dense.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_label_impact_clean_gen1_passes_h3() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        series = make_series(tmp)
        gen1 = make_gen1(tmp, series, corrupt_labels=False)
        impact = acc.label_impact_vs_gen1(series, gen1)
        assert impact["rows_matched_on_canonical"] > 50
        assert impact["flips_0_to_1_real_tails_missed_by_gen1"] == 0
        assert impact["flips_1_to_0_fake_tails_in_gen1"] == 0
        assert impact["h3_violated"] is False


def test_label_impact_corrupt_gen1_violates_h3() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        series = make_series(tmp)
        gen1 = make_gen1(tmp, series, corrupt_labels=True)
        impact = acc.label_impact_vs_gen1(series, gen1)
        assert impact["flips_0_to_1_real_tails_missed_by_gen1"] > 0
        assert impact["h3_violated"] is True
        assert impact["direction"] == "up"


def test_lockbox_manifest_immutable() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        acc.LOCKBOX_MANIFEST_PATH = tmp / "GEN2_LOCKBOX_MANIFEST.json"
        manifest = acc.write_lockbox_manifest(
            {"artifact_id": "d3-v1-x", "timestamp_min": "2024-07-11", "timestamp_max": "2026-07-11"},
            "2026-07-11 09:00:00",
        )
        assert manifest["current_query_count"] == 0
        assert manifest["allowed_query_count_per_candidate"] == 1
        assert manifest["contains_no_forward_metrics"] is True
        data = json.loads(acc.LOCKBOX_MANIFEST_PATH.read_text())
        assert "forbidden_uses" in data and len(data["forbidden_uses"]) >= 5
        try:
            acc.write_lockbox_manifest({"artifact_id": "y", "timestamp_min": "a", "timestamp_max": "b"}, "z")
            raise AssertionError("second write must fail (immutability)")
        except FileExistsError:
            pass


if __name__ == "__main__":
    test_label_impact_clean_gen1_passes_h3()
    test_label_impact_corrupt_gen1_violates_h3()
    test_lockbox_manifest_immutable()
    print("test_gen2_d3_acceptance: OK")
