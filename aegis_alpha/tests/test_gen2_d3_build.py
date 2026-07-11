#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

import aegis_alpha.tools.gen2_d3_build as build
import aegis_alpha.tools.gen2_d3_common as common
import aegis_alpha.tools.gen2_d3_snapshot as snap
from test_gen2_d3_snapshot import fake_fetch_factory, make_market

BAR_MS = 5 * 60_000


def make_snapshot(tmp: Path, n_bars: int = 9000, symbols=("BTCUSDT", "ETHUSDT", "ADAUSDT"), broken: str | None = None) -> Path:
    market = make_market(n_bars)
    root = tmp / "snaps"
    fetch = fake_fetch_factory(market)
    result = snap.build_snapshot(symbols=list(symbols), range_days=40, fetch_fn=fetch, snapshot_root=root, spacing=0, end_override_ms=max(market) + BAR_MS)
    d = Path(result["directory"])
    if broken == "duplicate":
        csv = d / "ADAUSDT_5m.csv"
        df = pd.read_csv(csv)
        pd.concat([df, df.iloc[[100]]]).sort_values("open_time").to_csv(csv, index=False)
    if broken == "bad_high":
        csv = d / "ADAUSDT_5m.csv"
        df = pd.read_csv(csv)
        df.loc[50, "high"] = df.loc[50, ["open", "close"]].min() - 5.0
        df.to_csv(csv, index=False)
    if broken == "gap":
        csv = d / "ADAUSDT_5m.csv"
        df = pd.read_csv(csv)
        df.drop(index=range(200, 260)).to_csv(csv, index=False)  # 60-bar hole
    return d


def _patch_roots(tmp: Path) -> None:
    for mod in (build, common):
        mod.SERIES_ROOT = tmp / "series"
        mod.DATASET_ROOT = tmp / "datasets"
        mod.REPORTS_ROOT = tmp / "reports"
    build.MIN_DAYS_PER_SYMBOL = 20  # fixture spans ~31 days
    build.GEN1_REFERENCE_DATASET = tmp / "gen1_ref.csv"


def _write_gen1_reference(tmp: Path, dataset_dir: Path) -> None:
    df = common.read_csv_exact(dataset_dir / "dense.csv", nrows=1)
    df.to_csv(tmp / "gen1_ref.csv", index=False)


def test_gates_reject_bad_inputs() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _patch_roots(tmp)
        for kind, needle in (("duplicate", "duplicated"), ("bad_high", "high <"), ("gap", "gap gate")):
            d = make_snapshot(tmp / kind, broken=kind)
            result = build.build_series(d, f"v-{kind}")
            assert result["excluded"], kind
            assert any(needle in e["reason"] for e in result["excluded"]), (kind, result["excluded"])


def test_series_dataset_determinism_and_replay() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        _patch_roots(tmp)
        snap_dir = make_snapshot(tmp)
        series = build.build_series(snap_dir, "v1")
        assert series["status"] == "OK", series
        # first build bootstraps the fixture's reference contract (enforce_contract=False)
        build.FROZEN_MAX_ROWS_PER_COMBO = 400
        try:
            a = build.build_dataset("v1", "build_a", enforce_contract=False)
            assert a["decision"] == "DATASET_BUILT", a
            dir_a = Path(a["directory"])
            _write_gen1_reference(tmp, dir_a)
            build.GEN1_REFERENCE_DATASET = tmp / "gen1_ref.csv"
            b = build.build_dataset("v1", "build_b")
            assert b["decision"] == "DATASET_BUILT"
            check = build.double_check(dir_a, Path(b["directory"]))
            assert check["passed"] is True, check
            # G7 replay: independent snapshot over the same market must reproduce rows
            replay_snap = make_snapshot(tmp / "replay")
            build.REPLAY_WARMUP_DAYS = 10
            replay = build.replay_check(dir_a, replay_snap)
            assert replay["rows_expected"] > 0
            assert replay["passed"] is True, replay
            # contract breach detection: reference with a renamed feature
            ref = common.read_csv_exact(tmp / "gen1_ref.csv", nrows=1)
            ref = ref.rename(columns={[c for c in ref.columns if c.startswith("feature.")][0]: "feature.renamed_x"})
            ref.to_csv(tmp / "gen1_ref.csv", index=False)
            broken = build.build_dataset("v1", "build_c")
            assert broken["decision"] == "D3_FEATURE_CONTRACT_BROKEN"
        finally:
            build.FROZEN_MAX_ROWS_PER_COMBO = 5000
            build.REPLAY_WARMUP_DAYS = 14


if __name__ == "__main__":
    test_gates_reject_bad_inputs()
    test_series_dataset_determinism_and_replay()
    print("test_gen2_d3_build: OK")
