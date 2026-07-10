#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from aegis_alpha.tools.audit_trrm_policy_metrics_fable_a import (
    TARGET,
    run_audit,
    shared_policy_metrics,
)


def test_shared_policy_metrics_basic() -> None:
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    reject = np.array([True, False, True, False, True, False, False, False, False, False])
    m = shared_policy_metrics(y, reject, budget=0.30)
    assert m["rows"] == 10
    assert m["positives"] == 3
    assert abs(m["realized_rejection_rate"] - 0.3) < 1e-12
    assert abs(m["tail_capture_rate"] - 2 / 3) < 1e-12
    assert abs(m["residual_tail_rate"] - 1 / 7) < 1e-12
    assert abs(m["absolute_budget_error"]) < 1e-12
    assert m["false_negatives"] == 1


def _lockbox_frame(n: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-05-01", periods=n, freq="h", tz="UTC")
    y = (rng.random(n) < 0.2).astype(int)
    score = np.clip(0.3 * y + rng.random(n) * 0.5, 0, 1)
    return pd.DataFrame(
        {
            "id.symbol": "ADAUSDT",
            "id.timestamp": ts.astype(str),
            "id.timeframe": "5m",
            "id.horizon": np.tile([6, 12, 24], n)[:n],
            TARGET: y,
            "risk_probability": score,
            "rejected_at_frozen_threshold": (score >= 0.391).astype(int),
        }
    )


def _policy_frame(base: pd.DataFrame, reject: np.ndarray) -> pd.DataFrame:
    out = base[["id.symbol", "id.timestamp", "id.timeframe", "id.horizon", TARGET]].copy()
    out["risk_score"] = base["risk_probability"]
    out["policy_threshold"] = 0.5
    out["reject"] = reject.astype(int)
    return out


def _fold_metrics(df: pd.DataFrame) -> dict:
    m = shared_policy_metrics(df[TARGET].to_numpy(int), df["reject"].astype(bool).to_numpy(), budget=0.30)
    return {k: m[k] for k in ("rows", "positives", "prevalence", "realized_rejection_rate", "tail_capture_rate", "residual_tail_rate", "lift")}


def _write_fixture(td: Path, *, mismatch_population: bool = False, with_scope_labels: bool = True) -> argparse.Namespace:
    lock = _lockbox_frame()
    reject = lock["risk_probability"].to_numpy() >= 0.45
    e21_lock = _policy_frame(lock, reject)
    e22_lock = _policy_frame(lock if not mismatch_population else lock.iloc[:-3], reject if not mismatch_population else reject[:-3])
    internal = []
    for i, fold in enumerate(("fold_1", "fold_2", "fold_3"), start=1):
        f = _policy_frame(lock, lock["risk_probability"].to_numpy() >= (0.40 + 0.02 * i))
        f["fold"] = fold
        f["policy"] = "HORIZON_ROLLING_QUANTILE"
        f["budget"] = 0.30
        internal.append(f)
    internal_df = pd.concat(internal, ignore_index=True)

    fold_rows = []
    for fold, g in internal_df.groupby("fold"):
        hrows = []
        for h, gh in g.groupby(g["id.horizon"].astype(int)):
            hm = shared_policy_metrics(gh[TARGET].to_numpy(int), gh["reject"].astype(bool).to_numpy(), budget=0.30)
            hrows.append({"horizon": int(h), "realized_rejection_rate": hm["realized_rejection_rate"], "tail_capture_rate": hm["tail_capture_rate"], "residual_tail_rate": hm["residual_tail_rate"], "lift": hm["lift"]})
        fold_rows.append({"fold": fold, "metrics": _fold_metrics(g), "horizon_metrics": hrows})
    rej_means = [f["metrics"]["realized_rejection_rate"] for f in fold_rows]
    cap_means = [f["metrics"]["tail_capture_rate"] for f in fold_rows]
    horizon_means = {}
    for h in (6, 12, 24):
        vals = [hr["realized_rejection_rate"] for f in fold_rows for hr in f["horizon_metrics"] if hr["horizon"] == h]
        horizon_means[h] = float(np.mean(vals))
    spread = max(horizon_means.values()) - min(horizon_means.values())

    e21_lock_metrics = shared_policy_metrics(e21_lock[TARGET].to_numpy(int), e21_lock["reject"].astype(bool).to_numpy())
    e21 = {
        "selected_policy": {"method": "ROLLING_GLOBAL_QUANTILE_PAST_ONLY", "budget": 0.30, "folds": fold_rows, "aggregate": {"mean_realized_rejection": float(np.mean(rej_means)), "mean_tail_capture": float(np.mean(cap_means)), "mean_residual_tail_rate": 0.0, "mean_lift": 1.0}},
        "opened_lockbox_diagnostic": {"selected_policy": e21_lock_metrics},
        "horizon_balance": fold_rows[-1]["horizon_metrics"],
    }
    if with_scope_labels:
        e21["horizon_balance_scope"] = "last_pre_lockbox_fold_only"
    e22_lock_metrics = shared_policy_metrics(e22_lock[TARGET].to_numpy(int), e22_lock["reject"].astype(bool).to_numpy())
    e22 = {
        "selected_policy": {"method": "HORIZON_ROLLING_QUANTILE", "budget": 0.30, "folds": fold_rows, "aggregate": {"mean_realized_rejection": float(np.mean(rej_means)), "horizon_rejection_spread": spread}},
        "selected_horizon_balance": fold_rows[-1]["horizon_metrics"],
        "opened_lockbox_diagnostic": {
            "selected_e22_policy": {"metrics": e22_lock_metrics},
            "global_e21_reference": {"metrics": e21_lock_metrics, "engine": "e22_day_batched"},
        },
    }
    if with_scope_labels:
        e22["selected_horizon_balance_scope"] = "last_pre_lockbox_fold_only"
    e2 = {"note": "fixture"}

    paths = {
        "e2_json": td / "e2.json",
        "e21_json": td / "e21.json",
        "e22_json": td / "e22.json",
    }
    paths["e2_json"].write_text(json.dumps(e2))
    paths["e21_json"].write_text(json.dumps(e21))
    paths["e22_json"].write_text(json.dumps(e22))
    lock.to_csv(td / "e2_lock.csv", index=False)
    e21_lock.to_csv(td / "e21_lock.csv", index=False)
    e22_lock.to_csv(td / "e22_lock.csv", index=False)
    internal_df.to_csv(td / "e22_internal.csv", index=False)
    return argparse.Namespace(
        e2_json=str(paths["e2_json"]),
        e2_lockbox_strided=str(td / "e2_lock.csv"),
        e21_json=str(paths["e21_json"]),
        e21_lockbox_csv=str(td / "e21_lock.csv"),
        e22_json=str(paths["e22_json"]),
        e22_internal_csv=str(td / "e22_internal.csv"),
        e22_lockbox_csv=str(td / "e22_lock.csv"),
        static_threshold=0.391,
        output_dir=str(td),
    )


def test_consistent_fixture_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        args = _write_fixture(td, with_scope_labels=True)
        result = run_audit(args)
        assert result["status"] == "METRICS_CONSISTENT", (result["status"], [k for k, v in result["checks"].items() if not v])
        assert result["checks"]["e22_aggregate_rejection_is_fold_mean"]
        assert result["checks"]["e22_horizon_spread_is_macro_over_folds"]
        assert Path(result["outputs"]["md"]).exists()


def test_missing_scope_labels_flagged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        args = _write_fixture(td, with_scope_labels=False)
        result = run_audit(args)
        assert result["status"] == "METRICS_SCOPE_AMBIGUOUS"
        kinds = {f["kind"] for f in result["findings"]}
        assert "METRICS_SCOPE_AMBIGUOUS" in kinds


def test_population_mismatch_detected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        args = _write_fixture(td, mismatch_population=True)
        result = run_audit(args)
        assert result["status"] == "DATASET_POPULATION_MISMATCH"


if __name__ == "__main__":
    test_shared_policy_metrics_basic()
    test_consistent_fixture_passes()
    test_missing_scope_labels_flagged()
    test_population_mismatch_detected()
    print("test_audit_trrm_policy_metrics_fable_a: OK")
