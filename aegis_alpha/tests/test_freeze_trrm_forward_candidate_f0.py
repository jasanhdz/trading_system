#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.trrm_forward_common_f0 as common
from aegis_alpha.tools.calibrate_trrm_operating_point_e21 import build_internal_folds, policy_predictions
from aegis_alpha.tools.freeze_trrm_forward_candidate_f0 import parse_args, run_freeze
from aegis_alpha.tools.train_trrm_honest_e2 import TARGET


def fixture(n: int = 3600) -> pd.DataFrame:
    start = pd.Timestamp("2025-07-09T00:00:00Z")
    rows = []
    for i in range(n):
        h = [6, 12, 24][i % 3]
        s = ((i % 70) / 70.0) + (0.10 if h == 12 else 0.18 if h == 24 else 0.0)
        y = int(s > 0.76)
        row = {
            "id.symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"][i % 3],
            "id.timestamp": str(start + pd.Timedelta(hours=2 * i)),
            "id.timeframe": "5m",
            "id.horizon": h,
            TARGET: y,
        }
        for j in range(108):
            row[f"feature.f{j:03d}"] = ((i + j) % 43) / 43.0
        row["feature.atr_proxy_24"] = min(0.99, s)
        row["feature.ema_slope_6"] = 0.01
        row["feature.ema_slope_12"] = 0.02
        row["feature.ema_slope_24"] = 0.03
        row["feature.ema_slope_48"] = 0.04
        rows.append(row)
    return pd.DataFrame(rows)


class DummyImputer:
    def transform(self, x):
        return x.fillna(0.0)


class DummyModel:
    def predict_proba(self, x):
        s = x["feature.atr_proxy_24"].to_numpy(float)
        s = np.clip(s, 0, 1)
        return np.vstack([1 - s, s]).T


def make_artifacts(root: Path) -> dict[str, Path]:
    df = fixture()
    dense = root / "dense.csv"
    df.to_csv(dense, index=False)
    features = [c for c in df.columns if c.startswith("feature.")] + ["horizon_6", "horizon_12", "horizon_24"]
    model_dir = root / "model"
    model_dir.mkdir()
    pipe = {
        "model": DummyModel(),
        "imputer": DummyImputer(),
        "scaler": None,
        "calibrator": None,
        "features": features,
        "target": TARGET,
        "threshold": 0.39101951472531293,
    }
    with (model_dir / "selected_pipeline.pkl").open("wb") as f:
        pickle.dump(pipe, f)
    (model_dir / "metadata.json").write_text(json.dumps({"target": {"name": TARGET}}))
    score = df["feature.atr_proxy_24"].to_numpy(float)
    fold = build_internal_folds(df, 120, 500, 20)[-1]
    s, thr, _ = policy_predictions(df, fold, score, 0.30, "ROLLING_GLOBAL_QUANTILE_PAST_ONLY", 500, 20, 30)
    internal = df.iloc[fold.evaluation_idx][["id.symbol", "id.timestamp", "id.timeframe", "id.horizon", TARGET]].copy()
    internal["risk_score"] = s
    internal["policy_threshold"] = thr
    internal["reject"] = (s >= thr).astype(int)
    internal_csv = root / "internal.csv"
    internal.to_csv(internal_csv, index=False)
    e2 = {
        "selected_candidate": {"model": "random_forest"},
    }
    e21 = {
        "target": TARGET,
        "paths": {"dense_csv": str(dense)},
        "selected_policy": {"method": "ROLLING_GLOBAL_QUANTILE_PAST_ONLY", "budget": 0.30, "rolling_window_days": 30},
    }
    fable = {
        "status": "METRICS_SCOPE_AMBIGUOUS",
        "findings": [{"detail": "Any F0 freeze must name exactly one engine."}],
    }
    e2_json = root / "e2.json"
    e21_json = root / "e21.json"
    fable_json = root / "fable.json"
    e2_json.write_text(json.dumps(e2))
    e21_json.write_text(json.dumps(e21))
    fable_json.write_text(json.dumps(fable))
    return {
        "dense": dense,
        "model_dir": model_dir,
        "policy_dir": root / "policy",
        "internal": internal_csv,
        "e2": e2_json,
        "e21": e21_json,
        "fable": fable_json,
        "feature_hash": common.feature_hash(features),
    }


def test_freeze_manifest_replay_seed_and_identity_documented() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        art = make_artifacts(root)
        common.EXPECTED_FEATURE_HASH = art["feature_hash"]
        args = parse_args(
            [
                "--e2-report-json",
                str(art["e2"]),
                "--e21-report-json",
                str(art["e21"]),
                "--fable-audit-json",
                str(art["fable"]),
                "--model-dir",
                str(art["model_dir"]),
                "--policy-dir",
                str(art["policy_dir"]),
                "--internal-predictions",
                str(art["internal"]),
                "--output-root",
                str(root / "out"),
                "--candidate-id",
                "candidate_test",
                "--freeze-time",
                "2026-07-10T20:00:00Z",
                "--feature-hash",
                art["feature_hash"],
            ]
        )
        payload = run_freeze(args)
        cdir = Path(payload["candidate_dir"])
        manifest = json.loads((cdir / "candidate_manifest.json").read_text())
        assert manifest["target"] == TARGET
        assert manifest["policy_method"] == "ROLLING_GLOBAL_QUANTILE_PAST_ONLY"
        assert manifest["budget"] == 0.30
        assert manifest["rolling_window_days"] == 30
        assert manifest["engine_name"] == "E21_PER_ROW_CANONICAL"
        assert manifest["primary_horizon"] == 12
        assert manifest["diagnostic_horizons"] == [6, 24]
        assert manifest["enforcement_enabled"] is False
        assert manifest["labels_enabled"] is False
        assert manifest["engine_replay"]["status"] == "OK"
        assert manifest["engine_replay"]["threshold_match"] is True
        assert manifest["engine_replay"]["decision_match"] is True
        assert "jasanhdzb@gmail.com" in manifest["git_identity"]["author_ident"]
        seed_text = (cdir / "history_seed.jsonl").read_text()
        assert TARGET not in seed_text
        assert "future" not in seed_text.lower()
        assert (cdir / "candidate_manifest.sha256").exists()
        assert (cdir / "schema.json").exists()
        assert "FORWARD_OUTCOMES_NOT_EVALUATED" in (cdir / "README.md").read_text()


def test_manifest_reuse_and_conflict() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        art = make_artifacts(root)
        common.EXPECTED_FEATURE_HASH = art["feature_hash"]
        base = [
            "--e2-report-json",
            str(art["e2"]),
            "--e21-report-json",
            str(art["e21"]),
            "--fable-audit-json",
            str(art["fable"]),
            "--model-dir",
            str(art["model_dir"]),
            "--policy-dir",
            str(art["policy_dir"]),
            "--internal-predictions",
            str(art["internal"]),
            "--output-root",
            str(root / "out"),
            "--candidate-id",
            "candidate_test",
            "--freeze-time",
            "2026-07-10T20:00:00Z",
            "--feature-hash",
            art["feature_hash"],
        ]
        run_freeze(parse_args(base))
        run_freeze(parse_args(base))
        changed = base.copy()
        changed[changed.index("--freeze-time") + 1] = "2026-07-10T20:01:00Z"
        try:
            run_freeze(parse_args(changed))
            raise AssertionError("expected conflict")
        except ValueError as exc:
            assert "FREEZE_MANIFEST_CONFLICT" in str(exc)


def test_feature_hash_mismatch_stops() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        art = make_artifacts(root)
        common.EXPECTED_FEATURE_HASH = art["feature_hash"]
        args = parse_args(
            [
                "--e2-report-json",
                str(art["e2"]),
                "--e21-report-json",
                str(art["e21"]),
                "--fable-audit-json",
                str(art["fable"]),
                "--model-dir",
                str(art["model_dir"]),
                "--policy-dir",
                str(art["policy_dir"]),
                "--internal-predictions",
                str(art["internal"]),
                "--output-root",
                str(root / "out"),
                "--feature-hash",
                "bad",
            ]
        )
        try:
            run_freeze(args)
            raise AssertionError("expected integrity error")
        except ValueError as exc:
            assert "ARTIFACT_INTEGRITY_ERROR" in str(exc)


if __name__ == "__main__":
    test_freeze_manifest_replay_seed_and_identity_documented()
    test_manifest_reuse_and_conflict()
    test_feature_hash_mismatch_stops()
    print("test_freeze_trrm_forward_candidate_f0: OK")
