#!/usr/bin/env python3
"""GEN2-RV2 trainer: TRRM V2 candidates + QMAE V2 (spec: GEN2_RV2_SPEC.md v1.0).

Research-only. Trains from scratch on the approved D3 canonical dataset,
development period only (< lockbox semi-blind start), four expanding folds with
120-minute embargo, identical protocol for every candidate. Freezes the winner
(refit on full development, calibrated) with manifest + hashes. Never touches
the semi-blind or forward windows (that single registered query belongs to the
acceptance auditor).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import (  # noqa: E402
    GEN2_ROOT,
    LOCKBOX_MANIFEST_PATH,
    read_csv_exact,
    sha256_file,
    sha256_text,
    utc_stamp,
    validate_gen2_path,
    write_manifest,
)

RV2_ROOT = GEN2_ROOT / "rv2"
DEFAULT_DATASET = GEN2_ROOT / "d3" / "datasets" / "d3-v1-build_a" / "dense.csv"
TARGET = "target.tail_risk_roe_030"
QMAE_TARGET = "future_eval.future_mae_roe_proxy"
EMBARGO_MINUTES = 120
FOLD_SPECS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90)]
CAL_FRACTION = 0.5  # within each validation block: first half calibration, second half evaluation
SEED = 42
QUANTILES = (0.5, 0.9)
COVERAGE_BAND = (0.87, 0.93)


class MedianImputer:
    def fit(self, x: pd.DataFrame) -> "MedianImputer":
        self.medians = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.replace([np.inf, -np.inf], np.nan).fillna(self.medians).fillna(0.0)


def load_development(dataset_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    lockbox = json.loads(LOCKBOX_MANIFEST_PATH.read_text())
    dev_end = pd.Timestamp(lockbox["historical_development_period"]["end"])
    df = read_csv_exact(dataset_csv)
    df["_ts"] = pd.to_datetime(df["id.timestamp"], errors="coerce")
    df = df.sort_values(["_ts", "id.symbol", "id.horizon"]).reset_index(drop=True)
    features = [c for c in df.columns if c.startswith("feature.")]
    for horizon in (6, 12, 24):
        df[f"horizon_{horizon}"] = (pd.to_numeric(df["id.horizon"], errors="coerce") == horizon).astype(float)
    feature_cols = features + ["horizon_6", "horizon_12", "horizon_24"]
    dev = df[df["_ts"] <= dev_end].reset_index(drop=True)
    semi = df[df["_ts"] > dev_end].reset_index(drop=True)
    return dev, semi, feature_cols, lockbox


def make_folds(dev: pd.DataFrame) -> list[dict[str, Any]]:
    ts = dev["_ts"]
    n = len(dev)
    embargo = pd.Timedelta(minutes=EMBARGO_MINUTES)
    folds = []
    for i, (tr_end, val_end) in enumerate(FOLD_SPECS, start=1):
        train_idx = np.arange(0, int(n * tr_end))
        val_raw = np.arange(int(n * tr_end), int(n * val_end))
        train_max = ts.iloc[train_idx].max()
        val_idx = val_raw[(ts.iloc[val_raw] > train_max + embargo).to_numpy()]
        half = ts.iloc[val_idx].quantile(CAL_FRACTION)
        cal_idx = val_idx[(ts.iloc[val_idx] <= half).to_numpy()]
        eval_raw = val_idx[(ts.iloc[val_idx] > half).to_numpy()]
        cal_max = ts.iloc[cal_idx].max() if len(cal_idx) else train_max
        eval_idx = eval_raw[(ts.iloc[eval_raw] > cal_max + embargo).to_numpy()]
        folds.append({
            "name": f"fold_{i}",
            "train_idx": train_idx, "cal_idx": cal_idx, "eval_idx": eval_idx,
            "periods": {"train_end": str(train_max), "cal_end": str(cal_max), "eval_start": str(ts.iloc[eval_idx].min()) if len(eval_idx) else None, "eval_end": str(ts.iloc[eval_idx].max()) if len(eval_idx) else None},
        })
    return folds


def trrm_candidates() -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "random_forest": (True, lambda: RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, class_weight="balanced_subsample", random_state=SEED, n_jobs=4)),
        "hist_gradient_boosting": (True, lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=None, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)),
        "gradient_boosting": (True, lambda: GradientBoostingClassifier(n_estimators=200, learning_rate=0.06, max_depth=3, subsample=0.7, random_state=SEED)),
        "extra_trees": (True, lambda: ExtraTreesClassifier(n_estimators=400, max_depth=14, min_samples_leaf=20, class_weight="balanced_subsample", random_state=SEED, n_jobs=4)),
        "logistic_reference": (False, lambda: LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
    }


def score_of(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    from scipy.special import expit  # pragma: no cover

    return expit(model.decision_function(x))


def fold_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    prevalence = float(y.mean())
    order = np.argsort(-score)
    k = max(1, int(round(0.30 * len(y))))
    top = order[:k]
    capture = float(y[top].sum() / y.sum()) if y.sum() else 0.0
    return {
        "rows": int(len(y)),
        "prevalence": prevalence,
        "pr_auc": float(average_precision_score(y, score)) if len(set(y)) > 1 else None,
        "roc_auc": float(roc_auc_score(y, score)) if len(set(y)) > 1 else None,
        "capture_at_30pct_budget": capture,
        "lift_at_30pct": capture / 0.30 if capture else 0.0,
        "brier_raw": float(brier_score_loss(y, np.clip(score, 0, 1))),
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.sum():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def evaluate_trrm(dev: pd.DataFrame, folds: list[dict[str, Any]], feature_cols: list[str]) -> dict[str, Any]:
    y_all = dev[TARGET].astype(int).to_numpy()
    x_raw = dev[feature_cols].apply(pd.to_numeric, errors="coerce")
    results: dict[str, Any] = {}
    for name, (eligible, factory) in trrm_candidates().items():
        fold_rows = []
        for fold in folds:
            t0 = time.time()
            imputer = MedianImputer().fit(x_raw.iloc[fold["train_idx"]])
            xt = imputer.transform(x_raw)
            model = factory()
            model.fit(xt.iloc[fold["train_idx"]], y_all[fold["train_idx"]])
            score = score_of(model, xt.iloc[fold["eval_idx"]])
            m = fold_metrics(y_all[fold["eval_idx"]], score)
            m["train_seconds"] = round(time.time() - t0, 1)
            fold_rows.append({"fold": fold["name"], "metrics": m})
        prs = [r["metrics"]["pr_auc"] for r in fold_rows if r["metrics"]["pr_auc"] is not None]
        prevs = [r["metrics"]["prevalence"] for r in fold_rows]
        agg = {
            "mean_pr_auc": float(np.mean(prs)),
            "worst_pr_auc": float(np.min(prs)),
            "std_pr_auc": float(np.std(prs)),
            "stability_ratio": float(np.min(prs) / np.mean(prs)) if np.mean(prs) else 0.0,
            "mean_capture_30": float(np.mean([r["metrics"]["capture_at_30pct_budget"] for r in fold_rows])),
            "mean_prevalence": float(np.mean(prevs)),
            "h1_all_folds_lift_ge_2x": bool(all(r["metrics"]["pr_auc"] >= 2 * r["metrics"]["prevalence"] for r in fold_rows)),
            "h2_stability_ge_080": bool(np.min(prs) >= 0.80 * np.mean(prs)),
            "total_train_seconds": float(sum(r["metrics"]["train_seconds"] for r in fold_rows)),
            "eligible_winner": eligible,
        }
        results[name] = {"folds": fold_rows, "aggregate": agg}
    return results


def select_winner(results: dict[str, Any]) -> str | None:
    eligible = {k: v for k, v in results.items() if v["aggregate"]["eligible_winner"] and v["aggregate"]["h1_all_folds_lift_ge_2x"]}
    if not eligible:
        return None
    ranked = sorted(
        eligible.items(),
        key=lambda kv: (
            -kv[1]["aggregate"]["worst_pr_auc"],
            -kv[1]["aggregate"]["mean_pr_auc"],
            kv[1]["aggregate"]["std_pr_auc"],
            kv[1]["aggregate"]["total_train_seconds"],
        ),
    )
    return ranked[0][0]


def calibrate_winner(dev: pd.DataFrame, folds: list[dict[str, Any]], feature_cols: list[str], winner: str) -> dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y_all = dev[TARGET].astype(int).to_numpy()
    x_raw = dev[feature_cols].apply(pd.to_numeric, errors="coerce")
    rows = []
    for fold in folds:
        imputer = MedianImputer().fit(x_raw.iloc[fold["train_idx"]])
        xt = imputer.transform(x_raw)
        model = trrm_candidates()[winner][1]()
        model.fit(xt.iloc[fold["train_idx"]], y_all[fold["train_idx"]])
        s_cal = score_of(model, xt.iloc[fold["cal_idx"]])
        s_eval = score_of(model, xt.iloc[fold["eval_idx"]])
        y_cal = y_all[fold["cal_idx"]]
        y_eval = y_all[fold["eval_idx"]]
        entry = {"fold": fold["name"], "raw": {"ece": ece(y_eval, s_eval), "brier": float(np.mean((y_eval - s_eval) ** 2))}}
        if len(set(y_cal)) > 1:
            platt = LogisticRegression(solver="liblinear").fit(s_cal.reshape(-1, 1), y_cal)
            p = platt.predict_proba(s_eval.reshape(-1, 1))[:, 1]
            entry["sigmoid"] = {"ece": ece(y_eval, p), "brier": float(np.mean((y_eval - p) ** 2))}
            iso = IsotonicRegression(out_of_bounds="clip").fit(s_cal, y_cal)
            q = np.asarray(iso.predict(s_eval), dtype=float)
            entry["isotonic"] = {"ece": ece(y_eval, q), "brier": float(np.mean((y_eval - q) ** 2))}
        rows.append(entry)
    means = {}
    for kind in ("raw", "sigmoid", "isotonic"):
        vals = [r[kind]["ece"] for r in rows if kind in r]
        if vals:
            means[kind] = float(np.mean(vals))
    best = min(means, key=means.get) if means else "raw"
    return {"folds": rows, "mean_ece": means, "selected_calibrator": best}


def evaluate_qmae(dev: pd.DataFrame, folds: list[dict[str, Any]], feature_cols: list[str]) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_pinball_loss

    y_all = pd.to_numeric(dev[QMAE_TARGET], errors="coerce").fillna(0.0).to_numpy()
    x_raw = dev[feature_cols].apply(pd.to_numeric, errors="coerce")
    h_all = pd.to_numeric(dev["id.horizon"], errors="coerce").astype(int).to_numpy()
    sym_all = dev["id.symbol"].astype(str).to_numpy()
    fold_rows = []
    for fold in folds:
        imputer = MedianImputer().fit(x_raw.iloc[fold["train_idx"]])
        xt = imputer.transform(x_raw)
        entry: dict[str, Any] = {"fold": fold["name"]}
        y_tr = y_all[fold["train_idx"]]
        y_cal = y_all[fold["cal_idx"]]
        y_ev = y_all[fold["eval_idx"]]
        for q in QUANTILES:
            model = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
            model.fit(xt.iloc[fold["train_idx"]], y_tr)
            pred_cal = model.predict(xt.iloc[fold["cal_idx"]])
            pred_ev = model.predict(xt.iloc[fold["eval_idx"]])
            baseline = float(np.quantile(y_tr, q))
            key = f"q{int(q*100)}"
            entry[key] = {
                "pinball": float(mean_pinball_loss(y_ev, pred_ev, alpha=q)),
                "pinball_baseline_unconditional": float(mean_pinball_loss(y_ev, np.full(len(y_ev), baseline), alpha=q)),
                "raw_coverage": float((y_ev <= pred_ev).mean()),
            }
            if q == 0.9:
                # split-conformal (CQR one-sided): shift q90 by the calibration residual quantile
                residuals = y_cal - pred_cal
                n_cal = len(residuals)
                k = int(np.ceil((n_cal + 1) * 0.9)) / n_cal if n_cal else 1.0
                adjust = float(np.quantile(residuals, min(1.0, k))) if n_cal else 0.0
                conf = pred_ev + adjust
                entry[key]["conformal_adjustment"] = adjust
                entry[key]["conformal_coverage"] = float((y_ev <= conf).mean())
                cov_h = {int(h): float((y_ev[h_all[fold["eval_idx"]] == h] <= conf[h_all[fold["eval_idx"]] == h]).mean()) for h in (6, 12, 24)}
                entry[key]["conformal_coverage_by_horizon"] = cov_h
                syms = pd.Series(y_ev <= conf).groupby(pd.Series(sym_all[fold["eval_idx"]])).mean()
                entry[key]["conformal_coverage_by_symbol_min"] = float(syms.min())
                entry[key]["conformal_coverage_by_symbol_max"] = float(syms.max())
        fold_rows.append(entry)
    covs = [r["q90"]["conformal_coverage"] for r in fold_rows]
    pin_ok = all(r["q90"]["pinball"] <= 0.9 * r["q90"]["pinball_baseline_unconditional"] for r in fold_rows)
    agg = {
        "q90_conformal_coverage_by_fold": covs,
        "coverage_band": list(COVERAGE_BAND),
        "h3_coverage_all_folds_in_band": bool(all(COVERAGE_BAND[0] <= c <= COVERAGE_BAND[1] for c in covs)),
        "h3_pinball_beats_baseline": bool(pin_ok),
        "h3_pass": bool(all(COVERAGE_BAND[0] <= c <= COVERAGE_BAND[1] for c in covs) and pin_ok),
    }
    return {"folds": fold_rows, "aggregate": agg}


def freeze_candidate(dev: pd.DataFrame, feature_cols: list[str], winner: str, calibrator_kind: str, out_dir: Path) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    ts = dev["_ts"]
    embargo = pd.Timedelta(minutes=EMBARGO_MINUTES)
    split_ts = ts.quantile(0.90)
    train_idx = np.flatnonzero((ts <= split_ts).to_numpy())
    cal_idx = np.flatnonzero((ts > ts.iloc[train_idx].max() + embargo).to_numpy())
    y_all = dev[TARGET].astype(int).to_numpy()
    x_raw = dev[feature_cols].apply(pd.to_numeric, errors="coerce")
    imputer = MedianImputer().fit(x_raw.iloc[train_idx])
    xt = imputer.transform(x_raw)
    model = trrm_candidates()[winner][1]()
    model.fit(xt.iloc[train_idx], y_all[train_idx])
    s_cal = score_of(model, xt.iloc[cal_idx])
    y_cal = y_all[cal_idx]
    calibrator: Any = None
    if calibrator_kind == "sigmoid" and len(set(y_cal)) > 1:
        calibrator = LogisticRegression(solver="liblinear").fit(s_cal.reshape(-1, 1), y_cal)
    elif calibrator_kind == "isotonic" and len(set(y_cal)) > 1:
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(s_cal, y_cal)
    qmae_models = {}
    qmae_conf_adjust = 0.0
    y_mae = pd.to_numeric(dev[QMAE_TARGET], errors="coerce").fillna(0.0).to_numpy()
    for q in QUANTILES:
        qm = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
        qm.fit(xt.iloc[train_idx], y_mae[train_idx])
        qmae_models[f"q{int(q*100)}"] = qm
        if q == 0.9 and len(cal_idx):
            res = y_mae[cal_idx] - qm.predict(xt.iloc[cal_idx])
            k = int(np.ceil((len(res) + 1) * 0.9)) / len(res)
            qmae_conf_adjust = float(np.quantile(res, min(1.0, k)))
    bundle = {
        "trrm_model": model, "imputer": imputer, "calibrator": calibrator, "calibrator_kind": calibrator_kind,
        "qmae_models": qmae_models, "qmae_q90_conformal_adjustment": qmae_conf_adjust,
        "features": feature_cols, "target": TARGET, "qmae_target": QMAE_TARGET, "winner": winner, "seed": SEED,
    }
    pkl = out_dir / "rv2_candidate.pkl"
    with pkl.open("wb") as f:
        pickle.dump(bundle, f)
    return {
        "winner": winner, "calibrator": calibrator_kind, "pickle": str(pkl), "pickle_sha256": sha256_file(pkl),
        "feature_hash": sha256_text("\n".join(feature_cols)), "train_rows": int(len(train_idx)), "cal_rows": int(len(cal_idx)),
        "qmae_q90_conformal_adjustment": qmae_conf_adjust,
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    stamp = utc_stamp()
    out_dir = RV2_ROOT / stamp
    validate_gen2_path(out_dir)
    out_dir.mkdir(parents=True)
    dataset_csv = Path(args.dataset_csv)
    dev, semi, feature_cols, lockbox = load_development(dataset_csv)
    folds = make_folds(dev)
    trrm = evaluate_trrm(dev, folds, feature_cols)
    winner = select_winner(trrm)
    calibration = calibrate_winner(dev, folds, feature_cols, winner) if winner else {}
    qmae = evaluate_qmae(dev, folds, feature_cols)
    frozen = freeze_candidate(dev, feature_cols, winner, calibration.get("selected_calibrator", "raw"), out_dir) if winner else {}
    payload = {
        "schema": "gen2_rv2_training_v1",
        "artifact_id": stamp,
        "spec": "GEN2_RV2_SPEC.md rv2-spec-v1.0",
        "dataset_csv": str(dataset_csv),
        "dataset_sha256": sha256_file(dataset_csv),
        "target": TARGET,
        "development_rows": int(len(dev)),
        "semi_blind_rows_not_touched_here": int(len(semi)),
        "feature_count": len(feature_cols),
        "feature_hash": sha256_text("\n".join(feature_cols)),
        "folds": [{k: f[k] for k in ("name", "periods")} | {"train_rows": len(f["train_idx"]), "cal_rows": len(f["cal_idx"]), "eval_rows": len(f["eval_idx"])} for f in folds],
        "trrm_results": trrm,
        "trrm_winner": winner,
        "calibration": calibration,
        "qmae_results": qmae,
        "frozen_candidate": frozen,
        "gen1_artifacts_reused": False,
        "safety": {"no_live": True, "no_semi_blind_used_for_selection": True, "no_forward_touched": True, "research_only": True},
    }
    (out_dir / "training_report.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    write_manifest(out_dir, {"schema": "gen2_rv2_artifact_v1", "artifact_id": stamp, "dataset_sha256": payload["dataset_sha256"], "trrm_winner": winner, "files": {p.name: sha256_file(p) for p in out_dir.iterdir() if p.is_file()}}, "rv2_manifest")
    print(json.dumps({
        "artifact": str(out_dir), "winner": winner,
        "winner_aggregate": trrm.get(winner, {}).get("aggregate") if winner else None,
        "qmae_h3": qmae["aggregate"], "calibrator": calibration.get("selected_calibrator"),
    }, indent=2, default=json_default))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEN2-RV2 TRRM V2 + QMAE V2 trainer")
    p.add_argument("--dataset-csv", default=str(DEFAULT_DATASET))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_training(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
