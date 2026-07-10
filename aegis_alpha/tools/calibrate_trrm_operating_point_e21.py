#!/usr/bin/env python3
"""FASE-E2.1 temporal calibration and stable TRRM operating point.

Research-only. Uses the frozen FASE-E2 model and target, selects threshold
policy using only pre-lockbox temporal folds, and reports the already-opened
lockbox only as diagnostic evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default
from aegis_alpha.tools.train_trrm_honest_e2 import (
    LOCKBOX_START,
    MedianImputer,
    PRIMARY_BUDGET,
    StandardScalerLite,
    TARGET,
    add_horizon_one_hot,
    budget_metrics,
    ece_score,
    eligible_features,
    safe_num,
    score_estimator,
    standard_metrics,
    target_values,
    threshold_for_budget,
)

DEFAULT_DENSE = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_20260710T051035Z.csv")
DEFAULT_STRIDED = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_strided_20260710T051035Z.csv")
DEFAULT_E2_JSON = Path("/home/jasan/Develop/aegis_phase_e2_trrm_honest_20260710T173714Z.json")
DEFAULT_E2_MODEL_DIR = Path("/home/jasan/Develop/aegis_research_models/trrm_e2/20260710T173714Z")
DEFAULT_VAL_PRED = Path("/home/jasan/Develop/aegis_phase_e2_validation_predictions_20260710T173714Z.csv")
DEFAULT_LOCKBOX_PRED = Path("/home/jasan/Develop/aegis_phase_e2_lockbox_predictions_strided_20260710T173714Z.csv")
DEFAULT_OUT = Path("/home/jasan/Develop")
DEFAULT_MODEL_OUT = Path("/home/jasan/Develop/aegis_research_models/trrm_e21")
RESEARCH_ROOT = Path("/home/jasan/Develop").resolve()
ORIGINAL_THRESHOLD = 0.39101951472531293


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FASE-E2.1 TRRM operating-point calibration")
    p.add_argument("--dense-csv", default=str(DEFAULT_DENSE))
    p.add_argument("--strided-csv", default=str(DEFAULT_STRIDED))
    p.add_argument("--e2-report-json", default=str(DEFAULT_E2_JSON))
    p.add_argument("--e2-model-dir", default=str(DEFAULT_E2_MODEL_DIR))
    p.add_argument("--validation-predictions", default=str(DEFAULT_VAL_PRED))
    p.add_argument("--opened-lockbox-predictions", default=str(DEFAULT_LOCKBOX_PRED))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--model-output-dir", default=str(DEFAULT_MODEL_OUT))
    p.add_argument("--target", default=TARGET)
    p.add_argument("--budgets", nargs="+", type=float, default=[0.20, 0.30, 0.40])
    p.add_argument("--rolling-window-days", nargs="+", type=int, default=[30, 60, 90])
    p.add_argument("--minimum-calibration-rows", type=int, default=500)
    p.add_argument("--minimum-positive-count", type=int, default=20)
    p.add_argument("--embargo-minutes", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write-artifacts", default="true")
    p.add_argument("--write-report", default="true")
    return p.parse_args(argv)


def validate_research_path(path: Path) -> None:
    resolved = path.resolve()
    if REPO in resolved.parents and "active" in resolved.parts:
        raise ValueError(f"refusing active/live path: {resolved}")
    if Path("/tmp") in resolved.parents or resolved == Path("/tmp"):
        return
    if RESEARCH_ROOT not in resolved.parents and resolved != RESEARCH_ROOT:
        raise ValueError(f"refusing non-research output path: {resolved}")


def feature_hash(features: list[str]) -> str:
    return hashlib.sha256("\n".join(features).encode()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def load_pipeline(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "selected_pipeline.pkl"
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("rb") as f:
        return pickle.load(f)


def artifact_integrity(args: argparse.Namespace, dense: pd.DataFrame, pipeline: dict[str, Any], e2: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(args.e2_model_dir)
    metadata = load_json(model_dir / "metadata.json")
    pipe_features = list(pipeline["features"])
    report_features = list(e2["selected_candidate"]["feature_columns"])
    dense_features, _, _ = eligible_features(dense)
    checks = {
        "model_dir_exists": model_dir.exists(),
        "metadata_exists": (model_dir / "metadata.json").exists(),
        "pipeline_exists": (model_dir / "selected_pipeline.pkl").exists(),
        "target_matches": pipeline.get("target") == TARGET == args.target,
        "threshold_matches": abs(float(pipeline.get("threshold")) - ORIGINAL_THRESHOLD) < 1e-9,
        "metadata_target_matches": metadata.get("target", {}).get("name") == TARGET,
        "report_target_matches": e2.get("target", {}).get("name") == TARGET,
        "feature_list_matches_report": pipe_features == report_features,
        "feature_list_subset_dense": set(pipe_features) <= set(dense_features + ["horizon_6", "horizon_12", "horizon_24"]),
        "lockbox_used_for_selection_false": e2.get("split_checks", {}).get("lockbox_used_for_selection") is False,
        "model_is_random_forest": e2.get("selected_candidate", {}).get("model") == "random_forest",
        "design_is_global_plus_horizon": e2.get("selected_candidate", {}).get("design") == "GLOBAL_CAUSAL_PLUS_HORIZON",
        "raw_close_absent": "feature.close" not in pipe_features,
        "symbol_absent": "id.symbol" not in pipe_features and "symbol" not in pipe_features,
        "no_target_label_future_features": not any(c.startswith(("target.", "label.", "future_eval.", "reference.", "id.")) for c in pipe_features),
    }
    status = "OK" if all(checks.values()) else "ARTIFACT_INTEGRITY_ERROR"
    return {
        "status": status,
        "checks": checks,
        "feature_count": len(pipe_features),
        "feature_hash": feature_hash(pipe_features),
        "original_threshold": float(pipeline.get("threshold")),
        "model_name": e2.get("selected_candidate", {}).get("name"),
    }


def score_frame(df: pd.DataFrame, pipeline: dict[str, Any]) -> np.ndarray:
    base_features = [c for c in pipeline["features"] if c.startswith("feature.")]
    x = df[base_features].apply(pd.to_numeric, errors="coerce")
    if any(c.startswith("horizon_") for c in pipeline["features"]):
        x = add_horizon_one_hot(df, x)
    x = x[pipeline["features"]]
    xi = pipeline["imputer"].transform(x)
    if pipeline.get("scaler") is not None:
        xi = pipeline["scaler"].transform(xi)
    raw = score_estimator(pipeline["model"], xi)
    cal = pipeline.get("calibrator")
    if cal is not None:
        return cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    return raw


@dataclass
class Fold:
    name: str
    train_idx: np.ndarray
    calibration_idx: np.ndarray
    evaluation_idx: np.ndarray
    periods: dict[str, str | None]


def build_internal_folds(df: pd.DataFrame, embargo_minutes: int, min_rows: int, min_pos: int) -> list[Fold]:
    ts = pd.to_datetime(df["id.timestamp"], errors="coerce", utc=True)
    y = target_values(df)
    pre = np.flatnonzero((ts < LOCKBOX_START).to_numpy())
    pre = pre[np.argsort(ts.iloc[pre].to_numpy())]
    n = len(pre)
    specs = [(0.40, 0.50, 0.60), (0.50, 0.60, 0.70), (0.60, 0.70, 0.80)]
    folds: list[Fold] = []
    embargo = pd.Timedelta(minutes=embargo_minutes)
    for i, (tr_end, cal_end, ev_end) in enumerate(specs, start=1):
        train = pre[: int(n * tr_end)]
        cal_raw = pre[int(n * tr_end): int(n * cal_end)]
        ev_raw = pre[int(n * cal_end): int(n * ev_end)]
        if not len(train) or not len(cal_raw) or not len(ev_raw):
            continue
        train_end_ts = ts.iloc[train].max()
        cal = cal_raw[(ts.iloc[cal_raw] > train_end_ts + embargo).to_numpy()]
        cal_end_ts = ts.iloc[cal].max() if len(cal) else ts.iloc[cal_raw].max()
        ev = ev_raw[(ts.iloc[ev_raw] > cal_end_ts + embargo).to_numpy()]
        if len(cal) < min_rows or int(y.iloc[cal].sum()) < min_pos or len(ev) < min_rows or int(y.iloc[ev].sum()) < min_pos:
            # Keep the fold for transparency if it has some samples; selection can mark insufficiency.
            pass
        periods = {
            "train_start": str(ts.iloc[train].min()) if len(train) else None,
            "train_end": str(ts.iloc[train].max()) if len(train) else None,
            "calibration_start": str(ts.iloc[cal].min()) if len(cal) else None,
            "calibration_end": str(ts.iloc[cal].max()) if len(cal) else None,
            "evaluation_start": str(ts.iloc[ev].min()) if len(ev) else None,
            "evaluation_end": str(ts.iloc[ev].max()) if len(ev) else None,
        }
        folds.append(Fold(f"fold_{i}", train, cal, ev, periods))
    return folds


def fit_sigmoid(score: np.ndarray, y: np.ndarray) -> Any | None:
    if len(set(y.astype(int))) < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(solver="liblinear")
        m.fit(score.reshape(-1, 1), y.astype(int))
        return m
    except Exception:
        return None


def fit_isotonic(score: np.ndarray, y: np.ndarray) -> Any | None:
    if len(score) < 500 or int(y.sum()) < 20:
        return None
    try:
        from sklearn.isotonic import IsotonicRegression

        m = IsotonicRegression(out_of_bounds="clip")
        m.fit(score, y.astype(int))
        return m
    except Exception:
        return None


def apply_calibrator(kind: str, calibrator: Any | None, score: np.ndarray) -> np.ndarray:
    if kind == "raw" or calibrator is None:
        return score
    if kind == "sigmoid":
        return calibrator.predict_proba(score.reshape(-1, 1))[:, 1]
    return np.asarray(calibrator.predict(score), dtype=float)


def static_thresholds_by_horizon(df: pd.DataFrame, cal_idx: np.ndarray, score: np.ndarray, budget: float, min_rows: int, min_pos: int) -> dict[int, float]:
    y = target_values(df)
    h = pd.to_numeric(df["id.horizon"], errors="coerce").fillna(-1).astype(int)
    global_thr = threshold_for_budget(score[cal_idx], budget)
    out: dict[int, float] = {}
    for horizon in (6, 12, 24):
        local = cal_idx[(h.iloc[cal_idx] == horizon).to_numpy()]
        if len(local) >= min_rows and int(y.iloc[local].sum()) >= min_pos:
            out[horizon] = threshold_for_budget(score[local], budget)
        else:
            out[horizon] = global_thr
    return out


def rolling_thresholds(df: pd.DataFrame, cal_idx: np.ndarray, eval_idx: np.ndarray, score: np.ndarray, budget: float, days: int, expanding: bool = False) -> np.ndarray:
    ts = pd.to_datetime(df["id.timestamp"], errors="coerce", utc=True)
    history_idx = np.concatenate([cal_idx, eval_idx])
    order = history_idx[np.argsort(ts.iloc[history_idx].to_numpy())]
    hist_ts = ts.iloc[order].reset_index(drop=True)
    hist_score = pd.Series(score[order]).reset_index(drop=True)
    eval_set = set(int(i) for i in eval_idx)
    thresholds = []
    window = pd.Timedelta(days=days)
    for idx in eval_idx:
        current = ts.iloc[idx]
        before = hist_ts < current
        if not expanding:
            before = before & (hist_ts >= current - window)
        vals = hist_score[before].to_numpy(float)
        if len(vals) < 50:
            vals = score[cal_idx]
        thresholds.append(threshold_for_budget(vals, budget))
    return np.asarray(thresholds, dtype=float)


def policy_predictions(
    df: pd.DataFrame,
    fold: Fold,
    score: np.ndarray,
    budget: float,
    method: str,
    min_rows: int,
    min_pos: int,
    window_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cal = fold.calibration_idx
    ev = fold.evaluation_idx
    y_cal = target_values(df).iloc[cal].to_numpy(int)
    cal_score = score[cal]
    eval_score = score[ev]
    meta: dict[str, Any] = {}
    if method == "STATIC_ABSOLUTE_THRESHOLD":
        return eval_score, np.full(len(ev), ORIGINAL_THRESHOLD), meta
    if method == "STATIC_GLOBAL_QUANTILE":
        return eval_score, np.full(len(ev), threshold_for_budget(cal_score, budget)), meta
    if method == "EXPANDING_GLOBAL_QUANTILE_PAST_ONLY":
        return eval_score, rolling_thresholds(df, cal, ev, score, budget, days=3650, expanding=True), meta
    if method == "ROLLING_GLOBAL_QUANTILE_PAST_ONLY":
        assert window_days is not None
        meta["rolling_window_days"] = window_days
        return eval_score, rolling_thresholds(df, cal, ev, score, budget, days=window_days, expanding=False), meta
    if method == "SIGMOID_CALIBRATED_STATIC_THRESHOLD":
        cal_model = fit_sigmoid(cal_score, y_cal)
        if cal_model is None:
            meta["skipped"] = "SKIPPED_INSUFFICIENT_CALIBRATION_DATA"
            return eval_score, np.full(len(ev), math.inf), meta
        cal_s = apply_calibrator("sigmoid", cal_model, cal_score)
        ev_s = apply_calibrator("sigmoid", cal_model, eval_score)
        return ev_s, np.full(len(ev), threshold_for_budget(cal_s, budget)), meta
    if method == "ISOTONIC_CALIBRATED_STATIC_THRESHOLD":
        cal_model = fit_isotonic(cal_score, y_cal)
        if cal_model is None:
            meta["skipped"] = "SKIPPED_INSUFFICIENT_CALIBRATION_DATA"
            return eval_score, np.full(len(ev), math.inf), meta
        cal_s = apply_calibrator("isotonic", cal_model, cal_score)
        ev_s = apply_calibrator("isotonic", cal_model, eval_score)
        return ev_s, np.full(len(ev), threshold_for_budget(cal_s, budget)), meta
    if method == "HORIZON_CONDITIONED_QUANTILE":
        thresholds = static_thresholds_by_horizon(df, cal, score, budget, min_rows, min_pos)
        h = pd.to_numeric(df["id.horizon"], errors="coerce").fillna(-1).astype(int)
        meta["thresholds_by_horizon"] = thresholds
        return eval_score, np.asarray([thresholds.get(int(v), threshold_for_budget(cal_score, budget)) for v in h.iloc[ev]], dtype=float), meta
    raise ValueError(method)


def metrics_for_policy(df: pd.DataFrame, idx: np.ndarray, y: np.ndarray, score: np.ndarray, thresholds: np.ndarray, budget: float) -> dict[str, Any]:
    reject = score >= thresholds
    fixed_score = score
    # For AUC/calibration, use score ranking; for operating metrics, use policy-specific thresholds.
    bm = budget_metrics(y, fixed_score, budget, threshold=math.inf)
    rejected = int(reject.sum())
    positives = int(y.sum())
    captured = int(((y == 1) & reject).sum())
    retained = y[~reject]
    prevalence = float(y.mean()) if len(y) else 0.0
    precision_rej = captured / rejected if rejected else 0.0
    residual = float(retained.mean()) if len(retained) else 0.0
    reduction = 1.0 - residual / prevalence if prevalence else 0.0
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        pr_auc = float(average_precision_score(y, fixed_score)) if len(set(y)) > 1 else None
        roc_auc = float(roc_auc_score(y, fixed_score)) if len(set(y)) > 1 else None
        brier = float(brier_score_loss(y, np.clip(fixed_score, 0, 1)))
    except Exception:
        pr_auc = roc_auc = brier = None
    ece, bins = ece_score(y, np.clip(fixed_score, 0, 1))
    return {
        "rows": int(len(y)),
        "positives": positives,
        "requested_budget": budget,
        "realized_rejection_rate": rejected / len(y) if len(y) else 0.0,
        "absolute_budget_error": abs((rejected / len(y) if len(y) else 0.0) - budget),
        "relative_budget_error": abs((rejected / len(y) if len(y) else 0.0) - budget) / budget if budget else 0.0,
        "prevalence": prevalence,
        "tail_capture_rate": captured / positives if positives else 0.0,
        "precision_among_rejected": precision_rej,
        "lift": precision_rej / prevalence if prevalence else 0.0,
        "residual_tail_rate": residual,
        "tail_risk_reduction": reduction,
        "false_negatives": int(positives - captured),
        "number_needed_to_reject": rejected / captured if captured else None,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "ece": ece,
        "calibration_bins": bins,
    }


def evaluate_method(df: pd.DataFrame, score: np.ndarray, folds: list[Fold], method: str, budget: float, min_rows: int, min_pos: int, window_days: int | None = None) -> dict[str, Any]:
    rows = []
    y_all = target_values(df)
    for fold in folds:
        y = y_all.iloc[fold.evaluation_idx].to_numpy(int)
        if len(fold.calibration_idx) < min_rows or int(y_all.iloc[fold.calibration_idx].sum()) < min_pos:
            rows.append({"fold": fold.name, "skipped": True, "reason": "insufficient_calibration_data", "periods": fold.periods})
            continue
        s, thr, meta = policy_predictions(df, fold, score, budget, method, min_rows, min_pos, window_days)
        if meta.get("skipped"):
            rows.append({"fold": fold.name, "skipped": True, "reason": meta["skipped"], "periods": fold.periods})
            continue
        rows.append({"fold": fold.name, "periods": fold.periods, "metrics": metrics_for_policy(df, fold.evaluation_idx, y, s, thr, budget), "meta": meta})
    good = [r["metrics"] for r in rows if not r.get("skipped")]
    if not good:
        agg = {"usable": False}
    else:
        agg = {
            "usable": True,
            "fold_count": len(good),
            "mean_realized_rejection": float(np.mean([m["realized_rejection_rate"] for m in good])),
            "std_realized_rejection": float(np.std([m["realized_rejection_rate"] for m in good])),
            "mean_absolute_budget_error": float(np.mean([m["absolute_budget_error"] for m in good])),
            "max_absolute_budget_error": float(np.max([m["absolute_budget_error"] for m in good])),
            "mean_tail_capture": float(np.mean([m["tail_capture_rate"] for m in good])),
            "minimum_tail_capture": float(np.min([m["tail_capture_rate"] for m in good])),
            "mean_residual_tail_rate": float(np.mean([m["residual_tail_rate"] for m in good])),
            "mean_tail_risk_reduction": float(np.mean([m["tail_risk_reduction"] for m in good])),
            "mean_lift": float(np.mean([m["lift"] for m in good])),
            "worst_fold_lift": float(np.min([m["lift"] for m in good])),
            "max_rejection": float(np.max([m["realized_rejection_rate"] for m in good])),
        }
    return {"method": method, "budget": budget, "rolling_window_days": window_days, "folds": rows, "aggregate": agg}


def policy_passes(result: dict[str, Any]) -> bool:
    a = result["aggregate"]
    if not a.get("usable"):
        return False
    return (
        a["mean_absolute_budget_error"] <= 0.05
        and a["max_absolute_budget_error"] <= 0.10
        and a["mean_tail_capture"] >= 0.55
        and a["minimum_tail_capture"] >= 0.40
        and a["mean_lift"] >= 1.75
        and a["worst_fold_lift"] >= 1.25
        and a["mean_tail_risk_reduction"] >= 0.35
        and not (result["budget"] == 0.30 and a["max_rejection"] > 0.50)
    )


def select_policy(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    pref = {
        "STATIC_GLOBAL_QUANTILE": 0,
        "EXPANDING_GLOBAL_QUANTILE_PAST_ONLY": 1,
        "ROLLING_GLOBAL_QUANTILE_PAST_ONLY": 2,
        "HORIZON_CONDITIONED_QUANTILE": 3,
        "SIGMOID_CALIBRATED_STATIC_THRESHOLD": 4,
        "ISOTONIC_CALIBRATED_STATIC_THRESHOLD": 5,
        "STATIC_ABSOLUTE_THRESHOLD": 6,
    }
    passing = [r for r in results if policy_passes(r)]
    if passing:
        return sorted(passing, key=lambda r: (abs(r["budget"] - PRIMARY_BUDGET), pref.get(r["method"], 99), r["aggregate"]["mean_absolute_budget_error"], -r["aggregate"]["mean_tail_capture"]))[0]
    usable = [r for r in results if r["aggregate"].get("usable")]
    return sorted(usable, key=lambda r: (abs(r["budget"] - PRIMARY_BUDGET), r["aggregate"]["mean_absolute_budget_error"], -r["aggregate"]["mean_tail_capture"]))[0] if usable else None


def horizon_balance(df: pd.DataFrame, idx: np.ndarray, y: np.ndarray, score: np.ndarray, thresholds: np.ndarray, budget: float) -> list[dict[str, Any]]:
    h = pd.to_numeric(df.iloc[idx]["id.horizon"], errors="coerce").fillna(-1).astype(int).to_numpy()
    rows = []
    for horizon in (6, 12, 24):
        mask = h == horizon
        if not mask.any():
            continue
        rows.append({"horizon": horizon, **metrics_for_policy(df, idx[mask], y[mask], score[mask], thresholds[mask], budget)})
    return rows


def monthly_drift(df: pd.DataFrame, score: np.ndarray, selected_threshold: float) -> list[dict[str, Any]]:
    work = df.copy()
    work["_ts"] = pd.to_datetime(work["id.timestamp"], errors="coerce", utc=True)
    work["_month"] = work["_ts"].dt.to_period("M").astype(str)
    work["_score"] = score
    work["_y"] = target_values(work)
    rows = []
    for month, g in work.groupby("_month"):
        scores = g["_score"].to_numpy(float)
        y = g["_y"].to_numpy(int)
        reject = scores >= selected_threshold
        rows.append({
            "month": month,
            "rows": int(len(g)),
            "prevalence": float(y.mean()) if len(y) else 0.0,
            "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
            "score_median": float(np.median(scores)) if len(scores) else 0.0,
            "score_p70": float(np.quantile(scores, 0.70)) if len(scores) else 0.0,
            "score_p80": float(np.quantile(scores, 0.80)) if len(scores) else 0.0,
            "score_p90": float(np.quantile(scores, 0.90)) if len(scores) else 0.0,
            "static_rejection_rate": float(reject.mean()) if len(reject) else 0.0,
            "tail_capture_static": int(((y == 1) & reject).sum()) / int(y.sum()) if int(y.sum()) else 0.0,
            "residual_tail_static": float(y[~reject].mean()) if (~reject).any() else 0.0,
        })
    return rows


def opened_lockbox_diagnostic(df: pd.DataFrame, score: np.ndarray, selected: dict[str, Any], budget: float, min_rows: int, min_pos: int) -> dict[str, Any]:
    ts = pd.to_datetime(df["id.timestamp"], errors="coerce", utc=True)
    idx = np.flatnonzero((ts >= LOCKBOX_START).to_numpy())
    y = target_values(df).iloc[idx].to_numpy(int)
    static_thr = np.full(len(idx), ORIGINAL_THRESHOLD)
    static = metrics_for_policy(df, idx, y, score[idx], static_thr, budget)
    # Diagnostic-only application: use policy mechanics with all pre-lockbox rows as calibration.
    cal = np.flatnonzero((ts < LOCKBOX_START).to_numpy())
    fold = Fold("opened_lockbox_diagnostic", np.array([], dtype=int), cal, idx, {})
    s, thr, meta = policy_predictions(df, fold, score, budget, selected["method"], min_rows, min_pos, selected.get("rolling_window_days"))
    selected_metrics = metrics_for_policy(df, idx, y, s, thr, budget)
    return {
        "status": "OPENED_LOCKBOX_DIAGNOSTIC_ONLY",
        "not_independent": True,
        "not_used_for_selection": True,
        "not_sufficient_for_promotion": True,
        "static_absolute_threshold": static,
        "selected_policy": selected_metrics,
        "selected_policy_meta": meta,
        "horizon_balance": horizon_balance(df, idx, y, s, thr, budget),
    }


def decision(selected: dict[str, Any] | None, opened: dict[str, Any] | None) -> tuple[str, str, str]:
    if not selected:
        return "RESEARCH_NOT_READY", "no usable policy selected from pre-lockbox folds", "E2.2 targeted calibration diagnostics."
    passes = policy_passes(selected)
    opened_bad = False
    if opened:
        m = opened["selected_policy"]
        opened_bad = m["realized_rejection_rate"] > 0.60 or m["lift"] < 1.0
    if passes and not opened_bad:
        return "TRRM_OPERATING_POLICY_READY_FOR_FORWARD_RESEARCH", "pre-lockbox folds meet operating policy criteria; opened lockbox does not contradict severely", "FASE-F0 - freeze research candidate and begin forward score collection, no enforcement."
    if selected["method"] == "STATIC_ABSOLUTE_THRESHOLD":
        return "STATIC_THRESHOLD_UNSTABLE_ADAPTIVE_POLICY_NEEDED", "static threshold is the best available but does not satisfy stability criteria", "E2.2 adaptive threshold diagnostics."
    if selected["method"].startswith("HORIZON_CONDITIONED"):
        return "HORIZON_CONDITIONING_REQUIRED", "horizon-conditioned policy is preferred but needs new confirmation", "E2.2 horizon-conditioned validation."
    if selected["aggregate"].get("mean_lift", 0) >= 1.25:
        return "TRRM_CALIBRATION_PROMISING", "policy has useful lift but misses one or more stability criteria", "E2.2 with the concrete failing stability metric."
    return "CALIBRATION_NOT_STABLE", "calibration/threshold policies vary too much across folds", "Do not proceed beyond research diagnostics."


def write_predictions(path: Path, df: pd.DataFrame, idx: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> None:
    cols = [c for c in ("id.symbol", "id.timestamp", "id.timeframe", "id.horizon", TARGET) if c in df.columns]
    out = df.iloc[idx][cols].copy()
    out["risk_score"] = score
    out["policy_threshold"] = thresholds
    out["reject"] = (score >= thresholds).astype(int)
    out.to_csv(path, index=False)


def render_markdown(payload: dict[str, Any]) -> str:
    sel = payload.get("selected_policy") or {}
    lines = [
        "# FASE-E2.1 TRRM Temporal Calibration and Operating Point",
        "",
        "## 1. Estado",
        f"- decision: {payload['decision']}",
        f"- reason: {payload['decision_reason']}",
        "- mode: research-only",
        "",
        "## 2. Input FASE-E2",
        f"- model: {payload['artifact_integrity']['model_name']}",
        f"- target: {payload['target']}",
        f"- original_threshold: {payload['original_threshold']}",
        "",
        "## 3. Artifact integrity",
        f"- status: {payload['artifact_integrity']['status']}",
        f"- feature_hash: {payload['artifact_integrity']['feature_hash']}",
        "",
        "## 4. Lockbox status",
        "- already opened in FASE-E2",
        "- diagnostic only",
        "- not used for selection",
        "",
        "## 5. Internal temporal folds",
        json.dumps(payload["folds"], indent=2, default=json_default),
        "",
        "## 6. Calibration methods",
        "- raw, sigmoid/Platt, isotonic when enough calibration positives exist",
        "",
        "## 7. Operating-point methods",
        ", ".join(sorted(set(r["method"] for r in payload["method_results"]))),
        "",
        "## 8. Budget stability",
        json.dumps(payload["selected_policy"], indent=2, default=json_default),
        "",
        "## 9. Fold robustness",
        json.dumps((sel.get("folds") or []), indent=2, default=json_default),
        "",
        "## 10. Horizon balance",
        json.dumps(payload["horizon_balance"], indent=2, default=json_default),
        "",
        "## 11. Prevalence and score drift",
        json.dumps(payload["monthly_drift"][:20], indent=2, default=json_default),
        "",
        "## 12. Probability calibration",
        json.dumps(payload["calibration_summary"], indent=2, default=json_default),
        "",
        "## 13. Selected policy",
        f"- method: {sel.get('method')}",
        f"- budget: {sel.get('budget')}",
        f"- rolling_window_days: {sel.get('rolling_window_days')}",
        "- fallback: global calibration threshold when horizon/rolling sample is insufficient",
        "",
        "## 14. Comparison with static threshold",
        json.dumps(payload["static_reference"], indent=2, default=json_default),
        "",
        "## 15. Opened-lockbox diagnostic",
        "- OPENED_LOCKBOX_DIAGNOSTIC_ONLY: not independent, not used for selection, not sufficient for promotion.",
        json.dumps(payload["opened_lockbox_diagnostic"], indent=2, default=json_default),
        "",
        "## 16. Limitations",
        "\n".join(f"- {x}" for x in payload["limitations"]),
        "",
        "## 17. New forward validation plan",
        "\n".join(f"- {x}" for x in payload["forward_validation_plan"]),
        "",
        "## 18. Decision",
        f"- {payload['decision']}: {payload['decision_reason']}",
        "",
        "## 19. Recommended next phase",
        f"- {payload['recommended_next_phase']}",
        "",
        "## 20. Safety confirmations",
        "- No live changes. No active_manifest. No YAML live. No PM2 restart. No orders. No .env. No TS changes. No push. No promotion. No shadow. Opened lockbox not used for selection. No future/label features.",
        "",
    ]
    return "\n".join(lines)


def run_e21(args: argparse.Namespace) -> dict[str, Any]:
    if args.target != TARGET:
        raise ValueError("ARTIFACT_INTEGRITY_ERROR: target is frozen to target.tail_risk_roe_030")
    out_dir = Path(args.output_dir)
    model_out = Path(args.model_output_dir)
    validate_research_path(out_dir)
    validate_research_path(model_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    dense = pd.read_csv(args.dense_csv)
    strided = pd.read_csv(args.strided_csv)
    e2 = load_json(Path(args.e2_report_json))
    pipeline = load_pipeline(Path(args.e2_model_dir))
    integ = artifact_integrity(args, dense, pipeline, e2)
    if integ["status"] != "OK":
        raise ValueError("ARTIFACT_INTEGRITY_ERROR")
    dense_score = score_frame(dense, pipeline)
    strided_score = score_frame(strided, pipeline)
    folds = build_internal_folds(dense, args.embargo_minutes, args.minimum_calibration_rows, args.minimum_positive_count)
    method_specs: list[tuple[str, int | None]] = [
        ("STATIC_ABSOLUTE_THRESHOLD", None),
        ("STATIC_GLOBAL_QUANTILE", None),
        ("EXPANDING_GLOBAL_QUANTILE_PAST_ONLY", None),
        ("SIGMOID_CALIBRATED_STATIC_THRESHOLD", None),
        ("ISOTONIC_CALIBRATED_STATIC_THRESHOLD", None),
        ("HORIZON_CONDITIONED_QUANTILE", None),
    ]
    for days in args.rolling_window_days:
        method_specs.append(("ROLLING_GLOBAL_QUANTILE_PAST_ONLY", int(days)))
    results = []
    for budget in args.budgets:
        for method, days in method_specs:
            results.append(evaluate_method(dense, dense_score, folds, method, float(budget), args.minimum_calibration_rows, args.minimum_positive_count, days))
    selected = select_policy(results)
    opened = opened_lockbox_diagnostic(strided, strided_score, selected, selected["budget"], args.minimum_calibration_rows, args.minimum_positive_count) if selected else None
    dec, reason, next_phase = decision(selected, opened)
    static_reference = [r for r in results if r["method"] == "STATIC_ABSOLUTE_THRESHOLD" and abs(r["budget"] - PRIMARY_BUDGET) < 1e-9]
    # Horizon balance on final pre-lockbox evaluation fold for the selected policy.
    horizon_rows = []
    if selected:
        last_fold = folds[-1]
        s, thr, _ = policy_predictions(dense, last_fold, dense_score, selected["budget"], selected["method"], args.minimum_calibration_rows, args.minimum_positive_count, selected.get("rolling_window_days"))
        y = target_values(dense).iloc[last_fold.evaluation_idx].to_numpy(int)
        horizon_rows = horizon_balance(dense, last_fold.evaluation_idx, y, s, thr, selected["budget"])
    calibration_summary = {
        "raw_static_ece_reference": (static_reference[0]["aggregate"] if static_reference else {}),
        "sigmoid_methods": [r for r in results if r["method"] == "SIGMOID_CALIBRATED_STATIC_THRESHOLD"],
        "isotonic_methods": [r for r in results if r["method"] == "ISOTONIC_CALIBRATED_STATIC_THRESHOLD"],
    }
    payload: dict[str, Any] = {
        "schema_version": "phase_e21_trrm_operating_policy_v1",
        "generated_at": stamp,
        "mode": "research-only",
        "target": TARGET,
        "original_threshold": ORIGINAL_THRESHOLD,
        "artifact_integrity": integ,
        "paths": {
            "dense_csv": str(args.dense_csv),
            "strided_csv": str(args.strided_csv),
            "e2_report_json": str(args.e2_report_json),
            "e2_model_dir": str(args.e2_model_dir),
            "validation_predictions": str(args.validation_predictions),
            "opened_lockbox_predictions": str(args.opened_lockbox_predictions),
        },
        "folds": [{"name": f.name, "periods": f.periods, "train_rows": len(f.train_idx), "calibration_rows": len(f.calibration_idx), "evaluation_rows": len(f.evaluation_idx), "calibration_positives": int(target_values(dense).iloc[f.calibration_idx].sum()), "evaluation_positives": int(target_values(dense).iloc[f.evaluation_idx].sum())} for f in folds],
        "method_results": results,
        "selected_policy": selected,
        "static_reference": static_reference[0] if static_reference else None,
        "horizon_balance": horizon_rows,
        "horizon_balance_scope": "last_pre_lockbox_fold_only",
        "aggregate_scope": "unweighted_mean_over_all_usable_folds",
        "policy_engine": "e21_per_row_rolling",
        "monthly_drift": monthly_drift(dense, dense_score, ORIGINAL_THRESHOLD),
        "calibration_summary": calibration_summary,
        "opened_lockbox_diagnostic": opened,
        "lockbox_status": {"already_opened": True, "diagnostic_only": True, "used_for_selection": False, "reselection_after_diagnostic": False},
        "limitations": [
            "Opened lockbox was already observed in FASE-E2 and is not independent evidence.",
            "The base model, features, target, and architecture remain frozen.",
            "No symbol-specific thresholds or models are selected.",
        ],
        "forward_validation_plan": [
            "Freeze research model and policy; collect forward scores with no enforcement.",
            "Do not inspect labels until at least 50 tail positives, preferably 100, across several months/regimes.",
            "Stop if realized rejection drifts outside agreed budget bands or any horizon collapses without sample-size justification.",
        ],
        "safety_confirmations": {
            "no_live_changes": True,
            "no_active_manifest": True,
            "no_yaml_live": True,
            "no_pm2_restart": True,
            "no_orders": True,
            "no_env": True,
            "no_ts_changes": True,
            "no_push": True,
            "no_model_promotion": True,
            "no_shadow": True,
            "opened_lockbox_not_used_for_selection": True,
            "no_future_or_label_features": True,
            "research_only_artifacts": True,
        },
    }
    payload["decision"] = dec
    payload["decision_reason"] = reason
    payload["recommended_next_phase"] = next_phase
    if parse_bool(args.write_artifacts):
        model_dir = model_out / stamp
        validate_research_path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "policy_metadata.json").write_text(json.dumps({
            "base_model_reference": str(args.e2_model_dir),
            "target": TARGET,
            "feature_hash": integ["feature_hash"],
            "selected_policy": selected,
            "decision": dec,
            "opened_lockbox_disclaimer": payload["lockbox_status"],
        }, indent=2, default=json_default))
        payload["research_policy_path"] = str(model_dir)
    if parse_bool(args.write_report):
        md = out_dir / f"aegis_phase_e21_trrm_calibration_{stamp}.md"
        js = out_dir / f"aegis_phase_e21_trrm_calibration_{stamp}.json"
        internal_csv = out_dir / f"aegis_phase_e21_internal_predictions_{stamp}.csv"
        lock_csv = out_dir / f"aegis_phase_e21_opened_lockbox_diagnostic_{stamp}.csv"
        if selected and folds:
            lf = folds[-1]
            s, thr, _ = policy_predictions(dense, lf, dense_score, selected["budget"], selected["method"], args.minimum_calibration_rows, args.minimum_positive_count, selected.get("rolling_window_days"))
            write_predictions(internal_csv, dense, lf.evaluation_idx, s, thr)
            lock_idx = np.flatnonzero((pd.to_datetime(strided["id.timestamp"], errors="coerce", utc=True) >= LOCKBOX_START).to_numpy())
            fold = Fold("opened_lockbox", np.array([], dtype=int), np.flatnonzero((pd.to_datetime(strided["id.timestamp"], errors="coerce", utc=True) < LOCKBOX_START).to_numpy()), lock_idx, {})
            ls, lthr, _ = policy_predictions(strided, fold, strided_score, selected["budget"], selected["method"], args.minimum_calibration_rows, args.minimum_positive_count, selected.get("rolling_window_days"))
            write_predictions(lock_csv, strided, lock_idx, ls, lthr)
        payload["artifacts"] = {"markdown": str(md), "json": str(js), "internal_predictions": str(internal_csv), "opened_lockbox_diagnostic": str(lock_csv)}
        js.write_text(json.dumps(payload, indent=2, default=json_default))
        md.write_text(render_markdown(payload))
    print(json.dumps({"decision": dec, "selected_policy": selected and {k: selected.get(k) for k in ("method", "budget", "rolling_window_days")}, "markdown": payload.get("artifacts", {}).get("markdown"), "json": payload.get("artifacts", {}).get("json"), "policy_path": payload.get("research_policy_path")}, indent=2, default=json_default))
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        run_e21(parse_args(argv))
        return 0
    except ValueError as exc:
        if "ARTIFACT_INTEGRITY_ERROR" in str(exc):
            print(json.dumps({"decision": "ARTIFACT_INTEGRITY_ERROR", "reason": str(exc)}))
            return 2
        if "LEAKAGE" in str(exc):
            print(json.dumps({"decision": "LEAKAGE_RISK_TOO_HIGH", "reason": str(exc)}))
            return 3
        print(json.dumps({"decision": "RESEARCH_NOT_READY", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
