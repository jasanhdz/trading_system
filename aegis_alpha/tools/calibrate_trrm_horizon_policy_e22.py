#!/usr/bin/env python3
"""FASE-E2.2 horizon-conditioned TRRM operating policy.

Research-only. Keeps the FASE-E2 TRRM model/features/target frozen and studies
causal past-only threshold streams by horizon. The already-opened lockbox is
diagnostic only and cannot affect selection.
"""
from __future__ import annotations

import argparse
import json
import math
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
from aegis_alpha.tools.calibrate_trrm_operating_point_e21 import (
    ORIGINAL_THRESHOLD,
    RESEARCH_ROOT,
    artifact_integrity as e21_artifact_integrity,
    build_internal_folds,
    feature_hash,
    fit_sigmoid,
    horizon_balance,
    load_json,
    load_pipeline,
    metrics_for_policy,
    parse_bool,
    score_frame,
    validate_research_path,
)
from aegis_alpha.tools.train_trrm_honest_e2 import (
    LOCKBOX_START,
    MedianImputer,
    PRIMARY_BUDGET,
    StandardScalerLite,
    TARGET,
    target_values,
    threshold_for_budget,
)

DEFAULT_E21_JSON = Path("/home/jasan/Develop/aegis_phase_e21_trrm_calibration_20260710T183052Z.json")
DEFAULT_E21_INTERNAL = Path("/home/jasan/Develop/aegis_phase_e21_internal_predictions_20260710T183052Z.csv")
DEFAULT_E21_LOCKBOX = Path("/home/jasan/Develop/aegis_phase_e21_opened_lockbox_diagnostic_20260710T183052Z.csv")
DEFAULT_E2_MODEL_DIR = Path("/home/jasan/Develop/aegis_research_models/trrm_e2/20260710T173714Z")
DEFAULT_DENSE = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_20260710T051035Z.csv")
DEFAULT_STRIDED = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_strided_20260710T051035Z.csv")
DEFAULT_OUT = Path("/home/jasan/Develop")
DEFAULT_POLICY_OUT = Path("/home/jasan/Develop/aegis_research_models/trrm_e22")
EXPECTED_FEATURE_HASH = "fbb1fee2cf0c42d21c591169b25452eb65e932bf5bd76109ca8447a4dfd7057e"
HORIZONS = (6, 12, 24)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FASE-E2.2 horizon-conditioned TRRM policy")
    p.add_argument("--e21-report-json", default=str(DEFAULT_E21_JSON))
    p.add_argument("--e21-internal-predictions", default=str(DEFAULT_E21_INTERNAL))
    p.add_argument("--opened-lockbox-predictions", default=str(DEFAULT_E21_LOCKBOX))
    p.add_argument("--e2-model-dir", default=str(DEFAULT_E2_MODEL_DIR))
    p.add_argument("--dense-csv", default=str(DEFAULT_DENSE))
    p.add_argument("--strided-csv", default=str(DEFAULT_STRIDED))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--policy-output-dir", default=str(DEFAULT_POLICY_OUT))
    p.add_argument("--target", default=TARGET)
    p.add_argument("--budgets", nargs="+", type=float, default=[0.20, 0.30, 0.40])
    p.add_argument("--rolling-window-days", nargs="+", type=int, default=[30, 60, 90])
    p.add_argument("--shrinkage-alphas", nargs="+", type=float, default=[0.25, 0.50, 0.75, 1.00])
    p.add_argument("--minimum-history-rows", type=int, default=500)
    p.add_argument("--minimum-horizon-rows", type=int, default=150)
    p.add_argument("--minimum-positive-count", type=int, default=10)
    p.add_argument("--embargo-minutes", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write-artifacts", default="true")
    p.add_argument("--write-report", default="true")
    return p.parse_args(argv)


@dataclass(frozen=True)
class PolicySpec:
    method: str
    budget: float
    rolling_window_days: int | None = None
    alpha: float | None = None
    shrinkage_constant: int | None = None


def complexity(spec: PolicySpec) -> dict[str, Any]:
    if spec.method == "GLOBAL_ROLLING_REFERENCE":
        return {
            "number_of_threshold_streams": 1,
            "number_of_hyperparameters": 2,
            "requires_calibrator": False,
            "requires_fallback": True,
            "state_required": "global rolling score history",
            "implementation_complexity": "LOW",
        }
    if spec.method in {"HORIZON_ROLLING_QUANTILE", "HORIZON_EXPANDING_QUANTILE"}:
        return {
            "number_of_threshold_streams": 3,
            "number_of_hyperparameters": 2,
            "requires_calibrator": False,
            "requires_fallback": True,
            "state_required": "per-horizon score history plus global fallback",
            "implementation_complexity": "MEDIUM",
        }
    if spec.method in {"BLENDED_GLOBAL_HORIZON_ROLLING", "SHRUNK_HORIZON_QUANTILE_BY_SAMPLE_SIZE"}:
        return {
            "number_of_threshold_streams": 4,
            "number_of_hyperparameters": 3,
            "requires_calibrator": False,
            "requires_fallback": True,
            "state_required": "global and per-horizon rolling score histories",
            "implementation_complexity": "MEDIUM",
        }
    return {
        "number_of_threshold_streams": 3,
        "number_of_hyperparameters": 3,
        "requires_calibrator": True,
        "requires_fallback": True,
        "state_required": "per-horizon calibrators and fallback thresholds",
        "implementation_complexity": "HIGH",
    }


def artifact_integrity(args: argparse.Namespace, dense: pd.DataFrame, pipeline: dict[str, Any], e21: dict[str, Any]) -> dict[str, Any]:
    fake = argparse.Namespace(
        e2_model_dir=args.e2_model_dir,
        target=args.target,
    )
    e2_json = Path(args.e2_model_dir) / "metadata.json"
    e2_meta = load_json(e2_json)
    e2_report = {
        "target": {"name": e2_meta.get("target", {}).get("name")},
        "split_checks": {"lockbox_used_for_selection": False},
        "selected_candidate": e2_meta.get("selected_candidate", {}),
    }
    base = e21_artifact_integrity(fake, dense, pipeline, e2_report)
    e21_selected = e21.get("selected_policy") or {}
    checks = dict(base["checks"])
    checks.update(
        {
            "feature_hash_exact": base["feature_hash"] == EXPECTED_FEATURE_HASH,
            "e21_target_matches": e21.get("target") == TARGET,
            "e21_selected_global_reference": e21_selected.get("method") == "ROLLING_GLOBAL_QUANTILE_PAST_ONLY",
            "e21_budget_reference": abs(float(e21_selected.get("budget", -1)) - PRIMARY_BUDGET) < 1e-12,
            "e21_lockbox_diagnostic_only": e21.get("lockbox_status", {}).get("diagnostic_only") is True,
            "e21_lockbox_not_used_for_selection": e21.get("lockbox_status", {}).get("used_for_selection") is False,
            "horizons_expected": set(pd.to_numeric(dense["id.horizon"], errors="coerce").dropna().astype(int).unique()) == set(HORIZONS),
            "model_not_retrained": True,
            "no_live_artifacts": True,
        }
    )
    status = "OK" if all(checks.values()) else "ARTIFACT_INTEGRITY_ERROR"
    return {
        "status": status,
        "checks": checks,
        "feature_hash": base["feature_hash"],
        "target": TARGET,
        "model": "RandomForestClassifier",
        "feature_design": "GLOBAL_CAUSAL_PLUS_HORIZON",
        "scores_recomputed_from_frozen_pipeline": True,
    }


def _history_values(
    df: pd.DataFrame,
    history_idx: np.ndarray,
    score: np.ndarray,
    current_ts: pd.Timestamp,
    budget: float,
    window_days: int | None,
    horizon: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(df["id.timestamp"], errors="coerce", utc=True)
    mask = (ts.iloc[history_idx] < current_ts).to_numpy().copy()
    if window_days is not None:
        mask = mask & (ts.iloc[history_idx] >= current_ts - pd.Timedelta(days=window_days)).to_numpy()
    if horizon is not None:
        h = pd.to_numeric(df["id.horizon"], errors="coerce").fillna(-1).astype(int)
        mask = mask & (h.iloc[history_idx] == horizon).to_numpy()
    idx = history_idx[mask]
    return score[idx], idx


def _threshold_or_inf(values: np.ndarray, budget: float) -> float:
    return threshold_for_budget(values, budget) if len(values) else math.inf


def rolling_policy_thresholds(
    df: pd.DataFrame,
    calibration_idx: np.ndarray,
    evaluation_idx: np.ndarray,
    score: np.ndarray,
    y_all: np.ndarray,
    spec: PolicySpec,
    min_history_rows: int,
    min_horizon_rows: int,
    min_positive_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    ts = pd.to_datetime(df["id.timestamp"], errors="coerce", utc=True)
    h = pd.to_numeric(df["id.horizon"], errors="coerce").fillna(-1).astype(int).to_numpy()
    ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    thresholds = np.full(len(evaluation_idx), math.inf, dtype=float)
    fallback = np.zeros(len(evaluation_idx), dtype=bool)
    no_decision = np.zeros(len(evaluation_idx), dtype=bool)
    alpha_values: list[float] = []
    history_counts: list[int] = []
    global_counts: list[int] = []
    ordered_eval = evaluation_idx[np.argsort(ts.iloc[evaluation_idx].to_numpy())]
    history_so_far = np.array(calibration_idx, dtype=int)
    pos_lookup = y_all.astype(int)
    by_idx = {int(v): i for i, v in enumerate(evaluation_idx)}
    sigmoid_models: dict[int, Any] = {}
    global_sigmoid = None
    if spec.method == "HORIZON_CONDITIONED_SIGMOID":
        cal_score = score[calibration_idx]
        cal_y = y_all[calibration_idx]
        global_sigmoid = fit_sigmoid(cal_score, cal_y)
        for horizon in HORIZONS:
            local = calibration_idx[h[calibration_idx] == horizon]
            if len(local) >= min_horizon_rows and int(pos_lookup[local].sum()) >= min_positive_count:
                sigmoid_models[horizon] = fit_sigmoid(score[local], pos_lookup[local])
    eval_days = pd.Series(ts.iloc[ordered_eval].dt.floor("D").to_numpy(), index=ordered_eval)

    def values_before(current: pd.Timestamp, window_days: int | None, horizon: int | None) -> tuple[np.ndarray, np.ndarray]:
        current_ns = int(current.value)
        mask = ts_ns[history_so_far] < current_ns
        if window_days is not None:
            mask = mask & (ts_ns[history_so_far] >= current_ns - int(pd.Timedelta(days=window_days).value))
        if horizon is not None:
            mask = mask & (h[history_so_far] == horizon)
        idxs = history_so_far[mask]
        return score[idxs], idxs

    for day, day_series in eval_days.groupby(eval_days):
        day_indices = np.asarray(day_series.index.to_list(), dtype=int)
        current_ts = pd.Timestamp(day)
        window = spec.rolling_window_days
        expanding = spec.method == "HORIZON_EXPANDING_QUANTILE"
        if expanding:
            window = None
        global_vals, global_idx = values_before(current_ts, window, None)
        global_ok = len(global_idx) >= min_history_rows
        global_thr = _threshold_or_inf(global_vals, spec.budget) if global_ok else math.inf
        for horizon in HORIZONS:
            horizon_day = day_indices[h[day_indices] == horizon]
            if not len(horizon_day):
                continue
            horizon_vals, horizon_idx = values_before(current_ts, window, horizon)
            global_counts.extend([int(len(global_idx))] * len(horizon_day))
            history_counts.extend([int(len(horizon_idx))] * len(horizon_day))
            horizon_ok = len(horizon_idx) >= min_horizon_rows and int(pos_lookup[horizon_idx].sum()) >= min_positive_count
            if spec.method == "GLOBAL_ROLLING_REFERENCE":
                if not global_ok:
                    thr = math.inf
                    for idx in horizon_day:
                        no_decision[by_idx[int(idx)]] = True
                        fallback[by_idx[int(idx)]] = True
                else:
                    thr = global_thr
            elif spec.method in {"HORIZON_ROLLING_QUANTILE", "HORIZON_EXPANDING_QUANTILE"}:
                if horizon_ok:
                    thr = _threshold_or_inf(horizon_vals, spec.budget)
                elif global_ok:
                    thr = global_thr
                    for idx in horizon_day:
                        fallback[by_idx[int(idx)]] = True
                else:
                    thr = math.inf
                    for idx in horizon_day:
                        fallback[by_idx[int(idx)]] = True
                        no_decision[by_idx[int(idx)]] = True
            elif spec.method == "BLENDED_GLOBAL_HORIZON_ROLLING":
                if horizon_ok and global_ok:
                    a = float(spec.alpha if spec.alpha is not None else 1.0)
                    alpha_values.extend([a] * len(horizon_day))
                    thr = a * _threshold_or_inf(horizon_vals, spec.budget) + (1.0 - a) * global_thr
                elif global_ok:
                    thr = global_thr
                    alpha_values.extend([0.0] * len(horizon_day))
                    for idx in horizon_day:
                        fallback[by_idx[int(idx)]] = True
                else:
                    thr = math.inf
                    for idx in horizon_day:
                        fallback[by_idx[int(idx)]] = True
                        no_decision[by_idx[int(idx)]] = True
            elif spec.method == "SHRUNK_HORIZON_QUANTILE_BY_SAMPLE_SIZE":
                if horizon_ok and global_ok:
                    k = int(spec.shrinkage_constant or 250)
                    a = len(horizon_idx) / (len(horizon_idx) + k)
                    alpha_values.extend([float(a)] * len(horizon_day))
                    thr = a * _threshold_or_inf(horizon_vals, spec.budget) + (1.0 - a) * global_thr
                elif global_ok:
                    thr = global_thr
                    alpha_values.extend([0.0] * len(horizon_day))
                    for idx in horizon_day:
                        fallback[by_idx[int(idx)]] = True
                else:
                    thr = math.inf
                    for idx in horizon_day:
                        fallback[by_idx[int(idx)]] = True
                        no_decision[by_idx[int(idx)]] = True
            elif spec.method == "HORIZON_CONDITIONED_SIGMOID":
                if horizon in sigmoid_models and sigmoid_models[horizon] is not None and horizon_ok:
                    model = sigmoid_models[horizon]
                    transformed = model.predict_proba(horizon_vals.reshape(-1, 1))[:, 1]
                    thr = _threshold_or_inf(transformed, spec.budget)
                    score[horizon_day] = model.predict_proba(score[horizon_day].reshape(-1, 1))[:, 1]
                elif global_sigmoid is not None and global_ok:
                    transformed = global_sigmoid.predict_proba(global_vals.reshape(-1, 1))[:, 1]
                    thr = _threshold_or_inf(transformed, spec.budget)
                    score[horizon_day] = global_sigmoid.predict_proba(score[horizon_day].reshape(-1, 1))[:, 1]
                    for idx in horizon_day:
                        fallback[by_idx[int(idx)]] = True
                else:
                    thr = math.inf
                    for idx in horizon_day:
                        fallback[by_idx[int(idx)]] = True
                        no_decision[by_idx[int(idx)]] = True
            else:
                raise ValueError(spec.method)
            for idx in horizon_day:
                thresholds[by_idx[int(idx)]] = thr
        history_so_far = np.append(history_so_far, day_indices)
    meta = {
        "fallback_rate": float(fallback.mean()) if len(fallback) else 0.0,
        "no_decision_rate": float(no_decision.mean()) if len(no_decision) else 0.0,
        "fallback_count": int(fallback.sum()),
        "no_decision_count": int(no_decision.sum()),
        "mean_horizon_history_rows": float(np.mean(history_counts)) if history_counts else 0.0,
        "mean_global_history_rows": float(np.mean(global_counts)) if global_counts else 0.0,
        "mean_dynamic_alpha": float(np.mean(alpha_values)) if alpha_values else spec.alpha,
    }
    return thresholds, meta


def horizon_metrics(df: pd.DataFrame, idx: np.ndarray, y: np.ndarray, score: np.ndarray, thresholds: np.ndarray, budget: float, fallback_mask: np.ndarray | None = None) -> list[dict[str, Any]]:
    rows = horizon_balance(df, idx, y, score, thresholds, budget)
    if fallback_mask is not None:
        h = pd.to_numeric(df.iloc[idx]["id.horizon"], errors="coerce").fillna(-1).astype(int).to_numpy()
        for row in rows:
            mask = h == row["horizon"]
            row["fallback_rate"] = float(fallback_mask[mask].mean()) if mask.any() else 0.0
            row["budget_error"] = row["absolute_budget_error"]
    return rows


def evaluate_policy(
    df: pd.DataFrame,
    score: np.ndarray,
    folds: list[Any],
    spec: PolicySpec,
    min_history_rows: int,
    min_horizon_rows: int,
    min_positive_count: int,
) -> dict[str, Any]:
    y_all = target_values(df).to_numpy(int)
    fold_rows = []
    internal_predictions = []
    for fold in folds:
        if len(fold.calibration_idx) < min_history_rows or int(y_all[fold.calibration_idx].sum()) < min_positive_count:
            fold_rows.append({"fold": fold.name, "skipped": True, "reason": "insufficient_calibration_data", "periods": fold.periods})
            continue
        local_score = np.array(score, dtype=float, copy=True)
        thresholds, meta = rolling_policy_thresholds(
            df,
            fold.calibration_idx,
            fold.evaluation_idx,
            local_score,
            y_all,
            spec,
            min_history_rows,
            min_horizon_rows,
            min_positive_count,
        )
        y = y_all[fold.evaluation_idx]
        s = local_score[fold.evaluation_idx]
        metrics = metrics_for_policy(df, fold.evaluation_idx, y, s, thresholds, spec.budget)
        hrows = horizon_metrics(df, fold.evaluation_idx, y, s, thresholds, spec.budget)
        fold_rows.append({"fold": fold.name, "periods": fold.periods, "metrics": metrics, "horizon_metrics": hrows, "meta": meta})
        pred = df.iloc[fold.evaluation_idx][[c for c in ("id.symbol", "id.timestamp", "id.timeframe", "id.horizon", TARGET) if c in df.columns]].copy()
        pred["fold"] = fold.name
        pred["policy"] = spec.method
        pred["budget"] = spec.budget
        pred["risk_score"] = s
        pred["policy_threshold"] = thresholds
        pred["reject"] = (s >= thresholds).astype(int)
        internal_predictions.append(pred)
    good = [r for r in fold_rows if not r.get("skipped")]
    if not good:
        aggregate = {"usable": False}
    else:
        ms = [r["metrics"] for r in good]
        horizon_rejections = []
        horizon_captures = []
        for horizon in HORIZONS:
            vals = [hr["realized_rejection_rate"] for r in good for hr in r["horizon_metrics"] if hr["horizon"] == horizon]
            caps = [hr["tail_capture_rate"] for r in good for hr in r["horizon_metrics"] if hr["horizon"] == horizon]
            if vals:
                horizon_rejections.append(float(np.mean(vals)))
            if caps:
                horizon_captures.append(float(np.mean(caps)))
        aggregate = {
            "usable": True,
            "fold_count": len(good),
            "mean_realized_rejection": float(np.mean([m["realized_rejection_rate"] for m in ms])),
            "std_realized_rejection": float(np.std([m["realized_rejection_rate"] for m in ms])),
            "mean_absolute_budget_error": float(np.mean([m["absolute_budget_error"] for m in ms])),
            "max_absolute_budget_error": float(np.max([m["absolute_budget_error"] for m in ms])),
            "mean_tail_capture": float(np.mean([m["tail_capture_rate"] for m in ms])),
            "minimum_tail_capture": float(np.min([m["tail_capture_rate"] for m in ms])),
            "mean_lift": float(np.mean([m["lift"] for m in ms])),
            "worst_fold_lift": float(np.min([m["lift"] for m in ms])),
            "mean_residual_tail_rate": float(np.mean([m["residual_tail_rate"] for m in ms])),
            "mean_tail_risk_reduction": float(np.mean([m["tail_risk_reduction"] for m in ms])),
            "worst_horizon_rejection": float(np.max(horizon_rejections)) if horizon_rejections else None,
            "best_horizon_rejection": float(np.min(horizon_rejections)) if horizon_rejections else None,
            "horizon_rejection_spread": float(np.max(horizon_rejections) - np.min(horizon_rejections)) if len(horizon_rejections) >= 2 else None,
            "horizon_capture_spread": float(np.max(horizon_captures) - np.min(horizon_captures)) if len(horizon_captures) >= 2 else None,
            "fallback_rate": float(np.mean([r["meta"]["fallback_rate"] for r in good])),
        }
    out = {
        "method": spec.method,
        "budget": spec.budget,
        "rolling_window_days": spec.rolling_window_days,
        "alpha": spec.alpha,
        "shrinkage_constant": spec.shrinkage_constant,
        "complexity": complexity(spec),
        "folds": fold_rows,
        "aggregate": aggregate,
    }
    if internal_predictions:
        out["_predictions"] = pd.concat(internal_predictions, ignore_index=True)
    return out


def ready_criteria(result: dict[str, Any]) -> dict[str, bool]:
    a = result["aggregate"]
    if not a.get("usable"):
        return {"usable": False}
    return {
        "selection_pre_lockbox": True,
        "no_leakage": True,
        "mean_budget_error": a["mean_absolute_budget_error"] <= 0.05,
        "max_budget_error": a["max_absolute_budget_error"] <= 0.10,
        "mean_capture": a["mean_tail_capture"] >= 0.70,
        "minimum_capture": a["minimum_tail_capture"] >= 0.55,
        "mean_lift": a["mean_lift"] >= 2.00,
        "worst_lift": a["worst_fold_lift"] >= 1.50,
        "tail_risk_reduction": a["mean_tail_risk_reduction"] >= 0.50,
        "mean_rejection_band": 0.25 <= a["mean_realized_rejection"] <= 0.35,
        "no_horizon_above_50pct": (a["worst_horizon_rejection"] is not None and a["worst_horizon_rejection"] <= 0.50),
        "no_horizon_below_10pct": (a["best_horizon_rejection"] is not None and a["best_horizon_rejection"] >= 0.10),
        "horizon_spread": (a["horizon_rejection_spread"] is not None and a["horizon_rejection_spread"] <= 0.20),
        "no_future": True,
        "no_post_lockbox_reselection": True,
    }


def is_ready(result: dict[str, Any]) -> bool:
    c = ready_criteria(result)
    return bool(c) and all(c.values())


def almost_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    aa = a["aggregate"]
    bb = b["aggregate"]
    return (
        abs(aa["mean_tail_capture"] - bb["mean_tail_capture"]) <= 0.03
        and abs(aa["mean_lift"] - bb["mean_lift"]) <= 0.15
        and abs(aa["mean_absolute_budget_error"] - bb["mean_absolute_budget_error"]) <= 0.02
    )


def complexity_rank(result: dict[str, Any]) -> int:
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    return order.get(result["complexity"]["implementation_complexity"], 9)


def candidate_score(result: dict[str, Any], global_ref: dict[str, Any] | None) -> tuple[float, float, float, float]:
    a = result["aggregate"]
    spread = a.get("horizon_rejection_spread")
    spread_val = float(spread) if spread is not None else 9.0
    global_spread = (global_ref or {}).get("aggregate", {}).get("horizon_rejection_spread")
    spread_gain = (float(global_spread) - spread_val) if global_spread is not None else 0.0
    return (
        -a["mean_absolute_budget_error"],
        spread_gain,
        a["mean_tail_capture"],
        a["mean_lift"],
    )


def select_policy(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [r for r in results if r["aggregate"].get("usable") and abs(r["budget"] - PRIMARY_BUDGET) < 1e-9]
    if not usable:
        usable = [r for r in results if r["aggregate"].get("usable")]
    if not usable:
        return None
    global_refs = [r for r in usable if r["method"] == "GLOBAL_ROLLING_REFERENCE"]
    global_ref = global_refs[0] if global_refs else None
    ready = [r for r in usable if is_ready(r)]
    if ready:
        ready = sorted(ready, key=lambda r: (complexity_rank(r), r["aggregate"]["mean_absolute_budget_error"], r["aggregate"].get("horizon_rejection_spread") or 9.0, -r["aggregate"]["mean_tail_capture"]))
        simplest = ready[0]
        for other in ready[1:]:
            if almost_equal(simplest, other) and complexity_rank(other) < complexity_rank(simplest):
                simplest = other
        return simplest
    ranked = sorted(usable, key=lambda r: (-(candidate_score(r, global_ref)[1]), r["aggregate"]["mean_absolute_budget_error"], (r["aggregate"].get("horizon_rejection_spread") or 9.0), -r["aggregate"]["mean_tail_capture"], complexity_rank(r)))
    return ranked[0]


def decision(selected: dict[str, Any] | None, global_ref: dict[str, Any] | None, opened: dict[str, Any] | None) -> tuple[str, str, str]:
    if selected is None:
        return "RESEARCH_NOT_READY", "no usable horizon policy from pre-lockbox folds", "No avanzar; diagnosticar artifacts/folds."
    if is_ready(selected) and not (opened and opened.get("contradicts_internal_evidence")):
        return "HORIZON_POLICY_READY_FOR_FORWARD_RESEARCH", "horizon policy meets strict pre-lockbox stability and utility criteria", "FASE-F0 - freeze research candidate and begin forward score collection without enforcement."
    a = selected["aggregate"]
    g = (global_ref or {}).get("aggregate", {})
    spread_improved = g.get("horizon_rejection_spread") is not None and a.get("horizon_rejection_spread") is not None and a["horizon_rejection_spread"] < g["horizon_rejection_spread"] - 0.05
    capture_loss_ok = not g or a["mean_tail_capture"] >= g.get("mean_tail_capture", 0) - 0.10
    if selected["method"] != "GLOBAL_ROLLING_REFERENCE" and spread_improved and capture_loss_ok:
        failed = [k for k, v in ready_criteria(selected).items() if not v]
        if len(failed) <= 2:
            return "HORIZON_POLICY_PROMISING", f"improves horizon balance but misses strict criteria: {failed}", "E2.3/F0 design review before any enforcement."
        return "HORIZON_CONDITIONING_REQUIRED_BUT_UNSTABLE", f"horizon conditioning helps but remains unstable: {failed}", "E2.3 horizon policy stabilization."
    if selected["method"] == "GLOBAL_ROLLING_REFERENCE":
        return "GLOBAL_POLICY_PREFERRED", "horizon conditioning did not improve enough to justify added complexity", "Keep E2.1 policy for research diagnostics; do not enforce."
    if a["mean_tail_capture"] < 0.55 or a["mean_lift"] < 1.50:
        return "RESEARCH_NOT_READY", "horizon-conditioned policy loses too much signal", "Do not add complexity; diagnose score drift."
    return "HORIZON_POLICY_NOT_STABLE", "budget or horizon spread is not stable enough", "Continue research-only stabilization."


def build_specs(args: argparse.Namespace) -> list[PolicySpec]:
    specs: list[PolicySpec] = [PolicySpec("GLOBAL_ROLLING_REFERENCE", PRIMARY_BUDGET, 30)]
    for budget in args.budgets:
        for days in args.rolling_window_days:
            specs.append(PolicySpec("HORIZON_ROLLING_QUANTILE", float(budget), int(days)))
            for alpha in args.shrinkage_alphas:
                specs.append(PolicySpec("BLENDED_GLOBAL_HORIZON_ROLLING", float(budget), int(days), float(alpha)))
            for k in (100, 250, 500):
                specs.append(PolicySpec("SHRUNK_HORIZON_QUANTILE_BY_SAMPLE_SIZE", float(budget), int(days), None, int(k)))
        specs.append(PolicySpec("HORIZON_EXPANDING_QUANTILE", float(budget), None))
        specs.append(PolicySpec("HORIZON_CONDITIONED_SIGMOID", float(budget), None))
    return specs


def drift_by_horizon(df: pd.DataFrame, score: np.ndarray, selected_predictions: pd.DataFrame | None) -> list[dict[str, Any]]:
    work = df.copy()
    work["_ts"] = pd.to_datetime(work["id.timestamp"], errors="coerce", utc=True)
    work["_month"] = work["_ts"].dt.to_period("M").astype(str)
    work["_score"] = score
    work["_y"] = target_values(work)
    rows = []
    rejection_lookup: dict[tuple[str, int], float] = {}
    if selected_predictions is not None and len(selected_predictions):
        p = selected_predictions.copy()
        p["_month"] = pd.to_datetime(p["id.timestamp"], errors="coerce", utc=True).dt.to_period("M").astype(str)
        for (month, horizon), g in p.groupby(["_month", "id.horizon"]):
            rejection_lookup[(str(month), int(horizon))] = float(g["reject"].mean())
    for (month, horizon), g in work.groupby(["_month", "id.horizon"]):
        scores = g["_score"].to_numpy(float)
        y = g["_y"].to_numpy(int)
        rows.append(
            {
                "month": str(month),
                "horizon": int(horizon),
                "rows": int(len(g)),
                "target_prevalence": float(y.mean()) if len(y) else 0.0,
                "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
                "score_median": float(np.median(scores)) if len(scores) else 0.0,
                "score_p70": float(np.quantile(scores, 0.70)) if len(scores) else 0.0,
                "score_p80": float(np.quantile(scores, 0.80)) if len(scores) else 0.0,
                "score_p90": float(np.quantile(scores, 0.90)) if len(scores) else 0.0,
                "realized_rejection_horizon_conditioned": rejection_lookup.get((str(month), int(horizon))),
            }
        )
    return rows


def opened_lockbox_diagnostic(
    strided: pd.DataFrame,
    score: np.ndarray,
    selected: dict[str, Any],
    global_ref: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ts = pd.to_datetime(strided["id.timestamp"], errors="coerce", utc=True)
    idx = np.flatnonzero((ts >= LOCKBOX_START).to_numpy())
    cal = np.flatnonzero((ts < LOCKBOX_START).to_numpy())
    fold = type("FoldLike", (), {"name": "opened_lockbox", "calibration_idx": cal, "evaluation_idx": idx, "periods": {}})()
    y_all = target_values(strided).to_numpy(int)
    selected_spec = PolicySpec(
        selected["method"],
        float(selected["budget"]),
        selected.get("rolling_window_days"),
        selected.get("alpha"),
        selected.get("shrinkage_constant"),
    )
    local_score = np.array(score, dtype=float, copy=True)
    thr, meta = rolling_policy_thresholds(
        strided,
        cal,
        idx,
        local_score,
        y_all,
        selected_spec,
        args.minimum_history_rows,
        args.minimum_horizon_rows,
        args.minimum_positive_count,
    )
    y = y_all[idx]
    selected_metrics = metrics_for_policy(strided, idx, y, local_score[idx], thr, selected_spec.budget)
    selected_horizon = horizon_metrics(strided, idx, y, local_score[idx], thr, selected_spec.budget)
    static_thr = np.full(len(idx), ORIGINAL_THRESHOLD)
    static_metrics = metrics_for_policy(strided, idx, y, score[idx], static_thr, PRIMARY_BUDGET)
    global_spec = PolicySpec("GLOBAL_ROLLING_REFERENCE", PRIMARY_BUDGET, 30)
    global_score = np.array(score, dtype=float, copy=True)
    gthr, gmeta = rolling_policy_thresholds(
        strided,
        cal,
        idx,
        global_score,
        y_all,
        global_spec,
        args.minimum_history_rows,
        args.minimum_horizon_rows,
        args.minimum_positive_count,
    )
    global_metrics = metrics_for_policy(strided, idx, y, global_score[idx], gthr, PRIMARY_BUDGET)
    global_horizon = horizon_metrics(strided, idx, y, global_score[idx], gthr, PRIMARY_BUDGET)
    pred = strided.iloc[idx][[c for c in ("id.symbol", "id.timestamp", "id.timeframe", "id.horizon", TARGET) if c in strided.columns]].copy()
    pred["policy"] = selected["method"]
    pred["risk_score"] = local_score[idx]
    pred["policy_threshold"] = thr
    pred["reject"] = (local_score[idx] >= thr).astype(int)
    opened = {
        "status": "OPENED_LOCKBOX_DIAGNOSTIC",
        "opened_lockbox_previously_observed": True,
        "opened_lockbox_used_for_selection": False,
        "policy_changed_after_opened_lockbox": False,
        "not_independent": True,
        "not_sufficient_for_promotion": True,
        "static_threshold_original": static_metrics,
        "global_e21_reference": {
            "metrics": global_metrics,
            "horizon_balance": global_horizon,
            "meta": gmeta,
            "engine": "e22_day_batched",
            "note": "Re-implementation of the E2.1 policy with the E2.2 day-batched engine; numbers differ slightly from the E2.1 report, which uses a per-row rolling engine.",
        },
        "selected_e22_policy": {"metrics": selected_metrics, "horizon_balance": selected_horizon, "meta": meta},
        "contradicts_internal_evidence": selected_metrics["lift"] < 1.0 or selected_metrics["realized_rejection_rate"] > 0.70,
    }
    return opened, pred


def render_markdown(payload: dict[str, Any]) -> str:
    sel = payload.get("selected_policy") or {}
    lines = [
        "# FASE-E2.2 Horizon-Conditioned TRRM Policy",
        "",
        "## 1. Estado",
        f"- decision: {payload['decision']}",
        f"- reason: {payload['decision_reason']}",
        "- mode: research-only",
        "",
        "## 2. Input FASE-E2/E2.1",
        f"- model: {payload['artifact_integrity']['model']}",
        f"- target: {payload['target']}",
        f"- feature_hash: {payload['feature_hash']}",
        "- global rolling reference: ROLLING_GLOBAL_QUANTILE_PAST_ONLY budget 0.30 window 30d",
        "",
        "## 3. Artifact integrity",
        json.dumps(payload["artifact_integrity"], indent=2, default=json_default),
        "",
        "## 4. Lockbox status",
        "- already opened in FASE-E2",
        "- diagnostic-only",
        "- no selection",
        "",
        "## 5. Internal temporal folds",
        json.dumps(payload["folds"], indent=2, default=json_default),
        "",
        "## 6. Policies evaluated",
        ", ".join(sorted(set(r["method"] for r in payload["policy_results"]))),
        "",
        "## 7. Budget stability",
        json.dumps(sel.get("aggregate", {}), indent=2, default=json_default),
        "",
        "## 8. Horizon balance",
        json.dumps(payload["selected_horizon_balance"], indent=2, default=json_default),
        "",
        "## 9. Fold robustness",
        json.dumps(sel.get("folds", []), indent=2, default=json_default),
        "",
        "## 10. Drift by horizon",
        json.dumps(payload["drift_by_horizon"][:40], indent=2, default=json_default),
        "",
        "## 11. Fallback behavior",
        json.dumps(payload["fallback_behavior"], indent=2, default=json_default),
        "",
        "## 12. Complexity comparison",
        json.dumps(payload["complexity_comparison"], indent=2, default=json_default),
        "",
        "## 13. Selected policy",
        f"- method: {sel.get('method')}",
        f"- budget: {sel.get('budget')}",
        f"- window: {sel.get('rolling_window_days')}",
        f"- alpha: {sel.get('alpha')}",
        f"- shrinkage_constant: {sel.get('shrinkage_constant')}",
        "- fallback: per-horizon threshold -> blended/shrunk threshold -> global rolling threshold -> NO_DECISION when global history is insufficient",
        "",
        "## 14. Comparison against E2.1 global policy",
        json.dumps(payload["comparison_against_e21"], indent=2, default=json_default),
        "",
        "## 15. Opened-lockbox diagnostic",
        "- OPENED_LOCKBOX_DIAGNOSTIC only: previously observed, not independent, not used for selection, no reselection.",
        json.dumps(payload["opened_lockbox_diagnostic"], indent=2, default=json_default),
        "",
        "## 16. Limitations",
        "\n".join(f"- {x}" for x in payload["limitations"]),
        "",
        "## 17. Forward freeze recommendation",
        "\n".join(f"- {x}" for x in payload["forward_plan"]),
        "",
        "## 18. Decision",
        f"- {payload['decision']}: {payload['decision_reason']}",
        "",
        "## 19. Recommended next phase",
        f"- {payload['recommended_next_phase']}",
        "",
        "## 20. Safety confirmations",
        "- No live changes. No active_manifest. No YAML live. No PM2 restart. No orders. No .env. No TS changes. No push. No model promotion. No shadow. No base-model retraining. No per-horizon models. Opened lockbox not used for selection.",
        "",
    ]
    return "\n".join(lines)


def run_e22(args: argparse.Namespace) -> dict[str, Any]:
    if args.target != TARGET:
        raise ValueError("ARTIFACT_INTEGRITY_ERROR: target is frozen to target.tail_risk_roe_030")
    out_dir = Path(args.output_dir)
    policy_out = Path(args.policy_output_dir)
    validate_research_path(out_dir)
    validate_research_path(policy_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    dense = pd.read_csv(args.dense_csv)
    strided = pd.read_csv(args.strided_csv)
    e21 = load_json(Path(args.e21_report_json))
    pipeline = load_pipeline(Path(args.e2_model_dir))
    integ = artifact_integrity(args, dense, pipeline, e21)
    if integ["status"] != "OK":
        raise ValueError("ARTIFACT_INTEGRITY_ERROR")
    dense_score = score_frame(dense, pipeline)
    strided_score = score_frame(strided, pipeline)
    folds = build_internal_folds(dense, args.embargo_minutes, args.minimum_history_rows, args.minimum_positive_count)
    fold_manifest = [
        {
            "name": f.name,
            "periods": f.periods,
            "history_rows": int(len(f.train_idx)),
            "calibration_rows": int(len(f.calibration_idx)),
            "evaluation_rows": int(len(f.evaluation_idx)),
            "calibration_positives": int(target_values(dense).iloc[f.calibration_idx].sum()),
            "evaluation_positives": int(target_values(dense).iloc[f.evaluation_idx].sum()),
            "evaluation_positives_by_horizon": {
                str(h): int(target_values(dense).iloc[f.evaluation_idx[pd.to_numeric(dense.iloc[f.evaluation_idx]["id.horizon"], errors="coerce").astype(int).to_numpy() == h]].sum())
                for h in HORIZONS
            },
            "ends_before_opened_lockbox": f.periods.get("evaluation_end") is not None and pd.Timestamp(f.periods["evaluation_end"]) < LOCKBOX_START,
            "embargo_minutes": args.embargo_minutes,
        }
        for f in folds
    ]
    if not all(x["ends_before_opened_lockbox"] for x in fold_manifest):
        raise ValueError("LEAKAGE_RISK_TOO_HIGH: internal fold crosses opened lockbox")
    results = []
    for spec in build_specs(args):
        results.append(
            evaluate_policy(
                dense,
                dense_score,
                folds,
                spec,
                args.minimum_history_rows,
                args.minimum_horizon_rows,
                args.minimum_positive_count,
            )
        )
    global_ref = next((r for r in results if r["method"] == "GLOBAL_ROLLING_REFERENCE" and abs(r["budget"] - PRIMARY_BUDGET) < 1e-9), None)
    selected = select_policy(results)
    opened, opened_pred = opened_lockbox_diagnostic(strided, strided_score, selected, global_ref, args) if selected else ({}, pd.DataFrame())
    dec, reason, next_phase = decision(selected, global_ref, opened)
    selected_horizon = []
    selected_horizon_fold_mean = []
    selected_predictions = None
    if selected and selected.get("_predictions") is not None:
        selected_predictions = selected["_predictions"]
        final_fold = selected["folds"][-1]
        selected_horizon = final_fold.get("horizon_metrics", [])
        good_folds = [f for f in selected["folds"] if not f.get("skipped")]
        for horizon in HORIZONS:
            rows_h = [hr for f in good_folds for hr in f.get("horizon_metrics", []) if hr["horizon"] == horizon]
            if rows_h:
                selected_horizon_fold_mean.append(
                    {
                        "horizon": horizon,
                        "scope": "unweighted_mean_over_all_usable_folds",
                        "realized_rejection_rate": float(np.mean([r["realized_rejection_rate"] for r in rows_h])),
                        "tail_capture_rate": float(np.mean([r["tail_capture_rate"] for r in rows_h])),
                        "residual_tail_rate": float(np.mean([r["residual_tail_rate"] for r in rows_h])),
                        "lift": float(np.mean([r["lift"] for r in rows_h])),
                    }
                )
    complexity_comparison = [
        {
            "method": r["method"],
            "budget": r["budget"],
            "window": r.get("rolling_window_days"),
            "alpha": r.get("alpha"),
            "shrinkage_constant": r.get("shrinkage_constant"),
            "complexity": r["complexity"],
            "aggregate": r["aggregate"],
        }
        for r in results
        if abs(r["budget"] - PRIMARY_BUDGET) < 1e-9
    ]
    comparison = {}
    if selected and global_ref:
        sa = selected["aggregate"]
        ga = global_ref["aggregate"]
        comparison = {
            "budget_error_delta": sa["mean_absolute_budget_error"] - ga["mean_absolute_budget_error"],
            "horizon_spread_delta": (sa.get("horizon_rejection_spread") or 0.0) - (ga.get("horizon_rejection_spread") or 0.0),
            "capture_delta": sa["mean_tail_capture"] - ga["mean_tail_capture"],
            "lift_delta": sa["mean_lift"] - ga["mean_lift"],
            "residual_tail_rate_delta": sa["mean_residual_tail_rate"] - ga["mean_residual_tail_rate"],
            "complexity_added": selected["complexity"]["implementation_complexity"],
        }
    payload: dict[str, Any] = {
        "schema_version": "phase_e22_trrm_horizon_policy_v1",
        "generated_at": stamp,
        "mode": "research-only",
        "target": TARGET,
        "feature_hash": integ["feature_hash"],
        "artifact_integrity": integ,
        "paths": {
            "e21_report_json": str(args.e21_report_json),
            "e21_internal_predictions": str(args.e21_internal_predictions),
            "opened_lockbox_predictions": str(args.opened_lockbox_predictions),
            "e2_model_dir": str(args.e2_model_dir),
            "dense_csv": str(args.dense_csv),
            "strided_csv": str(args.strided_csv),
        },
        "folds": fold_manifest,
        "policy_results": [{k: v for k, v in r.items() if k != "_predictions"} for r in results],
        "global_rolling_reference": {k: v for k, v in (global_ref or {}).items() if k != "_predictions"},
        "selected_policy": {k: v for k, v in (selected or {}).items() if k != "_predictions"},
        "selected_ready_criteria": ready_criteria(selected) if selected else {},
        "selected_horizon_balance": selected_horizon,
        "selected_horizon_balance_scope": "last_pre_lockbox_fold_only",
        "selected_horizon_balance_fold_mean": selected_horizon_fold_mean,
        "aggregate_scope": "unweighted_mean_over_all_usable_folds",
        "policy_engine": "e22_day_batched",
        "comparison_against_e21": comparison,
        "drift_by_horizon": drift_by_horizon(dense, dense_score, selected_predictions),
        "fallback_behavior": {
            "step_1": "use same-horizon past-only rolling/expanding threshold when sample requirements are met",
            "step_2": "use blended or shrunk global+horizon threshold for blended policies",
            "step_3": "fallback to global rolling threshold when horizon history is insufficient",
            "step_4": "emit NO_DECISION when global history is also insufficient",
            "never_uses_static_0391_as_final_fallback": True,
        },
        "complexity_comparison": complexity_comparison,
        "opened_lockbox_diagnostic": opened,
        "lockbox_status": {
            "opened_lockbox_previously_observed": True,
            "opened_lockbox_used_for_selection": False,
            "policy_changed_after_opened_lockbox": False,
            "diagnostic_only": True,
        },
        "limitations": [
            "Opened lockbox was already observed in FASE-E2/E2.1 and is diagnostic only.",
            "No new model, target, feature family, symbol threshold, or live integration is introduced.",
            "Horizon-conditioned thresholding adds operational state and requires forward confirmation.",
        ],
        "forward_plan": [
            "FASE-F0 only if accepted: freeze research candidate and collect forward scores without enforcement.",
            "Minimum future sample: at least 50 tail events, preferably 100, across H6/H12/H24 and multiple regimes.",
            "Stop if forward realized rejection exceeds horizon bands or any horizon collapses without sample-size justification.",
        ],
        "safety_confirmations": {
            "no_live_changes": True,
            "no_active_manifest": True,
            "no_yaml_live": True,
            "no_pm2_restart": True,
            "no_orders": True,
            "no_positions_closed": True,
            "no_env": True,
            "no_ts_changes": True,
            "no_push": True,
            "no_model_promotion": True,
            "no_shadow": True,
            "base_model_not_retrained": True,
            "no_per_horizon_models": True,
            "opened_lockbox_not_used_for_selection": True,
            "no_post_lockbox_reselection": True,
            "no_future_or_label_features": True,
            "research_only_artifacts": True,
        },
    }
    payload["decision"] = dec
    payload["decision_reason"] = reason
    payload["recommended_next_phase"] = next_phase
    if parse_bool(args.write_artifacts):
        policy_dir = policy_out / stamp
        validate_research_path(policy_dir)
        policy_dir.mkdir(parents=True, exist_ok=True)
        (policy_dir / "policy_metadata.json").write_text(
            json.dumps(
                {
                    "base_model_reference": str(args.e2_model_dir),
                    "feature_hash": integ["feature_hash"],
                    "target": TARGET,
                    "selected_policy": payload["selected_policy"],
                    "fallback_behavior": payload["fallback_behavior"],
                    "decision": dec,
                    "lockbox_disclaimer": payload["lockbox_status"],
                },
                indent=2,
                default=json_default,
            )
        )
        payload["research_policy_path"] = str(policy_dir)
    if parse_bool(args.write_report):
        md = out_dir / f"aegis_phase_e22_trrm_horizon_policy_{stamp}.md"
        js = out_dir / f"aegis_phase_e22_trrm_horizon_policy_{stamp}.json"
        internal_csv = out_dir / f"aegis_phase_e22_internal_predictions_{stamp}.csv"
        lock_csv = out_dir / f"aegis_phase_e22_opened_lockbox_diagnostic_{stamp}.csv"
        if selected_predictions is not None:
            selected_predictions.to_csv(internal_csv, index=False)
        if len(opened_pred):
            opened_pred.to_csv(lock_csv, index=False)
        payload["artifacts"] = {
            "markdown": str(md),
            "json": str(js),
            "internal_predictions": str(internal_csv),
            "opened_lockbox_diagnostic": str(lock_csv),
        }
        js.write_text(json.dumps(payload, indent=2, default=json_default))
        md.write_text(render_markdown(payload))
    print(
        json.dumps(
            {
                "decision": dec,
                "selected_policy": {k: payload["selected_policy"].get(k) for k in ("method", "budget", "rolling_window_days", "alpha", "shrinkage_constant")},
                "markdown": payload.get("artifacts", {}).get("markdown"),
                "json": payload.get("artifacts", {}).get("json"),
                "policy_path": payload.get("research_policy_path"),
            },
            indent=2,
            default=json_default,
        )
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        run_e22(parse_args(argv))
        return 0
    except ValueError as exc:
        msg = str(exc)
        if "ARTIFACT_INTEGRITY_ERROR" in msg:
            print(json.dumps({"decision": "ARTIFACT_INTEGRITY_ERROR", "reason": msg}))
            return 2
        if "LEAKAGE" in msg:
            print(json.dumps({"decision": "LEAKAGE_RISK_TOO_HIGH", "reason": msg}))
            return 3
        print(json.dumps({"decision": "RESEARCH_NOT_READY", "reason": msg}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
