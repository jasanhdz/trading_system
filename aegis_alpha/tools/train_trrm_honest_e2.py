#!/usr/bin/env python3
"""FASE-E2 honest TRRM evaluation over long-horizon tail risk.

Research-only. Trains small sklearn classifiers on the FASE-D2 causal dataset,
selects model/threshold on validation only, and opens lockbox only for final
evaluation. It never writes live/active artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default

warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_DENSE = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_20260710T051035Z.csv")
DEFAULT_STRIDED = Path("/home/jasan/Develop/aegis_trrm_causal_feature_dataset_d2_strided_20260710T051035Z.csv")
DEFAULT_D2_JSON = Path("/home/jasan/Develop/aegis_phase_d2_tail_target_review_20260710T051035Z.json")
DEFAULT_OUT = Path("/home/jasan/Develop")
DEFAULT_MODEL_ROOT = Path("/home/jasan/Develop/aegis_research_models/trrm_e2")
TARGET = "target.tail_risk_roe_030"
TIMEFRAME_MINUTES = 5
TRAIN_START = pd.Timestamp("2025-07-09T00:00:00Z")
VALIDATION_START = pd.Timestamp("2026-02-13T00:00:00Z")
LOCKBOX_START = pd.Timestamp("2026-04-27T00:00:00Z")
LOCKBOX_END = pd.Timestamp("2026-07-09T00:00:00Z")
BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
PRIMARY_BUDGET = 0.30
RESEARCH_ROOT = Path("/home/jasan/Develop").resolve()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FASE-E2 honest TRRM evaluation")
    p.add_argument("--dense-csv", default=str(DEFAULT_DENSE))
    p.add_argument("--strided-csv", default=str(DEFAULT_STRIDED))
    p.add_argument("--d2-report-json", default=str(DEFAULT_D2_JSON))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--model-output-dir", default=str(DEFAULT_MODEL_ROOT))
    p.add_argument("--target", default=TARGET)
    p.add_argument("--embargo-minutes", type=int, default=120)
    p.add_argument("--write-models", default="true")
    p.add_argument("--write-report", default="true")
    p.add_argument("--max-train-rows", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"1", "true", "yes"}).astype(int)


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def target_values(df: pd.DataFrame, target: str = TARGET) -> pd.Series:
    if target not in df.columns:
        raise KeyError(f"target missing: {target}")
    return bool_series(df[target])


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path)


def duplicate_key_columns(df: pd.DataFrame) -> list[str]:
    for cols in (["id.symbol", "id.horizon", "id.timestamp"], ["symbol", "horizon", "timestamp"]):
        if set(cols) <= set(df.columns):
            return cols
    return []


def dedupe_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = duplicate_key_columns(df)
    if not keys:
        return df, {"key_columns": [], "exact_duplicates_removed": 0, "contradictory_duplicates": 0}
    dup_mask = df.duplicated(subset=keys, keep=False)
    if not dup_mask.any():
        return df, {"key_columns": keys, "exact_duplicates_removed": 0, "contradictory_duplicates": 0}
    dup = df.loc[dup_mask].copy()
    contradictory = 0
    keep_drop: list[int] = []
    for _, g in dup.groupby(keys, dropna=False):
        check = g.drop(columns=[], errors="ignore").fillna("__NA__").astype(str)
        if len(check.drop_duplicates()) > 1:
            contradictory += len(g)
        else:
            keep_drop.extend(g.index[1:].tolist())
    if contradictory:
        raise ValueError(f"DATASET_INTEGRITY_ERROR: contradictory duplicate keys={contradictory}")
    out = df.drop(index=keep_drop).reset_index(drop=True)
    return out, {"key_columns": keys, "exact_duplicates_removed": len(keep_drop), "contradictory_duplicates": 0}


def eligible_features(df: pd.DataFrame) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    features: list[str] = []
    excluded: list[dict[str, str]] = []
    manual: list[dict[str, str]] = []
    blocked_prefix = ("target.", "label.", "future_eval.", "reference.", "id.")
    blocked_exact = {"symbol", "raw_symbol", "row_id", "group_id", "timestamp", "open_time", "close_time"}
    for col in df.columns:
        if col == "feature.close":
            excluded.append({"column": col, "reason": "raw_close_reference_not_trainable"})
            continue
        if col.startswith(blocked_prefix) or col in blocked_exact:
            if col.startswith(("target.", "label.", "future_eval.")):
                excluded.append({"column": col, "reason": "target_label_future_namespace"})
            continue
        if col.startswith("feature."):
            features.append(col)
        elif any(token in col.lower() for token in ("target", "future", "mae", "mfe", "pnl", "quality", "tail")):
            manual.append({"column": col, "reason": "ambiguous_non_feature_outcome_name"})
    return sorted(features), excluded, manual


def add_horizon_one_hot(df: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    h = pd.to_numeric(df.get("id.horizon", pd.Series(index=df.index)), errors="coerce").fillna(-1).astype(int)
    for val in (6, 12, 24):
        out[f"horizon_{val}"] = (h == val).astype(float)
    return out


@dataclass
class SplitManifest:
    train_idx: np.ndarray
    validation_idx: np.ndarray
    lockbox_idx: np.ndarray
    rows_purged: int
    rows_embargoed: int
    overlap_ok: bool
    periods: dict[str, Any]


def make_split(df: pd.DataFrame, embargo_minutes: int) -> SplitManifest:
    ts = pd.to_datetime(df["id.timestamp"], errors="coerce", utc=True)
    horizon = pd.to_numeric(df["id.horizon"], errors="coerce").fillna(24).astype(int)
    outcome_end = ts + pd.to_timedelta(horizon * TIMEFRAME_MINUTES, unit="m")
    embargo = pd.Timedelta(minutes=embargo_minutes)

    raw_train = (ts >= TRAIN_START) & (ts < VALIDATION_START)
    raw_val = (ts >= VALIDATION_START) & (ts < LOCKBOX_START)
    raw_lock = (ts >= LOCKBOX_START) & (ts < LOCKBOX_END)

    train = raw_train & (outcome_end < VALIDATION_START)
    val = raw_val & (ts >= VALIDATION_START + embargo) & (outcome_end < LOCKBOX_START)
    lock = raw_lock & (ts >= LOCKBOX_START + embargo)

    rows_purged = int((raw_train.sum() - train.sum()) + (raw_val.sum() - (raw_val & (outcome_end < LOCKBOX_START)).sum()))
    rows_embargoed = int(((raw_val & (outcome_end < LOCKBOX_START)).sum() - val.sum()) + (raw_lock.sum() - lock.sum()))
    train_idx = np.flatnonzero(train.to_numpy())
    val_idx = np.flatnonzero(val.to_numpy())
    lock_idx = np.flatnonzero(lock.to_numpy())
    overlap_ok = bool(
        len(train_idx) and len(val_idx) and len(lock_idx)
        and outcome_end.iloc[train_idx].max() < ts.iloc[val_idx].min()
        and outcome_end.iloc[val_idx].max() < ts.iloc[lock_idx].min()
        and set(ts.iloc[train_idx].astype(str)).isdisjoint(set(ts.iloc[val_idx].astype(str)))
        and set(ts.iloc[val_idx].astype(str)).isdisjoint(set(ts.iloc[lock_idx].astype(str)))
    )
    periods = {
        "train_start": str(ts.iloc[train_idx].min()) if len(train_idx) else None,
        "train_end": str(ts.iloc[train_idx].max()) if len(train_idx) else None,
        "validation_start": str(ts.iloc[val_idx].min()) if len(val_idx) else None,
        "validation_end": str(ts.iloc[val_idx].max()) if len(val_idx) else None,
        "lockbox_start": str(ts.iloc[lock_idx].min()) if len(lock_idx) else None,
        "lockbox_end": str(ts.iloc[lock_idx].max()) if len(lock_idx) else None,
        "max_train_outcome_end": str(outcome_end.iloc[train_idx].max()) if len(train_idx) else None,
        "max_validation_outcome_end": str(outcome_end.iloc[val_idx].max()) if len(val_idx) else None,
    }
    return SplitManifest(train_idx, val_idx, lock_idx, rows_purged, rows_embargoed, overlap_ok, periods)


class MedianImputer:
    def fit(self, x: pd.DataFrame) -> "MedianImputer":
        self.medians = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.replace([np.inf, -np.inf], np.nan).fillna(self.medians).fillna(0.0)


class StandardScalerLite:
    def fit(self, x: pd.DataFrame) -> "StandardScalerLite":
        self.mean = x.mean()
        self.std = x.std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return (x - self.mean) / self.std


def fit_preprocess(x: pd.DataFrame, train_idx: np.ndarray, scale: bool) -> tuple[MedianImputer, StandardScalerLite | None, pd.DataFrame]:
    imputer = MedianImputer().fit(x.iloc[train_idx])
    xi = imputer.transform(x)
    scaler = StandardScalerLite().fit(xi.iloc[train_idx]) if scale else None
    if scaler is not None:
        xi = scaler.transform(xi)
    return imputer, scaler, xi


def threshold_for_budget(score: np.ndarray, budget: float) -> float:
    score = np.asarray(score, dtype=float)
    if len(score) == 0:
        return math.inf
    return float(np.quantile(score, max(0.0, min(1.0, 1.0 - budget))))


def ece_score(y: np.ndarray, score: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if not mask.any():
            rows.append({"low": float(lo), "high": float(hi), "count": 0, "predicted": 0.0, "observed": 0.0})
            continue
        pred = float(score[mask].mean())
        obs = float(y[mask].mean())
        ece += (mask.sum() / total) * abs(pred - obs) if total else 0.0
        rows.append({"low": float(lo), "high": float(hi), "count": int(mask.sum()), "predicted": pred, "observed": obs})
    return float(ece), rows


def standard_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        pr_auc = float(average_precision_score(y, score)) if len(set(y)) > 1 else None
        roc_auc = float(roc_auc_score(y, score)) if len(set(y)) > 1 else None
        brier = float(brier_score_loss(y, np.clip(score, 0, 1)))
    except Exception:
        pr_auc = roc_auc = brier = None
    ece, bins = ece_score(y, np.clip(score, 0, 1))
    return {
        "rows": int(len(y)),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "ece": ece,
        "calibration_bins": bins,
        "false_negatives": fn,
        "false_positives": fp,
        "rejection_rate": float(pred.mean()) if len(pred) else 0.0,
        "retained_rate": 1.0 - (float(pred.mean()) if len(pred) else 0.0),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def budget_metrics(y: np.ndarray, score: np.ndarray, budget: float, threshold: float | None = None) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    thr = threshold_for_budget(score, budget) if threshold is None else float(threshold)
    reject = score >= thr
    rejected = int(reject.sum())
    positives = int(y.sum())
    captured = int(((y == 1) & reject).sum())
    retained_y = y[~reject]
    prevalence = float(y.mean()) if len(y) else 0.0
    precision_rej = captured / rejected if rejected else 0.0
    residual = float(retained_y.mean()) if len(retained_y) else 0.0
    reduction = 1.0 - (residual / prevalence) if prevalence > 0 else 0.0
    return {
        "budget": float(budget),
        "threshold": thr,
        "rows": int(len(y)),
        "prevalence": prevalence,
        "rejection_rate": rejected / len(y) if len(y) else 0.0,
        "retained_rate": 1.0 - (rejected / len(y) if len(y) else 0.0),
        "tail_capture_rate": captured / positives if positives else 0.0,
        "precision_among_rejected": precision_rej,
        "residual_tail_rate": residual,
        "tail_risk_reduction": reduction,
        "retained_tail_events": int(positives - captured),
        "lift_in_rejected_group": precision_rej / prevalence if prevalence > 0 else 0.0,
        "top_decile_lift": top_decile_lift(y, score),
        "number_needed_to_reject": rejected / captured if captured else None,
        "captured_tail_events": captured,
    }


def top_decile_lift(y: np.ndarray, score: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    n = max(1, int(math.ceil(len(y) * 0.10)))
    order = np.argsort(np.asarray(score))[-n:]
    prevalence = float(np.mean(y))
    return float(np.mean(np.asarray(y)[order]) / prevalence) if prevalence > 0 else 0.0


def rejection_budget_table(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    return {f"{int(b * 100)}pct": budget_metrics(y, score, b) for b in BUDGETS}


def available_models(seed: int) -> dict[str, tuple[str, Any]]:
    models: dict[str, tuple[str, Any]] = {}
    try:
        from sklearn.linear_model import LogisticRegression, SGDClassifier

        models["logistic_regression"] = ("linear", LogisticRegression(max_iter=800, class_weight="balanced", solver="liblinear", random_state=seed))
        models["sgd_log_loss"] = ("linear", SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000, tol=1e-3, random_state=seed))
    except Exception:
        pass
    try:
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier

        models["hist_gradient_boosting"] = ("tree", HistGradientBoostingClassifier(max_iter=80, learning_rate=0.07, max_leaf_nodes=31, random_state=seed))
        models["gradient_boosting"] = ("tree", GradientBoostingClassifier(n_estimators=70, learning_rate=0.06, max_depth=3, random_state=seed))
        models["random_forest"] = ("tree", RandomForestClassifier(n_estimators=90, max_depth=8, min_samples_leaf=25, class_weight="balanced_subsample", random_state=seed, n_jobs=1))
        models["extra_trees"] = ("tree", ExtraTreesClassifier(n_estimators=90, max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=seed, n_jobs=1))
    except Exception:
        pass
    return models


def score_estimator(model: Any, x_eval: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_eval)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x_eval), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -40, 40)))
    return np.asarray(model.predict(x_eval), dtype=float)


def fit_platt(score_val: np.ndarray, y_val: np.ndarray) -> Any | None:
    if len(set(y_val.astype(int))) < 2:
        return None
    try:
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression(solver="liblinear")
        calibrator.fit(np.asarray(score_val).reshape(-1, 1), y_val.astype(int))
        return calibrator
    except Exception:
        return None


def apply_platt(calibrator: Any | None, scores: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return scores
    return np.asarray(calibrator.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1], dtype=float)


def fit_model(name: str, kind: str, model: Any, x: pd.DataFrame, y: pd.Series, split: SplitManifest, design_cols: list[str], max_train_rows: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    train_idx = split.train_idx
    if max_train_rows and len(train_idx) > max_train_rows:
        train_idx = np.sort(rng.choice(train_idx, size=max_train_rows, replace=False))
    t0 = time.time()
    imputer, scaler, xp = fit_preprocess(x[design_cols], train_idx, scale=(kind == "linear"))
    model.fit(xp.iloc[train_idx], y.iloc[train_idx])
    train_seconds = time.time() - t0
    raw_val = score_estimator(model, xp.iloc[split.validation_idx])
    y_val = y.iloc[split.validation_idx].to_numpy(int)
    cal = fit_platt(raw_val, y_val)
    variants = {
        "raw": {"calibrator": None, "validation_score": raw_val},
        "sigmoid": {"calibrator": cal, "validation_score": apply_platt(cal, raw_val)} if cal is not None else None,
    }
    out: dict[str, Any] = {
        "model": model,
        "kind": kind,
        "imputer": imputer,
        "scaler": scaler,
        "xp": xp,
        "train_idx_used": train_idx,
        "train_seconds": train_seconds,
        "variants": {},
    }
    for cal_name, data in variants.items():
        if data is None:
            continue
        score_val = data["validation_score"]
        threshold = threshold_for_budget(score_val, PRIMARY_BUDGET)
        out["variants"][cal_name] = {
            "calibrator": data["calibrator"],
            "validation_score": score_val,
            "validation_standard": standard_metrics(y_val, score_val, threshold),
            "validation_budget": budget_metrics(y_val, score_val, PRIMARY_BUDGET, threshold),
            "validation_budget_table": rejection_budget_table(y_val, score_val),
            "selected_threshold": threshold,
        }
    return out


def transform_scores(fit: dict[str, Any], idx: np.ndarray, cal_name: str) -> np.ndarray:
    raw = score_estimator(fit["model"], fit["xp"].iloc[idx])
    calibrator = fit["variants"][cal_name]["calibrator"]
    return apply_platt(calibrator, raw)


def score_to_percentile(vals: pd.Series, train_idx: np.ndarray, high_is_risk: bool = True) -> np.ndarray:
    x = safe_num(vals).fillna(safe_num(vals).iloc[train_idx].median())
    unique = set(np.unique(x.dropna().to_numpy(float)))
    if unique.issubset({0.0, 1.0}):
        raw = x.to_numpy(float)
        return raw if high_is_risk else 1.0 - raw
    train = x.iloc[train_idx].to_numpy(float)
    ranks = np.searchsorted(np.sort(train), x.to_numpy(float), side="right") / max(1, len(train))
    return ranks if high_is_risk else 1.0 - ranks


def baseline_scores(df: pd.DataFrame, y: pd.Series, split: SplitManifest) -> dict[str, dict[str, Any]]:
    zeros = np.zeros(len(df))
    ones = np.ones(len(df))
    scores: dict[str, dict[str, Any]] = {
        "reject_none": {"score": zeros, "causal": True, "eligible_for_selection": False, "eligible_for_promotion": False},
        "reject_all": {"score": ones, "causal": True, "eligible_for_selection": False, "eligible_for_promotion": False},
        "prevalence_baseline": {"score": np.full(len(df), float(y.iloc[split.train_idx].mean())), "causal": True, "eligible_for_selection": False, "eligible_for_promotion": False},
    }
    groups = {
        "causal_volatility_baseline": ["feature.rolling_range_mean_24", "feature.rolling_range_std_24"],
        "causal_atr_range_percentile_baseline": ["feature.atr_proxy_24", "feature.rolling_range_mean_24"],
        "causal_rebound_squeeze_baseline": ["feature.rebound_risk_proxy", "feature.squeeze_risk_proxy_causal"],
    }
    for name, cols in groups.items():
        parts = []
        used = []
        for col in cols:
            if col in df:
                parts.append(score_to_percentile(df[col], split.train_idx))
                used.append(col)
        score = np.max(np.vstack(parts), axis=0) if parts else zeros.copy()
        scores[name] = {"score": score, "columns": used, "causal": True, "eligible_for_selection": True, "eligible_for_promotion": False}
    oracle_parts = []
    for col in ("target.bad_entry_v4", "target.early_mae_v4", "feature.squeeze_risk_proxy_causal"):
        if col in df:
            oracle_parts.append(bool_series(df[col]).to_numpy(float))
    oracle_score = np.max(np.vstack(oracle_parts), axis=0) if oracle_parts else zeros.copy()
    scores["diagnostic_oracle_upper_bound"] = {
        "score": oracle_score,
        "causal": False,
        "live_usable": False,
        "eligible_for_selection": False,
        "eligible_for_promotion": False,
        "diagnostic_only": True,
    }
    return scores


def evaluate_baseline_set(scores: dict[str, dict[str, Any]], y: pd.Series, split: SplitManifest) -> dict[str, Any]:
    out: dict[str, Any] = {}
    y_val = y.iloc[split.validation_idx].to_numpy(int)
    y_lock = y.iloc[split.lockbox_idx].to_numpy(int)
    for name, data in scores.items():
        score = np.asarray(data["score"], dtype=float)
        if name == "reject_all":
            val_threshold = -math.inf
        elif name in {"reject_none", "prevalence_baseline"} or len(np.unique(score[split.validation_idx])) <= 1:
            val_threshold = math.inf
        elif set(np.unique(score[split.validation_idx])).issubset({0.0, 1.0}):
            val_threshold = 0.5
        else:
            val_threshold = threshold_for_budget(score[split.validation_idx], PRIMARY_BUDGET)
        out[name] = {
            "name": name,
            **{k: v for k, v in data.items() if k != "score"},
            "selected_threshold_from_validation": val_threshold,
            "validation": {
                "standard": standard_metrics(y_val, score[split.validation_idx], val_threshold),
                "budget_30": budget_metrics(y_val, score[split.validation_idx], PRIMARY_BUDGET, val_threshold),
                "budget_table": rejection_budget_table(y_val, score[split.validation_idx]),
            },
            "lockbox": {
                "standard": standard_metrics(y_lock, score[split.lockbox_idx], val_threshold),
                "budget_30_frozen_threshold": budget_metrics(y_lock, score[split.lockbox_idx], PRIMARY_BUDGET, val_threshold),
                "budget_table": rejection_budget_table(y_lock, score[split.lockbox_idx]),
            },
        }
    return out


def candidate_key(candidate: dict[str, Any], baseline_30: dict[str, Any]) -> tuple[float, float, float, float]:
    b = candidate["validation"]["budget_30"]
    eligible = (
        b["rejection_rate"] <= 0.35
        and b["retained_rate"] >= 0.65
        and b["lift_in_rejected_group"] > 1.0
        and (
            b["tail_capture_rate"] > baseline_30["tail_capture_rate"]
            or b["residual_tail_rate"] < baseline_30["residual_tail_rate"]
        )
    )
    return (1.0 if eligible else 0.0, b["tail_capture_rate"], b["lift_in_rejected_group"], candidate["validation"]["standard"].get("pr_auc") or 0.0)


def select_candidate(model_results: list[dict[str, Any]], baseline_eval: dict[str, Any]) -> dict[str, Any] | None:
    causal = [
        v for k, v in baseline_eval.items()
        if v.get("eligible_for_selection")
        and k != "diagnostic_oracle_upper_bound"
        and v["validation"]["budget_30"]["rejection_rate"] <= 0.85
    ]
    best_base = max(causal, key=lambda r: (r["validation"]["budget_30"]["tail_capture_rate"], -r["validation"]["budget_30"]["residual_tail_rate"])) if causal else None
    baseline_30 = best_base["validation"]["budget_30"] if best_base else {"tail_capture_rate": 0.0, "residual_tail_rate": 1.0}
    eligible = [m for m in model_results if not m.get("skipped")]
    if not eligible:
        return None
    selected = max(eligible, key=lambda r: candidate_key(r, baseline_30))
    assert selected["name"] != "diagnostic_oracle_upper_bound"
    selected["best_validation_causal_baseline"] = best_base["name"] if best_base else None
    selected["baseline_for_selection"] = baseline_30
    selected["beats_validation_causal_baseline"] = candidate_key(selected, baseline_30)[0] > 0
    return selected


def segment_metrics(df: pd.DataFrame, idx: np.ndarray, y: np.ndarray, score: np.ndarray, threshold: float, keys: list[str]) -> list[dict[str, Any]]:
    local = df.iloc[idx][keys].copy()
    local["_y"] = y
    local["_score"] = score
    out = []
    for group, g in local.groupby(keys, dropna=False):
        if not isinstance(group, tuple):
            group = (group,)
        bm = budget_metrics(g["_y"].to_numpy(int), g["_score"].to_numpy(float), PRIMARY_BUDGET, threshold)
        sm = standard_metrics(g["_y"].to_numpy(int), g["_score"].to_numpy(float), threshold)
        row = {k: str(v) for k, v in zip(keys, group)}
        row.update({
            "rows": int(len(g)),
            "positives": int(g["_y"].sum()),
            "sample_status": "INSUFFICIENT_SEGMENT_SAMPLE" if int(g["_y"].sum()) < 10 else "OK",
            "prevalence": bm["prevalence"],
            "rejection_rate": bm["rejection_rate"],
            "tail_capture_rate": bm["tail_capture_rate"],
            "residual_tail_rate": bm["residual_tail_rate"],
            "lift": bm["lift_in_rejected_group"],
            "false_negatives": sm["false_negatives"],
        })
        out.append(row)
    return out


def month_metrics(df: pd.DataFrame, idx: np.ndarray, y: np.ndarray, score: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    local = df.iloc[idx][["id.timestamp"]].copy()
    local["month"] = pd.to_datetime(local["id.timestamp"], errors="coerce").dt.to_period("M").astype(str)
    local["_y"] = y
    local["_score"] = score
    return segment_metrics(local.rename(columns={"month": "id.month"}), np.arange(len(local)), local["_y"].to_numpy(int), local["_score"].to_numpy(float), threshold, ["id.month"])


def top_importance(model: Any, features: list[str], limit: int = 30) -> list[dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.asarray(model.coef_[0], dtype=float))
    else:
        return []
    rows = [{"feature": f, "importance": float(v)} for f, v in zip(features, vals)]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)[:limit]


def feature_family_filter(features: list[str], family: str) -> list[str]:
    tokens = {
        "volatility": ("atr", "range", "volatility"),
        "btc_eth": ("btc_", "eth_", "symbol_vs_btc", "symbol_vs_eth"),
        "ema_trend": ("ema_", "trend", "slope"),
        "risk_proxy": ("risk_proxy", "rebound", "squeeze", "breakdown", "room_to_fall", "extension", "exhaustion"),
        "horizon": ("horizon_",),
    }[family]
    return [c for c in features if not any(t in c for t in tokens)]


def train_one_candidate(
    df: pd.DataFrame,
    y: pd.Series,
    split: SplitManifest,
    base_x: pd.DataFrame,
    feature_cols: list[str],
    design_name: str,
    kind: str,
    model_name: str,
    model: Any,
    seed: int,
    max_train_rows: int,
) -> dict[str, Any]:
    x = base_x if design_name == "GLOBAL_CAUSAL" else add_horizon_one_hot(df, base_x)
    cols = list(x.columns)
    try:
        fit = fit_model(model_name, kind, model, x, y, split, cols, max_train_rows, seed)
        rows = []
        for cal_name, variant in fit["variants"].items():
            threshold = variant["selected_threshold"]
            rows.append({
                "name": f"{design_name}:{model_name}:{cal_name}",
                "design": design_name,
                "model": model_name,
                "calibration": cal_name,
                "feature_columns": cols,
                "threshold": threshold,
                "train_seconds": fit["train_seconds"],
                "fit": fit,
                "validation": {
                    "standard": variant["validation_standard"],
                    "budget_30": variant["validation_budget"],
                    "budget_table": variant["validation_budget_table"],
                },
            })
        return {"items": rows}
    except Exception as exc:
        return {"items": [{"name": f"{design_name}:{model_name}", "skipped": True, "reason": str(exc), "design": design_name, "model": model_name}]}


def evaluate_selected_on_lockbox(selected: dict[str, Any], df_dense: pd.DataFrame, y_dense: pd.Series, dense_split: SplitManifest, df_strided: pd.DataFrame, y_strided: pd.Series, strided_split: SplitManifest, strided_x: pd.DataFrame) -> dict[str, Any]:
    fit = selected["fit"]
    cal = selected["calibration"]
    threshold = selected["threshold"]
    dense_score = transform_scores(fit, dense_split.lockbox_idx, cal)
    if selected["design"] == "GLOBAL_CAUSAL_PLUS_HORIZON":
        # Rebuild preprocessing for strided using the fitted train medians/scaler columns.
        sx = add_horizon_one_hot(df_strided, strided_x)[selected["feature_columns"]]
    else:
        sx = strided_x[selected["feature_columns"]]
    si = fit["imputer"].transform(sx)
    if fit["scaler"] is not None:
        si = fit["scaler"].transform(si)
    raw_strided = score_estimator(fit["model"], si.iloc[strided_split.lockbox_idx])
    strided_score = apply_platt(fit["variants"][cal]["calibrator"], raw_strided)
    y_lock_strided = y_strided.iloc[strided_split.lockbox_idx].to_numpy(int)
    y_lock_dense = y_dense.iloc[dense_split.lockbox_idx].to_numpy(int)
    return {
        "strided_score": strided_score,
        "dense_score": dense_score,
        "strided_standard": standard_metrics(y_lock_strided, strided_score, threshold),
        "strided_budget_30": budget_metrics(y_lock_strided, strided_score, PRIMARY_BUDGET, threshold),
        "strided_budget_table": rejection_budget_table(y_lock_strided, strided_score),
        "dense_standard": standard_metrics(y_lock_dense, dense_score, threshold),
        "dense_budget_30": budget_metrics(y_lock_dense, dense_score, PRIMARY_BUDGET, threshold),
        "dense_budget_table": rejection_budget_table(y_lock_dense, dense_score),
    }


def walk_forward_diagnostics(df: pd.DataFrame, x: pd.DataFrame, y: pd.Series, selected: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    ts = pd.to_datetime(df["id.timestamp"], errors="coerce", utc=True)
    mask = (ts >= TRAIN_START) & (ts < LOCKBOX_START)
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) < 100:
        return []
    idx = idx[np.argsort(ts.iloc[idx].to_numpy())]
    folds = []
    for train_frac, val_frac in ((0.35, 0.15), (0.50, 0.15), (0.65, 0.15)):
        train_end = int(len(idx) * train_frac)
        val_end = min(len(idx), train_end + int(len(idx) * val_frac))
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        if len(train_idx) < 50 or len(val_idx) < 20:
            continue
        sm = SplitManifest(train_idx, val_idx, np.array([], dtype=int), 0, 0, True, {})
        name = selected["model"]
        kind, model = available_models(seed).get(name, (None, None))
        if model is None:
            continue
        design_x = add_horizon_one_hot(df, x) if selected["design"] == "GLOBAL_CAUSAL_PLUS_HORIZON" else x
        cols = selected["feature_columns"]
        try:
            fit = fit_model(name, kind, model, design_x[cols], y, sm, cols, 0, seed)
            score = fit["variants"]["raw"]["validation_score"]
            yy = y.iloc[val_idx].to_numpy(int)
            folds.append({
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "pr_auc": standard_metrics(yy, score, threshold_for_budget(score, PRIMARY_BUDGET))["pr_auc"],
                "budget_30": budget_metrics(yy, score, PRIMARY_BUDGET),
            })
        except Exception as exc:
            folds.append({"skipped": True, "reason": str(exc)})
    return folds


def symbol_holdout(df: pd.DataFrame, x: pd.DataFrame, y: pd.Series, selected: dict[str, Any], split: SplitManifest, seed: int) -> list[dict[str, Any]]:
    out = []
    symbols = [s for s in ("BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "SUIUSDT") if s in set(df["id.symbol"].astype(str))]
    kind, model_template = available_models(seed).get(selected["model"], (None, None))
    if model_template is None:
        return out
    import copy

    design_x = add_horizon_one_hot(df, x) if selected["design"] == "GLOBAL_CAUSAL_PLUS_HORIZON" else x
    cols = selected["feature_columns"]
    for sym in symbols:
        train_idx = split.train_idx[df.iloc[split.train_idx]["id.symbol"].astype(str).to_numpy() != sym]
        val_idx = split.validation_idx[df.iloc[split.validation_idx]["id.symbol"].astype(str).to_numpy() == sym]
        if len(train_idx) < 100 or len(val_idx) < 20:
            out.append({"symbol": sym, "status": "INSUFFICIENT_SEGMENT_SAMPLE"})
            continue
        sm = SplitManifest(train_idx, val_idx, np.array([], dtype=int), 0, 0, True, {})
        try:
            fit = fit_model(selected["model"], kind, copy.deepcopy(model_template), design_x[cols], y, sm, cols, 0, seed)
            score = fit["variants"]["raw"]["validation_score"]
            yy = y.iloc[val_idx].to_numpy(int)
            out.append({"symbol": sym, "rows": int(len(val_idx)), "prevalence": float(yy.mean()), "pr_auc": standard_metrics(yy, score, threshold_for_budget(score, PRIMARY_BUDGET))["pr_auc"], "budget_30": budget_metrics(yy, score, PRIMARY_BUDGET)})
        except Exception as exc:
            out.append({"symbol": sym, "skipped": True, "reason": str(exc)})
    return out


def ablation_diagnostics(df: pd.DataFrame, x: pd.DataFrame, y: pd.Series, selected: dict[str, Any], split: SplitManifest, seed: int) -> dict[str, Any]:
    out = {}
    kind, model_template = available_models(seed).get(selected["model"], (None, None))
    if model_template is None:
        return {"status": "NOT_RUN", "reason": "selected model unavailable"}
    import copy

    full_cols = selected["feature_columns"]
    design_x = add_horizon_one_hot(df, x) if selected["design"] == "GLOBAL_CAUSAL_PLUS_HORIZON" else x
    base = selected["validation"]["budget_30"]
    for family in ("volatility", "btc_eth", "ema_trend", "risk_proxy", "horizon"):
        cols = feature_family_filter(full_cols, family)
        removed = [c for c in full_cols if c not in cols]
        if len(cols) < 20 or not removed:
            out[family] = {"status": "NOT_RUN", "removed": removed[:20], "reason": "insufficient removed or remaining features"}
            continue
        try:
            fit = fit_model(selected["model"], kind, copy.deepcopy(model_template), design_x[cols], y, split, cols, 0, seed)
            score = fit["variants"]["raw"]["validation_score"]
            yy = y.iloc[split.validation_idx].to_numpy(int)
            bm = budget_metrics(yy, score, PRIMARY_BUDGET)
            out[family] = {"removed_count": len(removed), "remaining_count": len(cols), "pr_auc": standard_metrics(yy, score, bm["threshold"])["pr_auc"], "budget_30": bm}
        except Exception as exc:
            out[family] = {"skipped": True, "reason": str(exc)}
    vol = out.get("volatility", {}).get("budget_30", {})
    if vol and vol.get("lift_in_rejected_group", 0) < max(1.0, base["lift_in_rejected_group"] * 0.65):
        status = "VOLATILITY_DEPENDENCE_HIGH"
    elif any(out.get(k, {}).get("budget_30", {}).get("lift_in_rejected_group", 0) >= 1.1 for k in ("btc_eth", "ema_trend", "risk_proxy")):
        status = "MULTI_SIGNAL_RISK_STRUCTURE_CONFIRMED"
    else:
        status = "WEAK_SIGNAL_STRUCTURE"
    out["status"] = status
    return out


def decision(payload: dict[str, Any]) -> tuple[str, str, str]:
    if payload["integrity"]["status"] != "OK":
        return "DATASET_INTEGRITY_ERROR", payload["integrity"]["status"], "Stop and fix dataset integrity."
    if payload["leakage_risk"] or not payload["split_checks"]["overlap_ok"]:
        return "LEAKAGE_RISK_TOO_HIGH", "feature or split leakage check failed", "Stop and fix leakage before modeling."
    selected = payload.get("selected_candidate")
    if not selected:
        return "HONEST_CAUSAL_BASELINE_NOT_BEATEN", "no eligible model beat causal baselines on validation", "Review feature families or use simple causal rules."
    lock = payload["lockbox"]["strided_budget_30"]
    std = payload["lockbox"]["strided_standard"]
    prevalence = std["prevalence"]
    best_base = payload["comparison"]["best_causal_baseline_lockbox_budget_30"]
    beats_base = lock["tail_capture_rate"] > best_base["tail_capture_rate"] or lock["residual_tail_rate"] < best_base["residual_tail_rate"]
    wf_good = sum(1 for f in payload["walk_forward"] if not f.get("skipped") and f.get("budget_30", {}).get("lift_in_rejected_group", 0) > 1.0) >= 2
    ready = (
        (std.get("pr_auc") or 0.0) >= prevalence * 1.5
        and lock["tail_capture_rate"] >= 0.55
        and lock["tail_risk_reduction"] >= 0.35
        and 0.25 <= lock["rejection_rate"] <= 0.35
        and lock["retained_rate"] >= 0.65
        and lock["lift_in_rejected_group"] >= 1.5
        and beats_base
        and wf_good
        and payload["ablation"].get("status") != "VOLATILITY_DEPENDENCE_HIGH"
    )
    if ready:
        return "TRRM_READY_FOR_PHASE_F_RETROSPECTIVE", "strided lockbox meets honest TRRM operating criteria", "FASE-F - retrospective validation against Phase O/live trade IDs, research-only."
    if beats_base and lock["lift_in_rejected_group"] > 1.0:
        return "TRRM_PROMISING_RESEARCH_ONLY", "model shows real lift but misses one or more readiness criteria", "FASE-E2.1 for robustness/calibration."
    if not beats_base:
        return "HONEST_CAUSAL_BASELINE_NOT_BEATEN", "selected model did not beat the best causal baseline on lockbox", "Review feature families or consider simple causal rules."
    return "RESEARCH_NOT_READY", "signal is weak or unstable under the frozen operating point", "Run targeted diagnostics; do not add features randomly."


def write_predictions(path: Path, df: pd.DataFrame, idx: np.ndarray, y: pd.Series, score: np.ndarray, threshold: float) -> None:
    cols = [c for c in ("id.symbol", "id.timestamp", "id.timeframe", "id.horizon") if c in df.columns]
    out = df.iloc[idx][cols].copy()
    out[TARGET] = y.iloc[idx].to_numpy(int)
    out["risk_probability"] = score
    out["rejected_at_frozen_threshold"] = (score >= threshold).astype(int)
    out.to_csv(path, index=False)


def render_markdown(payload: dict[str, Any]) -> str:
    selected = payload.get("selected_candidate") or {}
    lock = payload.get("lockbox", {})
    lines = [
        "# FASE-E2 Honest TRRM Evaluation",
        "",
        "## 1. Estado",
        f"- decision: {payload['decision']}",
        f"- reason: {payload['decision_reason']}",
        "- mode: research-only",
        "",
        "## 2. Input FASE-D2",
        f"- dense: {payload['paths']['dense_csv']}",
        f"- strided: {payload['paths']['strided_csv']}",
        f"- d2_json: {payload['paths']['d2_report_json']}",
        f"- target: {payload['target']['name']}",
        f"- dense_rows: {payload['dataset']['dense_rows']}",
        f"- strided_rows: {payload['dataset']['strided_rows']}",
        "",
        "## 3. Dataset integrity",
        f"- status: {payload['integrity']['status']}",
        f"- duplicate removals dense/strided: {payload['integrity']['dense_duplicates']['exact_duplicates_removed']} / {payload['integrity']['strided_duplicates']['exact_duplicates_removed']}",
        f"- feature_count: {payload['features']['eligible_count']}",
        f"- leakage_risk: {payload['leakage_risk']}",
        "",
        "## 4. Target congelado",
        f"- definition: future_mae_roe_proxy >= 0.300 ROE",
        f"- train_rate: {payload['target']['rates']['train']:.6f}",
        f"- validation_rate: {payload['target']['rates']['validation']:.6f}",
        f"- lockbox_rate: {payload['target']['rates']['lockbox']:.6f}",
        "",
        "## 5. Features elegibles",
        f"- global causal: {payload['features']['eligible_count']}",
        "- global causal plus horizon: enabled",
        f"- excluded: {len(payload['features']['excluded'])}",
        f"- manual_review: {len(payload['features']['manual_review'])}",
        "",
        "## 6. Splits and embargo",
        f"- train: {payload['splits']['dense']['periods']['train_start']} -> {payload['splits']['dense']['periods']['train_end']}",
        f"- validation: {payload['splits']['dense']['periods']['validation_start']} -> {payload['splits']['dense']['periods']['validation_end']}",
        f"- lockbox: {payload['splits']['dense']['periods']['lockbox_start']} -> {payload['splits']['dense']['periods']['lockbox_end']}",
        f"- embargo_minutes: {payload['splits']['embargo_minutes']}",
        f"- overlap_ok: {payload['split_checks']['overlap_ok']}",
        "",
        "## 7. Honest causal baselines",
    ]
    for name, data in payload["baselines"].items():
        if name == "diagnostic_oracle_upper_bound":
            continue
        b = data["lockbox"]["budget_30_frozen_threshold"]
        lines.append(f"- {name}: PR-AUC={data['lockbox']['standard'].get('pr_auc')} lift@30={b['lift_in_rejected_group']:.3f} capture@30={b['tail_capture_rate']:.3f} residual={b['residual_tail_rate']:.4f} reject={b['rejection_rate']:.3f}")
    lines += [
        "",
        "## 8. Oracle diagnostic-only",
        "- diagnostic_oracle_upper_bound is not selectable, not promotable, non-causal, and not live-usable.",
        "",
        "## 9. Models trained",
    ]
    for row in payload["models_trained"]:
        lines.append(f"- {row['name']}: skipped={row.get('skipped', False)} seconds={row.get('train_seconds')}")
    lines += [
        "",
        "## 10. Validation model selection",
        f"- selected: {selected.get('name')}",
        f"- design: {selected.get('design')}",
        f"- calibration: {selected.get('calibration')}",
        f"- threshold: {selected.get('threshold')}",
        f"- budget: {PRIMARY_BUDGET}",
        "",
        "## 11. Walk-forward robustness",
        f"- folds: {len(payload['walk_forward'])}",
        "",
        "## 12. Lockbox results",
        f"- primary_strided: {lock.get('strided_standard')}",
        f"- secondary_dense: {lock.get('dense_standard')}",
        "",
        "## 13. Rejection-budget analysis",
        f"- strided_lockbox: {lock.get('strided_budget_table')}",
        "",
        "## 14. Comparison against causal baselines",
        f"- best_causal_baseline: {payload['comparison']['best_causal_baseline_name']}",
        f"- conclusion: {payload['comparison']['conclusion']}",
        "",
        "## 15. Per-symbol results",
        json.dumps(payload["segments"]["per_symbol"][:20], indent=2, default=json_default),
        "",
        "## 16. Per-horizon results",
        json.dumps(payload["segments"]["per_horizon"], indent=2, default=json_default),
        "",
        "## 17. Per-symbol+horizon results",
        json.dumps(payload["segments"]["per_symbol_horizon"][:40], indent=2, default=json_default),
        "",
        "## 18. Monthly lockbox stability",
        json.dumps(payload["segments"]["monthly_lockbox"], indent=2, default=json_default),
        "",
        "## 19. Symbol-holdout diagnostics",
        json.dumps(payload["symbol_holdout"], indent=2, default=json_default),
        "",
        "## 20. Ablations",
        json.dumps(payload["ablation"], indent=2, default=json_default),
        "",
        "## 21. Feature importance",
        json.dumps(payload["feature_importance"][:30], indent=2, default=json_default),
        "",
        "## 22. Calibration",
        json.dumps(lock.get("strided_standard", {}).get("calibration_bins", []), indent=2, default=json_default),
        "",
        "## 23. Comparison against FASE-C/E",
        "FASE-C/E used an oracle-like label-derived baseline as a benchmark. E2 keeps that oracle diagnostic-only and compares only against causal baselines selected without lockbox.",
        "",
        "## 24. Limitations",
        "\n".join(f"- {x}" for x in payload["limitations"]),
        "",
        "## 25. Decision",
        f"- {payload['decision']}: {payload['decision_reason']}",
        "",
        "## 26. Recommended next phase",
        f"- {payload['recommended_next_phase']}",
        "",
        "## 27. Safety confirmations",
        "- No live changes. No active_manifest. No YAML live. No PM2 restart. No orders. No .env. No TS changes. No push. No promotion. No shadow. Oracle not used for selection. Lockbox not used for model/threshold selection. No future/label features.",
        "",
    ]
    return "\n".join(lines)


def validate_research_path(path: Path) -> None:
    resolved = path.resolve()
    if REPO in resolved.parents and "active" in resolved.parts:
        raise ValueError(f"refusing active/live path: {resolved}")
    if Path("/tmp") in resolved.parents or resolved == Path("/tmp"):
        return
    if RESEARCH_ROOT not in resolved.parents and resolved != RESEARCH_ROOT:
        raise ValueError(f"refusing non-research output path: {resolved}")


def run_e2(args: argparse.Namespace) -> dict[str, Any]:
    stamp = utc_stamp()
    dense_path = Path(args.dense_csv)
    strided_path = Path(args.strided_csv)
    d2_json = Path(args.d2_report_json)
    out_dir = Path(args.output_dir)
    model_root = Path(args.model_output_dir)
    validate_research_path(out_dir)
    validate_research_path(model_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.target != TARGET:
        raise ValueError(f"FASE-E2 target is frozen to {TARGET}")
    if not d2_json.exists():
        raise FileNotFoundError(str(d2_json))
    d2 = json.loads(d2_json.read_text())
    if "20260710T051035Z" not in str(d2):
        raise ValueError("D2 report JSON does not appear to match run 20260710T051035Z")

    dense, dense_dup = dedupe_dataset(load_csv(dense_path))
    strided, strided_dup = dedupe_dataset(load_csv(strided_path))
    features, excluded, manual = eligible_features(dense)
    leakage_risk = any(not c.startswith("feature.") or c == "feature.close" or c.startswith(("target.", "label.", "future_eval.", "reference.", "id.")) for c in features)
    if len(features) < 100:
        raise ValueError("DATASET_INTEGRITY_ERROR: feature_count below 100")
    x_dense = dense[features].apply(pd.to_numeric, errors="coerce")
    x_strided = strided[features].apply(pd.to_numeric, errors="coerce")
    y_dense = target_values(dense, TARGET)
    y_strided = target_values(strided, TARGET)
    dense_split = make_split(dense, args.embargo_minutes)
    strided_split = make_split(strided, args.embargo_minutes)
    if not dense_split.overlap_ok or not strided_split.overlap_ok:
        leakage_risk = True

    constant_features = [c for c in features if x_dense[c].nunique(dropna=True) <= 1]
    missing_rates = x_dense.isna().mean()
    if constant_features:
        leakage_risk = True

    baselines_raw = baseline_scores(dense, y_dense, dense_split)
    baselines_raw_strided = baseline_scores(strided, y_strided, strided_split)
    baseline_eval = evaluate_baseline_set(baselines_raw, y_dense, dense_split)
    assert not baseline_eval["diagnostic_oracle_upper_bound"]["eligible_for_selection"]

    model_results: list[dict[str, Any]] = []
    models_trained: list[dict[str, Any]] = []
    for design in ("GLOBAL_CAUSAL", "GLOBAL_CAUSAL_PLUS_HORIZON"):
        base_x = x_dense
        for name, (kind, model) in available_models(args.seed).items():
            result = train_one_candidate(dense, y_dense, dense_split, base_x, features, design, kind, name, model, args.seed, args.max_train_rows)
            for item in result["items"]:
                if item.get("skipped"):
                    model_results.append(item)
                    models_trained.append({"name": item["name"], "skipped": True, "reason": item["reason"]})
                    continue
                model_results.append(item)
                models_trained.append({"name": item["name"], "skipped": False, "train_seconds": item["train_seconds"], "design": item["design"], "model": item["model"], "calibration": item["calibration"]})

    selected = select_candidate(model_results, baseline_eval)
    if selected and not selected.get("beats_validation_causal_baseline"):
        selected = None

    if selected:
        lock_eval = evaluate_selected_on_lockbox(selected, dense, y_dense, dense_split, strided, y_strided, strided_split, x_strided)
        selected_clean = {k: v for k, v in selected.items() if k != "fit"}
    else:
        zeros_s = np.zeros(len(strided_split.lockbox_idx))
        zeros_d = np.zeros(len(dense_split.lockbox_idx))
        lock_eval = {
            "strided_score": zeros_s,
            "dense_score": zeros_d,
            "strided_standard": standard_metrics(y_strided.iloc[strided_split.lockbox_idx].to_numpy(int), zeros_s, 0.5),
            "strided_budget_30": budget_metrics(y_strided.iloc[strided_split.lockbox_idx].to_numpy(int), zeros_s, PRIMARY_BUDGET, 0.5),
            "strided_budget_table": rejection_budget_table(y_strided.iloc[strided_split.lockbox_idx].to_numpy(int), zeros_s),
            "dense_standard": standard_metrics(y_dense.iloc[dense_split.lockbox_idx].to_numpy(int), zeros_d, 0.5),
            "dense_budget_30": budget_metrics(y_dense.iloc[dense_split.lockbox_idx].to_numpy(int), zeros_d, PRIMARY_BUDGET, 0.5),
            "dense_budget_table": rejection_budget_table(y_dense.iloc[dense_split.lockbox_idx].to_numpy(int), zeros_d),
        }
        selected_clean = None

    eligible_baseline_names = [
        k for k, v in baseline_eval.items()
        if v.get("eligible_for_selection") and v["validation"]["budget_30"]["rejection_rate"] <= 0.85
    ]
    best_base_name = max(
        eligible_baseline_names,
        key=lambda k: (
            baseline_eval[k]["lockbox"]["budget_30_frozen_threshold"]["tail_capture_rate"],
            -baseline_eval[k]["lockbox"]["budget_30_frozen_threshold"]["residual_tail_rate"],
        ),
    ) if eligible_baseline_names else "causal_atr_range_percentile_baseline"
    best_base_budget = baseline_eval[best_base_name]["lockbox"]["budget_30_frozen_threshold"]
    comparison = {
        "best_causal_baseline_name": best_base_name,
        "best_causal_baseline_lockbox_budget_30": best_base_budget,
        "conclusion": "model_evaluated" if selected else "no_model_selected_over_causal_baseline_on_validation",
    }
    model_candidates_summary = []
    for item in model_results:
        if item.get("skipped"):
            model_candidates_summary.append({"name": item["name"], "skipped": True, "reason": item.get("reason")})
        else:
            model_candidates_summary.append({
                "name": item["name"],
                "design": item["design"],
                "model": item["model"],
                "calibration": item["calibration"],
                "train_seconds": item["train_seconds"],
                "validation_standard": item["validation"]["standard"],
                "validation_budget_30": item["validation"]["budget_30"],
            })

    segments = {"per_symbol": [], "per_horizon": [], "per_symbol_horizon": [], "monthly_lockbox": []}
    feature_importance: list[dict[str, Any]] = []
    wf: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    ablation: dict[str, Any] = {"status": "NOT_RUN"}
    if selected:
        threshold = selected["threshold"]
        ys = y_strided.iloc[strided_split.lockbox_idx].to_numpy(int)
        ss = lock_eval["strided_score"]
        segments = {
            "per_symbol": segment_metrics(strided, strided_split.lockbox_idx, ys, ss, threshold, ["id.symbol"]),
            "per_horizon": segment_metrics(strided, strided_split.lockbox_idx, ys, ss, threshold, ["id.horizon"]),
            "per_symbol_horizon": segment_metrics(strided, strided_split.lockbox_idx, ys, ss, threshold, ["id.symbol", "id.horizon"]),
            "monthly_lockbox": month_metrics(strided, strided_split.lockbox_idx, ys, ss, threshold),
        }
        wf = walk_forward_diagnostics(dense, x_dense, y_dense, selected, args.seed)
        holdout = symbol_holdout(dense, x_dense, y_dense, selected, dense_split, args.seed)
        ablation = ablation_diagnostics(dense, x_dense, y_dense, selected, dense_split, args.seed)
        feature_importance = top_importance(selected["fit"]["model"], selected["feature_columns"])

    limitations = [
        "Target remains fixed across horizons; horizon 6 has known lower prevalence.",
        "Models are small predefined sklearn configurations, not a broad hyperparameter search.",
        "Dense lockbox is secondary because dense windows are highly overlapping.",
        "No live, shadow, or guard integration is performed.",
    ]
    payload: dict[str, Any] = {
        "schema_version": "phase_e2_honest_trrm_v1",
        "generated_at": stamp,
        "mode": "research-only",
        "paths": {"dense_csv": str(dense_path), "strided_csv": str(strided_path), "d2_report_json": str(d2_json)},
        "dataset": {
            "dense_rows": int(len(dense)),
            "strided_rows": int(len(strided)),
            "symbols": sorted(dense["id.symbol"].astype(str).unique().tolist()),
            "horizons": sorted(pd.to_numeric(dense["id.horizon"], errors="coerce").dropna().astype(int).unique().tolist()),
            "timeframe": sorted(dense["id.timeframe"].astype(str).unique().tolist()) if "id.timeframe" in dense else [],
            "timestamp_min": str(dense["id.timestamp"].min()),
            "timestamp_max": str(dense["id.timestamp"].max()),
        },
        "integrity": {
            "status": "OK",
            "dense_duplicates": dense_dup,
            "strided_duplicates": strided_dup,
            "constant_features": constant_features,
            "mean_missing_rate": float(missing_rates.mean()),
            "high_missing_features": missing_rates[missing_rates > 0.40].index.tolist(),
        },
        "features": {"eligible_count": len(features), "eligible_feature_columns": features, "excluded": excluded, "manual_review": manual},
        "leakage_risk": bool(leakage_risk),
        "target": {
            "name": TARGET,
            "definition": "future_mae_roe_proxy >= 0.300 ROE",
            "rates": {
                "global": float(y_dense.mean()),
                "train": float(y_dense.iloc[dense_split.train_idx].mean()),
                "validation": float(y_dense.iloc[dense_split.validation_idx].mean()),
                "lockbox": float(y_dense.iloc[dense_split.lockbox_idx].mean()),
                "strided_lockbox": float(y_strided.iloc[strided_split.lockbox_idx].mean()),
            },
        },
        "splits": {
            "embargo_minutes": args.embargo_minutes,
            "dense": {"train_rows": len(dense_split.train_idx), "validation_rows": len(dense_split.validation_idx), "lockbox_rows": len(dense_split.lockbox_idx), "rows_purged": dense_split.rows_purged, "rows_embargoed": dense_split.rows_embargoed, "periods": dense_split.periods},
            "strided": {"train_rows": len(strided_split.train_idx), "validation_rows": len(strided_split.validation_idx), "lockbox_rows": len(strided_split.lockbox_idx), "rows_purged": strided_split.rows_purged, "rows_embargoed": strided_split.rows_embargoed, "periods": strided_split.periods},
        },
        "split_checks": {"overlap_ok": bool(dense_split.overlap_ok and strided_split.overlap_ok), "lockbox_opened_once": True, "lockbox_used_for_selection": False},
        "baselines": baseline_eval,
        "models_trained": models_trained,
        "model_candidates_summary": model_candidates_summary,
        "selected_candidate": selected_clean,
        "lockbox": {k: v for k, v in lock_eval.items() if not k.endswith("_score")},
        "comparison": comparison,
        "segments": segments,
        "walk_forward": wf,
        "symbol_holdout": holdout,
        "ablation": ablation,
        "feature_importance": feature_importance,
        "limitations": limitations,
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
            "oracle_not_used_for_selection": True,
            "lockbox_not_used_for_selection": True,
            "no_future_or_label_features": True,
            "research_only_artifacts": True,
        },
    }
    dec, reason, next_phase = decision(payload)
    payload["decision"] = dec
    payload["decision_reason"] = reason
    payload["recommended_next_phase"] = next_phase

    if parse_bool(args.write_report):
        md_path = out_dir / f"aegis_phase_e2_trrm_honest_{stamp}.md"
        json_path = out_dir / f"aegis_phase_e2_trrm_honest_{stamp}.json"
        val_pred = out_dir / f"aegis_phase_e2_validation_predictions_{stamp}.csv"
        lock_pred = out_dir / f"aegis_phase_e2_lockbox_predictions_strided_{stamp}.csv"
        dense_pred = out_dir / f"aegis_phase_e2_lockbox_predictions_dense_{stamp}.csv"
        payload["artifacts"] = {"markdown": str(md_path), "json": str(json_path), "validation_predictions": str(val_pred), "lockbox_predictions_strided": str(lock_pred), "lockbox_predictions_dense": str(dense_pred)}
        if selected:
            val_score = selected["fit"]["variants"][selected["calibration"]]["validation_score"]
            write_predictions(val_pred, dense, dense_split.validation_idx, y_dense, val_score, selected["threshold"])
            write_predictions(lock_pred, strided, strided_split.lockbox_idx, y_strided, lock_eval["strided_score"], selected["threshold"])
            write_predictions(dense_pred, dense, dense_split.lockbox_idx, y_dense, lock_eval["dense_score"], selected["threshold"])
        else:
            ref = baselines_raw[best_base_name]["score"]
            ref_strided = baselines_raw_strided.get(best_base_name, {"score": np.zeros(len(strided))})["score"]
            threshold = baseline_eval[best_base_name]["selected_threshold_from_validation"]
            write_predictions(val_pred, dense, dense_split.validation_idx, y_dense, ref[dense_split.validation_idx], threshold)
            write_predictions(lock_pred, strided, strided_split.lockbox_idx, y_strided, ref_strided[strided_split.lockbox_idx], threshold)
            write_predictions(dense_pred, dense, dense_split.lockbox_idx, y_dense, ref[dense_split.lockbox_idx], threshold)
        json_path.write_text(json.dumps(payload, indent=2, default=json_default))
        md_path.write_text(render_markdown(payload))

    if selected and parse_bool(args.write_models):
        model_dir = model_root / stamp
        validate_research_path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "selected_pipeline.pkl").open("wb") as f:
            pickle.dump({
                "model": selected["fit"]["model"],
                "imputer": selected["fit"]["imputer"],
                "scaler": selected["fit"]["scaler"],
                "calibrator": selected["fit"]["variants"][selected["calibration"]]["calibrator"],
                "features": selected["feature_columns"],
                "target": TARGET,
                "threshold": selected["threshold"],
                "operating_budget": PRIMARY_BUDGET,
                "mode": "research-only",
            }, f)
        (model_dir / "metadata.json").write_text(json.dumps({k: payload[k] for k in ("schema_version", "generated_at", "decision", "target", "splits", "features", "selected_candidate", "comparison", "safety_confirmations")}, indent=2, default=json_default))
        payload["model_research_path"] = str(model_dir)
        if parse_bool(args.write_report):
            Path(payload["artifacts"]["json"]).write_text(json.dumps(payload, indent=2, default=json_default))

    print(json.dumps({"decision": payload["decision"], "markdown": payload.get("artifacts", {}).get("markdown"), "json": payload.get("artifacts", {}).get("json"), "model_research_path": payload.get("model_research_path"), "selected": (payload.get("selected_candidate") or {}).get("name")}, indent=2, default=json_default))
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        run_e2(parse_args(argv))
        return 0
    except ValueError as exc:
        if "DATASET_INTEGRITY_ERROR" in str(exc):
            print(json.dumps({"decision": "DATASET_INTEGRITY_ERROR", "reason": str(exc)}))
            return 2
        print(json.dumps({"decision": "RESEARCH_NOT_READY", "reason": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
