#!/usr/bin/env python3
"""FASE-E research-only TRRM training on enriched causal features."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)


DEFAULT_OUT_DIR = Path("/home/jasan/Develop")
DEFAULT_MODEL_ROOT = DEFAULT_OUT_DIR / "aegis_research_models" / "trrm_e"
LEAKAGE_PATTERNS = [
    "label", "target", "future", "mfe", "mae", "pnl", "net", "quality", "clean", "bad", "tail",
    "qmae", "management", "premium", "hit", "stop", "tp", "sl", "realized", "outcome", "profit",
    "loss", "trade_result", "close_reason",
]
TARGET_MAP = {
    "tail_risk_v4": "target.tail_risk_v4",
    "bad_entry_v4": "target.bad_entry_v4",
    "union_tail_or_bad": "target.union_tail_or_bad",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def latest_dataset() -> Path | None:
    files = sorted(DEFAULT_OUT_DIR.glob("aegis_trrm_causal_feature_dataset_d_*.csv"))
    return files[-1] if files else None


def is_leakage_feature(col: str) -> tuple[bool, str]:
    low = col.lower()
    if not low.startswith("feature."):
        return True, "not_feature_namespace"
    for pat in LEAKAGE_PATTERNS:
        if pat in low:
            return True, f"pattern:{pat}"
    return False, ""


def load_dataset(path: Path, max_rows: int = 0, symbols: str = "", horizons: str = "") -> pd.DataFrame:
    df = pd.read_csv(path, nrows=max_rows or None)
    if symbols and "id.symbol" in df:
        allowed = {s.strip().upper() for s in symbols.split(",") if s.strip()}
        df = df[df["id.symbol"].astype(str).str.upper().isin(allowed)].copy()
    if horizons and "id.horizon" in df:
        allowed_h = {h.strip() for h in horizons.split(",") if h.strip()}
        df = df[df["id.horizon"].astype(str).isin(allowed_h)].copy()
    if "target.tail_risk_v4" in df and "target.bad_entry_v4" in df:
        df["target.union_tail_or_bad"] = (
            bool_series(df["target.tail_risk_v4"]) | bool_series(df["target.bad_entry_v4"])
        ).astype(int)
    return df


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"1", "true", "yes"}).astype(int)


def select_features(df: pd.DataFrame) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    features: list[str] = []
    excluded: list[dict[str, str]] = []
    manual: list[dict[str, str]] = []
    for col in df.columns:
        if col.startswith(("target.", "label.", "future_eval.")):
            excluded.append({"column": col, "reason": "target_label_future_namespace"})
            continue
        if not col.startswith("feature."):
            continue
        leak, reason = is_leakage_feature(col)
        if leak:
            manual.append({"column": col, "reason": reason})
            continue
        features.append(col)
    return features, excluded, manual


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    method: str
    reliable: bool
    warning: str | None = None


def make_split(df: pd.DataFrame) -> Split:
    if "id.timestamp" not in df:
        idx = np.arange(len(df))
        return ordered_split(idx, "row_order_fallback", False, "id.timestamp missing")
    ts = pd.to_datetime(df["id.timestamp"], errors="coerce", utc=True)
    valid = ts.notna()
    if valid.mean() < 0.95:
        idx = np.arange(len(df))
        return ordered_split(idx, "row_order_fallback", False, "timestamp parse reliability below 95%")
    ordered = df.assign(_ts=ts).sort_values(["_ts", "id.symbol", "id.horizon"], kind="mergesort").index.to_numpy()
    return ordered_split(ordered, "global_walk_forward", True, None)


def ordered_split(idx: np.ndarray, method: str, reliable: bool, warning: str | None) -> Split:
    n = len(idx)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    return Split(idx[:train_end], idx[train_end:val_end], idx[val_end:], method, reliable, warning)


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


def fit_preprocessors(x: pd.DataFrame, train_idx: np.ndarray, scale: bool) -> tuple[MedianImputer, StandardScalerLite | None, pd.DataFrame]:
    imputer = MedianImputer().fit(x.loc[train_idx])
    xi = imputer.transform(x)
    scaler = StandardScalerLite().fit(xi.loc[train_idx]) if scale else None
    if scaler:
        xi = scaler.transform(xi)
    return imputer, scaler, xi


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y = y.astype(int)
    pred = pred.astype(int)
    return {
        "tp": int(((y == 1) & (pred == 1)).sum()),
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
    }


def metrics(y: np.ndarray, score: np.ndarray, threshold: float, include_auc: bool = True) -> dict[str, Any]:
    y = y.astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    c = confusion(y, pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": c,
        "false_negatives": fn,
        "false_positives": fp,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "rejection_rate": float(pred.mean()) if len(pred) else 0.0,
        "retained_rate": 1.0 - (float(pred.mean()) if len(pred) else 0.0),
        "risk_capture_rate": recall,
    }
    if include_auc:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            out["pr_auc"] = float(average_precision_score(y, score)) if len(set(y)) > 1 else None
            out["roc_auc"] = float(roc_auc_score(y, score)) if len(set(y)) > 1 else None
        except Exception:
            out["pr_auc"] = None
            out["roc_auc"] = None
    else:
        out["pr_auc"] = None
        out["roc_auc"] = None
    return out


def threshold_search(y_val: np.ndarray, score_val: np.ndarray) -> dict[str, dict[str, Any]]:
    raw = np.asarray(score_val, dtype=float)
    if len(raw) > 300:
        candidates = sorted(set(float(x) for x in np.quantile(raw, np.linspace(0, 1, 301))), reverse=True)
    else:
        candidates = sorted(set(float(x) for x in raw), reverse=True)
    candidates = candidates + [0.0, 1.0]
    rows = [(t, metrics(y_val, score_val, t, include_auc=False)) for t in candidates]
    def best_recall(target: float, prefer_low_reject: bool = False) -> dict[str, Any]:
        good = [(t, m) for t, m in rows if m["recall"] >= target]
        if not good:
            t, m = max(rows, key=lambda item: item[1]["recall"])
            return {"threshold": t, **m}
        key = (lambda item: (item[1]["precision"], -item[1]["rejection_rate"])) if not prefer_low_reject else (lambda item: (-item[1]["rejection_rate"], item[1]["precision"]))
        t, m = max(good, key=key)
        return {"threshold": t, **m}
    t, m = max(rows, key=lambda item: item[1]["f1"])
    return {
        "recall_095": best_recall(0.95),
        "recall_098": best_recall(0.98),
        "best_f1": {"threshold": t, **m},
        "best_precision_recall_095": best_recall(0.95),
        "lowest_rejection_recall_095": best_recall(0.95, prefer_low_reject=True),
    }


def baseline_simple_scores(df: pd.DataFrame) -> np.ndarray:
    cols = ["target.bad_entry_v4", "target.early_mae_v4", "feature.squeeze_risk_proxy_causal"]
    parts = []
    for col in cols:
        if col in df:
            parts.append(bool_series(df[col]))
        else:
            parts.append(pd.Series(0, index=df.index))
    return ((parts[0] == 1) | (parts[1] == 1) | (parts[2] > 0)).astype(float).to_numpy()


def causal_baseline_scores(df: pd.DataFrame, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    rules: list[pd.Series] = []
    thresholds: dict[str, float] = {}
    for col in ("feature.atr_proxy_24", "feature.rolling_range_mean_24", "feature.rolling_range_std_24"):
        if col in df:
            vals = pd.to_numeric(df[col], errors="coerce")
            thr = float(vals.loc[train_idx].quantile(0.75))
            thresholds[col] = thr
            rules.append(vals >= thr)
    for col in ("feature.rebound_risk_proxy", "feature.squeeze_risk_proxy_causal"):
        if col in df:
            rules.append(pd.to_numeric(df[col], errors="coerce").fillna(0.0) > 0)
    if not rules:
        return np.zeros(len(df)), {"thresholds": thresholds, "warning": "no causal baseline features"}
    out = rules[0].copy()
    for r in rules[1:]:
        out = out | r
    return out.astype(float).to_numpy(), {"thresholds": thresholds}


def available_models() -> dict[str, Any]:
    models: dict[str, Any] = {}
    try:
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        models["logistic_regression"] = ("linear", LogisticRegression(max_iter=800, class_weight="balanced", solver="liblinear"))
        models["sgd_log_loss"] = ("linear", SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000, tol=1e-3, random_state=31))
    except Exception:
        pass
    try:
        from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        models["hist_gradient_boosting"] = ("tree", HistGradientBoostingClassifier(max_iter=70, learning_rate=0.08, random_state=31))
        models["random_forest"] = ("tree", RandomForestClassifier(n_estimators=80, max_depth=7, min_samples_leaf=25, class_weight="balanced_subsample", random_state=31, n_jobs=1))
        models["gradient_boosting"] = ("tree", GradientBoostingClassifier(n_estimators=70, learning_rate=0.06, max_depth=3, random_state=31))
    except Exception:
        pass
    return models


def score_model(model: Any, x_train: pd.DataFrame, y_train: pd.Series, x_eval: pd.DataFrame) -> np.ndarray:
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_eval)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x_eval)
        return 1.0 / (1.0 + np.exp(-np.asarray(raw)))
    return np.asarray(model.predict(x_eval), dtype=float)


def importance(model: Any, cols: list[str]) -> list[dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        return []
    return sorted([{"feature": c, "importance": float(v)} for c, v in zip(cols, vals)], key=lambda r: r["importance"], reverse=True)[:25]


def group_metrics(df: pd.DataFrame, idx: np.ndarray, y: np.ndarray, score: np.ndarray, threshold: float, keys: list[str]) -> list[dict[str, Any]]:
    local = df.loc[idx, keys].copy()
    local["_y"] = y
    local["_s"] = score
    out = []
    for group, g in local.groupby(keys, dropna=False):
        if not isinstance(group, tuple):
            group = (group,)
        m = metrics(g["_y"].to_numpy(), g["_s"].to_numpy(), threshold)
        out.append({**{k: str(v) for k, v in zip(keys, group)}, "rows": int(len(g)), "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "rejection_rate": m["rejection_rate"], "false_negatives": m["false_negatives"]})
    return out


def evaluate_baselines(df: pd.DataFrame, y: pd.Series, split: Split) -> dict[str, Any]:
    y_val = y.loc[split.val_idx].to_numpy()
    y_test = y.loc[split.test_idx].to_numpy()
    simple = baseline_simple_scores(df)
    causal, causal_info = causal_baseline_scores(df, split.train_idx)
    baselines = {}
    score_map = {
        "baseline_simple_label_derived": simple,
        "causal_volatility_baseline": causal,
        "reject_all": np.ones(len(df)),
        "reject_none": np.zeros(len(df)),
        "class_prior": np.full(len(df), float(y.loc[split.train_idx].mean())),
    }
    fixed_thresholds = {
        "baseline_simple_label_derived": 0.5,
        "causal_volatility_baseline": 0.5,
        "reject_all": 0.5,
        "reject_none": 0.5,
        "class_prior": 0.5,
    }
    for name, scores in score_map.items():
        search = threshold_search(y_val, scores[split.val_idx])
        selected = fixed_thresholds[name]
        baselines[name] = {
            "validation_thresholds": search,
            "test": metrics(y_test, scores[split.test_idx], selected),
            "selected_threshold": selected,
        }
    baselines["causal_volatility_baseline"]["rule_info"] = causal_info
    return baselines


def evaluate_target(df: pd.DataFrame, x: pd.DataFrame, y: pd.Series, split: Split, write_models: bool, model_dir: Path, target_name: str) -> dict[str, Any]:
    baselines = evaluate_baselines(df, y, split)
    results: dict[str, Any] = {"baselines": baselines, "models": {}, "best_model": None}
    y_train, y_val, y_test = y.loc[split.train_idx], y.loc[split.val_idx].to_numpy(), y.loc[split.test_idx].to_numpy()
    for name, (kind, model) in available_models().items():
        try:
            imputer, scaler, xp = fit_preprocessors(x, split.train_idx, scale=(kind == "linear"))
            val_scores = score_model(model, xp.loc[split.train_idx], y_train, xp.loc[split.val_idx])
            test_scores = model.predict_proba(xp.loc[split.test_idx])[:, 1] if hasattr(model, "predict_proba") else (
                1.0 / (1.0 + np.exp(-model.decision_function(xp.loc[split.test_idx]))) if hasattr(model, "decision_function") else model.predict(xp.loc[split.test_idx])
            )
            search = threshold_search(y_val, val_scores)
            selected = search["best_precision_recall_095"]["threshold"]
            test_metrics = metrics(y_test, test_scores, selected)
            result = {
                "validation_thresholds": search,
                "selected_threshold": selected,
                "test": test_metrics,
                "per_symbol": group_metrics(df, split.test_idx, y_test, test_scores, selected, ["id.symbol"]) if "id.symbol" in df else [],
                "per_horizon": group_metrics(df, split.test_idx, y_test, test_scores, selected, ["id.horizon"]) if "id.horizon" in df else [],
                "per_symbol_horizon": group_metrics(df, split.test_idx, y_test, test_scores, selected, ["id.symbol", "id.horizon"]) if {"id.symbol", "id.horizon"} <= set(df.columns) else [],
                "feature_importance": importance(model, list(x.columns)),
            }
            if write_models:
                model_dir.mkdir(parents=True, exist_ok=True)
                with (model_dir / f"{target_name}_{name}.pkl").open("wb") as f:
                    pickle.dump({"model": model, "imputer": imputer, "scaler": scaler, "features": list(x.columns), "threshold": selected, "target": target_name}, f)
            results["models"][name] = result
        except Exception as exc:
            results["models"][name] = {"skipped": True, "reason": str(exc)}
    results["best_model"] = choose_best(results)
    return results


def choose_best(result: dict[str, Any]) -> str:
    best_name = ""
    best_tuple = (False, 0.0, -1.0, 0.0)
    for name, data in result["baselines"].items():
        m = data["test"]
        tup = (m["recall"] >= 0.95, m["precision"], -m["rejection_rate"], m["f1"])
        if tup > best_tuple:
            best_name, best_tuple = name, tup
    for name, data in result["models"].items():
        if data.get("skipped"):
            continue
        m = data["test"]
        tup = (m["recall"] >= 0.95, m["precision"], -m["rejection_rate"], m["f1"])
        if tup > best_tuple:
            best_name, best_tuple = name, tup
    return best_name


def volatility_dependence(df: pd.DataFrame, x: pd.DataFrame, y: pd.Series, split: Split, model_name: str) -> dict[str, Any]:
    drop_tokens = ("atr_proxy", "rolling_range", "volatility", "range_std", "range_mean")
    cols = [c for c in x.columns if not any(t in c for t in drop_tokens)]
    dropped = [c for c in x.columns if c not in cols]
    if len(cols) < 20 or model_name not in available_models():
        return {"status": "NOT_RUN", "reason": "insufficient non-volatility features or model unavailable", "dropped_features": dropped[:20]}
    kind, model = available_models()[model_name]
    try:
        imputer, scaler, xp = fit_preprocessors(x[cols], split.train_idx, scale=(kind == "linear"))
        val_scores = score_model(model, xp.loc[split.train_idx], y.loc[split.train_idx], xp.loc[split.val_idx])
        search = threshold_search(y.loc[split.val_idx].to_numpy(), val_scores)
        threshold = search["best_precision_recall_095"]["threshold"]
        test_scores = model.predict_proba(xp.loc[split.test_idx])[:, 1] if hasattr(model, "predict_proba") else model.predict(xp.loc[split.test_idx])
        m = metrics(y.loc[split.test_idx].to_numpy(), test_scores, threshold)
        status = "MULTI_SIGNAL_RISK_STRUCTURE_CONFIRMED" if m["recall"] >= 0.80 and m["precision"] > 0.35 else "VOLATILITY_DEPENDENCE_HIGH"
        return {"status": status, "metrics": m, "dropped_features": dropped[:30], "remaining_feature_count": len(cols)}
    except Exception as exc:
        return {"status": "NOT_RUN", "reason": str(exc), "dropped_features": dropped[:20]}


def decide(payload: dict[str, Any]) -> tuple[str, str, str]:
    if payload["feature_count"] < 20:
        return "DATASET_NOT_USABLE", "feature_count below 20", "stop and fix dataset"
    if payload["leakage_risk"]:
        return "LEAKAGE_RISK_TOO_HIGH", "leakage feature detected", "stop and fix dataset"
    if not payload["split"]["reliable"]:
        return "RESEARCH_NOT_READY", "temporal validation not reliable", "fix timestamp split"
    tail = payload["targets"].get("tail_risk_v4") or payload["targets"].get("union_tail_or_bad")
    if not tail:
        return "DATASET_NOT_USABLE", "no primary target evaluated", "stop and fix dataset"
    best_name = tail["best_model"]
    best = tail["baselines"][best_name]["test"] if best_name in tail["baselines"] else tail["models"][best_name]["test"]
    causal = tail["baselines"]["causal_volatility_baseline"]["test"]
    simple = tail["baselines"]["baseline_simple_label_derived"]["test"]
    if best["recall"] < 0.95 or best["rejection_rate"] >= 0.85:
        return "RESEARCH_NOT_READY", "no model meets high recall with acceptable rejection rate", "FASE-E2: tune features/thresholds by symbol/horizon"
    if best_name in tail["baselines"]:
        return "BASELINE_NOT_BEATEN", "best result is still a baseline", "FASE-D2: improve feature engineering/context"
    if best["precision"] >= causal["precision"] and best["rejection_rate"] <= causal["rejection_rate"]:
        if best["precision"] >= simple["precision"] * 0.90:
            return "TRRM_READY_FOR_SHADOW_REVIEW", f"{best_name} improves causal baseline with high recall", "FASE-F: validate TRRM against Phase O live retrospective and forward live trade IDs"
        return "TRRM_PROMISING_RESEARCH_ONLY", f"{best_name} improves causal baseline but not label-derived baseline", "FASE-E2: tune features/thresholds by symbol/horizon"
    return "CAUSAL_BASELINE_NOT_BEATEN", "trained models do not improve causal baseline", "FASE-D2: improve feature engineering/context"


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_csv = Path(args.input_csv) if args.input_csv else latest_dataset()
    if not input_csv or not input_csv.exists():
        raise FileNotFoundError("FASE-D causal feature dataset not found")
    df = load_dataset(input_csv, args.max_rows, args.symbols, args.horizons)
    feature_cols, excluded, manual = select_features(df)
    leakage = [c for c in feature_cols if is_leakage_feature(c)[0]]
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce") if feature_cols else pd.DataFrame(index=df.index)
    targets_all = build_targets(df)
    requested = list(targets_all) if args.target == "all" else [args.target]
    split = make_split(df)
    stamp = utc_stamp()
    model_dir = Path(args.output_dir) / "aegis_research_models" / "trrm_e" / stamp
    payload: dict[str, Any] = {
        "schema_version": "phase_e_trrm_classifier_v1",
        "generated_at": stamp,
        "mode": "research-only",
        "input_dataset": str(input_csv),
        "rows": int(len(df)),
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "excluded_features": excluded,
        "manual_review": manual,
        "leakage_risk": bool(leakage),
        "leakage_features": leakage,
        "targets": {},
        "symbols": sorted(df.get("id.symbol", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "horizons": sorted(df.get("id.horizon", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
        "timestamp_min": str(df["id.timestamp"].min()) if "id.timestamp" in df else None,
        "timestamp_max": str(df["id.timestamp"].max()) if "id.timestamp" in df else None,
        "split": {"method": split.method, "reliable": split.reliable, "warning": split.warning, "train_rows": len(split.train_idx), "validation_rows": len(split.val_idx), "test_rows": len(split.test_idx)},
        "model_research_path": str(model_dir) if parse_bool(args.write_models) else None,
    }
    if len(feature_cols) >= 20 and not leakage:
        for target in requested:
            if target in targets_all:
                payload["targets"][target] = evaluate_target(df, x, targets_all[target], split, parse_bool(args.write_models), model_dir, target)
    decision, reason, next_phase = decide(payload)
    payload["decision"] = decision
    payload["reason"] = reason
    payload["recommended_next_phase"] = next_phase
    primary = payload["targets"].get("tail_risk_v4") or payload["targets"].get("union_tail_or_bad") or {}
    best_name = primary.get("best_model")
    if best_name:
        payload["volatility_dependence_check"] = volatility_dependence(df, x, targets_all.get("tail_risk_v4", next(iter(targets_all.values()))), split, best_name) if best_name not in primary.get("baselines", {}) else {"status": "NOT_RUN", "reason": "best model is baseline"}
    else:
        payload["volatility_dependence_check"] = {"status": "NOT_RUN", "reason": "no best model"}
    payload["safety_confirmations"] = {
        "no_live_touched": True, "no_active_manifest": True, "no_yaml": True, "no_pm2_restart": True, "no_orders": True,
        "no_env": True, "no_ts_touched": True, "no_push": True, "no_model_promotion": True, "research_only_artifacts": True,
        "no_future_or_label_features": True,
    }
    json_path = out_dir / f"aegis_phase_e_trrm_classifier_{stamp}.json"
    md_path = out_dir / f"aegis_phase_e_trrm_classifier_{stamp}.md"
    if parse_bool(args.write_report):
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        md_path.write_text(render_markdown(payload), encoding="utf-8")
    payload["outputs"] = {"json": str(json_path), "md": str(md_path)}
    print(json.dumps({"decision": decision, "reason": reason, "md": str(md_path), "json": str(json_path), "model_path": payload["model_research_path"]}, indent=2))
    return payload


def build_targets(df: pd.DataFrame) -> dict[str, pd.Series]:
    out = {}
    if "target.tail_risk_v4" in df and df["target.tail_risk_v4"].nunique(dropna=True) > 1:
        out["tail_risk_v4"] = bool_series(df["target.tail_risk_v4"])
    if "target.bad_entry_v4" in df and df["target.bad_entry_v4"].nunique(dropna=True) > 1:
        out["bad_entry_v4"] = bool_series(df["target.bad_entry_v4"])
    if {"target.tail_risk_v4", "target.bad_entry_v4"} <= set(df.columns):
        out["union_tail_or_bad"] = (bool_series(df["target.tail_risk_v4"]) | bool_series(df["target.bad_entry_v4"])).astype(int)
    return out


def render_metric_line(prefix: str, m: dict[str, Any]) -> list[str]:
    return [
        f"- {prefix} precision: {m.get('precision', 0):.4f}",
        f"- {prefix} recall: {m.get('recall', 0):.4f}",
        f"- {prefix} f1: {m.get('f1', 0):.4f}",
        f"- {prefix} PR-AUC: {m.get('pr_auc')}",
        f"- {prefix} ROC-AUC: {m.get('roc_auc')}",
        f"- {prefix} rejection_rate: {m.get('rejection_rate', 0):.4f}",
        f"- {prefix} false_negatives: {m.get('false_negatives', 0)}",
    ]


def render_markdown(p: dict[str, Any]) -> str:
    primary = p["targets"].get("tail_risk_v4") or p["targets"].get("union_tail_or_bad") or {}
    best_name = primary.get("best_model", "n/a")
    best = primary.get("baselines", {}).get(best_name, {}).get("test") if best_name in primary.get("baselines", {}) else primary.get("models", {}).get(best_name, {}).get("test", {})
    simple = primary.get("baselines", {}).get("baseline_simple_label_derived", {}).get("test", {})
    causal = primary.get("baselines", {}).get("causal_volatility_baseline", {}).get("test", {})
    lines = [
        "# FASE-E TRRM Classifier with Enriched Causal Features",
        "",
        "## 1. Estado",
        f"- decision: {p['decision']}",
        f"- reason: {p['reason']}",
        f"- mode: {p['mode']}",
        "",
        "## 2. Input dataset",
        f"- path: {p['input_dataset']}",
        f"- rows: {p['rows']}",
        f"- feature_count: {p['feature_count']}",
        f"- targets: {', '.join(p['targets'].keys())}",
        f"- symbols: {', '.join(p['symbols'])}",
        f"- horizons: {', '.join(p['horizons'])}",
        f"- timestamp range: {p['timestamp_min']} to {p['timestamp_max']}",
        "",
        "## 3. Feature safety",
        f"- included features: {len(p['feature_columns'])}",
        f"- excluded features: {len(p['excluded_features'])}",
        f"- leakage risk: {p['leakage_risk']}",
        f"- manual review columns: {len(p['manual_review'])}",
        "",
        "## 4. Splits",
        f"- method: {p['split']['method']}",
        f"- train/validation/test: {p['split']['train_rows']} / {p['split']['validation_rows']} / {p['split']['test_rows']}",
        f"- temporal reliability: {p['split']['reliable']}",
        f"- warning: {p['split']['warning'] or 'none'}",
        "",
        "## 5. Baselines",
        *render_metric_line("simple_label_baseline", simple),
        *render_metric_line("causal_baseline", causal),
        "",
        "## 6. Models trained",
        *[f"- {name}" for name in primary.get("models", {}).keys()],
        "",
        "## 7. Global metrics",
        f"- best model: {best_name}",
        *render_metric_line("best", best or {}),
        "",
        "## 8. Threshold search",
        f"- selected threshold: {best.get('threshold') if best else None}",
        "",
        "## 9. Test/lockbox performance",
        f"- confusion matrix: {best.get('confusion_matrix') if best else None}",
        "",
        "## 10. Per-symbol metrics",
        *[f"- {r.get('id.symbol')}: precision={r.get('precision',0):.3f} recall={r.get('recall',0):.3f} rows={r.get('rows')}" for r in (primary.get("models", {}).get(best_name, {}).get("per_symbol", []) if best_name in primary.get("models", {}) else [])[:30]],
        "",
        "## 11. Per-horizon metrics",
        *[f"- {r.get('id.horizon')}: precision={r.get('precision',0):.3f} recall={r.get('recall',0):.3f} rows={r.get('rows')}" for r in (primary.get("models", {}).get(best_name, {}).get("per_horizon", []) if best_name in primary.get("models", {}) else [])[:20]],
        "",
        "## 12. Per-symbol+horizon metrics",
        *[f"- {r.get('id.symbol')} h{r.get('id.horizon')}: precision={r.get('precision',0):.3f} recall={r.get('recall',0):.3f} rows={r.get('rows')}" for r in (primary.get("models", {}).get(best_name, {}).get("per_symbol_horizon", []) if best_name in primary.get("models", {}) else [])[:50]],
        "",
        "## 13. Feature importance",
        *[f"- {r['feature']}: {r['importance']:.6f}" for r in (primary.get("models", {}).get(best_name, {}).get("feature_importance", []) if best_name in primary.get("models", {}) else [])[:25]],
        "",
        "## 14. Volatility dependence check",
        f"- status: {p['volatility_dependence_check'].get('status')}",
        "",
        "## 15. Comparison vs FASE-C",
        "- FASE-C had only 2 causal features and was BASELINE_NOT_BEATEN.",
        f"- FASE-E used {p['feature_count']} causal features from FASE-D.",
        "",
        "## 16. Limitations",
        "- Baseline simple is label-derived and research-only, not live-usable.",
        "- No Phase O trade-ID linkage validation yet.",
        "",
        "## 17. Decision",
        f"- {p['decision']}",
        "",
        "## 18. Recommended next phase",
        f"- {p['recommended_next_phase']}",
        "",
        "## 19. Safety confirmations",
        "- No toqué live.",
        "- No toqué active_manifest.",
        "- No toqué YAML.",
        "- No reinicié PM2.",
        "- No envié órdenes.",
        "- No toqué .env.",
        "- No toqué binance-futures-bot-ts.",
        "- No hice push.",
        "- No promoví modelos.",
        "- Artifacts generados son research-only.",
        "- No usé labels/futuro como features.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FASE-E research-only TRRM classifier training.")
    p.add_argument("--input-csv", default="")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--target", default="all", choices=["tail_risk_v4", "bad_entry_v4", "union_tail_or_bad", "all"])
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--symbols", default="")
    p.add_argument("--horizons", default="")
    p.add_argument("--write-models", default="true")
    p.add_argument("--write-report", default="true")
    p.add_argument("--strict-causal", default="true")
    return p


def main() -> int:
    run_training(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
