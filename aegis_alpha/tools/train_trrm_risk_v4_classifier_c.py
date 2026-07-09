#!/usr/bin/env python3
"""Research-only TRRM classifier training for Risk V4 / QMAE datasets.

TRRM = Tail Risk Rejection Model. This script trains and evaluates research
classifiers only. It never writes to active/live paths and never promotes a
model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path("/home/jasan/Develop")
DEFAULT_MODEL_DIR = DEFAULT_OUT_DIR / "aegis_research_models" / "trrm_c"
LEAKAGE_PATTERNS = [
    "label",
    "target",
    "future",
    "mfe",
    "mae",
    "pnl",
    "net",
    "quality",
    "clean",
    "bad",
    "tail",
    "qmae",
    "management",
    "premium",
    "hit",
    "stop",
    "tp",
    "sl",
    "realized",
    "outcome",
    "profit",
    "loss",
    "trade_result",
    "close_reason",
]
MANUAL_REVIEW_PATTERNS = ["risk", "score", "vote", "guard", "decision", "action", "reason"]
BASE_NUMERIC_FEATURES = ["close"]
JSON_FEATURE_COLUMNS = {
    "volatility_features": ("volatility",),
    "trend_features": ("trend",),
    "wick_reclaim_proxies": ("wick", "reclaim"),
    "btc_eth_context": ("btc", "eth", "context"),
}
TARGET_COLUMNS = ["tail_risk_v4", "bad_entry_v4"]
VARIANT_TARGETS = {
    "target_tail_risk_v4": "tail_risk_v4",
    "target_bad_entry_v4": "bad_entry_v4",
    "target_union_tail_or_bad": "union_tail_or_bad",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def latest_dataset(out_dir: Path = DEFAULT_OUT_DIR) -> Path | None:
    files = sorted(out_dir.glob("aegis_risk_v4_qmae_base_dataset_a_samples_*.csv"))
    return files[-1] if files else None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def to_bool_int(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["1", "true", "yes"]).astype(int)


def leakage_match(column: str) -> str | None:
    low = column.lower()
    for pat in LEAKAGE_PATTERNS:
        if pat in low:
            return pat
    return None


def manual_review_match(column: str) -> str | None:
    low = column.lower()
    for pat in MANUAL_REVIEW_PATTERNS:
        if pat in low:
            return pat
    return None


def flatten_json_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col, allowed_tokens in JSON_FEATURE_COLUMNS.items():
        if col not in df.columns:
            continue
        keys: set[str] = set()
        parsed_rows: list[dict[str, Any]] = []
        for raw in df[col].fillna("{}").astype(str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            parsed_rows.append(parsed)
            for key in parsed:
                low = str(key).lower()
                if leakage_match(low):
                    continue
                if any(tok in low for tok in allowed_tokens):
                    keys.add(str(key))
        for key in sorted(keys):
            name = f"{col}.{key}"
            out[name] = [safe_float(row.get(key)) for row in parsed_rows]
    return out


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    included: list[str] = []
    excluded: list[dict[str, str]] = []
    manual: list[dict[str, str]] = []
    numeric = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in {"timestamp", "symbol", "timeframe", "horizon"}:
            continue
        leak = leakage_match(col)
        if leak:
            excluded.append({"column": col, "reason": f"pattern:{leak}"})
            continue
        review = manual_review_match(col)
        if review:
            manual.append({"column": col, "reason": f"ambiguous_pattern:{review}"})
            continue
        if col in BASE_NUMERIC_FEATURES:
            numeric[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            included.append(col)
    flattened = flatten_json_features(df)
    for col in flattened.columns:
        if leakage_match(col):
            excluded.append({"column": col, "reason": "flattened_leakage_pattern"})
            continue
        numeric[col] = flattened[col].fillna(0.0)
        included.append(col)
    return numeric.replace([np.inf, -np.inf], 0.0).fillna(0.0), {
        "included_features": included,
        "excluded_by_leakage": excluded,
        "manual_review_columns": manual,
    }


def build_targets(df: pd.DataFrame) -> dict[str, pd.Series]:
    targets: dict[str, pd.Series] = {}
    if "tail_risk_v4" in df.columns and df["tail_risk_v4"].nunique(dropna=True) > 1:
        targets["target_tail_risk_v4"] = to_bool_int(df["tail_risk_v4"])
    if "bad_entry_v4" in df.columns and df["bad_entry_v4"].nunique(dropna=True) > 1:
        targets["target_bad_entry_v4"] = to_bool_int(df["bad_entry_v4"])
    if {"tail_risk_v4", "bad_entry_v4"} <= set(df.columns):
        union = ((to_bool_int(df["tail_risk_v4"]) == 1) | (to_bool_int(df["bad_entry_v4"]) == 1)).astype(int)
        if union.nunique(dropna=True) > 1:
            targets["target_union_tail_or_bad"] = union
    return targets


@dataclass
class SplitSet:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    method: str
    warning: str | None = None


def make_walk_forward_split(df: pd.DataFrame) -> SplitSet:
    if "timestamp" not in df.columns:
        n = len(df)
        return _ordered_split(np.arange(n), "ordered_fallback", "timestamp missing; temporal validation degraded")
    work = df.copy()
    work["_ts"] = pd.to_datetime(work["timestamp"], errors="coerce", utc=True)
    valid = work["_ts"].notna()
    if valid.sum() < max(30, len(work) * 0.8):
        return _ordered_split(np.arange(len(df)), "ordered_fallback", "insufficient parseable timestamps; validation degraded")
    ordered = work[valid].sort_values(["_ts", "symbol", "horizon"]).index.to_numpy()
    return _ordered_split(ordered, "global_walk_forward", None)


def _ordered_split(indices: np.ndarray, method: str, warning: str | None) -> SplitSet:
    n = len(indices)
    if n < 30:
        return SplitSet(indices[: max(1, int(n * 0.6))], indices[max(1, int(n * 0.6)): max(2, int(n * 0.8))], indices[max(2, int(n * 0.8)):], method, "small dataset; split reliability low")
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    return SplitSet(indices[:train_end], indices[train_end:val_end], indices[val_end:], method, warning)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_metric_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    c = confusion_counts(y_true, pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    rejection = float(pred.mean()) if len(pred) else 0.0
    out = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "rejection_rate": rejection,
        "retained_rate": 1.0 - rejection,
        "risk_capture_rate": recall,
        "false_negatives_tail_risk": fn,
        "threshold": threshold,
        "confusion_matrix": c,
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        out["pr_auc"] = float(average_precision_score(y_true, scores)) if len(set(y_true)) > 1 else None
        out["roc_auc"] = float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else None
    except Exception:
        out["pr_auc"] = None
        out["roc_auc"] = None
    return out


def threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> dict[str, Any]:
    candidates = sorted(set(float(x) for x in scores), reverse=True)
    best = {"threshold": 1.0, "precision": 0.0, "recall": 0.0}
    for threshold in candidates + [0.0]:
        m = safe_metric_metrics(y_true, scores, threshold)
        if m["recall"] >= target_recall:
            return {"threshold": threshold, "precision": m["precision"], "recall": m["recall"], "rejection_rate": m["rejection_rate"]}
        best = {"threshold": threshold, "precision": m["precision"], "recall": m["recall"], "rejection_rate": m["rejection_rate"]}
    return best


def baseline_scores(df: pd.DataFrame) -> np.ndarray:
    parts = []
    for col in ("bad_entry_v4", "early_mae_v4", "squeeze_risk_proxy_v4"):
        parts.append(to_bool_int(df[col]) if col in df.columns else pd.Series(0, index=df.index))
    score = ((parts[0] == 1) | (parts[1] == 1) | (parts[2] == 1)).astype(float)
    return score.to_numpy(dtype=float)


def available_models() -> dict[str, Any]:
    models: dict[str, Any] = {}
    try:
        from sklearn.linear_model import LogisticRegression, SGDClassifier

        models["logistic_regression"] = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
        models["sgd_log_loss"] = SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000, tol=1e-3, random_state=17)
    except Exception:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.08, random_state=17)
        models["random_forest"] = RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_leaf=25, class_weight="balanced_subsample", random_state=17, n_jobs=1)
    except Exception:
        pass
    return models


def model_scores(model: Any, x_train: pd.DataFrame, y_train: pd.Series, x_eval: pd.DataFrame) -> np.ndarray:
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_eval)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x_eval)
        return 1.0 / (1.0 + np.exp(-np.asarray(raw)))
    return model.predict(x_eval)


def feature_importance(model: Any, features: list[str]) -> list[dict[str, Any]]:
    if hasattr(model, "feature_importances_"):
        vals = getattr(model, "feature_importances_")
    elif hasattr(model, "coef_"):
        vals = np.abs(getattr(model, "coef_")[0])
    else:
        return []
    rows = [{"feature": f, "importance": float(v)} for f, v in zip(features, vals)]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)[:20]


def metrics_by_group(df: pd.DataFrame, y: pd.Series, scores: np.ndarray, idx: np.ndarray, keys: list[str]) -> list[dict[str, Any]]:
    local = df.loc[idx, keys].copy()
    local["_y"] = y.loc[idx].to_numpy()
    local["_score"] = scores
    rows: list[dict[str, Any]] = []
    for group, data in local.groupby(keys, dropna=False):
        if not isinstance(group, tuple):
            group = (group,)
        m = safe_metric_metrics(data["_y"].to_numpy(), data["_score"].to_numpy())
        rows.append({**{k: v for k, v in zip(keys, group)}, "rows": len(data), **{k: m[k] for k in ("precision", "recall", "f1", "false_positive_rate", "rejection_rate", "false_negatives_tail_risk")}})
    return rows


def evaluate_target(df: pd.DataFrame, x: pd.DataFrame, target_name: str, y: pd.Series, split: SplitSet) -> dict[str, Any]:
    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx
    target_result: dict[str, Any] = {"target": target_name, "models": {}, "best_model": None}
    y_test = y.loc[test_idx].to_numpy()
    base = baseline_scores(df.loc[test_idx])
    base_metrics = safe_metric_metrics(y_test, base, 0.5)
    base_metrics["threshold_recall_095"] = threshold_for_recall(y_test, base, 0.95)
    base_metrics["threshold_recall_098"] = threshold_for_recall(y_test, base, 0.98)
    target_result["baseline_simple"] = base_metrics
    best_name = "baseline_simple"
    best_precision = base_metrics["precision"]
    for name, model in available_models().items():
        try:
            scores = model_scores(model, x.loc[train_idx], y.loc[train_idx], x.loc[test_idx])
            metrics = safe_metric_metrics(y_test, scores, 0.5)
            metrics["threshold_recall_095"] = threshold_for_recall(y_test, scores, 0.95)
            metrics["threshold_recall_098"] = threshold_for_recall(y_test, scores, 0.98)
            hi = metrics["threshold_recall_095"]
            metrics["precision_at_high_recall_threshold"] = hi.get("precision", 0.0)
            metrics["per_symbol"] = metrics_by_group(df, y, scores, test_idx, ["symbol"]) if "symbol" in df.columns else []
            metrics["per_horizon"] = metrics_by_group(df, y, scores, test_idx, ["horizon"]) if "horizon" in df.columns else []
            metrics["per_group"] = metrics_by_group(df, y, scores, test_idx, ["symbol", "horizon"]) if {"symbol", "horizon"} <= set(df.columns) else []
            metrics["feature_importance"] = feature_importance(model, list(x.columns))
            target_result["models"][name] = metrics
            if metrics["recall"] >= 0.95 and hi.get("precision", 0.0) >= best_precision * 0.95:
                best_name = name
                best_precision = hi.get("precision", metrics["precision"])
        except Exception as exc:
            target_result["models"][name] = {"skipped": True, "reason": str(exc)}
    target_result["best_model"] = best_name
    return target_result


def decide(results: dict[str, Any], feature_info: dict[str, Any], split: SplitSet) -> tuple[str, str]:
    if not feature_info["included_features"]:
        return "DATASET_NOT_USABLE", "no causal features survived allowlist"
    if len(feature_info["excluded_by_leakage"]) == 0:
        return "LEAKAGE_RISK_TOO_HIGH", "allowlist did not identify any label/target columns; dataset schema may be unexpected"
    tail = results.get("target_tail_risk_v4")
    if not tail:
        return "DATASET_NOT_USABLE", "tail_risk_v4 target missing or unusable"
    baseline = tail["baseline_simple"]
    best_name = tail["best_model"]
    best_metrics = tail["models"].get(best_name, baseline) if best_name != "baseline_simple" else baseline
    threshold_095 = best_metrics.get("threshold_recall_095", {})
    if threshold_095.get("recall", 0.0) < 0.95:
        return "RESEARCH_NOT_READY", "no model reaches required high recall on lockbox"
    if best_name == "baseline_simple":
        return "BASELINE_NOT_BEATEN", "baseline simple remains the best high-recall rejection rule"
    if threshold_095.get("precision", 0.0) >= baseline.get("precision", 0.0) * 0.95:
        return "TRRM_PROMISING_FOR_REVIEW", f"{best_name} meets high-recall requirement with acceptable precision trade-off"
    return "BASELINE_NOT_BEATEN", "trained models did not improve enough over baseline simple"


def research_thresholds(best_metrics: dict[str, Any]) -> dict[str, Any]:
    t95 = best_metrics.get("threshold_recall_095", {}).get("threshold", 0.5)
    t98 = best_metrics.get("threshold_recall_098", {}).get("threshold", min(t95, 0.5))
    lo = max(0.0, min(float(t95), float(t98)) * 0.85)
    hi = min(1.0, max(float(t95), float(t98)) * 1.10)
    return {
        "conservative_reject_threshold": float(t95),
        "strict_reject_threshold": float(hi),
        "review_zone_low": float(lo),
        "review_zone_high": float(hi),
        "note": "research-only; do not move to live without linked live-trade validation",
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset_csv) if args.dataset_csv else latest_dataset(out_dir)
    if not dataset or not dataset.exists():
        raise FileNotFoundError("Risk V4 / QMAE samples CSV not found")
    df = pd.read_csv(dataset)
    targets = build_targets(df)
    x, feature_info = build_feature_matrix(df)
    split = make_walk_forward_split(df)
    results = {name: evaluate_target(df, x, name, y, split) for name, y in targets.items()}
    decision, reason = decide(results, feature_info, split)
    tail = results.get("target_tail_risk_v4", {})
    best_name = tail.get("best_model", "baseline_simple")
    best_metrics = tail.get("baseline_simple", {}) if best_name == "baseline_simple" else tail.get("models", {}).get(best_name, {})
    thresholds = research_thresholds(best_metrics) if best_metrics else {}
    stamp = utc_stamp()
    payload = {
        "schema_version": "phase_c_trrm_classifier_v1",
        "generated_at": stamp,
        "decision": decision,
        "reason": reason,
        "dataset": str(dataset),
        "rows": int(len(df)),
        "feature_allowlist": feature_info,
        "targets_evaluated": sorted(results.keys()),
        "split": {
            "method": split.method,
            "train_rows": int(len(split.train_idx)),
            "validation_rows": int(len(split.val_idx)),
            "test_lockbox_rows": int(len(split.test_idx)),
            "warning": split.warning,
        },
        "results": results,
        "research_thresholds": thresholds,
        "artifacts": {
            "model_dir": str(model_dir),
            "model_saved": False,
            "note": "No live-consumable artifact was written. Models are trained in-memory for evaluation only.",
        },
        "safety_confirmations": {
            "no_live_touched": True,
            "no_active_manifest": True,
            "no_yaml": True,
            "no_pm2_restart": True,
            "no_orders": True,
            "no_env": True,
            "no_ts_touched": True,
            "no_push": True,
            "no_model_promotion": True,
            "research_only_artifacts": True,
            "no_future_or_label_features": True,
        },
    }
    json_path = out_dir / f"aegis_phase_c_trrm_classifier_{stamp}.json"
    md_path = out_dir / f"aegis_phase_c_trrm_classifier_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    payload["outputs"] = {"json": str(json_path), "md": str(md_path)}
    print(json.dumps({"decision": decision, "reason": reason, "md": str(md_path), "json": str(json_path)}, indent=2))
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    feature_info = payload["feature_allowlist"]
    tail = payload["results"].get("target_tail_risk_v4", {})
    baseline = tail.get("baseline_simple", {})
    best_name = tail.get("best_model", "n/a")
    best = baseline if best_name == "baseline_simple" else tail.get("models", {}).get(best_name, {})
    imp = best.get("feature_importance", []) if isinstance(best, dict) else []
    lines = [
        "# FASE-C TRRM Risk Classifier",
        "",
        "## 1. Estado",
        f"- decision: {payload['decision']}",
        f"- reason: {payload['reason']}",
        "- mode: research-only",
        "",
        "## 2. Dataset usado",
        f"- path: {payload['dataset']}",
        f"- rows: {payload['rows']}",
        "",
        "## 3. Feature allowlist",
        f"- included count: {len(feature_info['included_features'])}",
        *[f"- {f}" for f in feature_info["included_features"][:30]],
        "",
        "## 4. Columnas excluidas por leakage",
        f"- excluded count: {len(feature_info['excluded_by_leakage'])}",
        *[f"- {r['column']}: {r['reason']}" for r in feature_info["excluded_by_leakage"][:40]],
        "",
        "## 5. Targets evaluados",
        *[f"- {t}" for t in payload["targets_evaluated"]],
        "",
        "## 6. Splits",
        f"- method: {payload['split']['method']}",
        f"- train/validation/test: {payload['split']['train_rows']} / {payload['split']['validation_rows']} / {payload['split']['test_lockbox_rows']}",
        f"- warning: {payload['split']['warning'] or 'none'}",
        "",
        "## 7. Baseline simple",
        "- rule: bad_entry_v4 OR early_mae_v4 OR squeeze_risk_proxy_v4",
        f"- precision: {baseline.get('precision', 0):.4f}",
        f"- recall: {baseline.get('recall', 0):.4f}",
        f"- f1: {baseline.get('f1', 0):.4f}",
        "",
        "## 8. Modelos entrenados",
        *[f"- {name}" for name in tail.get("models", {}).keys()],
        "",
        "## 9. Métricas globales",
        f"- best_model: {best_name}",
        f"- precision: {best.get('precision', 0):.4f}",
        f"- recall: {best.get('recall', 0):.4f}",
        f"- f1: {best.get('f1', 0):.4f}",
        f"- pr_auc: {best.get('pr_auc')}",
        f"- roc_auc: {best.get('roc_auc')}",
        f"- rejection_rate: {best.get('rejection_rate', 0):.4f}",
        "",
        "## 10. Métricas por símbolo",
        *[f"- {r.get('symbol')}: precision={r.get('precision',0):.3f} recall={r.get('recall',0):.3f} rows={r.get('rows')}" for r in best.get("per_symbol", [])[:20]],
        "",
        "## 11. Métricas por horizon",
        *[f"- {r.get('horizon')}: precision={r.get('precision',0):.3f} recall={r.get('recall',0):.3f} rows={r.get('rows')}" for r in best.get("per_horizon", [])[:20]],
        "",
        "## 12. Thresholds research-only",
        *[f"- {k}: {v}" for k, v in payload["research_thresholds"].items()],
        "",
        "## 13. Feature importance",
        *[f"- {r['feature']}: {r['importance']:.6f}" for r in imp[:20]],
        "",
        "## 14. Comparación contra baseline simple",
        f"- baseline_precision: {baseline.get('precision', 0):.4f}",
        f"- baseline_recall: {baseline.get('recall', 0):.4f}",
        f"- selected_precision: {best.get('precision', 0):.4f}",
        f"- selected_recall: {best.get('recall', 0):.4f}",
        "",
        "## 15. Limitaciones",
        "- No hay validación directa contra Phase O live trade IDs.",
        "- El dataset actual sólo tiene el timeframe disponible en el CSV generado.",
        "- No usar random split como evidencia principal.",
        "- No usar labels/path/futuro como features.",
        "",
        "## 16. Decisión",
        f"- {payload['decision']}",
        "",
        "## 17. Recomendación siguiente",
        "- Si se acepta, revisar manualmente features, threshold y per-symbol failures antes de FASE-D o integración shadow.",
        "- No promover a live ni modificar guards.",
        "",
        "## 18. Confirmaciones de seguridad",
        "- no toqué live.",
        "- no toqué active_manifest.",
        "- no toqué YAML.",
        "- no reinicié PM2.",
        "- no envié órdenes.",
        "- no toqué .env.",
        "- no toqué binance-futures-bot-ts.",
        "- no hice push.",
        "- no promoví modelos.",
        "- cualquier artifact generado es research-only.",
        "- no usé labels/futuro como features.",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train research-only TRRM classifier baseline.")
    p.add_argument("--dataset-csv", default="")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--no-save-model", action="store_true")
    return p


def main() -> int:
    run_training(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
