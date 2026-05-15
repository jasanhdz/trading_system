#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FALLBACK))


DEFAULT_DATASET = "aegis_alpha/data/processed/decision_brain/decision_brain_dataset_v010.npz"
DEFAULT_OUTPUT_DIR = "aegis_alpha/models/decision_brain/v010"
DEFAULT_REPORT_JSON = "aegis_alpha/logs/decision_brain/decision_brain_train_report_v010.json"
DEFAULT_REPORT_MD = "aegis_alpha/logs/decision_brain/decision_brain_train_report_v010.md"
DEFAULT_CHAT_MD = "aegis_alpha/logs/decision_brain/decision_brain_recommendations_for_chat_v010.md"

ALT_SYMBOLS = {
    "ADAUSDT",
    "AVAXUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "XRPUSDT",
}
MODE_RISK = {
    "NORMAL": 0,
    "CAUTION": 1,
    "RISK_OFF": 2,
    "MANUAL_ONLY": 3,
    "UNKNOWN": -1,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Aegis Decision Brain v0.1 meta-model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model-version", default="v010")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-mode", default="global_and_symbol_optional")
    parser.add_argument("--max-baseline-train-rows", type=int, default=250_000)
    parser.add_argument("--max-rf-train-rows", type=int, default=180_000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(np.asarray(value).item())
    except Exception:
        return default
    return out if math.isfinite(out) else default


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"dataset_not_found: {path}")
    data = np.load(path, allow_pickle=True)
    required = ["X", "y", "feature_names", "label_names", "timestamp", "symbol", "side", "label"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise SystemExit(f"dataset_missing_keys: {missing}")
    return {key: data[key] for key in data.files}


def temporal_split(n_rows: int) -> dict[str, np.ndarray]:
    train_end = int(n_rows * 0.70)
    val_end = int(n_rows * 0.85)
    return {
        "train": np.arange(0, train_end),
        "val": np.arange(train_end, val_end),
        "test": np.arange(val_end, n_rows),
    }


def choose_sample(indices: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(indices) <= max_rows:
        return indices
    rng = np.random.default_rng(seed)
    sampled = rng.choice(indices, size=max_rows, replace=False)
    return np.sort(sampled)


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    labels = list(range(len(label_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "per_class": {
            label_names[i]: {
                "precision": round(float(precision[i]), 6),
                "recall": round(float(recall[i]), 6),
                "f1": round(float(f1[i]), 6),
                "support": int(support[i]),
            }
            for i in labels
        },
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def multiclass_probabilities(model: Any, X: np.ndarray, class_count: int) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        pred = model.predict(X)
        proba = np.zeros((len(pred), class_count), dtype=np.float32)
        proba[np.arange(len(pred)), pred.astype(int)] = 1.0
        return proba
    proba = model.predict_proba(X)
    out = np.zeros((len(X), class_count), dtype=np.float32)
    classes = getattr(model, "classes_", np.arange(class_count))
    for src_idx, klass in enumerate(classes):
        out[:, int(klass)] = proba[:, src_idx]
    return out


def make_buckets(values: np.ndarray, kind: str) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(values, dtype=np.float32)
    if kind == "mae":
        labels = ["LOW_MAE", "MEDIUM_MAE", "HIGH_MAE", "EXTREME_MAE"]
        y = np.zeros(len(arr), dtype=np.int64)
        y[arr <= -0.08] = 1
        y[arr <= -0.15] = 2
        y[arr <= -0.25] = 3
        return y, labels
    labels = ["FAST_GREEN", "SLOW_GREEN", "TRAPPED", "NO_GREEN_OR_UNKNOWN"]
    y = np.full(len(arr), 3, dtype=np.int64)
    finite = np.isfinite(arr)
    y[finite & (arr <= 30.0)] = 0
    y[finite & (arr > 30.0) & (arr <= 60.0)] = 1
    y[finite & (arr > 60.0)] = 2
    return y, labels


def feature_index(feature_names: list[str], name: str) -> int | None:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def event_mode_counts(rows: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for key in ("event_risk_mode", "event_risk_auto_mode", "news_sentiment_mode"):
        if key in rows:
            out[key] = dict(Counter(str(v) for v in rows[key][indices].tolist()))
    return out


def group_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    label_names: list[str],
    min_rows: int = 30,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in sorted(set(str(item) for item in groups.tolist())):
        mask = np.asarray([str(item) == group for item in groups], dtype=bool)
        if int(mask.sum()) < min_rows:
            continue
        result[group] = {
            "rows": int(mask.sum()),
            "accuracy": round(float(accuracy_score(y_true[mask], y_pred[mask])), 6),
            "macro_f1": round(float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)), 6),
            "labels": dict(Counter(label_names[int(v)] for v in y_true[mask].tolist())),
        }
    return result


def btc_eth_context_groups(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    btc_idx = feature_index(feature_names, "btc_action_score")
    eth_idx = feature_index(feature_names, "eth_action_score")
    if btc_idx is None or eth_idx is None:
        return np.asarray(["BTC_ETH_CONTEXT_UNAVAILABLE"] * len(X), dtype=object)
    btc = X[:, btc_idx]
    eth = X[:, eth_idx]
    groups: list[str] = []
    for btc_value, eth_value in zip(btc, eth, strict=False):
        if not math.isfinite(float(btc_value)) or not math.isfinite(float(eth_value)) or (btc_value < 0 and eth_value < 0):
            groups.append("BTC_ETH_UNKNOWN")
        elif btc_value == 1.0 and eth_value == 1.0:
            groups.append("BTC_ETH_LONG_CONFIRM")
        elif btc_value == 0.0 or eth_value == 0.0:
            groups.append("BTC_ETH_HOLD_OR_WEAK")
        elif btc_value < 0.0 or eth_value < 0.0:
            groups.append("BTC_ETH_SHORT_OR_MIXED")
        else:
            groups.append("BTC_ETH_MIXED")
    return np.asarray(groups, dtype=object)


def profit_factor(final_roe: np.ndarray) -> float | None:
    wins = final_roe[final_roe > 0]
    losses = final_roe[final_roe < 0]
    gross_profit = float(np.nansum(wins))
    gross_loss = abs(float(np.nansum(losses)))
    if gross_loss <= 1e-12:
        return None if gross_profit <= 0 else 999.0
    return gross_profit / gross_loss


def trades_per_day(timestamps: np.ndarray, mask: np.ndarray) -> float | None:
    selected = timestamps[mask]
    if len(selected) == 0:
        return None
    parsed: list[datetime] = []
    for value in selected:
        try:
            text = str(value)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed.append(datetime.fromisoformat(text))
        except ValueError:
            continue
    if len(parsed) < 2:
        return None
    span_days = max((max(parsed) - min(parsed)).total_seconds() / 86400.0, 1.0)
    return len(parsed) / span_days


def policy_metrics(
    *,
    name: str,
    mask: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    final_roe: np.ndarray,
    mae_roe: np.ndarray,
    tail_risk: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, Any]:
    allowed = int(mask.sum())
    total = int(len(mask))
    selected_final = final_roe[mask]
    selected_mae = mae_roe[mask]
    selected_tail = tail_risk[mask]
    true_labels = np.asarray([label_names[int(v)] for v in y_true], dtype=object)
    return {
        "policy": name,
        "trades_allowed": allowed,
        "trades_blocked": total - allowed,
        "allowed_pct": round(float(allowed / max(total, 1)), 6),
        "net_proxy_pnl": round(float(np.nansum(selected_final)), 6) if allowed else 0.0,
        "avg_final_roe": round(float(np.nanmean(selected_final)), 6) if allowed else None,
        "avg_mae": round(float(np.nanmean(selected_mae)), 6) if allowed else None,
        "bad_entry_rate": round(float(np.mean(true_labels[mask] == "DO_NOT_ENTER")), 6) if allowed else None,
        "tail_risk_rate": round(float(np.mean(selected_tail > 0)), 6) if allowed else None,
        "win_rate": round(float(np.mean(selected_final > 0)), 6) if allowed else None,
        "profit_factor_proxy": None if allowed == 0 else (round(float(profit_factor(selected_final)), 6) if profit_factor(selected_final) is not None else None),
        "trades_per_day_estimate": None if allowed == 0 else (round(float(trades_per_day(timestamps, mask)), 4) if trades_per_day(timestamps, mask) is not None else None),
        "predicted_label_distribution": dict(Counter(label_names[int(v)] for v in y_pred[mask].tolist())) if allowed else {},
    }


def simulate_policies(
    *,
    rows: dict[str, np.ndarray],
    test_idx: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    label_names: list[str],
    feature_names: list[str],
) -> list[dict[str, Any]]:
    y_true = rows["y"][test_idx].astype(int)
    labels_pred = np.asarray([label_names[int(v)] for v in y_pred], dtype=object)
    symbols = np.asarray([str(v) for v in rows["symbol"][test_idx].tolist()], dtype=object)
    sides = np.asarray([str(v).upper() for v in rows["side"][test_idx].tolist()], dtype=object)
    final_roe = rows["final_roe"][test_idx].astype(np.float32)
    mae_roe = rows["future_mae_roe"][test_idx].astype(np.float32)
    tail_risk = rows["tail_risk"][test_idx].astype(np.int8)
    timestamps = rows["timestamp"][test_idx]
    total = len(test_idx)

    event_auto = np.asarray([str(v).upper() for v in rows.get("event_risk_auto_mode", np.asarray(["UNKNOWN"] * len(rows["y"]), dtype=object))[test_idx].tolist()], dtype=object)
    btc_idx = feature_index(feature_names, "btc_action_score")
    eth_idx = feature_index(feature_names, "eth_action_score")
    X_test = rows["X"][test_idx]
    btc_weak = np.zeros(total, dtype=bool)
    eth_weak = np.zeros(total, dtype=bool)
    if btc_idx is not None:
        btc_weak = X_test[:, btc_idx] == 0.0
    if eth_idx is not None:
        eth_weak = X_test[:, eth_idx] == 0.0
    major_weak = btc_weak | eth_weak | np.isin(event_auto, ["CAUTION", "RISK_OFF", "MANUAL_ONLY"])
    alt_long = np.isin(symbols, sorted(ALT_SYMBOLS)) & (sides == "LONG")

    masks = {
        "A_baseline_turbo_actual": np.ones(total, dtype=bool),
        "B_enter_only_ENTER_NOW": labels_pred == "ENTER_NOW",
        "C_block_DO_NOT_ENTER": labels_pred != "DO_NOT_ENTER",
        "D_block_DO_NOT_ENTER_and_skip_WAIT": np.isin(labels_pred, ["ENTER_NOW", "MANUAL_ONLY"]),
        "E_short_only_enforce": (sides == "SHORT") & (labels_pred != "DO_NOT_ENTER"),
        "F_alt_long_block_when_btc_eth_weak": ~((alt_long & major_weak) | (labels_pred == "DO_NOT_ENTER")),
    }
    return [
        policy_metrics(
            name=name,
            mask=mask,
            y_true=y_true,
            y_pred=y_pred,
            label_names=label_names,
            final_roe=final_roe,
            mae_roe=mae_roe,
            tail_risk=tail_risk,
            timestamps=timestamps,
        )
        for name, mask in masks.items()
    ]


def recommend_policy(policy_results: list[dict[str, Any]], metrics: dict[str, Any], coverage_notes: list[str]) -> dict[str, Any]:
    baseline = next(item for item in policy_results if item["policy"] == "A_baseline_turbo_actual")
    candidates = [item for item in policy_results if item["policy"] != "A_baseline_turbo_actual" and item["trades_allowed"] > 0]
    viable = [
        item for item in candidates
        if (item.get("net_proxy_pnl") or 0.0) > (baseline.get("net_proxy_pnl") or 0.0)
        and (item.get("bad_entry_rate") or 1.0) <= (baseline.get("bad_entry_rate") or 1.0)
    ]
    if metrics["macro_f1"] < 0.45 or coverage_notes:
        return {
            "policy": "SHADOW_ONLY",
            "reason": "model_not_ready_for_enforce_or_context_coverage_incomplete",
            "thresholds": {
                "enter_now_prob_shadow": 0.55,
                "do_not_enter_prob_shadow": 0.55,
            },
        }
    if viable:
        best = sorted(viable, key=lambda item: (item.get("net_proxy_pnl") or 0.0, -(item.get("bad_entry_rate") or 1.0)), reverse=True)[0]
        return {
            "policy": best["policy"],
            "reason": "best_oos_proxy_policy_with_lower_bad_entry_rate",
            "thresholds": {
                "enter_now_prob": 0.60,
                "do_not_enter_prob": 0.55,
            },
        }
    return {
        "policy": "SHADOW_ONLY",
        "reason": "no_oos_policy_improved_baseline_proxy_enough",
        "thresholds": {
            "enter_now_prob_shadow": 0.55,
            "do_not_enter_prob_shadow": 0.55,
        },
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_path = Path(args.dataset)
    rows = load_dataset(dataset_path)
    X = rows["X"].astype(np.float32, copy=False)
    y = rows["y"].astype(np.int64, copy=False)
    feature_names = [str(v) for v in rows["feature_names"].tolist()]
    label_names = [str(v) for v in rows["label_names"].tolist()]
    split = temporal_split(len(y))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(DEFAULT_REPORT_JSON).parent.mkdir(parents=True, exist_ok=True)

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train = imputer.fit_transform(X[split["train"]])
    X_val = imputer.transform(X[split["val"]])
    X_test = imputer.transform(X[split["test"]])
    y_train = y[split["train"]]
    y_val = y[split["val"]]
    y_test = y[split["test"]]

    baseline_idx = choose_sample(split["train"], args.max_baseline_train_rows, args.random_seed)
    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=160, class_weight="balanced", n_jobs=-1)),
    ])
    baseline.fit(X[baseline_idx], y[baseline_idx])
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = class_metrics(y_test, baseline_pred, label_names)

    rf_idx = choose_sample(split["train"], args.max_rf_train_rows, args.random_seed + 1)
    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=40,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.random_seed,
        )),
    ])
    random_forest.fit(X[rf_idx], y[rf_idx])
    rf_pred = random_forest.predict(X_test)
    rf_metrics = class_metrics(y_test, rf_pred, label_names)

    main_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.07,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        class_weight="balanced",
        random_state=args.random_seed,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=12,
    )
    main_model.fit(X_train, y_train)
    val_pred = main_model.predict(X_val)
    test_pred = main_model.predict(X_test)
    val_metrics = class_metrics(y_val, val_pred, label_names)
    test_metrics = class_metrics(y_test, test_pred, label_names)
    test_proba = multiclass_probabilities(main_model, X_test, len(label_names))

    enter_now_y = rows["enter_now_binary"][split["train"]].astype(np.int8)
    do_not_y = rows["do_not_enter_binary"][split["train"]].astype(np.int8)
    enter_now_model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=31, random_state=args.random_seed)
    do_not_enter_model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=31, random_state=args.random_seed + 1)
    enter_now_model.fit(X_train, enter_now_y)
    do_not_enter_model.fit(X_train, do_not_y)

    mae_bucket_y, mae_bucket_labels = make_buckets(rows["future_mae_roe"], "mae")
    ttg_bucket_y, ttg_bucket_labels = make_buckets(rows["time_to_green"], "time_to_green")
    mae_bucket_model = HistGradientBoostingClassifier(max_iter=90, learning_rate=0.08, max_leaf_nodes=24, random_state=args.random_seed + 2)
    ttg_bucket_model = HistGradientBoostingClassifier(max_iter=90, learning_rate=0.08, max_leaf_nodes=24, random_state=args.random_seed + 3)
    mae_bucket_model.fit(X_train, mae_bucket_y[split["train"]])
    ttg_bucket_model.fit(X_train, ttg_bucket_y[split["train"]])
    mae_bucket_metrics = class_metrics(mae_bucket_y[split["test"]], mae_bucket_model.predict(X_test), mae_bucket_labels)
    ttg_bucket_metrics = class_metrics(ttg_bucket_y[split["test"]], ttg_bucket_model.predict(X_test), ttg_bucket_labels)

    policy_results = simulate_policies(
        rows=rows,
        test_idx=split["test"],
        y_pred=test_pred,
        proba=test_proba,
        label_names=label_names,
        feature_names=feature_names,
    )

    coverage_notes: list[str] = []
    for key in ("event_risk_mode", "event_risk_auto_mode", "news_sentiment_mode"):
        if key in rows and set(str(v) for v in rows[key].tolist()) <= {"UNKNOWN"}:
            coverage_notes.append(f"{key}_all_unknown")

    recommended = recommend_policy(policy_results, test_metrics, coverage_notes)
    model_package = {
        "model_version": args.model_version,
        "status": "RESEARCH_CANDIDATE_NOT_LIVE",
        "label_names": label_names,
        "feature_names": feature_names,
        "main_model": main_model,
        "baseline_logistic": baseline,
        "baseline_random_forest": random_forest,
        "enter_now_model": enter_now_model,
        "do_not_enter_model": do_not_enter_model,
        "mae_bucket_model": mae_bucket_model,
        "mae_bucket_labels": mae_bucket_labels,
        "time_to_green_bucket_model": ttg_bucket_model,
        "time_to_green_bucket_labels": ttg_bucket_labels,
        "class_probability_fields": {
            label_names[0]: "enter_now_prob",
            label_names[1]: "wait_prob",
            label_names[2]: "manual_prob",
            label_names[3]: "do_not_enter_prob",
        },
    }
    joblib.dump(model_package, output_dir / "decision_brain_model.joblib")
    joblib.dump(imputer, output_dir / "preprocessor.joblib")
    (output_dir / "feature_columns.json").write_text(json.dumps(feature_names, indent=2) + "\n", encoding="utf-8")

    test_symbols = rows["symbol"][split["test"]]
    test_sides = rows["side"][split["test"]]
    group_report = {
        "by_symbol": group_metrics(y_true=y_test, y_pred=test_pred, groups=test_symbols, label_names=label_names),
        "by_side": group_metrics(y_true=y_test, y_pred=test_pred, groups=test_sides, label_names=label_names),
        "by_btc_eth_context": group_metrics(y_true=y_test, y_pred=test_pred, groups=btc_eth_context_groups(X_test, feature_names), label_names=label_names),
        "by_event_risk_mode": group_metrics(y_true=y_test, y_pred=test_pred, groups=rows.get("event_risk_mode", np.asarray(["UNKNOWN"] * len(y), dtype=object))[split["test"]], label_names=label_names),
        "by_event_risk_auto_mode": group_metrics(y_true=y_test, y_pred=test_pred, groups=rows.get("event_risk_auto_mode", np.asarray(["UNKNOWN"] * len(y), dtype=object))[split["test"]], label_names=label_names),
        "by_news_sentiment_mode": group_metrics(y_true=y_test, y_pred=test_pred, groups=rows.get("news_sentiment_mode", np.asarray(["UNKNOWN"] * len(y), dtype=object))[split["test"]], label_names=label_names),
    }

    manifest = {
        "created_at": utc_now_iso(),
        "model_version": args.model_version,
        "status": "RESEARCH_CANDIDATE_NOT_LIVE",
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "train_mode": args.train_mode,
        "rows": {
            "total": int(len(y)),
            "train": int(len(split["train"])),
            "val": int(len(split["val"])),
            "test": int(len(split["test"])),
        },
        "features_count": int(len(feature_names)),
        "label_names": label_names,
        "metrics": {
            "baseline_logistic_test": baseline_metrics,
            "baseline_random_forest_test": rf_metrics,
            "main_validation": val_metrics,
            "main_test": test_metrics,
            "mae_bucket_test": mae_bucket_metrics,
            "time_to_green_bucket_test": ttg_bucket_metrics,
        },
        "replay_policy_simulation": policy_results,
        "recommended_policy": recommended,
        "thresholds": recommended.get("thresholds", {}),
        "coverage_notes": coverage_notes,
        "notes": [
            "Offline research model only. Not integrated with live trading.",
            "Temporal split uses first 70% train, next 15% validation, last 15% OOS test.",
            "Outcome arrays are used only for labels, metrics, and policy simulation.",
            "Event/news context coverage is currently incomplete because those layers were added after most historical rows.",
        ],
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        **manifest,
        "group_metrics": group_report,
        "event_mode_counts": event_mode_counts(rows, np.arange(len(y))),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    Path(DEFAULT_REPORT_JSON).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(DEFAULT_REPORT_MD).write_text(render_report_md(report), encoding="utf-8")
    Path(DEFAULT_CHAT_MD).write_text(render_chat_md(report), encoding="utf-8")
    return report


def render_report_md(report: dict[str, Any]) -> str:
    main = report["metrics"]["main_test"]
    recommendation = report["recommended_policy"]
    lines = [
        "# Aegis Decision Brain Train Report v010",
        "",
        f"- Created: {report['created_at']}",
        f"- Status: `{report['status']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Rows: train={report['rows']['train']}, val={report['rows']['val']}, test={report['rows']['test']}",
        f"- Features: {report['features_count']}",
        "",
        "## Main Test Metrics",
        "",
        f"- Accuracy: {main['accuracy']}",
        f"- Macro F1: {main['macro_f1']}",
        "",
        "## Per Class",
        "",
    ]
    for label, data in main["per_class"].items():
        lines.append(f"- {label}: precision={data['precision']} recall={data['recall']} f1={data['f1']} support={data['support']}")
    lines.extend(["", "## Replay Policies", ""])
    for item in report["replay_policy_simulation"]:
        lines.append(
            f"- {item['policy']}: allowed={item['trades_allowed']} net_proxy_pnl={item['net_proxy_pnl']} "
            f"avg_mae={item['avg_mae']} bad_entry_rate={item['bad_entry_rate']} win_rate={item['win_rate']}"
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"- Policy: `{recommendation['policy']}`",
        f"- Reason: {recommendation['reason']}",
        "",
        "## Coverage Notes",
        "",
    ])
    for note in report["coverage_notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def render_chat_md(report: dict[str, Any]) -> str:
    main = report["metrics"]["main_test"]
    rec = report["recommended_policy"]
    best_policy = sorted(report["replay_policy_simulation"], key=lambda item: item.get("net_proxy_pnl") or -999, reverse=True)[0]
    edge_line = "No recomiendo live/enforce todavía." if rec["policy"] == "SHADOW_ONLY" else f"Mejor política OOS: {rec['policy']}."
    lines = [
        "# Decision Brain v010 - Resumen para chat",
        "",
        f"Estado: `{report['status']}`",
        "",
        f"Main model OOS: accuracy={main['accuracy']}, macro_f1={main['macro_f1']}.",
        f"{edge_line}",
        "",
        f"Mejor replay proxy por PnL: `{best_policy['policy']}` con net_proxy_pnl={best_policy['net_proxy_pnl']}, "
        f"bad_entry_rate={best_policy['bad_entry_rate']}, avg_mae={best_policy['avg_mae']}.",
        "",
        "Recomendación:",
        f"- `{rec['policy']}`",
        f"- Razón: {rec['reason']}",
        "",
        "Limitación principal:",
        "- EventRiskAuto y NewsSentiment aún no tienen cobertura histórica suficiente; deben seguir en SHADOW juntando datos.",
        "",
        "Uso recomendado ahora:",
        "- Solo investigación/shadow.",
        "- No conectar a TradingService.",
        "- Revisar nuevas muestras después de varios días de logs con event/news completos.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = train(args)
    print(json.dumps({
        "status": report["status"],
        "rows": report["rows"],
        "main_test": {
            "accuracy": report["metrics"]["main_test"]["accuracy"],
            "macro_f1": report["metrics"]["main_test"]["macro_f1"],
        },
        "recommended_policy": report["recommended_policy"],
        "coverage_notes": report["coverage_notes"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
