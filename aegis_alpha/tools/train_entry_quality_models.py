#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        precision_recall_fscore_support,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_sample_weight
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"sklearn_import_failed: {exc!r}") from exc


DEFAULT_DATASET = REPO_ROOT / "aegis_alpha/data/processed/entry_quality/entry_quality_dataset_v020.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "aegis_alpha/models/entry_quality/v020"
REPORT_DIR = REPO_ROOT / "aegis_alpha/logs/entry_quality"
REPORT_JSON = REPORT_DIR / "entry_quality_train_report_v020.json"
REPORT_MD = REPORT_DIR / "entry_quality_train_report_v020.md"
CHAT_REPORT_MD = REPORT_DIR / "entry_quality_recommendations_for_chat_v020.md"
TARGETS = {
    "entry_quality": "label_good_entry_v1",
    "tail_risk": "label_tail_risk_v1",
}
LABEL_COLUMNS = {
    "label_good_entry_v1",
    "label_bad_entry_v1",
    "label_tail_risk_v1",
    "quality_class",
}
FUTURE_PREFIXES = (
    "future_",
    "time_to_",
    "hit_",
    "final_",
)
ENTRY_THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75)
TAIL_THRESHOLDS = (0.30, 0.40, 0.50, 0.60)
COMBO_THRESHOLDS = (
    (0.60, 0.50),
    (0.65, 0.50),
    (0.70, 0.40),
    (0.60, 0.40),
    (0.65, 0.40),
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace("/", "").replace("-", "")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(np.asarray(value).item())
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def read_dataset(dataset_path: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    meta_path = dataset_path.with_name(f"{dataset_path.stem}_meta.json")
    if not meta_path.exists() and dataset_path.suffix == ".parquet":
        meta_path = dataset_path.with_suffix(".npz").with_name(f"{dataset_path.stem}_meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    if dataset_path.exists() and dataset_path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(dataset_path), metadata, str(dataset_path)
        except Exception:
            pass

    npz_path = dataset_path.with_suffix(".npz")
    if not npz_path.exists() and dataset_path.suffix.lower() == ".npz":
        npz_path = dataset_path
    if not npz_path.exists():
        raise FileNotFoundError(f"dataset_not_found: {dataset_path} or {npz_path}")
    return read_npz_dataset(npz_path), metadata, str(npz_path)


def read_npz_dataset(npz_path: Path) -> pd.DataFrame:
    data = np.load(npz_path, allow_pickle=True)
    numeric = np.asarray(data["numeric"], dtype=np.float32)
    numeric_cols = [str(item) for item in data["numeric_columns"].tolist()]
    df = pd.DataFrame(numeric, columns=numeric_cols)
    seen: set[str] = set(df.columns)
    for col in [str(item) for item in data["string_columns"].tolist()]:
        if col in seen:
            continue
        key = f"str_{col}"
        if key not in data:
            continue
        values = np.asarray(data[key], dtype=object).ravel()
        if len(values) != len(df):
            values = values[: len(df)]
        df[col] = pd.Series(values).astype(str)
        seen.add(col)
    for col in ("final_roe_8h", "future_mae_roe", "future_mfe_roe"):
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_encoded_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    symbols = sorted(str(item) for item in out["symbol"].dropna().unique())
    symbol_map = {symbol: idx for idx, symbol in enumerate(symbols)}
    out["side_long"] = out["side"].astype(str).str.upper().eq("LONG").astype(np.float32)
    out["side_short"] = out["side"].astype(str).str.upper().eq("SHORT").astype(np.float32)
    out["symbol_code"] = out["symbol"].map(symbol_map).astype(np.float32)
    out["turbo_action_long"] = out.get("turbo_action", "").astype(str).str.upper().eq("LONG").astype(np.float32)
    out["turbo_action_short"] = out.get("turbo_action", "").astype(str).str.upper().eq("SHORT").astype(np.float32)
    out["turbo_action_hold"] = out.get("turbo_action", "").astype(str).str.upper().eq("HOLD").astype(np.float32)
    out["candidate_reason_code"] = pd.factorize(out.get("candidate_reason", pd.Series("", index=out.index)).astype(str))[0].astype(np.float32)
    for symbol in symbols:
        out[f"symbol_is_{symbol}"] = out["symbol"].astype(str).eq(symbol).astype(np.float32)
    return out, symbol_map


def leakage_column(name: str) -> bool:
    if name in LABEL_COLUMNS:
        return True
    if any(name.startswith(prefix) for prefix in FUTURE_PREFIXES):
        return True
    return name in {
        "entry_price",
        "leverage",
        "timestamp",
        "symbol",
        "side",
        "timeframe",
        "quality_class",
        "candidate_generation_method",
    }


def select_feature_columns(df: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
    requested = list(metadata.get("feature_columns") or [])
    model_cols = [
        "long_score_7d",
        "long_score_14d",
        "long_score_30d",
        "short_score_7d",
        "short_score_14d",
        "short_score_30d",
        "votes_long",
        "votes_short",
        "votes_neutral",
        "turbo_score",
        "score_gap",
        "side_long",
        "side_short",
        "symbol_code",
        "turbo_action_long",
        "turbo_action_short",
        "turbo_action_hold",
        "candidate_reason_code",
    ]
    model_cols.extend(col for col in df.columns if col.startswith("symbol_is_"))
    ordered: list[str] = []
    for col in requested + model_cols:
        if col in df.columns and col not in ordered and not leakage_column(col):
            ordered.append(col)
    return ordered


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp_dt"] = pd.to_datetime(out["timestamp"], errors="coerce")
    for col in ("label_good_entry_v1", "label_bad_entry_v1", "label_tail_risk_v1"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int8)
    for col in ("future_mae_roe", "future_mfe_roe", "final_roe_8h", "turbo_score"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp_dt", "symbol", "side"])
    out = out[out["timestamp_dt"].notna()].reset_index(drop=True)
    return out


def split_indices(frame: pd.DataFrame, time_based: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(frame["timestamp_dt"].to_numpy()) if time_based else np.arange(len(frame))
    n = len(order)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return order[:train_end], order[train_end:val_end], order[val_end:]


def make_preprocessor() -> SimpleImputer:
    return SimpleImputer(strategy="median")


def maybe_sample_indices(indices: np.ndarray, max_rows: int | None) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or len(indices) <= max_rows:
        return indices
    positions = np.linspace(0, len(indices) - 1, max_rows).round().astype(int)
    return indices[positions]


def train_logistic_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_rows: int = 250_000,
) -> Pipeline | None:
    if len(np.unique(y_train)) < 2:
        return None
    if len(y_train) > max_rows:
        idx = np.linspace(0, len(y_train) - 1, max_rows).round().astype(int)
        x_fit = x_train[idx]
        y_fit = y_train[idx]
    else:
        x_fit = x_train
        y_fit = y_train
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    solver="saga",
                    C=0.5,
                ),
            ),
        ]
    )
    model.fit(x_fit, y_fit)
    return model


def train_hgb_classifier(x_train: np.ndarray, y_train: np.ndarray, *, symbol_model: bool) -> HistGradientBoostingClassifier | None:
    if len(np.unique(y_train)) < 2:
        return None
    weights = compute_sample_weight("balanced", y_train)
    model = HistGradientBoostingClassifier(
        max_iter=80 if symbol_model else 120,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        early_stopping=True,
        random_state=42,
    )
    model.fit(x_train, y_train, sample_weight=weights)
    return model


def train_random_forest_optional(x_train: np.ndarray, y_train: np.ndarray, *, enabled: bool) -> RandomForestClassifier | None:
    if not enabled or len(np.unique(y_train)) < 2:
        return None
    max_rows = min(150_000, len(y_train))
    idx = np.linspace(0, len(y_train) - 1, max_rows).round().astype(int)
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=50,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model.fit(x_train[idx], y_train[idx])
    return model


def predict_prob(model: Any, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full((len(x),), np.nan, dtype=np.float32)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)
    pred = np.asarray(model.predict(x), dtype=np.float32)
    return np.clip(pred, 0.0, 1.0)


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    valid = np.isfinite(scores)
    y_true = y_true[valid].astype(int)
    scores = scores[valid]
    if len(y_true) == 0:
        return {}
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "rows": int(len(y_true)),
        "positive_rate": safe_float(np.mean(y_true)),
        "accuracy": safe_float(accuracy_score(y_true, y_pred)),
        "precision": safe_float(precision),
        "recall": safe_float(recall),
        "f1": safe_float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = safe_float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = safe_float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    metrics["calibration_buckets"] = calibration_buckets(y_true, scores)
    return metrics


def calibration_buckets(y_true: np.ndarray, scores: np.ndarray, buckets: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for low in np.linspace(0.0, 0.9, buckets):
        high = low + 0.1
        mask = (scores >= low) & (scores < high if high < 1.0 else scores <= high)
        if not np.any(mask):
            continue
        out.append(
            {
                "bucket": f"{low:.1f}-{high:.1f}",
                "rows": int(mask.sum()),
                "avg_score": safe_float(np.mean(scores[mask])),
                "actual_rate": safe_float(np.mean(y_true[mask])),
            }
        )
    return out


def proxy_profit_factor(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    wins = vals[vals > 0].sum()
    losses = -vals[vals < 0].sum()
    if losses <= 1e-12:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)


def gate_simulation_entry(frame: pd.DataFrame, quality_scores: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(frame)
    for threshold in ENTRY_THRESHOLDS:
        allow = quality_scores >= threshold
        block = ~allow
        allowed = frame[allow]
        blocked = frame[block]
        rows.append(
            {
                "quality_threshold": threshold,
                "trades_allowed": int(allow.sum()),
                "trades_blocked": int(block.sum()),
                "allowed_pct": safe_float(allow.mean(), 0.0),
                "bad_entries_blocked": int(blocked["label_bad_entry_v1"].sum()),
                "good_entries_blocked": int(blocked["label_good_entry_v1"].sum()),
                "bad_block_rate": safe_float(blocked["label_bad_entry_v1"].mean()) if len(blocked) else None,
                "good_block_rate": safe_float(blocked["label_good_entry_v1"].mean()) if len(blocked) else None,
                "net_proxy_pnl_allowed": safe_float(allowed["final_roe_8h"].sum(), 0.0),
                "net_proxy_pnl_blocked": safe_float(blocked["final_roe_8h"].sum(), 0.0),
                "proxy_pf_allowed": proxy_profit_factor(allowed["final_roe_8h"]),
                "avg_future_mae_allowed": safe_float(allowed["future_mae_roe"].mean()) if len(allowed) else None,
                "avg_future_mae_blocked": safe_float(blocked["future_mae_roe"].mean()) if len(blocked) else None,
                "tail_risk_rate_allowed": safe_float(allowed["label_tail_risk_v1"].mean()) if len(allowed) else None,
                "tail_risk_rate_blocked": safe_float(blocked["label_tail_risk_v1"].mean()) if len(blocked) else None,
                "bad_entry_rate_allowed": safe_float(allowed["label_bad_entry_v1"].mean()) if len(allowed) else None,
                "trades_per_day_estimate": trades_per_day(allowed),
                "total_trades": n,
            }
        )
    return rows


def gate_simulation_tail(frame: pd.DataFrame, tail_scores: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in TAIL_THRESHOLDS:
        allow = tail_scores <= threshold
        block = ~allow
        allowed = frame[allow]
        blocked = frame[block]
        rows.append(
            {
                "tail_threshold": threshold,
                "trades_allowed": int(allow.sum()),
                "trades_blocked": int(block.sum()),
                "allowed_pct": safe_float(allow.mean(), 0.0),
                "tail_losses_blocked": int(blocked["label_tail_risk_v1"].sum()),
                "good_trades_wrongly_blocked": int(blocked["label_good_entry_v1"].sum()),
                "avg_mae_allowed": safe_float(allowed["future_mae_roe"].mean()) if len(allowed) else None,
                "worst_mae_allowed": safe_float(allowed["future_mae_roe"].min()) if len(allowed) else None,
                "tail_risk_rate_allowed": safe_float(allowed["label_tail_risk_v1"].mean()) if len(allowed) else None,
                "tail_risk_rate_blocked": safe_float(blocked["label_tail_risk_v1"].mean()) if len(blocked) else None,
                "net_proxy_pnl_allowed": safe_float(allowed["final_roe_8h"].sum(), 0.0),
                "proxy_pf_allowed": proxy_profit_factor(allowed["final_roe_8h"]),
            }
        )
    return rows


def combo_simulation(frame: pd.DataFrame, quality_scores: np.ndarray, tail_scores: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for quality_threshold, tail_threshold in COMBO_THRESHOLDS:
        allow = (quality_scores >= quality_threshold) & (tail_scores <= tail_threshold)
        allowed = frame[allow]
        blocked = frame[~allow]
        rows.append(
            {
                "quality_threshold": quality_threshold,
                "tail_threshold": tail_threshold,
                "allowed_pct": safe_float(allow.mean(), 0.0),
                "trades_allowed": int(allow.sum()),
                "trades_blocked": int((~allow).sum()),
                "net_proxy_pnl": safe_float(allowed["final_roe_8h"].sum(), 0.0),
                "proxy_pf": proxy_profit_factor(allowed["final_roe_8h"]),
                "avg_mae": safe_float(allowed["future_mae_roe"].mean()) if len(allowed) else None,
                "blocked_avg_mae": safe_float(blocked["future_mae_roe"].mean()) if len(blocked) else None,
                "bad_entry_rate": safe_float(allowed["label_bad_entry_v1"].mean()) if len(allowed) else None,
                "tail_risk_rate": safe_float(allowed["label_tail_risk_v1"].mean()) if len(allowed) else None,
                "good_entries_blocked": int(blocked["label_good_entry_v1"].sum()),
                "bad_entries_blocked": int(blocked["label_bad_entry_v1"].sum()),
                "tail_entries_blocked": int(blocked["label_tail_risk_v1"].sum()),
                "trades_per_day_estimate": trades_per_day(allowed),
                "by_symbol": group_rates(allowed, "symbol"),
                "by_side": group_rates(allowed, "side"),
            }
        )
    return rows


def trades_per_day(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return 0.0
    days = max((frame["timestamp_dt"].max() - frame["timestamp_dt"].min()).total_seconds() / 86400.0, 1.0)
    return safe_float(len(frame) / days, 0.0)


def group_rates(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if frame.empty or column not in frame:
        return out
    for key, group in frame.groupby(column):
        out[str(key)] = {
            "rows": int(len(group)),
            "bad_entry_rate": safe_float(group["label_bad_entry_v1"].mean()),
            "tail_risk_rate": safe_float(group["label_tail_risk_v1"].mean()),
            "avg_mae": safe_float(group["future_mae_roe"].mean()),
            "net_proxy_pnl": safe_float(group["final_roe_8h"].sum(), 0.0),
        }
    return out


def score_bucket_rates(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    buckets = [0.0, 0.50, 0.60, 0.65, 0.70, 0.75, 0.85, 1.01]
    for low, high in zip(buckets[:-1], buckets[1:]):
        mask = (scores >= low) & (scores < high)
        group = frame[mask]
        out[f"{low:.2f}-{high:.2f}"] = {
            "rows": int(len(group)),
            "good_rate": safe_float(group["label_good_entry_v1"].mean()) if len(group) else None,
            "bad_rate": safe_float(group["label_bad_entry_v1"].mean()) if len(group) else None,
            "tail_rate": safe_float(group["label_tail_risk_v1"].mean()) if len(group) else None,
        }
    return out


def evaluate_model_slice(frame: pd.DataFrame, scores: np.ndarray, target_col: str) -> dict[str, Any]:
    frame = frame.reset_index(drop=True)
    scores = np.asarray(scores, dtype=np.float32)
    y = frame[target_col].to_numpy(dtype=int)
    metrics = binary_metrics(y, scores)
    if target_col == "label_good_entry_v1":
        pred_good = scores >= 0.5
        pred_bad = ~pred_good
        metrics["precision_good"] = safe_float(precision_score(y, pred_good, zero_division=0))
        metrics["recall_good"] = safe_float(recall_score(y, pred_good, zero_division=0))
        bad_true = frame["label_bad_entry_v1"].to_numpy(dtype=int)
        metrics["precision_bad_proxy"] = safe_float(precision_score(bad_true, pred_bad, zero_division=0))
        metrics["score_buckets"] = score_bucket_rates(frame, scores)
    else:
        pred_tail = scores >= 0.5
        metrics["precision_tail"] = safe_float(precision_score(y, pred_tail, zero_division=0))
        metrics["recall_tail"] = safe_float(recall_score(y, pred_tail, zero_division=0))
        cm = confusion_matrix(y, pred_tail, labels=[0, 1])
        metrics["false_positives"] = int(cm[0, 1])
        metrics["false_negatives"] = int(cm[1, 0])
    metrics["by_symbol"] = {
        symbol: binary_metrics(group[target_col].to_numpy(dtype=int), scores[group.index.to_numpy()])
        for symbol, group in frame.groupby("symbol")
        if len(group) > 0
    }
    metrics["by_side"] = {
        side: binary_metrics(group[target_col].to_numpy(dtype=int), scores[group.index.to_numpy()])
        for side, group in frame.groupby("side")
        if len(group) > 0
    }
    return metrics


def recent_date_evaluation(frame: pd.DataFrame, quality_scores: np.ndarray, tail_scores: np.ndarray) -> dict[str, Any]:
    if frame.empty:
        return {}
    cutoff = frame["timestamp_dt"].max() - pd.Timedelta(days=30)
    recent = frame[frame["timestamp_dt"] >= cutoff]
    idx = recent.index.to_numpy()
    return {
        "cutoff": str(cutoff),
        "rows": int(len(recent)),
        "entry_quality": evaluate_model_slice(recent, quality_scores[idx], "label_good_entry_v1"),
        "tail_risk": evaluate_model_slice(recent, tail_scores[idx], "label_tail_risk_v1"),
        "entry_gate": gate_simulation_entry(recent, quality_scores[idx]),
        "tail_gate": gate_simulation_tail(recent, tail_scores[idx]),
        "combo_gate": combo_simulation(recent, quality_scores[idx], tail_scores[idx]),
    }


def choose_recommendations(combo_rows: list[dict[str, Any]], entry_rows: list[dict[str, Any]], tail_rows: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [
        row
        for row in combo_rows
        if (row.get("allowed_pct") or 0.0) >= 0.20 and (row.get("trades_allowed") or 0) >= 100
    ]
    if viable:
        best_combo = sorted(
            viable,
            key=lambda row: (
                row.get("proxy_pf") or 0.0,
                -(row.get("bad_entry_rate") or 1.0),
                -(row.get("tail_risk_rate") or 1.0),
            ),
            reverse=True,
        )[0]
    else:
        best_combo = max(combo_rows, key=lambda row: row.get("proxy_pf") or 0.0) if combo_rows else {}
    best_entry = max(entry_rows, key=lambda row: row.get("proxy_pf_allowed") or 0.0) if entry_rows else {}
    best_tail = min(
        tail_rows,
        key=lambda row: (
            row.get("tail_risk_rate_allowed") if row.get("tail_risk_rate_allowed") is not None else 999,
            -(row.get("allowed_pct") or 0.0),
        ),
    ) if tail_rows else {}
    return {
        "recommended_quality_threshold": best_combo.get("quality_threshold", best_entry.get("quality_threshold")),
        "recommended_tail_threshold": best_combo.get("tail_threshold", best_tail.get("tail_threshold")),
        "best_combo": best_combo,
        "best_entry_only": best_entry,
        "best_tail_only": best_tail,
        "live_status": "RESEARCH_CANDIDATE_NOT_LIVE",
        "enforce_recommendation": "Keep SHADOW. If tested further, consider SHORT-only shadow/limited enforcement first after forward validation.",
    }


def fit_target(
    name: str,
    target_col: str,
    frame: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    *,
    symbol_model: bool,
    train_baseline: bool,
    train_rf: bool,
    max_train_rows: int | None,
    time_based: bool,
) -> dict[str, Any]:
    train_idx, val_idx, test_idx = split_indices(frame, time_based)
    train_fit_idx = maybe_sample_indices(train_idx, max_train_rows)
    preprocessor = make_preprocessor()
    x_train_raw = frame.iloc[train_fit_idx][feature_cols].to_numpy(dtype=np.float32)
    x_train = preprocessor.fit_transform(x_train_raw).astype(np.float32)
    y_train = frame.iloc[train_fit_idx][target_col].to_numpy(dtype=np.int8)
    x_val = preprocessor.transform(frame.iloc[val_idx][feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    y_val = frame.iloc[val_idx][target_col].to_numpy(dtype=np.int8)
    x_test = preprocessor.transform(frame.iloc[test_idx][feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    y_test = frame.iloc[test_idx][target_col].to_numpy(dtype=np.int8)

    hgb = train_hgb_classifier(x_train, y_train, symbol_model=symbol_model)
    baseline = train_logistic_baseline(x_train, y_train) if train_baseline else None
    rf = train_random_forest_optional(x_train, y_train, enabled=train_rf)

    val_scores = predict_prob(hgb, x_val)
    test_scores = predict_prob(hgb, x_test)
    baseline_test_scores = predict_prob(baseline, x_test) if baseline is not None else np.full(len(test_idx), np.nan)
    rf_test_scores = predict_prob(rf, x_test) if rf is not None else np.full(len(test_idx), np.nan)

    model_path = output_dir / f"{name}_model.joblib"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "estimator": hgb,
            "preprocessor": preprocessor,
            "feature_columns": feature_cols,
            "target": target_col,
            "created_at": utc_now_iso(),
            "status": "RESEARCH_CANDIDATE_NOT_LIVE",
        },
        model_path,
    )
    if baseline is not None:
        joblib.dump(baseline, output_dir / f"{name}_baseline_logistic.joblib")
    if rf is not None:
        joblib.dump(rf, output_dir / f"{name}_random_forest.joblib")

    return {
        "model_path": str(model_path),
        "baseline_logistic_path": str(output_dir / f"{name}_baseline_logistic.joblib") if baseline is not None else None,
        "random_forest_path": str(output_dir / f"{name}_random_forest.joblib") if rf is not None else None,
        "rows": {
            "total": int(len(frame)),
            "train": int(len(train_fit_idx)),
            "train_available_before_cap": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "positive_rates": {
            "train": safe_float(np.mean(y_train)),
            "validation": safe_float(np.mean(y_val)) if len(y_val) else None,
            "test": safe_float(np.mean(y_test)) if len(y_test) else None,
        },
        "validation_metrics": binary_metrics(y_val, val_scores),
        "test_metrics": binary_metrics(y_test, test_scores),
        "baseline_logistic_test_metrics": binary_metrics(y_test, baseline_test_scores) if baseline is not None else None,
        "random_forest_test_metrics": binary_metrics(y_test, rf_test_scores) if rf is not None else None,
        "test_indices": test_idx.astype(int).tolist(),
        "test_scores": test_scores.astype(float).tolist(),
    }


def train_scope(
    scope_name: str,
    frame: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
    *,
    symbol_model: bool,
    train_baseline: bool,
    train_rf: bool,
    max_train_rows: int | None,
    time_based: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scope": scope_name,
        "rows": int(len(frame)),
        "models": {},
    }
    entry = fit_target(
        "entry_quality",
        TARGETS["entry_quality"],
        frame,
        feature_cols,
        output_dir,
        symbol_model=symbol_model,
        train_baseline=train_baseline,
        train_rf=train_rf,
        max_train_rows=max_train_rows,
        time_based=time_based,
    )
    tail = fit_target(
        "tail_risk",
        TARGETS["tail_risk"],
        frame,
        feature_cols,
        output_dir,
        symbol_model=symbol_model,
        train_baseline=train_baseline,
        train_rf=train_rf,
        max_train_rows=max_train_rows,
        time_based=time_based,
    )
    result["models"]["entry_quality"] = strip_arrays(entry)
    result["models"]["tail_risk"] = strip_arrays(tail)

    test_idx = np.asarray(entry["test_indices"], dtype=int)
    # Split is identical for both targets within this scope.
    test_frame = frame.iloc[test_idx].copy().reset_index(drop=False)
    quality_scores = np.asarray(entry["test_scores"], dtype=np.float32)
    tail_scores = np.asarray(tail["test_scores"], dtype=np.float32)
    result["evaluation"] = {
        "entry_quality": evaluate_model_slice(test_frame, quality_scores, "label_good_entry_v1"),
        "tail_risk": evaluate_model_slice(test_frame, tail_scores, "label_tail_risk_v1"),
        "entry_gate_simulation": gate_simulation_entry(test_frame, quality_scores),
        "tail_gate_simulation": gate_simulation_tail(test_frame, tail_scores),
        "combo_gate_simulation": combo_simulation(test_frame, quality_scores, tail_scores),
        "recent_date_evaluation": recent_date_evaluation(test_frame, quality_scores, tail_scores),
    }
    result["recommendations"] = choose_recommendations(
        result["evaluation"]["combo_gate_simulation"],
        result["evaluation"]["entry_gate_simulation"],
        result["evaluation"]["tail_gate_simulation"],
    )
    return result


def strip_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in {"test_indices", "test_scores"}}


def write_feature_columns(path: Path, feature_cols: list[str], symbol_map: dict[str, int]) -> None:
    write_json(
        path,
        {
            "feature_columns": feature_cols,
            "symbol_encoding": symbol_map,
            "created_at": utc_now_iso(),
        },
    )


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    dataset_path_used: str,
    feature_cols: list[str],
    symbol_map: dict[str, int],
    global_result: dict[str, Any] | None,
    symbol_results: dict[str, Any],
    recommendations: dict[str, Any],
) -> None:
    write_json(
        path,
        {
            "model_version": args.model_version,
            "created_at": utc_now_iso(),
            "dataset_path_requested": args.dataset,
            "dataset_path_used": dataset_path_used,
            "feature_columns": feature_cols,
            "symbol_encoding": symbol_map,
            "targets": TARGETS,
            "sklearn_version": sklearn.__version__,
            "metrics": {
                "global": global_result,
                "symbols": symbol_results,
            },
            "recommended_thresholds": recommendations,
            "status": "RESEARCH_CANDIDATE_NOT_LIVE",
        },
    )


def write_reports(report: dict[str, Any]) -> None:
    write_json(REPORT_JSON, report)
    write_markdown_report(REPORT_MD, report)
    write_chat_report(CHAT_REPORT_MD, report)


def compact_metric(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "n/a"
    return (
        f"auc={metrics.get('roc_auc')}, pr_auc={metrics.get('pr_auc')}, "
        f"precision={metrics.get('precision')}, recall={metrics.get('recall')}"
    )


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    global_eval = ((report.get("global") or {}).get("evaluation") or {})
    global_models = ((report.get("global") or {}).get("models") or {})
    recommendations = (report.get("global") or {}).get("recommendations") or report.get("recommended_thresholds") or {}
    lines = [
        "# Aegis Entry Quality Model Training v0.2.0",
        "",
        f"- Created at: `{report['created_at']}`",
        f"- Dataset: `{report['dataset_path_used']}`",
        f"- Rows: `{report['rows_total']}`",
        f"- Symbols: `{', '.join(report['symbols'])}`",
        f"- Status: `{report['status']}`",
        "",
        "## Global Metrics",
        "",
        f"- Entry Quality: {compact_metric((global_models.get('entry_quality') or {}).get('test_metrics'))}",
        f"- Tail Risk: {compact_metric((global_models.get('tail_risk') or {}).get('test_metrics'))}",
        "",
        "## Recommended Thresholds",
        "",
        f"- quality >= `{recommendations.get('recommended_quality_threshold')}`",
        f"- tail <= `{recommendations.get('recommended_tail_threshold')}`",
        "",
        "## Combo Simulation",
        "",
    ]
    for row in global_eval.get("combo_gate_simulation", [])[:10]:
        lines.append(
            f"- q>={row['quality_threshold']} tail<={row['tail_threshold']}: "
            f"allowed={row['allowed_pct']:.3f}, pf={row['proxy_pf']:.3f}, "
            f"bad={row['bad_entry_rate']:.3f}, tail={row['tail_risk_rate']:.3f}, "
            f"avg_mae={row['avg_mae']:.4f}"
        )
    lines.extend(["", "## Symbol Models", ""])
    for symbol, result in report.get("symbols_results", {}).items():
        models = result.get("models", {})
        lines.append(
            f"- `{symbol}` rows={result.get('rows')}: "
            f"EQ {compact_metric((models.get('entry_quality') or {}).get('test_metrics'))}; "
            f"Tail {compact_metric((models.get('tail_risk') or {}).get('test_metrics'))}"
        )
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings") or []
    lines.extend(f"- {warning}" for warning in warnings) if warnings else lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chat_report(path: Path, report: dict[str, Any]) -> None:
    global_result = report.get("global") or {}
    models = global_result.get("models") or {}
    rec = global_result.get("recommendations") or {}
    combo = rec.get("best_combo") or {}
    entry_gate = ((global_result.get("evaluation") or {}).get("entry_gate_simulation") or [])
    baseline_bad = report.get("baseline_bad_entry_rate")
    baseline_mae = report.get("baseline_avg_mae")
    allowed_bad = combo.get("bad_entry_rate")
    allowed_mae = combo.get("avg_mae")
    bad_reduction = None
    mae_reduction = None
    if baseline_bad is not None and allowed_bad is not None and baseline_bad > 0:
        bad_reduction = (baseline_bad - allowed_bad) / baseline_bad
    if baseline_mae is not None and allowed_mae is not None and baseline_mae < 0:
        mae_reduction = (allowed_mae - baseline_mae) / abs(baseline_mae)
    lines = [
        "# Entry Quality v0.2 Research Recommendation",
        "",
        f"Status: `{report['status']}`. These models are research candidates only and are not wired into live inference.",
        "",
        "## Edge Check",
        "",
        f"- Entry Quality ROC AUC: `{(models.get('entry_quality') or {}).get('test_metrics', {}).get('roc_auc')}`",
        f"- Entry Quality PR AUC: `{(models.get('entry_quality') or {}).get('test_metrics', {}).get('pr_auc')}`",
        f"- Tail Risk ROC AUC: `{(models.get('tail_risk') or {}).get('test_metrics', {}).get('roc_auc')}`",
        f"- Tail Risk PR AUC: `{(models.get('tail_risk') or {}).get('test_metrics', {}).get('pr_auc')}`",
        "",
        "## Suggested Research Threshold",
        "",
        f"- Quality threshold: `>= {rec.get('recommended_quality_threshold')}`",
        f"- Tail threshold: `<= {rec.get('recommended_tail_threshold')}`",
        f"- Allowed pct: `{combo.get('allowed_pct')}`",
        f"- Trades blocked: `{combo.get('trades_blocked')}`",
        f"- Bad-entry rate allowed: `{allowed_bad}`",
        f"- Tail-risk rate allowed: `{combo.get('tail_risk_rate')}`",
        f"- Avg MAE allowed: `{allowed_mae}`",
        f"- Proxy PF: `{combo.get('proxy_pf')}`",
        "",
        "## Impact",
        "",
        f"- Baseline bad-entry rate: `{baseline_bad}`",
        f"- Baseline avg MAE: `{baseline_mae}`",
        f"- Bad-entry reduction estimate: `{bad_reduction}`",
        f"- MAE improvement estimate: `{mae_reduction}`",
        f"- Frequency estimate: `{combo.get('trades_per_day_estimate')}` candidate trades/day",
        "",
        "## Recommendation",
        "",
        rec.get("enforce_recommendation", "Keep SHADOW until forward validation confirms the historical edge."),
        "",
        "## Entry-Only Gate Table",
        "",
    ]
    for row in entry_gate:
        lines.append(
            f"- q>={row['quality_threshold']}: allowed={row['allowed_pct']:.3f}, "
            f"blocked={row['trades_blocked']}, bad_blocked={row['bad_entries_blocked']}, "
            f"good_blocked={row['good_entries_blocked']}, avg_mae_allowed={row['avg_future_mae_allowed']}, "
            f"tail_allowed={row['tail_risk_rate_allowed']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Aegis v0.2 Entry Quality and Tail Risk research models.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--symbols", default="all", help="all or comma-separated symbols")
    parser.add_argument("--model-version", default="v020")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-mode", default="global_and_symbol", choices=["global", "symbol", "global_and_symbol"])
    parser.add_argument("--min-rows-per-symbol", type=int, default=500)
    parser.add_argument("--test-split-time-based", default="true")
    parser.add_argument("--train-random-forest", default="false")
    parser.add_argument("--max-global-train-rows", type=int, default=700_000)
    parser.add_argument("--max-symbol-train-rows", type=int, default=120_000)
    return parser.parse_args()


def main() -> int:
    started = time.time()
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    time_based = parse_bool(args.test_split_time_based)
    train_rf = parse_bool(args.train_random_forest)

    df_raw, metadata, dataset_path_used = read_dataset(dataset_path)
    df_raw = clean_dataset(df_raw)
    df, symbol_map = add_encoded_columns(df_raw)
    feature_cols = select_feature_columns(df, metadata)
    missing_counts = {col: int(df[col].isna().sum()) for col in feature_cols if int(df[col].isna().sum()) > 0}

    requested_symbols = (
        sorted(df["symbol"].astype(str).unique())
        if str(args.symbols).strip().lower() == "all"
        else [normalize_symbol(item) for item in str(args.symbols).split(",") if item.strip()]
    )
    df = df[df["symbol"].isin(requested_symbols)].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no_rows_after_symbol_filter")

    write_feature_columns(output_dir / "feature_columns.json", feature_cols, symbol_map)
    joblib.dump(make_preprocessor().fit(df[feature_cols].to_numpy(dtype=np.float32)), output_dir / "scaler_or_preprocessor.joblib")

    global_result: dict[str, Any] | None = None
    if args.train_mode in {"global", "global_and_symbol"}:
        print(f"training global models rows={len(df)} features={len(feature_cols)}")
        global_result = train_scope(
            "global",
            df,
            feature_cols,
            output_dir,
            symbol_model=False,
            train_baseline=True,
            train_rf=train_rf,
            max_train_rows=args.max_global_train_rows,
            time_based=time_based,
        )
        # Required artifact names.
        for src_name, dst_name in (
            ("entry_quality_model.joblib", "global_entry_quality_model.joblib"),
            ("tail_risk_model.joblib", "global_tail_risk_model.joblib"),
        ):
            src = output_dir / src_name
            dst = output_dir / dst_name
            if src.exists():
                dst.write_bytes(src.read_bytes())

    symbol_results: dict[str, Any] = {}
    if args.train_mode in {"symbol", "global_and_symbol"}:
        for symbol in requested_symbols:
            symbol_frame = df[df["symbol"] == symbol].copy().reset_index(drop=True)
            if len(symbol_frame) < args.min_rows_per_symbol:
                symbol_results[symbol] = {"skipped": True, "reason": "below_min_rows", "rows": int(len(symbol_frame))}
                continue
            print(f"training symbol models symbol={symbol} rows={len(symbol_frame)}")
            symbol_results[symbol] = train_scope(
                symbol,
                symbol_frame,
                feature_cols,
                output_dir / symbol,
                symbol_model=True,
                train_baseline=False,
                train_rf=False,
                max_train_rows=args.max_symbol_train_rows,
                time_based=time_based,
            )

    recommendations = (global_result or {}).get("recommendations", {})
    baseline_bad = safe_float(df["label_bad_entry_v1"].mean())
    baseline_mae = safe_float(df["future_mae_roe"].mean())
    report = {
        "created_at": utc_now_iso(),
        "runtime_seconds": round(time.time() - started, 3),
        "model_version": args.model_version,
        "dataset_path_requested": str(dataset_path),
        "dataset_path_used": dataset_path_used,
        "output_dir": str(output_dir),
        "rows_total": int(len(df)),
        "symbols": requested_symbols,
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "missing_features": missing_counts,
        "targets": TARGETS,
        "split": {"type": "time_based_70_15_15", "enabled": time_based},
        "baseline_bad_entry_rate": baseline_bad,
        "baseline_tail_risk_rate": safe_float(df["label_tail_risk_v1"].mean()),
        "baseline_avg_mae": baseline_mae,
        "baseline_proxy_pf": proxy_profit_factor(df["final_roe_8h"]),
        "global": global_result,
        "symbols_results": symbol_results,
        "recommended_thresholds": recommendations,
        "warnings": [
            "Research candidate only; not promoted to API or live runtime.",
            "Parquet unavailable in environment if pyarrow/fastparquet is not installed; NPZ loader was used when needed.",
        ],
        "status": "RESEARCH_CANDIDATE_NOT_LIVE",
    }
    write_reports(report)
    write_manifest(
        output_dir / "model_manifest.json",
        args,
        dataset_path_used,
        feature_cols,
        symbol_map,
        global_result,
        symbol_results,
        recommendations,
    )
    print(
        json.dumps(
            {
                "runtime_seconds": report["runtime_seconds"],
                "rows_total": report["rows_total"],
                "feature_count": report["feature_count"],
                "global_entry_quality_auc": (((global_result or {}).get("models") or {}).get("entry_quality") or {}).get("test_metrics", {}).get("roc_auc"),
                "global_tail_risk_auc": (((global_result or {}).get("models") or {}).get("tail_risk") or {}).get("test_metrics", {}).get("roc_auc"),
                "recommended_thresholds": recommendations,
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
