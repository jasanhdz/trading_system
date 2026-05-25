#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from aegis_alpha.turbo.train_operable_edge_v2 import (
    MODEL_FAMILIES,
    _classifier,
    _model_seed,
    _regressor,
    _side_arrays,
    finite_float,
    safe_corr,
)
from aegis_alpha.turbo.walk_forward_operable_v2 import temporal_folds


def feature_family(name: str) -> tuple[str, str]:
    prefixes = ("mean_64", "mean_12", "mean_6", "std_64", "std_12", "delta_12", "delta_6", "last")
    for prefix in prefixes:
        if name.startswith(f"{prefix}_"):
            return prefix, name[len(prefix) + 1 :]
    return "operable_v2", name


def feature_target_statistics(dataset: dict[str, Any], side: str, horizon: int) -> list[dict[str, Any]]:
    x = np.asarray(dataset["X"], dtype=np.float64)
    names = [str(item) for item in np.asarray(dataset["feature_names"]).tolist()]
    arrays = _side_arrays(dataset, side.lower(), horizon)
    clean = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    hit8 = arrays["hit8"].astype(np.int8)
    quality = arrays["quality"].astype(np.float64)
    danger = arrays["danger"].astype(np.int8)
    mi_hit8 = mutual_info_classif(clean, hit8, discrete_features=False, random_state=13) if len(np.unique(hit8)) > 1 else np.zeros(clean.shape[1])
    mi_quality = mutual_info_regression(clean, quality, random_state=13) if float(np.std(quality)) > 1e-12 else np.zeros(clean.shape[1])
    mi_danger = mutual_info_classif(clean, danger, discrete_features=False, random_state=13) if len(np.unique(danger)) > 1 else np.zeros(clean.shape[1])
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        col = x[:, index]
        finite = col[np.isfinite(col)]
        family, indicator = feature_family(name)
        rows.append({
            "feature_name": name,
            "feature_family": family,
            "base_indicator": indicator,
            "mean": finite_float(np.mean(finite)) if len(finite) else None,
            "std": finite_float(np.std(finite)) if len(finite) else None,
            "min": finite_float(np.min(finite)) if len(finite) else None,
            "max": finite_float(np.max(finite)) if len(finite) else None,
            "nan_rate": finite_float(np.isnan(col).mean()),
            "inf_rate": finite_float(np.isinf(col).mean()),
            "zero_rate": finite_float(np.mean(np.nan_to_num(col) == 0.0)),
            "constant_rate": float(len(finite) > 0 and float(np.std(finite)) <= 1e-12),
            "unique_count": int(len(np.unique(finite))) if len(finite) else 0,
            "corr_hit8": safe_corr(col, hit8),
            "corr_trade_quality": safe_corr(col, quality),
            "corr_mae_danger": safe_corr(col, danger),
            "mi_hit8": finite_float(mi_hit8[index]),
            "mi_trade_quality": finite_float(mi_quality[index]),
            "mi_mae_danger": finite_float(mi_danger[index]),
        })
    return rows


def highly_correlated_features(dataset: dict[str, Any], threshold: float = 0.995) -> list[dict[str, Any]]:
    x = np.nan_to_num(np.asarray(dataset["X"], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    names = [str(item) for item in np.asarray(dataset["feature_names"]).tolist()]
    std = np.std(x, axis=0)
    usable = np.where(std > 1e-12)[0]
    if len(usable) < 2:
        return []
    corr = np.corrcoef(x[:, usable], rowvar=False)
    rows: list[dict[str, Any]] = []
    for left in range(len(usable)):
        for right in range(left + 1, len(usable)):
            value = float(corr[left, right])
            if math.isfinite(value) and abs(value) >= threshold:
                rows.append({
                    "feature_a": names[int(usable[left])],
                    "feature_b": names[int(usable[right])],
                    "correlation": value,
                })
    return rows


def candidate_feature_indices(stats: list[dict[str, Any]], maximum: int) -> list[int]:
    ranked = sorted(
        enumerate(stats),
        key=lambda item: max(
            abs(float(item[1].get("corr_hit8") or 0.0)),
            abs(float(item[1].get("corr_trade_quality") or 0.0)),
            abs(float(item[1].get("corr_mae_danger") or 0.0)),
            float(item[1].get("mi_trade_quality") or 0.0),
        ),
        reverse=True,
    )
    return [index for index, _ in ranked[:maximum]]


def permutation_importance_by_fold(
    dataset: dict[str, Any],
    *,
    symbol: str,
    side: str,
    lookback_days: int,
    horizon: int,
    fold_count: int = 4,
    max_features: int = 50,
    sample_size: int = 1000,
    fast: bool = False,
) -> dict[str, Any]:
    stats = feature_target_statistics(dataset, side, horizon)
    selected = candidate_feature_indices(stats, max_features)
    names = [str(item) for item in np.asarray(dataset["feature_names"]).tolist()]
    x = np.asarray(dataset["X"], dtype=np.float32)[:, selected]
    selected_names = [names[index] for index in selected]
    arrays = _side_arrays(dataset, side.lower(), horizon)
    folds = temporal_folds(len(x), fold_count=fold_count)
    rows: list[dict[str, Any]] = []
    top_by_family: dict[str, list[set[str]]] = defaultdict(list)
    for fold in folds:
        train = fold["train"]
        test = fold["test"]
        if sample_size > 0 and len(test) > sample_size:
            test = test[-sample_size:]
        for family, spec in MODEL_FAMILIES.items():
            target_alias = {
                "hit8_before_minus5": "hit8",
                "trade_quality": "quality",
                "mae_danger": "danger",
            }[spec["target_suffix"]]
            target = arrays[target_alias]
            if spec["kind"] == "classifier" and (len(np.unique(target[train])) < 2 or len(np.unique(target[test])) < 2):
                continue
            if spec["kind"] == "regressor" and float(np.std(target[train])) <= 1e-12:
                continue
            seed = _model_seed(symbol, side.lower(), lookback_days, f"importance:{family}:{fold['fold']}")
            estimator = _classifier(60 if fast else 120, seed) if spec["kind"] == "classifier" else _regressor(60 if fast else 120, seed)
            estimator.fit(x[train], target[train])
            scoring = "roc_auc" if spec["kind"] == "classifier" else "neg_mean_absolute_error"
            importance = permutation_importance(
                estimator,
                x[test],
                target[test],
                scoring=scoring,
                n_repeats=1 if fast else 3,
                random_state=seed,
            )
            order = np.argsort(importance.importances_mean)[::-1]
            top_by_family[family].append({selected_names[int(index)] for index in order[:10]})
            for rank, index in enumerate(order, start=1):
                rows.append({
                    "symbol": symbol,
                    "side": side.upper(),
                    "fold": int(fold["fold"]),
                    "model_family": family,
                    "feature_name": selected_names[int(index)],
                    "rank": int(rank),
                    "permutation_importance_mean": finite_float(importance.importances_mean[int(index)]),
                    "permutation_importance_std": finite_float(importance.importances_std[int(index)]),
                    "built_in_importance": None,
                    "built_in_note": "HistGradientBoosting_has_no_supported_builtin_feature_importances",
                })
    stability: list[dict[str, Any]] = []
    for family, groups in top_by_family.items():
        overlaps: list[float] = []
        for left, right in zip(groups, groups[1:]):
            union = left | right
            overlaps.append(len(left & right) / len(union) if union else 0.0)
        stability.append({
            "model_family": family,
            "valid_fold_count": len(groups),
            "adjacent_top10_jaccard_mean": finite_float(np.mean(overlaps)) if overlaps else None,
            "stable_top_features": sorted(set.intersection(*groups)) if groups else [],
        })
    return {
        "selected_features": selected_names,
        "importance_rows": rows,
        "stability": stability,
    }
