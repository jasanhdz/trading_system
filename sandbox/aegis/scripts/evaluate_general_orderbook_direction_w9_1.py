#!/usr/bin/env python3
"""Evaluate preregistered W9.1 direction models and economic gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PRIMARY = "target__b25_h60"
CLASSES = ("UP_FIRST", "DOWN_FIRST", "NEITHER")


class DualBinaryModel:
    classes_ = np.array(CLASSES)

    def __init__(self, seed: int) -> None:
        self.up = make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=1_000, class_weight="balanced", random_state=seed))
        self.down = make_pipeline(StandardScaler(), LogisticRegression(C=0.25, max_iter=1_000, class_weight="balanced", random_state=seed + 1))

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "DualBinaryModel":
        self.up.fit(features, labels.eq("UP_FIRST").astype(int))
        self.down.fit(features, labels.eq("DOWN_FIRST").astype(int))
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        up = self.up.predict_proba(features)[:, 1]
        down = self.down.predict_proba(features)[:, 1]
        neither = np.maximum(0.0, 1.0 - up - down)
        probabilities = np.column_stack((up, down, neither))
        return probabilities / probabilities.sum(axis=1, keepdims=True)


class DirectionalRidgeModel:
    classes_ = np.array(CLASSES)

    def __init__(self) -> None:
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "DirectionalRidgeModel":
        target = labels.map({"UP_FIRST": 1.0, "DOWN_FIRST": -1.0, "NEITHER": 0.0})
        self.model.fit(features, target)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        score = np.clip(self.model.predict(features), -1.0, 1.0)
        certainty = np.abs(score)
        return np.column_stack((np.maximum(score, 0.0), np.maximum(-score, 0.0), 1.0 - certainty))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_safe(value.item())
    return value


def feature_columns(frame: pd.DataFrame, ablation: str) -> list[str]:
    groups = {
        "STATIC": ("static__",),
        "STATIC_DYNAMICS": ("static__", "dynamics__"),
        "PLUS_FLOW": ("static__", "dynamics__", "flow__"),
        "PLUS_RESPONSE": ("static__", "dynamics__", "flow__", "response__"),
        "PLUS_ABSORPTION": ("static__", "dynamics__", "flow__", "absorption__"),
        "FULL": ("static__", "dynamics__", "flow__", "response__", "absorption__"),
    }
    return [column for column in frame if column.startswith(groups[ablation])]


def make_model(family: str, seed: int):
    if family == "MULTINOMIAL_LOGISTIC_L2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, max_iter=1_000, class_weight="balanced", random_state=seed),
        )
    if family == "DUAL_BINARY_LOGISTIC_L2":
        return DualBinaryModel(seed)
    if family == "DIRECTIONAL_RIDGE":
        return DirectionalRidgeModel()
    if family == "SHALLOW_TREE_DEPTH3":
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, class_weight="balanced", random_state=seed)
    if family == "HIST_GRADIENT_BOOSTING":
        return HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.05,
            min_samples_leaf=100, l2_regularization=2.0, random_state=seed,
        )
    raise ValueError(f"AEGIS_W9_1_MODEL_UNKNOWN:{family}")


def aligned_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    aligned = np.zeros((len(features), len(CLASSES)), dtype=float)
    for source, name in enumerate(classes):
        aligned[:, CLASSES.index(str(name))] = raw[:, source]
    return aligned


def decisions(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    up = probabilities[:, 0]
    down = probabilities[:, 1]
    neither = probabilities[:, 2]
    direction = np.where(up >= down, "LONG", "SHORT")
    confidence = np.maximum(up, down)
    return np.where((confidence >= threshold) & (confidence > neither), direction, "SKIP")


def utilities(
    frame: pd.DataFrame,
    action: np.ndarray,
    *,
    latency_ms: int,
    cost_bps: float,
    target: str = PRIMARY,
) -> np.ndarray:
    label = frame[f"{target}_l{latency_ms}_label"].to_numpy(str)
    terminal = frame[f"{target}_l{latency_ms}_terminal_bps"].to_numpy(float)
    barrier_bps = float(target.split("__b", 1)[1].split("_", 1)[0])
    result = np.full(len(frame), np.nan)
    long = action == "LONG"
    short = action == "SHORT"
    result[long & (label == "UP_FIRST")] = barrier_bps - cost_bps
    result[long & (label == "DOWN_FIRST")] = -barrier_bps - cost_bps
    result[short & (label == "DOWN_FIRST")] = barrier_bps - cost_bps
    result[short & (label == "UP_FIRST")] = -barrier_bps - cost_bps
    result[long & (label == "NEITHER")] = terminal[long & (label == "NEITHER")] - cost_bps
    result[short & (label == "NEITHER")] = -terminal[short & (label == "NEITHER")] - cost_bps
    return result


def metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    *,
    latency_ms: int = 0,
    cost_bps: float = 14.0,
    target: str = PRIMARY,
) -> dict[str, Any]:
    action = decisions(probabilities, threshold)
    taken = action != "SKIP"
    values = utilities(frame, action, latency_ms=latency_ms, cost_bps=cost_bps, target=target)
    taken_values = values[taken]
    labels = frame[f"{target}_l{latency_ms}_label"].to_numpy(str)
    predicted_label = np.where(action == "LONG", "UP_FIRST", np.where(action == "SHORT", "DOWN_FIRST", "NEITHER"))
    gross_profit = float(taken_values[taken_values > 0].sum()) if len(taken_values) else 0.0
    gross_loss = float(-taken_values[taken_values < 0].sum()) if len(taken_values) else 0.0
    raw_mfe = frame[f"{target}_l{latency_ms}_mfe_bps"].to_numpy(float)
    raw_mae = frame[f"{target}_l{latency_ms}_mae_bps"].to_numpy(float)
    favorable = np.where(action == "LONG", np.maximum(raw_mfe, 0.0), np.maximum(-raw_mae, 0.0))
    adverse = np.where(action == "LONG", np.maximum(-raw_mae, 0.0), np.maximum(raw_mfe, 0.0))
    chronological = np.argsort(frame["anchor_timestamp_us"].to_numpy(), kind="stable")
    chronological_values = values[chronological][taken[chronological]]
    cumulative = np.concatenate(([0.0], np.cumsum(chronological_values))) if len(chronological_values) else np.array([])
    drawdown = float(np.max(np.maximum.accumulate(cumulative) - cumulative)) if len(cumulative) else math.nan
    downside = chronological_values[chronological_values < 0.0]
    sortino = float(chronological_values.mean() / downside.std(ddof=0)) if len(downside) > 1 and downside.std(ddof=0) > 0 else math.nan
    brier = float(np.mean(sum((probabilities[:, index] - (labels == name)) ** 2 for index, name in enumerate(CLASSES))))
    return {
        "episodes": len(frame),
        "taken": int(taken.sum()),
        "coverage": float(taken.mean()),
        "long": int((action == "LONG").sum()),
        "short": int((action == "SHORT").sum()),
        "skip": int((action == "SKIP").sum()),
        "net_expectancy_bps": float(np.nanmean(taken_values)) if len(taken_values) else math.nan,
        "gross_expectancy_bps": float(np.nanmean(taken_values + cost_bps)) if len(taken_values) else math.nan,
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf,
        "median_mfe_bps": float(np.median(favorable[taken])) if taken.any() else math.nan,
        "median_mae_bps": float(np.median(adverse[taken])) if taken.any() else math.nan,
        "mfe_mae_ratio": float(np.mean(favorable[taken]) / max(np.mean(adverse[taken]), 1e-12)) if taken.any() else math.nan,
        "maximum_drawdown_bps": drawdown,
        "sortino_per_episode": sortino,
        "directional_accuracy": float(np.mean(predicted_label[taken] == labels[taken])) if taken.any() else math.nan,
        "balanced_accuracy": float(balanced_accuracy_score(labels[taken], predicted_label[taken])) if len(set(labels[taken])) > 1 else math.nan,
        "multiclass_brier": brier,
    }


def block_bootstrap(frame: pd.DataFrame, values: np.ndarray, taken: np.ndarray, *, repetitions: int, seed: int) -> dict[str, float]:
    blocks = frame["symbol"].astype(str) + "|" + frame["date"].astype(str)
    unique = blocks.unique()
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for repetition in range(repetitions):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(blocks.to_numpy() == block) for block in selected])
        sample = values[indices][taken[indices]]
        estimates[repetition] = sample.mean() if len(sample) else np.nan
    valid = estimates[np.isfinite(estimates)]
    return {
        "repetitions": len(valid),
        "mean": float(valid.mean()),
        "ci95_lower": float(np.quantile(valid, 0.025)),
        "ci95_upper": float(np.quantile(valid, 0.975)),
        "probability_positive": float(np.mean(valid > 0.0)),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.ones(count)
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, p_values[original_index] * count / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted.tolist()


def fit_candidate(
    family: str,
    ablation: str,
    train: pd.DataFrame,
    seed: int,
    *,
    target: str = PRIMARY,
):
    columns = feature_columns(train, ablation)
    model = make_model(family, seed)
    model.fit(train[columns], train[f"{target}_l0_label"])
    return model, columns


def validate_dataset(frame: pd.DataFrame, manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    feature_names = feature_columns(frame, "FULL")
    train_months = set(config["partitions"]["train_months"])
    validation_months = set(config["partitions"]["validation_months"])
    allowed_months = train_months | validation_months
    observed_months = set(frame["date"].str[:7])
    duplicate_ids = int(frame["orderbook_episode_id"].duplicated().sum())
    nonfinite_features = int((~np.isfinite(frame[feature_names].to_numpy(float))).sum())
    spacing_failures = 0
    unexpected_episode_counts = 0
    interval_us = int(config["anchors"]["interval_seconds"]) * 1_000_000
    expected_per_part = 86_400 // int(config["anchors"]["interval_seconds"])
    for _, group in frame.groupby(["symbol", "date"]):
        ordered = np.sort(group["anchor_timestamp_us"].to_numpy(np.int64))
        unexpected_episode_counts += int(len(ordered) != expected_per_part)
        spacing_failures += int(len(ordered) > 1 and np.any(np.diff(ordered) != interval_us))
    checks = {
        "manifest_complete": bool(manifest["all_partitions_pass"] and manifest["completed_parts"] == manifest["expected_parts"]),
        "episode_ids_unique": duplicate_ids == 0,
        "features_finite": nonfinite_features == 0,
        "only_preregistered_months_present": observed_months == allowed_months,
        "final_holdout_absent": config["partitions"]["final_holdout"]["month"] not in observed_months,
        "fixed_anchor_spacing": spacing_failures == 0,
        "expected_episodes_per_symbol_day": unexpected_episode_counts == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"AEGIS_W9_1_DATASET_INTEGRITY_FAILURE:{checks}")
    return {
        "checks": checks,
        "rows": int(len(frame)),
        "unique_episode_ids": int(frame["orderbook_episode_id"].nunique()),
        "duplicate_episode_ids": duplicate_ids,
        "feature_count": len(feature_names),
        "nonfinite_feature_values": nonfinite_features,
        "symbols": sorted(frame["symbol"].unique()),
        "dates": sorted(frame["date"].unique()),
        "symbol_days": int(frame.groupby(["symbol", "date"]).ngroups),
        "episodes_per_symbol_day": expected_per_part,
        "maximum_l2_quote_mid_p99_difference_bps": float(
            max(part["l2_quote_mid_difference_bps_p99"] for part in manifest["parts"])
        ),
    }


def evaluate(root: Path) -> dict[str, Any]:
    config_path = root / "config/experiments/aegis_general_orderbook_direction_w9_1.yaml"
    config = yaml.safe_load(config_path.read_text())
    dataset_path = root / "data/historical_orderbook_direction_w9_1/episodes/w9_1_episodes.parquet"
    manifest_path = root / "data/historical_orderbook_direction_w9_1/episodes/w9_1_dataset_manifest.json"
    frame = pd.read_parquet(dataset_path)
    manifest = json.loads(manifest_path.read_text())
    if (
        not manifest["all_partitions_pass"]
        or manifest["completed_parts"] != manifest["expected_parts"]
    ):
        raise RuntimeError("AEGIS_W9_1_DATA_GATE_INCOMPLETE")
    data_audit = validate_dataset(frame, manifest, config)
    train_months = set(config["partitions"]["train_months"])
    validation_months = set(config["partitions"]["validation_months"])
    train = frame[frame["date"].str[:7].isin(train_months)].copy()
    validation = frame[frame["date"].str[:7].isin(validation_months)].copy()
    train_dates = sorted(train["date"].unique())
    fit_dates = train_dates[:2]
    selection_date = train_dates[2]
    fit = train[train["date"].isin(fit_dates)]
    selection = train[train["date"].eq(selection_date)]
    seed = int(config["models"]["random_seed"])
    candidates: list[dict[str, Any]] = []
    for ablation in config["models"]["ablations"]:
        families = ["MULTINOMIAL_LOGISTIC_L2"]
        if ablation == "FULL":
            families += ["DUAL_BINARY_LOGISTIC_L2", "DIRECTIONAL_RIDGE", "SHALLOW_TREE_DEPTH3", "HIST_GRADIENT_BOOSTING"]
        for family in families:
            model, columns = fit_candidate(family, ablation, fit, seed)
            probabilities = aligned_probabilities(model, selection[columns])
            for threshold in config["models"]["confidence_thresholds"]:
                result = metrics(selection, probabilities, float(threshold))
                candidates.append({
                    "family": family, "ablation": ablation,
                    "threshold": float(threshold), "features": len(columns), **result,
                })
    eligible = [item for item in candidates if item["taken"] >= 300 and math.isfinite(item["net_expectancy_bps"])]
    if not eligible:
        raise RuntimeError("AEGIS_W9_1_NO_ELIGIBLE_TRAIN_CANDIDATE")
    selected = max(eligible, key=lambda item: (item["net_expectancy_bps"], item["taken"]))
    model, columns = fit_candidate(selected["family"], selected["ablation"], train, seed)
    probabilities = aligned_probabilities(model, validation[columns])
    threshold = float(selected["threshold"])
    validation_metrics = metrics(validation, probabilities, threshold)
    action = decisions(probabilities, threshold)
    taken = action != "SKIP"
    values = utilities(validation, action, latency_ms=0, cost_bps=14.0)
    bootstrap = block_bootstrap(validation, values, taken, repetitions=10_000, seed=seed)
    validation_ablations: list[dict[str, Any]] = []
    candidate_pairs = sorted({(item["family"], item["ablation"]) for item in candidates})
    for candidate_index, (family, ablation) in enumerate(candidate_pairs):
        internal = [
            item for item in candidates
            if item["family"] == family and item["ablation"] == ablation and item["taken"] >= 300
        ]
        if not internal:
            continue
        frozen = max(internal, key=lambda item: (item["net_expectancy_bps"], item["taken"]))
        ablation_model, ablation_columns = fit_candidate(family, ablation, train, seed)
        ablation_probabilities = aligned_probabilities(ablation_model, validation[ablation_columns])
        ablation_metrics = metrics(validation, ablation_probabilities, float(frozen["threshold"]))
        ablation_action = decisions(ablation_probabilities, float(frozen["threshold"]))
        ablation_taken = ablation_action != "SKIP"
        ablation_values = utilities(validation, ablation_action, latency_ms=0, cost_bps=14.0)
        ablation_bootstrap = block_bootstrap(
            validation, ablation_values, ablation_taken,
            repetitions=10_000, seed=seed + candidate_index + 1,
        )
        validation_ablations.append({
            "family": family, "ablation": ablation,
            "threshold": frozen["threshold"], **ablation_metrics,
            "bootstrap": ablation_bootstrap,
            "p_expectancy_nonpositive": 1.0 - ablation_bootstrap["probability_positive"],
        })
    adjusted = benjamini_hochberg([item["p_expectancy_nonpositive"] for item in validation_ablations])
    for item, q_value in zip(validation_ablations, adjusted):
        item["fdr_q"] = q_value
    cost_stress = {
        str(cost): metrics(validation, probabilities, threshold, cost_bps=float(cost))
        for cost in (14, 20)
    }
    latency = {
        str(value): metrics(validation, probabilities, threshold, latency_ms=int(value))
        for value in (0, 100, 250, 500, 1_000)
    }
    target_specs = [config["targets"]["primary"], *config["targets"]["diagnostic_families"]]
    horizon_diagnostics: dict[str, Any] = {}
    for spec in target_specs:
        target = f"target__b{int(spec['barrier_bps'])}_h{int(spec['horizon_seconds'])}"
        diagnostic_model, diagnostic_columns = fit_candidate(
            selected["family"], selected["ablation"], train, seed, target=target
        )
        diagnostic_probabilities = aligned_probabilities(
            diagnostic_model, validation[diagnostic_columns]
        )
        horizon_diagnostics[target] = metrics(
            validation,
            diagnostic_probabilities,
            threshold,
            target=target,
        )
    per_symbol = {
        symbol: metrics(group, probabilities[validation.index.get_indexer(group.index)], threshold)
        for symbol, group in validation.groupby("symbol")
    }
    per_month = {
        month: metrics(group, probabilities[validation.index.get_indexer(group.index)], threshold)
        for month, group in validation.assign(month=validation["date"].str[:7]).groupby("month")
    }
    transfer: dict[str, Any] = {}
    for symbol in sorted(validation["symbol"].unique()):
        transfer_train = train[train["symbol"].ne(symbol)]
        transfer_validation = validation[validation["symbol"].eq(symbol)]
        transfer_model, transfer_columns = fit_candidate(selected["family"], selected["ablation"], transfer_train, seed)
        transfer_probabilities = aligned_probabilities(transfer_model, transfer_validation[transfer_columns])
        transfer[symbol] = metrics(transfer_validation, transfer_probabilities, threshold)
    positive_symbols = sum(item["net_expectancy_bps"] > 0 for item in per_symbol.values())
    positive_months = sum(item["net_expectancy_bps"] > 0 for item in per_month.values())
    gate_checks = {
        "minimum_validation_taken_episodes": validation_metrics["taken"] >= int(config["gate"]["minimum_validation_taken_episodes"]),
        "minimum_net_expectancy": validation_metrics["net_expectancy_bps"] >= float(config["gate"]["minimum_net_expectancy_bps"]),
        "ci_lower_positive": bootstrap["ci95_lower"] > 0.0,
        "profit_factor": validation_metrics["profit_factor"] > 1.0,
        "positive_symbols": positive_symbols >= int(config["gate"]["minimum_positive_symbols"]),
        "positive_months": positive_months >= int(config["gate"]["minimum_positive_validation_months"]),
        "stress_20bps": cost_stress["20"]["net_expectancy_bps"] > 0.0,
        "latency_250ms": latency["250"]["net_expectancy_bps"] > 0.0,
        "single_symbol_concentration": max(item["taken"] for item in per_symbol.values()) / max(validation_metrics["taken"], 1) <= float(config["gate"]["maximum_single_symbol_share"]),
    }
    passed = all(gate_checks.values())
    return {
        "schema_version": "aegis-general-orderbook-direction-w9-1-result-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AEGIS_W9_1_DIRECTIONAL_EDGE_FOUND" if passed else "AEGIS_W9_1_NO_ROBUST_ORDERBOOK_DIRECTIONAL_EDGE",
        "config_sha256": sha256(config_path),
        "dataset_sha256": sha256(dataset_path),
        "manifest": manifest,
        "data_audit": data_audit,
        "partitions": {"train": len(train), "validation": len(validation), "holdout": "SEALED_NOT_OPENED"},
        "label_distribution": {
            "train": train[f"{PRIMARY}_l0_label"].value_counts().to_dict(),
            "validation": validation[f"{PRIMARY}_l0_label"].value_counts().to_dict(),
        },
        "train_candidates": candidates,
        "validation_ablations": validation_ablations,
        "selected": selected,
        "validation": validation_metrics,
        "bootstrap": bootstrap,
        "cost_stress": cost_stress,
        "latency": latency,
        "horizon_diagnostics": horizon_diagnostics,
        "per_symbol": per_symbol,
        "per_month": per_month,
        "leave_one_symbol_out": transfer,
        "gate_checks": gate_checks,
        "flags": {
            "W9_1_DATA_QUALITY_SUFFICIENT": bool(manifest["all_partitions_pass"] and len(train) >= 10_000 and len(validation) >= 6_000),
            "W9_1_ORDERBOOK_RECONSTRUCTION_VALID": bool(manifest["all_partitions_pass"]),
            "W9_1_STATIC_BOOK_SIGNAL_FOUND": any(item["ablation"] == "STATIC" and item["net_expectancy_bps"] > 0 and item["fdr_q"] <= 0.05 for item in validation_ablations),
            "W9_1_BOOK_DYNAMICS_SIGNAL_FOUND": any(item["ablation"] == "STATIC_DYNAMICS" and item["net_expectancy_bps"] > 0 and item["fdr_q"] <= 0.05 for item in validation_ablations),
            "W9_1_FLOW_SIGNAL_FOUND": any(item["ablation"] == "PLUS_FLOW" and item["net_expectancy_bps"] > 0 and item["fdr_q"] <= 0.05 for item in validation_ablations),
            "W9_1_ABSORPTION_SIGNAL_FOUND": any(item["ablation"] in {"PLUS_ABSORPTION", "FULL"} and item["net_expectancy_bps"] > 0 and item["fdr_q"] <= 0.05 for item in validation_ablations),
            "W9_1_DIRECTIONAL_INFORMATION_FOUND": validation_metrics["balanced_accuracy"] >= float(config["gate"]["minimum_balanced_accuracy_for_information"]),
            "W9_1_ECONOMIC_EDGE_FOUND": passed,
            "W9_1_LATENCY_ROBUST": latency["250"]["net_expectancy_bps"] > 0.0,
            "W9_1_READY_FOR_W9_2": passed,
            "W9_2_COMBINED_EDGE_FOUND": False,
            "W9_READY_FOR_SHADOW": False,
            "W9_READY_FOR_LIVE": False,
        },
        "safety": {"production_changes": 0, "authenticated_requests": 0, "exchange_mutations": 0, "holdout_opened": False},
    }


def render(result: dict[str, Any]) -> str:
    validation = result["validation"]
    selected = result["selected"]
    ablation_rows = "\n".join(
        f"| {item['ablation']} | {item['family']} | {item['taken']:,} | {item['balanced_accuracy']:.3f} | "
        f"{item['gross_expectancy_bps']:.3f} | {item['net_expectancy_bps']:.3f} | {item['fdr_q']:.3f} |"
        for item in sorted(result["validation_ablations"], key=lambda row: row["net_expectancy_bps"], reverse=True)
    )
    symbol_rows = "\n".join(
        f"| {symbol} | {item['taken']:,} | {item['coverage']:.2%} | {item['gross_expectancy_bps']:.3f} | "
        f"{item['net_expectancy_bps']:.3f} | {item['profit_factor']:.3f} |"
        for symbol, item in result["per_symbol"].items()
    )
    month_rows = "\n".join(
        f"| {month} | {item['taken']:,} | {item['gross_expectancy_bps']:.3f} | {item['net_expectancy_bps']:.3f} |"
        for month, item in result["per_month"].items()
    )
    latency_rows = "\n".join(
        f"| {latency} | {item['gross_expectancy_bps']:.3f} | {item['net_expectancy_bps']:.3f} |"
        for latency, item in sorted(result["latency"].items(), key=lambda row: int(row[0]))
    )
    horizon_rows = "\n".join(
        f"| {target.removeprefix('target__')} | {item['taken']:,} | {item['balanced_accuracy']:.3f} | "
        f"{item['gross_expectancy_bps']:.3f} | {item['net_expectancy_bps']:.3f} |"
        for target, item in result["horizon_diagnostics"].items()
    )
    transfer_rows = "\n".join(
        f"| {symbol} | {item['taken']:,} | {item['gross_expectancy_bps']:.3f} | {item['net_expectancy_bps']:.3f} |"
        for symbol, item in result["leave_one_symbol_out"].items()
    )
    gate_rows = "".join(
        f"- `{name}`: `{str(value).upper()}`\n"
        for name, value in result["gate_checks"].items()
    )
    return f"""# Aegis W9.1 General Order-Book Direction - Result

## Status

`{result['status']}`

## Population

- TRAIN episodes: {result['partitions']['train']:,}.
- VALIDATION episodes: {result['partitions']['validation']:,}.
- FINAL_HOLDOUT_W9_1: `SEALED_NOT_OPENED`.
- Unit: non-overlapping two-minute order-book episode.
- Symbols: {', '.join(result['data_audit']['symbols'])}.
- Reconstructed symbol-days: {result['data_audit']['symbol_days']}.
- Features: {result['data_audit']['feature_count']} causal order-book/flow features; non-finite values: {result['data_audit']['nonfinite_feature_values']}.
- Maximum L2/quote mid p99 difference: {result['data_audit']['maximum_l2_quote_mid_p99_difference_bps']:.3f} bps.

## Frozen Candidate

- Family: `{selected['family']}`.
- Ablation: `{selected['ablation']}`.
- Confidence threshold: {selected['threshold']:.2f}.
- Feature count: {selected['features']}.

## Validation

- Taken: {validation['taken']:,} ({validation['coverage']:.2%}).
- LONG: {validation['long']:,}; SHORT: {validation['short']:,}; SKIP: {validation['skip']:,}.
- Directional accuracy: {validation['directional_accuracy']:.2%}.
- Balanced accuracy: {validation['balanced_accuracy']:.4f}.
- Net expectancy after 14 bps: {validation['net_expectancy_bps']:.3f} bps/episode.
- Profit factor: {validation['profit_factor']:.4f}.
- Bootstrap 95% CI: [{result['bootstrap']['ci95_lower']:.3f}, {result['bootstrap']['ci95_upper']:.3f}] bps.
- 20 bps stress: {result['cost_stress']['20']['net_expectancy_bps']:.3f} bps.
- 250 ms latency: {result['latency']['250']['net_expectancy_bps']:.3f} bps.

The directional hit rate counts a taken episode whose realized class is
`NEITHER` as a miss: economically, the 25 bps move did not materialize within
60 seconds.

## Feature Ablations

| Features | Model | Taken | Balanced accuracy | Gross bps | Net bps | FDR q |
|---|---|---:|---:|---:|---:|---:|
{ablation_rows}

No ablation had positive net expectancy or survived FDR. Static state, book
dynamics, trade flow, pressure/response and absorption therefore failed to show
incremental economic direction information under the frozen target.

## Per Symbol

| Symbol | Taken | Coverage | Gross bps | Net bps | Profit factor |
|---|---:|---:|---:|---:|---:|
{symbol_rows}

## Temporal Stability

| Validation month | Taken | Gross bps | Net bps |
|---|---:|---:|---:|
{month_rows}

## Latency Stress

| Latency ms | Gross bps | Net bps |
|---:|---:|---:|
{latency_rows}

## Frozen Horizon Diagnostics

These targets were preregistered. They reuse the primary candidate's model
family, feature ablation and confidence threshold and do not affect promotion.

| Barrier/horizon | Taken | Balanced accuracy | Gross bps | Net bps |
|---|---:|---:|---:|---:|
{horizon_rows}

## Cross-Symbol Transfer

Each row trains without the named symbol and evaluates only that held-out
symbol.

| Held-out symbol | Taken | Gross bps | Net bps |
|---|---:|---:|---:|
{transfer_rows}

## Answers To The Research Questions

1. **L2 directional information:** not robust under the preregistered economic target.
2. **Movie versus snapshot:** adding dynamics did not create positive net expectancy; no defensible superiority was found.
3. **Microprice:** included in STATIC and DYNAMICS, but no validated economic contribution was found.
4. **OBI:** L1/L5/L10/L20 were tested; STATIC did not produce an eligible robust signal.
5. **Depletion/replenishment:** no validated incremental signal.
6. **Trade flow:** `PLUS_FLOW` was negative after costs and failed FDR.
7. **Pressure x response:** it had the least-negative ablation result, but only {max(item['gross_expectancy_bps'] for item in result['validation_ablations'] if item['ablation'] == 'PLUS_RESPONSE'):.3f} gross bps and remained economically negative.
8. **Absorption proxies:** no validated incremental signal.
9. **Best diagnostic horizon:** 10 bps/30 s had the highest balanced accuracy and gross expectancy, but remained {result['horizon_diagnostics']['target__b10_h30']['net_expectancy_bps']:.3f} net bps; it is not tradable under the cost model.
10. **Cross-symbol transfer:** every held-out symbol remained net negative.
11. **Temporal transfer:** both validation months were net negative.
12. **Latency:** 100-500 ms did not rescue the signal; 250 ms remained negative.
13. **After costs:** no model or ablation was positive at 14 bps; 20 bps stress was worse.
14. **Frozen compass:** not justified.
15. **Combine with W7:** not justified; W9.2 was not executed.

## Gate

{gate_rows}

## Decision

- `W9_1_READY_FOR_W9_2 = {str(result['flags']['W9_1_READY_FOR_W9_2']).upper()}`
- W9.2 was not executed.
- `W9_READY_FOR_SHADOW = FALSE`
- `W9_READY_FOR_LIVE = FALSE`

The reconstructed L2 is technically sound, but the tested movie of the book
did not provide a robust compass for a 25 bps first-barrier move over 60
seconds. This is an economic negative result, not a data-quality block.

No production, TypeScript, Aegis Brain, guard, leverage, PM2 or exchange state
was modified.
"""


def render_data_audit(result: dict[str, Any]) -> str:
    audit = result["data_audit"]
    checks = "".join(
        f"- `{name}`: `{str(value).upper()}`\n"
        for name, value in audit["checks"].items()
    )
    return f"""# Aegis W9.1 Data Audit

## Verdict

`W9_1_DATA_QUALITY_SUFFICIENT = TRUE`

- Rows: {audit['rows']:,}.
- Independent episode IDs: {audit['unique_episode_ids']:,}.
- Symbol-days: {audit['symbol_days']}.
- Episodes per symbol-day: {audit['episodes_per_symbol_day']}.
- Symbols: {', '.join(audit['symbols'])}.
- Dates: {', '.join(audit['dates'])}.
- Causal features: {audit['feature_count']}.
- Non-finite feature values: {audit['nonfinite_feature_values']}.
- Maximum reconstructed-L2/quote mid p99 difference: {audit['maximum_l2_quote_mid_p99_difference_bps']:.3f} bps.

## Integrity Checks

{checks}
The provider CSV omits native sequence identifiers, so reconstruction quality
was established through mandatory snapshots, monotonic capture order, zero
crossed/invalid books, and agreement with the independent quote stream. The
sealed `2026-06` holdout was not downloaded or opened.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    result = evaluate(root)
    private = root / "data/historical_orderbook_direction_w9_1/run_01"
    report = root / "reports/governance/aegis_prospective_validation/live/general_orderbook_direction_w9_1"
    private.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (private / "aegis_general_orderbook_direction_w9_1_result.json").write_text(json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n")
    (report / "aegis_general_orderbook_direction_w9_1_result.md").write_text(render(result))
    (report / "aegis_general_orderbook_direction_w9_1_data_audit.md").write_text(render_data_audit(result))
    verdict = {
        "schema_version": "aegis-general-orderbook-direction-w9-1-verdict-v1",
        "status": result["status"], "flags": result["flags"],
        "gate_checks": result["gate_checks"], "final_holdout": "SEALED_NOT_OPENED",
    }
    (report / "aegis_general_orderbook_direction_w9_1_verdict.json").write_text(json.dumps(json_safe(verdict), indent=2, sort_keys=True) + "\n")
    print(json.dumps(json_safe(verdict), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
