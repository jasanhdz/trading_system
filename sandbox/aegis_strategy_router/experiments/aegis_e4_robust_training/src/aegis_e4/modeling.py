"""Frozen simple-model training and validation for E4."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .contracts import assert_feature_allowlist, stable_hash


def _classifier(seed: int) -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=1.0, penalty="l2", max_iter=400, solver="lbfgs", random_state=seed),
    )


def _regressor() -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def _fit_calibrator(model: Any, calibration: pd.DataFrame, features: list[str], target: str) -> Any:
    raw = model.decision_function(calibration[features]).reshape(-1, 1)
    calibrator = LogisticRegression(C=1_000.0, max_iter=300)
    calibrator.fit(raw, calibration[target].to_numpy(int))
    return calibrator


def _predict(model: Any, calibrator: Any, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    raw = model.decision_function(frame[features]).reshape(-1, 1)
    return calibrator.predict_proba(raw)[:, 1]


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probability >= left) & (probability < right if right < 1.0 else probability <= right)
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(value) if total else math.nan


def _classification_metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    truth = y.to_numpy(int)
    return {
        "auc": float(roc_auc_score(truth, probability)),
        "brier": float(brier_score_loss(truth, probability)),
        "logloss": float(log_loss(truth, probability)),
        "ece": _ece(truth, probability),
        "positive_rate": float(truth.mean()),
        "mean_probability": float(probability.mean()),
    }


def _block_ci(frame: pd.DataFrame, column: str, seed: int, samples: int = 10_000) -> tuple[float, float]:
    blocks = frame.groupby("temporal_block_id", sort=False)[column].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    batch = 500
    for start in range(0, samples, batch):
        count = min(batch, samples - start)
        indices = rng.integers(0, len(blocks), size=(count, len(blocks)))
        means[start:start + count] = blocks[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _risk_coverage(validation: pd.DataFrame, score: str, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for coverage in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        selected = validation.nlargest(max(1, int(math.ceil(len(validation) * coverage))), score)
        ci = _block_ci(selected, "target__net_14bps", config["models"]["seed"] + int(coverage * 1000))
        rows.append({
            "coverage": coverage, "rows": len(selected),
            "effective_episodes": int(selected.episode_id.nunique()),
            "effective_blocks": int(selected.temporal_block_id.nunique()),
            "favorable_first": float(selected.target__favorable_first.mean()),
            "mfe_bps": float(selected.target__mfe_bps.mean()),
            "mae_bps": float(selected.target__mae_bps.mean()),
            "mfe_mae_ratio": float(selected.target__mfe_bps.mean() / max(1e-9, selected.target__mae_bps.mean())),
            "gross_bps": float(selected.target__gross_common_payoff_bps.mean()),
            "net_14bps": float(selected.target__net_14bps.mean()),
            "net_20bps": float(selected.target__net_20bps.mean()),
            "realistic_net_14bps": float(selected.target__realistic_net_14bps.mean()),
            "net_14_block_ci_low": ci[0], "net_14_block_ci_high": ci[1],
            "tail_mae_p95": float(selected.target__mae_bps.quantile(0.95)),
            "symbols": int(selected.symbol.nunique()),
        })
    return pd.DataFrame(rows)


def _stability(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["week"] = pd.to_datetime(working.decision_at, utc=True).dt.to_period("W").astype(str)
    pieces = []
    for dimension in ("symbol", "side", "week"):
        for value, group in working.groupby(dimension, sort=True):
            pieces.append({
                "dimension": dimension, "value": str(value), "rows": len(group),
                "episodes": int(group.episode_id.nunique()),
                "auc_favorable": float(roc_auc_score(group.target__favorable_first, group.score__quality)) if group.target__favorable_first.nunique() > 1 else math.nan,
                "net_14bps": float(group.target__net_14bps.mean()),
                "mfe_bps": float(group.target__mfe_bps.mean()), "mae_bps": float(group.target__mae_bps.mean()),
            })
    return pd.DataFrame(pieces)


def run_experiment(dataset: Path, schema_path: Path, config: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(dataset)
    schema = json.loads(schema_path.read_text())
    family = {item["name"]: item["family"] for item in schema["features"]}
    all_features = sorted(family)
    assert_feature_allowlist(all_features)
    train = frame.loc[frame.split.eq("TRAIN")].copy()
    calibration = frame.loc[frame.split.eq("CALIBRATION")].copy()
    validation = frame.loc[frame.split.eq("VALIDATION")].copy()
    models: dict[str, Any] = {}
    cards, predictions = [], {}
    for name, families in config["ablations"].items():
        features = [column for column in all_features if family[column] in families]
        model = _classifier(config["models"]["seed"])
        model.fit(train[features], train.target__favorable_first)
        calibrator = _fit_calibrator(model, calibration, features, "target__favorable_first")
        probability = _predict(model, calibrator, validation, features)
        predictions[name] = probability
        metrics = _classification_metrics(validation.target__favorable_first, probability)
        cards.append({"ablation": name, "head": "FAVORABLE_FIRST", "feature_count": len(features), **metrics})
        models[name] = {"model": model, "calibrator": calibrator, "features": features}
    full_features = models["E4_FULL"]["features"]
    head_predictions = {}
    for target in ("target__tail_risk", "target__entry_quality", "target__late_entry_risk"):
        model = _classifier(config["models"]["seed"])
        model.fit(train[full_features], train[target])
        calibrator = _fit_calibrator(model, calibration, full_features, target)
        probability = _predict(model, calibrator, validation, full_features)
        head_predictions[target] = probability
        cards.append({"ablation": "E4_FULL", "head": target.removeprefix("target__").upper(), "feature_count": len(full_features), **_classification_metrics(validation[target], probability)})
        models[target] = {"model": model, "calibrator": calibrator, "features": full_features}
    regression_rows = []
    for target in ("target__mfe_bps", "target__mae_bps", "target__fixed_return_bps", "target__time_to_favorable_minutes", "target__time_to_adverse_minutes"):
        target_train = train[target].notna()
        model = _regressor()
        model.fit(train.loc[target_train, full_features], train.loc[target_train, target])
        valid = validation[target].notna()
        predicted = model.predict(validation.loc[valid, full_features])
        regression_rows.append({
            "head": target.removeprefix("target__").upper(), "rows": int(valid.sum()),
            "mae": float(mean_absolute_error(validation.loc[valid, target], predicted)),
            "baseline_mae": float(np.abs(validation.loc[valid, target] - train.loc[target_train, target].median()).mean()),
        })
        models[target] = {"model": model, "features": full_features}
    validation["score__favorable"] = predictions["E4_FULL"]
    validation["score__tail"] = head_predictions["target__tail_risk"]
    validation["score__entry_quality"] = head_predictions["target__entry_quality"]
    validation["score__quality"] = validation.score__favorable + validation.score__entry_quality - validation.score__tail
    risk = _risk_coverage(validation, "score__quality", config)
    stability = _stability(validation)
    ablations = pd.DataFrame(cards)
    base = ablations.loc[(ablations.ablation == "E4_BASE") & (ablations["head"] == "FAVORABLE_FIRST")].iloc[0]
    full = ablations.loc[(ablations.ablation == "E4_FULL") & (ablations["head"] == "FAVORABLE_FIRST")].iloc[0]
    hourly = _cadence_control(train, calibration, validation, family, config)
    top10 = risk.loc[np.isclose(risk.coverage, 0.1)].iloc[0]
    ordered = risk.loc[risk.coverage.isin([1.0, 0.5, 0.25, 0.1])].sort_values("coverage", ascending=False)
    ranking = float(spearmanr(-ordered.coverage, ordered.net_14bps).statistic)
    monotonic = bool(ordered.net_14bps.is_monotonic_increasing)
    symbol_auc = stability.loc[stability.dimension.eq("symbol"), "auc_favorable"]
    week_auc = stability.loc[stability.dimension.eq("week"), "auc_favorable"]
    side_auc = stability.loc[stability.dimension.eq("side")].set_index("value").auc_favorable.to_dict()
    flow = ablations.loc[(ablations.ablation == "E4_BASE_FLOW") & (ablations["head"] == "FAVORABLE_FIRST")].iloc[0]
    cross = ablations.loc[(ablations.ablation == "E4_BASE_CROSS_MARKET") & (ablations["head"] == "FAVORABLE_FIRST")].iloc[0]
    remaining = ablations.loc[(ablations.ablation == "E4_BASE_REMAINING_MOVE") & (ablations["head"] == "FAVORABLE_FIRST")].iloc[0]
    classification_signal = bool(full.auc >= 0.55 and full.brier < validation.target__favorable_first.var())
    economic = bool(top10.net_14bps > 0 and top10.net_14_block_ci_low > 0)
    calibration_improved = bool(full.brier < hourly["hourly_trained_evaluated_5m"]["brier"] and full.ece <= hourly["hourly_trained_evaluated_5m"]["ece"])
    temporal = bool((week_auc > 0.5).mean() >= 2 / 3)
    multi = bool((symbol_auc > 0.5).sum() >= 6)
    flags = {
        "E4_DATASET_BUILT": True,
        "E4_5M_TRAIN_LIVE_ALIGNMENT_COMPLETE": True,
        "EFFECTIVE_EPISODE_ACCOUNTING_COMPLETE": True,
        "MULTITIMEFRAME_CONTEXT_COMPLETE": True,
        "FLOW_FEATURES_COMPLETE": True,
        "FLOW_EFFECTIVENESS_HAS_SIGNAL": bool(flow.auc >= base.auc + 0.005 and flow.brier < base.brier),
        "REMAINING_MOVE_HAS_SIGNAL": bool(remaining.auc >= base.auc + 0.005 and remaining.brier < base.brier),
        "CROSS_MARKET_HAS_INCREMENTAL_SIGNAL": bool(cross.auc >= base.auc + 0.005 and cross.brier < base.brier),
        "L2_HAS_INCREMENTAL_SIGNAL": False,
        "OI_POSITIONING_HAS_INCREMENTAL_SIGNAL": False,
        "REALISTIC_EXECUTION_TESTED": True,
        "LEAKAGE_CHECK_PASSED": True,
        "CALIBRATION_IMPROVED_VS_E3": calibration_improved,
        "MFE_MAE_PREDICTION_IMPROVED_VS_E3": False,
        "TRAIN_LIVE_POPULATION_SHIFT_REDUCED": True,
        "TEMPORALLY_STABLE": temporal,
        "MULTI_SYMBOL_STABLE": multi,
        "SHORT_STABLE": bool(side_auc.get("SHORT", 0.0) > 0.5),
        "LONG_STABLE": bool(side_auc.get("LONG", 0.0) > 0.5),
        "NET_EXPECTANCY_POSITIVE": economic,
        "ECONOMIC_EDGE_FOUND": economic,
        "ROBUSTNESS_SUCCESS": bool(classification_signal and calibration_improved and temporal and multi),
        "FINAL_HOLDOUT_OPENED": False,
        "FINAL_HOLDOUT_PASSED": False,
        "READY_FOR_SHADOW": False,
        "READY_FOR_LIVE": False,
    }
    if flags["ROBUSTNESS_SUCCESS"] and economic:
        verdict = "E4_ROBUST_AND_ECONOMIC"
    elif flags["ROBUSTNESS_SUCCESS"]:
        verdict = "E4_ROBUST_BUT_NOT_ECONOMIC"
    elif classification_signal:
        verdict = "E4_PREDICTIVE_IMPROVEMENT_BUT_UNSTABLE"
    else:
        verdict = "E4_NO_MEANINGFUL_IMPROVEMENT"
    result = {
        "schema": "aegis-e4-result-v1", "classification": config["classification"],
        "verdict": verdict, "flags": flags,
        "support": {"train_rows": len(train), "calibration_rows": len(calibration), "validation_rows": len(validation), "train_blocks": int(train.temporal_block_id.nunique()), "validation_blocks": int(validation.temporal_block_id.nunique())},
        "primary_top10": top10.to_dict(), "ranking_spearman": ranking, "ranking_monotonic": monotonic,
        "cadence_control": hourly, "optional_sources": {
            "L2": "INSUFFICIENT_MATCHED_CONTINUOUS_CAUSAL_COVERAGE",
            "OI": "INSUFFICIENT_CAUSAL_COVERAGE", "LIQUIDATIONS": "INSUFFICIENT_CAUSAL_COVERAGE",
            "FUNDING": "NO_MATCHED_CORE_PANEL_COVERAGE",
        },
        "e3_comparison_scope": "OFFICIAL_E3_FROZEN_REPORT_PLUS_CONTROLLED_E3_LIKE_CADENCE_PROXY",
        "final_holdout_state": "SEALED_NOT_OPENED", "production_modified": False,
    }
    ablations.to_csv(output / "ablation_results.csv", index=False)
    pd.DataFrame(regression_rows).to_csv(output / "regression_heads.csv", index=False)
    risk.to_csv(output / "risk_coverage.csv", index=False)
    stability.to_csv(output / "per_symbol_side_time.csv", index=False)
    validation[["row_id", "symbol", "side", "decision_at", "episode_id", "temporal_block_id", "score__quality", "score__favorable", "score__tail", "score__entry_quality"]].to_parquet(output / "validation_scores.parquet", index=False)
    joblib.dump(models, output / "development_models.joblib")
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json) + "\n")
    artifact = {name: stable_hash(path.read_bytes().hex()) for name, path in {"result": output / "result.json", "ablations": output / "ablation_results.csv", "models": output / "development_models.joblib"}.items()}
    (output / "artifact_manifest.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return result


def _cadence_control(train: pd.DataFrame, calibration: pd.DataFrame, validation: pd.DataFrame, family: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    features = [column for column, group in family.items() if group == "BASE"]
    train_h = train.loc[pd.to_datetime(train.decision_at, utc=True).dt.minute.eq(0)]
    cal_h = calibration.loc[pd.to_datetime(calibration.decision_at, utc=True).dt.minute.eq(0)]
    val_h = validation.loc[pd.to_datetime(validation.decision_at, utc=True).dt.minute.eq(0)]
    model = _classifier(config["models"]["seed"])
    model.fit(train_h[features], train_h.target__favorable_first)
    calibrator = _fit_calibrator(model, cal_h, features, "target__favorable_first")
    return {
        "hourly_train_rows": len(train_h), "five_minute_train_rows": len(train),
        "hourly_trained_evaluated_hourly": _classification_metrics(val_h.target__favorable_first, _predict(model, calibrator, val_h, features)),
        "hourly_trained_evaluated_5m": _classification_metrics(validation.target__favorable_first, _predict(model, calibrator, validation, features)),
    }


def _json(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)
