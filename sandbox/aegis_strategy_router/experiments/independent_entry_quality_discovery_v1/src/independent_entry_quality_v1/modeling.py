"""Frozen simple-model evaluation for entry-quality discovery."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from independent_entry_quality_v1.features import assert_feature_allowlist


def _linear_classifier(features: list[str], config: dict[str, Any]) -> Pipeline:
    spec = config["models"]["logistic"]
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=float(spec["C"]), penalty=spec["penalty"], max_iter=int(spec["max_iter"]),
            class_weight=spec["class_weight"], random_state=int(config["statistics"]["seed"]),
        )),
    ])


def _ridge(config: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=float(config["models"]["ridge"]["alpha"]))),
    ])


def _tree(config: dict[str, Any]) -> Pipeline:
    spec = config["models"]["shallow_tree"]
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=int(spec["max_depth"]), min_samples_leaf=int(spec["min_samples_leaf"]),
            random_state=int(config["statistics"]["seed"]),
        )),
    ])


def _calibrate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(x, y)
    return calibrated


def _classification_metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    prevalence = float(y.mean())
    constant = np.full(len(y), prevalence)
    positive = probability[y.to_numpy(int) == 1]
    negative = probability[y.to_numpy(int) == 0]
    p_value = float(mannwhitneyu(positive, negative, alternative="two-sided").pvalue)
    return {
        "n": len(y),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "constant_brier": float(brier_score_loss(y, constant)),
        "constant_log_loss": float(log_loss(y, constant, labels=[0, 1])),
        "ece": _ece(y.to_numpy(int), probability),
        "auc_mann_whitney_p_value": p_value,
    }


def _ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(probability)
    pieces = np.array_split(order, bins)
    return float(sum(
        len(piece) / len(y) * abs(float(y[piece].mean()) - float(probability[piece].mean()))
        for piece in pieces if len(piece)
    ))


def _calibration_rows(name: str, y: pd.Series, probability: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(probability)
    rows = []
    for index, piece in enumerate(np.array_split(order, 10), start=1):
        rows.append({
            "model": name, "decile": index, "n": len(piece),
            "mean_probability": float(probability[piece].mean()),
            "observed_rate": float(y.iloc[piece].mean()),
        })
    return rows


def run_experiment(
    *, dataset: pd.DataFrame, feature_dictionary: dict[str, Any], config: dict[str, Any], output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    feature_columns = sorted(item["name"] for item in feature_dictionary["features"])
    assert_feature_allowlist(feature_columns)
    groups: dict[str, list[str]] = {}
    for item in feature_dictionary["features"]:
        groups.setdefault(item["family"], []).append(item["name"])
    train = dataset.loc[dataset.split.eq("TRAIN")].copy()
    calibration = dataset.loc[dataset.split.eq("CALIBRATION")].copy()
    validation = dataset.loc[dataset.split.eq("VALIDATION")].copy()
    opportunity_train = train.loc[train.side.eq("LONG")]
    opportunity_cal = calibration.loc[calibration.side.eq("LONG")]
    opportunity_val = validation.loc[validation.side.eq("LONG")]
    opportunity_model = _linear_classifier(feature_columns, config)
    opportunity_model.fit(opportunity_train[feature_columns], opportunity_train["target__opportunity"])
    opportunity_model = _calibrate(
        opportunity_model, opportunity_cal[feature_columns], opportunity_cal["target__opportunity"]
    )
    opp_probability_state = opportunity_model.predict_proba(opportunity_val[feature_columns])[:, 1]
    opp_by_state = dict(zip(opportunity_val.market_state_group_id, opp_probability_state))
    direction_train = train.loc[train["target__opportunity"].eq(1)]
    direction_cal = calibration.loc[calibration["target__opportunity"].eq(1)]
    direction_val = validation.loc[validation["target__opportunity"].eq(1)]
    direction_model = _linear_classifier(feature_columns, config)
    direction_model.fit(direction_train[feature_columns], direction_train["target__favorable_first"])
    direction_model = _calibrate(
        direction_model, direction_cal[feature_columns], direction_cal["target__favorable_first"]
    )
    direction_probability = direction_model.predict_proba(validation[feature_columns])[:, 1]
    ridge = _ridge(config)
    ridge.fit(train[feature_columns], train["target__net_common_payoff_bps"])
    tree = _tree(config)
    tree.fit(train[feature_columns], train["target__net_positive"])
    validation = validation.copy()
    validation["score__opportunity"] = validation.market_state_group_id.map(opp_by_state)
    if validation["score__opportunity"].isna().any():
        raise ValueError("OPPORTUNITY_STATE_JOIN_MISSING")
    validation["score__direction"] = direction_probability
    validation["score__two_stage"] = validation["score__opportunity"] * validation["score__direction"]
    validation["score__ridge_net"] = ridge.predict(validation[feature_columns])
    validation["score__tree_net_positive"] = tree.predict_proba(validation[feature_columns])[:, 1]
    cards = {
        "opportunity_logistic": {
            "target": "target__opportunity", "fit_rows": len(opportunity_train),
            "calibration_rows": len(opportunity_cal),
            "validation": _classification_metrics(
                opportunity_val["target__opportunity"], opp_probability_state
            ),
        },
        "direction_logistic": {
            "target": "target__favorable_first", "conditioning": "target__opportunity == 1 in fit/calibration",
            "fit_rows": len(direction_train), "calibration_rows": len(direction_cal),
            "validation": _classification_metrics(
                direction_val["target__favorable_first"],
                direction_model.predict_proba(direction_val[feature_columns])[:, 1],
            ),
        },
        "joint_ridge": {"target": "target__net_common_payoff_bps", "fit_rows": len(train)},
        "joint_shallow_tree": {"target": "target__net_positive", "fit_rows": len(train)},
    }
    calibration_rows = _calibration_rows(
        "opportunity_logistic", opportunity_val["target__opportunity"], opp_probability_state
    ) + _calibration_rows(
        "direction_logistic", direction_val["target__favorable_first"],
        direction_model.predict_proba(direction_val[feature_columns])[:, 1],
    )
    risk_rows = []
    score_map = {
        "TWO_STAGE_LOGISTIC": "score__two_stage",
        "JOINT_RIDGE_NET": "score__ridge_net",
        "JOINT_SHALLOW_TREE": "score__tree_net_positive",
    }
    for model_name, score in score_map.items():
        risk_rows.extend(_risk_coverage(validation, score, model_name, config))
    risk = pd.DataFrame(risk_rows)
    primary = risk.loc[risk.model.eq("TWO_STAGE_LOGISTIC")].sort_values("coverage", ascending=False)
    required = primary.loc[primary.coverage.isin(config["success"]["monotonic_required_levels"])]
    ranking_spearman = float(spearmanr(-required.coverage, required.net_mean_bps).statistic)
    monotonic = bool(required.sort_values("coverage", ascending=False).net_mean_bps.is_monotonic_increasing)
    primary_level = float(config["success"]["primary_coverage"])
    primary_row = primary.loc[np.isclose(primary.coverage, primary_level)].iloc[0].to_dict()
    selected_count = max(1, int(math.ceil(len(validation) * primary_level)))
    selected = validation.nlargest(selected_count, "score__two_stage")
    stability = _stability(selected)
    baselines = _baselines(train, validation, config)
    ablations = _ablations(train, calibration, validation, groups, config)
    economics = _economics(selected, config)
    opportunity_signal = _signal_gate(cards["opportunity_logistic"]["validation"], config)
    direction_signal = _signal_gate(cards["direction_logistic"]["validation"], config)
    support = _support(train, validation, config)
    net_positive = bool(
        primary_row["net_mean_bps"] > 0
        and primary_row["net_block_ci_lower_bps"] > 0
        and primary_row["effective_groups"] >= config["support"]["minimum_selected_effective_groups"]
    )
    multi_stable = stability["positive_symbols"] >= config["success"]["minimum_positive_symbols"]
    temporal_stable = stability["positive_week_fraction"] >= config["success"]["minimum_positive_week_fraction"]
    long_short_stable = all(value > 0 for value in stability["net_by_side"].values()) and len(stability["net_by_side"]) == 2
    promising = all((
        support["train"], support["validation"], opportunity_signal, direction_signal,
        monotonic, ranking_spearman >= config["success"]["ranking_spearman_minimum"],
        net_positive, multi_stable, temporal_stable, long_short_stable,
    ))
    flags = {
        "ENTRY_QUALITY_DATASET_BUILT": True,
        "LEAKAGE_CHECK_PASSED": True,
        "TRAIN_SUPPORT_SUFFICIENT": support["train"],
        "VALIDATION_SUPPORT_SUFFICIENT": support["validation"],
        "OPPORTUNITY_MODEL_HAS_SIGNAL": opportunity_signal,
        "DIRECTION_MODEL_HAS_SIGNAL": direction_signal,
        "QUALITY_RANKING_MONOTONIC": monotonic and ranking_spearman >= config["success"]["ranking_spearman_minimum"],
        "OUT_OF_SAMPLE_EDGE_POSITIVE": net_positive,
        "NET_EDGE_POSITIVE": net_positive,
        "NET_EDGE_ABOVE_20BPS": bool(net_positive and primary_row["net_mean_bps"] > 20.0),
        "MULTI_SYMBOL_STABLE": multi_stable,
        "TEMPORALLY_STABLE": temporal_stable,
        "LONG_SHORT_STABLE": long_short_stable,
        "FINAL_HOLDOUT_OPENED": False,
        "FINAL_HOLDOUT_PASSED": False,
        "INDEPENDENT_ENTRY_QUALITY_PROMISING": promising,
        "READY_FOR_PROSPECTIVE_COLLECTION": promising,
        "READY_FOR_SHADOW": False,
        "READY_FOR_LIVE": False,
    }
    if promising:
        recommendation = "1_CONTINUE_ENTRY_QUALITY_EDGE_INDEPENDENT"
    elif opportunity_signal or direction_signal or flags["QUALITY_RANKING_MONOTONIC"]:
        recommendation = "2_PREDICTIVE_SIGNAL_NOT_YET_ECONOMIC"
    elif support["train"] and support["validation"]:
        recommendation = "3_REDESIGN_LABELS_OR_FEATURES_NO_SIGNAL"
    else:
        recommendation = "4_STOP_INSUFFICIENT_SUPPORT"
    result = {
        "schema": "independent-entry-quality-discovery-v1-result",
        "classification": "RETROSPECTIVE_DISCOVERY_WITH_TEMPORAL_OOS_VALIDATION",
        "support": support, "model_cards": cards, "primary_risk_coverage": primary.to_dict("records"),
        "primary_top10": primary_row, "ranking_spearman": ranking_spearman,
        "stability": stability, "baselines": baselines, "flags": flags,
        "recommendation": recommendation,
        "final_holdout_state": "SEALED_NOT_OPENED",
        "gradient_boosting_run": False, "production_modified": False,
    }
    pd.DataFrame(calibration_rows).to_csv(output / "calibration.csv", index=False)
    risk.to_csv(output / "risk_coverage.csv", index=False)
    pd.DataFrame(ablations).to_csv(output / "ablations.csv", index=False)
    pd.DataFrame(economics).to_csv(output / "economics_cost_stress.csv", index=False)
    pd.DataFrame(stability["detail"]).to_csv(output / "per_symbol_side_time_results.csv", index=False)
    (output / "baseline_report.json").write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
    (output / "model_cards.json").write_text(json.dumps(cards, indent=2, sort_keys=True) + "\n")
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n")
    joblib.dump({"opportunity": opportunity_model, "direction": direction_model, "ridge": ridge, "tree": tree, "features": feature_columns}, output / "development_models.joblib")
    validation.loc[:, ["row_id", "market_state_group_id", "symbol", "decision_at", "side", *score_map.values()]].to_parquet(output / "validation_scores.parquet", index=False)
    return result


def _signal_gate(metrics: dict[str, float], config: dict[str, Any]) -> bool:
    return bool(
        metrics["roc_auc"] >= config["success"]["classification_auc_minimum"]
        and metrics["ece"] <= config["success"]["classification_ece_maximum"]
        and metrics["log_loss"] < metrics["constant_log_loss"]
    )


def _risk_coverage(frame: pd.DataFrame, score: str, model: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    ordered = frame.sort_values(score, ascending=False, kind="mergesort")
    for coverage in config["coverage_levels"]:
        count = max(1, int(math.ceil(len(ordered) * float(coverage))))
        selected = ordered.head(count)
        ci = _block_bootstrap(selected, "target__net_common_payoff_bps", config)
        tail_count = max(1, int(math.ceil(len(selected) * 0.05)))
        rows.append({
            "model": model, "coverage": float(coverage), "rows": len(selected),
            "effective_groups": selected.temporal_block_id.nunique(),
            "favorable_first_rate": float(selected["target__favorable_first"].mean()),
            "mfe_mean_bps": float(selected["target__mfe_bps"].mean()),
            "mae_mean_bps": float(selected["target__mae_bps"].mean()),
            "gross_mean_bps": float(selected["target__gross_common_payoff_bps"].mean()),
            "net_mean_bps": float(selected["target__net_common_payoff_bps"].mean()),
            "latency_net_mean_bps": float(selected["target__latency_stressed_net_bps"].mean()),
            "tail_mae_bps": float(selected["target__mae_bps"].nlargest(tail_count).mean()),
            "expected_shortfall_net_bps": float(selected["target__net_common_payoff_bps"].nsmallest(tail_count).mean()),
            "net_block_ci_lower_bps": ci[0], "net_block_ci_upper_bps": ci[1],
        })
    return rows


def _block_bootstrap(frame: pd.DataFrame, column: str, config: dict[str, Any]) -> tuple[float, float]:
    grouped = frame.groupby("temporal_block_id")[column].mean().to_numpy(float)
    samples = int(config["statistics"]["block_bootstrap_samples"])
    rng = np.random.default_rng(int(config["statistics"]["seed"]))
    draws = np.empty(samples)
    for start in range(0, samples, 500):
        size = min(500, samples - start)
        indices = rng.integers(0, len(grouped), size=(size, len(grouped)))
        draws[start:start + size] = grouped[indices].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _support(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    rule = config["support"]
    result = {
        "train_rows": len(train), "train_hour_groups": train.temporal_block_id.nunique(),
        "train_symbols": train.symbol.nunique(), "validation_rows": len(validation),
        "validation_hour_groups": validation.temporal_block_id.nunique(),
        "validation_symbols": validation.symbol.nunique(),
    }
    result["train"] = bool(
        result["train_rows"] >= rule["minimum_train_rows"]
        and result["train_hour_groups"] >= rule["minimum_train_hour_groups"]
        and result["train_symbols"] >= rule["minimum_symbols"]
    )
    result["validation"] = bool(
        result["validation_rows"] >= rule["minimum_validation_rows"]
        and result["validation_hour_groups"] >= rule["minimum_validation_hour_groups"]
        and result["validation_symbols"] >= rule["minimum_symbols"]
    )
    return result


def _baselines(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    persistence = validation.loc[validation["feature__tf15m__directional_return_3_bps"] > 0]
    threshold = float(train["feature__tf15m__atr_percentile_96"].quantile(0.9))
    volatility = validation.loc[validation["feature__tf15m__atr_percentile_96"] >= threshold]
    rng = np.random.default_rng(int(config["statistics"]["seed"]))
    random = validation.iloc[np.sort(rng.choice(len(validation), size=max(1, len(validation) // 10), replace=False))]
    def summary(frame: pd.DataFrame) -> dict[str, float]:
        return {"n": len(frame), "favorable_first": float(frame["target__favorable_first"].mean()), "gross_bps": float(frame["target__gross_common_payoff_bps"].mean()), "net_bps": float(frame["target__net_common_payoff_bps"].mean())}
    return {
        "validation_unconditional": summary(validation),
        "validation_by_side": {side: summary(group) for side, group in validation.groupby("side")},
        "directional_persistence": summary(persistence),
        "top_decile_volatility": {**summary(volatility), "train_threshold": threshold},
        "random_10pct": summary(random),
    }


def _ablations(train: pd.DataFrame, calibration: pd.DataFrame, validation: pd.DataFrame, groups: dict[str, list[str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    families = list(config["feature_ablations"])
    for family in families:
        columns = sorted({column for values in groups.values() for column in values} if family == "FULL" else groups.get(family, []))
        if not columns:
            rows.append({"family": family, "status": "NO_AVAILABLE_FEATURES"})
            continue
        fit = train.loc[train["target__opportunity"].eq(1)]
        cal = calibration.loc[calibration["target__opportunity"].eq(1)]
        val = validation.loc[validation["target__opportunity"].eq(1)]
        model = _linear_classifier(columns, config)
        model.fit(fit[columns], fit["target__favorable_first"])
        model = _calibrate(model, cal[columns], cal["target__favorable_first"])
        probability = model.predict_proba(val[columns])[:, 1]
        metrics = _classification_metrics(val["target__favorable_first"], probability)
        scored = val.loc[:, ["symbol", "decision_at", "target__favorable_first"]].copy()
        scored["probability"] = probability
        scored["week"] = scored.decision_at.dt.strftime("%Y-W%V")
        symbol_aucs = [_safe_auc(group.target__favorable_first, group.probability) for _, group in scored.groupby("symbol")]
        week_aucs = [_safe_auc(group.target__favorable_first, group.probability) for _, group in scored.groupby("week")]
        rows.append({
            "family": family, "status": "EVALUATED", **metrics,
            "symbols_auc_above_half": sum(value > 0.5 for value in symbol_aucs if math.isfinite(value)),
            "symbols_evaluated": sum(math.isfinite(value) for value in symbol_aucs),
            "symbol_auc_std": float(np.nanstd(symbol_aucs)),
            "weeks_auc_above_half": sum(value > 0.5 for value in week_aucs if math.isfinite(value)),
            "weeks_evaluated": sum(math.isfinite(value) for value in week_aucs),
            "week_auc_std": float(np.nanstd(week_aucs)),
        })
    evaluated = [row for row in rows if row.get("status") == "EVALUATED"]
    adjusted = _benjamini_hochberg([row["auc_mann_whitney_p_value"] for row in evaluated])
    for row, q_value in zip(evaluated, adjusted):
        row["fdr_q_value"] = q_value
        row["fdr_pass"] = q_value <= float(config["statistics"]["fdr_alpha"])
    return rows


def _safe_auc(y: pd.Series, probability: pd.Series) -> float:
    return float(roc_auc_score(y, probability)) if y.nunique() == 2 else math.nan


def _benjamini_hochberg(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def _stability(selected: pd.DataFrame) -> dict[str, Any]:
    data = selected.copy()
    data["week"] = data.decision_at.dt.strftime("%Y-W%V")
    data["volatility_regime"] = pd.cut(
        data["feature__tf15m__atr_percentile_96"], [-np.inf, 0.25, 0.75, np.inf],
        labels=["LOW", "MID", "HIGH"],
    ).astype(str)
    data["btc_regime"] = np.where(
        data["feature__cross__btcusdt__tf1h__directional_return_3_bps"] * data["feature__context__side_sign"] >= 0,
        "BTC_UP", "BTC_DOWN",
    )
    detail = []
    for dimension in ("symbol", "side", "week", "volatility_regime", "btc_regime"):
        for value, group in data.groupby(dimension, observed=True):
            detail.append({"dimension": dimension, "value": str(value), "n": len(group), "net_bps": float(group["target__net_common_payoff_bps"].mean()), "gross_bps": float(group["target__gross_common_payoff_bps"].mean()), "favorable_first": float(group["target__favorable_first"].mean())})
    symbol_net = data.groupby("symbol")["target__net_common_payoff_bps"].mean()
    week_net = data.groupby("week")["target__net_common_payoff_bps"].mean()
    return {
        "positive_symbols": int((symbol_net > 0).sum()),
        "negative_symbols": int((symbol_net < 0).sum()),
        "positive_week_fraction": float((week_net > 0).mean()),
        "net_by_side": data.groupby("side")["target__net_common_payoff_bps"].mean().to_dict(),
        "leave_one_symbol_out_net_bps": {symbol: float(data.loc[data.symbol.ne(symbol), "target__net_common_payoff_bps"].mean()) for symbol in sorted(data.symbol.unique())},
        "detail": detail,
    }


def _economics(selected: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    gross = selected["target__gross_common_payoff_bps"]
    return [{
        "cost_bps": float(cost), "mean_net_bps": float((gross - float(cost)).mean()),
        "median_net_bps": float((gross - float(cost)).median()),
        "latency_adjusted_mean_bps": float((gross - float(cost) - selected["target__latency_shortfall_bps"]).mean()),
    } for cost in config["cost_scenarios_bps"]]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)
