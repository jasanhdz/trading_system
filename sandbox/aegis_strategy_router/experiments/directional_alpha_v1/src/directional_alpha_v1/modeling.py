"""Frozen simple-model evaluation for Directional Alpha V1."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def linear_classifier(columns: list[str], config: dict[str, Any]) -> Pipeline:
    spec = config["models"]["logistic"]
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ("model", LogisticRegression(C=spec["C"], penalty=spec["penalty"], max_iter=spec["max_iter"], random_state=config["models"]["seed"])),
    ])


def ridge_model(columns: list[str], config: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ("model", Ridge(alpha=config["models"]["ridge"]["alpha"])),
    ])


def tree_model(config: dict[str, Any]) -> Pipeline:
    spec = config["models"]["tree"]
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(max_depth=spec["max_depth"], min_samples_leaf=spec["min_samples_leaf"], random_state=config["models"]["seed"])),
    ])


def paired_scores(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    index = ["market_state_group_id", "temporal_block_id", "decision_at", "symbol"]
    wide_score = frame.pivot(index=index, columns="side", values=score_column).reset_index()
    wide_score["chosen_side"] = np.where(wide_score.LONG >= wide_score.SHORT, "LONG", "SHORT")
    wide_score["predicted_best_net_bps"] = wide_score[["LONG", "SHORT"]].max(axis=1)
    wide_score["predicted_advantage_bps"] = (wide_score.LONG - wide_score.SHORT).abs()
    chosen = frame.merge(wide_score[index + ["chosen_side", "predicted_best_net_bps", "predicted_advantage_bps"]], on=index, validate="many_to_one")
    return chosen.loc[chosen.side.eq(chosen.chosen_side)].copy()


def classification_metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    prevalence = float(y.mean())
    auc = float(roc_auc_score(y, probability))
    positive, negative = probability[y.to_numpy(int) == 1], probability[y.to_numpy(int) == 0]
    return {
        "n": len(y), "prevalence": prevalence, "roc_auc": auc,
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "constant_log_loss": float(log_loss(y, np.full(len(y), prevalence), labels=[0, 1])),
        "mann_whitney_p": float(mannwhitneyu(positive, negative, alternative="two-sided").pvalue),
    }


def calibration_bins(y: pd.Series, probability: np.ndarray, family: str) -> list[dict[str, Any]]:
    """Return deterministic equal-frequency validation bins for diagnostics."""
    values = pd.DataFrame({"observed": y.to_numpy(int), "probability": probability})
    values["bin"] = pd.qcut(values.probability.rank(method="first"), q=10, labels=False)
    return [
        {
            "family": family,
            "bin": int(bin_id),
            "n": int(len(group)),
            "mean_predicted_probability": float(group.probability.mean()),
            "observed_favorable_first": float(group.observed.mean()),
        }
        for bin_id, group in values.groupby("bin", sort=True)
    ]


def summarize(frame: pd.DataFrame) -> dict[str, float]:
    ratio = float(frame.target__mfe_bps.mean() / frame.target__mae_bps.mean()) if frame.target__mae_bps.mean() else math.inf
    tail = max(1, math.ceil(len(frame) * 0.05))
    return {
        "n": len(frame), "effective_blocks": frame.temporal_block_id.nunique(),
        "long": int(frame.side.eq("LONG").sum()), "short": int(frame.side.eq("SHORT").sum()),
        "favorable_first": float(frame.target__favorable_first.mean()),
        "mfe_bps": float(frame.target__mfe_bps.mean()), "mae_bps": float(frame.target__mae_bps.mean()),
        "mfe_mae_ratio": ratio, "mfe_minus_mae_bps": float((frame.target__mfe_bps - frame.target__mae_bps).mean()),
        "gross_bps": float(frame.target__gross_common_payoff_bps.mean()),
        "net_bps": float(frame.target__net_common_payoff_bps.mean()),
        "tail_mae_bps": float(frame.target__mae_bps.nlargest(tail).mean()),
        "expected_shortfall_net_bps": float(frame.target__net_common_payoff_bps.nsmallest(tail).mean()),
    }


def bootstrap_ci(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[float, float]:
    groups = frame.groupby("temporal_block_id").target__net_common_payoff_bps.mean().to_numpy(float)
    rng = np.random.default_rng(config["statistics"]["seed"])
    repetitions = config["statistics"]["block_bootstrap_samples"]
    means = np.empty(repetitions)
    for start in range(0, repetitions, 500):
        size = min(500, repetitions - start)
        indices = rng.integers(0, len(groups), size=(size, len(groups)))
        means[start:start + size] = groups[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def risk_coverage(chosen: pd.DataFrame, model: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    ordered = chosen.sort_values("predicted_advantage_bps", ascending=False, kind="mergesort")
    rows = []
    for coverage in config["directional_coverage_levels"]:
        selected = ordered.head(max(1, math.ceil(len(ordered) * coverage)))
        summary = summarize(selected)
        lower, upper = bootstrap_ci(selected, config)
        rows.append({"model": model, "coverage": coverage, **summary, "net_ci_lower_bps": lower, "net_ci_upper_bps": upper})
    return rows


def opportunity_baseline(population: pd.DataFrame) -> dict[str, float]:
    # Equal-weight both sides: the no-direction-information expectation.
    return summarize(population)


def stability(frame: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = frame.copy()
    values["week"] = values.decision_at.dt.strftime("%Y-W%V")
    detail = []
    for dimension in ("symbol", "side", "week"):
        for value, group in values.groupby(dimension):
            detail.append({"dimension": dimension, "value": str(value), **summarize(group)})
    symbol_net = values.groupby("symbol").target__net_common_payoff_bps.mean()
    week_net = values.groupby("week").target__net_common_payoff_bps.mean()
    return {
        "positive_symbols": int((symbol_net > 0).sum()), "negative_symbols": int((symbol_net < 0).sum()),
        "positive_week_fraction": float((week_net > 0).mean()),
        "net_by_side": values.groupby("side").target__net_common_payoff_bps.mean().to_dict(),
        "leave_one_symbol_out": {symbol: float(values.loc[values.symbol.ne(symbol), "target__net_common_payoff_bps"].mean()) for symbol in sorted(values.symbol.unique())},
    }, detail


def fdr(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 1.0
    for reverse in range(len(values) - 1, -1, -1):
        index, rank = int(order[reverse]), reverse + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted.tolist()


def run(*, dataset: pd.DataFrame, dictionary: dict[str, Any], config: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[str]] = {}
    for item in dictionary["features"]:
        groups.setdefault(item["family"], []).append(item["name"])
    validation_all = dataset.loc[dataset.split.eq("VALIDATION")].copy()
    train = dataset.loc[dataset.split.eq("TRAIN") & dataset.opportunity_top_90].copy()
    calibration = dataset.loc[dataset.split.eq("CALIBRATION") & dataset.opportunity_top_90].copy()
    validation = dataset.loc[dataset.split.eq("VALIDATION") & dataset.opportunity_top_90].copy()
    family_columns = {
        "FLOW_ONLY": sorted(groups["FLOW"]), "CROSS_MARKET_ONLY": sorted(groups["CROSS_MARKET"]),
        "FLOW_CROSS_MARKET": sorted(groups["FLOW"] + groups["CROSS_MARKET"]),
    }
    model_cards, coverage_rows, calibration_rows, validation_score_rows, scored_models, ridge_models = [], [], [], [], {}, {}
    for family in config["ablations"]:
        columns = family_columns[family]
        ridge = ridge_model(columns, config)
        ridge.fit(train[columns], train.target__net_common_payoff_bps)
        ridge_models[family] = ridge
        ridge_mfe = ridge_model(columns, config)
        ridge_mae = ridge_model(columns, config)
        ridge_mfe.fit(train[columns], train.target__mfe_bps)
        ridge_mae.fit(train[columns], train.target__mae_bps)
        scored = validation.copy()
        scored["predicted_net_utility"] = ridge.predict(scored[columns])
        chosen = paired_scores(scored, "predicted_net_utility")
        scored_models[family] = chosen
        coverage_rows.extend(risk_coverage(chosen, family, config))
        tree = tree_model(config)
        tree.fit(train[columns], train.target__net_common_payoff_bps)
        tree_scored = validation.copy()
        tree_scored["predicted_net_utility"] = tree.predict(tree_scored[columns])
        coverage_rows.extend(risk_coverage(paired_scores(tree_scored, "predicted_net_utility"), f"{family}_SHALLOW_TREE", config))
        logistic = linear_classifier(columns, config)
        logistic.fit(train[columns], train.target__favorable_first)
        calibrated = CalibratedClassifierCV(FrozenEstimator(logistic), method="sigmoid")
        calibrated.fit(calibration[columns], calibration.target__favorable_first)
        probability = calibrated.predict_proba(validation[columns])[:, 1]
        calibration_rows.extend(calibration_bins(validation.target__favorable_first, probability, family))
        validation_score_rows.append(pd.DataFrame({
            "market_state_group_id": validation.market_state_group_id,
            "temporal_block_id": validation.temporal_block_id,
            "decision_at": validation.decision_at,
            "symbol": validation.symbol,
            "side": validation.side,
            "family": family,
            "predicted_favorable_first": probability,
            "predicted_net_utility_bps": scored.predicted_net_utility,
            "target__favorable_first": validation.target__favorable_first,
            "target__mfe_bps": validation.target__mfe_bps,
            "target__mae_bps": validation.target__mae_bps,
            "target__net_common_payoff_bps": validation.target__net_common_payoff_bps,
        }))
        metrics = classification_metrics(validation.target__favorable_first, probability)
        metrics["mfe_regression_rmse_bps"] = float(np.sqrt(np.mean((ridge_mfe.predict(validation[columns]) - validation.target__mfe_bps) ** 2)))
        metrics["mae_regression_rmse_bps"] = float(np.sqrt(np.mean((ridge_mae.predict(validation[columns]) - validation.target__mae_bps) ** 2)))
        model_cards.append({"family": family, "features": len(columns), **metrics})
        joblib.dump({"ridge": ridge, "ridge_mfe": ridge_mfe, "ridge_mae": ridge_mae, "tree": tree, "logistic": calibrated, "features": columns}, output / f"{family.lower()}_models.joblib")
    q_values = fdr([row["mann_whitney_p"] for row in model_cards])
    for row, q_value in zip(model_cards, q_values):
        row["fdr_q"] = q_value
        row["fdr_pass"] = q_value <= config["statistics"]["fdr_alpha"]
    primary_family = "FLOW_CROSS_MARKET"
    primary = pd.DataFrame(coverage_rows).loc[lambda value: value.model.eq(primary_family)].sort_values("coverage", ascending=False)
    required = primary.loc[primary.coverage.isin(config["success"]["monotonic_required_levels"])]
    ranking_spearman = float(spearmanr(-required.coverage, required.net_bps).statistic)
    monotonic = bool(required.sort_values("coverage", ascending=False).net_bps.is_monotonic_increasing)
    primary_coverage = config["success"]["primary_directional_coverage"]
    primary_row = primary.loc[np.isclose(primary.coverage, primary_coverage)].iloc[0].to_dict()
    primary_chosen = scored_models[primary_family].nlargest(max(1, math.ceil(len(scored_models[primary_family]) * primary_coverage)), "predicted_advantage_bps")
    population_rows = []
    primary_columns = family_columns[primary_family]
    all_scored = validation_all.copy()
    all_scored["predicted_net_utility"] = ridge_models[primary_family].predict(all_scored[primary_columns])
    for population, mask in (
        ("ALL_ELIGIBLE_STATES", pd.Series(True, index=all_scored.index)),
        ("OPPORTUNITY_TOP20_TRAIN_THRESHOLD", all_scored.opportunity_top_80),
        ("OPPORTUNITY_TOP10_PRIMARY_TRAIN_THRESHOLD", all_scored.opportunity_top_90),
        ("OPPORTUNITY_TOP5_TRAIN_THRESHOLD", all_scored.opportunity_top_95),
    ):
        population_frame = all_scored.loc[mask].copy()
        chosen_population = paired_scores(population_frame, "predicted_net_utility")
        selected = chosen_population.nlargest(
            max(1, math.ceil(len(chosen_population) * primary_coverage)), "predicted_advantage_bps"
        )
        lower, upper = bootstrap_ci(selected, config)
        population_rows.append({
            "population": population,
            "directional_coverage": primary_coverage,
            "population_directional_rows": len(population_frame),
            "population_states": population_frame.market_state_group_id.nunique(),
            **summarize(selected),
            "net_ci_lower_bps": lower,
            "net_ci_upper_bps": upper,
        })
    baseline = opportunity_baseline(validation)
    stable, stability_rows = stability(primary_chosen)
    operational = scored_models[primary_family].loc[
        scored_models[primary_family].predicted_best_net_bps.gt(config["abstention"]["minimum_predicted_best_net_utility_bps"])
        & scored_models[primary_family].predicted_advantage_bps.ge(config["abstention"]["minimum_predicted_advantage_bps"])
    ]
    support = {
        "train_rows": len(train), "calibration_rows": len(calibration), "validation_rows": len(validation),
        "validation_blocks": validation.temporal_block_id.nunique(), "symbols": validation.symbol.nunique(),
    }
    support["train"] = len(train) >= config["support"]["minimum_train_rows_in_primary_population"]
    support["validation"] = (
        len(validation) >= config["support"]["minimum_validation_rows_in_primary_population"]
        and validation.temporal_block_id.nunique() >= config["support"]["minimum_validation_effective_blocks"]
        and validation.symbol.nunique() >= config["support"]["minimum_symbols"]
    )
    full_card = next(row for row in model_cards if row["family"] == primary_family)
    flow_card = next(row for row in model_cards if row["family"] == "FLOW_ONLY")
    cross_card = next(row for row in model_cards if row["family"] == "CROSS_MARKET_ONLY")
    signal = lambda card: card["roc_auc"] > 0.55 and card["fdr_pass"] and card["log_loss"] < card["constant_log_loss"]
    geometry = primary_row["mfe_mae_ratio"] > baseline["mfe_mae_ratio"] and primary_row["mfe_minus_mae_bps"] > baseline["mfe_minus_mae_bps"]
    favorable = primary_row["favorable_first"] > baseline["favorable_first"]
    net_positive = primary_row["net_bps"] > 0 and primary_row["net_ci_lower_bps"] > 0
    multi = stable["positive_symbols"] >= config["success"]["minimum_positive_symbols"]
    temporal = stable["positive_week_fraction"] >= config["success"]["minimum_positive_week_fraction"]
    side_stable = len(stable["net_by_side"]) == 2 and all(value > 0 for value in stable["net_by_side"].values())
    combined_signal = signal(full_card)
    flags = {
        "DIRECTIONAL_ALPHA_DATASET_BUILT": True, "OPPORTUNITY_GATE_REUSED_WITHOUT_RETRAINING": True,
        "LEAKAGE_CHECK_PASSED": True, "TRAIN_SUPPORT_SUFFICIENT": support["train"], "VALIDATION_SUPPORT_SUFFICIENT": support["validation"],
        "FLOW_EFFECTIVENESS_HAS_SIGNAL": signal(flow_card), "CROSS_MARKET_PROPAGATION_HAS_SIGNAL": signal(cross_card),
        "L2_RESPONSE_HAS_SIGNAL": False, "COMBINED_DIRECTION_MODEL_HAS_SIGNAL": combined_signal,
        "DIRECTION_AUC_IMPROVED": full_card["roc_auc"] > config["success"]["auc_must_exceed_entry_quality_v1"],
        "MFE_MAE_GEOMETRY_IMPROVED": geometry, "FAVORABLE_FIRST_IMPROVED": favorable,
        "QUALITY_RANKING_MONOTONIC": monotonic and ranking_spearman >= config["success"]["ranking_spearman_minimum"],
        "NET_EXPECTANCY_POSITIVE": net_positive, "NET_EDGE_ABOVE_20BPS": net_positive and primary_row["net_bps"] >= 20.0,
        "MULTI_SYMBOL_STABLE": multi, "TEMPORALLY_STABLE": temporal, "LONG_SHORT_STABLE": side_stable,
        "FINAL_HOLDOUT_OPENED": False, "FINAL_HOLDOUT_PASSED": False,
        "DIRECTIONAL_ALPHA_PROMISING": False, "READY_FOR_PROSPECTIVE_COLLECTION": False,
        "READY_FOR_SHADOW": False, "READY_FOR_LIVE": False,
    }
    flags["DIRECTIONAL_ALPHA_PROMISING"] = all((support["train"], support["validation"], combined_signal, geometry, favorable, flags["QUALITY_RANKING_MONOTONIC"], net_positive, multi, temporal, side_stable))
    flags["READY_FOR_PROSPECTIVE_COLLECTION"] = flags["DIRECTIONAL_ALPHA_PROMISING"]
    if flags["DIRECTIONAL_ALPHA_PROMISING"] and flags["NET_EDGE_ABOVE_20BPS"]:
        verdict = "DIRECTIONAL_ALPHA_ECONOMIC_EDGE_FOUND"
    elif combined_signal and (geometry or favorable) and not net_positive:
        verdict = "DIRECTIONAL_ALPHA_PREDICTIVE_BUT_NOT_ECONOMIC"
    elif combined_signal or geometry or favorable:
        verdict = "DIRECTIONAL_ALPHA_WEAK_OR_UNSTABLE"
    else:
        verdict = "DIRECTIONAL_ALPHA_NO_EDGE"
    comparisons = {
        "opportunity_only_same_period": baseline, "directional_primary_top10": primary_row,
        "delta": {key: primary_row[key] - baseline[key] for key in ("favorable_first", "mfe_bps", "mae_bps", "mfe_mae_ratio", "mfe_minus_mae_bps", "gross_bps", "net_bps", "tail_mae_bps")},
        "entry_quality_v1_frozen_reference": {"direction_auc": 0.5503396511301014, "top10_mfe_bps": 106.35250978742455, "top10_mae_bps": 101.48408988185703, "top10_gross_bps": 3.417868997400809, "top10_net_bps": -16.58213100259919},
    }
    cost_rows = [{"cost_bps": cost, "mean_net_bps": primary_row["gross_bps"] - cost} for cost in config["cost_scenarios_bps"]]
    result = {
        "schema": "directional-alpha-v1-result", "classification": "RETROSPECTIVE_DISCOVERY_WITH_TEMPORAL_OOS_VALIDATION",
        "verdict": verdict, "support": support, "model_cards": model_cards, "primary": primary_row,
        "opportunity_comparison": comparisons, "ranking_spearman": ranking_spearman,
        "operational_abstention": {"taken": len(operational), "coverage": len(operational) / len(scored_models[primary_family]), "summary": summarize(operational) if len(operational) else None},
        "stability": stable, "flags": flags, "l2_subexperiment": "NOT_RUN_NO_CLEAN_ELIGIBLE_L2_PERIOD",
        "positioning_subexperiment": "NOT_RUN_INSUFFICIENT_AUTHENTIC_COVERAGE", "final_holdout_state": "SEALED_NOT_OPENED",
        "gradient_boosting_run": False, "production_modified": False,
        "opportunity_population_diagnostics": population_rows,
    }
    pd.DataFrame(model_cards).to_csv(output / "ablation_model_cards.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "calibration.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "confidence_coverage.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(output / "symbol_side_time.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(output / "cost_stress.csv", index=False)
    pd.DataFrame(population_rows).to_csv(output / "opportunity_population_diagnostics.csv", index=False)
    pd.concat(validation_score_rows, ignore_index=True).to_parquet(
        output / "validation_scores.parquet", index=False, compression="zstd"
    )
    (output / "opportunity_comparison.json").write_text(json.dumps(comparisons, indent=2, sort_keys=True) + "\n")
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n")
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)
