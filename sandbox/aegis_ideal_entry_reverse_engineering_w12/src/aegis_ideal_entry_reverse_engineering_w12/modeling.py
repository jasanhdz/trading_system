"""Frozen W12 model formulations, ranked metrics, and negative controls."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_COMPLEXITY = {"LOGISTIC": 0, "HIST_GB": 1, "RANDOM_FOREST": 2, "HIST_GB_REGRESSION": 1, "OPPORTUNITY_THEN_SIDE": 1}


@dataclass
class FrozenCandidate:
    name: str
    formulation: str
    side: str
    horizon_minutes: int
    model: Any
    feature_names: tuple[str, ...]
    thresholds: dict[int, float]
    validation_primary_net14_bps: float
    validation_primary_precision: float

    def score(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = frame.loc[:, list(self.feature_names)]
        if self.formulation == "QUALITY_REGRESSION":
            raw = np.asarray(self.model.predict(matrix), dtype=float)
            score = np.clip(raw / 100.0, 0.0, 1.0)
            return score, np.repeat(self.side, len(frame))
        if self.formulation == "OPPORTUNITY_THEN_SIDE":
            opportunity, direction = self.model
            p_opportunity = opportunity.predict_proba(matrix)[:, 1]
            p_long = direction.predict_proba(matrix)[:, 1]
            side = np.where(p_long >= 0.5, "LONG", "SHORT")
            confidence = np.maximum(p_long, 1.0 - p_long)
            return p_opportunity * confidence, side
        return self.model.predict_proba(matrix)[:, 1], np.repeat(self.side, len(frame))


def _classification_pipeline(spec: Mapping[str, Any], seed: int, n_jobs: int = 1) -> Pipeline:
    name = spec["name"]
    if name == "LOGISTIC":
        estimator = LogisticRegression(
            C=float(spec["c"]), class_weight=spec["class_weight"], max_iter=2000,
            random_state=seed, solver="liblinear",
        )
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler()), ("model", estimator)]
    elif name == "HIST_GB":
        estimator = HistGradientBoostingClassifier(
            max_iter=int(spec["max_iter"]), max_leaf_nodes=int(spec["max_leaf_nodes"]),
            learning_rate=float(spec["learning_rate"]), random_state=seed,
            class_weight="balanced",
        )
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", estimator)]
    elif name == "RANDOM_FOREST":
        estimator = RandomForestClassifier(
            n_estimators=int(spec["trees"]), max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]), class_weight=spec["class_weight"],
            random_state=seed, n_jobs=n_jobs,
        )
        steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", estimator)]
    else:
        raise ValueError(f"unknown frozen model: {name}")
    return Pipeline(steps)


def regression_pipeline(spec: Mapping[str, Any], seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", HistGradientBoostingRegressor(
            max_iter=int(spec["max_iter"]), max_leaf_nodes=int(spec["max_leaf_nodes"]),
            learning_rate=float(spec["learning_rate"]), random_state=seed,
        )),
    ])


def temporal_negative_mask(labels: pd.DataFrame, minutes: int) -> np.ndarray:
    """Keep negatives only when farther than the frozen radius from any ideal timestamp."""
    if "symbol" in labels.columns and labels["symbol"].nunique() > 1:
        keep = np.zeros(len(labels), dtype=bool)
        for _, positions in labels.groupby("symbol", sort=True).groups.items():
            positional = labels.index.get_indexer(positions)
            keep[positional] = temporal_negative_mask(labels.loc[positions].reset_index(drop=True), minutes)
        return keep
    times = labels["decision_at"].astype("int64").to_numpy()
    positive_times = np.sort(times[labels["majority_ideal"].to_numpy(bool)])
    keep = np.ones(len(labels), dtype=bool)
    if not len(positive_times):
        return keep
    positions = np.searchsorted(positive_times, times)
    distance = np.full(len(times), np.iinfo(np.int64).max, dtype=np.int64)
    left = positions > 0
    right = positions < len(positive_times)
    distance[left] = np.minimum(distance[left], np.abs(times[left] - positive_times[positions[left] - 1]))
    distance[right] = np.minimum(distance[right], np.abs(times[right] - positive_times[positions[right]]))
    keep[~labels["majority_ideal"].to_numpy(bool)] = distance[~labels["majority_ideal"].to_numpy(bool)] > minutes * 60 * 1_000_000_000
    return keep


def ranked_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    gross_bps: np.ndarray,
    timestamps: pd.Series,
    symbols: pd.Series,
    percentiles: list[int],
    thresholds: dict[int, float] | None = None,
) -> tuple[dict[str, Any], dict[int, float], pd.DataFrame]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    gross = np.asarray(gross_bps, dtype=float)
    valid = np.isfinite(scores) & np.isfinite(gross)
    labels, scores, gross = labels[valid], scores[valid], gross[valid]
    ts = pd.Series(timestamps).reset_index(drop=True).loc[valid].reset_index(drop=True)
    syms = pd.Series(symbols).reset_index(drop=True).loc[valid].reset_index(drop=True)
    frozen = thresholds or {percent: float(np.quantile(scores, 1.0 - percent / 100.0)) for percent in percentiles}
    cuts = {}
    selected_rows = []
    prevalence = float(labels.mean()) if len(labels) else 0.0
    for percent in percentiles:
        selected = scores >= frozen[percent]
        selected_gross = gross[selected]
        precision = float(labels[selected].mean()) if selected.any() else None
        cuts[str(percent)] = {
            "threshold": frozen[percent], "selected": int(selected.sum()),
            "selection_fraction": float(selected.mean()) if len(selected) else 0.0,
            "precision": precision,
            "precision_lift": precision / prevalence if precision is not None and prevalence else None,
            "gross_mean_bps": float(selected_gross.mean()) if len(selected_gross) else None,
            "net14_mean_bps": float(selected_gross.mean() - 14.0) if len(selected_gross) else None,
            "net20_mean_bps": float(selected_gross.mean() - 20.0) if len(selected_gross) else None,
            "net30_mean_bps": float(selected_gross.mean() - 30.0) if len(selected_gross) else None,
        }
        if percent == 2:
            positions = np.flatnonzero(valid)[selected]
            selected_rows.append(pd.DataFrame({
                "row_position": positions, "decision_at": ts[selected].to_numpy(),
                "symbol": syms[selected].to_numpy(), "score": scores[selected],
                "actual_ideal": labels[selected], "gross_bps": selected_gross,
            }))
    metrics = {
        "rows": len(labels), "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "brier": float(brier_score_loss(labels, np.clip(scores, 0, 1))) if len(labels) else None,
        "precision_at_0_5": float(precision_score(labels, scores >= 0.5, zero_division=0)) if len(labels) else None,
        "recall_at_0_5": float(recall_score(labels, scores >= 0.5, zero_division=0)) if len(labels) else None,
        "top": cuts,
    }
    return metrics, frozen, pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()


def fit_direct_candidates(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str],
    config: Mapping[str, Any],
    *,
    include_two_stage: bool = True,
) -> tuple[list[FrozenCandidate], pd.DataFrame]:
    candidates: list[FrozenCandidate] = []
    reports = []
    seed = int(config["seed"])
    percentiles = [int(value) for value in config["selection"]["top_percentiles"]]
    for horizon in config["teachers"]["horizons_minutes"]:
        for side in config["teachers"]["sides"]:
            train = discovery[(discovery["horizon_minutes"].eq(horizon)) & (discovery["side"].eq(side))].copy()
            score_set = validation[(validation["horizon_minutes"].eq(horizon)) & (validation["side"].eq(side))].copy()
            keep = train["zone_best"].to_numpy(bool) | (~train["majority_ideal"].to_numpy(bool) & temporal_negative_mask(train, int(config["zones"]["negative_exclusion_minutes"])))
            train = train.loc[keep]
            target = train["zone_best"].to_numpy(int)
            if len(np.unique(target)) < 2 or target.sum() < 20:
                continue
            for spec in config["models"]["classification"]:
                model = _classification_pipeline(spec, seed, n_jobs=1).fit(train[feature_names], target)
                scores = model.predict_proba(score_set[feature_names])[:, 1]
                metrics, thresholds, _ = ranked_metrics(
                    score_set["zone_best"], scores, score_set["policy_gross_bps"],
                    score_set["decision_at"], score_set["symbol"], percentiles,
                )
                primary = metrics["top"]["2"]
                name = f"{spec['name']}_{side}_{horizon}M"
                candidates.append(FrozenCandidate(
                    name, "DIRECT_CLASSIFICATION", side, int(horizon), model,
                    tuple(feature_names), thresholds,
                    primary["net14_mean_bps"] if primary["net14_mean_bps"] is not None else -math.inf,
                    primary["precision"] if primary["precision"] is not None else 0.0,
                ))
                reports.append({"candidate": name, "formulation": "DIRECT_CLASSIFICATION", "side": side, "horizon_minutes": horizon, **metrics})
            regression = regression_pipeline(config["models"]["regression"], seed).fit(train[feature_names], train["entry_quality_score"])
            scores = np.clip(regression.predict(score_set[feature_names]) / 100.0, 0, 1)
            quality_ranks = pd.Series(score_set["entry_quality_score"]).rank(method="average").to_numpy(float)
            score_ranks = pd.Series(scores).rank(method="average").to_numpy(float)
            quality_spearman = float(np.corrcoef(quality_ranks, score_ranks)[0, 1])
            metrics, thresholds, _ = ranked_metrics(
                score_set["zone_best"], scores, score_set["policy_gross_bps"],
                score_set["decision_at"], score_set["symbol"], percentiles,
            )
            primary = metrics["top"]["2"]
            name = f"HIST_GB_REGRESSION_{side}_{horizon}M"
            candidates.append(FrozenCandidate(
                name, "QUALITY_REGRESSION", side, int(horizon), regression,
                tuple(feature_names), thresholds,
                primary["net14_mean_bps"] if primary["net14_mean_bps"] is not None else -math.inf,
                primary["precision"] if primary["precision"] is not None else 0.0,
            ))
            reports.append({"candidate": name, "formulation": "QUALITY_REGRESSION", "side": side, "horizon_minutes": horizon, "quality_mae": float(mean_absolute_error(score_set["entry_quality_score"], regression.predict(score_set[feature_names]))), "quality_spearman": quality_spearman, **metrics})
        if not include_two_stage:
            continue
        train_both = discovery[discovery["horizon_minutes"].eq(horizon)].copy()
        validation_both = validation[validation["horizon_minutes"].eq(horizon)].copy()
        keys = ["decision_at", "symbol"]
        train_base = train_both.drop_duplicates(keys).sort_values(keys, kind="mergesort")
        validation_base = validation_both.drop_duplicates(keys).sort_values(keys, kind="mergesort")
        train_positive = train_both[train_both["zone_best"]].sort_values(
            [*keys, "entry_quality_score", "side"], ascending=[True, True, False, True], kind="mergesort"
        ).drop_duplicates(keys)
        positive_index = pd.MultiIndex.from_frame(train_positive[keys])
        train_index = pd.MultiIndex.from_frame(train_base[keys])
        opportunity_target = train_index.isin(positive_index).astype(int)
        any_ideal = train_both.groupby(keys, sort=True)["majority_ideal"].any().reindex(train_index).to_numpy(bool)
        zone_proxy = train_base[keys].copy()
        zone_proxy["majority_ideal"] = any_ideal
        keep_base = opportunity_target.astype(bool) | (~any_ideal & temporal_negative_mask(zone_proxy, int(config["zones"]["negative_exclusion_minutes"])))
        train_base = train_base.loc[keep_base].reset_index(drop=True)
        opportunity_target = opportunity_target[keep_base]
        if opportunity_target.sum() >= 20 and len(np.unique(opportunity_target)) == 2 and train_positive["side"].nunique() == 2:
            logistic_spec = next(spec for spec in config["models"]["classification"] if spec["name"] == "LOGISTIC")
            opportunity_model = _classification_pipeline(logistic_spec, seed, n_jobs=1).fit(train_base[feature_names], opportunity_target)
            direction_model = _classification_pipeline(logistic_spec, seed + int(horizon), n_jobs=1).fit(
                train_positive[feature_names], train_positive["side"].eq("LONG").astype(int)
            )
            p_opportunity = opportunity_model.predict_proba(validation_base[feature_names])[:, 1]
            p_long = direction_model.predict_proba(validation_base[feature_names])[:, 1]
            predicted_side = np.where(p_long >= 0.5, "LONG", "SHORT")
            scores = p_opportunity * np.maximum(p_long, 1.0 - p_long)
            lookup = validation_both.set_index(["decision_at", "symbol", "side"])
            selected_label = []
            selected_gross = []
            for row, side_value in zip(validation_base.itertuples(), predicted_side, strict=True):
                outcome = lookup.loc[(row.decision_at, row.symbol, side_value)]
                selected_label.append(bool(outcome["zone_best"]))
                selected_gross.append(float(outcome["policy_gross_bps"]))
            metrics, thresholds, _ = ranked_metrics(
                np.asarray(selected_label), scores, np.asarray(selected_gross),
                validation_base["decision_at"], validation_base["symbol"], percentiles,
            )
            primary = metrics["top"]["2"]
            name = f"OPPORTUNITY_THEN_SIDE_{horizon}M"
            candidates.append(FrozenCandidate(
                name, "OPPORTUNITY_THEN_SIDE", "DYNAMIC", int(horizon),
                (opportunity_model, direction_model), tuple(feature_names), thresholds,
                primary["net14_mean_bps"] if primary["net14_mean_bps"] is not None else -math.inf,
                primary["precision"] if primary["precision"] is not None else 0.0,
            ))
            reports.append({"candidate": name, "formulation": "OPPORTUNITY_THEN_SIDE", "side": "DYNAMIC", "horizon_minutes": horizon, **metrics})
    return candidates, pd.DataFrame(reports)


def select_candidate(candidates: list[FrozenCandidate]) -> FrozenCandidate:
    if not candidates:
        raise ValueError("no trainable W12 candidates")
    def complexity(item: FrozenCandidate) -> int:
        if item.name.startswith("LOGISTIC"):
            return 0
        if item.name.startswith("HIST_GB") or item.formulation == "OPPORTUNITY_THEN_SIDE":
            return 1
        if item.name.startswith("RANDOM_FOREST"):
            return 2
        return 9

    return sorted(candidates, key=lambda item: (
        -item.validation_primary_net14_bps,
        -item.validation_primary_precision,
        complexity(item),
        item.horizon_minutes,
        item.name,
    ))[0]
