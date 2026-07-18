"""Pre-registered stability-first model competition infrastructure."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..models import CalibrationMethod, CalibratorSpec
from ..tree_models import DecisionTree, EnsembleAggregation, TreeEnsemble, TreeNode
from ..utils import Sha256HashProvider
from .train import calibration_metrics, fit_platt_calibrator


@dataclass(frozen=True)
class FoldScore:
    fold: int
    score: float


@dataclass(frozen=True)
class StabilitySummary:
    candidate_id: str
    fold_scores: tuple[FoldScore, ...]
    worst_fold: float
    mean: float
    standard_deviation: float
    compute_cost: float


@dataclass(frozen=True)
class QmaeQuantileResult:
    q50: TreeEnsemble
    q90: TreeEnsemble
    conformal_adjustment: float
    empirical_coverage: float
    q50_pinball: float
    q90_pinball: float
    baseline_q90_pinball: float

    @property
    def valid(self) -> bool:
        return (
            0.87 <= self.empirical_coverage <= 0.93
            and self.q90_pinball <= 0.90 * self.baseline_q90_pinball
        )


@dataclass(frozen=True)
class ProbabilisticCandidateResult:
    candidate_id: str
    average_precision: float
    prevalence: float
    calibrator: CalibratorSpec
    calibration_report: Mapping[str, Mapping[str, float]]
    artifact: Mapping[str, Any]


@dataclass(frozen=True)
class RegressionCandidateResult:
    candidate_id: str
    mean_absolute_error: float
    artifact: Mapping[str, Any]


def summarize_stability(candidate_id: str, scores: Sequence[float], compute_cost: float) -> StabilitySummary:
    values = tuple(float(value) for value in scores)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("fold scores must be finite and non-empty")
    return StabilitySummary(
        candidate_id, tuple(FoldScore(index + 1, value) for index, value in enumerate(values)),
        min(values), float(np.mean(values)), float(np.std(values)), float(compute_cost),
    )


def rank_stability_first(candidates: Sequence[StabilitySummary]) -> tuple[StabilitySummary, ...]:
    return tuple(sorted(candidates, key=lambda item: (
        -item.worst_fold, -item.mean, item.standard_deviation, item.compute_cost, item.candidate_id,
    )))


def select_calibrator(
    fit_probabilities: Sequence[float], fit_labels: Sequence[int],
    score_probabilities: Sequence[float], score_labels: Sequence[int],
) -> tuple[CalibratorSpec, Mapping[str, Mapping[str, float]]]:
    """Select raw/Platt/isotonic by held-out ECE then Brier."""
    fit_p = np.asarray(fit_probabilities, dtype=np.float64); fit_y = np.asarray(fit_labels, dtype=np.float64)
    score_p = np.asarray(score_probabilities, dtype=np.float64); score_y = np.asarray(score_labels, dtype=np.float64)
    raw_ece, raw_brier = calibration_metrics(score_p, score_y)
    raw = CalibratorSpec(CalibrationMethod.IDENTITY, raw_ece, raw_brier, len(score_p))
    platt_fit = fit_platt_calibrator(fit_p, fit_y)
    platt_values = np.asarray([platt_fit.apply(float(value)) for value in score_p])
    platt_ece, platt_brier = calibration_metrics(platt_values, score_y)
    platt = CalibratorSpec(
        CalibrationMethod.PLATT, platt_ece, platt_brier, len(score_p), parameters=platt_fit.parameters,
    )
    isotonic_model = IsotonicRegression(out_of_bounds="clip").fit(fit_p, fit_y)
    iso_x = tuple(float(value) for value in isotonic_model.X_thresholds_)
    iso_y = tuple(float(value) for value in isotonic_model.y_thresholds_)
    isotonic_values = np.asarray(isotonic_model.predict(score_p), dtype=np.float64)
    iso_ece, iso_brier = calibration_metrics(isotonic_values, score_y)
    isotonic = CalibratorSpec(CalibrationMethod.ISOTONIC, iso_ece, iso_brier, len(score_p), x=iso_x, y=iso_y)
    candidates = (raw, platt, isotonic)
    selected = min(candidates, key=lambda item: (item.ece, item.brier, item.method.value))
    report = {item.method.value: {"ece": item.ece, "brier": item.brier} for item in candidates}
    return selected, report


def _tree_nodes(tree: Any, *, classifier: bool) -> tuple[TreeNode, ...]:
    state = tree.tree_
    missing = getattr(state, "missing_go_to_left", np.zeros(state.node_count, dtype=np.uint8))
    nodes = []
    for index in range(state.node_count):
        is_leaf = int(state.children_left[index]) == int(state.children_right[index])
        if classifier:
            counts = np.asarray(state.value[index][0], dtype=np.float64)
            value = float(counts[1] / counts.sum()) if len(counts) == 2 and counts.sum() > 0 else 0.0
        else:
            value = float(np.asarray(state.value[index]).reshape(-1)[0])
        nodes.append(TreeNode(
            int(state.feature[index]), float(state.threshold[index]), int(state.children_left[index]),
            int(state.children_right[index]), value, is_leaf, bool(missing[index]),
        ))
    return tuple(nodes)


def _finalize(
    ensemble_id: str, feature_names: Sequence[str], aggregation: EnsembleAggregation,
    base_value: float, trees: Sequence[DecisionTree],
) -> TreeEnsemble:
    provisional = TreeEnsemble(
        ensemble_id, "aegis-tree-ensemble-v1", tuple(feature_names), aggregation,
        float(base_value), tuple(trees), "",
    )
    return TreeEnsemble.from_payload(provisional.to_payload())


def export_random_forest(model: Any, ensemble_id: str, feature_names: Sequence[str], *, classifier: bool) -> TreeEnsemble:
    if int(model.n_features_in_) != len(feature_names):
        raise ValueError("forest feature dimension mismatch")
    if classifier and tuple(model.classes_) != (0, 1):
        raise ValueError("only binary forests are publishable")
    trees = tuple(DecisionTree(_tree_nodes(tree, classifier=classifier)) for tree in model.estimators_)
    return _finalize(ensemble_id, feature_names, EnsembleAggregation.AVERAGE, 0.0, trees)


def export_hist_gradient_boosting(
    model: Any, ensemble_id: str, feature_names: Sequence[str], *, classifier: bool,
) -> TreeEnsemble:
    if int(model.n_features_in_) != len(feature_names) or int(model.n_trees_per_iteration_) != 1:
        raise ValueError("only scalar HGB outputs are publishable")
    if np.any(getattr(model, "is_categorical_", np.zeros(len(feature_names), dtype=bool))):
        raise ValueError("categorical HGB nodes are not supported")
    trees = []
    for iteration in model._predictors:
        raw_nodes = iteration[0].nodes
        nodes = tuple(TreeNode(
            int(node["feature_idx"]), float(node["num_threshold"]), int(node["left"]), int(node["right"]),
            float(node["value"]), bool(node["is_leaf"]), bool(node["missing_go_to_left"]),
        ) for node in raw_nodes)
        trees.append(DecisionTree(nodes))
    aggregation = EnsembleAggregation.ADDITIVE_LOGIT if classifier else EnsembleAggregation.ADDITIVE
    return _finalize(ensemble_id, feature_names, aggregation, float(np.asarray(model._baseline_prediction).reshape(-1)[0]), trees)


def pinball_loss(actual: Sequence[float], predicted: Sequence[float], quantile: float) -> float:
    y = np.asarray(actual, dtype=np.float64); values = np.asarray(predicted, dtype=np.float64)
    residual = y - values
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def fit_qmae_quantiles(
    x_train: np.ndarray, y_train: np.ndarray, x_calibration: np.ndarray, y_calibration: np.ndarray,
    x_score: np.ndarray, y_score: np.ndarray, feature_names: Sequence[str], *, seed: int,
) -> QmaeQuantileResult:
    from sklearn.ensemble import HistGradientBoostingRegressor

    common = dict(max_iter=80, max_depth=4, min_samples_leaf=10, learning_rate=0.05, random_state=seed)
    q50_model = HistGradientBoostingRegressor(loss="quantile", quantile=0.50, **common).fit(x_train, y_train)
    q90_model = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **common).fit(x_train, y_train)
    calibration_residuals = y_calibration - q90_model.predict(x_calibration)
    rank = min(len(calibration_residuals) - 1, math.ceil((len(calibration_residuals) + 1) * 0.90) - 1)
    adjustment = max(0.0, float(np.sort(calibration_residuals)[rank]))
    q50_values = q50_model.predict(x_score)
    q90_values = q90_model.predict(x_score) + adjustment
    baseline = float(np.quantile(y_train, 0.90, method="higher"))
    baseline_loss = pinball_loss(y_score, np.full(len(y_score), baseline), 0.90)
    return QmaeQuantileResult(
        export_hist_gradient_boosting(q50_model, "qmae-q50", feature_names, classifier=False),
        export_hist_gradient_boosting(q90_model, "qmae-q90", feature_names, classifier=False),
        adjustment, float(np.mean(y_score <= q90_values)), pinball_loss(y_score, q50_values, 0.50),
        pinball_loss(y_score, q90_values, 0.90), baseline_loss,
    )


def _linear_artifact(model: Any, artifact_id: str, feature_names: Sequence[str]) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "aegis-linear-model-v1", "artifact_id": artifact_id,
        "feature_names": list(feature_names),
        "coefficients": np.asarray(model.coef_, dtype=np.float64).reshape(-1).tolist(),
        "intercept": float(np.asarray(model.intercept_).reshape(-1)[0]),
    }
    payload["content_hash"] = Sha256HashProvider().digest_value(payload)
    return payload


def run_trrm_fold_competition(
    x_train: np.ndarray, y_train: np.ndarray, x_calibration: np.ndarray, y_calibration: np.ndarray,
    x_score: np.ndarray, y_score: np.ndarray, feature_names: Sequence[str], *, seed: int,
) -> tuple[ProbabilisticCandidateResult, ...]:
    """Run every pre-registered TRRM family for one temporal fold."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    models = (
        ("trrm_logistic_baseline", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
        ("trrm_random_forest", RandomForestClassifier(
            n_estimators=80, max_depth=8, min_samples_leaf=8, random_state=seed, n_jobs=1,
        )),
        ("trrm_hist_gradient_boosting", HistGradientBoostingClassifier(
            max_iter=80, max_depth=5, min_samples_leaf=10, learning_rate=0.05, random_state=seed,
        )),
    )
    reports = []
    for candidate_id, model in models:
        model.fit(x_train, y_train)
        calibration_raw = model.predict_proba(x_calibration)[:, 1]
        score_raw = model.predict_proba(x_score)[:, 1]
        calibrator, calibration_report = select_calibrator(calibration_raw, y_calibration, score_raw, y_score)
        calibrated = np.asarray([calibrator.apply(float(value)) for value in score_raw])
        if candidate_id.endswith("logistic_baseline"):
            artifact = _linear_artifact(model, candidate_id, feature_names)
        elif candidate_id.endswith("random_forest"):
            artifact = export_random_forest(model, candidate_id, feature_names, classifier=True).to_payload()
        else:
            artifact = export_hist_gradient_boosting(model, candidate_id, feature_names, classifier=True).to_payload()
        reports.append(ProbabilisticCandidateResult(
            candidate_id, float(average_precision_score(y_score, calibrated)), float(np.mean(y_score)),
            calibrator, calibration_report, artifact,
        ))
    return tuple(reports)


def run_eqm_fold_competition(
    x_train: np.ndarray, clean_train: np.ndarray, net_train: np.ndarray,
    x_calibration: np.ndarray, clean_calibration: np.ndarray,
    x_score: np.ndarray, clean_score: np.ndarray, net_score: np.ndarray,
    feature_names: Sequence[str], *, seed: int,
) -> tuple[tuple[ProbabilisticCandidateResult, ...], tuple[RegressionCandidateResult, ...]]:
    """Evaluate EQM clean classification and net-quality regression as separate tasks."""
    from sklearn.ensemble import (
        ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import average_precision_score, mean_absolute_error

    clean_models = (
        ("eqm_logistic_clean_baseline", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
        ("eqm_random_forest_clean", RandomForestClassifier(
            n_estimators=80, max_depth=8, min_samples_leaf=8, random_state=seed, n_jobs=1,
        )),
        ("eqm_hgb_clean", HistGradientBoostingClassifier(
            max_iter=80, max_depth=5, min_samples_leaf=10, learning_rate=0.05, random_state=seed,
        )),
    )
    clean_reports = []
    for candidate_id, model in clean_models:
        model.fit(x_train, clean_train)
        calibration_raw = model.predict_proba(x_calibration)[:, 1]
        score_raw = model.predict_proba(x_score)[:, 1]
        calibrator, calibration_report = select_calibrator(
            calibration_raw, clean_calibration, score_raw, clean_score,
        )
        calibrated = np.asarray([calibrator.apply(float(value)) for value in score_raw])
        if "logistic" in candidate_id:
            artifact = _linear_artifact(model, candidate_id, feature_names)
        elif "random_forest" in candidate_id:
            artifact = export_random_forest(model, candidate_id, feature_names, classifier=True).to_payload()
        else:
            artifact = export_hist_gradient_boosting(model, candidate_id, feature_names, classifier=True).to_payload()
        clean_reports.append(ProbabilisticCandidateResult(
            candidate_id, float(average_precision_score(clean_score, calibrated)), float(np.mean(clean_score)),
            calibrator, calibration_report, artifact,
        ))

    net_models = (
        ("eqm_linear_net_baseline", Ridge(alpha=1.0)),
        ("eqm_extra_trees_net", ExtraTreesRegressor(
            n_estimators=80, max_depth=8, min_samples_leaf=5, random_state=seed, n_jobs=1,
        )),
        ("eqm_hgb_net", HistGradientBoostingRegressor(
            max_iter=80, max_depth=5, min_samples_leaf=10, learning_rate=0.05, random_state=seed,
        )),
    )
    net_reports = []
    for candidate_id, model in net_models:
        model.fit(x_train, net_train)
        predicted = model.predict(x_score)
        if "linear" in candidate_id:
            artifact = _linear_artifact(model, candidate_id, feature_names)
        elif "extra_trees" in candidate_id:
            artifact = export_random_forest(model, candidate_id, feature_names, classifier=False).to_payload()
        else:
            artifact = export_hist_gradient_boosting(model, candidate_id, feature_names, classifier=False).to_payload()
        net_reports.append(RegressionCandidateResult(
            candidate_id, float(mean_absolute_error(net_score, predicted)), artifact,
        ))
    return tuple(clean_reports), tuple(net_reports)


def registered_candidate_families() -> Mapping[str, tuple[str, ...]]:
    return {
        "TRRM": ("LOGISTIC_LINEAR", "RANDOM_FOREST", "HIST_GRADIENT_BOOSTING"),
        "QMAE": ("UNCONDITIONAL_QUANTILE", "HIST_GRADIENT_BOOSTING_QUANTILE"),
        "EQM_CLEAN": ("LOGISTIC_LINEAR", "RANDOM_FOREST", "HIST_GRADIENT_BOOSTING"),
        "EQM_NET_QUALITY": ("RIDGE_LINEAR", "EXTRA_TREES", "HIST_GRADIENT_BOOSTING"),
    }
