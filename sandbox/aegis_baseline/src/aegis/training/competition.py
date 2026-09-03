"""Pre-registered stability-first model competition infrastructure."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.isotonic import IsotonicRegression

from ..models import CalibrationMethod, CalibratorSpec
from ..tree_models import DecisionTree, EnsembleAggregation, TreeEnsemble, TreeNode
from ..utils import Sha256HashProvider, sha256_file
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


@dataclass(frozen=True)
class ScientificCompetitionContract:
    """Hash-bound model and governance parameters for productive competition."""

    source_path: Path
    physical_sha256: str
    production_hyperparameters: Mapping[str, Mapping[str, Any]]
    smoke_overrides: Mapping[str, Any]
    eqm_training_population: Mapping[str, str]
    trrm_veto: Mapping[str, Any]
    econ_baselines: Mapping[str, Any]
    base_seed: int

    def parameters(self, candidate_id: str, *, smoke: bool = False) -> dict[str, Any]:
        try:
            values = dict(self.production_hyperparameters[candidate_id])
        except KeyError as exc:
            raise ValueError(f"candidate lacks frozen hyperparameters: {candidate_id}") from exc
        if smoke:
            if self.smoke_overrides.get("production_use") != "FORBIDDEN":
                raise ValueError("smoke override governance is invalid")
            for key in ("n_estimators", "max_iter"):
                if key in values and key in self.smoke_overrides:
                    values[key] = int(self.smoke_overrides[key])
        return values


@dataclass(frozen=True)
class RankBasedPopulation:
    train_indices: np.ndarray
    scoring_indices: np.ndarray
    train_total: int
    scoring_total: int
    veto_budget: float


_REQUIRED_HYPERPARAMETERS = {
    "trrm_logistic_baseline", "trrm_random_forest", "trrm_hist_gradient_boosting",
    "eqm_linear_net_baseline", "eqm_extra_trees_net", "eqm_hgb_net",
    "eqm_logistic_clean_baseline", "eqm_random_forest_clean", "eqm_hgb_clean",
    "qmae_hist_gradient_boosting",
}


def load_scientific_competition_contract(
    path: Path, *, expected_sha256: str | None = None,
) -> ScientificCompetitionContract:
    resolved = path.resolve()
    physical = sha256_file(resolved)
    if expected_sha256 is not None and physical != expected_sha256:
        raise ValueError("scientific competition physical hash mismatch")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "aegis-scientific-competition-v2":
        raise ValueError("productive competition requires schema v2")
    parameters = payload.get("production_hyperparameters")
    if not isinstance(parameters, Mapping) or set(parameters) != _REQUIRED_HYPERPARAMETERS:
        raise ValueError("productive competition hyperparameter matrix is incomplete")
    smoke = payload.get("smoke_overrides")
    if not isinstance(smoke, Mapping) or smoke.get("production_use") != "FORBIDDEN":
        raise ValueError("productive competition must forbid smoke overrides")
    population = payload.get("eqm_training_population")
    if population != {
        "fold_train": "TRRM_VETO_SURVIVORS_OF_FOLD_TRAIN",
        "fold_scoring": "TRRM_VETO_SURVIVORS_OF_FOLD_SCORING",
        "refit": "TRRM_VETO_SURVIVORS_OF_FINAL_TRAIN",
    }:
        raise ValueError("EQM population contract mismatch")
    veto = payload.get("trrm_veto")
    if not isinstance(veto, Mapping) or veto.get("mechanics") != "RANK_BASED_BUDGET" or float(veto.get("veto_budget", -1)) != 0.30:
        raise ValueError("TRRM veto contract mismatch")
    if veto.get("raw_probability_0_70") != "FORBIDDEN":
        raise ValueError("raw TRRM probability 0.70 is forbidden")
    baselines = payload.get("econ_baselines")
    if not isinstance(baselines, Mapping) or tuple(baselines.get("directional", ())) != (
        "no_trade", "random_directional_with_veto", "momentum_rule",
        "mean_reversion_rule", "volatility_rule",
    ) or tuple(baselines.get("diagnostic", ())) != ("eqm_only", "trrm_only"):
        raise ValueError("ECON baseline contract is missing")
    if not all(bool(baselines.get(key)) for key in ("same_costs", "same_holding_period", "same_selection_budget")):
        raise ValueError("ECON baseline equality contract is incomplete")
    return ScientificCompetitionContract(
        resolved, physical, parameters, smoke, population, veto, baselines,
        int(payload["base_seed"]),
    )


def rank_based_survivor_indices(probabilities: Sequence[float], *, veto_budget: float = 0.30) -> np.ndarray:
    """Keep the lowest-risk rank budget with stable input-order tie breaking."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("TRRM probabilities must be a finite non-empty vector")
    if not 0.0 < veto_budget < 1.0:
        raise ValueError("TRRM veto budget must be between zero and one")
    keep = max(1, int(round((1.0 - veto_budget) * len(values))))
    return np.argsort(values, kind="stable")[:keep]


def build_rank_based_eqm_population(
    train_probabilities: Sequence[float], scoring_probabilities: Sequence[float], *, veto_budget: float,
) -> RankBasedPopulation:
    return RankBasedPopulation(
        rank_based_survivor_indices(train_probabilities, veto_budget=veto_budget),
        rank_based_survivor_indices(scoring_probabilities, veto_budget=veto_budget),
        len(train_probabilities), len(scoring_probabilities), veto_budget,
    )


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


def fit_calibrator_family(
    method: CalibrationMethod, probabilities: Sequence[float], labels: Sequence[int],
) -> CalibratorSpec:
    """Refit a previously selected calibrator family without model selection."""
    values = np.asarray(probabilities, dtype=np.float64)
    actual = np.asarray(labels, dtype=np.float64)
    if method is CalibrationMethod.IDENTITY:
        ece, brier = calibration_metrics(values, actual)
        return CalibratorSpec(method, ece, brier, len(values))
    if method is CalibrationMethod.PLATT:
        return fit_platt_calibrator(values, actual)
    model = IsotonicRegression(out_of_bounds="clip").fit(values, actual)
    predicted = np.asarray(model.predict(values), dtype=np.float64)
    ece, brier = calibration_metrics(predicted, actual)
    return CalibratorSpec(
        CalibrationMethod.ISOTONIC, ece, brier, len(values),
        x=tuple(float(value) for value in model.X_thresholds_),
        y=tuple(float(value) for value in model.y_thresholds_),
    )


def _probabilistic_model(
    candidate_id: str, seed: int, contract: ScientificCompetitionContract, *, smoke: bool = False,
) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    parameters = contract.parameters(candidate_id, smoke=smoke)
    if candidate_id in {"trrm_logistic_baseline", "eqm_logistic_clean_baseline"}:
        return LogisticRegression(**parameters, random_state=seed)
    if candidate_id in {"trrm_random_forest", "eqm_random_forest_clean"}:
        return RandomForestClassifier(**parameters, random_state=seed, n_jobs=1)
    if candidate_id in {"trrm_hist_gradient_boosting", "eqm_hgb_clean"}:
        return HistGradientBoostingClassifier(**parameters, random_state=seed)
    raise ValueError(f"unknown probabilistic candidate: {candidate_id}")


def _regression_model(
    candidate_id: str, seed: int, contract: ScientificCompetitionContract, *, smoke: bool = False,
) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    parameters = contract.parameters(candidate_id, smoke=smoke)
    if candidate_id == "eqm_linear_net_baseline":
        return Ridge(**parameters)
    if candidate_id == "eqm_extra_trees_net":
        return ExtraTreesRegressor(**parameters, random_state=seed, n_jobs=1)
    if candidate_id == "eqm_hgb_net":
        return HistGradientBoostingRegressor(**parameters, random_state=seed)
    raise ValueError(f"unknown regression candidate: {candidate_id}")


def _export_fitted(model: Any, candidate_id: str, feature_names: Sequence[str], *, classifier: bool) -> Mapping[str, Any]:
    if "logistic" in candidate_id or "linear" in candidate_id:
        return _linear_artifact(model, candidate_id, feature_names)
    if "random_forest" in candidate_id or "extra_trees" in candidate_id:
        return export_random_forest(model, candidate_id, feature_names, classifier=classifier).to_payload()
    return export_hist_gradient_boosting(model, candidate_id, feature_names, classifier=classifier).to_payload()


def fit_selected_probabilistic_candidate(
    candidate_id: str, x_train: np.ndarray, y_train: np.ndarray,
    feature_names: Sequence[str], *, seed: int, contract: ScientificCompetitionContract,
) -> tuple[Any, Mapping[str, Any]]:
    model = _probabilistic_model(candidate_id, seed, contract).fit(x_train, y_train)
    return model, _export_fitted(model, candidate_id, feature_names, classifier=True)


def fit_selected_regression_candidate(
    candidate_id: str, x_train: np.ndarray, y_train: np.ndarray,
    feature_names: Sequence[str], *, seed: int, contract: ScientificCompetitionContract,
) -> tuple[Any, Mapping[str, Any]]:
    model = _regression_model(candidate_id, seed, contract).fit(x_train, y_train)
    return model, _export_fitted(model, candidate_id, feature_names, classifier=False)


def fit_qmae_refit(
    x_train: np.ndarray, y_train: np.ndarray, x_reserve: np.ndarray,
    y_reserve: np.ndarray, feature_names: Sequence[str], *, seed: int,
    contract: ScientificCompetitionContract,
) -> QmaeQuantileResult:
    """Fit final quantiles on train and conformal adjustment only on reserve."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    common = contract.parameters("qmae_hist_gradient_boosting")
    common["random_state"] = seed
    q50_model = HistGradientBoostingRegressor(loss="quantile", quantile=0.50, **common).fit(x_train, y_train)
    q90_model = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **common).fit(x_train, y_train)
    residuals = y_reserve - q90_model.predict(x_reserve)
    rank = min(len(residuals) - 1, math.ceil((len(residuals) + 1) * 0.90) - 1)
    adjustment = max(0.0, float(np.sort(residuals)[rank]))
    q50_values = q50_model.predict(x_reserve)
    q90_values = q90_model.predict(x_reserve) + adjustment
    baseline = float(np.quantile(y_train, 0.90, method="higher"))
    return QmaeQuantileResult(
        export_hist_gradient_boosting(q50_model, "qmae-final-q50", feature_names, classifier=False),
        export_hist_gradient_boosting(q90_model, "qmae-final-q90", feature_names, classifier=False),
        adjustment, float(np.mean(y_reserve <= q90_values)),
        pinball_loss(y_reserve, q50_values, 0.50),
        pinball_loss(y_reserve, q90_values, 0.90),
        pinball_loss(y_reserve, np.full(len(y_reserve), baseline), 0.90),
    )


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
    contract: ScientificCompetitionContract, smoke: bool = False,
) -> QmaeQuantileResult:
    from sklearn.ensemble import HistGradientBoostingRegressor

    common = contract.parameters("qmae_hist_gradient_boosting", smoke=smoke)
    common["random_state"] = seed
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
    contract: ScientificCompetitionContract, smoke: bool = False,
) -> tuple[ProbabilisticCandidateResult, ...]:
    """Run every pre-registered TRRM family for one temporal fold."""
    from sklearn.metrics import average_precision_score

    models = tuple((candidate_id, _probabilistic_model(candidate_id, seed, contract, smoke=smoke)) for candidate_id in (
        "trrm_logistic_baseline", "trrm_random_forest", "trrm_hist_gradient_boosting",
    ))
    reports = []
    for candidate_id, model in models:
        model.fit(x_train, y_train)
        calibration_raw = model.predict_proba(x_calibration)[:, 1]
        score_raw = model.predict_proba(x_score)[:, 1]
        calibrator, calibration_report = select_calibrator(calibration_raw, y_calibration, score_raw, y_score)
        calibrated = np.asarray([calibrator.apply(float(value)) for value in score_raw])
        artifact = _export_fitted(model, candidate_id, feature_names, classifier=True)
        reports.append(ProbabilisticCandidateResult(
            candidate_id, float(average_precision_score(y_score, calibrated)), float(np.mean(y_score)),
            calibrator, calibration_report, artifact,
        ))
    return tuple(reports)


def run_eqm_fold_competition(
    x_train: np.ndarray, clean_train: np.ndarray, net_train: np.ndarray,
    x_calibration: np.ndarray, clean_calibration: np.ndarray,
    x_score: np.ndarray, clean_score: np.ndarray, net_score: np.ndarray,
    feature_names: Sequence[str], *, seed: int, contract: ScientificCompetitionContract,
    smoke: bool = False,
) -> tuple[tuple[ProbabilisticCandidateResult, ...], tuple[RegressionCandidateResult, ...]]:
    """Evaluate EQM clean classification and net-quality regression as separate tasks."""
    from sklearn.metrics import average_precision_score, mean_absolute_error

    clean_models = tuple((candidate_id, _probabilistic_model(candidate_id, seed, contract, smoke=smoke)) for candidate_id in (
        "eqm_logistic_clean_baseline", "eqm_random_forest_clean", "eqm_hgb_clean",
    ))
    clean_reports = []
    for candidate_id, model in clean_models:
        model.fit(x_train, clean_train)
        calibration_raw = model.predict_proba(x_calibration)[:, 1]
        score_raw = model.predict_proba(x_score)[:, 1]
        calibrator, calibration_report = select_calibrator(
            calibration_raw, clean_calibration, score_raw, clean_score,
        )
        calibrated = np.asarray([calibrator.apply(float(value)) for value in score_raw])
        artifact = _export_fitted(model, candidate_id, feature_names, classifier=True)
        clean_reports.append(ProbabilisticCandidateResult(
            candidate_id, float(average_precision_score(clean_score, calibrated)), float(np.mean(clean_score)),
            calibrator, calibration_report, artifact,
        ))

    net_models = tuple((candidate_id, _regression_model(candidate_id, seed, contract, smoke=smoke)) for candidate_id in (
        "eqm_linear_net_baseline", "eqm_extra_trees_net", "eqm_hgb_net",
    ))
    net_reports = []
    for candidate_id, model in net_models:
        model.fit(x_train, net_train)
        predicted = model.predict(x_score)
        artifact = _export_fitted(model, candidate_id, feature_names, classifier=False)
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
