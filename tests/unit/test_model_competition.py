import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier,
)

from aegis.training.competition import (
    export_hist_gradient_boosting, export_random_forest, fit_qmae_quantiles,
    load_scientific_competition_contract,
    rank_stability_first, registered_candidate_families, run_eqm_fold_competition,
    run_trrm_fold_competition, select_calibrator, summarize_stability,
)


CONTRACT = load_scientific_competition_contract(
    Path(__file__).resolve().parents[2] / "config" / "scientific_competition_v2.yaml",
)
from aegis.features import DeterministicFeaturePipeline, FEATURE_NAMES
from aegis.models import DeterministicModelRuntime, TreeHead
from aegis.tree_models import (
    DecisionTree, EnsembleAggregation, TreeEnsemble, TreeModelError, TreeNode,
)


def _data():
    rng = np.random.default_rng(20260718)
    x = rng.normal(size=(240, 4))
    y_class = (x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.3, size=len(x)) > 0).astype(int)
    y_value = 0.01 + 0.006 * np.abs(x[:, 0]) + 0.002 * x[:, 2] + rng.normal(scale=0.001, size=len(x))
    return x, y_class, y_value


def test_random_forest_and_extra_trees_json_export_match_sklearn() -> None:
    x, y_class, y_value = _data(); names = tuple(f"f{index}" for index in range(x.shape[1]))
    classifier = RandomForestClassifier(n_estimators=9, max_depth=5, random_state=7, n_jobs=1).fit(x[:180], y_class[:180])
    exported_classifier = export_random_forest(classifier, "rf", names, classifier=True)
    expected = classifier.predict_proba(x[180:])[:, 1]
    actual = np.asarray([exported_classifier.evaluate(row) for row in x[180:]])
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-12)

    regressor = ExtraTreesRegressor(n_estimators=9, max_depth=5, random_state=7, n_jobs=1).fit(x[:180], y_value[:180])
    exported_regressor = export_random_forest(regressor, "extra", names, classifier=False)
    assert np.allclose(
        [exported_regressor.evaluate(row) for row in x[180:]], regressor.predict(x[180:]),
        rtol=0.0, atol=1e-12,
    )


def test_hist_gradient_boosting_json_export_matches_sklearn() -> None:
    x, y_class, y_value = _data(); names = tuple(f"f{index}" for index in range(x.shape[1]))
    classifier = HistGradientBoostingClassifier(max_iter=25, max_depth=3, min_samples_leaf=8, random_state=9).fit(x[:180], y_class[:180])
    exported_classifier = export_hist_gradient_boosting(classifier, "hgb-c", names, classifier=True)
    assert np.allclose(
        [exported_classifier.evaluate(row) for row in x[180:]], classifier.predict_proba(x[180:])[:, 1],
        rtol=0.0, atol=1e-12,
    )
    regressor = HistGradientBoostingRegressor(max_iter=25, max_depth=3, min_samples_leaf=8, random_state=9).fit(x[:180], y_value[:180])
    exported_regressor = export_hist_gradient_boosting(regressor, "hgb-r", names, classifier=False)
    assert np.allclose(
        [exported_regressor.evaluate(row) for row in x[180:]], regressor.predict(x[180:]),
        rtol=0.0, atol=1e-12,
    )


def test_tree_payload_is_content_hashed_and_deterministic() -> None:
    x, y_class, _ = _data(); names = tuple(f"f{index}" for index in range(x.shape[1]))
    model = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=1, n_jobs=1).fit(x, y_class)
    first = export_random_forest(model, "rf", names, classifier=True)
    second = export_random_forest(model, "rf", names, classifier=True)
    assert first.to_payload() == second.to_payload()
    mutated = dict(first.to_payload()); mutated["base_value"] = 1.0
    try:
        TreeEnsemble.from_payload(mutated)
    except TreeModelError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("mutated tree artifact was accepted")


def test_runtime_can_consume_a_bundle_v2_tree_head(snapshot_factory, scenario_bundle_factory) -> None:
    provisional = TreeEnsemble(
        "tail-tree", "aegis-tree-ensemble-v1", FEATURE_NAMES, EnsembleAggregation.AVERAGE,
        0.0, (DecisionTree((TreeNode(-1, -2.0, -1, -1, 0.2, True),)),), "",
    )
    tree = TreeEnsemble.from_payload(provisional.to_payload())
    bundle = scenario_bundle_factory("SHORT")
    estimator = replace(bundle.estimators[0], tail_risk=TreeHead("tail-tree", "PROBABILITY"))
    bundle = replace(bundle, schema_version="aegis-model-bundle-v2", estimators=(estimator,), tree_ensembles=(tree,))
    features = DeterministicFeaturePipeline().transform(snapshot_factory())
    predictions = DeterministicModelRuntime(bundle).predict(features)
    assert all(item.tail_risk_probability == 0.2 for item in predictions.predictions)


def test_tree_training_and_export_are_deterministic_between_processes() -> None:
    code = """
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from aegis.training.competition import export_random_forest
rng=np.random.default_rng(99)
x=rng.normal(size=(120,3)); y=(x[:,0]-x[:,1]>0).astype(int)
model=RandomForestClassifier(n_estimators=7,max_depth=4,random_state=5,n_jobs=1).fit(x,y)
print(export_random_forest(model,'det',('a','b','c'),classifier=True).content_hash)
"""
    environment = {**os.environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).parents[2] / "src")}
    first = subprocess.check_output([sys.executable, "-c", code], env=environment, text=True)
    second = subprocess.check_output([sys.executable, "-c", code], env=environment, text=True)
    assert first == second


def test_calibration_competition_reports_all_pre_registered_candidates() -> None:
    raw = np.linspace(0.02, 0.98, 120)
    labels = (raw + 0.15 * np.sin(np.arange(len(raw))) > 0.5).astype(int)
    selected, report = select_calibrator(raw[:80], labels[:80], raw[80:], labels[80:])
    assert set(report) == {"IDENTITY", "PLATT", "ISOTONIC"}
    assert selected.method.value in report
    assert math.isfinite(selected.ece) and math.isfinite(selected.brier)


def test_qmae_quantile_smoke_exports_models_and_reports_conformal_coverage() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(600, 3))
    y = np.maximum(0.0, 0.008 + 0.004 * np.abs(x[:, 0]) + rng.normal(scale=0.0015, size=len(x)))
    result = fit_qmae_quantiles(
        x[:300], y[:300], x[300:450], y[300:450], x[450:], y[450:],
        ("a", "b", "c"), seed=11, contract=CONTRACT, smoke=True,
    )
    q90 = np.asarray([result.q90.evaluate(row) + result.conformal_adjustment for row in x[450:]])
    assert abs(float(np.mean(y[450:] <= q90)) - result.empirical_coverage) <= 1e-15
    assert 0.0 <= result.empirical_coverage <= 1.0
    assert result.q50_pinball >= 0.0 and result.q90_pinball >= 0.0 and result.baseline_q90_pinball >= 0.0


def test_stability_ranking_and_candidate_matrix_are_pre_registered() -> None:
    stable = summarize_stability("stable", (0.61, 0.62, 0.63, 0.64), 2.0)
    volatile = summarize_stability("volatile", (0.58, 0.90, 0.91, 0.92), 1.0)
    assert rank_stability_first((volatile, stable))[0].candidate_id == "stable"
    families = registered_candidate_families()
    assert families["TRRM"] == ("LOGISTIC_LINEAR", "RANDOM_FOREST", "HIST_GRADIENT_BOOSTING")
    assert "HIST_GRADIENT_BOOSTING_QUANTILE" in families["QMAE"]
    assert "EXTRA_TREES" in families["EQM_NET_QUALITY"]


def test_trrm_and_eqm_smoke_run_every_family_with_separate_targets() -> None:
    x, tail, net = _data(); clean = (net > np.median(net)).astype(int)
    names = tuple(f"f{index}" for index in range(x.shape[1]))
    trrm = run_trrm_fold_competition(
        x[:120], tail[:120], x[120:180], tail[120:180], x[180:], tail[180:], names,
        seed=17, contract=CONTRACT, smoke=True,
    )
    assert {item.candidate_id for item in trrm} == {
        "trrm_logistic_baseline", "trrm_random_forest", "trrm_hist_gradient_boosting",
    }
    assert all(set(item.calibration_report) == {"IDENTITY", "PLATT", "ISOTONIC"} for item in trrm)
    clean_results, net_results = run_eqm_fold_competition(
        x[:120], clean[:120], net[:120], x[120:180], clean[120:180],
        x[180:], clean[180:], net[180:], names, seed=17, contract=CONTRACT, smoke=True,
    )
    assert {item.candidate_id for item in clean_results} == {
        "eqm_logistic_clean_baseline", "eqm_random_forest_clean", "eqm_hgb_clean",
    }
    assert {item.candidate_id for item in net_results} == {
        "eqm_linear_net_baseline", "eqm_extra_trees_net", "eqm_hgb_net",
    }
    assert all(item.mean_absolute_error >= 0.0 for item in net_results)
