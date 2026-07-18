from pathlib import Path

import numpy as np
import pytest
import yaml

from aegis.training.competition import (
    build_rank_based_eqm_population,
    load_scientific_competition_contract,
    rank_based_survivor_indices,
)
from aegis.training.phase_e import ProductionScientificBackend
from aegis.training.run_state import PhaseETechnicalError


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "config" / "scientific_competition_v2.yaml"
E2 = ROOT / "config" / "experiments" / "aegis_short_candidate_e2.yaml"
E3 = ROOT / "config" / "experiments" / "aegis_short_candidate_e3.yaml"


EXPECTED = {
    "trrm_logistic_baseline": {"C": 1.0, "max_iter": 1000},
    "trrm_random_forest": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20, "class_weight": "balanced_subsample"},
    "trrm_hist_gradient_boosting": {"max_iter": 300, "learning_rate": 0.06, "max_depth": None, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0},
    "eqm_linear_net_baseline": {"alpha": 1.0},
    "eqm_extra_trees_net": {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 25},
    "eqm_hgb_net": {"max_iter": 250, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0},
    "eqm_logistic_clean_baseline": {"max_iter": 1000, "class_weight": "balanced", "solver": "liblinear"},
    "eqm_random_forest_clean": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 25, "class_weight": "balanced_subsample"},
    "eqm_hgb_clean": {"max_iter": 250, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0},
    "qmae_hist_gradient_boosting": {"max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 40, "l2_regularization": 1.0},
}


def test_productive_capacities_are_exactly_the_frozen_v2_matrix() -> None:
    contract = load_scientific_competition_contract(V2)
    assert {key: contract.parameters(key) for key in EXPECTED} == EXPECTED
    assert all(contract.parameters(key) != contract.parameters(key, smoke=True) for key in EXPECTED if "n_estimators" in EXPECTED[key] or "max_iter" in EXPECTED[key])


def test_e3_backend_hash_binds_v2_and_rejects_e2_execution() -> None:
    e3 = yaml.safe_load(E3.read_text(encoding="utf-8"))
    backend = ProductionScientificBackend(competition_path=V2)
    contract = backend._require_e3(e3)
    assert contract.physical_sha256 == e3["models"]["competition_protocol"]["physical_sha256"]
    with pytest.raises(PhaseETechnicalError, match="PROTOCOL_VERSION_NOT_EXECUTABLE"):
        backend._require_e3(yaml.safe_load(E2.read_text(encoding="utf-8")))


def test_rank_veto_is_exactly_thirty_percent_and_deterministic_on_ties() -> None:
    probabilities = np.asarray([0.4, 0.1, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.6, 0.5])
    first = rank_based_survivor_indices(probabilities, veto_budget=0.30)
    second = rank_based_survivor_indices(probabilities, veto_budget=0.30)
    assert first.tolist() == [1, 2, 4, 6, 0, 9, 8]
    assert np.array_equal(first, second)


def test_eqm_fold_train_and_scoring_populations_are_rank_veto_survivors() -> None:
    population = build_rank_based_eqm_population(
        [0.9, 0.1, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0],
        [0.8, 0.2, 0.1, 0.7, 0.3], veto_budget=0.30,
    )
    assert population.train_total == 10 and len(population.train_indices) == 7
    assert population.scoring_total == 5 and len(population.scoring_indices) == 4
    assert population.veto_budget == 0.30


def test_v2_rejects_smoke_governance_or_full_population_mutation(tmp_path: Path) -> None:
    payload = yaml.safe_load(V2.read_text(encoding="utf-8"))
    payload["smoke_overrides"]["production_use"] = "ALLOWED"
    path = tmp_path / "bad.yaml"; path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbid smoke"):
        load_scientific_competition_contract(path)
    payload = yaml.safe_load(V2.read_text(encoding="utf-8"))
    payload["eqm_training_population"]["fold_train"] = "FULL_FOLD_POPULATION"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="population"):
        load_scientific_competition_contract(path)
