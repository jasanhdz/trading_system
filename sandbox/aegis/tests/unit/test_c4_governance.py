import json
from pathlib import Path

import yaml

from aegis.utils import sha256_file


ROOT = Path(__file__).resolve().parents[2]
ABLATIONS = ROOT / "reports" / "governance" / "additional_ablations"
C4 = ROOT / "reports" / "governance" / "c4"


def test_frozen_inputs_and_prior_ablation_evidence_are_unchanged() -> None:
    expected = {
        "config/experiments/aegis_short_candidate_e1.yaml": "2604df3461ca891db05b6d877ab2e6373eac8fc7b7412d08571b2644f351ae39",
        "config/experiments/aegis_short_candidate_e2.yaml": "759a7c87d2afaec2f6ede7b6451154a287ea5527e00036e040beb26c32732b8e",
        "config/experiments/aegis_short_candidate_e3.yaml": "281e27f93f8be0f9c4fbe78673d55f1a4f3391aa421c3fcaf90823113dc81e62",
        "config/scientific_competition_v1.yaml": "6eb6e89b42ec518ddc7b277cec1ec7df22dd5f5b2fdbb78341d9885aaf27988c",
        "config/scientific_competition_v2.yaml": "70c889223b1466ed3e0817a63e7cfafb5b0966bf7c4d3eb3d06544c70c097f79",
        "reports/compatibility_replay/aegis-gen2-compatibility-replay-v1/stage_0.json": "6a10dccbea5ede8a1e863bd14d93b1a7e71b5d68f4403b509e4484b3e157ba17",
        "reports/compatibility_replay/aegis-gen2-compatibility-replay-v1/stage_1.json": "f9934bc318f6ab6f1aef8074e30b4e923bf61dc9b169804a535efc2fade9264f",
        "reports/compatibility_replay/aegis-gen2-compatibility-replay-v1/stage_2.json": "f213c521df29e4b4c7d9fecfa72e56f3d54d3f95ada8ae85409ef00fec8efb38",
        "reports/compatibility_replay/aegis-gen2-compatibility-replay-v1/stage_3.json": "d6c349bf6da4e3f307c6048a578f28de1e7160c476231c042a6994712313303a",
        "reports/compatibility_replay/aegis-gen2-compatibility-replay-v1/stage_4.json": "f434f0ff45a331552ac0877c6c2faead2125abea478be3f6d21141ad1c2307a3",
        "reports/compatibility_replay/aegis-gen2-compatibility-replay-v1/stage_5.json": "952cb81b68c9413c7df89af6fdeb0124707cf49f7ac1eec37facdbb0dee7e5af",
    }
    assert {path: sha256_file(ROOT / path) for path in expected} == expected


def test_stage_1b_is_closed_hourly_coherent_and_not_executed() -> None:
    payload = yaml.safe_load((ABLATIONS / "stage_1b_preregistration.yaml").read_text())
    assert payload["status"] == "PRE_REGISTERED_NOT_EXECUTED"
    assert payload["owner_decision"] == "OWNER_DECISION_O1"
    assert payload["data"]["training_sampling"] == payload["data"]["scoring_sampling"] == "E2_HOURLY_ANCHORS"
    assert payload["features"]["schema"] == "HISTORICAL_114_RECOMPUTED_ON_HOURLY_POPULATION"
    assert payload["folds"]["calibration_fraction"] == 0.50
    assert payload["folds"]["embargo_minutes"] == 120
    assert payload["folds"]["minimum_trades_each_fold"] == 100
    assert payload["training"]["seeds"] == "BASE_42_PLUS_FOLD_ID"
    assert payload["evaluation"]["deterministic_runs"] == 2
    assert payload["evaluation"]["numeric_tolerance"] == 1e-12
    assert payload["selection"]["absolute_threshold"] == "FORBIDDEN"


def test_stage_4b_has_only_three_single_lever_variants_in_frozen_order() -> None:
    payload = yaml.safe_load((ABLATIONS / "stage_4b_preregistration.yaml").read_text())
    assert payload["status"] == "PRE_REGISTERED_NOT_EXECUTED"
    assert payload["frozen_chain"] == ["scores", "veto", "threshold", "ranking", "dedup", "budget"]
    variants = payload["variants"]
    assert [item["id"] for item in variants] == payload["closed_variant_ids"] == ["STAGE_4B_A", "STAGE_4B_B", "STAGE_4B_C"]
    assert [item["changed_axis"] for item in variants] == ["veto", "selection", "threshold"]
    assert variants[0]["threshold"] == "NONE" and variants[0]["selection"] == "HISTORICAL_POOLED_TOP_DECILE"
    assert variants[1]["veto"]["mechanics"] == "RANK_BASED_BUDGET" and variants[1]["selection"] == "TOP_1_PER_CYCLE"
    assert variants[2]["threshold"]["application_order"] == "AFTER_VETO_BEFORE_RANKING"
    assert payload["additional_variants"] == "FORBIDDEN" and payload["execution_authorized"] is False


def test_owner_decision_and_optional_stage_5b_are_recorded_without_authorization() -> None:
    owner = json.loads((ABLATIONS / "owner_decision_o1.json").read_text())
    stage_5b = json.loads((ABLATIONS / "stage_5b_decision.json").read_text())
    assert owner["decision_id"] == "OWNER_DECISION_O1"
    assert owner["stage_1b_execution_authorized"] is False and owner["e3_validation_authorized"] is False
    assert stage_5b["status"] == "OPTIONAL_GOVERNANCE_EVIDENCE"
    assert stage_5b["owner_decision_required"] is True and stage_5b["executed"] is False


def test_baseline_contract_freezes_formulas_budget_and_directional_maximum() -> None:
    payload = yaml.safe_load((C4 / "baseline_contract.yaml").read_text())
    assert payload["common"]["indicators_source"] == "RAW_CANONICAL_BARS_ONLY"
    assert payload["budget"]["selected"] == "min(N_fold, eligible_cycle_selections)"
    assert payload["budget"]["fill_deficit"] == payload["budget"]["redistribute"] == "FORBIDDEN"
    assert payload["strategies"]["momentum_rule"]["score"] == "-ret_12"
    assert payload["strategies"]["mean_reversion_rule"]["score"] == "ret_12"
    assert payload["strategies"]["volatility_rule"]["minimum_window"] == "24/24"
    assert payload["strategies"]["no_trade"]["profit_factor"] is None
    assert payload["directional_maximum"]["exclude"] == ["eqm_only", "trrm_only"]


def test_lockbox_and_typescript_execution_remain_closed() -> None:
    authority = json.loads((ROOT / "reports/experiments/lockbox_semi_blind_20260427_20260711.json").read_text())
    assert authority["status"] == "NOT_CONSUMED" and authority["consumed_queries"] == []
    ts_config = yaml.safe_load((ROOT / "binance-futures-bot-ts/config/regimen.config.yaml").read_text())
    assert ts_config["execution"]["enabledByConfig"] is False
    assert ts_config["brain"]["allowedSides"] == ["SHORT"]


def test_governance_manifest_binds_every_materialized_contract() -> None:
    manifest = json.loads((C4 / "manifest.json").read_text())
    paths = {
        "baseline_contract.yaml": C4 / "baseline_contract.yaml",
        "parameter_matrix.json": C4 / "parameter_matrix.json",
        "implementation_report.md": C4 / "implementation_report.md",
        "test_evidence.json": C4 / "test_evidence.json",
        "stage_1b_preregistration.yaml": ABLATIONS / "stage_1b_preregistration.yaml",
        "stage_4b_preregistration.yaml": ABLATIONS / "stage_4b_preregistration.yaml",
        "owner_decision_o1.json": ABLATIONS / "owner_decision_o1.json",
        "stage_5b_decision.json": ABLATIONS / "stage_5b_decision.json",
    }
    assert {name: sha256_file(path) for name, path in paths.items()} == manifest["artifacts"]
    assert set(manifest["execution"].values()) <= {"NOT_EXECUTED", "NOT_ACCESSED"}
