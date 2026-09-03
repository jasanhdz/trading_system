from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import inspect
from pathlib import Path

import pytest
import yaml

from aegis.live_decision import CurrentBrainEngine
from aegis.research.committee_v2_replay import (
    CommitteeV2ReplayError,
    assess_committee_v2_replay,
    load_committee_v2_replay_config,
)
from aegis.research.committee_v2_shadow import (
    REVERSAL_FLAG_FEATURES,
    committee_v2_counterfactual,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "config/experiments/aegis_committee_v2_replay_v1.yaml"


def _features(flagged: bool) -> dict[str, float]:
    return {
        name: float(flagged and index == 0)
        for index, name in enumerate(REVERSAL_FLAG_FEATURES)
    }


def _episode(index: int) -> dict:
    flagged = index % 2 == 0
    timestamp = datetime(2026, 7, 12, tzinfo=timezone.utc) + timedelta(hours=index)
    counterfactual = committee_v2_counterfactual(
        _features(flagged),
        control_selected=True,
        control_side="SHORT",
    )
    return {
        "source": "FIXTURE",
        "symbol": "BTCUSDT",
        "signal_timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "control_selected": True,
        "control_side": "SHORT",
        "committee_action": counterfactual["paper_action"],
        "committee_reason": counterfactual["reason"],
        "risk_flag_count": counterfactual["observed_risk_count"],
        "risk_flags": counterfactual["risk_flags"],
        "net": -0.02 if flagged else 0.01,
        "mae": 0.03 if flagged else 0.005,
        "mfe": 0.002 if flagged else 0.02,
        "exchange_mutations": 0,
    }


def test_repository_replay_contract_is_frozen_and_non_mutating() -> None:
    config = load_committee_v2_replay_config(CONFIG, repo_root=ROOT)

    assert config.replay_start > config.information_cutoff
    assert config.horizon_bars == 12
    assert config.bootstrap_resamples == 2000
    assert config.output_root.is_relative_to(ROOT / "data")


def test_network_or_exchange_authority_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["authority"]["network_access"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(
        CommitteeV2ReplayError,
        match="REPLAY_AUTHORITY_INVALID",
    ):
        load_committee_v2_replay_config(path, repo_root=ROOT)


def test_counterfactual_is_exact_and_does_not_fabricate_votes() -> None:
    clear = committee_v2_counterfactual(
        _features(False),
        control_selected=True,
        control_side="TradeSide.SHORT",
    )
    flagged = committee_v2_counterfactual(
        _features(True),
        control_selected=True,
        control_side="SHORT",
    )

    assert clear["paper_action"] == "ENTER_NOW"
    assert flagged["paper_action"] == "WAIT_CONFIRMATION"
    assert flagged["would_change_control"] is True
    assert flagged["observed_risk_count"] == 1


def test_paired_replay_detects_incremental_filter_value() -> None:
    config = replace(
        load_committee_v2_replay_config(CONFIG, repo_root=ROOT),
        minimum_preliminary_global_episodes=60,
        minimum_robust_global_episodes=70,
        bootstrap_resamples=200,
    )
    report = assess_committee_v2_replay(
        tuple(_episode(index) for index in range(80)),
        config,
        source="FIXTURE",
        provenance={"fixture": True},
    )

    assert report["population"]["global_purged_episodes"] == 80
    assert report["population"]["retained_coverage"] == 0.5
    assert report["performance"]["mean_paired_delta_fraction"] == pytest.approx(0.01)
    assert report["performance"]["retained_enter_now"][
        "mean_mae_fraction"
    ] == pytest.approx(0.005)
    assert report["walk_forward"]["positive_folds"] == 4
    assert report["verdict"] == "ROBUST_INCREMENTAL_VALUE_SUPPORTED"
    assert report["training_performed"] is False
    assert report["automatic_promotion"] is False
    assert report["exchange_mutations"] == 0


def test_assessment_is_deterministic_and_has_no_mutation_surface() -> None:
    from aegis.research import committee_v2_replay

    config = replace(
        load_committee_v2_replay_config(CONFIG, repo_root=ROOT),
        bootstrap_resamples=200,
    )
    episodes = tuple(_episode(index) for index in range(20))
    first = assess_committee_v2_replay(
        episodes,
        config,
        source="FIXTURE",
        provenance={"fixture": True},
    )
    second = assess_committee_v2_replay(
        episodes,
        config,
        source="FIXTURE",
        provenance={"fixture": True},
    )
    source = inspect.getsource(committee_v2_replay).lower()

    assert first == second
    assert all(
        value not in source
        for value in (
            "create_order",
            "cancel_order",
            "change_leverage",
            "change_margin",
            "api_secret",
            "api_key",
            "requests.",
        )
    )


def test_live_engine_has_explicit_historical_replay_clock(
    snapshot_factory,
) -> None:
    engine = CurrentBrainEngine()
    engine.initialize()
    snapshot = snapshot_factory(closed_at=datetime(2025, 1, 1, 1, tzinfo=timezone.utc))

    batch = engine.evaluate_replay(snapshot)

    assert batch["feature_count"] == 83
    assert set(batch["results"]) == {
        "ETHUSDT",
        "BTCUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "AVAXUSDT",
        "LINKUSDT",
        "SUIUSDT",
        "LTCUSDT",
    }
