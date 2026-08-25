from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from aegis_ephemeral_regime_w11.data import load_frozen_config
from aegis_ephemeral_regime_w11.experiment import FRAME_NAMES, run_experiment
from aegis_ephemeral_regime_w11.reporting import ARTIFACT_FILES, write_results


PROJECT = Path(__file__).resolve().parents[1]


def synthetic_config() -> dict:
    config = load_frozen_config(PROJECT / "config" / "w11_frozen.json")
    config["source"]["symbols"] = ["ADAUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"]
    config["partitions"]["validation"] = ["2023-01-04T00:00:00Z", "2023-01-04T06:00:00Z"]
    config["partitions"]["prospective"] = ["2023-01-05T00:00:00Z", "2023-01-05T06:00:00Z"]
    return config


def synthetic_panel(config: dict) -> pd.DataFrame:
    rng = np.random.default_rng(20260825)
    times = pd.date_range("2023-01-01", "2023-01-05 06:00", freq="15min", tz="UTC")
    records = []
    features = config["features"]["names"]
    for time_index, decision_at in enumerate(times):
        for symbol_index, symbol in enumerate(config["source"]["symbols"]):
            values = rng.normal(size=len(features))
            opportunity_state = 1.0 if (time_index + symbol_index) % 3 else -1.0
            direction_state = 1.0 if (time_index + 2 * symbol_index) % 4 < 2 else -1.0
            values[0] = opportunity_state * 3.0 + rng.normal(scale=0.1)
            values[1] = direction_state * 3.0 + rng.normal(scale=0.1)
            gross = direction_state * (48.0 if opportunity_state > 0 else 4.0)
            row = {
                "decision_at": decision_at, "symbol": symbol,
                "feature_available_at": decision_at,
                "outcome_available_at": decision_at + pd.Timedelta(minutes=60),
            }
            row.update(dict(zip(features, values, strict=True)))
            for horizon in config["experts"]["horizons_minutes"]:
                row[f"gross_target_{horizon}m_bps"] = gross + horizon / 100.0
                row[f"opportunity_{horizon}m"] = abs(row[f"gross_target_{horizon}m_bps"]) >= 19
            records.append(row)
    frame = pd.DataFrame(records).sort_values(["decision_at", "symbol"], kind="mergesort")
    return frame.set_index(["decision_at", "symbol"])


@pytest.fixture(scope="module")
def completed():
    config = synthetic_config()
    return config, synthetic_panel(config), run_experiment(synthetic_panel(config), config)


def test_temporal_bounds_and_no_future_labels_at_creation(completed):
    _, _, result = completed
    rows = result.candidate_evaluations
    assert not rows.empty
    created = pd.to_datetime(rows["created_at"], utc=True)
    assert (pd.to_datetime(rows["validation_start"], utc=True) == created - pd.Timedelta(hours=7)).all()
    assert (pd.to_datetime(rows["validation_end"], utc=True) == created - pd.Timedelta(hours=1)).all()
    assert (pd.to_datetime(rows["training_end"], utc=True) == created - pd.Timedelta(hours=8)).all()
    assert (pd.to_datetime(rows["maximum_training_outcome_available_at"], utc=True) <= created).dropna().all()


def test_ids_are_unique_and_expiration_never_reactivates(completed):
    _, _, result = completed
    ids = result.instances["instance_id"]
    assert ids.is_unique
    if not result.expiration_registry.empty:
        expired = result.expiration_registry.sort_values("expired_at").groupby(["mode", "instance_id"])
        assert expired.size().max() == 1
        for (_, instance_id), group in expired:
            later = result.decisions[
                result.decisions["model_instance_id"].eq(instance_id)
                & result.decisions["mode"].eq(group.iloc[0]["mode"])
                & pd.to_datetime(result.decisions["decision_at"], utc=True).ge(pd.Timestamp(group.iloc[0]["expired_at"]))
            ]
            assert later.empty


def test_one_open_position_per_symbol(completed):
    _, _, result = completed
    for (_, _), trades in result.trades.groupby(["partition", "mode"]):
        for _, symbol_trades in trades.groupby("symbol"):
            ordered = symbol_trades.sort_values("opened_at")
            assert (pd.to_datetime(ordered["opened_at"], utc=True).iloc[1:].to_numpy()
                    >= pd.to_datetime(ordered["resolved_at"], utc=True).iloc[:-1].to_numpy()).all()


def test_result_is_deterministic(completed):
    config, panel, first = completed
    second = run_experiment(panel, config)
    assert first.summary == second.summary
    for name in FRAME_NAMES:
        pdt.assert_frame_equal(getattr(first, name), getattr(second, name), check_like=False)


def test_output_contract_and_verdict(tmp_path, completed):
    config, _, result = completed
    sandbox = tmp_path / "sandbox"
    (sandbox / "config").mkdir(parents=True)
    (sandbox / "src").mkdir()
    (sandbox / "config" / "w11_frozen.json").write_text(json.dumps(config), encoding="utf-8")
    repository = tmp_path / "repository"
    manifest_path = repository / config["source"]["manifest"]
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"symbols": {
        symbol: {"parquet_sha256": f"{index:064x}"}
        for index, symbol in enumerate(config["source"]["symbols"], 1)
    }}), encoding="utf-8")

    write_results(result, config, sandbox, repository_dir=repository)

    expected = set(ARTIFACT_FILES.values()) | {"summary.json", "manifest.json"}
    assert expected.issubset(path.name for path in (sandbox / "artifacts").iterdir())
    verdict = json.loads((sandbox / "w11_ephemeral_regime_verdict.json").read_text())
    assert verdict["grade"] in {"A", "B", "C", "D"}
    assert verdict["verdict"] in {
        "EPHEMERAL_ALPHA_CONFIRMED", "EPHEMERAL_SIGNAL_DETECTED_NOT_YET_ECONOMIC",
        "NO_EPHEMERAL_EDGE_FOUND", "INSUFFICIENT_DATA",
    }
    summary_text = (sandbox / "artifacts" / "summary.json").read_text()
    assert "NaN" not in summary_text
    report = (sandbox / "w11_ephemeral_regime_result.md").read_text()
    assert report.count("### ") == 20
    assert "no production" in report.lower()
