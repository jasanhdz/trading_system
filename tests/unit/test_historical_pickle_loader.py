from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest
import sklearn._loss._loss as runtime_loss

from scripts.diagnostics.compat_replay import historical_loader as loader
from scripts.diagnostics.compat_replay.historical_adapter import historical_trade_pnl


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["CyPinballLoss", "CyHalfBinomialLoss"])
def test_legacy_loss_resolution_is_exact_and_replay_only(name: str) -> None:
    resolved = loader.resolve_legacy_serialization_global("_loss", name)
    assert resolved is getattr(runtime_loss, name)
    assert resolved.__name__ == name
    assert resolved.__module__ == "_loss"
    assert loader.resolve_legacy_serialization_global("_loss", "OtherClass") is None
    assert ("_loss", "*") not in loader.ALLOWED_GLOBALS
    assert not (ROOT / "_loss.py").exists()


@pytest.mark.parametrize("name", ["CyPinballLoss", "CyHalfBinomialLoss"])
def test_serialized_legacy_global_fails_normally_and_loads_only_with_exact_remap(name: str) -> None:
    payload = f"c_loss\n{name}\n.".encode()
    assert "_loss" not in sys.modules
    with pytest.raises(ModuleNotFoundError, match="_loss"):
        pickle.loads(payload)
    assert loader.AllowlistedUnpickler(io.BytesIO(payload)).load() is getattr(runtime_loss, name)
    assert "_loss" not in sys.modules


def test_legacy_loss_remapping_has_no_prefix_or_dynamic_fallback() -> None:
    with pytest.raises(loader.HistoricalCompatibilityError, match="UNINVENTORIED"):
        loader.AllowlistedUnpickler(io.BytesIO(b"c_loss\nOtherClass\n.")).load()
    assert loader.resolve_legacy_serialization_global("_loss.extra", "CyPinballLoss") is None
    assert loader.resolve_legacy_serialization_global("_loss", "*") is None


def test_public_pandas_globals_use_exact_identity_without_wildcard() -> None:
    assert loader.ALLOWED_GLOBALS >= {("pandas", "Index"), ("pandas", "StringDtype")}
    assert ("pandas", "*") not in loader.ALLOWED_GLOBALS
    payload = pickle.dumps(pd.Index(["x"]))
    loaded = loader.AllowlistedUnpickler(io.BytesIO(payload)).load()
    assert type(loaded) is pd.Index
    with pytest.raises(loader.HistoricalCompatibilityError, match="UNINVENTORIED"):
        loader.AllowlistedUnpickler(io.BytesIO(b"\x80\x04cpandas\nDataFrame\n.")).load()


def test_historical_namespace_is_removed_after_context() -> None:
    with loader.historical_namespace() as symbol:
        assert symbol.__module__ == "aegis_alpha.tools.gen2_rv2_train"
        assert "aegis_alpha.tools.gen2_rv2_train" in sys.modules
    assert "aegis_alpha" not in sys.modules
    assert "aegis_alpha.tools" not in sys.modules
    assert "aegis_alpha.tools.gen2_rv2_train" not in sys.modules


def test_frozen_pickle_and_protocol_hashes_are_unchanged() -> None:
    expected = {
        ROOT / "config/experiments/aegis_short_candidate_e1.yaml": "2604df3461ca891db05b6d877ab2e6373eac8fc7b7412d08571b2644f351ae39",
        ROOT / "config/experiments/aegis_short_candidate_e2.yaml": "759a7c87d2afaec2f6ede7b6451154a287ea5527e00036e040beb26c32732b8e",
        ROOT / "config/experiments/aegis_short_candidate_e3.yaml": "281e27f93f8be0f9c4fbe78673d55f1a4f3391aa421c3fcaf90823113dc81e62",
        ROOT / "config/scientific_competition_v1.yaml": "6eb6e89b42ec518ddc7b277cec1ec7df22dd5f5b2fdbb78341d9885aaf27988c",
        ROOT / "config/scientific_competition_v2.yaml": "70c889223b1466ed3e0817a63e7cfafb5b0966bf7c4d3eb3d06544c70c097f79",
    }
    expected.update({path: digest for path, digest in loader.FROZEN_PICKLES.values()})
    assert {path: loader.sha256_file(path) for path in expected} == expected


def test_compatibility_matrix_is_closed_and_replay_only() -> None:
    matrix = json.loads((ROOT / "reports/diagnostics/compat_replay/historical_compatibility_matrix.json").read_text())
    rows = matrix["rows"]
    assert {(row["serialized_module"], row["serialized_name"]) for row in rows} == {
        ("__main__", "MedianImputer"),
        ("aegis_alpha.tools.gen2_rv2_train", "MedianImputer"),
        ("pandas", "Index"),
        ("pandas", "StringDtype"),
        ("_loss", "CyPinballLoss"),
        ("_loss", "CyHalfBinomialLoss"),
    }
    assert all(row["replay_only"] and not row["production_allowed"] for row in rows)
    assert all(not row["wildcard_allowed"] for row in rows)


def test_productive_code_does_not_reference_compatibility_replay() -> None:
    references = []
    for path in (ROOT / "src/aegis").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "compat_replay" in text or "historical_loader" in text or "aegis_alpha" in text:
            references.append(path)
    assert references == []


def test_lockbox_remains_unconsumed() -> None:
    authority = json.loads(
        (ROOT / "reports/experiments/lockbox_semi_blind_20260427_20260711.json").read_text()
    )
    assert authority["status"] == "NOT_CONSUMED"
    assert authority["consumed_queries"] == []
    assert authority["maximum_queries_total"] == 1


def test_historical_trade_pnl_uses_next_bar_open_h12_and_frozen_costs() -> None:
    timestamp = pd.Timestamp("2025-01-01 00:00:00")
    index = pd.date_range(timestamp, periods=14, freq="5min")
    frame = pd.DataFrame({"open": [100.0] * 14, "close": [100.0] * 13 + [99.0], "_i": range(14)}, index=index)
    result = historical_trade_pnl(
        {"ADAUSDT": frame}, "ADAUSDT", timestamp, {"fee": 5.0, "slip": 2.0, "funding_h": 1.0}
    )
    assert result == {"entry": 100.0, "exit": 99.0, "gross": 1.0, "cost": 0.15, "net": 0.85}
