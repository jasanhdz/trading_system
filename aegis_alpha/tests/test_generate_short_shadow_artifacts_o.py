from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools import generate_short_shadow_artifacts_o as gen


def _args(tmp: Path, **overrides):
    data = dict(symbols="ALL", only_symbol=None, metadata_only=True, out_dir=str(tmp / "reports"), shadow_model_dir=str(tmp / "models" / "turbo"), fast=True, dry_run=False, skip_training=False, validate_only=False)
    data.update(overrides)
    return argparse.Namespace(**data)


def test_generate_metadata_only_11_manifests_and_pointer():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        original = gen.turbo_symbol_model_dir
        gen.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
        try:
            result = gen.generate(_args(tmp))
        finally:
            gen.turbo_symbol_model_dir = original
        assert len(result.symbol_manifests) == 11
        assert sum(1 for m in result.symbol_manifests if m["entry_enabled"]) == 10
        assert sum(1 for m in result.symbol_manifests if m["avoid_only"]) == 1
        link = next(m for m in result.symbol_manifests if m["symbol"] == "LINKUSDT")
        assert link["avoid_only"] is True
        assert link["entry_enabled"] is False
        assert link["affects_gating"] is False
        assert link["affects_sizing"] is False
        assert link["affects_orders"] is False
        assert "open_position" in link["forbidden_runtime_use"]
        assert link["feature_schema_hash"]
        pointer = json.loads(Path(result.pointer_manifest_path).read_text())
        assert pointer["latest_artifact_stamp"] == result.artifact_stamp
        assert pointer["active_manifest_touched"] is True


def test_dry_run_does_not_write_files():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        original = gen.turbo_symbol_model_dir
        gen.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
        try:
            result = gen.generate(_args(tmp, dry_run=True))
        finally:
            gen.turbo_symbol_model_dir = original
        assert result.dry_run is True
        assert result.pointer_manifest_path == ""
        assert not (tmp / "models").exists()


def test_unknown_symbol_rejected():
    try:
        gen.select_symbols("FAKEUSDT", None)
    except SystemExit as exc:
        assert "Unknown" in str(exc)
    else:
        raise AssertionError("unknown symbol accepted")


def test_live_compatible_path_guard():
    gen.assert_live_compatible_path("/x/aegis_alpha/models/turbo/LTCUSDT/active/phase_o_20260101/model.joblib")
    try:
        gen.assert_live_compatible_path("/x/aegis_alpha/models/turbo/LTCUSDT/active/model.joblib")
    except ValueError:
        pass
    else:
        raise AssertionError("non phase_o active path accepted")


def test_metadata_only_has_no_joblib_requirement():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        original = gen.turbo_symbol_model_dir
        gen.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
        try:
            result = gen.generate(_args(tmp, only_symbol="LTCUSDT"))
        finally:
            gen.turbo_symbol_model_dir = original
        manifest = result.symbol_manifests[0]
        assert manifest["metadata_only"] is True
        assert manifest["model_files"] == []
        active = json.loads((tmp / "models" / "turbo" / "LTCUSDT" / "active_manifest.json").read_text())
        assert active["phase_o_prod_ready"] is True


if __name__ == "__main__":
    test_generate_metadata_only_11_manifests_and_pointer()
    test_dry_run_does_not_write_files()
    test_unknown_symbol_rejected()
    test_live_compatible_path_guard()
    test_metadata_only_has_no_joblib_requirement()
    print("manual_generate_short_shadow_artifacts_o_tests_passed")
