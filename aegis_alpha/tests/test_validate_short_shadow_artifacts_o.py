from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.tools import generate_short_shadow_artifacts_o as gen
from aegis_alpha.tools import validate_short_shadow_artifacts_o as val


def _args(tmp: Path):
    return argparse.Namespace(symbols="ALL", only_symbol=None, metadata_only=True, out_dir=str(tmp / "reports"), shadow_model_dir=str(tmp / "models" / "turbo"), fast=True, dry_run=False, skip_training=False, validate_only=False)


def _generate(tmp: Path):
    original_gen = gen.turbo_symbol_model_dir
    original_val = val.turbo_symbol_model_dir
    gen.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
    val.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
    try:
        result = gen.generate(_args(tmp))
        validation = val.validate(tmp / "models" / "turbo", tmp / "validation")
    finally:
        gen.turbo_symbol_model_dir = original_gen
        val.turbo_symbol_model_dir = original_val
    return result, validation


def test_validator_passes_valid_metadata_structure():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, validation = _generate(tmp)
        assert validation["status"] == "passed", validation["errors"]
        assert validation["entry_count"] == 10
        assert validation["avoid_only_count"] == 1
        assert Path(validation["report_paths"]["json"]).exists()
        assert Path(validation["report_paths"]["csv"]).exists()


def test_validator_fails_missing_symbol():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        original_gen = gen.turbo_symbol_model_dir
        original_val = val.turbo_symbol_model_dir
        gen.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
        val.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
        try:
            result = gen.generate(_args(tmp))
            global_path = Path(result.global_manifest_path)
            payload = json.loads(global_path.read_text())
            payload["artifact_paths"].pop("SOLUSDT")
            global_path.write_text(json.dumps(payload), encoding="utf-8")
            validation = val.validate(tmp / "models" / "turbo", tmp / "validation")
        finally:
            gen.turbo_symbol_model_dir = original_gen
            val.turbo_symbol_model_dir = original_val
        assert validation["status"] == "failed"
        assert any("expected 11" in err or "missing symbol" in err for err in validation["errors"])


def test_validator_fails_link_entry_enabled():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        original_gen = gen.turbo_symbol_model_dir
        original_val = val.turbo_symbol_model_dir
        gen.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
        val.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
        try:
            result = gen.generate(_args(tmp))
            link_path = Path(next(m for m in result.symbol_manifests if m["symbol"] == "LINKUSDT")["artifact_dir"]) / "symbol_shadow_manifest.json"
            payload = json.loads(link_path.read_text())
            payload["entry_enabled"] = True
            link_path.write_text(json.dumps(payload), encoding="utf-8")
            validation = val.validate(tmp / "models" / "turbo", tmp / "validation")
        finally:
            gen.turbo_symbol_model_dir = original_gen
            val.turbo_symbol_model_dir = original_val
        assert validation["status"] == "failed"
        assert any("LINKUSDT" in err for err in validation["errors"])


def test_validator_fails_affects_orders_true_and_missing_models_when_required():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        original_gen = gen.turbo_symbol_model_dir
        original_val = val.turbo_symbol_model_dir
        gen.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
        val.turbo_symbol_model_dir = lambda symbol: tmp / "models" / "turbo" / symbol
        try:
            result = gen.generate(_args(tmp))
            ltc_path = Path(next(m for m in result.symbol_manifests if m["symbol"] == "LTCUSDT")["artifact_dir"]) / "symbol_shadow_manifest.json"
            payload = json.loads(ltc_path.read_text())
            payload["affects_orders"] = True
            ltc_path.write_text(json.dumps(payload), encoding="utf-8")
            validation = val.validate(tmp / "models" / "turbo", tmp / "validation", require_model_files=True)
        finally:
            gen.turbo_symbol_model_dir = original_gen
            val.turbo_symbol_model_dir = original_val
        assert validation["status"] == "failed"
        assert any("affects_orders" in err for err in validation["errors"])
        assert any("no model_files" in err for err in validation["errors"])


if __name__ == "__main__":
    test_validator_passes_valid_metadata_structure()
    test_validator_fails_missing_symbol()
    test_validator_fails_link_entry_enabled()
    test_validator_fails_affects_orders_true_and_missing_models_when_required()
    print("manual_validate_short_shadow_artifacts_o_tests_passed")
