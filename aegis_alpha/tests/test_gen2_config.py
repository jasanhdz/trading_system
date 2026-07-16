#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_config as cfg

CID = "gen2-test"

FREEZE = {"candidate_id": CID, "trrm_v2_sha256": "a", "eqm1_sha256": "b",
          "d3_dataset_sha256": "c", "feature_hash": "d",
          "veto": {"threshold_full_dev_informational": 0.5}}


def setup(tmp: Path, yaml_text: str) -> Path:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "freeze.json"
    core.FREEZE_PATH.write_text(json.dumps(FREEZE))
    core.init_canary(CID)
    conf = tmp / "gen2_config.yaml"
    conf.write_text(yaml_text)
    cfg.CONFIG_PATH = conf
    return conf


GOOD = """
candidate_id: gen2-test
mode: EXPERIMENTAL_CONTINUOUS
capital:
  initial_equity: 200.0
symbols: [ADAUSDT, DOGEUSDT]
execution_enabled: false
telegram_enabled: true
risk_overrides: {}
"""


def test_load_config_valid_and_coherent() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t), GOOD)
        c = cfg.load_config(CID)
        assert c["mode"] == "EXPERIMENTAL_CONTINUOUS"
        assert c["symbols"] == ["ADAUSDT", "DOGEUSDT"]
        assert c["execution_enabled"] is False
        assert c["max_orders_per_day"] == 20
        assert c["equity_floor"] == 200.0 * 0.75
        assert "config_checksum" in c


def test_incoherent_override_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as t:
        # huge balance fraction -> per-stop loss exceeds the daily cap (incoherent)
        setup(Path(t), GOOD.replace("risk_overrides: {}", "risk_overrides:\n  balance_fraction: 0.9"))
        try:
            cfg.load_config(CID)
            raise AssertionError("incoherent config must fail closed")
        except ValueError as e:
            assert "INCOHERENT" in str(e)


def test_non_overridable_key_rejected() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t), GOOD.replace("risk_overrides: {}", "risk_overrides:\n  martingale: true"))
        try:
            cfg.load_config(CID)
            raise AssertionError("non-overridable key must be rejected")
        except ValueError as e:
            assert "NON_OVERRIDABLE" in str(e)


def test_missing_config_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t), GOOD)
        cfg.CONFIG_PATH = Path(t) / "nope.yaml"
        try:
            cfg.load_config(CID)
            raise AssertionError("missing config must raise")
        except FileNotFoundError:
            pass


def test_startup_audit_written_without_secrets() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t), GOOD)
        rec = cfg.write_startup_audit(CID)
        assert rec["event"] == "GEN2_STARTUP" and rec["mode"] == "EXPERIMENTAL_CONTINUOUS"
        assert rec["execution_enabled"] is False
        audit = (core.canary_dir(CID) / "startup_audit.jsonl").read_text()
        assert "GEN2_STARTUP" in audit


def test_config_change_detection_diff() -> None:
    with tempfile.TemporaryDirectory() as t:
        conf = setup(Path(t), GOOD)
        cfg.snapshot_for_change_tracking(CID)
        c0 = cfg.config_checksum()
        # no change -> None
        assert cfg.detect_config_change(c0, CID) is None
        # flip execution_enabled -> CONFIGURATION_CHANGED with a diff
        conf.write_text(GOOD.replace("execution_enabled: false", "execution_enabled: true"))
        change = cfg.detect_config_change(c0, CID)
        assert change is not None and change["event"] == "CONFIGURATION_CHANGED"
        assert change["new_config_valid"] is True
        assert change["diff"]["execution_enabled"] == {"old": False, "new": True}


def test_config_change_to_invalid_is_flagged_not_applied() -> None:
    with tempfile.TemporaryDirectory() as t:
        conf = setup(Path(t), GOOD)
        cfg.snapshot_for_change_tracking(CID)
        c0 = cfg.config_checksum()
        conf.write_text(GOOD.replace("mode: EXPERIMENTAL_CONTINUOUS", "mode: NONSENSE"))
        change = cfg.detect_config_change(c0, CID)
        assert change["new_config_valid"] is False
        assert "UNKNOWN_MODE" in change["invalid_reason"]


if __name__ == "__main__":
    test_load_config_valid_and_coherent()
    test_incoherent_override_is_rejected()
    test_non_overridable_key_rejected()
    test_missing_config_fails_closed()
    test_startup_audit_written_without_secrets()
    test_config_change_detection_diff()
    test_config_change_to_invalid_is_flagged_not_applied()
    print("test_gen2_config: OK")
