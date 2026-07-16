#!/usr/bin/env python3
"""Truth table for the unified execution enablement:
config = the only enabler, GEN2_EXECUTION_ENABLED env = deny-only."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_config as cfg


def _set_env(v):
    if v is None:
        os.environ.pop(cfg.DENY_ONLY_ENV, None)
    else:
        os.environ[cfg.DENY_ONLY_ENV] = v


def test_truth_table() -> None:
    old = os.environ.get(cfg.DENY_ONLY_ENV)
    try:
        cfg_true = {"execution_enabled": True}
        cfg_false = {"execution_enabled": False}
        # (config, env)              -> effective
        cases = [
            (cfg_false, None, False),    # config false, env unset -> disabled
            (cfg_false, "true", False),  # config false + env true  -> DISABLED (never enable via env)
            (cfg_true, None, True),      # config true, env unset   -> enabled (config enables)
            (cfg_true, "true", True),    # config true + env true   -> enabled (env doesn't deny)
            (cfg_true, "false", False),  # config true + env false  -> DISABLED (emergency deny)
            (cfg_true, "0", False),      # deny-like value
            (cfg_true, "off", False),    # deny-like value
            (cfg_true, "garbage", True), # junk env is NOT a deny (and never enables)
            (None, "true", False),       # config missing (None) -> disabled
            (None, None, False),
        ]
        for contract, env, expected in cases:
            _set_env(env)
            got = cfg.effective_execution_enabled(contract)
            assert got == expected, f"config={contract} env={env}: expected {expected} got {got}"
        # default is always false
        _set_env(None)
        assert cfg.effective_execution_enabled({}) is False
    finally:
        _set_env(old)


def test_emergency_deny_detection() -> None:
    old = os.environ.get(cfg.DENY_ONLY_ENV)
    try:
        for v, deny in [(None, False), ("true", False), ("1", False), ("false", True),
                        ("FALSE", True), ("off", True), ("0", True), ("no", True), ("", False)]:
            _set_env(v)
            assert cfg.emergency_deny_active() is deny, f"{v!r}"
    finally:
        _set_env(old)


if __name__ == "__main__":
    test_truth_table()
    test_emergency_deny_detection()
    print("test_gen2_execution_enablement: OK")
