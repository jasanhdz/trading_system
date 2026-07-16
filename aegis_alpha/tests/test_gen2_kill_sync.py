#!/usr/bin/env python3
"""Kill switch synchronization: the startup gauntlet clears BOTH the Python and
the bridge kill switches atomically (both or neither)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_config as cfg
import aegis_alpha.tools.gen2_decision_loop as loop

CID = "gen2-test"


def setup(tmp: Path) -> tuple[Path, Path]:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "freeze.json"
    core.FREEZE_PATH.write_text(json.dumps({"candidate_id": CID, "trrm_v2_sha256": "a",
                                            "eqm1_sha256": "b", "d3_dataset_sha256": "c", "feature_hash": "d"}))
    core.init_canary(CID)
    loop.BRIDGE_STATE_DIR = tmp / "bridge_state"
    (tmp / "bridge_state").mkdir()
    cfg.CONFIG_PATH = tmp / "nope.yaml"  # startup audit fails-caught; irrelevant to sync
    py = core.canary_dir(CID) / "KILL_SWITCH"
    br = loop.bridge_kill_path()
    return py, br


def engage(py: Path, br: Path) -> None:
    py.write_text(json.dumps({"reason": "test"}))
    br.write_text(json.dumps({"reason": "test"}))


def test_both_kills_cleared_when_gauntlet_passes() -> None:
    with tempfile.TemporaryDirectory() as t:
        py, br = setup(Path(t))
        engage(py, br)
        loop.startup_gauntlet = lambda c, s=None, o=None: (True, [])  # monkeypatch: pass
        r = loop.startup_arm(CID)
        assert r["kills_cleared"] is True
        assert not py.exists() and not br.exists()  # BOTH gone


def test_gauntlet_fail_keeps_both_engaged() -> None:
    with tempfile.TemporaryDirectory() as t:
        py, br = setup(Path(t))
        # only the bridge kill is engaged; python is clear
        br.write_text(json.dumps({"reason": "bridge"}))
        assert not py.exists()
        loop.startup_gauntlet = lambda c, s=None, o=None: (False, ["OPEN_POSITION_PRESENT"])
        r = loop.startup_arm(CID)
        assert r["kills_cleared"] is False
        # both must be consistently ENGAGED after a failed gauntlet
        assert py.exists() and br.exists()


def test_repeated_restarts_idempotent() -> None:
    with tempfile.TemporaryDirectory() as t:
        py, br = setup(Path(t))
        engage(py, br)
        loop.startup_gauntlet = lambda c, s=None, o=None: (True, [])
        loop.startup_arm(CID)
        assert not py.exists() and not br.exists()
        # second restart: nothing engaged -> no-op, still clear, no error
        r2 = loop.startup_arm(CID)
        assert r2["kills_cleared"] is False and not py.exists() and not br.exists()
        # third restart with a fresh kill -> clears again
        engage(py, br)
        r3 = loop.startup_arm(CID)
        assert r3["kills_cleared"] is True and not py.exists() and not br.exists()


def test_atomic_rollback_keeps_both_when_bridge_unlink_fails() -> None:
    with tempfile.TemporaryDirectory() as t:
        py, br = setup(Path(t))
        py.write_text(json.dumps({"reason": "py"}))
        # make the bridge kill path un-unlinkable: a non-empty directory at that path
        br.rmdir() if br.is_dir() else None
        brdir = br
        brdir.mkdir()
        (brdir / "child").write_text("x")  # non-empty dir -> unlink raises
        cleared, why = loop._clear_both_kills(CID)
        assert cleared is False
        assert py.exists()  # rolled back: python kill restored (both-or-neither)


def test_gauntlet_flags_positions_orders_and_bridge() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        # open position present
        ok, f = loop.startup_gauntlet(CID, status_fn=lambda: {"gen2_enabled": True, "open_positions": ["X"]},
                                      orders_fn=lambda c: 0)
        assert not ok and "OPEN_POSITION_PRESENT" in f
        # open orders present
        _, f2 = loop.startup_gauntlet(CID, status_fn=lambda: {"gen2_enabled": True, "open_positions": []},
                                      orders_fn=lambda c: 3)
        assert "OPEN_ORDERS_PRESENT" in f2
        # orders unverifiable (no creds)
        _, f3 = loop.startup_gauntlet(CID, status_fn=lambda: {"gen2_enabled": True, "open_positions": []},
                                      orders_fn=lambda c: None)
        assert "OPEN_ORDERS_UNVERIFIABLE" in f3
        # bridge unreachable
        def boom():
            raise RuntimeError("refused")
        _, f4 = loop.startup_gauntlet(CID, status_fn=boom, orders_fn=lambda c: 0)
        assert any(x.startswith("BRIDGE_UNREACHABLE") for x in f4)
        # bridge unhealthy (gen2_enabled false)
        _, f5 = loop.startup_gauntlet(CID, status_fn=lambda: {"gen2_enabled": False, "open_positions": []},
                                      orders_fn=lambda c: 0)
        assert "BRIDGE_NOT_HEALTHY" in f5


if __name__ == "__main__":
    _orig = loop.startup_gauntlet
    try:
        test_both_kills_cleared_when_gauntlet_passes()
        loop.startup_gauntlet = _orig
        test_gauntlet_fail_keeps_both_engaged()
        loop.startup_gauntlet = _orig
        test_repeated_restarts_idempotent()
        loop.startup_gauntlet = _orig
        test_atomic_rollback_keeps_both_when_bridge_unlink_fails()
        test_gauntlet_flags_positions_orders_and_bridge()
    finally:
        loop.startup_gauntlet = _orig
    print("test_gen2_kill_sync: OK")
