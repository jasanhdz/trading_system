#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_telegram as tg

CID = "gen2-test"


def setup(tmp: Path) -> None:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "freeze.json"
    core.FREEZE_PATH.write_text(json.dumps({"candidate_id": CID, "trrm_v2_sha256": "a", "eqm1_sha256": "b", "d3_dataset_sha256": "c", "feature_hash": "d"}))
    core.init_canary(CID)


def test_severity_cooldown_and_dedup() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        sent = []
        now = 1_000_000.0
        # CRITICAL: first send goes out, identical fingerprint within 10 min is suppressed
        r1 = tg.notify(CID, "CRITICAL", "orphan position", "x", send_fn=lambda m: sent.append(m) or True, now_epoch=now)
        r2 = tg.notify(CID, "CRITICAL", "orphan position", "x", send_fn=lambda m: sent.append(m) or True, now_epoch=now + 60)
        assert r1["sent"] is True and r2["reason"] == "COOLDOWN_DEDUP"
        # after the cooldown it re-sends
        r3 = tg.notify(CID, "CRITICAL", "orphan position", "x", send_fn=lambda m: sent.append(m) or True, now_epoch=now + 700)
        assert r3["sent"] is True
        # a NEW critical fingerprint always goes out immediately
        r4 = tg.notify(CID, "CRITICAL", "bracket failure", "y", send_fn=lambda m: sent.append(m) or True, now_epoch=now + 61)
        assert r4["sent"] is True
        # INFO long cooldown
        tg.notify(CID, "INFO", "heartbeat", "", send_fn=lambda m: sent.append(m) or True, now_epoch=now)
        r5 = tg.notify(CID, "INFO", "heartbeat", "", send_fn=lambda m: sent.append(m) or True, now_epoch=now + 3600)
        assert r5["reason"] == "COOLDOWN_DEDUP"
        assert len(sent) == 4
        # audit log exists without raising
        log = core.canary_dir(CID) / "telegram_log.jsonl"
        assert log.exists() and len(log.read_text().splitlines()) == 6


def test_secrets_are_redacted_and_failures_never_raise() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        os.environ["GEN2_BRIDGE_SECRET"] = "supersecret-hmac-value-123"
        try:
            captured = []
            r = tg.notify(CID, "CRITICAL", "leak test supersecret-hmac-value-123",
                          "body has supersecret-hmac-value-123 too",
                          send_fn=lambda m: captured.append(m) or True, now_epoch=1_000_000.0)
            assert r["sent"] is True
            assert "supersecret-hmac-value-123" not in captured[0]
            assert "<REDACTED>" in captured[0]
            # sender explosion -> reason recorded, no exception
            def boom(m):
                raise RuntimeError("telegram down")

            r2 = tg.notify(CID, "CRITICAL", "other", "x", send_fn=boom, now_epoch=1_000_000.0)
            assert r2["sent"] is False and "NOTIFY_FAILED" in r2["reason"]
        finally:
            os.environ.pop("GEN2_BRIDGE_SECRET", None)


def test_startup_message_reflects_real_state_without_secrets() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        state = {
            "contract": {"mode": "EXPERIMENTAL", "sizing_kind": "BALANCE_FRACTION", "default_leverage": 10,
                         "max_leverage": 10, "daily_loss_pct": 0.10, "total_loss_pct": 0.25,
                         "equity_floor_fraction": 0.75, "max_concurrent_positions": 1},
            "armed": False, "arm_reason": "CANARY_UNARMED_NO_TOKEN", "risk_reason": "RISK_OK",
            "kill_switch": False, "phase_o_paused": True,
            "symbols_analyzed": ["BTCUSDT", "ADAUSDT"], "symbols_executable": [],
            "bridge": {"gen2_enabled": True, "execution_enabled": False, "available_balance": None, "open_positions": []},
            "models": {"freeze_valid": True, "environment": "valid", "trrm": "hash-verified", "qmae": "hash-verified", "eqm": "hash-verified"},
            "evidence": {"paper_decisions": 33, "outcomes": 22, "dryrun_requests": 5, "real_order_submissions": 0, "real_fills": 0, "incidents": 0},
        }
        msg = tg.build_startup_message(CID, state)
        for needle in ("GEN2 CANARY ONLINE", "EXPERIMENTAL", "Armed: NO", "PAUSED", "hash-verified",
                       "Paper decisions: 33", "Órdenes reales: 0", "Fills reales: 0", "dry-run", "Ninguna", "BTC ADA"):
            assert needle in msg, f"missing: {needle}"
        assert "secret" not in msg.lower()


def test_unconfigured_telegram_is_reported_not_fatal() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        old_repo = tg.REPO
        old_env = {k: os.environ.pop(k, None) for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")}
        try:
            tg.REPO = Path(t)  # no .env files here
            r = tg.notify(CID, "INFO", "no creds", now_epoch=1_000_000.0)
            assert r["sent"] is False and r["reason"] == "TELEGRAM_NOT_CONFIGURED"
        finally:
            tg.REPO = old_repo
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v


if __name__ == "__main__":
    test_severity_cooldown_and_dedup()
    test_secrets_are_redacted_and_failures_never_raise()
    test_startup_message_reflects_real_state_without_secrets()
    test_unconfigured_telegram_is_reported_not_fatal()
    print("test_gen2_telegram: OK")
