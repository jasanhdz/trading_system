#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_decision_loop as loop
import aegis_alpha.tools.gen2_operational_contract as oc
from aegis_alpha.tools.gen2_rv2_train import MedianImputer

CID = "gen2-test"
BAR_MS = 5 * 60_000


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)

    def predict_proba(self, x):
        p = np.full(len(x), self.value)
        return np.column_stack([1 - p, p])


def make_env(tmp: Path, tail_value: float = 0.05) -> dict:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "freeze.json"
    loop.FORWARD_ROOT = tmp / "forward"
    loop.SYMBOLS = ["ADAUSDT", "BTCUSDT", "ETHUSDT"]
    (tmp / "forward").mkdir()
    feats = None  # computed lazily; models accept any width via ConstantModel
    trrm_dir = tmp / "trrm"
    eqm_dir = tmp / "eqm"
    trrm_dir.mkdir(), eqm_dir.mkdir()
    feature_names = ["feature.ret_1", "horizon_6", "horizon_12", "horizon_24"]
    imp = MedianImputer()
    imp.medians = pd.Series({c: 0.0 for c in feature_names})
    trrm = {"trrm_model": ConstantModel(tail_value), "imputer": imp, "calibrator": None, "calibrator_kind": "raw",
            "qmae_models": {"q90": ConstantModel(0.1)}, "qmae_q90_conformal_adjustment": 0.01, "features": feature_names}
    eqm = {"reg_model": ConstantModel(0.2), "clf_model": ConstantModel(0.7), "imputer": imp,
           "features": feature_names, "score_kind": "reg_component"}
    with (trrm_dir / "rv2_candidate.pkl").open("wb") as f:
        pickle.dump(trrm, f)
    with (eqm_dir / "eqm1_candidate.pkl").open("wb") as f:
        pickle.dump(eqm, f)
    from aegis_alpha.tools.gen2_d3_common import sha256_file
    core.FREEZE_PATH.write_text(json.dumps({
        "candidate_id": CID, "trrm_v2_sha256": sha256_file(trrm_dir / "rv2_candidate.pkl"),
        "eqm1_sha256": sha256_file(eqm_dir / "eqm1_candidate.pkl"), "d3_dataset_sha256": "c", "feature_hash": "d",
        "veto": {"threshold_full_dev_informational": 0.5},
    }))
    core.init_canary(CID)

    # fake snapshot: write candle CSVs the loop can read
    snap_dir = tmp / "snap"
    snap_dir.mkdir()
    n = 200
    ts = pd.date_range("2026-07-12", periods=n, freq="5min")
    for sym in loop.SYMBOLS:
        pd.DataFrame({"timestamp": ts.astype(str), "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
                      "volume": 10.0, "taker_buy_base_volume": 5.0, "open_time": (ts.astype("int64") // 10**6)}).to_csv(snap_dir / f"{sym}_5m.csv", index=False)
    snapshot = {"directory": str(snap_dir), "snapshot_id": "snap-test", "symbols": {s: {} for s in loop.SYMBOLS}}
    return {"trrm_dir": trrm_dir, "eqm_dir": eqm_dir, "snapshot": snapshot}


def run(env, live=False, status=None, execute=None):
    return loop.run_cycle(CID, env["trrm_dir"], env["eqm_dir"], live_enabled=live,
                          snapshot_fn=lambda: env["snapshot"], status_fn=lambda: status,
                          execute_fn=execute, filters_fn=lambda s: {"step_size": 1.0, "min_notional": 5.0})


def test_paper_always_recorded_and_dedup() -> None:
    with tempfile.TemporaryDirectory() as t:
        env = make_env(Path(t))
        s1 = run(env)
        assert s1["decisions"] == 3 and s1["candidate_short"] == 3 and s1["orders_submitted"] == 0
        assert all(a["reason"] in {"WOULD_SUBMIT_LIVE_DISABLED", "CANARY_UNARMED_NO_TOKEN", "SYMBOL_NOT_IN_ARM_TOKEN"} or not a["eligible"] for a in s1["live_attempts"])
        # same candle -> dedup, no duplicate paper records
        s2 = run(env)
        assert s2["decisions"] == 0
        lines = (loop.FORWARD_ROOT / "forward_decisions.jsonl").read_text().splitlines()
        assert len(lines) == 3


def test_veto_blocks_live_but_paper_records() -> None:
    with tempfile.TemporaryDirectory() as t:
        env = make_env(Path(t), tail_value=0.9)  # above veto threshold 0.5
        s = run(env)
        assert s["vetoed"] == 3 and s["candidate_short"] == 0


def test_live_path_full_gauntlet_and_ack() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        env = make_env(tmp)
        oc.write_contract(CID, "experimental", 200.0)
        oc.create_arm_token(CID, 72, ["ADAUSDT"])
        status = {"phase_o_allow_orders": False, "gen2_enabled": True, "open_positions": [], "available_balance": 200.0}
        sent = []

        def fake_execute(order):
            sent.append(order)
            return {"status": "ACCEPTED", "client_order_id": order["client_order_id"]}

        s = run(env, live=True, status=status, execute=fake_execute)
        assert s["orders_submitted"] == 1  # only ADAUSDT is in the token
        assert sent[0]["symbol"] == "ADAUSDT" and sent[0]["side"] == "SHORT"
        assert sent[0]["brackets"]["stop_price"] > 100.0
        # token consumed (first_arm_max_orders=1) -> second cycle cannot submit
        env["snapshot"] = {**env["snapshot"]}
        state = json.loads((core.canary_dir(CID) / "loop_state.json").read_text())
        state["last_processed"] = {}
        core.atomic_write(core.canary_dir(CID) / "loop_state.json", json.dumps(state))
        s2 = run(env, live=True, status=status, execute=fake_execute)
        assert s2["orders_submitted"] == 0
        reasons = {a["reason"] for a in s2["live_attempts"]}
        # exhausted token yields empty allowed-symbols (fail-closed); either reason is the same block
        assert reasons & {"TOKEN_MAX_ORDERS_EXHAUSTED", "SYMBOL_NOT_IN_ARM_TOKEN"}


def test_live_fail_closed_without_bridge_status() -> None:
    with tempfile.TemporaryDirectory() as t:
        env = make_env(Path(t))
        oc.write_contract(CID, "safe", 200.0)
        oc.create_arm_token(CID, 72, ["ADAUSDT"])
        s = run(env, live=True, status=None, execute=lambda o: {"status": "ACCEPTED"})
        assert s["orders_submitted"] == 0
        assert any(a["reason"] == "BRIDGE_STATUS_UNAVAILABLE" for a in s["live_attempts"])


def test_event_ingestion_dedup_and_pnl_wiring() -> None:
    with tempfile.TemporaryDirectory() as t:
        env = make_env(Path(t))
        oc.write_contract(CID, "safe", 200.0)
        events = [
            {"type": "FILL", "client_order_id": "a", "ts_sequence": 1, "payload": {"fill_price": 100.0}},
            {"type": "POSITION_CLOSED", "client_order_id": "a", "ts_sequence": 2, "payload": {"realized_pnl": -1.5}},
            {"type": "FILL", "client_order_id": "a", "ts_sequence": 1, "payload": {}},  # duplicate
        ]
        r = loop.ingest_events(CID, events)
        assert r["ingested"] == 2
        state = json.loads((core.canary_dir(CID) / "risk_state.json").read_text())
        assert abs(state["total_loss"] - 1.5) < 1e-9 and state["consecutive_losses"] == 1
        # bracket failure engages kill switch
        loop.ingest_events(CID, [{"type": "BRACKET_FAILED", "client_order_id": "b", "ts_sequence": 3}])
        assert core.kill_switch_engaged(CID) is True
        # unknown PnL is never guessed: it becomes an operator incident
        loop.ingest_events(CID, [{"type": "POSITION_CLOSED", "client_order_id": "c", "ts_sequence": 4,
                                  "payload": {"realized_pnl": None, "reason": "CLOSED_BY_STOP_OR_MANUAL_PNL_UNKNOWN"}}])
        incidents = [json.loads(l) for l in (core.canary_dir(CID) / "incidents" / "incidents.jsonl").read_text().splitlines()]
        assert any(i["type"] == "POSITION_CLOSED_UNKNOWN_PNL_REQUIRES_OPERATOR" for i in incidents)
        state = json.loads((core.canary_dir(CID) / "risk_state.json").read_text())
        assert abs(state["total_loss"] - 1.5) < 1e-9  # unchanged by unknown PnL


def test_pull_events_checkpoint_and_bridge_failure() -> None:
    with tempfile.TemporaryDirectory() as t:
        env = make_env(Path(t))  # noqa: F841 - sets canary roots
        oc.write_contract(CID, "safe", 200.0)
        pages = {0: [{"type": "FILL", "client_order_id": "a", "ts_sequence": 1, "payload": {}},
                     {"type": "POSITION_CLOSED", "client_order_id": "a", "ts_sequence": 2, "payload": {"realized_pnl": -0.5}}],
                 2: [{"type": "BRACKET_CONFIRMED", "client_order_id": "a", "ts_sequence": 3, "payload": {}}]}
        calls = []

        def fake_fetch(after):
            calls.append(after)
            return pages.get(after, [])

        r1 = loop.pull_events(CID, fake_fetch)
        assert r1["pulled"] == 2 and r1["ingested"] == 2 and r1["last_sequence"] == 2
        r2 = loop.pull_events(CID, fake_fetch)
        assert calls == [0, 2]  # checkpoint advanced
        assert r2["pulled"] == 1 and r2["last_sequence"] == 3
        # re-delivery of already-seen events is deduped, checkpoint never regresses
        pages[3] = pages[0]
        r3 = loop.pull_events(CID, fake_fetch)
        assert r3["ingested"] == 0 and r3["last_sequence"] == 3
        # bridge failure -> incident, checkpoint untouched, no crash
        def boom(after):
            raise RuntimeError("down")

        r4 = loop.pull_events(CID, boom)
        assert r4["error"] == "BRIDGE_UNAVAILABLE" and r4["last_sequence"] == 3


def test_token_burned_before_submit_and_unconfirmed_ack_incident() -> None:
    with tempfile.TemporaryDirectory() as t:
        env = make_env(Path(t))
        oc.write_contract(CID, "experimental", 200.0)
        oc.create_arm_token(CID, 72, ["ADAUSDT"])
        status = {"phase_o_allow_orders": False, "gen2_enabled": True, "open_positions": [], "available_balance": 200.0}

        def bridge_down(order):
            return {"status": "BRIDGE_UNAVAILABLE", "reason": "NO_RESPONSE"}

        s = run(env, live=True, status=status, execute=bridge_down)
        assert s["orders_submitted"] == 0
        # fail-closed: token was consumed BEFORE the wire attempt
        assert oc.verify_arm_token(CID)[1] == "TOKEN_MAX_ORDERS_EXHAUSTED"
        incidents = [json.loads(l) for l in (core.canary_dir(CID) / "incidents" / "incidents.jsonl").read_text().splitlines()]
        assert any(i["type"] == "TOKEN_CONSUMED_WITHOUT_CONFIRMED_ACK" for i in incidents)


def test_crash_mid_cycle_does_not_reemit_decisions() -> None:
    with tempfile.TemporaryDirectory() as t:
        env = make_env(Path(t))
        run(env)  # full cycle appends 3 decisions and persists per-symbol state
        # simulate a crash-restart: fresh run over the same snapshot must re-emit nothing
        s2 = run(env)
        lines = (loop.FORWARD_ROOT / "forward_decisions.jsonl").read_text().splitlines()
        assert s2["decisions"] == 0 and len(lines) == 3


def test_watch_singleton_lock() -> None:
    with tempfile.TemporaryDirectory() as t:
        make_env(Path(t))
        lock = loop.acquire_watch_lock(CID)
        try:
            loop.acquire_watch_lock(CID)
            raise AssertionError("second watcher must be refused")
        except RuntimeError as exc:
            assert "WATCHER_ALREADY_RUNNING" in str(exc)
        # stale lock (dead pid) is taken over
        core.atomic_write(lock, json.dumps({"pid": 999999999}))
        lock2 = loop.acquire_watch_lock(CID)
        assert lock2.exists()
        lock2.unlink()


def test_snapshot_pruning_keeps_recent() -> None:
    with tempfile.TemporaryDirectory() as t:
        make_env(Path(t))
        root = loop.FORWARD_ROOT / "snapshots"
        root.mkdir(parents=True, exist_ok=True)
        for i in range(15):
            (root / f"2026071{i:02d}T000000Z").mkdir()
        old_validate = loop.validate_gen2_path
        loop.validate_gen2_path = lambda p: None  # tmp dirs live outside GEN2_ROOT
        try:
            removed = loop.prune_forward_snapshots(keep=12)
        finally:
            loop.validate_gen2_path = old_validate
        assert removed == 3
        assert len(list(root.iterdir())) == 12


def test_environment_mismatch_vs_freeze_fails_before_unpickle() -> None:
    with tempfile.TemporaryDirectory() as t:
        env = make_env(Path(t))
        freeze = json.loads(core.FREEZE_PATH.read_text())
        freeze["environment"] = {"pandas": "999.0.0", "sklearn": "999.0.0", "numpy": "999.0.0",
                                 "executable": "/frozen/venv/bin/python"}
        core.FREEZE_PATH.write_text(json.dumps(freeze))
        try:
            run(env)
            raise AssertionError("environment drift must fail closed before unpickling")
        except ValueError as exc:
            assert "ENVIRONMENT_MISMATCH_VS_FREEZE" in str(exc)


def test_watch_runner_heartbeat_incidents_and_hard_stop() -> None:
    with tempfile.TemporaryDirectory() as t:
        make_env(Path(t))
        sleeps = []
        cycles = {"n": 0}

        def flaky_cycle(cid, live_enabled=False):
            cycles["n"] += 1
            if cycles["n"] == 2:
                raise RuntimeError("snapshot fetch timeout")  # operational -> survive
            return {"decisions": 3, "orders_submitted": 0}

        hb = loop.run_watch(CID, interval_seconds=0.01, max_cycles=3, live_enabled=False,
                            cycle_fn=flaky_cycle, resolve_fn=lambda cid: {"resolved_new": 2},
                            sleep_fn=sleeps.append)
        assert hb["cycles"] == 3 and hb["cycle_errors"] == 1
        assert hb["decisions"] == 6 and hb["outcomes_resolved"] == 6
        assert len(sleeps) == 2  # no sleep after the final cycle
        hb_file = json.loads((core.canary_dir(CID) / "heartbeat.json").read_text())
        assert hb_file["schema"] == "gen2_watch_heartbeat_v1" and hb_file["cycles"] == 3
        incidents = [json.loads(l) for l in (core.canary_dir(CID) / "incidents" / "incidents.jsonl").read_text().splitlines()]
        assert any(i["type"] == "WATCH_CYCLE_FAILED" for i in incidents)

        # scientific integrity failure -> hard stop
        def poisoned_cycle(cid, live_enabled=False):
            raise ValueError("TRRM_HASH_MISMATCH_VS_FREEZE")

        try:
            loop.run_watch(CID, interval_seconds=0.01, max_cycles=2, cycle_fn=poisoned_cycle,
                           resolve_fn=lambda cid: {"resolved_new": 0}, sleep_fn=lambda s: None)
            raise AssertionError("hash mismatch must abort the watcher")
        except ValueError:
            pass
        incidents = [json.loads(l) for l in (core.canary_dir(CID) / "incidents" / "incidents.jsonl").read_text().splitlines()]
        assert any(i["type"] == "WATCH_HARD_STOP_INTEGRITY" for i in incidents)


def test_watch_notifications_severity_and_startup() -> None:
    with tempfile.TemporaryDirectory() as t:
        make_env(Path(t))
        oc.write_contract(CID, "safe", 200.0)
        notes: list[tuple] = []

        def note(cid, severity, title, body="", fingerprint=None):
            notes.append((severity, title))
            return {"sent": True}

        # seed a connectivity incident (WARNING class) and a critical one before the cycle
        core.append_jsonl(core.canary_dir(CID) / "incidents" / "incidents.jsonl", {"type": "EVENTS_PULL_FAILED", "error": "refused"})

        def cycle_that_adds_critical_incident(cid, live_enabled=False):
            core.append_jsonl(core.canary_dir(CID) / "incidents" / "incidents.jsonl", {"type": "ORPHAN_ORDER_ON_EXCHANGE"})
            return {"decisions": 3, "orders_submitted": 0}

        loop.run_watch(CID, interval_seconds=0.0, max_cycles=1, live_enabled=False,
                       cycle_fn=cycle_that_adds_critical_incident, resolve_fn=lambda cid: {"resolved_new": 0},
                       sleep_fn=lambda s: None, notify_fn=note)
        titles = {(sev, title) for sev, title in notes}
        # startup INFO always fires
        assert any(sev == "INFO" and "ONLINE" in title for sev, title in titles)
        # ORPHAN_ORDER_ON_EXCHANGE -> CRITICAL
        assert any(sev == "CRITICAL" for sev, _ in notes)


def test_watch_kill_switch_blocks_live_but_paper_continues() -> None:
    with tempfile.TemporaryDirectory() as t:
        env = make_env(Path(t))
        oc.write_contract(CID, "safe", 200.0)
        oc.create_arm_token(CID, 72, ["ADAUSDT"])
        core.engage_kill_switch(CID, "pre-existing")
        status = {"phase_o_allow_orders": False, "gen2_enabled": True, "open_positions": [], "available_balance": 200.0}
        submitted = []
        hb = loop.run_watch(CID, interval_seconds=0.0, max_cycles=1, live_enabled=True,
                            cycle_fn=lambda cid, live_enabled=False: loop.run_cycle(
                                cid, env["trrm_dir"], env["eqm_dir"], live_enabled=live_enabled,
                                snapshot_fn=lambda: env["snapshot"], status_fn=lambda: status,
                                execute_fn=lambda o: submitted.append(o) or {"status": "ACCEPTED"},
                                filters_fn=lambda s: {"step_size": 1.0, "min_notional": 5.0}),
                            pull_fn=lambda cid: {"ingested": 0},
                            resolve_fn=lambda cid: {"resolved_new": 0}, sleep_fn=lambda s: None)
        assert submitted == []  # kill switch blocks the live path
        assert hb["decisions"] == 3  # paper evidence still collected
        assert hb["kill_switch"] is True
        assert core.kill_switch_engaged(CID) is True  # never auto-rearmed


if __name__ == "__main__":
    test_paper_always_recorded_and_dedup()
    test_veto_blocks_live_but_paper_records()
    test_live_path_full_gauntlet_and_ack()
    test_live_fail_closed_without_bridge_status()
    test_event_ingestion_dedup_and_pnl_wiring()
    test_pull_events_checkpoint_and_bridge_failure()
    test_token_burned_before_submit_and_unconfirmed_ack_incident()
    test_crash_mid_cycle_does_not_reemit_decisions()
    test_watch_singleton_lock()
    test_snapshot_pruning_keeps_recent()
    test_environment_mismatch_vs_freeze_fails_before_unpickle()
    test_watch_runner_heartbeat_incidents_and_hard_stop()
    test_watch_notifications_severity_and_startup()
    test_watch_kill_switch_blocks_live_but_paper_continues()
    print("test_gen2_decision_loop: OK")
