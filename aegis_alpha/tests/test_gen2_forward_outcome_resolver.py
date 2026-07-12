#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_forward_outcome_resolver as resolver

CID = "gen2-test"
BAR_MS = 5 * 60_000


def setup(tmp: Path) -> Path:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "freeze.json"
    core.FREEZE_PATH.write_text(json.dumps({"candidate_id": CID, "trrm_v2_sha256": "a", "eqm1_sha256": "b", "d3_dataset_sha256": "c", "feature_hash": "d"}))
    core.init_canary(CID)
    decisions = tmp / "decisions.jsonl"
    return decisions


def fake_fetch_factory(entry_open: float = 100.0, drift_per_bar: float = -0.1, spike_high: float | None = None):
    def fake_fetch(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        n = int((end_ms - start_ms) // BAR_MS) + 1
        opens = entry_open + drift_per_bar * np.arange(n)
        rows = []
        for i in range(n):
            high = opens[i] + 0.05
            if spike_high is not None and i == 3:
                high = spike_high
            rows.append({"open_time": start_ms + i * BAR_MS, "open": opens[i], "high": high,
                         "low": opens[i] - 0.2, "close": opens[i] + drift_per_bar, "volume": 1.0,
                         "close_time": start_ms + (i + 1) * BAR_MS - 1})
        return pd.DataFrame(rows)
    return fake_fetch


def write_decision(path: Path, ts: str, symbol: str = "ADAUSDT", action: str = "CANDIDATE_SHORT", veto: bool = False) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"candidate_id": CID, "symbol": symbol, "ts": ts,
                            "hypothetical_action": action, "vetoed_by_trrm": veto,
                            "eqm_score": 0.1, "tail_score": 0.05, "qmae_q90": 0.1}) + "\n")


def test_resolver_computes_outcomes_and_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        decisions = setup(tmp)
        base = pd.Timestamp("2026-07-12 00:00:00")
        write_decision(decisions, str(base))                                   # mature
        write_decision(decisions, str(base + pd.Timedelta(minutes=30)), veto=True)  # mature, vetoed
        write_decision(decisions, str(base + pd.Timedelta(hours=5)))           # immature
        now = pd.Timestamp("2026-07-12 02:00:00")
        s1 = resolver.resolve(CID, decisions, fake_fetch_factory(), now)
        assert s1["resolved_new"] == 2 and s1["skipped_immature"] == 1
        # short in downtrend: positive net return; tail label 0
        rows = resolver.read_jsonl(core.canary_dir(CID) / "forward_outcomes.jsonl")
        taken = [r for r in rows if r["hypothetical_action"] == "CANDIDATE_SHORT" and not r["vetoed_by_trrm"]]
        assert taken and taken[0]["net_short_return_pct"] > 0
        assert taken[0]["tail_risk_roe_030"] == 0
        assert taken[0]["final_candles_used"] == 13
        # idempotent: second run resolves nothing new, evidence not mutated
        before = (core.canary_dir(CID) / "forward_outcomes.jsonl").read_bytes()
        s2 = resolver.resolve(CID, decisions, fake_fetch_factory(), now)
        assert s2["resolved_new"] == 0 and s2["skipped_already_resolved"] == 2
        assert (core.canary_dir(CID) / "forward_outcomes.jsonl").read_bytes() == before
        assert s2["policy_changed"] is False and s2["orders_submitted"] == 0


def test_tail_label_from_spike() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        decisions = setup(tmp)
        base = pd.Timestamp("2026-07-12 00:00:00")
        write_decision(decisions, str(base))
        # spike high +2% -> mae_roe 0.4 >= 0.30 -> tail=1
        s = resolver.resolve(CID, decisions, fake_fetch_factory(spike_high=102.5), pd.Timestamp("2026-07-12 02:00:00"))
        assert s["candidate_short_tail_rate"] == 1.0


def test_fetch_failure_is_incident_not_crash() -> None:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        decisions = setup(tmp)
        write_decision(decisions, "2026-07-12 00:00:00")

        def boom(symbol, s, e):
            raise RuntimeError("network down")

        s = resolver.resolve(CID, decisions, boom, pd.Timestamp("2026-07-12 02:00:00"))
        assert s["resolved_new"] == 0 and s["failed_no_candles"] == 1
        incidents = resolver.read_jsonl(core.canary_dir(CID) / "incidents" / "incidents.jsonl")
        assert any(i["type"] == "OUTCOME_FETCH_FAILED" for i in incidents)


if __name__ == "__main__":
    test_resolver_computes_outcomes_and_is_idempotent()
    test_tail_label_from_spike()
    test_fetch_failure_is_incident_not_crash()
    print("test_gen2_forward_outcome_resolver: OK")
