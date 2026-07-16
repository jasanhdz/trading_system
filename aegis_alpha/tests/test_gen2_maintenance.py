#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aegis_alpha.tools.gen2_canary_core as core
import aegis_alpha.tools.gen2_maintenance as m

CID = "gen2-test"


def setup(tmp: Path) -> Path:
    core.CANARY_ROOT = tmp / "live_canary"
    core.FREEZE_PATH = tmp / "freeze.json"
    core.FREEZE_PATH.write_text(json.dumps({"candidate_id": CID, "trrm_v2_sha256": "a",
                                            "eqm1_sha256": "b", "d3_dataset_sha256": "c", "feature_hash": "d"}))
    core.init_canary(CID)
    m.FORWARD_ROOT = tmp / "forward"
    (m.FORWARD_ROOT).mkdir()
    return core.canary_dir(CID)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_evidence_segmentation_preserves_all_rows() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        now = datetime(2026, 7, 16, tzinfo=timezone.utc)
        fd = m.FORWARD_ROOT / "forward_decisions.jsonl"
        rows = ([{"collected_at": "2026-05-10T00:00:00+00:00", "symbol": f"OLD{i}"} for i in range(5)]
                + [{"collected_at": "2026-06-01T00:00:00+00:00", "symbol": f"JUN{i}"} for i in range(3)]
                + [{"collected_at": "2026-07-15T00:00:00+00:00", "symbol": f"CUR{i}"} for i in range(4)]
                + [{"symbol": "UNDATED"}])  # undated stays active
        write_rows(fd, rows)
        r = m.segment_evidence(fd, now, dry_run=False)
        assert r["moved"] == 8 and r["kept"] == 5  # 4 current + 1 undated
        assert r["archives"] == {"2026-05": 5, "2026-06": 3}
        # total rows preserved across active + archives
        active = m.read_jsonl(fd)
        arch_may = m.read_jsonl(m.FORWARD_ROOT / "forward_decisions_2026-05.jsonl")
        arch_jun = m.read_jsonl(m.FORWARD_ROOT / "forward_decisions_2026-06.jsonl")
        assert len(active) + len(arch_may) + len(arch_jun) == 13
        assert any(x["symbol"] == "UNDATED" for x in active)


def test_segmentation_idempotent() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        now = datetime(2026, 7, 16, tzinfo=timezone.utc)
        fd = m.FORWARD_ROOT / "selection_outcomes.jsonl"
        write_rows(fd, [{"collected_at": "2026-05-10T00:00:00+00:00", "x": i} for i in range(4)])
        m.segment_evidence(fd, now, dry_run=False)
        a1 = m.read_jsonl(m.FORWARD_ROOT / "selection_outcomes_2026-05.jsonl")
        m.segment_evidence(fd, now, dry_run=False)  # second run: nothing left to move
        a2 = m.read_jsonl(m.FORWARD_ROOT / "selection_outcomes_2026-05.jsonl")
        assert len(a1) == 4 and len(a2) == 4  # not duplicated


def test_dry_run_does_not_modify_files() -> None:
    with tempfile.TemporaryDirectory() as t:
        cdir = setup(Path(t))
        fd = m.FORWARD_ROOT / "forward_decisions.jsonl"
        write_rows(fd, [{"collected_at": "2026-01-01T00:00:00+00:00", "x": i} for i in range(6)])
        before = fd.read_bytes()
        m.run_maintenance(CID, "dry-run")
        assert fd.read_bytes() == before  # untouched
        assert not (m.FORWARD_ROOT / "forward_decisions_2026-01.jsonl").exists()


def test_log_rotation_and_incident_summary() -> None:
    with tempfile.TemporaryDirectory() as t:
        cdir = setup(Path(t))
        now = datetime(2026, 7, 16, tzinfo=timezone.utc)
        inc = cdir / "incidents" / "incidents.jsonl"
        old = "2026-01-01T00:00:00+00:00"
        recent = "2026-07-15T00:00:00+00:00"
        write_rows(inc, [{"recorded_at": old, "type": "A"}, {"recorded_at": old, "type": "B"},
                         {"recorded_at": recent, "type": "C"}])
        r = m.rotate_log(inc, 45, now, dry_run=False)
        assert r["rotated"] == 2 and r["kept"] == 1
        assert m.read_jsonl(cdir / "incidents" / "incidents_summary.jsonl")[0]["by_type"] == {"A": 1, "B": 1}
        assert len(m.read_jsonl(inc)) == 1  # only the recent one remains active


def test_torn_last_line_tolerated() -> None:
    with tempfile.TemporaryDirectory() as t:
        setup(Path(t))
        fd = m.FORWARD_ROOT / "forward_decisions.jsonl"
        fd.write_text('{"collected_at": "2026-05-01T00:00:00+00:00", "x": 1}\n{"collected_at": "2026-05')
        rows = m.read_jsonl(fd)
        assert len(rows) == 1  # torn line skipped, intact row read


def test_state_files_are_report_only() -> None:
    with tempfile.TemporaryDirectory() as t:
        cdir = setup(Path(t))
        (cdir / "events_seen.json").write_text(json.dumps(["a|1", "b|2"]))
        a = m.audit(CID)
        state = [i for i in a["items"] if i["class"] == "B_STATE_DEDUP"]
        assert any("events_seen" in i["path"] for i in state)
        assert all("report only" in i["policy"] for i in state)


if __name__ == "__main__":
    test_evidence_segmentation_preserves_all_rows()
    test_segmentation_idempotent()
    test_dry_run_does_not_modify_files()
    test_log_rotation_and_incident_summary()
    test_torn_last_line_tolerated()
    test_state_files_are_report_only()
    print("test_gen2_maintenance: OK")
