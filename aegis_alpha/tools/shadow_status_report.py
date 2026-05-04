#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_CONFIG = "aegis_alpha/configs/base.yaml"
DEFAULT_LOG_GLOB = "aegis_alpha/logs/shadow/*.jsonl"
MAX_EVAL_HORIZON = 48


def _read_rows(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(glob.glob(pattern)):
        path = Path(item)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_log_file"] = str(path)
                    rows.append(row)
    return rows


def shadow_status_report(config_path: str, log_glob: str) -> dict[str, Any]:
    rows = _read_rows(log_glob)
    long_rows = [row for row in rows if row.get("action") == "LONG" and bool(row.get("would_execute"))]
    pending = len(long_rows)
    ready = 0
    warning = None
    if rows and long_rows:
        try:
            from aegis_alpha.signals.common import load_signal_market

            market = load_signal_market(config_path)
            timestamp_to_step = {str(ts): idx for idx, ts in enumerate(market.timestamps)}
            pending = 0
            ready = 0
            for row in long_rows:
                step = timestamp_to_step.get(str(row.get("timestamp")))
                if step is None or step + MAX_EVAL_HORIZON >= len(market.close):
                    pending += 1
                else:
                    ready += 1
        except Exception as exc:
            warning = f"market_data_unavailable: {exc!r}"
    latest = rows[-1] if rows else None
    report = {
        "total_shadow_evaluations": int(len(rows)),
        "long_count": int(len(long_rows)),
        "hold_count": int(sum(1 for row in rows if row.get("action") in {"HOLD", "IDLE"})),
        "reasons_distribution": dict(Counter(str(row.get("reason", "unknown")) for row in rows)),
        "latest_signal": latest,
        "signals_pending_evaluation": int(pending),
        "signals_ready_for_evaluation": int(ready),
    }
    if warning is not None:
        report["evaluation_status_warning"] = warning
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--logs", default=DEFAULT_LOG_GLOB)
    args = parser.parse_args()
    shadow_status_report(args.config, args.logs)


if __name__ == "__main__":
    main()
