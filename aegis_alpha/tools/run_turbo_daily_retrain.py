#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG  # noqa: E402
from aegis_alpha.turbo.evaluate_recent_models import evaluate_recent_models  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402
from aegis_alpha.turbo.train_recent_edge import train_recent_edge_models  # noqa: E402


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_turbo_daily_retrain(symbol: str = DEFAULT_TURBO_CONFIG.symbol) -> dict[str, Any]:
    cfg = DEFAULT_TURBO_CONFIG
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    dataset_reports = [build_recent_dataset(symbol, int(days), save=True)["report"] for days in cfg.lookback_days]
    train_report = train_recent_edge_models(symbol)
    eval_report = evaluate_recent_models(symbol)
    status = {
        "schema_version": "aegis_turbo_daily_status_v1",
        "created_at": _utc_stamp(),
        "symbol": symbol,
        "lookback_days": list(cfg.lookback_days),
        "dataset_reports": dataset_reports,
        "train_report_path": train_report.get("report_path"),
        "eval_report_path": eval_report.get("report_path"),
        "turbo_research_enabled": bool(eval_report.get("turbo_research_enabled", False)),
        "note": "daily retrain only prepares Turbo shadow models; it does not restart inference or touch live trading",
    }
    path = cfg.log_dir / f"turbo_daily_status_{_utc_stamp()}.json"
    path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    status["status_path"] = str(path)
    print(json.dumps(status, indent=2, sort_keys=True))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_TURBO_CONFIG.symbol)
    args = parser.parse_args()
    run_turbo_daily_retrain(args.symbol)


if __name__ == "__main__":
    main()
