#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import profit_factor, safe_float, write_json  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402


DEFAULT_CONFIG = "aegis_alpha/configs/base.yaml"
DEFAULT_LOG_GLOB = "aegis_alpha/logs/shadow/*.jsonl"
DEFAULT_OUTPUT_DIR = Path("aegis_alpha/logs/shadow")
HORIZONS = (6, 12, 24, 48)


def _read_jsonl(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path().glob(pattern)):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _path_stats(close: np.ndarray, idx: int, horizon: int, fee_round_trip: float) -> dict[str, float | bool]:
    if idx < 0 or idx + horizon >= len(close):
        return {"pending": True}
    entry = float(close[idx])
    future = close[idx + 1 : idx + horizon + 1]
    if entry <= 0.0 or len(future) == 0:
        return {"pending": True}
    path = future / entry - 1.0
    future_return = float(close[idx + horizon] / entry - 1.0)
    return {
        "pending": False,
        "future_return": safe_float(future_return),
        "mfe": safe_float(np.max(path)),
        "mae": safe_float(max(0.0, -np.min(path))),
        "estimated_net_after_fees": safe_float(future_return - fee_round_trip),
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for value in sorted({str(row.get(key, "unknown")) for row in rows}):
        subset = [row for row in rows if str(row.get(key, "unknown")) == value]
        returns = np.asarray([row["h12_net"] for row in subset if "h12_net" in row], dtype=np.float32)
        out[value] = {
            "count": int(len(subset)),
            "estimated_win_rate": safe_float(np.mean(returns > 0.0)) if len(returns) else 0.0,
            "estimated_profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
            "avg_h12_net": safe_float(np.mean(returns)) if len(returns) else 0.0,
        }
    return out


def evaluate_shadow_log(config_path: str, log_glob: str, output_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(log_glob)
    market = load_signal_market(config_path)
    timestamp_to_step = {str(ts): idx for idx, ts in enumerate(market.timestamps)}
    fee_round_trip = market.cfg.risk.total_fee * 2.0
    long_rows = [row for row in rows if row.get("action") == "LONG" and bool(row.get("would_execute"))]
    enriched_longs: list[dict[str, Any]] = []
    pending_count = 0

    for row in long_rows:
        idx = timestamp_to_step.get(str(row.get("timestamp")))
        enriched = dict(row)
        if idx is None:
            pending_count += 1
            enriched["pending"] = True
            enriched_longs.append(enriched)
            continue
        row_pending = False
        for horizon in HORIZONS:
            stats = _path_stats(market.close, idx, horizon, fee_round_trip)
            if stats.get("pending"):
                row_pending = True
                continue
            enriched[f"future_return_{horizon}"] = stats["future_return"]
            if horizon in (12, 24, 48):
                enriched[f"mfe_{horizon}"] = stats["mfe"]
                enriched[f"mae_{horizon}"] = stats["mae"]
            enriched[f"net_{horizon}"] = stats["estimated_net_after_fees"]
        if row_pending:
            pending_count += 1
        enriched["pending"] = row_pending
        if "net_12" in enriched:
            enriched["h12_net"] = enriched["net_12"]
        enriched_longs.append(enriched)

    completed = [row for row in enriched_longs if not row.get("pending") and "h12_net" in row]
    h12_returns = np.asarray([row["h12_net"] for row in completed], dtype=np.float32)
    report = {
        "schema_version": "aegis_shadow_log_eval_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config_path": config_path,
        "log_glob": log_glob,
        "total_signals": int(len(rows)),
        "hold_count": int(sum(1 for row in rows if row.get("action") in {"HOLD", "IDLE"})),
        "long_shadow_count": int(len(long_rows)),
        "long_shadow_rate": safe_float(len(long_rows) / max(len(rows), 1)),
        "would_execute_true_count": int(sum(1 for row in rows if bool(row.get("would_execute")))),
        "reasons_distribution": dict(Counter(str(row.get("reason", "unknown")) for row in rows)),
        "regimes_distribution": dict(Counter(str(row.get("regime", "unknown")) for row in rows)),
        "size_mode_distribution": dict(Counter(str(row.get("size_mode", "unknown")) for row in rows)),
        "pending_count": int(pending_count),
        "long_shadow": {
            "completed_count": int(len(completed)),
            "pending_count": int(pending_count),
            "estimated_net_after_fees": safe_float(np.mean(h12_returns)) if len(h12_returns) else 0.0,
            "estimated_win_rate": safe_float(np.mean(h12_returns > 0.0)) if len(h12_returns) else 0.0,
            "estimated_profit_factor": safe_float(profit_factor(h12_returns)) if len(h12_returns) else 0.0,
            "avg_edge_score_h12": safe_float(np.mean([row.get("edge_score_h12", 0.0) for row in completed])) if completed else 0.0,
            "avg_tail_risk_score": safe_float(np.mean([row.get("tail_risk_score", 0.0) for row in completed])) if completed else 0.0,
            "by_regime": _group_summary(completed, "regime"),
            "by_size_mode": _group_summary(completed, "size_mode"),
        },
    }
    for horizon in HORIZONS:
        values = [row[f"future_return_{horizon}"] for row in completed if f"future_return_{horizon}" in row]
        report["long_shadow"][f"future_return_{horizon}"] = safe_float(np.mean(values)) if values else 0.0
    for horizon in (12, 24, 48):
        mfes = [row[f"mfe_{horizon}"] for row in completed if f"mfe_{horizon}" in row]
        maes = [row[f"mae_{horizon}"] for row in completed if f"mae_{horizon}" in row]
        report["long_shadow"][f"MFE_{horizon}"] = safe_float(np.mean(mfes)) if mfes else 0.0
        report["long_shadow"][f"MAE_{horizon}"] = safe_float(np.mean(maes)) if maes else 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"shadow_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Shadow eval report -> {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--logs", default=DEFAULT_LOG_GLOB)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    evaluate_shadow_log(args.config, args.logs, Path(args.output_dir))


if __name__ == "__main__":
    main()
