#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import profit_factor, safe_float  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402


DEFAULT_CONFIG = "aegis_alpha/configs/base.yaml"
DEFAULT_LOG_GLOB = "aegis_alpha/logs/turbo/turbo_shadow_*.jsonl"
DEFAULT_OUTPUT_DIR = Path("aegis_alpha/logs/turbo")
HORIZONS = (3, 6, 12, 24)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_rows(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(glob.glob(pattern)):
        path = Path(item)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _path_stats(close: np.ndarray, idx: int, horizon: int, action: str) -> dict[str, Any]:
    if idx < 0 or idx + horizon >= len(close):
        return {"pending": True}
    entry = float(close[idx])
    future = close[idx + 1 : idx + horizon + 1]
    if entry <= 0.0 or len(future) < horizon:
        return {"pending": True}
    if action == "SHORT":
        path = entry / np.maximum(future, 1e-10) - 1.0
        ret = entry / float(close[idx + horizon]) - 1.0
    else:
        path = future / entry - 1.0
        ret = float(close[idx + horizon]) / entry - 1.0
    return {
        "pending": False,
        "future_return": safe_float(ret),
        "mfe": safe_float(np.max(path)),
        "mae": safe_float(max(0.0, -np.min(path))),
    }


def _trailing_exit_roe(path_returns: list[float], leverage: float, activation_roe: float, callback_roe: float) -> float | None:
    if not path_returns or leverage <= 0.0:
        return None
    activation = activation_roe / 100.0
    callback = callback_roe / 100.0
    best_roe = -999.0
    activated = False
    for ret in path_returns:
        roe = ret * leverage
        best_roe = max(best_roe, roe)
        if best_roe >= activation:
            activated = True
        if activated and best_roe - roe >= callback:
            return safe_float(roe * 100.0)
    return None


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for value in sorted({str(row.get(key, "unknown")) for row in rows}):
        subset = [row for row in rows if str(row.get(key, "unknown")) == value]
        returns = np.asarray([row.get("estimated_account_return", 0.0) for row in subset], dtype=np.float32)
        out[value] = {
            "count": int(len(subset)),
            "avg_estimated_account_return": safe_float(np.mean(returns)) if len(returns) else 0.0,
            "win_rate": safe_float(np.mean(returns > 0.0)) if len(returns) else 0.0,
            "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
        }
    return out


def _max_loss_streak(returns: list[float]) -> int:
    best = 0
    current = 0
    for value in returns:
        if value < 0.0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def evaluate_turbo_shadow_log(config_path: str, log_glob: str, output_dir: Path) -> dict[str, Any]:
    rows = _read_rows(log_glob)
    market = load_signal_market(config_path)
    timestamp_to_step = {str(ts): idx for idx, ts in enumerate(market.timestamps)}
    completed: list[dict[str, Any]] = []
    pending_count = 0

    for row in rows:
        if row.get("action") not in {"LONG", "SHORT"} or not bool(row.get("would_execute")):
            continue
        idx = timestamp_to_step.get(str(row.get("timestamp")))
        if idx is None:
            pending_count += 1
            continue
        enriched = dict(row)
        action = str(row.get("action"))
        leverage = float(row.get("leverage_suggestion", 0.0) or 0.0)
        fraction = float(row.get("position_fraction", 0.0) or 0.0)
        row_pending = False
        for horizon in HORIZONS:
            stats = _path_stats(market.close, idx, horizon, action)
            if stats.get("pending"):
                row_pending = True
                continue
            ret = float(stats["future_return"])
            enriched[f"future_return_{horizon}"] = safe_float(ret)
            enriched[f"MFE_{horizon}"] = stats["mfe"]
            enriched[f"MAE_{horizon}"] = stats["mae"]
            enriched[f"estimated_roe_{horizon}"] = safe_float(ret * leverage * 100.0)
            enriched[f"estimated_account_return_{horizon}"] = safe_float(ret * leverage * fraction)
        if row_pending:
            pending_count += 1
        if "estimated_roe_12" in enriched:
            enriched["estimated_roe"] = enriched["estimated_roe_12"]
            enriched["estimated_account_return"] = enriched["estimated_account_return_12"]
        safe_context = row.get("safe_context") or {}
        enriched["regime"] = safe_context.get("regime", "unknown")
        completed.append(enriched)

    account_returns = [float(row.get("estimated_account_return", 0.0)) for row in completed if "estimated_account_return" in row]
    roe_values = [float(row.get("estimated_roe", 0.0)) for row in completed if "estimated_roe" in row]
    report = {
        "schema_version": "aegis_turbo_shadow_eval_v1",
        "created_at": _utc_stamp(),
        "config_path": config_path,
        "log_glob": log_glob,
        "total_evaluations": int(len(rows)),
        "would_execute_count": int(sum(1 for row in rows if bool(row.get("would_execute")))),
        "long_count": int(sum(1 for row in rows if row.get("action") == "LONG" and bool(row.get("would_execute")))),
        "short_count": int(sum(1 for row in rows if row.get("action") == "SHORT" and bool(row.get("would_execute")))),
        "hold_count": int(sum(1 for row in rows if row.get("action") == "HOLD")),
        "pending_count": int(pending_count),
        "avg_estimated_roe": safe_float(np.mean(roe_values)) if roe_values else 0.0,
        "avg_estimated_account_return": safe_float(np.mean(account_returns)) if account_returns else 0.0,
        "win_rate": safe_float(np.mean(np.asarray(account_returns) > 0.0)) if account_returns else 0.0,
        "profit_factor": safe_float(profit_factor(np.asarray(account_returns, dtype=np.float32))) if account_returns else 0.0,
        "max_loss_streak": _max_loss_streak(account_returns),
        "reasons_distribution": dict(Counter(str(row.get("reason", "unknown")) for row in rows)),
        "by_confidence": _group(completed, "confidence"),
        "by_leverage_suggestion": _group(completed, "leverage_suggestion"),
        "by_regime": _group(completed, "regime"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"turbo_shadow_eval_{_utc_stamp()}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Turbo shadow eval report -> {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--logs", default=DEFAULT_LOG_GLOB)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    evaluate_turbo_shadow_log(args.config, args.logs, Path(args.output_dir))


if __name__ == "__main__":
    main()
