#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import safe_float  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.signals.horizon_targets import HORIZONS, build_horizon_targets  # noqa: E402


DEFAULT_OUTPUT = Path("aegis_alpha/data/processed/signal_lab_dataset_v050.npz")
DEFAULT_REPORT = Path("aegis_alpha/logs/signals/signal_lab_dataset_v050.json")
DEFAULT_CONFIG = "aegis_alpha/configs/base.yaml"
DEFAULT_PROFIT_THRESHOLD = 0.0030
DEFAULT_RISK_THRESHOLD = 0.0030


def _target_summary(values: np.ndarray, positive_mask: np.ndarray | None = None) -> dict[str, float]:
    arr = np.asarray(values)
    out = {
        "count": safe_float(len(arr)),
        "rate": safe_float(np.mean(arr > 0.0)) if arr.dtype.kind == "f" else safe_float(np.mean(arr.astype(np.float32) > 0.0)),
        "mean": safe_float(np.mean(arr)) if len(arr) else 0.0,
    }
    if positive_mask is not None and len(arr):
        out["filtered_rate"] = safe_float(np.mean(arr[positive_mask] > 0.0)) if np.any(positive_mask) else 0.0
    return out


def build_signal_lab_dataset(
    config_path: str,
    output_path: Path,
    report_path: Path,
    profit_threshold: float,
    risk_threshold: float,
) -> dict[str, Any]:
    market = load_signal_market(config_path)
    cfg = market.cfg
    max_horizon = max(HORIZONS)
    last_step = len(market.close) - max_horizon - 1
    if last_step <= cfg.model.window_size:
        raise RuntimeError(f"Not enough rows for window={cfg.model.window_size}, horizon={max_horizon}")

    x_full = market.signal_features
    steps = np.arange(cfg.model.window_size, last_step + 1, dtype=np.int64)
    x = x_full[: len(steps)].astype(np.float32)
    timestamps = market.timestamps[steps]
    close = market.close[steps]
    regimes = market.regimes[steps]
    targets = build_horizon_targets(
        close=market.close,
        steps=steps,
        total_fee=cfg.risk.total_fee,
        horizons=HORIZONS,
        profit_threshold=profit_threshold,
        risk_threshold=risk_threshold,
    )

    save_kwargs: dict[str, Any] = {
        "X": x,
        "x": x,
        "timestamp": timestamps,
        "step": steps,
        "close": close,
        "regime": regimes,
        "regimes": regimes,
        "feature_names": np.asarray(market.feature_names),
        "horizons": np.asarray(HORIZONS, dtype=np.int16),
        "profit_threshold": np.asarray(profit_threshold, dtype=np.float32),
        "risk_threshold": np.asarray(risk_threshold, dtype=np.float32),
        "window_size": np.asarray(cfg.model.window_size, dtype=np.int16),
        "fee_round_trip": np.asarray(cfg.risk.total_fee * 2.0, dtype=np.float32),
        "config": np.asarray(json.dumps({"config_path": config_path}), dtype=object),
    }
    save_kwargs.update(targets)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save_kwargs)

    target_report: dict[str, Any] = {}
    for horizon in HORIZONS:
        long_good = targets[f"h{horizon}_long_good"].astype(np.float32)
        short_good = targets[f"h{horizon}_short_good"].astype(np.float32)
        no_trade = targets[f"h{horizon}_no_trade"].astype(np.float32)
        failure_bad_long = targets[f"h{horizon}_failure_bad_long"].astype(np.float32)
        failure_bad_short = targets[f"h{horizon}_failure_bad_short"].astype(np.float32)
        target_report[f"h{horizon}"] = {
            "long_good_rate": safe_float(np.mean(long_good)),
            "short_good_rate": safe_float(np.mean(short_good)),
            "no_trade_rate": safe_float(np.mean(no_trade)),
            "avg_long_return": safe_float(np.mean(targets[f"h{horizon}_long_net_return"])),
            "avg_short_return": safe_float(np.mean(targets[f"h{horizon}_short_net_return"])),
            "failure_bad_long_rate": safe_float(np.mean(failure_bad_long)),
            "failure_bad_short_rate": safe_float(np.mean(failure_bad_short)),
        }

    report = {
        "schema_version": "aegis_signal_lab_dataset_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config_path": config_path,
        "output_path": str(output_path),
        "sample_count": int(len(x)),
        "feature_count": int(x.shape[1]),
        "date_start": str(timestamps[0]),
        "date_end": str(timestamps[-1]),
        "window_size": int(cfg.model.window_size),
        "profit_threshold": float(profit_threshold),
        "risk_threshold": float(risk_threshold),
        "target_distributions": target_report,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Dataset saved -> {output_path}")
    print(f"Report saved -> {report_path}")
    print(f"Samples: {len(x):,} features: {x.shape[1]:,}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--profit-threshold", type=float, default=DEFAULT_PROFIT_THRESHOLD)
    parser.add_argument("--risk-threshold", type=float, default=DEFAULT_RISK_THRESHOLD)
    args = parser.parse_args()
    build_signal_lab_dataset(
        config_path=args.config,
        output_path=Path(args.output),
        report_path=Path(args.report),
        profit_threshold=args.profit_threshold,
        risk_threshold=args.risk_threshold,
    )


if __name__ == "__main__":
    main()
