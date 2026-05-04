#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import safe_float, write_json  # noqa: E402
from aegis_alpha.signals.combination_utils import RuleCondition, predict_scores, threshold_for_rule  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.tools.evaluate_tail_risk_calibration import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    CalibrationConfig,
    SizeBand,
    _evaluate_window,
    _load_estimator,
    _select_windows,
    _summary,
)


DEFAULT_CANDIDATE = Path("aegis_alpha/models/strategy_candidates/aegis_h12_tail_risk_candidate_v052.json")
DEFAULT_REPORT = Path("aegis_alpha/logs/signals/candidate_repro_eval_v053.json")
COMPARE_METRICS = (
    "median_balance",
    "p25_balance",
    "worst_balance",
    "p25_pf",
    "profitable_window_pct",
    "median_trades",
    "worst_max_dd",
)
TOLERANCES = {
    "median_balance": 1e-4,
    "p25_balance": 1e-4,
    "worst_balance": 1e-4,
    "p25_pf": 1e-4,
    "profitable_window_pct": 1e-8,
    "median_trades": 1e-8,
    "worst_max_dd": 1e-8,
}


def _candidate_config(candidate: dict[str, Any]) -> CalibrationConfig:
    sizing = candidate.get("sizing_config") or {}
    bands_raw = sizing.get("bands") or candidate.get("oos_metrics", {}).get("bands") or []
    bands = tuple(
        SizeBand(
            max_pct=float(item["max_pct"]),
            fraction=float(item["fraction"]),
            label=str(item["label"]),
        )
        for item in bands_raw
    )
    if not bands:
        raise ValueError("Candidate has no sizing bands")
    return CalibrationConfig(
        config_id=str(candidate["config_id"]),
        sizing_mode=str(sizing.get("mode", candidate.get("sizing_mode"))),
        tail_threshold=float(sizing.get("tail_threshold", candidate.get("tail_threshold"))),
        bands=bands,
    )


def evaluate_strategy_candidate_from_json(
    candidate_path: Path,
    report_path: Path,
    config_path: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_config = _candidate_config(candidate)
    market = load_signal_market(config_path)
    frozen_windows = candidate.get("oos_evaluation", {}).get("windows")
    if frozen_windows:
        windows = [(int(item["start_step"]), str(item["source"])) for item in frozen_windows]
        window_source = "candidate_json"
    else:
        windows = _select_windows(market, seeds)
        window_source = "selected_from_seeds"
    model_paths = candidate["model_paths"]
    models = {
        "long_edge_h12": _load_estimator(Path(model_paths["long_edge_h12"])),
        "long_tail_risk_h12": _load_estimator(Path(model_paths["long_tail_risk_h12"])),
    }
    preds = predict_scores(market, models)
    entry = candidate.get("entry_rule", {"mode": "top_pct", "value": 0.03})
    edge_threshold = threshold_for_rule(
        preds["long_edge_h12"],
        RuleCondition("long_edge_h12", str(entry.get("mode", "top_pct")), float(entry.get("value", 0.03))),
    )
    band_pcts = sorted({float(band.max_pct) for band in candidate_config.bands})
    tail_thresholds = {
        pct: threshold_for_rule(preds["long_tail_risk_h12"], RuleCondition("long_tail_risk_h12", "bottom_pct", pct))
        for pct in band_pcts
    }
    fee_multiplier = float(candidate.get("fee_multiplier", candidate.get("oos_metrics", {}).get("fee_multiplier", 1.0)))
    combo_windows = [
        _evaluate_window(
            market=market,
            preds=preds,
            config=candidate_config,
            edge_threshold=edge_threshold,
            tail_thresholds=tail_thresholds,
            start_step=int(start_step),
            source=source,
            fee_multiplier=fee_multiplier,
        )
        for start_step, source in windows
    ]
    reproduced = _summary(candidate_config, fee_multiplier, combo_windows, market.cfg.risk.initial_balance)
    expected = candidate.get("oos_metrics", {})
    comparisons: dict[str, dict[str, float | bool]] = {}
    reproducibility_passed = True
    for metric in COMPARE_METRICS:
        expected_value = float(expected[metric])
        reproduced_value = float(reproduced[metric])
        diff = abs(reproduced_value - expected_value)
        tolerance = TOLERANCES[metric]
        passed = diff <= tolerance
        reproducibility_passed = reproducibility_passed and passed
        comparisons[metric] = {
            "expected": safe_float(expected_value),
            "reproduced": safe_float(reproduced_value),
            "abs_diff": safe_float(diff),
            "tolerance": safe_float(tolerance),
            "passed": bool(passed),
        }

    report = {
        "schema_version": "aegis_candidate_repro_eval_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "candidate_path": str(candidate_path),
        "candidate_status": candidate.get("status"),
        "sklearn_version": {
            "candidate": candidate.get("sklearn_version"),
            "runtime": sklearn.__version__,
        },
        "config_path": config_path,
        "seeds": list(seeds),
        "window_count": len(windows),
        "window_source": window_source,
        "model_paths": model_paths,
        "entry_rule": candidate.get("entry_rule"),
        "sizing_config": candidate.get("sizing_config"),
        "fee_multiplier": fee_multiplier,
        "thresholds": {
            "long_edge_h12": safe_float(edge_threshold),
            **{f"long_tail_risk_h12_bottom{int(pct * 100)}": safe_float(value) for pct, value in tail_thresholds.items()},
        },
        "reproducibility_passed": bool(reproducibility_passed),
        "comparisons": comparisons,
        "reproduced_metrics": {metric: reproduced[metric] for metric in COMPARE_METRICS},
        "expected_metrics": {metric: expected[metric] for metric in COMPARE_METRICS},
        "windows": combo_windows,
    }
    write_json(report_path, report)
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    args = parser.parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    evaluate_strategy_candidate_from_json(
        candidate_path=Path(args.candidate),
        report_path=Path(args.report),
        config_path=args.config,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
