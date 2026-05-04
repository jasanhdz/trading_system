#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import sklearn

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import write_json  # noqa: E402
from aegis_alpha.inference.shadow_schema import build_shadow_signal  # noqa: E402
from aegis_alpha.signals.common import load_signal_market  # noqa: E402
from aegis_alpha.signals.combination_utils import RuleCondition, predict_scores, threshold_for_rule  # noqa: E402
from aegis_alpha.tools.evaluate_long_edge_robustness import ALLOWED_REGIMES  # noqa: E402
from aegis_alpha.tools.evaluate_strategy_candidate_from_json import _candidate_config  # noqa: E402
from aegis_alpha.tools.evaluate_tail_risk_calibration import DEFAULT_CONFIG, _load_estimator, _position_size  # noqa: E402


DEFAULT_CANDIDATE = Path("aegis_alpha/models/strategy_candidates/aegis_h12_tail_risk_candidate_v052.json")
DEFAULT_OUTPUT = Path("aegis_alpha/logs/signals/latest_shadow_signal_v053.json")


def shadow_candidate_dry_run(candidate_path: Path, output_path: Path, config_path: str) -> dict:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    config = _candidate_config(candidate)
    market = load_signal_market(config_path)
    models = {
        "long_edge_h12": _load_estimator(Path(candidate["model_paths"]["long_edge_h12"])),
        "long_tail_risk_h12": _load_estimator(Path(candidate["model_paths"]["long_tail_risk_h12"])),
    }
    preds = predict_scores(market, models)
    entry = candidate.get("entry_rule", {"mode": "top_pct", "value": 0.03})
    edge_threshold = threshold_for_rule(
        preds["long_edge_h12"],
        RuleCondition("long_edge_h12", str(entry.get("mode", "top_pct")), float(entry.get("value", 0.03))),
    )
    band_pcts = sorted({float(band.max_pct) for band in config.bands})
    tail_thresholds = {
        pct: threshold_for_rule(preds["long_tail_risk_h12"], RuleCondition("long_tail_risk_h12", "bottom_pct", pct))
        for pct in band_pcts
    }

    rel_idx = len(preds["long_edge_h12"]) - 1
    step = market.cfg.model.window_size + rel_idx
    timestamp = str(market.timestamps[step])
    regime = str(market.regimes[step])
    edge_score = float(preds["long_edge_h12"][rel_idx])
    tail_score = float(preds["long_tail_risk_h12"][rel_idx])
    position_fraction, size_mode = _position_size(config, tail_score, tail_thresholds)

    action = "HOLD"
    reason = "not_live_shadow_only"
    risk_tier = "blocked"
    if candidate.get("status") != "OFFLINE_CANDIDATE_NOT_LIVE":
        reason = "candidate_status_not_shadow_ready"
    elif candidate.get("live_enabled") is True:
        reason = "candidate_live_flag_unexpected"
    elif regime not in ALLOWED_REGIMES:
        reason = "regime_block"
        position_fraction = 0.0
        size_mode = "skip"
    elif edge_score < edge_threshold:
        reason = "edge_below_h12_top3"
        position_fraction = 0.0
        size_mode = "skip"
    elif position_fraction <= 0.0:
        reason = "tail_risk_block"
        size_mode = "skip"
    else:
        action = "LONG"
        reason = "shadow_entry_conditions_met_not_live"
        risk_tier = "full" if size_mode == "full" else "reduced"

    signal = build_shadow_signal(
        strategy_name=str(candidate.get("config_id", "aegis_h12_tail_risk_candidate")),
        strategy_version="v053-shadow",
        symbol=market.cfg.symbol,
        timestamp=timestamp,
        action=action,
        reason=reason,
        size_mode=size_mode,
        position_fraction=position_fraction,
        edge_score_h12=edge_score,
        tail_risk_score=tail_score,
        regime=regime,
        risk_tier=risk_tier,
        model_status=str(candidate.get("status", "UNKNOWN")),
        not_live_reason=candidate.get("reason_not_live", []),
    )
    payload = {
        "schema_version": "aegis_shadow_candidate_dry_run_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "candidate_path": str(candidate_path),
        "sklearn_version": {
            "candidate": candidate.get("sklearn_version"),
            "runtime": sklearn.__version__,
        },
        "thresholds": {
            "long_edge_h12_top3": float(edge_threshold),
            **{f"long_tail_risk_h12_bottom{int(pct * 100)}": float(value) for pct, value in tail_thresholds.items()},
        },
        "shadow_signal": signal,
    }
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    shadow_candidate_dry_run(Path(args.candidate), Path(args.output), args.config)


if __name__ == "__main__":
    main()
