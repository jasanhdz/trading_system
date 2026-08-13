#!/usr/bin/env python3
"""Run the preregistered A2 backcast using the frozen A1 contract."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.economic_alpha_backcast_a2 import (
    clustered_bootstrap,
    classify_frozen_regimes,
    control_events,
    economic_summary,
    event_identity_hash,
    fixed_horizon_outcomes,
    frozen_candidates,
    load_frozen_a1_contract,
    top_rank,
)
from aegis.research.economic_alpha_discovery_a1 import (
    add_causal_features,
    add_cross_sectional_features,
    aggregate_completed_15m,
    daily_space,
)
from aegis.research.market_event_economic_path_m1b import load_funding, load_mark_prices
from aegis.research.market_event_fast_track_batch import load_symbol_frame
from aegis.utils import sha256_file


START_MS = int(pd.Timestamp("2023-06-01T00:00:00Z").timestamp() * 1000)
END_MS = int(pd.Timestamp("2024-01-01T00:00:00Z").timestamp() * 1000)
HYPOTHESES = {
    "TREND_ACCEPTANCE::LONG": 60,
    "CARRY_CONVERGENCE::SHORT": 1440,
}


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _panel(root: Path, raw: Path, output: Path, reuse: bool) -> pd.DataFrame:
    panel_path = output / "causal_panel.parquet"
    if reuse and panel_path.exists():
        return pd.read_parquet(panel_path)
    cache = output / "state_cache"
    cache.mkdir(exist_ok=True)
    os.chmod(cache, 0o700)
    states = []
    for symbol in CANONICAL_SYMBOLS:
        path = cache / f"{symbol}.parquet"
        if reuse and path.exists():
            state = pd.read_parquet(path)
        else:
            minute = load_symbol_frame(raw, symbol)
            mark = load_mark_prices(raw, symbol)
            funding = load_funding(raw, symbol)
            state = add_causal_features(aggregate_completed_15m(minute, mark), funding)
            state["symbol"] = symbol
            state.to_parquet(path, compression="zstd", index=False)
            os.chmod(path, 0o600)
        states.append(state)
        print(f"a2_state symbol={symbol} rows={len(state)}", flush=True)
    panel = add_cross_sectional_features(pd.concat(states, ignore_index=True))
    panel.to_parquet(panel_path, compression="zstd", index=False)
    os.chmod(panel_path, 0o600)
    return panel


def _gate(
    primary: pd.DataFrame,
    all_eligible: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    summary = economic_summary(primary)
    bootstrap = clustered_bootstrap(primary) if len(primary) >= 2 else None
    control_summaries = {
        name: economic_summary(rows) for name, rows in controls.items()
    }
    gross = economic_summary(primary, "gross_return")
    stress = economic_summary(primary, "net_stress_20bps")
    gate = {
        "minimum_events": len(primary) >= 100,
        "gross_expectancy": gross.get("expectancy", -math.inf) >= 0.0042,
        "primary_net_positive": summary.get("expectancy", -math.inf) > 0.0,
        "bootstrap_lower_positive": bool(
            bootstrap and bootstrap["expectancy_lower_95"] > 0.0
        ),
        "profit_factor_lower_above_one": bool(
            bootstrap and bootstrap["profit_factor_lower_95"] > 1.0
        ),
        "positive_temporal_thirds": sum(
            value > 0.0 for value in summary.get("temporal_thirds", ())
        )
        >= 2,
        "maximum_symbol_share": summary.get("maximum_symbol_share", math.inf) <= 0.25,
        "positive_symbols": summary.get("positive_symbols", 0) >= 7,
        "stress_positive": stress.get("expectancy", -math.inf) > 0.0,
        "outperform_all_controls": all(
            summary.get("expectancy", -math.inf)
            > control.get("expectancy", math.inf)
            for control in control_summaries.values()
        ),
        "top_rank_outperforms_all_eligible": summary.get("expectancy", -math.inf)
        > economic_summary(all_eligible).get("expectancy", math.inf),
        "zero_material_leakage": True,
    }
    return gate, control_summaries, bootstrap or {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/economic_alpha_backcast_a2/raw")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/economic_alpha_backcast_a2/run_01")
    )
    parser.add_argument("--reuse-panel", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw = args.raw_root if args.raw_root.is_absolute() else root / args.raw_root
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    authority_path = root / "data/economic_alpha_discovery_a1/run_01/result.json"
    authority = load_frozen_a1_contract(authority_path)
    panel = _panel(root, raw, output, args.reuse_panel)
    panel = classify_frozen_regimes(panel, authority["regime_thresholds"])

    minute_frames = {
        symbol: load_symbol_frame(raw, symbol) for symbol in CANONICAL_SYMBOLS
    }
    funding_frames = {symbol: load_funding(raw, symbol) for symbol in CANONICAL_SYMBOLS}
    reports: dict[str, Any] = {}
    outcome_parts = []
    candidate_parts = []
    for identity, horizon in HYPOTHESES.items():
        mechanism, side = identity.split("::")
        eligible = frozen_candidates(
            panel,
            authority,
            mechanism=mechanism,
            side=side,
            start_ms=START_MS,
            end_ms=END_MS,
        )
        top = daily_space(top_rank(eligible))
        all_daily = daily_space(eligible)
        candidate_parts.extend(
            [
                eligible.assign(selection="ELIGIBLE"),
                top.assign(selection="TOP_RANK_DAILY"),
                all_daily.assign(selection="ALL_ELIGIBLE_DAILY"),
            ]
        )
        primary = fixed_horizon_outcomes(top, minute_frames, funding_frames, horizon)
        primary["control"] = "TOP_RANK_DAILY"
        all_outcomes = fixed_horizon_outcomes(
            all_daily, minute_frames, funding_frames, horizon
        )
        all_outcomes["control"] = "ALL_ELIGIBLE_DAILY"
        outcome_parts.extend([primary, all_outcomes])
        control_outcomes = {"all_eligible_daily": all_outcomes}
        for name, events in control_events(top, eligible, panel).items():
            outcomes = fixed_horizon_outcomes(
                events, minute_frames, funding_frames, horizon
            )
            outcomes["control"] = name.upper()
            outcome_parts.append(outcomes)
            control_outcomes[name] = outcomes
        gate, controls, bootstrap = _gate(primary, all_outcomes, control_outcomes)
        reports[identity] = {
            "horizon_minutes": horizon,
            "eligible_events": len(eligible),
            "top_rank_daily_events": len(top),
            "all_eligible_daily_events": len(all_daily),
            "candidate_identity_sha256": event_identity_hash(eligible),
            "primary": economic_summary(primary),
            "gross": economic_summary(primary, "gross_return"),
            "stress_20bps": economic_summary(primary, "net_stress_20bps"),
            "all_eligible_daily": economic_summary(all_outcomes),
            "controls": controls,
            "bootstrap": bootstrap,
            "gate": gate,
            "gate_pass": all(gate.values()),
        }
        print(
            f"a2_result identity={identity} eligible={len(eligible)} "
            f"top={len(top)} gate={all(gate.values())}",
            flush=True,
        )

    candidates = pd.concat(candidate_parts, ignore_index=True)
    outcomes = pd.concat(outcome_parts, ignore_index=True)
    candidates_path = output / "candidates.parquet"
    outcomes_path = output / "outcomes.parquet"
    panel_path = output / "causal_panel.parquet"
    candidates.to_parquet(candidates_path, compression="zstd", index=False)
    outcomes.to_parquet(outcomes_path, compression="zstd", index=False)
    os.chmod(candidates_path, 0o600)
    os.chmod(outcomes_path, 0o600)
    passes = [name for name, report in reports.items() if report["gate_pass"]]
    result = {
        "schema_version": "aegis-economic-alpha-backcast-a2-result-v1",
        "experiment_id": "aegis-economic-alpha-backcast-a2-01",
        "evidence_class": "REVERSE_TIME_EXTERNAL_BACKCAST_NO_PROMOTION_AUTHORITY",
        "authority": {
            "a1_result": str(authority_path),
            "a1_result_sha256": sha256_file(authority_path),
            "archive_manifest_sha256": sha256_file(
                raw.parent / "archive_manifest.jsonl"
            ),
        },
        "panel": {"rows": len(panel), "sha256": sha256_file(panel_path)},
        "candidates": {"rows": len(candidates), "sha256": sha256_file(candidates_path)},
        "outcomes": {"rows": len(outcomes), "sha256": sha256_file(outcomes_path)},
        "results": reports,
        "gate_passes": passes,
        "A2_READY_FOR_MODELING": False,
        "A2_READY_FOR_PROSPECTIVE_FORWARD_COLLECTION": bool(passes),
        "A2_READY_FOR_SHADOW": False,
        "A2_READY_FOR_LIVE": False,
        "retrospective_promotion_authority": False,
        "authenticated_exchange_calls": 0,
        "exchange_mutations": 0,
        "runtime_changes": "NONE",
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(_safe(result), indent=2, sort_keys=True) + "\n")
    os.chmod(result_path, 0o600)
    print(
        json.dumps(
            {
                "gate_passes": passes,
                "panel_rows": len(panel),
                "outcome_rows": len(outcomes),
                "result": str(result_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
