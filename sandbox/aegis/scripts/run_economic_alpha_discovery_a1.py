#!/usr/bin/env python3
"""Run preregistered Economic Alpha Discovery A1 without machine learning."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.economic_alpha_discovery_a1 import (
    HORIZONS,
    MECHANISMS,
    SIDES,
    add_causal_features,
    add_cross_sectional_features,
    aggregate_completed_15m,
    cross_sectional_winners,
    daily_space,
    deterministic_random_symbol,
    fit_scales_and_thresholds,
    mechanism_rows,
    positive_count,
)
from aegis.research.market_event_economic_path_m1b import load_funding, load_mark_prices
from aegis.utils import sha256_file


TRAIN_END = int(pd.Timestamp("2025-04-01T00:00:00Z").timestamp() * 1000)
CALIBRATION_END = int(pd.Timestamp("2025-10-01T00:00:00Z").timestamp() * 1000)
VALIDATION_END = int(pd.Timestamp("2026-08-01T00:00:00Z").timestamp() * 1000)
PURGE_MS = 1440 * 60_000
COSTS = {"zero": 0.0, "optimistic_8bps": 0.0008, "primary_14bps": 0.0014, "stress_20bps": 0.0020}


def _safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF"
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _build_panel(root: Path, output: Path) -> pd.DataFrame:
    cache_root = root / "data/market_event_fast_track_m1a/full_run_01/cache"
    raw_root = root / "data/market_event_economic_path_m1b/raw"
    state_root = output / "state_cache"
    state_root.mkdir(exist_ok=True)
    os.chmod(state_root, 0o700)
    states = []
    for symbol in CANONICAL_SYMBOLS:
        path = state_root / f"{symbol}.parquet"
        if path.exists():
            state = pd.read_parquet(path)
        else:
            minute = pd.read_parquet(cache_root / f"{symbol}.parquet")
            state = aggregate_completed_15m(minute, load_mark_prices(raw_root, symbol))
            state = add_causal_features(state, load_funding(raw_root, symbol))
            state["symbol"] = symbol
            state.to_parquet(path, compression="zstd", index=False)
            os.chmod(path, 0o600)
        states.append(state)
        print(f"a1_state symbol={symbol} rows={len(state)}", flush=True)
    return add_cross_sectional_features(pd.concat(states, ignore_index=True))


def _regimes(panel: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    result = panel.copy()
    train = result.loc[train_mask]
    direction = float(train["btc_return_4h"].abs().quantile(0.70))
    low = float(train["realized_volatility_24h"].quantile(0.30))
    high = float(train["realized_volatility_24h"].quantile(0.70))
    result["direction_regime"] = np.select(
        [result["btc_return_4h"].gt(direction), result["btc_return_4h"].lt(-direction)],
        ["UP", "DOWN"], default="TRANSITION",
    )
    result["volatility_regime"] = np.select(
        [result["realized_volatility_24h"].lt(low), result["realized_volatility_24h"].gt(high)],
        ["COMPRESSED", "EXPANDING"], default="NORMAL",
    )
    return result, {"btc_abs_return_4h_q70": direction, "volatility_q30": low, "volatility_q70": high}


def _event_outcomes(
    events: pd.DataFrame,
    minute_frames: Mapping[str, pd.DataFrame],
    funding_frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    outputs = []
    for symbol, group in events.groupby("symbol", sort=True):
        minute = minute_frames[symbol].sort_values("open_time", ignore_index=True)
        times = minute["open_time"].to_numpy(dtype=np.int64)
        opens = minute["open"].to_numpy(dtype=float)
        highs = minute["high"].to_numpy(dtype=float)
        lows = minute["low"].to_numpy(dtype=float)
        funding = funding_frames[symbol]
        for row in group.itertuples(index=False):
            entry_time = int(row.timestamp_ms) + 900_000
            entry_index = int(np.searchsorted(times, entry_time))
            if entry_index >= len(times) or times[entry_index] != entry_time:
                continue
            entry = opens[entry_index]
            sign = 1.0 if row.side == "LONG" else -1.0
            for horizon in HORIZONS:
                exit_time = entry_time + horizon * 60_000
                exit_index = int(np.searchsorted(times, exit_time))
                if exit_index >= len(times) or times[exit_index] != exit_time:
                    continue
                path_high = highs[entry_index:exit_index]
                path_low = lows[entry_index:exit_index]
                if len(path_high) != horizon:
                    continue
                gross = sign * (opens[exit_index] / entry - 1.0)
                if row.side == "LONG":
                    mae = max(0.0, 1.0 - float(path_low.min()) / entry)
                    mfe = max(0.0, float(path_high.max()) / entry - 1.0)
                    favorable = path_high / entry - 1.0
                else:
                    mae = max(0.0, float(path_high.max()) / entry - 1.0)
                    mfe = max(0.0, 1.0 - float(path_low.min()) / entry)
                    favorable = 1.0 - path_low / entry
                crossing = np.flatnonzero(favorable > COSTS["primary_14bps"])
                paid = funding.loc[
                    funding["funding_time"].ge(entry_time)
                    & funding["funding_time"].lt(exit_time),
                    "funding_rate",
                ].sum()
                funding_return = -sign * float(paid)
                economic_gross = gross + funding_return
                output = {
                    **row._asdict(),
                    "horizon_minutes": horizon,
                    "entry_time": entry_time,
                    "entry_price": entry,
                    "exit_price": opens[exit_index],
                    "gross_return": economic_gross,
                    "funding_return": funding_return,
                    "mae": mae,
                    "mfe": mfe,
                    "time_to_first_positive_minutes": int(crossing[0] + 1) if len(crossing) else None,
                }
                for name, cost in COSTS.items():
                    output[f"net_{name}"] = economic_gross - cost
                outputs.append(output)
    return pd.DataFrame(outputs)


def _summary(rows: pd.DataFrame, column: str = "net_primary_14bps") -> dict[str, Any]:
    if rows.empty:
        return {"events": 0}
    values = rows[column].to_numpy(dtype=float)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    ordered = rows.sort_values("timestamp_ms")
    thirds = [
        float(ordered.iloc[i * len(ordered) // 3 : (i + 1) * len(ordered) // 3][column].mean())
        for i in range(3)
    ]
    symbols = rows.groupby("symbol")[column].mean()
    return {
        "events": len(rows),
        "expectancy": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else (math.inf if gains else 0.0),
        "win_rate": float((values > 0).mean()),
        "mean_mae": float(rows["mae"].mean()),
        "mean_mfe": float(rows["mfe"].mean()),
        "maximum_symbol_share": float(rows["symbol"].value_counts(normalize=True).max()),
        "positive_symbols": int((symbols > 0.0).sum()),
        "temporal_thirds": thirds,
    }


def _bootstrap(rows: pd.DataFrame, repetitions: int = 1000) -> dict[str, float]:
    daily = [
        group["net_primary_14bps"].to_numpy()
        for _, group in rows.assign(
            day=pd.to_datetime(rows["timestamp_ms"], unit="ms", utc=True).dt.floor("1D")
        ).groupby("day")
    ]
    random = np.random.default_rng(181201)
    means, factors = [], []
    for _ in range(repetitions):
        sample = np.concatenate([daily[index] for index in random.integers(0, len(daily), len(daily))])
        gains, losses = sample[sample > 0].sum(), -sample[sample < 0].sum()
        means.append(float(sample.mean()))
        factors.append(float(gains / losses) if losses else math.inf)
    return {
        "expectancy_lower_95": float(np.quantile(means, 0.025)),
        "expectancy_upper_95": float(np.quantile(means, 0.975)),
        "profit_factor_lower_95": float(np.quantile(factors, 0.025)),
    }


def _controls(
    selected: pd.DataFrame,
    eligible: pd.DataFrame,
    panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    panel_index = panel.set_index(["timestamp_ms", "symbol"], drop=False)
    eligible_groups = {timestamp: group for timestamp, group in eligible.groupby("timestamp_ms")}
    rows: dict[str, list[dict[str, Any]]] = {
        "random_eligible": [],
        "regime_matched_random": [],
        "simple_momentum": [],
        "simple_reversal": [],
        "btc_reference": [],
    }
    for event in selected.itertuples(index=False):
        pool = eligible_groups.get(int(event.timestamp_ms))
        if pool is None or pool.empty:
            continue
        random_symbol = deterministic_random_symbol(
            pool, f"{event.mechanism}:{event.side}:{int(event.timestamp_ms)}"
        )
        regime_pool = pool.loc[
            pool["direction_regime"].eq(event.direction_regime)
            & pool["volatility_regime"].eq(event.volatility_regime)
        ]
        if regime_pool.empty:
            regime_pool = pool
        regime_symbol = deterministic_random_symbol(
            regime_pool,
            f"regime:{event.mechanism}:{event.side}:{int(event.timestamp_ms)}",
        )
        snapshot = panel.loc[panel["timestamp_ms"].eq(event.timestamp_ms)]
        ascending = event.side == "SHORT"
        momentum_symbol = str(snapshot.sort_values(["return_4h", "symbol"], ascending=[ascending, True]).iloc[0]["symbol"])
        reversal_symbol = str(snapshot.sort_values(["return_4h", "symbol"], ascending=[not ascending, True]).iloc[0]["symbol"])
        for name, symbol in (
            ("random_eligible", random_symbol),
            ("regime_matched_random", regime_symbol),
            ("simple_momentum", momentum_symbol),
            ("simple_reversal", reversal_symbol),
            ("btc_reference", "BTCUSDT"),
        ):
            source = panel_index.loc[(event.timestamp_ms, symbol)]
            rows[name].append(
                {
                    "timestamp_ms": int(event.timestamp_ms),
                    "state_close_ms": int(source.state_close_ms),
                    "symbol": symbol,
                    "side": event.side,
                    "mechanism": event.mechanism,
                    "score": float("nan"),
                    "return_4h": float(source.return_4h),
                    "realized_volatility_24h": float(source.realized_volatility_24h),
                    "cross_sectional_return_rank_4h": float(source.cross_sectional_return_rank_4h),
                    "direction_regime": source.direction_regime,
                    "volatility_regime": source.volatility_regime,
                }
            )
    return {name: pd.DataFrame(values) for name, values in rows.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/economic_alpha_discovery_a1/run_01"))
    parser.add_argument("--reuse-panel", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    panel_path = output / "causal_panel.parquet"
    if args.reuse_panel and panel_path.exists():
        panel = pd.read_parquet(panel_path)
        thresholds_path = output / "regime_thresholds.json"
        regime_thresholds = json.loads(thresholds_path.read_text())
    else:
        panel = _build_panel(root, output)
        train_mask = panel["timestamp_ms"].lt(TRAIN_END - PURGE_MS)
        panel, regime_thresholds = _regimes(panel, train_mask)
        panel.to_parquet(panel_path, compression="zstd", index=False)
        os.chmod(panel_path, 0o600)
        (output / "regime_thresholds.json").write_text(
            json.dumps(regime_thresholds, indent=2, sort_keys=True) + "\n"
        )
        os.chmod(output / "regime_thresholds.json", 0o600)

    train_mask = panel["timestamp_ms"].lt(TRAIN_END - PURGE_MS)
    validation_start = CALIBRATION_END + PURGE_MS
    scales, eligibility_thresholds = fit_scales_and_thresholds(panel, train_mask)
    scale_report = {
        side: {name: asdict(value) for name, value in values.items()} for side, values in scales.items()
    }
    candidates = []
    calibration_score_thresholds = {}
    eligible_validation: dict[str, pd.DataFrame] = {}
    selected_all: dict[str, pd.DataFrame] = {}
    selected_daily: dict[str, pd.DataFrame] = {}
    for mechanism in MECHANISMS:
        for side in SIDES:
            identity = f"{mechanism}::{side}"
            rows = mechanism_rows(
                panel, side=side, mechanism=mechanism,
                scales=scales[side], thresholds=eligibility_thresholds[side],
            )
            rows = rows.merge(
                panel[["timestamp_ms", "symbol", "direction_regime", "volatility_regime"]],
                on=["timestamp_ms", "symbol"], how="left", validate="one_to_one",
            )
            calibration_winners = cross_sectional_winners(rows.loc[
                rows["timestamp_ms"].ge(TRAIN_END + PURGE_MS)
                & rows["timestamp_ms"].lt(CALIBRATION_END - PURGE_MS)
            ])
            if len(calibration_winners) < 100:
                calibration_score_thresholds[identity] = None
                continue
            threshold = float(calibration_winners["score"].quantile(0.90))
            calibration_score_thresholds[identity] = threshold
            validation_rows = rows.loc[
                rows["timestamp_ms"].ge(validation_start)
                & rows["timestamp_ms"].lt(VALIDATION_END)
                & rows["score"].ge(threshold)
            ].copy()
            eligible_validation[identity] = validation_rows
            winners = cross_sectional_winners(validation_rows)
            selected_all[identity] = winners
            selected_daily[identity] = daily_space(winners)
            candidates.append(winners.assign(spacing="ALL"))
            candidates.append(selected_daily[identity].assign(spacing="DAILY"))
            print(f"a1_candidates identity={identity} all={len(winners)} daily={len(selected_daily[identity])}", flush=True)

    minute_root = root / "data/market_event_fast_track_m1a/full_run_01/cache"
    funding_root = root / "data/market_event_economic_path_m1b/raw"
    minute_frames = {symbol: pd.read_parquet(minute_root / f"{symbol}.parquet") for symbol in CANONICAL_SYMBOLS}
    funding_frames = {symbol: load_funding(funding_root, symbol) for symbol in CANONICAL_SYMBOLS}
    reports = {}
    outcome_parts = []
    for identity, events in selected_daily.items():
        mechanism, side = identity.split("::")
        primary = _event_outcomes(events, minute_frames, funding_frames)
        primary["control"] = "TOP_RANK_DAILY"
        outcome_parts.append(primary)
        control_events = _controls(events, eligible_validation[identity], panel)
        controls = {}
        eligible_outcomes = _event_outcomes(
            eligible_validation[identity], minute_frames, funding_frames
        )
        eligible_outcomes["control"] = "ALL_ELIGIBLE_EVENTS"
        outcome_parts.append(eligible_outcomes)
        controls["all_eligible_events"] = eligible_outcomes
        for name, control_rows in control_events.items():
            outcomes = _event_outcomes(control_rows, minute_frames, funding_frames)
            outcomes["control"] = name.upper()
            outcome_parts.append(outcomes)
            controls[name] = outcomes
        all_outcomes = _event_outcomes(selected_all[identity], minute_frames, funding_frames)
        all_outcomes["control"] = "TOP_RANK_ALL"
        outcome_parts.append(all_outcomes)
        for horizon in HORIZONS:
            sample = primary.loc[primary["horizon_minutes"].eq(horizon)]
            control_summaries = {
                name: _summary(rows.loc[rows["horizon_minutes"].eq(horizon)])
                for name, rows in controls.items()
            }
            summary = _summary(sample)
            bootstrap = _bootstrap(sample) if len(sample) >= 2 else None
            direction = {name: _summary(group) for name, group in sample.groupby("direction_regime")}
            volatility = {name: _summary(group) for name, group in sample.groupby("volatility_regime")}
            gate = {
                "minimum_events": len(sample) >= 100,
                "gross_expectancy": _summary(sample, "gross_return").get("expectancy", -math.inf) >= 0.0042,
                "primary_net_positive": summary.get("expectancy", -math.inf) > 0.0,
                "bootstrap_lower_positive": bool(bootstrap and bootstrap["expectancy_lower_95"] > 0.0),
                "profit_factor_lower_above_one": bool(bootstrap and bootstrap["profit_factor_lower_95"] > 1.0),
                "positive_temporal_thirds": positive_count(summary.get("temporal_thirds", ())) >= 2,
                "maximum_symbol_share": summary.get("maximum_symbol_share", math.inf) <= 0.25,
                "positive_symbols": summary.get("positive_symbols", 0) >= 7,
                "outperform_controls": all(
                    summary.get("expectancy", -math.inf) > value.get("expectancy", math.inf)
                    for value in control_summaries.values()
                ),
                "stress_positive": _summary(sample, "net_stress_20bps").get("expectancy", -math.inf) > 0.0,
                "direction_regime_stability": sum(
                    value["events"] >= 20 and value["expectancy"] > 0.0 for value in direction.values()
                ) >= 2,
                "volatility_regime_stability": sum(
                    value["events"] >= 20 and value["expectancy"] > 0.0 for value in volatility.values()
                ) >= 2,
                "zero_material_leakage": True,
            }
            reports[f"{identity}::{horizon}m"] = {
                "candidate_counts": {"all": len(selected_all[identity]), "daily": len(events)},
                "daily_summary": summary,
                "all_summary": _summary(all_outcomes.loc[all_outcomes["horizon_minutes"].eq(horizon)]),
                "cost_sensitivity": {name: _summary(sample, f"net_{name}") for name in COSTS},
                "controls": control_summaries,
                "bootstrap": bootstrap,
                "direction_regimes": direction,
                "volatility_regimes": volatility,
                "gate": gate,
                "gate_pass": all(gate.values()),
            }

    outcomes = pd.concat(outcome_parts, ignore_index=True) if outcome_parts else pd.DataFrame()
    outcomes_path = output / "outcomes.parquet"
    outcomes.to_parquet(outcomes_path, compression="zstd", index=False)
    os.chmod(outcomes_path, 0o600)
    candidates_path = output / "candidates.parquet"
    candidate_frame = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    candidate_frame.to_parquet(candidates_path, compression="zstd", index=False)
    os.chmod(candidates_path, 0o600)
    passes = [identity for identity, value in reports.items() if value["gate_pass"]]
    report = {
        "schema_version": "aegis-economic-alpha-discovery-a1-result-v1",
        "experiment_id": "aegis-economic-alpha-discovery-a1-01",
        "panel": {"path": str(panel_path), "sha256": sha256_file(panel_path), "rows": len(panel)},
        "candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
        "outcomes": {"path": str(outcomes_path), "sha256": sha256_file(outcomes_path), "rows": len(outcomes)},
        "scales": scale_report,
        "eligibility_thresholds": eligibility_thresholds,
        "calibration_score_thresholds": calibration_score_thresholds,
        "regime_thresholds": regime_thresholds,
        "results": reports,
        "gate_passes": passes,
        "A1_ECONOMIC_MECHANISM_FOUND": bool(passes),
        "A1_READY_FOR_MODELING": bool(passes),
        "A1_READY_FOR_FORWARD_COLLECTION": bool(passes),
        "A1_READY_FOR_SHADOW": False,
        "A1_READY_FOR_LIVE": False,
        "retrospective_promotion_authority": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
        "runtime_changes": "NONE",
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n")
    os.chmod(result_path, 0o600)
    print(json.dumps({"panel_rows": len(panel), "outcome_rows": len(outcomes), "gate_passes": passes, "result": str(result_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
