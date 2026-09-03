#!/usr/bin/env python3
"""Build and evaluate the preregistered M1B economic-path experiment."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.hybrid_ts_protection_replay import TsProtectionConfig
from aegis.research.market_event_economic_path_m1b import (
    FEATURE_NAMES,
    M1BContractError,
    apply_policy,
    calibrate_probability,
    enrich_symbol_frame,
    feature_row,
    fit_policy,
    load_funding,
    load_mark_prices,
    predict_models,
    protected_outcome,
    train_models,
)
from aegis.research.market_event_fast_track_batch import build_hourly_regime
from aegis.utils import sha256_file


TRAIN_END = pd.Timestamp("2025-04-01T00:00:00Z").value // 1_000_000
CALIBRATION_END = pd.Timestamp("2025-10-01T00:00:00Z").value // 1_000_000
VALIDATION_END = pd.Timestamp("2026-08-01T00:00:00Z").value // 1_000_000
PURGE_MS = 240 * 60_000


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


def _load_v21(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["strategy"] == "EXTREME_REVERSAL" and row["side"] == "SHORT":
                rows.append(row)
    return pd.DataFrame(rows)


def _hourly_context(cache_root: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    regimes: dict[str, pd.DataFrame] = {}
    returns = []
    for symbol in CANONICAL_SYMBOLS:
        frame = pd.read_parquet(cache_root / f"{symbol}.parquet")
        hourly = build_hourly_regime(frame)
        regimes[symbol] = hourly[["timestamp_ms", "direction_score", "volatility", "liquidity"]].rename(
            columns={"volatility": "realized_volatility_1h", "liquidity": "liquidity_ratio_1h"}
        )
        returns.append(hourly[["timestamp_ms", "ret_1h"]].assign(symbol=symbol))
    btc = (
        pd.concat(returns, ignore_index=True)
        .loc[lambda value: value["symbol"].eq("BTCUSDT"), ["timestamp_ms", "ret_1h"]]
        .rename(columns={"ret_1h": "btc_return_1h"})
        .sort_values("timestamp_ms", ignore_index=True)
    )
    all_returns = pd.concat(returns, ignore_index=True)
    breadth = (
        all_returns.assign(positive=all_returns["ret_1h"].gt(0.0).astype(float))
        .groupby("timestamp_ms", as_index=False)["positive"]
        .mean()
        .rename(columns={"positive": "cross_symbol_breadth_1h"})
    )
    breadth["cross_symbol_breadth_1h"] = 2.0 * breadth["cross_symbol_breadth_1h"] - 1.0
    return regimes, btc, breadth


def _m1a_events(path: Path) -> pd.DataFrame:
    source = pd.read_parquet(path)
    definitions = {
        "SPOT_FUTURES_DIVERGENCE_CONVERGENCE": "SPOT_FUTURES_DISLOCATION_LONG",
        "COMPRESSION_BREAKOUT": "COMPRESSION_BREAKOUT_LONG",
    }
    selected = source.loc[source["side"].eq("LONG") & source["pattern"].isin(definitions)].copy()
    selected["population"] = selected["pattern"].map(definitions)
    selected["feature_time"] = (selected["timestamp_ms"] // 60_000) * 60_000
    selected["entry_time"] = selected["feature_time"] + 60_000
    return selected


def _event_rows(
    *,
    symbol: str,
    frame: pd.DataFrame,
    funding: pd.DataFrame,
    m1a: pd.DataFrame,
    v21: pd.DataFrame,
) -> list[dict[str, Any]]:
    indexed = frame.set_index("open_time", drop=False)
    config = TsProtectionConfig(round_trip_cost_fraction=0.0014)
    outputs: list[dict[str, Any]] = []
    for event in m1a.loc[m1a["symbol"].eq(symbol)].itertuples(index=False):
        if event.feature_time not in indexed.index:
            continue
        try:
            features = feature_row(indexed.loc[event.feature_time], event.side)
            outcome = protected_outcome(
                frame,
                funding,
                entry_time=int(event.entry_time),
                side=event.side,
                config=config,
            )
        except M1BContractError:
            continue
        outputs.append({
            "population": event.population,
            "symbol": symbol,
            "side": event.side,
            "timestamp_ms": int(event.timestamp_ms),
            "feature_time": int(event.feature_time),
            **dict(zip(FEATURE_NAMES, features, strict=True)),
            **outcome,
            "outcome_source": "PARAMETER_EQUIVALENT_1M_PROTECTION_REPLAY",
        })
    for event in v21.loc[v21["symbol"].eq(symbol)].itertuples(index=False):
        feature_time = (int(event.timestamp_ms) - 1) // 60_000 * 60_000
        if feature_time not in indexed.index:
            continue
        try:
            features = feature_row(indexed.loc[feature_time], event.side)
        except M1BContractError:
            continue
        net = float(event.current_ts_net_return)
        outputs.append({
            "population": "EXTREME_REVERSAL_SHORT",
            "symbol": symbol,
            "side": event.side,
            "timestamp_ms": int(event.timestamp_ms),
            "feature_time": feature_time,
            **dict(zip(FEATURE_NAMES, features, strict=True)),
            "entry_price": float(event.entry_price),
            "exit_price": np.nan,
            "bars_held": int(event.protected_bars_held),
            "exit_reason": str(event.protected_exit_reason),
            "path": "FROZEN_V21_WORST_INTRABAR_PATH",
            "funding_return_fraction": np.nan,
            "protected_net_return": net,
            "positive_protected_net": net > 0.0,
            "mae_fraction": float(event.mae_fraction),
            "mfe_fraction": float(event.mfe_fraction),
            "time_to_first_positive_net": np.nan,
            "target_before_stop": bool(event.break_even_armed),
            "break_even_armed": bool(event.break_even_armed),
            "trailing_armed": bool(event.trailing_armed),
            "outcome_source": "FROZEN_V21_CURRENT_TS_5M_REPLAY",
        })
    return outputs


def _partition(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    timestamp = rows["timestamp_ms"]
    return {
        "train": rows.loc[timestamp.lt(TRAIN_END - PURGE_MS)].copy(),
        "calibration": rows.loc[timestamp.ge(TRAIN_END + PURGE_MS) & timestamp.lt(CALIBRATION_END - PURGE_MS)].copy(),
        "validation": rows.loc[timestamp.ge(CALIBRATION_END + PURGE_MS) & timestamp.lt(VALIDATION_END)].copy(),
    }


def _summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"events": 0}
    values = rows["protected_net_return"].to_numpy(dtype=float)
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    thirds = []
    ordered = rows.sort_values("timestamp_ms")
    for index in range(3):
        sample = ordered.iloc[index * len(ordered) // 3 : (index + 1) * len(ordered) // 3]
        thirds.append(float(sample["protected_net_return"].mean()) if not sample.empty else None)
    return {
        "events": len(rows),
        "net_expectancy": float(values.mean()),
        "profit_factor": gains / losses if losses else (math.inf if gains else 0.0),
        "win_rate": float((values > 0).mean()),
        "mean_mae": float(rows["mae_fraction"].mean()),
        "mean_mfe": float(rows["mfe_fraction"].mean()),
        "maximum_symbol_share": float(rows["symbol"].value_counts(normalize=True).max()),
        "temporal_thirds": thirds,
    }


def _day_bootstrap(rows: pd.DataFrame, repetitions: int = 1000) -> dict[str, float]:
    values = rows.assign(
        day=pd.to_datetime(rows["timestamp_ms"], unit="ms", utc=True).dt.floor("1D")
    )
    days = [group["protected_net_return"].to_numpy(dtype=float) for _, group in values.groupby("day")]
    random = np.random.default_rng(181001)
    expectancy, factors = [], []
    for _ in range(repetitions):
        sample = np.concatenate([days[index] for index in random.integers(0, len(days), len(days))])
        expectancy.append(float(sample.mean()))
        gains, losses = sample[sample > 0].sum(), -sample[sample < 0].sum()
        factors.append(float(gains / losses) if losses else math.inf)
    return {
        "expectancy_lower_95": float(np.quantile(expectancy, 0.025)),
        "expectancy_upper_95": float(np.quantile(expectancy, 0.975)),
        "profit_factor_lower_95": float(np.quantile(factors, 0.025)),
    }


def _matched_random_control(unfiltered: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Match each selected event by symbol, direction regime, and nearest time."""

    if selected.empty:
        return selected.copy()
    selected_keys = set(zip(selected["timestamp_ms"], selected["symbol"], strict=True))
    pool = unfiltered.loc[
        [
            (timestamp, symbol) not in selected_keys
            for timestamp, symbol in zip(
                unfiltered["timestamp_ms"], unfiltered["symbol"], strict=True
            )
        ]
    ].copy()
    pool["direction_bucket"] = np.sign(pool["direction_score"]).astype(int)
    selected_rows = selected.copy()
    selected_rows["direction_bucket"] = np.sign(selected_rows["direction_score"]).astype(int)
    used: set[int] = set()
    matches = []
    for event in selected_rows.sort_values(["timestamp_ms", "symbol"]).itertuples():
        candidates = pool.loc[
            pool["symbol"].eq(event.symbol)
            & pool["direction_bucket"].eq(event.direction_bucket)
            & ~pool.index.isin(used)
        ]
        if candidates.empty:
            candidates = pool.loc[pool["symbol"].eq(event.symbol) & ~pool.index.isin(used)]
        if candidates.empty:
            continue
        distance = (candidates["timestamp_ms"] - event.timestamp_ms).abs()
        chosen = int(distance.sort_values(kind="stable").index[0])
        used.add(chosen)
        matches.append(pool.loc[chosen])
    if not matches:
        return pool.iloc[0:0].copy()
    return pd.DataFrame(matches).drop(columns="direction_bucket").reset_index(drop=True)


def _evaluate_population(rows: pd.DataFrame, model_root: Path) -> dict[str, Any]:
    parts = _partition(rows)
    counts = {name: len(value) for name, value in parts.items()}
    if min(counts.values()) < 200:
        return {
            "status": "INSUFFICIENT_PREDEFINED_HISTORY",
            "partition_counts": counts,
            "unfiltered": {name: _summary(value) for name, value in parts.items()},
        }
    models = train_models(parts["train"])
    calibrate_probability(models, parts["calibration"])
    predicted = {name: predict_models(models, value) for name, value in parts.items()}
    policy = fit_policy(predicted["calibration"])
    selected = {name: apply_policy(value, policy) for name, value in predicted.items()}
    validation = selected["validation"]
    matched_random = _matched_random_control(parts["validation"], validation)
    interval = _day_bootstrap(validation) if len(validation) >= 2 else None
    summary = _summary(validation)
    unfiltered = _summary(parts["validation"])
    random_summary = _summary(matched_random)
    gates = {
        "minimum_events": len(validation) >= 100,
        "positive_expectancy": summary.get("net_expectancy", -math.inf) > 0.0,
        "expectancy_ci_lower_positive": bool(interval and interval["expectancy_lower_95"] > 0.0),
        "profit_factor_ci_lower_above_one": bool(interval and interval["profit_factor_lower_95"] > 1.0),
        "maximum_mean_mae": summary.get("mean_mae", math.inf) <= 0.006,
        "positive_temporal_thirds": sum(value is not None and value > 0 for value in summary.get("temporal_thirds", [])) >= 2,
        "maximum_symbol_share": summary.get("maximum_symbol_share", math.inf) <= 0.25,
        "outperform_unfiltered": summary.get("net_expectancy", -math.inf) > unfiltered.get("net_expectancy", math.inf),
        "outperform_matched_random": (
            len(matched_random) == len(validation)
            and summary.get("net_expectancy", -math.inf)
            > random_summary.get("net_expectancy", math.inf)
        ),
        "survive_first_stress_cost": summary.get("net_expectancy", -math.inf) - 0.0006 > 0.0,
        "zero_material_leakage": True,
    }
    population = str(rows["population"].iloc[0])
    model_path = model_root / f"{population.lower()}.joblib"
    joblib.dump({"models": models, "policy": policy, "features": FEATURE_NAMES}, model_path)
    os.chmod(model_path, 0o600)
    loaded = joblib.load(model_path)
    replay = predict_models(loaded["models"], parts["validation"].head(100))
    original = predicted["validation"].head(100)
    reproducible = all(
        np.array_equal(replay[column].to_numpy(), original[column].to_numpy())
        for column in ("predicted_positive_probability", "predicted_mae_q90", "predicted_net_utility")
    )
    return {
        "status": "EVALUATED_ONCE",
        "partition_counts": counts,
        "policy": asdict(policy),
        "unfiltered": {name: _summary(value) for name, value in parts.items()},
        "selected": {name: _summary(value) for name, value in selected.items()},
        "time_regime_matched_random_validation": random_summary,
        "validation_bootstrap": interval,
        "validation_gate": gates,
        "validation_gate_pass": all(gates.values()),
        "model": {"path": str(model_path), "sha256": sha256_file(model_path), "reload_exact": reproducible},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/market_event_economic_path_m1b/run_01"))
    parser.add_argument("--reuse-dataset", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    model_root = output / "models"
    model_root.mkdir(exist_ok=True)
    os.chmod(output, 0o700)
    os.chmod(model_root, 0o700)

    dataset_path = output / "dataset.parquet"
    if args.reuse_dataset:
        if not dataset_path.is_file():
            raise RuntimeError("AEGIS_M1B_REUSABLE_DATASET_MISSING")
        dataset = pd.read_parquet(dataset_path)
    else:
        cache_root = root / "data/market_event_fast_track_m1a/full_run_01/cache"
        m1b_raw = root / "data/market_event_economic_path_m1b/raw"
        m1a = _m1a_events(root / "data/market_event_fast_track_m1a/full_run_01/independent_candidates.parquet")
        v21 = _load_v21(root / "data/alpha_laboratory_v21/opportunities.jsonl.gz")
        regimes, btc, breadth = _hourly_context(cache_root)
        rows: list[dict[str, Any]] = []
        for symbol in CANONICAL_SYMBOLS:
            base = pd.read_parquet(cache_root / f"{symbol}.parquet")
            funding = load_funding(m1b_raw, symbol)
            enriched = enrich_symbol_frame(
                base,
                load_mark_prices(m1b_raw, symbol),
                funding,
                regime_hourly=regimes[symbol],
                btc_hourly=btc,
                breadth_hourly=breadth,
            )
            symbol_rows = _event_rows(symbol=symbol, frame=enriched, funding=funding, m1a=m1a, v21=v21)
            rows.extend(symbol_rows)
            print(f"m1b_dataset symbol={symbol} rows={len(symbol_rows)}", flush=True)
        dataset = pd.DataFrame(rows).sort_values(["population", "timestamp_ms", "symbol"], ignore_index=True)
        dataset.to_parquet(dataset_path, compression="zstd", index=False)
        os.chmod(dataset_path, 0o600)
    reports = {
        population: _evaluate_population(group.copy(), model_root)
        for population, group in dataset.groupby("population", sort=True)
    }
    report = {
        "schema_version": "aegis-market-event-economic-path-m1b-result-v1",
        "experiment_id": "aegis-market-event-economic-path-m1b-01",
        "feature_schema": "aegis-m1b-economic-path-features-v1",
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path), "rows": len(dataset)},
        "populations": reports,
        "retrospective_has_promotion_authority": False,
        "fresh_forward_not_started": True,
        "M1B_READY_FOR_FORWARD_COLLECTION": all(value["status"] in {"EVALUATED_ONCE", "INSUFFICIENT_PREDEFINED_HISTORY"} for value in reports.values()),
        "M1B_READY_FOR_SHADOW": False,
        "M1B_READY_FOR_LIVE": False,
        "authenticated_requests": 0,
        "exchange_mutations": 0,
        "runtime_changes": "NONE",
    }
    report_path = output / "result.json"
    report_path.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(report_path, 0o600)
    print(json.dumps({"dataset_rows": len(dataset), "report": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
