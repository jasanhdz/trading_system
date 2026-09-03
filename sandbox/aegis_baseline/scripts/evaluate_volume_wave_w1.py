#!/usr/bin/env python3
"""Evaluate W1 on TRAIN/VALIDATION while keeping final holdout sealed."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.volume_wave_w1 import (
    ENTRY_VARIANTS,
    LADDERS,
    benjamini_hochberg,
    clustered_economic_metrics,
    path_outcomes,
    registered_contracts,
)
from aegis.utils import sha256_file


def _milliseconds(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1_000)


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _best_contract(
    events: pd.DataFrame,
    contracts: tuple[tuple[str, int, float, float], ...],
    *,
    cost_bps: float,
    minimum_events: int,
) -> tuple[str, int, float, float] | None:
    if len(events) < minimum_events:
        return None
    ranked = []
    for identity, horizon, favorable, adverse in contracts:
        outcomes = path_outcomes(
            events, horizon_bars=horizon, favorable_atr=favorable,
            adverse_atr=adverse, cost_bps=cost_bps,
        )
        ranked.append((float(outcomes["net_utility"].mean()), identity, horizon, favorable, adverse))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    _, identity, horizon, favorable, adverse = ranked[0]
    return identity, horizon, favorable, adverse


def _matched_control(controls: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected = selected.assign(
        month=pd.to_datetime(selected["event_timestamp_ms"], unit="ms", utc=True).dt.strftime("%Y-%m")
    )
    controls = controls.assign(
        month=pd.to_datetime(controls["event_timestamp_ms"], unit="ms", utc=True).dt.strftime("%Y-%m")
    )
    samples = []
    for (symbol, month), rows in selected.groupby(["symbol", "month"], sort=True):
        candidates = controls.loc[
            controls["symbol"].eq(symbol) & controls["month"].eq(month)
        ].copy()
        if len(candidates) < len(rows):
            raise RuntimeError("AEGIS_W1_VALIDATION_CONTROL_INSUFFICIENT")
        identity = (
            candidates["event_timestamp_ms"].astype(str) + ":"
            + candidates["symbol"] + ":" + candidates["side"] + ":"
            + candidates["entry_variant"]
        )
        candidates["order"] = pd.util.hash_pandas_object(
            identity, index=False, hash_key="1814021814021814"
        ).to_numpy(dtype=np.uint64)
        samples.append(candidates.sort_values(["order", "event_timestamp_ms"]).head(len(rows)))
    return pd.concat(samples, ignore_index=True).drop(columns=["order", "month"])


def _walk_forward(
    events: pd.DataFrame,
    contracts: tuple[tuple[str, int, float, float], ...],
    *,
    cost_bps: float,
    minimum_events: int,
) -> dict[str, object]:
    folds = []
    for test_month in ("2025-11", "2025-12", "2026-01", "2026-02"):
        test_start = pd.Timestamp(f"{test_month}-01T00:00:00Z")
        test_end = test_start + pd.offsets.MonthBegin(1)
        train = events.loc[
            events["event_timestamp_ms"].ge(_milliseconds("2025-08-01T00:00:00Z"))
            & events["event_timestamp_ms"].lt(int(test_start.timestamp() * 1_000))
        ]
        test = events.loc[
            events["event_timestamp_ms"].ge(int(test_start.timestamp() * 1_000))
            & events["event_timestamp_ms"].lt(int(test_end.timestamp() * 1_000))
        ]
        selected = _best_contract(
            train, contracts, cost_bps=cost_bps, minimum_events=minimum_events
        )
        if selected is None or test.empty:
            folds.append({"test_month": test_month, "state": "INSUFFICIENT"})
            continue
        identity, horizon, favorable, adverse = selected
        outcomes = path_outcomes(
            test, horizon_bars=horizon, favorable_atr=favorable,
            adverse_atr=adverse, cost_bps=cost_bps,
        )
        folds.append({
            "test_month": test_month, "state": "EVALUATED",
            "contract": identity, "events": len(test),
            "net_expectancy": float(outcomes["net_utility"].mean()),
        })
    evaluated = [fold for fold in folds if fold["state"] == "EVALUATED"]
    return {
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "positive_folds": sum(float(fold["net_expectancy"]) > 0.0 for fold in evaluated),
    }


def _bins(values: list[Any]) -> list[float]:
    result = []
    for value in values:
        if value == "INF":
            result.append(math.inf)
        elif value == "-INF":
            result.append(-math.inf)
        else:
            result.append(float(value))
    return result


def _descriptive_bins(validation: pd.DataFrame, config: dict[str, Any]) -> dict[str, object]:
    immediate = validation.loc[
        validation["sample_source"].eq("WAVE_CANDIDATE")
        & validation["entry_variant"].eq("A_IMMEDIATE")
    ].copy()
    outcomes = path_outcomes(
        immediate, horizon_bars=3, favorable_atr=0.50, adverse_atr=0.25,
        cost_bps=float(config["economics"]["base_round_trip_cost_bps"]),
    )
    immediate = immediate.join(outcomes[["net_utility", "mfe_atr", "mae_atr"]])
    definitions = {
        "volume_ratio_20": _bins(config["features"]["volume_ratio_bins"]),
        "volume_z_20": _bins(config["features"]["volume_z_bins"]),
        "body_ratio": _bins(config["features"]["body_ratio_bins"]),
        "side_taker_imbalance": _bins(config["features"]["side_taker_imbalance_bins"]),
        "side_rsi_space": [0.0, 15.0, 25.0, 40.0, 60.0, 75.0, 85.0, 100.0001],
        "side_extension_ma25_atr": [-math.inf, 0.0, 1.0, 2.0, 3.0, math.inf],
    }
    result = {}
    for side in ("LONG", "SHORT"):
        side_rows = immediate.loc[immediate["side"].eq(side)]
        side_result = {}
        for feature, boundaries in definitions.items():
            categories = pd.cut(side_rows[feature], boundaries, right=False, duplicates="drop")
            table = side_rows.assign(bin=categories).groupby("bin", observed=True).agg(
                events=("net_utility", "size"),
                net_expectancy=("net_utility", "mean"),
                mean_mfe_atr=("mfe_atr", "mean"),
                mean_mae_atr=("mae_atr", "mean"),
            )
            side_result[feature] = [
                {"bin": str(index), **{key: float(value) if key != "events" else int(value) for key, value in row.items()}}
                for index, row in table.iterrows()
            ]
        result[side] = side_result
    return result


def _exhaustion_profile(validation: pd.DataFrame) -> dict[str, object]:
    rows = validation.loc[
        validation["sample_source"].eq("WAVE_CANDIDATE")
        & validation["entry_variant"].eq("A_IMMEDIATE")
    ]
    result = {}
    for side in ("LONG", "SHORT"):
        selected = rows.loc[rows["side"].eq(side)]
        sign = 1.0 if side == "LONG" else -1.0
        entry = selected["entry_price"].to_numpy(dtype=np.float64)
        profile = []
        peak = np.zeros(len(selected), dtype=np.float64)
        for offset in range(1, 7):
            high = selected[f"future_high_{offset}"].to_numpy(dtype=np.float64)
            low = selected[f"future_low_{offset}"].to_numpy(dtype=np.float64)
            close = selected[f"future_close_{offset}"].to_numpy(dtype=np.float64)
            favorable = np.where(sign > 0.0, high - entry, entry - low) / entry
            peak = np.maximum(peak, favorable)
            current = sign * (close - entry) / entry
            giveback = np.divide(
                peak - current, peak, out=np.zeros_like(peak), where=peak > 0.0
            )
            meaningful_peak = (
                peak
                >= 0.05
                * selected["entry_atr"].to_numpy(dtype=np.float64)
                / entry
            )
            meaningful_giveback = giveback[meaningful_peak]
            profile.append({
                "bar": offset,
                "positive_close_rate": float((current > 0.0).mean()),
                "mean_peak_mfe_fraction": float(peak.mean()),
                "meaningful_peak_events": int(meaningful_peak.sum()),
                "median_giveback_ratio": float(np.median(meaningful_giveback))
                if len(meaningful_giveback) else 0.0,
                "giveback_over_50pct_rate": float(
                    (meaningful_giveback > 0.50).mean()
                ) if len(meaningful_giveback) else 0.0,
                "giveback_over_100pct_rate": float(
                    (meaningful_giveback > 1.0).mean()
                ) if len(meaningful_giveback) else 0.0,
                "mean_side_taker_imbalance": float(
                    (sign * selected[f"future_taker_imbalance_{offset}"]).mean()
                ),
                "mean_side_velocity_atr": float(
                    (sign * selected[f"future_velocity_atr_1_{offset}"]).mean()
                ),
            })
        result[side] = profile
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("data/volume_wave_w1/dataset_2025_08_2026_07"),
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("config/experiments/aegis_volume_wave_w1.yaml"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/volume_wave_w1/evaluation_train_validation_01.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else root / args.dataset_root
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = yaml.safe_load(config_path.read_text())
    manifest = json.loads((dataset_root / "manifest.json").read_text())
    if manifest.get("final_holdout_state") != "SEALED":
        raise RuntimeError("AEGIS_W1_HOLDOUT_AUTHORITY_INVALID")
    first_schema = pq.read_schema(dataset_root / f"{CANONICAL_SYMBOLS[0]}.parquet")
    available = set(first_schema.names)
    columns = {
        "sample_source", "symbol", "side", "entry_variant", "event_timestamp_ms",
        "entry_price", "entry_atr", "volume_ratio_20", "volume_z_20", "body_ratio",
        "side_taker_imbalance", "side_rsi_space", "side_extension_ma25_atr",
        *(f"ladder_{ladder}" for ladder in LADDERS),
        *(f"future_{field}_{offset}" for field in (
            "high", "low", "close", "taker_imbalance", "velocity_atr_1"
        ) for offset in range(1, 7)),
        *(
            f"future_1m_{field}_{offset}"
            for field in ("high", "low", "close")
            for offset in range(1, 31)
        ),
    }
    if not columns.issubset(available):
        raise RuntimeError("AEGIS_W1_EVALUATION_SCHEMA_MISMATCH")
    validation_end = _milliseconds(config["partitions"]["validation"][1])
    frames = [
        pd.read_parquet(
            dataset_root / f"{symbol}.parquet", columns=sorted(columns),
            filters=[("event_timestamp_ms", "<", validation_end)],
        )
        for symbol in CANONICAL_SYMBOLS
    ]
    population = pd.concat(frames, ignore_index=True)
    train_end = _milliseconds(config["partitions"]["train"][1])
    train = population.loc[population["event_timestamp_ms"].lt(train_end)]
    validation = population.loc[population["event_timestamp_ms"].ge(train_end)]
    wave_train = train.loc[train["sample_source"].eq("WAVE_CANDIDATE")]
    wave_validation = validation.loc[validation["sample_source"].eq("WAVE_CANDIDATE")]
    controls_validation = validation.loc[
        validation["sample_source"].eq("MATCHED_PRICE_ONLY_CONTROL")
    ]
    contracts = registered_contracts(config)
    cost = float(config["economics"]["base_round_trip_cost_bps"])
    minimum_train = int(config["contract_selection"]["minimum_train_events"])
    repetitions = int(config["statistics"]["bootstrap_repetitions"])
    results = {}
    pvalues = {}
    for ladder in LADDERS:
        for side in ("LONG", "SHORT"):
            for variant in ENTRY_VARIANTS:
                identity = f"{ladder}:{side}:{variant}"
                train_rows = wave_train.loc[
                    wave_train["side"].eq(side)
                    & wave_train["entry_variant"].eq(variant)
                    & wave_train[f"ladder_{ladder}"]
                ]
                selected_contract = _best_contract(
                    train_rows, contracts, cost_bps=cost, minimum_events=minimum_train
                )
                if selected_contract is None:
                    results[identity] = {"state": "TRAIN_EVENTS_LT_500", "train_events": len(train_rows)}
                    continue
                contract_id, horizon, favorable, adverse = selected_contract
                validation_rows = wave_validation.loc[
                    wave_validation["side"].eq(side)
                    & wave_validation["entry_variant"].eq(variant)
                    & wave_validation[f"ladder_{ladder}"]
                ]
                if validation_rows.empty:
                    results[identity] = {"state": "VALIDATION_EMPTY", "contract": contract_id}
                    continue
                outcomes = path_outcomes(
                    validation_rows, horizon_bars=horizon, favorable_atr=favorable,
                    adverse_atr=adverse, cost_bps=cost,
                )
                metrics = clustered_economic_metrics(
                    validation_rows, outcomes, repetitions=repetitions,
                    seed=181401 + len(results),
                )
                control_pool = controls_validation.loc[
                    controls_validation["side"].eq(side)
                    & controls_validation["entry_variant"].eq(variant)
                ]
                control_rows = _matched_control(control_pool, validation_rows)
                control_outcomes = path_outcomes(
                    control_rows, horizon_bars=horizon, favorable_atr=favorable,
                    adverse_atr=adverse, cost_bps=cost,
                )
                control_metrics = clustered_economic_metrics(
                    control_rows, control_outcomes, repetitions=repetitions,
                    seed=181901 + len(results),
                )
                stress = {}
                for stress_cost in config["economics"]["stress_round_trip_cost_bps"]:
                    stress_outcomes = path_outcomes(
                        validation_rows, horizon_bars=horizon, favorable_atr=favorable,
                        adverse_atr=adverse, cost_bps=float(stress_cost),
                    )
                    stress[str(stress_cost)] = float(stress_outcomes["net_utility"].mean())
                walk_forward_rows = population.loc[
                    population["sample_source"].eq("WAVE_CANDIDATE")
                    & population["side"].eq(side)
                    & population["entry_variant"].eq(variant)
                    & population[f"ladder_{ladder}"]
                ]
                walk_forward = _walk_forward(
                    walk_forward_rows, contracts, cost_bps=cost,
                    minimum_events=minimum_train,
                )
                result = {
                    "state": "EVALUATED", "contract": contract_id,
                    "horizon_bars": horizon, "favorable_atr": favorable,
                    "adverse_atr": adverse, "train_events": len(train_rows),
                    "validation": metrics, "matched_price_only_control": control_metrics,
                    "stress_expectancy": stress, "walk_forward": walk_forward,
                }
                results[identity] = result
                pvalues[identity] = float(metrics["bootstrap_probability_expectancy_le_zero"])
                print(json.dumps({"hypothesis": identity, "state": "EVALUATED"}), flush=True)
    fdr = benjamini_hochberg(pvalues) if pvalues else {}
    passes = []
    for identity, accepted in fdr.items():
        result = results[identity]
        metrics = result["validation"]
        control = result["matched_price_only_control"]
        blockers = []
        if not accepted:
            blockers.append("FDR_NOT_SIGNIFICANT")
        if metrics["events"] < int(config["evidence_gate"]["minimum_validation_events_per_side"]):
            blockers.append("VALIDATION_EVENTS_LT_1000")
        if metrics["expectancy_ci_95"][0] <= 0.0:
            blockers.append("EXPECTANCY_CI_LOWER_NOT_POSITIVE")
        if metrics["profit_factor_ci_95"][0] <= 1.0:
            blockers.append("PROFIT_FACTOR_CI_LOWER_NOT_ABOVE_ONE")
        if result["stress_expectancy"]["20"] <= 0.0:
            blockers.append("FAILS_20BPS_COST")
        if metrics["positive_symbols"] < int(config["evidence_gate"]["positive_symbols"]):
            blockers.append("POSITIVE_SYMBOLS_LT_7")
        if result["walk_forward"]["positive_folds"] < int(config["evidence_gate"]["positive_walk_forward_folds"]):
            blockers.append("POSITIVE_WALK_FORWARD_FOLDS_LT_3")
        if metrics["net_expectancy"] <= control["net_expectancy"]:
            blockers.append("DOES_NOT_BEAT_MATCHED_PRICE_ONLY")
        if metrics["maximum_symbol_share"] > float(config["evidence_gate"]["maximum_symbol_share"]):
            blockers.append("SYMBOL_SHARE_GT_20_PERCENT")
        result["fdr_significant"] = accepted
        result["gate_blockers"] = blockers
        result["gate_pass"] = not blockers
        if not blockers:
            passes.append(identity)
    report = {
        "schema_version": "aegis-volume-wave-w1-evaluation-v1",
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
        "final_holdout_state": "SEALED",
        "train_rows": len(train), "validation_rows": len(validation),
        "results": results,
        "fdr": fdr,
        "passing_hypotheses": passes,
        "descriptive_bins_fixed_h3_f050_a025": _descriptive_bins(validation, config),
        "exhaustion_profile": _exhaustion_profile(validation),
        "W1_RULE_EDGE_FOUND": bool(passes),
        "W1_MODELING_JUSTIFIED": bool(passes),
        "W1_READY_FOR_SHADOW": False,
        "W1_READY_FOR_LIVE": False,
        "authenticated_requests": 0,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n")
    os.chmod(output, 0o600)
    print(json.dumps({
        "output": str(output), "passing_hypotheses": passes,
        "holdout": "SEALED",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
