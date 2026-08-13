#!/usr/bin/env python3
"""Run the preregistered residual-regime alpha B2 diagnostic."""

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
from aegis.research.opportunity_atlas_b1 import economic_summary, partition_mask
from aegis.research.residual_regime_alpha_b2 import (
    PRIMARY_COST,
    RANK_FEATURES,
    STRESS_COST,
    add_barrier_outcomes,
    add_causal_residuals,
    assign_regimes,
    choose_regime_mechanisms,
    contract_hash,
    fit_pairwise_rankers,
    fit_path_models,
    fit_regime_thresholds,
    grouped_rank_metrics,
    select_combined,
    select_mechanism_rows,
)
from aegis.utils import sha256_file


PARTITIONS = {
    "TRAIN": ("2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    "CALIBRATION": ("2025-01-02T00:00:00Z", "2025-07-01T00:00:00Z"),
    "VALIDATION": ("2025-07-02T00:00:00Z", "2026-01-01T00:00:00Z"),
    "PSEUDO_FORWARD": ("2026-01-02T00:00:00Z", "2026-08-01T00:00:00Z"),
}
SOURCE_HASHES = {
    60: {
        "events": "8ba9d92c3bf6b5f5bb2497fe123a23c757d86da96e4277a6e3112664774a63ef",
        "rows": "60c8341cac908ffc8882c150617bbfe91fe936c6a9352325c3068893b65508f5",
    },
    240: {
        "events": "961f0217398aee865935099b2e447ed1a97c52b024176ce45ae92abd5ad55608",
        "rows": "3e5b7fcfa4847c2933f101010f51759a9d66e4b9c93e9ebce8fa4574552f6a70",
    },
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


def _bootstrap(rows: pd.DataFrame, repetitions: int = 1000) -> dict[str, float] | None:
    if rows.empty:
        return None
    days = list(
        rows.assign(day=pd.to_datetime(rows.timestamp_ms, unit="ms", utc=True).dt.floor("1D"))
        .groupby("day")["net_primary"].apply(np.asarray)
    )
    random = np.random.default_rng(1812)
    values = []
    for _ in range(repetitions):
        sample = np.concatenate([days[index] for index in random.integers(0, len(days), len(days))])
        values.append(float(sample.mean()))
    return {"lower_95": float(np.quantile(values, 0.025)), "upper_95": float(np.quantile(values, 0.975))}


def _baseline_rows(rows: pd.DataFrame, timestamps: set[int], choices: dict[str, tuple[str, str]]) -> pd.DataFrame:
    selected = []
    sample = rows.loc[rows.timestamp_ms.isin(timestamps)]
    for regime, (side, mechanism) in choices.items():
        selected.append(select_mechanism_rows(sample.loc[sample.regime.eq(regime)], side, mechanism))
    if not selected:
        return pd.DataFrame()
    result = pd.concat(selected, ignore_index=True)
    result["net_primary"] = result.gross_return - PRIMARY_COST
    result["net_stress"] = result.gross_return - STRESS_COST
    return result


def _path_metrics(rows: pd.DataFrame, path_models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for side in ("LONG", "SHORT"):
        sample = rows.loc[rows.side.eq(side)]
        predicted_mae = path_models[side]["mae"].predict(sample.loc[:, list(RANK_FEATURES)])
        predicted_mfe = path_models[side]["mfe"].predict(sample.loc[:, list(RANK_FEATURES)])
        result[side] = {
            "rows": len(sample),
            "mae_spearman": float(pd.Series(sample.mae.to_numpy()).corr(pd.Series(predicted_mae), method="spearman")),
            "mfe_spearman": float(pd.Series(sample.mfe.to_numpy()).corr(pd.Series(predicted_mfe), method="spearman")),
        }
    return result


def _barrier_metrics(selected: pd.DataFrame, population: pd.DataFrame) -> dict[str, Any]:
    return {
        "selected_events": len(selected),
        "selected_favorable_first_rate": float(selected.favorable_first.mean()) if len(selected) else 0.0,
        "population_favorable_first_rate": float(population.favorable_first.mean()) if len(population) else 0.0,
    }


def _random_control(selected: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    output = []
    for event in selected.itertuples(index=False):
        pool = population.loc[population.timestamp_ms.eq(event.timestamp_ms)]
        if pool.empty:
            continue
        ordered = pool.sort_values(["symbol", "side"])
        digest = __import__("hashlib").sha256(f"b2-combined:{event.timestamp_ms}".encode("ascii")).digest()
        candidate = ordered.iloc[int.from_bytes(digest[:8], "big") % len(ordered)]
        output.append({
            **candidate.to_dict(), "net_primary": float(candidate.gross_return - PRIMARY_COST),
            "net_stress": float(candidate.gross_return - STRESS_COST),
        })
    return pd.DataFrame(output)


def _combined_gate(
    selected: pd.DataFrame,
    random: pd.DataFrame,
    bootstrap: dict[str, float] | None,
    component_gate: dict[str, bool],
) -> dict[str, bool]:
    summary = economic_summary(selected)
    random_summary = economic_summary(random)
    return {
        "minimum_events": len(selected) >= 100,
        "positive_net": summary.get("expectancy", -math.inf) > 0.0,
        "bootstrap_lower_positive": bool(bootstrap and bootstrap["lower_95"] > 0.0),
        "profit_factor": summary.get("profit_factor", 0.0) > 1.10,
        "stress_positive": economic_summary(selected, "net_stress").get("expectancy", -math.inf) > 0.0,
        "positive_thirds": sum(value > 0 for value in summary.get("temporal_thirds", ())) >= 2,
        "positive_symbols": summary.get("positive_symbols", 0) >= 7,
        "symbol_concentration": summary.get("maximum_symbol_share", math.inf) <= 0.25,
        "outperform_random": summary.get("expectancy", -math.inf) > random_summary.get("expectancy", math.inf),
        "component_gates": all(component_gate.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/residual_regime_alpha_b2/run_01"))
    parser.add_argument("--reuse-dataset", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    source = root / "data/opportunity_atlas_b1/run_01"
    minute_root = root / "data/market_event_fast_track_m1a/full_run_01/cache"
    minute = {symbol: pd.read_parquet(minute_root / f"{symbol}.parquet") for symbol in CANONICAL_SYMBOLS}
    reports: dict[str, Any] = {}
    artifacts: list[Path] = []
    horizon_passes = []
    for horizon in (60, 240):
        event_source = source / f"events_{horizon}m.parquet"
        row_source = source / f"symbol_sides_{horizon}m.parquet"
        if sha256_file(event_source) != SOURCE_HASHES[horizon]["events"] or sha256_file(row_source) != SOURCE_HASHES[horizon]["rows"]:
            raise RuntimeError("AEGIS_B2_SOURCE_HASH_MISMATCH")
        derived_path = output / f"residual_symbol_sides_{horizon}m.parquet"
        events = pd.read_parquet(event_source)
        if args.reuse_dataset and derived_path.exists():
            rows = pd.read_parquet(derived_path)
        else:
            rows = add_causal_residuals(pd.read_parquet(row_source))
            event_available = events.loc[events.timestamp_ms.isin(rows.timestamp_ms.unique())].copy()
            train_mask = partition_mask(event_available, *PARTITIONS["TRAIN"])
            thresholds = fit_regime_thresholds(event_available.loc[train_mask])
            event_available = assign_regimes(event_available, thresholds)
            rows = rows.merge(event_available[["timestamp_ms", "regime"]], on="timestamp_ms", validate="many_to_one")
            rows = add_barrier_outcomes(rows, minute, horizon)
            rows.to_parquet(derived_path, compression="zstd", index=False)
            os.chmod(derived_path, 0o600)
        events = events.loc[events.timestamp_ms.isin(rows.timestamp_ms.unique())].copy()
        masks = {name: partition_mask(events, *bounds) for name, bounds in PARTITIONS.items()}
        train_events = events.loc[masks["TRAIN"]]
        thresholds = fit_regime_thresholds(train_events)
        events = assign_regimes(events, thresholds)
        if "regime" in rows.columns:
            rows = rows.drop(columns="regime")
        rows = rows.merge(events[["timestamp_ms", "regime"]], on="timestamp_ms", validate="many_to_one")
        timestamps = {name: set(events.loc[mask, "timestamp_ms"].astype(int)) for name, mask in masks.items()}
        choices, choice_evidence = choose_regime_mechanisms(rows, timestamps["TRAIN"], timestamps["CALIBRATION"])
        rankers = fit_pairwise_rankers(rows, timestamps["TRAIN"])
        path_models = fit_path_models(rows, timestamps["TRAIN"])
        calibration_rows = rows.loc[rows.timestamp_ms.isin(timestamps["CALIBRATION"])]
        mae_limits, mfe_limits = {}, {}
        for side in ("LONG", "SHORT"):
            sample = calibration_rows.loc[calibration_rows.side.eq(side)]
            mae_limits[side] = float(np.quantile(path_models[side]["mae"].predict(sample.loc[:, list(RANK_FEATURES)]), 0.50))
            mfe_limits[side] = float(np.quantile(path_models[side]["mfe"].predict(sample.loc[:, list(RANK_FEATURES)]), 0.50))
        partition_reports = {}
        component_by_partition = {}
        selections = {}
        for name in ("VALIDATION", "PSEUDO_FORWARD"):
            population = rows.loc[rows.timestamp_ms.isin(timestamps[name])]
            baseline = _baseline_rows(rows, timestamps[name], choices)
            ranking = grouped_rank_metrics(population, rankers)
            path = _path_metrics(population, path_models)
            selected = select_combined(population, choices, rankers, path_models, mae_limits, mfe_limits)
            random = _random_control(selected, population)
            bootstrap = _bootstrap(selected)
            barrier = _barrier_metrics(selected, population)
            selections[name] = selected
            component_by_partition[name] = {"baseline": economic_summary(baseline), "ranking": ranking, "path": path, "barrier": barrier}
            partition_reports[name] = {
                "baseline": economic_summary(baseline), "selected": economic_summary(selected),
                "selected_stress": economic_summary(selected, "net_stress"),
                "random": economic_summary(random), "bootstrap": bootstrap, "barrier": barrier,
            }
            if len(selected):
                selected.to_parquet(output / f"selected_{horizon}m_{name.lower()}.parquet", compression="zstd", index=False)
        validation, forward = component_by_partition["VALIDATION"], component_by_partition["PSEUDO_FORWARD"]
        component_gate = {
            "direction_validation": validation["baseline"].get("expectancy", -math.inf) > 0.0,
            "direction_forward": forward["baseline"].get("expectancy", -math.inf) > 0.0,
            "ranking_validation": validation["ranking"].get("grouped_spearman", 0.0) >= 0.05,
            "ranking_forward": forward["ranking"].get("grouped_spearman", 0.0) >= 0.05,
            "ranking_random_validation": bool(validation["ranking"].get("top_outperforms_random", False)),
            "ranking_random_forward": bool(forward["ranking"].get("top_outperforms_random", False)),
            "path_validation": all(item["mae_spearman"] >= 0.10 for item in validation["path"].values()),
            "path_forward": all(item["mae_spearman"] >= 0.10 for item in forward["path"].values()),
            "barrier_validation": validation["barrier"]["selected_favorable_first_rate"] > validation["barrier"]["population_favorable_first_rate"],
            "barrier_forward": forward["barrier"]["selected_favorable_first_rate"] > forward["barrier"]["population_favorable_first_rate"],
        }
        combined_gates = {}
        horizon_pass = True
        for name in ("VALIDATION", "PSEUDO_FORWARD"):
            selected = selections[name]
            random = _random_control(selected, rows.loc[rows.timestamp_ms.isin(timestamps[name])])
            gate = _combined_gate(selected, random, _bootstrap(selected), component_gate)
            combined_gates[name] = gate
            horizon_pass &= all(gate.values())
        reports[f"{horizon}m"] = {
            "events": len(events), "rows": len(rows), "regime_thresholds": thresholds.__dict__,
            "confirmed_regime_mechanisms": {key: list(value) for key, value in choices.items()},
            "mechanism_evidence": choice_evidence, "ranker_count": len(rankers),
            "path_limits": {"mae": mae_limits, "mfe": mfe_limits},
            "components": component_by_partition, "component_gate": component_gate,
            "component_gate_pass": all(component_gate.values()), "partitions": partition_reports,
            "combined_gates": combined_gates, "horizon_gate_pass": horizon_pass,
        }
        artifacts.append(derived_path)
        horizon_passes.append(horizon_pass)
        print(f"b2_horizon={horizon} rows={len(rows)} regimes={len(choices)} pass={horizon_pass}", flush=True)
    path_confirmed = all(
        all(item["mae_spearman"] >= 0.10 for item in report["components"][partition]["path"].values())
        for report in reports.values() for partition in ("VALIDATION", "PSEUDO_FORWARD")
    )
    result = {
        "schema_version": "aegis-residual-regime-alpha-b2-result-v1",
        "experiment_id": "aegis-residual-regime-alpha-b2-01",
        "evidence_class": "CONTAMINATED_ARCHITECTURE_DIAGNOSTIC_NO_PROMOTION_AUTHORITY",
        "feature_contract": {"names": RANK_FEATURES, "sha256": contract_hash(RANK_FEATURES)},
        "artifacts": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in artifacts},
        "results": reports,
        "B2_RESIDUAL_ALPHA_FOUND": all(
            report["component_gate"]["ranking_validation"] and report["component_gate"]["ranking_forward"]
            for report in reports.values()
        ),
        "B2_REGIME_SPECIALIZATION_FOUND": all(
            report["component_gate"]["direction_validation"] and report["component_gate"]["direction_forward"]
            for report in reports.values()
        ),
        "B2_PATH_RISK_CONFIRMED": path_confirmed,
        "B2_COMBINED_POLICY_FOUND": any(horizon_passes),
        "B2_READY_FOR_FORWARD_EXPERIMENT": any(horizon_passes),
        "B2_READY_FOR_SHADOW": False, "B2_READY_FOR_LIVE": False,
        "exchange_calls": 0, "exchange_mutations": 0, "runtime_changes": "NONE",
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(_safe(result), indent=2, sort_keys=True) + "\n")
    os.chmod(result_path, 0o600)
    print(json.dumps({"result": str(result_path), "horizon_passes": horizon_passes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
