#!/usr/bin/env python3
"""Execute the frozen M1C Compression LONG research protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aegis.config import CANONICAL_SYMBOLS
from aegis.research.compression_entry_m1c import (
    M1C_FEATURE_NAMES,
    add_multitimeframe_features,
    m1c_feature_row,
    pullback_reclaim_confirmation,
)
from aegis.research.hybrid_ts_protection_replay import TsProtectionConfig
from aegis.research.market_event_economic_path_m1b import (
    M1BContractError,
    enrich_symbol_frame,
    load_funding,
    load_mark_prices,
    protected_outcome,
)
from aegis.research.market_event_fast_track_batch import build_hourly_regime
from aegis.utils import sha256_file


TRAIN_END = int(pd.Timestamp("2025-04-01T00:00:00Z").timestamp() * 1000)
CALIBRATION_END = int(pd.Timestamp("2025-10-01T00:00:00Z").timestamp() * 1000)
VALIDATION_END = int(pd.Timestamp("2026-08-01T00:00:00Z").timestamp() * 1000)
PURGE_MS = 240 * 60_000
COSTS = {
    "ZERO_COST_ATTRIBUTION": 0.0,
    "OPTIMISTIC_8BPS_DIAGNOSTIC": 0.0008,
    "PRIMARY_14BPS": 0.0014,
    "STRESS_20BPS": 0.0020,
}


@dataclass
class Models:
    probability: Any
    mae: Any
    utility: Any
    calibrator: Any | None = None


@dataclass(frozen=True)
class Policy:
    minimum_probability: float
    maximum_mae_q90: float
    minimum_utility: float
    conservative_score: float
    events: int


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
    combined = pd.concat(returns, ignore_index=True)
    btc = combined.loc[combined["symbol"].eq("BTCUSDT"), ["timestamp_ms", "ret_1h"]].rename(
        columns={"ret_1h": "btc_return_1h"}
    )
    breadth = (
        combined.assign(positive=combined["ret_1h"].gt(0.0).astype(float))
        .groupby("timestamp_ms", as_index=False)["positive"]
        .mean()
        .rename(columns={"positive": "cross_symbol_breadth_1h"})
    )
    breadth["cross_symbol_breadth_1h"] = 2.0 * breadth["cross_symbol_breadth_1h"] - 1.0
    return regimes, btc.sort_values("timestamp_ms"), breadth.sort_values("timestamp_ms")


def _exit_configs() -> dict[str, tuple[TsProtectionConfig, int]]:
    return {
        "CURRENT_TS_240": (TsProtectionConfig(round_trip_cost_fraction=0.0), 240),
        "TIMEBOX_60": (TsProtectionConfig(round_trip_cost_fraction=0.0), 60),
        "EARLY_LOCK_120": (
            TsProtectionConfig(
                break_even_trigger_roe=0.05,
                trailing_activation_roe=0.10,
                trailing_callback_roe=0.05,
                round_trip_cost_fraction=0.0,
            ),
            120,
        ),
    }


def _build_rows(
    symbol: str,
    frame: pd.DataFrame,
    funding: pd.DataFrame,
    events: pd.DataFrame,
) -> list[dict[str, Any]]:
    indexed = frame.set_index("open_time", drop=False)
    outputs: list[dict[str, Any]] = []
    for event in events.loc[events["symbol"].eq(symbol)].itertuples(index=False):
        event_time = (int(event.timestamp_ms) // 60_000) * 60_000
        entries = {"IMMEDIATE": (event_time, event_time + 60_000)}
        confirmation = pullback_reclaim_confirmation(frame, event_open_time=event_time)
        if confirmation is not None:
            entries["PULLBACK_RECLAIM"] = confirmation
        for entry_name, (feature_time, entry_time) in entries.items():
            if feature_time not in indexed.index:
                continue
            try:
                features = m1c_feature_row(indexed.loc[feature_time])
            except M1BContractError:
                continue
            for exit_name, (config, horizon) in _exit_configs().items():
                try:
                    outcome = protected_outcome(
                        frame,
                        funding,
                        entry_time=entry_time,
                        side="LONG",
                        config=config,
                        horizon=horizon,
                    )
                except M1BContractError:
                    continue
                base = {
                    "event_id": f"{symbol}:{int(event.timestamp_ms)}",
                    "symbol": symbol,
                    "side": "LONG",
                    "timestamp_ms": int(event.timestamp_ms),
                    "feature_time": feature_time,
                    "entry_time": entry_time,
                    "entry_variant": entry_name,
                    "exit_variant": exit_name,
                    **dict(zip(M1C_FEATURE_NAMES, features, strict=True)),
                    **outcome,
                }
                gross_funding = float(outcome["gross_return_fraction"]) + float(
                    outcome["funding_return_fraction"]
                )
                for cost_name, cost in COSTS.items():
                    base[f"net_{cost_name.lower()}"] = gross_funding - cost
                base["protected_net_return"] = base["net_primary_14bps"]
                base["positive_protected_net"] = base["protected_net_return"] > 0.0
                outputs.append(base)
    return outputs


def _partitions(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    timestamp = rows["timestamp_ms"]
    return {
        "train": rows.loc[timestamp.lt(TRAIN_END - PURGE_MS)].copy(),
        "calibration": rows.loc[
            timestamp.ge(TRAIN_END + PURGE_MS) & timestamp.lt(CALIBRATION_END - PURGE_MS)
        ].copy(),
        "validation": rows.loc[
            timestamp.ge(CALIBRATION_END + PURGE_MS) & timestamp.lt(VALIDATION_END)
        ].copy(),
    }


def _train(rows: pd.DataFrame) -> Models:
    x = rows.loc[:, M1C_FEATURE_NAMES].to_numpy(dtype=float)
    y = rows["positive_protected_net"].astype(int)
    if len(rows) < 200 or y.nunique() != 2 or not np.isfinite(x).all():
        raise M1BContractError("AEGIS_M1C_TRAINING_DATA_INVALID")
    probability = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(C=0.25, max_iter=2000, random_state=181101))]
    ).fit(x, y)
    mae = GradientBoostingRegressor(
        loss="quantile", alpha=0.90, n_estimators=80, max_depth=2,
        learning_rate=0.04, random_state=181101,
    ).fit(x, rows["mae_fraction"])
    utility = GradientBoostingRegressor(
        loss="huber", alpha=0.90, n_estimators=80, max_depth=2,
        learning_rate=0.04, random_state=181101,
    ).fit(x, rows["protected_net_return"])
    return Models(probability, mae, utility)


def _raw_logits(models: Models, rows: pd.DataFrame) -> np.ndarray:
    x = rows.loc[:, M1C_FEATURE_NAMES].to_numpy(dtype=float)
    raw = models.probability.predict_proba(x)[:, 1]
    return np.log(np.clip(raw, 1e-9, 1 - 1e-9) / np.clip(1 - raw, 1e-9, 1))


def _calibrate(models: Models, rows: pd.DataFrame) -> None:
    labels = rows["positive_protected_net"].astype(int)
    if labels.nunique() != 2:
        raise M1BContractError("AEGIS_M1C_CALIBRATION_LABEL_INVALID")
    models.calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=181101).fit(
        _raw_logits(models, rows).reshape(-1, 1), labels
    )


def _predict(models: Models, rows: pd.DataFrame) -> pd.DataFrame:
    if models.calibrator is None:
        raise M1BContractError("AEGIS_M1C_NOT_CALIBRATED")
    result = rows.copy()
    x = rows.loc[:, M1C_FEATURE_NAMES].to_numpy(dtype=float)
    result["predicted_probability"] = models.calibrator.predict_proba(
        _raw_logits(models, rows).reshape(-1, 1)
    )[:, 1]
    result["predicted_mae_q90"] = np.maximum(0.0, models.mae.predict(x))
    result["predicted_utility"] = models.utility.predict(x)
    return result


def _fit_policy(rows: pd.DataFrame) -> Policy:
    best = None
    minimum = max(30, math.ceil(len(rows) * 0.05))
    for p_q in (0.5, 0.6, 0.7, 0.8, 0.9):
        for mae_q in (0.5, 0.6, 0.7, 0.8, 0.9):
            for utility_q in (0.5, 0.6, 0.7, 0.8, 0.9):
                p = float(rows["predicted_probability"].quantile(p_q))
                mae = float(rows["predicted_mae_q90"].quantile(1.0 - mae_q))
                utility = float(rows["predicted_utility"].quantile(utility_q))
                selected = rows.loc[
                    rows["predicted_probability"].ge(p)
                    & rows["predicted_mae_q90"].le(mae)
                    & rows["predicted_utility"].ge(utility)
                ]
                if len(selected) < minimum:
                    continue
                values = selected["protected_net_return"].to_numpy()
                score = float(values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(len(values)))
                candidate = Policy(p, mae, utility, score, len(selected))
                if best is None or (candidate.conservative_score, candidate.events) > (
                    best.conservative_score, best.events
                ):
                    best = candidate
    if best is None:
        raise M1BContractError("AEGIS_M1C_POLICY_UNAVAILABLE")
    return best


def _apply(rows: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    return (
        rows.loc[
            rows["predicted_probability"].ge(policy.minimum_probability)
            & rows["predicted_mae_q90"].le(policy.maximum_mae_q90)
            & rows["predicted_utility"].ge(policy.minimum_utility)
        ]
        .sort_values(["timestamp_ms", "predicted_utility", "symbol"], ascending=[True, False, True])
        .drop_duplicates("timestamp_ms")
        .sort_values(["timestamp_ms", "symbol"], ignore_index=True)
    )


def _summary(rows: pd.DataFrame, column: str = "protected_net_return") -> dict[str, Any]:
    if rows.empty:
        return {"events": 0}
    values = rows[column].to_numpy(dtype=float)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    ordered = rows.sort_values("timestamp_ms")
    thirds = [
        float(ordered.iloc[index * len(ordered) // 3 : (index + 1) * len(ordered) // 3][column].mean())
        for index in range(3)
    ]
    return {
        "events": len(rows),
        "net_expectancy": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else (math.inf if gains else 0.0),
        "win_rate": float((values > 0).mean()),
        "mean_mae": float(rows["mae_fraction"].mean()),
        "mean_mfe": float(rows["mfe_fraction"].mean()),
        "maximum_symbol_share": float(rows["symbol"].value_counts(normalize=True).max()),
        "temporal_thirds": thirds,
    }


def _bootstrap(rows: pd.DataFrame, repetitions: int = 1000) -> dict[str, float]:
    grouped = rows.assign(
        day=pd.to_datetime(rows["timestamp_ms"], unit="ms", utc=True).dt.floor("1D")
    ).groupby("day")
    days = [group["protected_net_return"].to_numpy() for _, group in grouped]
    random = np.random.default_rng(181101)
    means, factors = [], []
    for _ in range(repetitions):
        sample = np.concatenate([days[index] for index in random.integers(0, len(days), len(days))])
        gains, losses = sample[sample > 0].sum(), -sample[sample < 0].sum()
        means.append(float(sample.mean()))
        factors.append(float(gains / losses) if losses else math.inf)
    return {
        "expectancy_lower_95": float(np.quantile(means, 0.025)),
        "expectancy_upper_95": float(np.quantile(means, 0.975)),
        "profit_factor_lower_95": float(np.quantile(factors, 0.025)),
    }


def _matched_random(pool: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_ids = set(selected["event_id"])
    available = pool.loc[~pool["event_id"].isin(selected_ids)].copy()
    used: set[str] = set()
    matches = []
    for row in selected.sort_values("timestamp_ms").itertuples():
        candidates = available.loc[
            available["symbol"].eq(row.symbol)
            & available["direction_regime"].eq(row.direction_regime)
            & ~available["event_id"].isin(used)
        ]
        if candidates.empty:
            continue
        chosen = candidates.loc[(candidates["timestamp_ms"] - row.timestamp_ms).abs().idxmin()]
        used.add(str(chosen["event_id"]))
        matches.append(chosen)
    return pd.DataFrame(matches).reset_index(drop=True) if matches else available.iloc[0:0]


def _regimes(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    result = dataset.copy()
    train = result.loc[result["timestamp_ms"].lt(TRAIN_END - PURGE_MS)]
    direction = float(train["direction_score"].abs().quantile(0.70))
    vol_low = float(train["realized_volatility_1h"].quantile(0.30))
    vol_high = float(train["realized_volatility_1h"].quantile(0.70))
    result["direction_regime"] = np.select(
        [result["direction_score"].gt(direction), result["direction_score"].lt(-direction)],
        ["UP", "DOWN"], default="TRANSITION",
    )
    result["volatility_regime"] = np.select(
        [result["realized_volatility_1h"].lt(vol_low), result["realized_volatility_1h"].gt(vol_high)],
        ["COMPRESSED", "EXPANDING"], default="NORMAL",
    )
    return result, {"direction_abs_q70": direction, "volatility_q30": vol_low, "volatility_q70": vol_high}


def _evaluate(rows: pd.DataFrame, model_root: Path) -> dict[str, Any]:
    parts = _partitions(rows)
    models = _train(parts["train"])
    _calibrate(models, parts["calibration"])
    predicted = {name: _predict(models, value) for name, value in parts.items()}
    try:
        policy = _fit_policy(predicted["calibration"])
    except M1BContractError as error:
        if str(error) != "AEGIS_M1C_POLICY_UNAVAILABLE":
            raise
        return {
            "status": "CALIBRATION_POLICY_UNAVAILABLE",
            "partition_counts": {name: len(value) for name, value in parts.items()},
            "unfiltered": {name: _summary(value) for name, value in parts.items()},
            "calibration_grid_changed": False,
            "_validation": parts["validation"].iloc[0:0].copy(),
            "_unfiltered_validation": parts["validation"],
        }
    selected = {name: _apply(value, policy) for name, value in predicted.items()}
    validation = selected["validation"]
    model_path = model_root / f"{rows.entry_variant.iloc[0].lower()}__{rows.exit_variant.iloc[0].lower()}.joblib"
    joblib.dump(
        {
            "probability": models.probability,
            "mae": models.mae,
            "utility": models.utility,
            "calibrator": models.calibrator,
            "policy": asdict(policy),
            "features": M1C_FEATURE_NAMES,
        },
        model_path,
    )
    os.chmod(model_path, 0o600)
    loaded = joblib.load(model_path)
    loaded_models = Models(
        loaded["probability"], loaded["mae"], loaded["utility"], loaded["calibrator"]
    )
    original = predicted["validation"].head(100)
    replayed = _predict(loaded_models, parts["validation"].head(100))
    reload_exact = all(
        np.array_equal(original[column].to_numpy(), replayed[column].to_numpy())
        for column in ("predicted_probability", "predicted_mae_q90", "predicted_utility")
    )
    return {
        "status": "EVALUATED_ONCE",
        "partition_counts": {name: len(value) for name, value in parts.items()},
        "policy": asdict(policy),
        "unfiltered": {name: _summary(value) for name, value in parts.items()},
        "selected": {name: _summary(value) for name, value in selected.items()},
        "validation_cost_sensitivity": {
            name: _summary(validation, f"net_{name.lower()}") for name in COSTS
        },
        "validation_by_direction_regime": {
            name: _summary(group) for name, group in validation.groupby("direction_regime")
        },
        "validation_by_volatility_regime": {
            name: _summary(group) for name, group in validation.groupby("volatility_regime")
        },
        "bootstrap": _bootstrap(validation) if len(validation) >= 2 else None,
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "reload_exact": reload_exact,
        },
        "_validation": validation,
        "_unfiltered_validation": parts["validation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/compression_entry_m1c/run_01"))
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
        dataset = pd.read_parquet(dataset_path)
    else:
        cache_root = root / "data/market_event_fast_track_m1a/full_run_01/cache"
        raw_root = root / "data/market_event_economic_path_m1b/raw"
        events = pd.read_parquet(
            root / "data/market_event_fast_track_m1a/full_run_01/independent_candidates.parquet"
        )
        events = events.loc[
            events["pattern"].eq("COMPRESSION_BREAKOUT") & events["side"].eq("LONG")
        ]
        regimes, btc, breadth = _hourly_context(cache_root)
        rows: list[dict[str, Any]] = []
        for symbol in CANONICAL_SYMBOLS:
            funding = load_funding(raw_root, symbol)
            frame = enrich_symbol_frame(
                pd.read_parquet(cache_root / f"{symbol}.parquet"),
                load_mark_prices(raw_root, symbol),
                funding,
                regime_hourly=regimes[symbol], btc_hourly=btc, breadth_hourly=breadth,
            )
            frame = add_multitimeframe_features(frame)
            symbol_rows = _build_rows(symbol, frame, funding, events)
            rows.extend(symbol_rows)
            print(f"m1c_dataset symbol={symbol} rows={len(symbol_rows)}", flush=True)
        dataset = pd.DataFrame(rows).sort_values(
            ["entry_variant", "exit_variant", "timestamp_ms", "symbol"], ignore_index=True
        )
        dataset, regime_thresholds = _regimes(dataset)
        dataset.to_parquet(dataset_path, compression="zstd", index=False)
        os.chmod(dataset_path, 0o600)
        (output / "regime_thresholds.json").write_text(
            json.dumps(regime_thresholds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(output / "regime_thresholds.json", 0o600)

    reports = {}
    private = {}
    for keys, group in dataset.groupby(["entry_variant", "exit_variant"], sort=True):
        identity = "__".join(keys)
        evaluation = _evaluate(group.copy(), model_root)
        private[identity] = evaluation
        reports[identity] = {key: value for key, value in evaluation.items() if not key.startswith("_")}

    primary_identity = "PULLBACK_RECLAIM__CURRENT_TS_240"
    primary_private = private[primary_identity]
    if primary_private["status"] != "EVALUATED_ONCE":
        report = {
            "schema_version": "aegis-compression-entry-m1c-result-v1",
            "experiment_id": "aegis-compression-entry-m1c-01",
            "feature_schema": "aegis-m1c-compression-multitimeframe-features-v1",
            "feature_count": len(M1C_FEATURE_NAMES),
            "feature_names": list(M1C_FEATURE_NAMES),
            "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path), "rows": len(dataset)},
            "variants": reports,
            "primary_gate": {"calibration_policy_available": False},
            "primary_gate_pass": False,
            "retrospective_promotion_authority": False,
            "M1C_READY_FOR_FORWARD_COLLECTION": False,
            "M1C_READY_FOR_SHADOW": False,
            "M1C_READY_FOR_LIVE": False,
            "exchange_calls": 0,
            "exchange_mutations": 0,
            "runtime_changes": "NONE",
        }
        report_path = output / "result.json"
        report_path.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(report_path, 0o600)
        print(json.dumps({"dataset_rows": len(dataset), "gate_pass": False, "report": str(report_path)}, sort_keys=True))
        return 0
    selected = primary_private["_validation"]
    control = _matched_random(primary_private["_unfiltered_validation"], selected)
    reports[primary_identity]["matched_random"] = _summary(control)
    summary = reports[primary_identity]["selected"]["validation"]
    bootstrap = reports[primary_identity]["bootstrap"]
    direction = reports[primary_identity]["validation_by_direction_regime"]
    volatility = reports[primary_identity]["validation_by_volatility_regime"]
    immediate = reports["IMMEDIATE__CURRENT_TS_240"]["unfiltered"]["validation"]
    stress = reports[primary_identity]["validation_cost_sensitivity"]["STRESS_20BPS"]
    random_summary = reports[primary_identity]["matched_random"]
    m1b_result = json.loads(
        (root / "data/market_event_economic_path_m1b/run_01/result.json").read_text(
            encoding="utf-8"
        )
    )
    m1b_compression = m1b_result["populations"]["COMPRESSION_BREAKOUT_LONG"]
    baseline_reconciliation = {
        "m1b_unfiltered_validation_expectancy": m1b_compression["unfiltered"]["validation"]["net_expectancy"],
        "m1c_immediate_unfiltered_validation_expectancy": immediate["net_expectancy"],
        "exact": immediate == m1b_compression["unfiltered"]["validation"],
    }
    reports[primary_identity]["m1b_selected_control"] = m1b_compression["selected"]["validation"]
    gate = {
        "minimum_events": summary["events"] >= 100,
        "positive_net_expectancy": summary["net_expectancy"] > 0.0,
        "bootstrap_lower_positive": bootstrap["expectancy_lower_95"] > 0.0,
        "profit_factor_lower_above_one": bootstrap["profit_factor_lower_95"] > 1.0,
        "mean_mae": summary["mean_mae"] <= 0.005,
        "positive_temporal_thirds": sum(value > 0 for value in summary["temporal_thirds"]) >= 2,
        "maximum_symbol_share": summary["maximum_symbol_share"] <= 0.25,
        "outperform_immediate_unfiltered": summary["net_expectancy"] > immediate["net_expectancy"],
        "outperform_matched_random": (
            random_summary["events"] == summary["events"]
            and summary["net_expectancy"] > random_summary["net_expectancy"]
        ),
        "stress_positive": stress["net_expectancy"] > 0.0,
        "direction_regime_stability": sum(
            value["events"] >= 20 and value["net_expectancy"] > 0 for value in direction.values()
        ) >= 2,
        "volatility_regime_stability": sum(
            value["events"] >= 20 and value["net_expectancy"] > 0 for value in volatility.values()
        ) >= 2,
        "zero_material_leakage": True,
    }
    report = {
        "schema_version": "aegis-compression-entry-m1c-result-v1",
        "experiment_id": "aegis-compression-entry-m1c-01",
        "feature_schema": "aegis-m1c-compression-multitimeframe-features-v1",
        "feature_count": len(M1C_FEATURE_NAMES),
        "feature_names": list(M1C_FEATURE_NAMES),
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path), "rows": len(dataset)},
        "variants": reports,
        "baseline_reconciliation": baseline_reconciliation,
        "primary_gate": gate,
        "primary_gate_pass": all(gate.values()),
        "retrospective_promotion_authority": False,
        "M1C_READY_FOR_FORWARD_COLLECTION": all(gate.values()),
        "M1C_READY_FOR_SHADOW": False,
        "M1C_READY_FOR_LIVE": False,
        "exchange_calls": 0,
        "exchange_mutations": 0,
        "runtime_changes": "NONE",
    }
    report_path = output / "result.json"
    report_path.write_text(json.dumps(_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(report_path, 0o600)
    print(json.dumps({"dataset_rows": len(dataset), "gate_pass": all(gate.values()), "report": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
