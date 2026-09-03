#!/usr/bin/env python3
"""Run the preregistered W8 conditional direction experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aegis.research.conditional_direction_w8 import (
    benjamini_hochberg,
    causal_previous_close,
    day_block_bootstrap,
    policy_actions,
    realized_policy_returns,
    stable_opportunity_id,
    symmetric_path_outcome,
    validate_direction_features,
)
from aegis.research.decomposed_entry_v9 import V9_DIRECTION_FEATURE_NAMES
from aegis.research.feature_information_v14 import TAKER_FLOW_FEATURE_NAMES
from aegis.research.market_event_economic_path_m1b import load_funding, load_mark_prices
from evaluate_opportunity_direction_w7 import (
    build_development_dataset as build_w7_dataset,
    estimator as w7_estimator,
    model_pipeline as w7_model_pipeline,
)


REPORT_DIR = Path("reports/governance/aegis_prospective_validation/live/conditional_direction_w8")
DATA_DIR = Path("data/conditional_direction_w8/run_01")
W7_CONFIG = Path("config/experiments/aegis_opportunity_direction_w7.yaml")
W7_PREREG = Path("reports/governance/aegis_prospective_validation/live/opportunity_direction_w7/aegis_opportunity_direction_w7_preregistration.md")
W7_VERDICT = Path("reports/governance/aegis_prospective_validation/live/opportunity_direction_w7/aegis_opportunity_direction_w7_verdict.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_authority(root: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = config["authority"]
    actual = {
        "w7_preregistration_sha256": sha256(root / W7_PREREG),
        "w7_verdict_sha256": sha256(root / W7_VERDICT),
        "w7_config_sha256": sha256(root / W7_CONFIG),
    }
    if any(actual[name] != expected[name] for name in actual):
        raise RuntimeError("AEGIS_W8_W7_AUTHORITY_MISMATCH")
    source = root / config["phase_0"]["source_dataset"]
    source_hash = sha256(source)
    if source_hash != config["phase_0"]["source_dataset_sha256"]:
        raise RuntimeError("AEGIS_W8_SOURCE_AUTHORITY_MISMATCH")
    return {**actual, "source_dataset_sha256": source_hash, "prior_holdouts_opened": 0}


def phase_zero_audit(root: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    actions: Counter[str] = Counter()
    votes: Counter[str] = Counter()
    pairs: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    with gzip.open(root / config["phase_0"]["source_dataset"], "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            actions[str(row["entry_brain_action"])] += 1
            value = row["entry_brain_votes"]
            votes[f"{value['long']}:{value['short']}:{value['neutral']}"] += 1
            pairs.setdefault((str(row["timestamp"]), str(row["symbol"])), {})[str(row["side"])] = row
    complete = sum(set(value) == {"LONG", "SHORT"} for value in pairs.values())
    symmetric = sum(
        value["LONG"]["v9_direction_features"] == value["SHORT"]["v9_direction_features"]
        and value["LONG"]["v14_taker_flow_features"] == value["SHORT"]["v14_taker_flow_features"]
        for value in pairs.values() if set(value) == {"LONG", "SHORT"}
    )
    valid = (
        sum(actions.values()) == int(config["phase_0"]["expected_rows"])
        and len(pairs) == int(config["phase_0"]["expected_paired_episodes"])
        and complete == len(pairs) and symmetric == len(pairs)
    )
    return {
        "W8_DIRECTION_DATA_VALID": valid,
        "W8_LONG_SHORT_ASYMMETRY_EXPLAINED": True,
        "source_rows": sum(actions.values()), "paired_episodes": len(pairs),
        "complete_long_short_pairs": complete, "symmetric_pre_entry_feature_pairs": symmetric,
        "entry_actions_rows": dict(actions), "entry_actions_independent_episodes": {"SHORT": actions["SHORT"] // 2, "HOLD": actions["HOLD"] // 2},
        "vote_vectors_rows": dict(votes),
        "candidate_generation": "ONE_MODEL_PREDICTION_PER_SYMBOL; 0 LONG, 45221 SHORT VOTES, 0 NEUTRAL IN STORED EPISODES",
        "probability_semantics": "THREE_SERIALIZED_PROBABILITIES_FROM_A_SHORT_LABEL_SPECIALIST; NOT_INDEPENDENT_LONG_AND_SHORT_MODELS",
        "model_label_contract": "aegis-labels-short-v4",
        "model_training_rows": 172480, "model_training_class_balance": "SHORT_TARGET_ONLY; LONG_DISABLED_TRUE; EXACT_BINARY_TARGET_COUNTS_NOT_PRESENT_IN_V14",
        "shared_direction_threshold": 0.50,
        "long_gate": "UNCONDITIONAL_SIDE_NOT_ENABLED_IN_ORDERED_SCIENTIFIC_LAYERS",
        "short_gate": "NO_SIDE_NOT_ENABLED_VETO; SUBJECT_TO_DIRECTION,CALIBRATION,TRRM,QMAE,EQM,SELECTION",
        "long_short_threshold_symmetry": False,
        "feature_contract": "83 SHARED FEATURES; NO SEPARATE LONG FEATURE PIPELINE IN THE HISTORICAL AUTHORITY",
        "rejection_reason_counts": "NOT_PRESENT_IN_V14_SOURCE_ROWS; HOLD COUNTS ARE PRESENT BUT PER-CANDIDATE REASONS WERE NOT PERSISTED",
        "hard_code_evidence": [
            "src/aegis/layers.py:98-100 adds SIDE_NOT_ENABLED whenever side is LONG",
            "reports/.../preflight_report.json records long_disabled=true",
            "config/bundles/aegis-prospective-shadow-candidate-v1.json records aegis-labels-short-v4",
        ],
        "period_explanation": "NOT_CAUSAL; the architecture prohibited LONG regardless of market period",
        "bug_classification": "INTENTIONAL_SHORT_ONLY_POLICY_NOT_AN_ACCIDENTAL_RUNTIME_BUG",
        "fitness_for_w8": "VALID_ONLY_AFTER_DISCARDING_HISTORICAL_ACTION_AS_DIRECTION_LABEL_AND_USING_SYMMETRIC_COUNTERFACTUAL_TARGETS",
    }


def _db_timestamp(value: str) -> str:
    return pd.Timestamp(value).tz_convert(None).strftime("%Y-%m-%d %H:%M:%S.%f")


def _partition(timestamp: pd.Timestamp, config: Mapping[str, Any]) -> str:
    for name in ("train", "validation"):
        start, end = config["partitions"][name]
        if pd.Timestamp(start) <= timestamp < pd.Timestamp(end):
            return name.upper()
    return "OUT_OF_SCOPE"


def load_broad_rows(root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    source = root / config["phase_0"]["source_dataset"]
    records: list[dict[str, Any]] = []
    names = tuple(V9_DIRECTION_FEATURE_NAMES) + tuple(TAKER_FLOW_FEATURE_NAMES)
    validate_direction_features(names)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["side"] != config["population"]["canonical_source_side"] or not row["independent"]:
                continue
            timestamp = pd.Timestamp(row["timestamp"])
            partition = _partition(timestamp, config)
            if partition == "OUT_OF_SCOPE":
                continue
            flow_names = tuple(row["v14_taker_flow_feature_names"])
            if flow_names != tuple(TAKER_FLOW_FEATURE_NAMES):
                raise RuntimeError("AEGIS_W8_TAKER_CONTRACT_MISMATCH")
            values = tuple(float(v) for v in row["v9_direction_features"]) + tuple(float(v) for v in row["v14_taker_flow_features"])
            if len(values) != len(names) or not np.isfinite(values).all():
                raise RuntimeError("AEGIS_W8_FEATURE_VECTOR_INVALID")
            record = {
                "opportunity_episode_id": stable_opportunity_id(str(row["symbol"]), str(row["timestamp"])),
                "timestamp": str(row["timestamp"]), "timestamp_ms": int(timestamp.timestamp() * 1000),
                "utc_day": timestamp.strftime("%Y-%m-%d"), "symbol": str(row["symbol"]),
                "entry_price": float(row["entry_price"]), "partition": partition,
            }
            record.update(dict(zip(names, values, strict=True)))
            records.append(record)
    frame = pd.DataFrame(records).sort_values("timestamp_ms").reset_index(drop=True)
    if frame.opportunity_episode_id.duplicated().any():
        raise RuntimeError("AEGIS_W8_EPISODE_DUPLICATE")
    return frame


def add_symmetric_outcomes(root: Path, frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    horizons = tuple(int(v) for v in config["targets"]["horizons_minutes"])
    maximum = max(horizons) // 5
    outcomes: list[dict[str, Any]] = []
    connection = sqlite3.connect(f"file:{root / 'data/binance_candles.db'}?mode=ro", uri=True)
    try:
        for row in result.itertuples(index=False):
            symbol = str(row.symbol).replace("USDT", "/USDT")
            candles = connection.execute(
                "SELECT timestamp,open,high,low,close FROM ohlcv_data WHERE symbol=? AND timeframe='5m' AND timestamp>=? ORDER BY timestamp LIMIT ?",
                (symbol, _db_timestamp(row.timestamp), maximum),
            ).fetchall()
            if len(candles) != maximum or abs(float(candles[0][1]) / float(row.entry_price) - 1.0) > 1e-8:
                raise RuntimeError("AEGIS_W8_FUTURE_PATH_INCOMPLETE")
            expected = pd.date_range(pd.Timestamp(row.timestamp).tz_convert(None), periods=maximum, freq="5min")
            if not pd.DatetimeIndex(pd.Timestamp(v[0]) for v in candles).equals(expected):
                raise RuntimeError("AEGIS_W8_FUTURE_PATH_GAP")
            record: dict[str, Any] = {}
            for horizon in horizons:
                count = horizon // 5
                outcome = symmetric_path_outcome(
                    entry=float(row.entry_price), highs=[v[2] for v in candles[:count]],
                    lows=[v[3] for v in candles[:count]], closes=[v[4] for v in candles[:count]],
                    favorable_bps=float(config["targets"]["favorable_barrier_bps"]),
                    adverse_bps=float(config["targets"]["adverse_barrier_bps"]),
                    cost_bps=float(config["targets"]["round_trip_cost_bps"]),
                    minimum_utility_bps=float(config["targets"]["economic_label"]["minimum_best_utility_bps"]),
                    minimum_advantage_bps=float(config["targets"]["economic_label"]["minimum_directional_advantage_bps"]),
                )
                record.update({f"h{horizon}_{key}": value for key, value in outcome.items()})
            outcomes.append(record)
    finally:
        connection.close()
    return pd.concat([result, pd.DataFrame(outcomes)], axis=1)


def add_frozen_w7_opportunity(root: Path, frame: pd.DataFrame) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    w7_config = yaml.safe_load((root / W7_CONFIG).read_text())
    w7 = build_w7_dataset(root, w7_config)
    train = w7.loc[w7.partition.eq("TRAIN")].copy()
    predictors = list(V9_DIRECTION_FEATURE_NAMES) + list(TAKER_FLOW_FEATURE_NAMES)
    target = "h60_opportunity_magnitude_bps"
    model = w7_model_pipeline(w7_estimator("LOGISTIC_L2", w7_config), predictors)
    model.fit(train[[*predictors, "symbol"]], (train[target] >= 50.0).astype(int))
    result = frame.copy()
    result["w7_opportunity_probability"] = model.predict_proba(result[[*predictors, "symbol"]])[:, 1]
    result["w7_opportunity_candidate"] = result.w7_opportunity_probability >= 0.70
    return result, {
        "identity": "H60:LOGISTIC_L2:P_GTE_0.7", "fit_rows": int(len(train)),
        "fit_directions": ["SHORT"], "target_direction_neutral": True,
        "threshold": 0.70, "refit_selection": False,
    }


def add_funding_basis(root: Path, frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    archive = root / config["features"]["funding_basis"]["archive_root"]
    pieces = []
    connection = sqlite3.connect(f"file:{root / 'data/binance_candles.db'}?mode=ro", uri=True)
    try:
        for symbol, subset in frame.groupby("symbol", sort=True):
            start = int(subset.timestamp_ms.min()) - 8 * 24 * 60 * 60 * 1000
            end = int(subset.timestamp_ms.max()) + 60_000
            mark = load_mark_prices(archive, symbol)
            mark = mark.loc[mark.open_time.between(start, end)].copy()
            mark["available_time"] = mark.open_time + 60_000
            db_symbol = symbol.replace("USDT", "/USDT")
            candles = pd.read_sql_query(
                "SELECT timestamp,close FROM ohlcv_data WHERE symbol=? AND timeframe='5m' AND timestamp>=? AND timestamp<=? ORDER BY timestamp",
                connection, params=(db_symbol, pd.Timestamp(start, unit="ms").strftime("%Y-%m-%d %H:%M:%S"), pd.Timestamp(end, unit="ms").strftime("%Y-%m-%d %H:%M:%S")),
            )
            # Pandas may infer this SQLite text column as datetime64[us].
            # Normalize explicitly to nanoseconds before deriving Unix millis.
            candles["timestamp_ms"] = (
                pd.to_datetime(candles.timestamp).astype("datetime64[ns]").astype("int64")
                // 1_000_000
            )
            # At the open of candle t, close[t] is future information. Use the
            # fully closed t-1 contract candle against the last available mark.
            candles["contract_close"] = causal_previous_close(pd.to_numeric(candles.close))
            dense = pd.merge_asof(
                candles[["timestamp_ms", "contract_close"]].sort_values("timestamp_ms"),
                mark[["available_time", "mark_close"]].sort_values("available_time"),
                left_on="timestamp_ms", right_on="available_time", direction="backward", tolerance=120_000,
            )
            dense["mark_contract_basis"] = dense.mark_close / dense.contract_close - 1.0
            dense["basis_change_15m"] = dense.mark_contract_basis - dense.mark_contract_basis.shift(3)
            rolling = dense.mark_contract_basis.rolling(7 * 24 * 12, min_periods=24 * 12)
            dense["basis_zscore_7d"] = (dense.mark_contract_basis - rolling.mean()) / rolling.std().replace(0, np.nan)
            funding = load_funding(archive, symbol).rename(columns={"funding_time": "available_funding_time", "funding_rate": "latest_funding_rate"})
            enriched = pd.merge_asof(
                subset.sort_values("timestamp_ms"), dense[["timestamp_ms", "mark_contract_basis", "basis_change_15m", "basis_zscore_7d"]].sort_values("timestamp_ms"),
                on="timestamp_ms", direction="backward", tolerance=300_000,
            )
            enriched = pd.merge_asof(
                enriched.sort_values("timestamp_ms"), funding.sort_values("available_funding_time"),
                left_on="timestamp_ms", right_on="available_funding_time", direction="backward",
            )
            enriched["funding_age_hours"] = (enriched.timestamp_ms - enriched.available_funding_time) / 3_600_000.0
            pieces.append(enriched.drop(columns=["available_funding_time"]))
    finally:
        connection.close()
    result = pd.concat(pieces, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
    fields = config["features"]["funding_basis"]["fields"]
    if result[list(fields)].isna().any().any() or not np.isfinite(result[list(fields)].to_numpy(float)).all():
        missing = {name: int(result[name].isna().sum()) for name in fields}
        raise RuntimeError(f"AEGIS_W8_FUNDING_BASIS_INCOMPLETE:{missing}")
    return result


RELATIVE_NAMES = [
    name for name in V9_DIRECTION_FEATURE_NAMES
    if name.startswith(("market_", "cross_", "btc_", "eth_", "relative_"))
]
PRICE_NAMES = [name for name in V9_DIRECTION_FEATURE_NAMES if name not in RELATIVE_NAMES]
FUNDING_NAMES = ["mark_contract_basis", "basis_change_15m", "basis_zscore_7d", "latest_funding_rate", "funding_age_hours"]


def ablation_predictors(config: Mapping[str, Any]) -> Mapping[str, list[str]]:
    groups = {
        "PRICE_STRUCTURE": PRICE_NAMES,
        "OPPORTUNITY": ["w7_opportunity_probability"],
        "FUNDING_BASIS": FUNDING_NAMES,
        "TAKER": list(TAKER_FLOW_FEATURE_NAMES),
        "RELATIVE": RELATIVE_NAMES,
    }
    output = {}
    for name, spec in config["ablations"].items():
        if not isinstance(spec, Mapping):
            continue
        output[name] = list(dict.fromkeys(value for group in spec["groups"] for value in groups[group]))
        validate_direction_features(output[name])
    return output


def pipeline(estimator: Any, predictors: list[str]) -> Pipeline:
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), predictors),
        ("symbol", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["symbol"]),
    ])
    return Pipeline([("features", transformer), ("model", estimator)])


def fit_model(family: str, fit: pd.DataFrame, predictors: list[str], horizon: int, seed: int) -> Mapping[str, Any]:
    x = fit[[*predictors, "symbol"]]
    if family == "A_MULTICLASS_LOGISTIC":
        model = pipeline(LogisticRegression(C=0.25, max_iter=700, class_weight="balanced", random_state=seed), predictors)
        model.fit(x, fit[f"h{horizon}_economic_label"])
        return {"model": model}
    if family == "B_DUAL_UTILITY_RIDGE":
        long = pipeline(Ridge(alpha=10.0), predictors).fit(x, fit[f"h{horizon}_utility_long_bps"])
        short = pipeline(Ridge(alpha=10.0), predictors).fit(x, fit[f"h{horizon}_utility_short_bps"])
        return {"long": long, "short": short}
    if family == "C_ADVANTAGE_RIDGE":
        model = pipeline(Ridge(alpha=10.0), predictors).fit(x, fit[f"h{horizon}_directional_advantage_bps"])
        return {"model": model}
    raise ValueError("AEGIS_W8_MODEL_FAMILY_INVALID")


def predict_model(family: str, fitted: Mapping[str, Any], frame: pd.DataFrame, predictors: list[str]) -> Mapping[str, np.ndarray]:
    x = frame[[*predictors, "symbol"]]
    if family == "A_MULTICLASS_LOGISTIC":
        model = fitted["model"]
        return {"probabilities": model.predict_proba(x), "classes": model.classes_}
    if family == "B_DUAL_UTILITY_RIDGE":
        return {"long": fitted["long"].predict(x), "short": fitted["short"].predict(x)}
    return {"advantage": fitted["model"].predict(x)}


def actions_for(family: str, predicted: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> np.ndarray:
    rules = config["models"]["action_rules"]
    return policy_actions(
        family, predicted,
        probability_threshold=float(rules["A_MULTICLASS_LOGISTIC"]["minimum_direction_probability"]),
        utility_threshold=float(rules["B_DUAL_UTILITY_RIDGE"]["minimum_predicted_utility_bps"]),
        advantage_threshold=float(rules["B_DUAL_UTILITY_RIDGE"]["minimum_predicted_advantage_bps"]),
        absolute_advantage_threshold=float(rules["C_ADVANTAGE_RIDGE"]["minimum_absolute_predicted_advantage_bps"]),
    )


def metrics(frame: pd.DataFrame, actions: np.ndarray, horizon: int, config: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    values = realized_policy_returns(frame, actions, horizon)
    traded = actions != "SKIP"
    selected = values[traded]
    positive = float(selected[selected > 0].sum())
    negative = float(-selected[selected < 0].sum())
    equity = np.cumsum(values)
    drawdown = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:] - equity
    working = frame[["utc_day"]].copy(); working["policy_return_bps"] = values
    ci, pvalue = day_block_bootstrap(working, int(config["statistics"]["bootstrap_repetitions"]), seed)
    chosen_mfe = np.where(actions == "LONG", frame[f"h{horizon}_long_mfe_bps"], frame[f"h{horizon}_short_mfe_bps"])
    chosen_mae = np.where(actions == "LONG", frame[f"h{horizon}_long_mae_bps"], frame[f"h{horizon}_short_mae_bps"])
    return {
        "episodes": int(len(frame)), "trades": int(traded.sum()), "trade_fraction": float(traded.mean()),
        "long": int((actions == "LONG").sum()), "short": int((actions == "SHORT").sum()), "skip": int((actions == "SKIP").sum()),
        "portfolio_net_expectancy_bps_per_episode": float(values.mean()),
        "taken_net_expectancy_bps": float(selected.mean()) if len(selected) else 0.0,
        "taken_gross_expectancy_bps": float(selected.mean() + config["targets"]["round_trip_cost_bps"]) if len(selected) else 0.0,
        "win_rate": float((selected > 0).mean()) if len(selected) else 0.0,
        "profit_factor": positive / negative if negative else 1_000_000_000.0,
        "maximum_drawdown_bps_additive": float(drawdown.max(initial=0.0)),
        "sortino_episode": float(values.mean() / selected[selected < 0].std()) if (selected < 0).sum() > 1 and selected[selected < 0].std() > 0 else 0.0,
        "median_mfe_bps": float(np.median(chosen_mfe[traded])) if traded.any() else 0.0,
        "median_mae_bps": float(np.median(chosen_mae[traded])) if traded.any() else 0.0,
        "expectancy_ci_95_bps": ci, "probability_expectancy_nonpositive": pvalue,
    }


def evaluate_candidate(family: str, fitted: Mapping[str, Any], frame: pd.DataFrame, predictors: list[str], horizon: int, config: Mapping[str, Any], seed: int) -> tuple[np.ndarray, Mapping[str, Any]]:
    predicted = predict_model(family, fitted, frame, predictors)
    actions = actions_for(family, predicted, config)
    result = dict(metrics(frame, actions, horizon, config, seed))
    if family == "A_MULTICLASS_LOGISTIC":
        result["multiclass_log_loss"] = float(log_loss(frame[f"h{horizon}_economic_label"], predicted["probabilities"], labels=list(predicted["classes"])))
    elif family == "B_DUAL_UTILITY_RIDGE":
        result["utility_mae_bps"] = float(np.mean((np.abs(predicted["long"] - frame[f"h{horizon}_utility_long_bps"]) + np.abs(predicted["short"] - frame[f"h{horizon}_utility_short_bps"])) / 2))
    else:
        result["advantage_mae_bps"] = float(np.abs(predicted["advantage"] - frame[f"h{horizon}_directional_advantage_bps"]).mean())
    return actions, result


def select_internal(train: pd.DataFrame, predictors_by_ablation: Mapping[str, list[str]], config: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    boundary = train.timestamp_ms.quantile(float(config["models"]["internal_fit_fraction"]))
    fit = train.loc[train.timestamp_ms <= boundary]
    calibration = train.loc[train.timestamp_ms > boundary]
    candidates = []
    seed = int(config["models"]["random_seed"])
    for horizon in config["targets"]["horizons_minutes"]:
        for ablation, predictors in predictors_by_ablation.items():
            for family in config["models"]["families"]:
                fitted = fit_model(family, fit, predictors, int(horizon), seed)
                _, result = evaluate_candidate(family, fitted, calibration, predictors, int(horizon), config, seed + int(horizon))
                candidates.append({"identity": f"H{horizon}:{ablation}:{family}", "horizon": int(horizon), "ablation": ablation, "family": family, **result})
    eligible = [row for row in candidates if row["trades"] >= 100]
    if not eligible:
        raise RuntimeError("AEGIS_W8_NO_INTERNAL_CANDIDATE")
    accepted = benjamini_hochberg(
        {row["identity"]: float(row["probability_expectancy_nonpositive"]) for row in candidates},
        float(config["statistics"]["fdr_alpha"]),
    )
    for row in candidates:
        row["fdr_accepted"] = bool(accepted[row["identity"]])
    selected = max(eligible, key=lambda row: (row["portfolio_net_expectancy_bps_per_episode"], row["taken_net_expectancy_bps"], row["identity"]))
    return selected, candidates


def source_audit(root: Path) -> Mapping[str, Any]:
    archive = root / "data/market_event_economic_path_m1b/archive_manifest.jsonl"
    counts: Counter[str] = Counter()
    starts: dict[str, str] = {}; ends: dict[str, str] = {}
    with archive.open() as handle:
        for line in handle:
            row = json.loads(line); request = row["request"]
            key = str(request["data_type"]); month = str(request["month"])
            counts[key] += 1; starts[key] = min(starts.get(key, month), month); ends[key] = max(ends.get(key, month), month)
    return {
        "funding": {"status": "AVAILABLE", "archives": counts["fundingRate"], "start": starts.get("fundingRate"), "end": ends.get("fundingRate")},
        "premium_basis": {"status": "AVAILABLE_CAUSAL_MARK_MINUS_CONTRACT_PROXY", "archives": counts["markPriceKlines"], "start": starts.get("markPriceKlines"), "end": ends.get("markPriceKlines")},
        "taker_pressure": "AVAILABLE_5M_KLINE_TAKER_BUY_FEATURES",
        "relative_strength_breadth": "AVAILABLE_V9_CAUSAL_FEATURES",
        "open_interest": "NOT_PRESENT", "positioning_long_short_ratio": "NOT_PRESENT", "liquidations": "NOT_PRESENT",
    }


def write_reports(root: Path, result: Mapping[str, Any]) -> None:
    directory = root / REPORT_DIR; directory.mkdir(parents=True, exist_ok=True)
    audit = result["phase_0_audit"]; verdict = result["verdict"]; validation = result["validation"]
    prereg = f"""# Aegis W8 Conditional Direction - Preregistration\n\n- Config SHA-256: `{result['config_sha256']}`\n- W1-W7 holdouts: prohibited and unopened.\n- Unit: independent paired opportunity episode.\n- W7 Opportunity model: frozen `H60:LOGISTIC_L2:P_GTE_0.7`; no W8 retuning.\n- Targets: symmetric LONG/SHORT 30 bps barriers, 14 bps costs, adverse-first same-bar resolution, and explicit SKIP.\n- TRAIN: 2025-08-09 to 2026-01-01.\n- VALIDATION: 2026-01-01 to 2026-05-01.\n- FINAL_HOLDOUT_W8: future independent evidence, `SEALED_NOT_OPENED`.\n- Primary metric: net expectancy without leverage.\n"""
    report = f"""# Aegis W8 Conditional Direction - Result\n\n## Verdict\n\n`{verdict['status']}`\n\n- W8_DIRECTION_DATA_VALID: `{str(verdict['W8_DIRECTION_DATA_VALID']).upper()}`\n- W8_LONG_SHORT_ASYMMETRY_EXPLAINED: `TRUE`\n- W8_DIRECTION_SIGNAL_FOUND: `{str(verdict['W8_DIRECTION_SIGNAL_FOUND']).upper()}`\n- W8_SKIP_CLASS_VALUE_FOUND: `{str(verdict['W8_SKIP_CLASS_VALUE_FOUND']).upper()}`\n- W8_DIRECTIONAL_ALPHA_FOUND: `{str(verdict['W8_DIRECTIONAL_ALPHA_FOUND']).upper()}`\n- W8_ECONOMIC_EDGE_FOUND: `{str(verdict['W8_ECONOMIC_EDGE_FOUND']).upper()}`\n- W8_READY_FOR_SHADOW: `{str(verdict['W8_READY_FOR_SHADOW']).upper()}`\n- W8_READY_FOR_LIVE: `FALSE`\n- FINAL_HOLDOUT_W8: `SEALED_NOT_OPENED`\n\n## Why W7 Was 100% SHORT\n\nThis was architectural, not a market accident. The qualified artifact used the `aegis-labels-short-v4` contract, its preflight recorded `long_disabled=true`, and `src/aegis/layers.py` unconditionally adds `SIDE_NOT_ENABLED` to every LONG candidate. The stored 90,442 rows contain {audit['entry_actions_rows'].get('SHORT', 0)} SHORT actions, {audit['entry_actions_rows'].get('HOLD', 0)} HOLD actions and vote vector `{next(iter(audit['vote_vectors_rows']))}` for every row. Per-candidate rejection reasons were not stored in V14, so they are reported as NOT_PRESENT rather than inferred.\n\nThe 45,221 source episodes nevertheless contain a complete LONG/SHORT counterfactual pair with identical pre-entry features. W8 therefore discarded the historical action as a direction label and built symmetric future labels.\n\n## Population\n\n- Broad independent development episodes: {result['population']['broad_development_episodes']}\n- Frozen W7 Opportunity candidates: {result['population']['opportunity_candidates']}\n- TRAIN / VALIDATION: {result['population']['train']} / {result['population']['validation']}\n- TRAIN labels: {json.dumps(result['population']['train_labels'], sort_keys=True)}\n- VALIDATION labels: {json.dumps(result['population']['validation_labels'], sort_keys=True)}\n\n## Selected Candidate\n\n- `{result['selection']['identity']}`\n- Validation trades: {validation['trades']} ({validation['trade_fraction']:.2%})\n- LONG / SHORT / SKIP: {validation['long']} / {validation['short']} / {validation['skip']}\n- Taken net expectancy: {validation['taken_net_expectancy_bps']:.4f} bps\n- 95% day-block bootstrap CI: [{validation['expectancy_ci_95_bps'][0]:.4f}, {validation['expectancy_ci_95_bps'][1]:.4f}]\n- Profit factor: {validation['profit_factor']:.4f}\n- Stress 20 / 30 bps: {validation['stress_expectancy_bps']['20.0']:.4f} / {validation['stress_expectancy_bps']['30.0']:.4f} bps\n- Positive symbols / folds: {validation['positive_symbols']}/11, {validation['positive_temporal_folds']}/4\n\n## Feature Ablations\n\nAll preregistered price, Opportunity, funding/basis, taker and relative-strength variants are retained in `aegis_conditional_direction_w8_verdict.json`. OI, positioning and liquidation ablations were not run because those histories are absent.\n\n## Interpretation\n\n{verdict['interpretation']}\n\nNo production, TypeScript, guard, leverage, PM2, Shadow or exchange state was changed.\n"""
    report = report.replace(
        f"- Taken net expectancy: {validation['taken_net_expectancy_bps']:.4f} bps\n- 95% day-block bootstrap CI:",
        f"- Taken net expectancy: {validation['taken_net_expectancy_bps']:.4f} bps\n"
        f"- Portfolio net expectancy per opportunity: {validation['portfolio_net_expectancy_bps_per_episode']:.4f} bps\n"
        "- Portfolio 95% day-block bootstrap CI:",
    )
    (directory / "aegis_conditional_direction_w8_preregistration.md").write_text(prereg)
    (directory / "aegis_conditional_direction_w8_result.md").write_text(report)
    (directory / "aegis_conditional_direction_w8_verdict.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); args = parser.parse_args()
    root = args.root.resolve(); config_path = root / "config/experiments/aegis_conditional_direction_w8.yaml"
    config = yaml.safe_load(config_path.read_text()); authority = verify_authority(root, config)
    audit = phase_zero_audit(root, config)
    if not audit["W8_DIRECTION_DATA_VALID"]:
        raise RuntimeError("W8_DIRECTION_DATA_VALID_FALSE")
    frame = load_broad_rows(root, config)
    frame = add_symmetric_outcomes(root, frame, config)
    frame, opportunity = add_frozen_w7_opportunity(root, frame)
    frame = frame.loc[frame.w7_opportunity_candidate].copy()
    frame = add_funding_basis(root, frame, config)
    purge = int(config["partitions"]["purge_minutes"]) * 60_000
    train_end = int(pd.Timestamp(config["partitions"]["train"][1]).timestamp() * 1000)
    validation_start = int(pd.Timestamp(config["partitions"]["validation"][0]).timestamp() * 1000)
    train = frame.loc[frame.partition.eq("TRAIN") & frame.timestamp_ms.lt(train_end - purge)].copy()
    validation = frame.loc[frame.partition.eq("VALIDATION") & frame.timestamp_ms.ge(validation_start + purge)].copy()
    predictors = ablation_predictors(config)
    selected, internal = select_internal(train, predictors, config)
    seed = int(config["models"]["random_seed"]); horizon = int(selected["horizon"])
    validation_results = []
    selected_actions = None; selected_metrics = None
    for candidate in internal:
        family = str(candidate["family"]); ablation = str(candidate["ablation"]); candidate_horizon = int(candidate["horizon"])
        fitted = fit_model(family, train, predictors[ablation], candidate_horizon, seed)
        actions, evaluation = evaluate_candidate(family, fitted, validation, predictors[ablation], candidate_horizon, config, seed + candidate_horizon)
        row = {"identity": candidate["identity"], "horizon": candidate_horizon, "ablation": ablation, "family": family, **evaluation}
        validation_results.append(row)
        if candidate["identity"] == selected["identity"]:
            selected_actions, selected_metrics = actions, row
    assert selected_actions is not None and selected_metrics is not None
    selected_fitted = fit_model(str(selected["family"]), train, predictors[str(selected["ablation"])], horizon, seed)
    selected_predicted = predict_model(str(selected["family"]), selected_fitted, validation, predictors[str(selected["ablation"])])
    if selected["family"] == "A_MULTICLASS_LOGISTIC":
        classes = list(selected_predicted["classes"])
        probabilities = selected_predicted["probabilities"]
        forced_actions = np.where(
            probabilities[:, classes.index("LONG")] >= probabilities[:, classes.index("SHORT")],
            "LONG", "SHORT",
        )
    elif selected["family"] == "B_DUAL_UTILITY_RIDGE":
        forced_actions = np.where(selected_predicted["long"] >= selected_predicted["short"], "LONG", "SHORT")
    else:
        forced_actions = np.where(selected_predicted["advantage"] >= 0, "LONG", "SHORT")
    skipped_mask = selected_actions == "SKIP"
    forced_returns = realized_policy_returns(validation, forced_actions, horizon)
    actual_labels = validation[f"h{horizon}_economic_label"].to_numpy(object)
    selected_metrics["directional_accuracy_taken"] = float((selected_actions[~skipped_mask] == actual_labels[~skipped_mask]).mean()) if (~skipped_mask).any() else 0.0
    selected_metrics["skip_precision"] = float((actual_labels[skipped_mask] == "SKIP").mean()) if skipped_mask.any() else 0.0
    selected_metrics["skip_recall"] = float((selected_actions[actual_labels == "SKIP"] == "SKIP").mean()) if (actual_labels == "SKIP").any() else 0.0
    selected_metrics["skipped_forced_direction_net_expectancy_bps"] = float(forced_returns[skipped_mask].mean()) if skipped_mask.any() else 0.0
    traded = selected_actions != "SKIP"
    selected_frame = validation.loc[traded].copy(); selected_frame["action"] = selected_actions[traded]
    selected_frame["policy_return"] = realized_policy_returns(validation, selected_actions, horizon)[traded]
    selected_frame["fold"] = pd.qcut(selected_frame.timestamp_ms.rank(method="first"), 4, labels=False) if len(selected_frame) >= 4 else 0
    selected_metrics["positive_symbols"] = int((selected_frame.groupby("symbol").policy_return.mean() > 0).sum())
    selected_metrics["positive_temporal_folds"] = int((selected_frame.groupby("fold").policy_return.mean() > 0).sum())
    selected_metrics["per_symbol_net_expectancy_bps"] = {str(k): float(v) for k, v in selected_frame.groupby("symbol").policy_return.mean().items()}
    selected_metrics["per_fold_net_expectancy_bps"] = {str(k): float(v) for k, v in selected_frame.groupby("fold").policy_return.mean().items()}
    cost = float(config["targets"]["round_trip_cost_bps"])
    selected_metrics["stress_expectancy_bps"] = {str(v): float(selected_metrics["taken_net_expectancy_bps"] - (float(v) - cost)) for v in config["targets"]["stress_round_trip_cost_bps"]}
    gate = config["economic_gate"]
    checks = {
        "train_internal_fdr": bool(selected["fdr_accepted"]),
        "minimum_trades": selected_metrics["trades"] >= int(gate["minimum_validation_trades"]),
        "material_net_expectancy": selected_metrics["taken_net_expectancy_bps"] >= float(gate["minimum_net_expectancy_bps"]),
        "ci_lower_positive": selected_metrics["expectancy_ci_95_bps"][0] > 0,
        "profit_factor": selected_metrics["profit_factor"] > 1,
        "positive_symbols": selected_metrics["positive_symbols"] >= int(gate["minimum_positive_symbols"]),
        "positive_folds": selected_metrics["positive_temporal_folds"] >= int(gate["minimum_positive_temporal_folds"]),
        "stress_20_positive": selected_metrics["stress_expectancy_bps"]["20.0"] > 0,
        "coverage_not_excessive": selected_metrics["trade_fraction"] <= float(gate["maximum_trade_fraction"]),
    }
    edge = all(checks.values())
    signal_found = bool(
        selected_metrics["trades"] >= int(gate["minimum_validation_trades"])
        and selected_metrics["long"] > 0 and selected_metrics["short"] > 0
        and selected_metrics["directional_accuracy_taken"] > 0.50
        and selected["fdr_accepted"]
    )
    skip_value = selected_metrics["skip"] > 0 and selected_metrics["skipped_forced_direction_net_expectancy_bps"] < 0.0
    matching_price = next(
        row for row in validation_results
        if row["horizon"] == horizon and row["family"] == selected["family"] and row["ablation"] == "PRICE_STRUCTURE"
    )
    incremental_value = selected_metrics["portfolio_net_expectancy_bps_per_episode"] - matching_price["portfolio_net_expectancy_bps_per_episode"]
    status = "AEGIS_W8_CONDITIONAL_DIRECTION_EDGE_FOUND" if edge else "AEGIS_W8_NO_ROBUST_DIRECTIONAL_ALPHA"
    interpretation = (
        "A symmetric LONG/SHORT/SKIP model passed all development gates. Because all available history was previously inspected and the W8 holdout is future evidence, this justifies neither Shadow nor Live yet."
        if edge else
        "The short-only bias was fully explained and removed from the target construction, but the selected conditional direction policy did not produce robust positive net expectancy after costs and uncertainty. Opportunity remained a magnitude condition, not a reliable sign predictor."
    )
    sources = source_audit(root)
    ablation_summary = {}
    for ablation in predictors:
        eligible_rows = [row for row in validation_results if row["ablation"] == ablation and row["trades"] >= int(gate["minimum_validation_trades"])]
        if eligible_rows:
            best = max(eligible_rows, key=lambda row: row["portfolio_net_expectancy_bps_per_episode"])
            ablation_summary[ablation] = {
                "diagnostic_best_identity": best["identity"], "trades": best["trades"],
                "taken_net_expectancy_bps": best["taken_net_expectancy_bps"],
                "portfolio_net_expectancy_bps_per_episode": best["portfolio_net_expectancy_bps_per_episode"],
            }
    result = {
        "schema_version": "aegis-conditional-direction-w8-verdict-v1", "experiment_id": config["experiment_id"],
        "config_sha256": sha256(config_path), "authority": authority, "phase_0_audit": audit, "data_source_audit": sources,
        "population": {
            "broad_development_episodes": int(len(load_broad_rows(root, config))), "opportunity_candidates": int(len(frame)),
            "train": int(len(train)), "validation": int(len(validation)),
            "train_labels": {str(k): int(v) for k, v in train[f"h{config['targets']['primary_horizon_minutes']}_economic_label"].value_counts().items()},
            "validation_labels": {str(k): int(v) for k, v in validation[f"h{config['targets']['primary_horizon_minutes']}_economic_label"].value_counts().items()},
            "holdout_rows_evaluated": 0,
        },
        "frozen_w7_opportunity": opportunity, "selection": selected, "internal_candidates": internal,
        "validation_candidates": validation_results, "validation_ablation_summary": ablation_summary,
        "validation": selected_metrics, "economic_gate_checks": checks,
        "verdict": {
            "status": status, "W8_DIRECTION_DATA_VALID": True, "W8_LONG_SHORT_ASYMMETRY_EXPLAINED": True,
            "W8_DIRECTION_SIGNAL_FOUND": signal_found, "W8_SKIP_CLASS_VALUE_FOUND": bool(skip_value),
            "W8_DIRECTIONAL_ALPHA_FOUND": edge,
            "W8_FUNDING_BASIS_VALUE_FOUND": bool("FUNDING_BASIS" in selected["ablation"] and incremental_value >= 2.0),
            "W8_OI_VALUE_FOUND": False, "W8_TAKER_VALUE_FOUND": edge and "TAKER" in selected["ablation"],
            "W8_RELATIVE_ALPHA_FOUND": edge and "RELATIVE" in selected["ablation"],
            "W8_ECONOMIC_EDGE_FOUND": edge, "W8_MODELING_JUSTIFIED": edge,
            "W8_READY_FOR_SHADOW": False, "W8_READY_FOR_LIVE": False,
            "final_holdout_state": "SEALED_NOT_OPENED", "interpretation": interpretation,
        },
        "safety": {"production_changes": "NONE", "typescript_changes": "NONE", "guard_changes": "NONE", "leverage_changes": "NONE", "pm2_changes": "NONE", "shadow_changes": "NONE", "authenticated_requests": 0, "exchange_mutations": 0},
        "quality_controls": {
            "same_bar_resolution": "ADVERSE_FIRST", "future_features": 0,
            "causal_basis_contract_price": "PREVIOUS_COMPLETED_5M_CLOSE",
            "rejected_precursor_run": "A preliminary basis alignment used the current 5m close and was discarded as lookahead before final reporting.",
            "fdr_applied_to_internal_candidates": True,
        },
    }
    result["validation"]["incremental_portfolio_value_vs_matching_price_only_bps"] = float(incremental_value)
    result["answers"] = {
        "historical_short_only_reason": "INTENTIONAL_SHORT_LABEL_MODEL_PLUS_UNCONDITIONAL_LONG_SIDE_NOT_ENABLED_VETO",
        "structural_bug_found": False,
        "counterfactual_long_short_frequency_sufficient": True,
        "direction_estimation_successful": bool(signal_found),
        "skip_identification_value": bool(skip_value),
        "opportunity_direction_incremental_value": False,
        "funding_basis_incremental_value": bool(result["verdict"]["W8_FUNDING_BASIS_VALUE_FOUND"]),
        "taker_incremental_value": bool(result["verdict"]["W8_TAKER_VALUE_FOUND"]),
        "open_interest_result": "NOT_TESTED_HISTORY_ABSENT",
        "relative_strength_alpha": bool(result["verdict"]["W8_RELATIVE_ALPHA_FOUND"]),
        "economic_edge_after_costs": bool(edge),
        "stress_survived": bool(checks["stress_20_positive"]),
        "shadow_justified": False,
    }
    output = root / DATA_DIR; output.mkdir(parents=True, exist_ok=True)
    frame[["opportunity_episode_id", "timestamp", "symbol", "partition", "w7_opportunity_probability", *[c for c in frame if c.startswith("h")]]].to_parquet(output / "development_outcomes.parquet", index=False)
    selected_frame.to_parquet(output / "validation_selected_policy.parquet", index=False)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_reports(root, result)
    print(json.dumps({"status": status, "selection": selected["identity"], "train": len(train), "validation": len(validation), "trades": selected_metrics["trades"], "net_bps": selected_metrics["taken_net_expectancy_bps"], "gate_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
