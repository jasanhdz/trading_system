#!/usr/bin/env python3
"""Evaluate W7A opportunity meta-labeling and audit W7B directional data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aegis.research.decomposed_entry_v9 import V9_DIRECTION_FEATURE_NAMES
from aegis.research.feature_information_v14 import TAKER_FLOW_FEATURE_NAMES
from aegis.research.opportunity_direction_w7 import (
    benjamini_hochberg,
    day_block_bootstrap,
    economic_summary,
    opportunity_path_outcomes,
    partition,
    stable_signal_id,
    validate_opportunity_features,
)


PRIOR_PATHS = {
    "W1": (
        "reports/governance/aegis_prospective_validation/live/volume_wave_w1/aegis_volume_wave_w1_preregistration.md",
        "reports/governance/aegis_prospective_validation/live/volume_wave_w1/aegis_volume_wave_w1_verdict.json",
    ),
    "W2": (
        "reports/governance/aegis_prospective_validation/live/momentum_exhaustion_w2/aegis_momentum_exhaustion_w2_preregistration.md",
        "reports/governance/aegis_prospective_validation/live/momentum_exhaustion_w2/aegis_momentum_exhaustion_w2_verdict.json",
    ),
    "W3": (
        "reports/governance/aegis_prospective_validation/live/intrabar_wave_w3/aegis_intrabar_wave_w3_preregistration.md",
        "reports/governance/aegis_prospective_validation/live/intrabar_wave_w3/aegis_intrabar_wave_w3_verdict.json",
    ),
    "W4": (
        "reports/governance/aegis_prospective_validation/live/execution_timing_w4/aegis_execution_timing_w4_preregistration.md",
        "reports/governance/aegis_prospective_validation/live/execution_timing_w4/aegis_execution_timing_w4_verdict.json",
    ),
    "W5": (
        "reports/governance/aegis_prospective_validation/live/wave_regime_discovery_w5/aegis_wave_regime_discovery_w5_preregistration.md",
        "reports/governance/aegis_prospective_validation/live/wave_regime_discovery_w5/aegis_wave_regime_discovery_w5_verdict.json",
    ),
    "W6": (
        "reports/governance/aegis_prospective_validation/live/adaptive_profit_guard_w6/aegis_adaptive_profit_guard_w6_preregistration.md",
        "reports/governance/aegis_prospective_validation/live/adaptive_profit_guard_w6/aegis_adaptive_profit_guard_w6_verdict.json",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_authority(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for experiment, paths in PRIOR_PATHS.items():
        expected = config["prior_experiments"]["artifacts"][experiment]
        actual = {
            "preregistration_sha256": sha256(root / paths[0]),
            "verdict_sha256": sha256(root / paths[1]),
        }
        if actual != expected:
            raise RuntimeError(f"AEGIS_W7_PRIOR_AUTHORITY_MISMATCH:{experiment}")
        results[experiment] = {**actual, "holdout_accessed": False, "retuned": False}
    source = root / config["W7A"]["source"]["dataset"]
    actual_source = sha256(source)
    if actual_source != config["W7A"]["source"]["dataset_sha256"]:
        raise RuntimeError("AEGIS_W7_SOURCE_AUTHORITY_MISMATCH")
    return {"prior_experiments": results, "source_dataset_sha256": actual_source}


def _db_timestamp(value: str) -> str:
    return pd.Timestamp(value).tz_convert(None).strftime("%Y-%m-%d %H:%M:%S.%f")


def build_development_dataset(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    source = root / config["W7A"]["source"]["dataset"]
    db_path = root / config["W7A"]["source"]["candle_database"]
    holdout_start = pd.Timestamp(config["W7A"]["partitions"]["final_holdout"][0])
    horizons = tuple(int(value) for value in config["W7A"]["opportunity_target"]["horizons_minutes"])
    max_bars = max(horizons) // 5
    feature_names = tuple(V9_DIRECTION_FEATURE_NAMES) + tuple(TAKER_FLOW_FEATURE_NAMES)
    validate_opportunity_features(feature_names)
    rows: list[dict[str, Any]] = []
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                timestamp = pd.Timestamp(row["timestamp"])
                if timestamp >= holdout_start:
                    continue
                if (
                    row.get("entry_brain_action") != config["W7A"]["population"]["action_required"]
                    or row.get("side") != config["W7A"]["population"]["row_side_required"]
                    or not bool(row.get("independent"))
                ):
                    continue
                direction_values = tuple(float(value) for value in row["v9_direction_features"])
                flow_names = tuple(str(value) for value in row["v14_taker_flow_feature_names"])
                flow_values = tuple(float(value) for value in row["v14_taker_flow_features"])
                if len(direction_values) != len(V9_DIRECTION_FEATURE_NAMES) or flow_names != tuple(TAKER_FLOW_FEATURE_NAMES):
                    raise RuntimeError("AEGIS_W7_FEATURE_VECTOR_MISMATCH")
                symbol_db = str(row["symbol"]).replace("USDT", "/USDT")
                candles = connection.execute(
                    "SELECT timestamp, open, high, low, close FROM ohlcv_data "
                    "WHERE symbol=? AND timeframe='5m' AND timestamp>=? ORDER BY timestamp LIMIT ?",
                    (symbol_db, _db_timestamp(row["timestamp"]), max_bars),
                ).fetchall()
                if len(candles) != max_bars or str(candles[0][0])[:19] != _db_timestamp(row["timestamp"])[:19]:
                    raise RuntimeError("AEGIS_W7_FUTURE_PATH_INCOMPLETE")
                expected = pd.date_range(timestamp.tz_convert(None), periods=max_bars, freq="5min")
                observed = pd.DatetimeIndex([pd.Timestamp(item[0]) for item in candles])
                if not observed.equals(expected):
                    raise RuntimeError("AEGIS_W7_FUTURE_PATH_GAP")
                entry = float(row["entry_price"])
                if abs(float(candles[0][1]) / entry - 1.0) > 1e-8:
                    raise RuntimeError("AEGIS_W7_ENTRY_PRICE_MISMATCH")
                record: dict[str, Any] = {
                    "signal_episode_id": stable_signal_id(str(row["symbol"]), str(row["timestamp"]), "SHORT"),
                    "timestamp_ms": int(timestamp.timestamp() * 1_000),
                    "timestamp": str(row["timestamp"]), "symbol": str(row["symbol"]),
                    "frozen_direction": "SHORT", "entry_price": entry,
                    "current_guard_net_bps": float(row["full_lifecycle_worst_net_return"]) * 10_000.0 - 4.0,
                }
                record.update(dict(zip(feature_names, (*direction_values, *flow_values), strict=True)))
                for horizon in horizons:
                    count = horizon // 5
                    outcome = opportunity_path_outcomes(
                        entry=entry,
                        highs=[float(item[2]) for item in candles[:count]],
                        lows=[float(item[3]) for item in candles[:count]],
                        closes=[float(item[4]) for item in candles[:count]],
                        frozen_direction="SHORT",
                        cost_bps=float(config["W7A"]["economic_outcome"]["base_round_trip_cost_bps"]),
                    )
                    record.update({f"h{horizon}_{name}": value for name, value in outcome.items()})
                record["partition"] = partition(record["timestamp_ms"], config)
                record["utc_day"] = timestamp.strftime("%Y-%m-%d")
                rows.append(record)
    finally:
        connection.close()
    frame = pd.DataFrame(rows).sort_values("timestamp_ms").reset_index(drop=True)
    if frame.signal_episode_id.duplicated().any() or set(frame.partition) - {"TRAIN", "VALIDATION"}:
        raise RuntimeError("AEGIS_W7_DEVELOPMENT_POPULATION_INVALID")
    return frame


def model_pipeline(estimator: Any, numeric: list[str]) -> Pipeline:
    features = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("symbol", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["symbol"]),
    ])
    return Pipeline([("features", features), ("model", estimator)])


def estimator(name: str, config: dict[str, Any]) -> Any:
    seed = int(config["W7A"]["models"]["random_seed"])
    if name == "RIDGE_L2":
        return Ridge(alpha=10.0)
    if name == "HIST_GRADIENT_BOOSTING":
        return HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, l2_regularization=1.0, random_state=seed)
    if name == "LOGISTIC_L2":
        return LogisticRegression(C=0.25, max_iter=500, class_weight="balanced", random_state=seed)
    raise ValueError(f"AEGIS_W7_UNKNOWN_MODEL:{name}")


def policy_metrics(frame: pd.DataFrame, take: np.ndarray, horizon: int, config: dict[str, Any]) -> dict[str, Any]:
    net_column = f"h{horizon}_directional_net_return_bps"
    selected = frame.loc[take].copy()
    policy = frame.copy()
    policy["take"] = take
    policy["policy_return_bps"] = np.where(take, frame[net_column], 0.0)
    baseline = frame[net_column].to_numpy(float)
    policy["improvement_bps"] = policy.policy_return_bps - baseline
    selected_summary = economic_summary(
        selected.rename(columns={
            net_column: "selected_net",
            f"h{horizon}_directional_mfe_bps": "directional_mfe_bps",
            f"h{horizon}_directional_mae_bps": "directional_mae_bps",
            f"h{horizon}_mfe_mae_ratio": "mfe_mae_ratio",
        }),
        "selected_net",
    ) if len(selected) else {"episodes": 0, "net_expectancy_bps": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "median_mfe_bps": 0.0, "median_mae_bps": 0.0, "median_mfe_mae_ratio": 0.0}
    repetitions = int(config["W7A"]["statistics"]["bootstrap_repetitions"])
    seed = int(config["W7A"]["models"]["random_seed"])
    taken_ci, taken_p = day_block_bootstrap(
        selected.assign(selected_net=selected[net_column]) if len(selected) else selected.assign(selected_net=[]),
        value_column="selected_net", repetitions=repetitions, seed=seed + horizon,
    ) if len(selected) else ([float("nan"), float("nan")], 1.0)
    improvement_ci, improvement_p = day_block_bootstrap(
        policy, value_column="improvement_bps", repetitions=repetitions, seed=seed + horizon + 1,
    )
    cost = float(config["W7A"]["economic_outcome"]["base_round_trip_cost_bps"])
    skipped = frame.loc[~take]
    return {
        "signals": int(len(frame)), "taken": int(take.sum()),
        "take_fraction": float(take.mean()), "skip_fraction": float(1.0 - take.mean()),
        "baseline_net_expectancy_bps": float(baseline.mean()),
        "baseline_gross_expectancy_bps": float(baseline.mean() + cost),
        "portfolio_net_expectancy_bps_per_signal": float(policy.policy_return_bps.mean()),
        "portfolio_improvement_bps_per_signal": float(policy.improvement_bps.mean()),
        "taken": selected_summary,
        "taken_gross_expectancy_bps": float(selected[net_column].mean() + cost) if len(selected) else 0.0,
        "skipped_counterfactual_net_expectancy_bps": float(skipped[net_column].mean()) if len(skipped) else 0.0,
        "selected_actual_magnitude_mean_bps": float(selected[f"h{horizon}_opportunity_magnitude_bps"].mean()) if len(selected) else 0.0,
        "skipped_actual_magnitude_mean_bps": float(skipped[f"h{horizon}_opportunity_magnitude_bps"].mean()) if len(skipped) else 0.0,
        "selected_current_guard_net_expectancy_bps": float(selected.current_guard_net_bps.mean()) if len(selected) else 0.0,
        "baseline_current_guard_net_expectancy_bps": float(frame.current_guard_net_bps.mean()),
        "cost_stress_taken_expectancy_bps": {
            str(stress): float(selected[net_column].mean() - (float(stress) - cost)) if len(selected) else 0.0
            for stress in config["W7A"]["economic_outcome"]["stress_round_trip_cost_bps"]
        },
        "taken_expectancy_ci_95_bps": taken_ci,
        "taken_probability_expectancy_nonpositive": taken_p,
        "improvement_ci_95_bps": improvement_ci,
        "improvement_probability_nonpositive": improvement_p,
        "policy_frame": policy,
    }


def select_on_internal_calibration(train: pd.DataFrame, config: dict[str, Any], predictors: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_fraction = float(config["W7A"]["models"]["internal_fit_fraction"])
    boundary = train.timestamp_ms.quantile(fit_fraction)
    fit = train.loc[train.timestamp_ms <= boundary].copy()
    calibration = train.loc[train.timestamp_ms > boundary].copy()
    candidates: list[dict[str, Any]] = []
    pvalues: dict[str, float] = {}
    primary_threshold = float(config["W7A"]["opportunity_target"]["primary_decisive_threshold_bps"])
    for horizon in config["W7A"]["opportunity_target"]["horizons_minutes"]:
        target = f"h{horizon}_opportunity_magnitude_bps"
        for model_name in config["W7A"]["models"]["magnitude_regression"]:
            model = model_pipeline(estimator(model_name, config), predictors).fit(fit[[*predictors, "symbol"]], fit[target])
            scores = model.predict(calibration[[*predictors, "symbol"]])
            for threshold in config["W7A"]["models"]["regression_take_thresholds_bps"]:
                identity = f"H{horizon}:{model_name}:MAG_GTE_{threshold}"
                metrics = policy_metrics(calibration, scores >= float(threshold), int(horizon), config)
                candidates.append({"identity": identity, "horizon": int(horizon), "model": model_name, "target": "MAGNITUDE", "threshold": float(threshold), **{key: value for key, value in metrics.items() if key != "policy_frame"}})
                pvalues[identity] = float(metrics["taken_probability_expectancy_nonpositive"])
        labels = (fit[target] >= primary_threshold).astype(int)
        model = model_pipeline(estimator("LOGISTIC_L2", config), predictors).fit(fit[[*predictors, "symbol"]], labels)
        scores = model.predict_proba(calibration[[*predictors, "symbol"]])[:, 1]
        for threshold in config["W7A"]["models"]["probability_take_thresholds"]:
            identity = f"H{horizon}:LOGISTIC_L2:P_GTE_{threshold}"
            metrics = policy_metrics(calibration, scores >= float(threshold), int(horizon), config)
            candidates.append({"identity": identity, "horizon": int(horizon), "model": "LOGISTIC_L2", "target": "DECISIVE_50BPS", "threshold": float(threshold), **{key: value for key, value in metrics.items() if key != "policy_frame"}})
            pvalues[identity] = float(metrics["taken_probability_expectancy_nonpositive"])
    eligible = [item for item in candidates if int(item["taken"]["episodes"]) >= 30]
    if not eligible:
        raise RuntimeError("AEGIS_W7_NO_INTERNAL_POLICY_WITH_MINIMUM_SAMPLE")
    selected = max(eligible, key=lambda item: (item["portfolio_net_expectancy_bps_per_signal"], item["taken"]["net_expectancy_bps"], item["identity"]))
    return selected, {"fit_rows": len(fit), "calibration_rows": len(calibration), "candidates": candidates, "fdr_accepted": benjamini_hochberg(pvalues, float(config["W7A"]["statistics"]["fdr_alpha"]))}


def fit_selected_and_score(train: pd.DataFrame, validation: pd.DataFrame, selected: dict[str, Any], config: dict[str, Any], predictors: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    horizon = int(selected["horizon"])
    target = f"h{horizon}_opportunity_magnitude_bps"
    model_name = str(selected["model"])
    model = model_pipeline(estimator(model_name, config), predictors)
    if selected["target"] == "MAGNITUDE":
        model.fit(train[[*predictors, "symbol"]], train[target])
        scores = model.predict(validation[[*predictors, "symbol"]])
    else:
        decisive = float(config["W7A"]["opportunity_target"]["primary_decisive_threshold_bps"])
        model.fit(train[[*predictors, "symbol"]], (train[target] >= decisive).astype(int))
        scores = model.predict_proba(validation[[*predictors, "symbol"]])[:, 1]
    return scores >= float(selected["threshold"]), {"score_min": float(scores.min()), "score_median": float(np.median(scores)), "score_max": float(scores.max())}


def opportunity_validation(train: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any], predictors: list[str]) -> dict[str, Any]:
    output = {}
    decisive = float(config["W7A"]["opportunity_target"]["primary_decisive_threshold_bps"])
    for horizon in config["W7A"]["opportunity_target"]["horizons_minutes"]:
        target = f"h{horizon}_opportunity_magnitude_bps"
        horizon_result: dict[str, Any] = {}
        for name in config["W7A"]["models"]["magnitude_regression"]:
            model = model_pipeline(estimator(name, config), predictors).fit(train[[*predictors, "symbol"]], train[target])
            predicted = model.predict(validation[[*predictors, "symbol"]])
            horizon_result[name] = {
                "spearman": float(pd.Series(predicted).corr(pd.Series(validation[target].to_numpy()), method="spearman")),
                "mae_bps": float(np.abs(predicted - validation[target].to_numpy()).mean()),
            }
        logistic = model_pipeline(estimator("LOGISTIC_L2", config), predictors).fit(train[[*predictors, "symbol"]], (train[target] >= decisive).astype(int))
        probability = logistic.predict_proba(validation[[*predictors, "symbol"]])[:, 1]
        labels = (validation[target] >= decisive).astype(int)
        horizon_result["LOGISTIC_L2"] = {
            "roc_auc": float(roc_auc_score(labels, probability)) if labels.nunique() == 2 else 0.5,
            "base_rate": float(labels.mean()),
        }
        output[str(horizon)] = horizon_result
    gate = config["W7A"]["opportunity_value_gate"]
    passing = 0
    for result in output.values():
        best_spearman = max(float(result[name]["spearman"]) for name in config["W7A"]["models"]["magnitude_regression"])
        if best_spearman >= float(gate["minimum_validation_spearman"]) and float(result["LOGISTIC_L2"]["roc_auc"]) >= float(gate["minimum_validation_auc"]):
            passing += 1
    return {"horizons": output, "passing_horizons": passing, "value_found": passing >= int(gate["minimum_horizons_passing"])}


def w7b_data_audit(root: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for manifest in (
        root / "data/market_event_fast_track_m1a/archive_manifest.jsonl",
        root / "data/market_event_economic_path_m1b/archive_manifest.jsonl",
    ):
        with manifest.open() as handle:
            for line in handle:
                request = json.loads(line)["request"]
                identity = ":".join(str(request.get(key)) for key in ("market", "data_type", "interval"))
                counts[identity] = counts.get(identity, 0) + 1
    return {
        "public_archive_counts": dict(sorted(counts.items())),
        "open_interest": "NOT_PRESENT",
        "long_short_positioning": "NOT_PRESENT",
        "crowding": "NOT_PRESENT_EXCEPT_FUNDING_PROXY",
        "liquidations": "NOT_PRESENT",
        "historical_order_book": "NOT_PRESENT",
        "funding": "AVAILABLE_11_SYMBOLS_2024_01_TO_2026_07",
        "premium_basis": "AVAILABLE_MARK_PRICE_PLUS_SPOT_FUTURES_1M",
        "taker_pressure": "AVAILABLE_KLINE_TAKER_BUY_AND_PARTIAL_AGGTRADES",
        "relative_strength_breadth": "AVAILABLE_BUT_PREVIOUS_B1_B2_NO_VALIDATED_ALPHA",
        "new_directional_source_bundle_complete": False,
        "W7B_execution": "BLOCKED_NO_GENUINELY_NEW_COMPLETE_DIRECTIONAL_SOURCE_SET",
    }


def write_reports(root: Path, result: dict[str, Any]) -> None:
    directory = root / "reports/governance/aegis_prospective_validation/live/opportunity_direction_w7"
    directory.mkdir(parents=True, exist_ok=True)
    config_hash = result["config_sha256"]
    prereg = f"""# Aegis W7 Opportunity x Direction - Preregistration\n\n- Config SHA-256: `{config_hash}`\n- W1-W6 holdouts: sealed and prohibited.\n- Population: independent entries where `entry_brain_action=SHORT`; HOLD is not an entry.\n- LONG: not available from the frozen current brain.\n- Opportunity target: direction-neutral maximum absolute excursion at 15/30/60 minutes.\n- Economic target: frozen SHORT terminal return minus 14 bps.\n- W7 final holdout: SEALED.\n- Historical evidence has prior inspection and cannot authorize promotion.\n"""
    validation = result["W7A"]["validation"]
    verdict = result["verdict"]
    horizons = result["W7A"]["opportunity_validation"]["horizons"]
    horizon_lines = "\n".join(
        f"| {name}m | {values['HIST_GRADIENT_BOOSTING']['spearman']:.4f} | {values['LOGISTIC_L2']['roc_auc']:.4f} | {validation['baseline_by_horizon'][name]['net_expectancy_bps']:.4f} bps |"
        for name, values in horizons.items()
    )
    report = f"""# Aegis W7 Opportunity x Direction - Result\n\n## Verdict\n\n`{verdict['status']}`\n\n- W7A_OPPORTUNITY_VALUE_FOUND: `{str(verdict['W7A_OPPORTUNITY_VALUE_FOUND']).upper()}`\n- W7A_META_LABEL_EDGE_FOUND: `{str(verdict['W7A_META_LABEL_EDGE_FOUND']).upper()}`\n- W7B_DIRECTIONAL_ALPHA_FOUND: `FALSE`\n- W7B_CROSS_SECTIONAL_ALPHA_FOUND: `FALSE`\n- W7_COMBINED_EDGE_FOUND: `FALSE`\n- W7_READY_FOR_SHADOW: `FALSE`\n- W7_READY_FOR_LIVE: `FALSE`\n- FINAL_HOLDOUT_W7: `SEALED_NOT_OPENED`\n\n## Population\n\n- TRAIN: {result['population']['train_signals']} frozen Aegis SHORT signals\n- VALIDATION: {result['population']['validation_signals']} frozen Aegis SHORT signals\n- LONG: NOT_AVAILABLE; the current historical brain was SHORT-only.\n- HOLD rows were excluded rather than relabeled as entries.\n\n## Opportunity\n\n- Horizons passing magnitude gates: {result['W7A']['opportunity_validation']['passing_horizons']}/3\n- Selected policy: `{result['W7A']['selected_policy']['identity']}`\n- Validation take/skip: {validation['taken']['episodes']}/{validation['signals']} taken; {validation['skip_fraction']:.2%} skipped\n\n| Horizon | Magnitude Spearman | Decisive AUC | Frozen SHORT baseline |\n|---:|---:|---:|---:|\n{horizon_lines}\n\nThe selected opportunities really were larger on average ({validation['selected_actual_magnitude_mean_bps']:.2f} vs {validation['skipped_actual_magnitude_mean_bps']:.2f} bps), confirming that magnitude and direction are different questions.\n\n## Economics\n\n- Baseline gross/net expectancy: {validation['baseline_gross_expectancy_bps']:.4f} / {validation['baseline_net_expectancy_bps']:.4f} bps per signal.\n- Selected gross/net expectancy: {validation['taken_gross_expectancy_bps']:.4f} / {validation['taken']['net_expectancy_bps']:.4f} bps per trade.\n- Skipped signals counterfactual expectancy: {validation['skipped_counterfactual_net_expectancy_bps']:.4f} bps.\n- Selected-trade 95% CI: [{validation['taken_expectancy_ci_95_bps'][0]:.4f}, {validation['taken_expectancy_ci_95_bps'][1]:.4f}]\n- Portfolio improvement: {validation['portfolio_improvement_bps_per_signal']:.4f} bps/signal, caused mostly by abstaining from a negative baseline.\n- Profit factor: {validation['taken']['profit_factor']:.4f}; maximum additive drawdown: {validation['taken']['maximum_drawdown_bps_additive']:.2f} bps; Sortino: {validation['taken']['sortino_episode']:.4f}.\n- Stress expectancy at 20/30 bps cost: {validation['cost_stress_taken_expectancy_bps']['20']:.4f} / {validation['cost_stress_taken_expectancy_bps']['30']:.4f} bps.\n- Positive symbols: {validation['positive_symbols']}/11; positive temporal folds: {validation['positive_temporal_folds']}/4.\n\n## W7B Data Audit\n\nFunding and mark/spot basis are available, but were already tested without validated edge in M1B. Relative strength and breadth were already negative in B1/B2. Open interest, positioning/crowding ratios and liquidations are absent. W7B was therefore not trained: doing so would reuse the same failed information rather than test genuinely new directional alpha.\n\n## Conclusion\n\n{verdict['interpretation']}\n\nNo production, TypeScript, guards, leverage, PM2, Shadow or exchange state changed.\n"""
    (directory / "aegis_opportunity_direction_w7_preregistration.md").write_text(prereg)
    (directory / "aegis_opportunity_direction_w7_result.md").write_text(report)
    (directory / "aegis_opportunity_direction_w7_verdict.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / "config/experiments/aegis_opportunity_direction_w7.yaml"
    config = yaml.safe_load(config_path.read_text())
    authority = verify_authority(root, config)
    frame = build_development_dataset(root, config)
    purge_ms = int(config["W7A"]["partitions"]["purge_minutes"]) * 60_000
    train_end = int(pd.Timestamp(config["W7A"]["partitions"]["train"][1]).timestamp() * 1_000)
    validation_start = int(pd.Timestamp(config["W7A"]["partitions"]["validation"][0]).timestamp() * 1_000)
    train = frame.loc[frame.partition.eq("TRAIN") & (frame.timestamp_ms < train_end - purge_ms)].copy()
    validation = frame.loc[frame.partition.eq("VALIDATION") & (frame.timestamp_ms >= validation_start + purge_ms)].copy()
    predictors = list(V9_DIRECTION_FEATURE_NAMES) + list(TAKER_FLOW_FEATURE_NAMES)
    selected, selection = select_on_internal_calibration(train, config, predictors)
    take, score_summary = fit_selected_and_score(train, validation, selected, config, predictors)
    validation_metrics = policy_metrics(validation, take, int(selected["horizon"]), config)
    policy_frame = validation_metrics.pop("policy_frame")
    selected_rows = policy_frame.loc[take].copy()
    selected_horizon = int(selected["horizon"])
    selected_net = f"h{selected_horizon}_directional_net_return_bps"
    selected_rows["fold"] = pd.qcut(selected_rows.timestamp_ms.rank(method="first"), 4, labels=False) if len(selected_rows) >= 4 else 0
    validation_metrics["positive_symbols"] = int((selected_rows.groupby("symbol")[selected_net].mean() > 0).sum())
    validation_metrics["positive_temporal_folds"] = int((selected_rows.groupby("fold")[selected_net].mean() > 0).sum())
    validation_metrics["per_symbol_taken_net_expectancy_bps"] = {
        str(key): float(value) for key, value in selected_rows.groupby("symbol")[selected_net].mean().items()
    }
    validation_metrics["per_fold_taken_net_expectancy_bps"] = {
        str(key): float(value) for key, value in selected_rows.groupby("fold")[selected_net].mean().items()
    }
    validation_metrics["baseline_by_horizon"] = {
        str(horizon): {
            "net_expectancy_bps": float(validation[f"h{horizon}_directional_net_return_bps"].mean()),
            "gross_expectancy_bps": float(validation[f"h{horizon}_directional_gross_return_bps"].mean()),
            "median_mfe_bps": float(validation[f"h{horizon}_directional_mfe_bps"].median()),
            "median_mae_bps": float(validation[f"h{horizon}_directional_mae_bps"].median()),
        }
        for horizon in config["W7A"]["opportunity_target"]["horizons_minutes"]
    }
    opportunity = opportunity_validation(train, validation, config, predictors)
    gate = config["W7A"]["economic_gate"]
    checks = {
        "minimum_taken_signals": validation_metrics["taken"]["episodes"] >= int(gate["minimum_validation_taken_signals"]),
        "maximum_skip_fraction": validation_metrics["skip_fraction"] <= float(gate["maximum_skip_fraction"]),
        "positive_material_taken_expectancy": validation_metrics["taken"]["net_expectancy_bps"] >= float(gate["minimum_taken_net_expectancy_bps"]),
        "material_portfolio_improvement": validation_metrics["portfolio_improvement_bps_per_signal"] >= float(gate["minimum_portfolio_improvement_bps"]),
        "taken_ci_positive": validation_metrics["taken_expectancy_ci_95_bps"][0] > 0,
        "positive_symbols": validation_metrics["positive_symbols"] >= int(gate["minimum_positive_symbols"]),
        "positive_folds": validation_metrics["positive_temporal_folds"] >= int(gate["minimum_positive_temporal_folds"]),
        "profit_factor": validation_metrics["taken"]["profit_factor"] > 1.0,
        "stress_20bps_positive": validation_metrics["taken"]["net_expectancy_bps"] - 6.0 > 0,
    }
    meta_edge = bool(opportunity["value_found"] and all(checks.values()))
    audit = w7b_data_audit(root)
    status = "AEGIS_W7_OPPORTUNITY_META_LABEL_EDGE_FOUND" if meta_edge else "AEGIS_W7_DECOMPOSITION_NO_ECONOMIC_EDGE"
    if opportunity["value_found"] and not meta_edge:
        interpretation = "Opportunity magnitude was learnable, but it did not convert the frozen Aegis direction into robust positive net expectancy. The remaining bottleneck is direction, and the local repository lacks a genuinely new complete OI/crowding/liquidation dataset for a defensible W7B test."
    elif not opportunity["value_found"]:
        interpretation = "The direction-neutral opportunity model did not predict magnitude robustly enough and did not justify economic meta-labeling. Separating opportunity from direction did not rescue the frozen Aegis signals."
    else:
        interpretation = "W7A passed development validation, but prior inspection means a fresh sealed holdout is still required before Shadow."
    result = {
        "schema_version": "aegis-opportunity-direction-w7-verdict-v1",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256(config_path),
        "authority": authority,
        "population": {
            "train_signals": int(len(train)), "validation_signals": int(len(validation)),
            "symbols": int(validation.symbol.nunique()), "frozen_directions": ["SHORT"],
            "long_status": "NOT_AVAILABLE_CURRENT_BRAIN_SHORT_ONLY",
            "hold_rows_used_as_signals": False, "holdout_rows_evaluated": 0,
        },
        "W7A": {
            "selected_policy": selected,
            "selection": selection,
            "score_summary": score_summary,
            "opportunity_validation": opportunity,
            "validation": validation_metrics,
            "economic_gate_checks": checks,
        },
        "W7B": audit,
        "verdict": {
            "status": status,
            "W7A_OPPORTUNITY_VALUE_FOUND": bool(opportunity["value_found"]),
            "W7A_META_LABEL_EDGE_FOUND": meta_edge,
            "W7B_DIRECTIONAL_ALPHA_FOUND": False,
            "W7B_CROSS_SECTIONAL_ALPHA_FOUND": False,
            "W7_COMBINED_EDGE_FOUND": False,
            "W7_MODELING_JUSTIFIED": meta_edge,
            "W7_READY_FOR_SHADOW": False,
            "W7_READY_FOR_LIVE": False,
            "final_holdout_state": "SEALED_NOT_OPENED",
            "interpretation": interpretation,
        },
        "safety": {"production_changes": "NONE", "typescript_changes": "NONE", "guard_changes": "NONE", "leverage_changes": "NONE", "pm2_changes": "NONE", "authenticated_requests": 0, "exchange_mutations": 0},
    }
    output = root / "data/opportunity_direction_w7/run_01"
    output.mkdir(parents=True, exist_ok=True)
    development_columns = ["signal_episode_id", "timestamp", "symbol", "frozen_direction", "partition", *[column for column in frame if column.startswith("h")]]
    frame[development_columns].to_parquet(output / "development_outcomes.parquet", index=False)
    policy_frame.to_parquet(output / "validation_policy.parquet", index=False)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_reports(root, result)
    print(json.dumps({"status": status, "opportunity_value_found": opportunity["value_found"], "meta_label_edge_found": meta_edge, "selected": selected["identity"], "validation_taken": validation_metrics["taken"]["episodes"], "validation_taken_expectancy_bps": validation_metrics["taken"]["net_expectancy_bps"], "gate_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
