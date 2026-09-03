#!/usr/bin/env python3
"""Train and evaluate the preregistered W10 sequential navigator."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


CLASSES = ("UP_FIRST", "DOWN_FIRST", "NEITHER")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_safe(value.item())
    return value


def feature_columns(frame: pd.DataFrame, ablation: str) -> list[str]:
    price = [column for column in frame if column.startswith("price__")]
    book = [column for column in frame if column.startswith("book__")]
    flow_base = [
        column for column in frame
        if column.startswith("flow__")
        and "response" not in column
        and "absorption" not in column
    ]
    response = [column for column in frame if column.startswith("flow__") and "response" in column]
    absorption = [column for column in frame if column.startswith("flow__") and "absorption" in column]
    groups = {
        "PRICE_ONLY": price,
        "PRICE_FLOW": price + flow_base,
        "PRICE_BOOK_FLOW": price + book + flow_base,
        "FULL": price + book + flow_base + response + absorption,
    }
    return groups[ablation]


def make_model(family: str, seed: int):
    if family == "MULTINOMIAL_LOGISTIC_L2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, max_iter=1_000, class_weight="balanced", random_state=seed),
        )
    if family == "SHALLOW_TREE_DEPTH3":
        return DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=250, class_weight="balanced", random_state=seed
        )
    if family == "HIST_GRADIENT_BOOSTING":
        return HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.05,
            min_samples_leaf=250,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"AEGIS_W10_UNKNOWN_MODEL:{family}")


def fit_model(family: str, ablation: str, frame: pd.DataFrame, seed: int):
    columns = feature_columns(frame, ablation)
    model = make_model(family, seed)
    model.fit(frame[columns], frame["label__b20_h60"])
    return model, columns


def aligned_probabilities(model, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    raw = model.predict_proba(frame[columns])
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    result = np.zeros((len(frame), 3), dtype=float)
    for source, name in enumerate(classes):
        result[:, CLASSES.index(str(name))] = raw[:, source]
    return result


@dataclass(frozen=True)
class Policy:
    entry_threshold: float
    entry_confirmations: int
    hold_threshold: float
    exit_confirmations: int
    cooldown_seconds: int
    max_entries: int

    @property
    def identity(self) -> str:
        return (
            f"E{self.entry_threshold:.2f}x{self.entry_confirmations}-"
            f"H{self.hold_threshold:.2f}-X{self.exit_confirmations}-"
            f"C{self.cooldown_seconds}-M{self.max_entries}"
        )


def policy_family(config: dict[str, Any]) -> list[Policy]:
    search = config["policy_search_train_only"]
    policies = []
    for values in itertools.product(
        search["entry_thresholds"],
        search["entry_confirmations"],
        search["hold_thresholds"],
        search["exit_confirmations"],
        search["cooldown_seconds"],
        search["max_entries_per_episode"],
    ):
        policy = Policy(*values)
        if policy.entry_threshold > policy.hold_threshold:
            policies.append(policy)
    return policies


def _trade_record(
    episode: pd.DataFrame,
    entry_index: int,
    exit_index: int,
    side: str,
    latency_ms: int,
    cost_bps: float,
    exit_reason: str,
) -> dict[str, Any]:
    entry = episode.iloc[entry_index]
    exit_row = episode.iloc[exit_index]
    entry_price = float(entry[f"execution_mid_l{latency_ms}"])
    exit_price = float(exit_row[f"execution_mid_l{latency_ms}"])
    direction = 1.0 if side == "LONG" else -1.0
    gross = direction * (exit_price / entry_price - 1.0) * 10_000.0
    active = episode.iloc[entry_index : exit_index + 1]
    high = float(active["path__interval_high"].max())
    low = float(active["path__interval_low"].min())
    if side == "LONG":
        mfe = max((high / entry_price - 1.0) * 10_000.0, 0.0)
        mae = max((1.0 - low / entry_price) * 10_000.0, 0.0)
    else:
        mfe = max((1.0 - low / entry_price) * 10_000.0, 0.0)
        mae = max((high / entry_price - 1.0) * 10_000.0, 0.0)
    peak_gross = mfe
    remaining = episode.iloc[exit_index:]
    if side == "LONG":
        future_best = (float(remaining["path__interval_high"].max()) / exit_price - 1.0) * 10_000.0
    else:
        future_best = (1.0 - float(remaining["path__interval_low"].min()) / exit_price) * 10_000.0
    observed_spread = (
        float(entry["book__spread_bps"]) + float(exit_row["book__spread_bps"])
    ) / 2.0
    return {
        "momentum_episode_id": str(entry["momentum_episode_id"]),
        "symbol": str(entry["symbol"]),
        "date": str(entry["date"]),
        "side": side,
        "entry_timestamp_us": int(entry["decision_timestamp_us"]),
        "exit_timestamp_us": int(exit_row["decision_timestamp_us"]),
        "holding_seconds": (int(exit_row["decision_timestamp_us"]) - int(entry["decision_timestamp_us"])) / 1_000_000.0,
        "gross_bps": gross,
        "net_bps": gross - cost_bps,
        "mfe_bps": mfe,
        "mae_bps": mae,
        "mfe_mae_ratio": mfe / max(mae, 1e-12),
        "observed_spread_component_bps": observed_spread,
        "residual_cost_assumption_bps": max(cost_bps - observed_spread, 0.0),
        "premature_exit_regret_bps": max(future_best, 0.0),
        "excessive_hold_regret_bps": max(peak_gross - gross, 0.0),
        "exit_reason": exit_reason,
    }


def simulate_policy(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    policy: Policy,
    *,
    latency_ms: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    working = frame.copy()
    working["_up"] = probabilities[:, 0]
    working["_down"] = probabilities[:, 1]
    working["_neither"] = probabilities[:, 2]
    trades: list[dict[str, Any]] = []
    transitions: dict[str, int] = {}
    episode_ids = working["momentum_episode_id"].to_numpy(str)
    boundaries = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(working)]))
    if len(starts) != working["momentum_episode_id"].nunique():
        raise RuntimeError("AEGIS_W10_NONCONTIGUOUS_EPISODES")
    for start, stop in zip(starts, stops):
        episode = working.iloc[int(start) : int(stop)].reset_index(drop=True)
        times = episode["decision_timestamp_us"].to_numpy(np.int64)
        ups = episode["_up"].to_numpy(float)
        downs = episode["_down"].to_numpy(float)
        neithers = episode["_neither"].to_numpy(float)
        state = "FLAT"
        candidate: str | None = None
        confirmation = decay = entries = 0
        cooldown_until = -1
        entry_index = -1
        flip_attempts = 0
        for index in range(len(episode)):
            now = int(times[index])
            up, down, neither = ups[index], downs[index], neithers[index]
            direction = "LONG" if up >= down else "SHORT"
            confidence = max(up, down)
            if state in {"LONG", "SHORT"}:
                own = up if state == "LONG" else down
                opposite = down if state == "LONG" else up
                if opposite >= policy.entry_threshold and opposite > own:
                    flip_attempts += 1
                if opposite >= policy.entry_threshold and opposite > neither:
                    trades.append(_trade_record(episode, entry_index, index, state, latency_ms, cost_bps, "FAST_INVALIDATION"))
                    transitions[f"{state}->COOLDOWN"] = transitions.get(f"{state}->COOLDOWN", 0) + 1
                    state = "COOLDOWN"
                    cooldown_until = now + policy.cooldown_seconds * 1_000_000
                    candidate = None
                    confirmation = decay = 0
                    continue
                decay = decay + 1 if own < policy.hold_threshold else 0
                if decay >= policy.exit_confirmations:
                    trades.append(_trade_record(episode, entry_index, index, state, latency_ms, cost_bps, "CONFIRMED_DECAY"))
                    transitions[f"{state}->COOLDOWN"] = transitions.get(f"{state}->COOLDOWN", 0) + 1
                    state = "COOLDOWN"
                    cooldown_until = now + policy.cooldown_seconds * 1_000_000
                    candidate = None
                    confirmation = decay = 0
                    continue
            elif state == "COOLDOWN":
                if now >= cooldown_until:
                    state = "FLAT"
                    transitions["COOLDOWN->FLAT"] = transitions.get("COOLDOWN->FLAT", 0) + 1
                else:
                    continue
            if state == "FLAT" and entries < policy.max_entries:
                qualified = confidence >= policy.entry_threshold and confidence > neither
                if qualified and direction == candidate:
                    confirmation += 1
                elif qualified:
                    candidate = direction
                    confirmation = 1
                    transitions["FLAT->WATCHING"] = transitions.get("FLAT->WATCHING", 0) + 1
                else:
                    candidate = None
                    confirmation = 0
                if confirmation >= policy.entry_confirmations:
                    state = direction
                    entry_index = index
                    entries += 1
                    transitions[f"WATCHING->{state}"] = transitions.get(f"WATCHING->{state}", 0) + 1
                    candidate = None
                    confirmation = decay = 0
        if state in {"LONG", "SHORT"}:
            trades.append(_trade_record(episode, entry_index, len(episode) - 1, state, latency_ms, cost_bps, "EPISODE_END"))
            transitions[f"{state}->EXITING"] = transitions.get(f"{state}->EXITING", 0) + 1
        transitions["flip_attempts"] = transitions.get("flip_attempts", 0) + flip_attempts
    return pd.DataFrame(trades), transitions


def simulate_baseline(frame: pd.DataFrame, name: str, *, latency_ms: int, cost_bps: float) -> pd.DataFrame:
    if name == "NO_TRADE":
        return pd.DataFrame()
    trades: list[dict[str, Any]] = []
    for _, episode in frame.groupby("momentum_episode_id", sort=False):
        episode = episode.sort_values("decision_timestamp_us").reset_index(drop=True)
        signal = episode[episode["price__return_10s_bps"].abs().ge(10.0) & episode["price__persistence_20s"].ge(0.5)]
        if signal.empty:
            continue
        entry_index = int(signal.index[0])
        side = "LONG" if float(episode.iloc[entry_index]["price__return_10s_bps"]) > 0 else "SHORT"
        if name.endswith("FIXED_30S"):
            exit_index = min(entry_index + 6, len(episode) - 1)
            reason = "FIXED_30S"
        elif name.endswith("FIXED_60S"):
            exit_index = min(entry_index + 12, len(episode) - 1)
            reason = "FIXED_60S"
        else:
            entry_price = float(episode.iloc[entry_index][f"execution_mid_l{latency_ms}"])
            peak = 0.0
            exit_index = len(episode) - 1
            reason = "EPISODE_END"
            for index in range(entry_index + 1, len(episode)):
                price = float(episode.iloc[index][f"execution_mid_l{latency_ms}"])
                current = (price / entry_price - 1.0) * 10_000.0 if side == "LONG" else (1.0 - price / entry_price) * 10_000.0
                peak = max(peak, current)
                if name.endswith("TRAILING_10BPS") and peak - current >= 10.0:
                    exit_index, reason = index, "TRAILING_10BPS"
                    break
                if name.endswith("GIVEBACK_40PCT") and peak >= 10.0 and current <= peak * 0.60:
                    exit_index, reason = index, "GIVEBACK_40PCT"
                    break
        trades.append(_trade_record(episode, entry_index, exit_index, side, latency_ms, cost_bps, reason))
    return pd.DataFrame(trades)


def performance(frame: pd.DataFrame, trades: pd.DataFrame, transitions: dict[str, int] | None = None) -> dict[str, Any]:
    episode_count = int(frame["momentum_episode_id"].nunique())
    symbol_days = int(frame.groupby(["symbol", "date"]).ngroups)
    transitions = transitions or {}
    if trades.empty:
        return {
            "episodes": episode_count, "trades": 0, "long": 0, "short": 0,
            "gross_bps_per_trade": 0.0, "net_bps_per_trade": 0.0,
            "net_bps_per_episode": 0.0, "profit_factor": math.nan,
            "win_rate": math.nan, "median_mfe_bps": math.nan, "median_mae_bps": math.nan,
            "mfe_mae_ratio": math.nan, "maximum_drawdown_bps": 0.0,
            "sortino": math.nan, "median_holding_seconds": math.nan,
            "trades_per_hour": 0.0, "entries_per_episode": 0.0,
            "reentries_per_episode": 0.0, "direct_flips": 0,
            "flip_attempts": int(transitions.get("flip_attempts", 0)),
            "total_turnover_legs": 0, "total_cost_bps": 0.0,
            "cost_to_positive_gross_ratio": math.nan,
            "median_premature_exit_regret_bps": math.nan,
            "median_excessive_hold_regret_bps": math.nan,
        }
    ordered = trades.sort_values("exit_timestamp_us")
    cumulative = np.concatenate(([0.0], ordered["net_bps"].cumsum().to_numpy(float)))
    drawdown = float(np.max(np.maximum.accumulate(cumulative) - cumulative))
    losses = -trades.loc[trades["net_bps"].lt(0), "net_bps"].sum()
    profits = trades.loc[trades["net_bps"].gt(0), "net_bps"].sum()
    downside = trades.loc[trades["net_bps"].lt(0), "net_bps"].to_numpy(float)
    episode_entries = trades.groupby("momentum_episode_id").size()
    positive_gross = float(trades.loc[trades["gross_bps"].gt(0), "gross_bps"].sum())
    total_cost = float((trades["gross_bps"] - trades["net_bps"]).sum())
    return {
        "episodes": episode_count,
        "trades": len(trades),
        "long": int(trades["side"].eq("LONG").sum()),
        "short": int(trades["side"].eq("SHORT").sum()),
        "gross_bps_per_trade": float(trades["gross_bps"].mean()),
        "net_bps_per_trade": float(trades["net_bps"].mean()),
        "net_bps_per_episode": float(trades["net_bps"].sum() / episode_count),
        "profit_factor": float(profits / losses) if losses else math.inf,
        "win_rate": float(trades["net_bps"].gt(0).mean()),
        "median_mfe_bps": float(trades["mfe_bps"].median()),
        "median_mae_bps": float(trades["mae_bps"].median()),
        "mfe_mae_ratio": float(trades["mfe_bps"].mean() / max(trades["mae_bps"].mean(), 1e-12)),
        "maximum_drawdown_bps": drawdown,
        "sortino": float(trades["net_bps"].mean() / downside.std(ddof=0)) if len(downside) > 1 and downside.std(ddof=0) else math.nan,
        "median_holding_seconds": float(trades["holding_seconds"].median()),
        "trades_per_hour": float(len(trades) / max(symbol_days * 24, 1)),
        "entries_per_episode": float(len(trades) / episode_count),
        "reentries_per_episode": float((episode_entries - 1).clip(lower=0).sum() / episode_count),
        "direct_flips": 0,
        "flip_attempts": int(transitions.get("flip_attempts", 0)),
        "total_turnover_legs": int(len(trades) * 2),
        "total_cost_bps": total_cost,
        "cost_to_positive_gross_ratio": total_cost / positive_gross if positive_gross > 0 else math.inf,
        "median_observed_spread_component_bps": float(trades["observed_spread_component_bps"].median()),
        "median_premature_exit_regret_bps": float(trades["premature_exit_regret_bps"].median()),
        "median_excessive_hold_regret_bps": float(trades["excessive_hold_regret_bps"].median()),
    }


def structural_churn_pass(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    gate = config["anti_churn_gate"]
    return bool(
        metrics["trades_per_hour"] <= float(gate["maximum_trades_per_hour"])
        and metrics["reentries_per_episode"] <= float(gate["maximum_reentries_per_episode"])
        and metrics["direct_flips"] == int(gate["required_direct_flips"])
        and metrics["median_holding_seconds"] >= float(gate["minimum_median_holding_seconds"])
    )


def anti_churn_pass(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    ratio = metrics["cost_to_positive_gross_ratio"]
    return bool(
        structural_churn_pass(metrics, config)
        and math.isfinite(ratio)
        and ratio <= float(config["anti_churn_gate"]["maximum_cost_to_positive_gross_ratio"])
    )


def block_bootstrap(frame: pd.DataFrame, trades: pd.DataFrame, repetitions: int, seed: int) -> dict[str, float]:
    episode_meta = frame.groupby("momentum_episode_id", as_index=False).first()[["momentum_episode_id", "symbol", "date"]]
    returns = trades.groupby("momentum_episode_id")["net_bps"].sum() if not trades.empty else pd.Series(dtype=float)
    episode_meta["return"] = episode_meta["momentum_episode_id"].map(returns).fillna(0.0)
    episode_meta["block"] = episode_meta["symbol"] + "|" + episode_meta["date"]
    blocks = episode_meta["block"].unique()
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for index in range(repetitions):
        selected = rng.choice(blocks, size=len(blocks), replace=True)
        sample = pd.concat([episode_meta[episode_meta["block"].eq(block)] for block in selected])
        estimates[index] = float(sample["return"].mean())
    return {
        "repetitions": repetitions,
        "mean_net_bps_per_episode": float(estimates.mean()),
        "ci95_lower": float(np.quantile(estimates, 0.025)),
        "ci95_upper": float(np.quantile(estimates, 0.975)),
        "probability_positive": float(np.mean(estimates > 0.0)),
    }


def validate_dataset(frame: pd.DataFrame, manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    features = feature_columns(frame, "FULL")
    episode_sizes = frame.groupby("momentum_episode_id").size()
    train_months = set(config["partitions"]["train_months"])
    validation_months = set(config["partitions"]["validation_months"])
    checks = {
        "manifest_complete": bool(manifest["all_partitions_pass"] and manifest["completed_parts"] == manifest["expected_parts"]),
        "finite_features": bool(np.isfinite(frame[features].to_numpy(float)).all()),
        "unique_state_keys": not frame.duplicated(["momentum_episode_id", "step"]).any(),
        "states_per_episode": bool((episode_sizes == 25).all()),
        "months_exact": set(frame["date"].str[:7]) == train_months | validation_months,
        "holdout_absent": config["partitions"]["final_holdout"]["state"] == "SEALED_NOT_OPENED",
    }
    if not all(checks.values()):
        raise RuntimeError(f"AEGIS_W10_DATA_INTEGRITY_FAILURE:{checks}")
    episode_meta = frame.groupby("momentum_episode_id", as_index=False).first()
    train_count = int(episode_meta["date"].str[:7].isin(train_months).sum())
    validation_count = int(episode_meta["date"].str[:7].isin(validation_months).sum())
    gate = config["data_gate"]
    checks["minimum_train_episodes"] = train_count >= int(gate["minimum_train_episodes"])
    checks["minimum_validation_episodes"] = validation_count >= int(gate["minimum_validation_episodes"])
    if not all(checks.values()):
        raise RuntimeError(f"AEGIS_W10_DATA_GATE_FAILURE:{checks}")
    return {
        "checks": checks,
        "rows": len(frame),
        "episodes": len(episode_meta),
        "train_episodes": train_count,
        "validation_episodes": validation_count,
        "features": len(features),
        "symbols": sorted(frame["symbol"].unique()),
        "dates": sorted(frame["date"].unique()),
        "w7_covered_episodes": int(episode_meta["w7_opportunity_probability"].notna().sum()),
    }


def evaluate(root: Path) -> dict[str, Any]:
    config_path = root / "config/experiments/aegis_reactive_sequential_momentum_w10.yaml"
    dataset_path = root / "data/reactive_sequential_momentum_w10/states/w10_states.parquet"
    manifest_path = root / "data/reactive_sequential_momentum_w10/states/w10_dataset_manifest.json"
    config = yaml.safe_load(config_path.read_text())
    frame = pd.read_parquet(dataset_path)
    manifest = json.loads(manifest_path.read_text())
    audit = validate_dataset(frame, manifest, config)
    train_months = set(config["partitions"]["train_months"])
    validation_months = set(config["partitions"]["validation_months"])
    train = frame[frame["date"].str[:7].isin(train_months)].copy().reset_index(drop=True)
    validation = frame[frame["date"].str[:7].isin(validation_months)].copy().reset_index(drop=True)
    train_dates = sorted(train["date"].unique())
    fit = train[train["date"].isin(train_dates[:2])].reset_index(drop=True)
    selection = train[train["date"].eq(train_dates[2])].reset_index(drop=True)
    seed = int(config["models"]["random_seed"])
    candidates: list[dict[str, Any]] = []
    policies = policy_family(config)
    for family in config["models"]["families"]:
        for ablation in config["models"]["ablations"]:
            model, columns = fit_model(family, ablation, fit, seed)
            probabilities = aligned_probabilities(model, selection, columns)
            for policy in policies:
                trades, transitions = simulate_policy(selection, probabilities, policy, latency_ms=0, cost_bps=14.0)
                result = performance(selection, trades, transitions)
                candidates.append({
                    "family": family,
                    "ablation": ablation,
                    "policy": policy.__dict__,
                    "policy_identity": policy.identity,
                    "feature_count": len(columns),
                    "structural_churn": structural_churn_pass(result, config),
                    "anti_churn": anti_churn_pass(result, config),
                    **result,
                })
    eligible = [
        item for item in candidates
        if item["structural_churn"] and item["trades"] >= 150 and math.isfinite(item["net_bps_per_episode"])
    ]
    if not eligible:
        raise RuntimeError("AEGIS_W10_NO_TRAIN_POLICY_PASSES_ANTI_CHURN")
    selected = max(eligible, key=lambda item: (item["net_bps_per_episode"], item["net_bps_per_trade"]))
    policy = Policy(**selected["policy"])
    model, columns = fit_model(selected["family"], selected["ablation"], train, seed)
    validation_probabilities = aligned_probabilities(model, validation, columns)
    validation_trades, transitions = simulate_policy(
        validation, validation_probabilities, policy, latency_ms=0, cost_bps=14.0
    )
    validation_metrics = performance(validation, validation_trades, transitions)
    labels = validation["label__b20_h60"].to_numpy(str)
    predicted = np.array(CLASSES)[validation_probabilities.argmax(axis=1)]
    model_information = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "class_distribution": pd.Series(labels).value_counts().to_dict(),
    }
    baselines = {
        name: performance(validation, simulate_baseline(validation, name, latency_ms=0, cost_bps=14.0))
        for name in config["baselines"]
    }
    latency: dict[str, Any] = {}
    for value in config["execution"]["latency_ms"]:
        trades, state_transitions = simulate_policy(
            validation, validation_probabilities, policy, latency_ms=int(value), cost_bps=14.0
        )
        latency[str(value)] = performance(validation, trades, state_transitions)
    cost_stress: dict[str, Any] = {}
    for cost in (14.0, 20.0):
        trades, state_transitions = simulate_policy(
            validation, validation_probabilities, policy, latency_ms=0, cost_bps=cost
        )
        cost_stress[str(int(cost))] = performance(validation, trades, state_transitions)
    per_symbol = {}
    for symbol, group in validation.groupby("symbol"):
        indices = group.index.to_numpy()
        trades, state_transitions = simulate_policy(
            group.reset_index(drop=True), validation_probabilities[indices], policy, latency_ms=0, cost_bps=14.0
        )
        per_symbol[symbol] = performance(group, trades, state_transitions)
    per_date = {}
    for date, group in validation.groupby("date"):
        indices = group.index.to_numpy()
        trades, state_transitions = simulate_policy(
            group.reset_index(drop=True), validation_probabilities[indices], policy, latency_ms=0, cost_bps=14.0
        )
        per_date[date] = performance(group, trades, state_transitions)
    bootstrap = block_bootstrap(validation, validation_trades, 10_000, seed)
    w7_meta = validation.groupby("momentum_episode_id", as_index=False).first()
    w7_train = train.groupby("momentum_episode_id", as_index=False).first()["w7_opportunity_probability"].notna().sum()
    w7_validation = w7_meta["w7_opportunity_probability"].notna().sum()
    w7_gate = config["opportunity_conditioning"]
    w7_diagnostic = {
        "train_episodes": int(w7_train),
        "validation_episodes": int(w7_validation),
        "testable": bool(w7_train >= int(w7_gate["minimum_train_episodes"]) and w7_validation >= int(w7_gate["minimum_validation_episodes"])),
        "executed": False,
        "reason": "INSUFFICIENT_EXISTING_FROZEN_W7_OVERLAP_NO_REFIT",
    }
    positive_symbols = sum(item["net_bps_per_episode"] > 0 for item in per_symbol.values())
    positive_dates = sum(item["net_bps_per_episode"] > 0 for item in per_date.values())
    anti_churn = anti_churn_pass(validation_metrics, config)
    best_baseline = max(item["net_bps_per_episode"] for item in baselines.values())
    gate = config["economic_gate"]
    gate_checks = {
        "minimum_trades": validation_metrics["trades"] >= int(gate["minimum_validation_trades"]),
        "minimum_net_per_trade": validation_metrics["net_bps_per_trade"] >= float(gate["minimum_net_bps_per_trade"]),
        "minimum_net_per_episode": validation_metrics["net_bps_per_episode"] >= float(gate["minimum_net_bps_per_episode"]),
        "profit_factor": validation_metrics["profit_factor"] >= float(gate["minimum_profit_factor"]),
        "bootstrap_ci": bootstrap["ci95_lower"] > 0.0,
        "positive_symbols": positive_symbols >= int(gate["minimum_positive_symbols"]),
        "positive_dates": positive_dates >= int(gate["minimum_positive_validation_dates"]),
        "beats_all_baselines": validation_metrics["net_bps_per_episode"] > best_baseline,
        "cost_20bps": cost_stress["20"]["net_bps_per_episode"] > 0.0,
        "latency_250ms": latency["250"]["net_bps_per_episode"] > 0.0,
        "single_symbol_concentration": max(item["trades"] for item in per_symbol.values()) / max(validation_metrics["trades"], 1) <= float(gate["maximum_single_symbol_trade_share"]),
        "anti_churn": anti_churn,
    }
    passed = all(gate_checks.values())
    baseline_fixed = max(
        baselines["MOMENTUM_10BPS_FIXED_30S"]["net_bps_per_episode"],
        baselines["MOMENTUM_10BPS_FIXED_60S"]["net_bps_per_episode"],
    )
    return {
        "schema_version": "aegis-reactive-sequential-momentum-w10-result-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AEGIS_W10_SEQUENTIAL_MOMENTUM_EDGE_FOUND" if passed else "AEGIS_W10_NO_ROBUST_SEQUENTIAL_MOMENTUM_EDGE",
        "config_sha256": sha256(config_path),
        "dataset_sha256": sha256(dataset_path),
        "data_audit": audit,
        "partitions": {"train_states": len(train), "validation_states": len(validation), "holdout": "SEALED_NOT_OPENED"},
        "selected": selected,
        "model_information": model_information,
        "validation": validation_metrics,
        "state_transitions": transitions,
        "baselines": baselines,
        "latency": latency,
        "cost_stress": cost_stress,
        "per_symbol": per_symbol,
        "per_date": per_date,
        "bootstrap": bootstrap,
        "w7_diagnostic": w7_diagnostic,
        "gate_checks": gate_checks,
        "flags": {
            "W10_MOMENTUM_DETECTION_FOUND": model_information["balanced_accuracy"] >= 0.38,
            "W10_MOMENTUM_PERSISTENCE_FOUND": validation_metrics["gross_bps_per_trade"] > 0.0,
            "W10_DECAY_INFORMATION_FOUND": validation_metrics["net_bps_per_episode"] > baseline_fixed,
            "W10_SEQUENTIAL_POLICY_EDGE_FOUND": passed,
            "W10_ANTI_CHURN_GATE_PASSED": anti_churn,
            "W10_COST_GATE_PASSED": gate_checks["minimum_net_per_trade"] and gate_checks["cost_20bps"],
            "W10_LATENCY_GATE_PASSED": gate_checks["latency_250ms"],
            "W10_MODELING_JUSTIFIED": passed,
            "W10_READY_FOR_SHADOW": passed,
            "W10_READY_FOR_LIVE": False,
        },
        "safety": {
            "production_changes": 0,
            "typescript_changes": 0,
            "authenticated_requests": 0,
            "exchange_mutations": 0,
            "holdout_opened": False,
        },
    }


def render(result: dict[str, Any]) -> str:
    validation = result["validation"]
    baseline_rows = "\n".join(
        f"| {name} | {item['trades']:,} | {item['gross_bps_per_trade']:.3f} | {item['net_bps_per_trade']:.3f} | {item['net_bps_per_episode']:.3f} |"
        for name, item in result["baselines"].items()
    )
    latency_rows = "\n".join(
        f"| {latency} | {item['trades']:,} | {item['net_bps_per_trade']:.3f} | {item['net_bps_per_episode']:.3f} |"
        for latency, item in sorted(result["latency"].items(), key=lambda row: int(row[0]))
    )
    symbol_rows = "\n".join(
        f"| {symbol} | {item['trades']:,} | {item['net_bps_per_trade']:.3f} | {item['net_bps_per_episode']:.3f} |"
        for symbol, item in result["per_symbol"].items()
    )
    date_rows = "\n".join(
        f"| {date} | {item['trades']:,} | {item['net_bps_per_trade']:.3f} | {item['net_bps_per_episode']:.3f} |"
        for date, item in result["per_date"].items()
    )
    gates = "".join(f"- `{name}`: `{str(value).upper()}`\n" for name, value in result["gate_checks"].items())
    flags = "".join(f"- `{name} = {str(value).upper()}`\n" for name, value in result["flags"].items())
    return f"""# Aegis W10 Reactive Sequential Momentum - Result

## Verdict

`{result['status']}`

- TRAIN episodes: {result['data_audit']['train_episodes']:,}.
- VALIDATION episodes: {result['data_audit']['validation_episodes']:,}.
- States: {result['data_audit']['rows']:,}; causal features: {result['data_audit']['features']}.
- FINAL_HOLDOUT_W10: `SEALED_NOT_OPENED`.

## Frozen Navigator

- Model: `{result['selected']['family']}` / `{result['selected']['ablation']}`.
- Policy: `{result['selected']['policy_identity']}`.
- Model balanced accuracy: {result['model_information']['balanced_accuracy']:.4f}.

## Validation

- Trades: {validation['trades']:,}; LONG/SHORT: {validation['long']:,}/{validation['short']:,}.
- Gross/net: {validation['gross_bps_per_trade']:.3f}/{validation['net_bps_per_trade']:.3f} bps per trade.
- Net per episode: {validation['net_bps_per_episode']:.3f} bps.
- Profit factor: {validation['profit_factor']:.4f}; win rate: {validation['win_rate']:.2%}.
- Median MFE/MAE: {validation['median_mfe_bps']:.3f}/{validation['median_mae_bps']:.3f} bps.
- Median hold: {validation['median_holding_seconds']:.1f}s; trades/hour: {validation['trades_per_hour']:.3f}.
- Reentries/episode: {validation['reentries_per_episode']:.4f}; direct flips: {validation['direct_flips']}.
- Cost/positive gross: {validation['cost_to_positive_gross_ratio']:.3f}.
- Bootstrap episode CI: [{result['bootstrap']['ci95_lower']:.3f}, {result['bootstrap']['ci95_upper']:.3f}] bps.

## Baselines

| Policy | Trades | Gross/trade | Net/trade | Net/episode |
|---|---:|---:|---:|---:|
{baseline_rows}

## Latency

| ms | Trades | Net/trade | Net/episode |
|---:|---:|---:|---:|
{latency_rows}

## Symbols

| Symbol | Trades | Net/trade | Net/episode |
|---|---:|---:|---:|
{symbol_rows}

## Validation Dates

| Date | Trades | Net/trade | Net/episode |
|---|---:|---:|---:|
{date_rows}

## Frozen W7 Diagnostic

- TRAIN overlap: {result['w7_diagnostic']['train_episodes']} episodes.
- VALIDATION overlap: {result['w7_diagnostic']['validation_episodes']} episodes.
- Testable without refit: `{str(result['w7_diagnostic']['testable']).upper()}`.
- Executed: `FALSE`; W7 was not refitted.

## Gates

{gates}
## Flags

{flags}
W10 did not modify production, TypeScript, Aegis Brain, guards, leverage, PM2,
Shadow, Live or exchange state.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    result = evaluate(root)
    private = root / "data/reactive_sequential_momentum_w10/run_01"
    report = root / "reports/governance/aegis_prospective_validation/live/reactive_sequential_momentum_w10"
    private.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (private / "aegis_reactive_sequential_momentum_w10_result.json").write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True) + "\n"
    )
    (report / "aegis_reactive_sequential_momentum_w10_result.md").write_text(render(result))
    verdict = {
        "schema_version": "aegis-reactive-sequential-momentum-w10-verdict-v1",
        "status": result["status"],
        "flags": result["flags"],
        "gate_checks": result["gate_checks"],
        "final_holdout": "SEALED_NOT_OPENED",
    }
    (report / "aegis_reactive_sequential_momentum_w10_verdict.json").write_text(
        json.dumps(json_safe(verdict), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(json_safe(verdict), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
