"""Causal orchestration for the frozen W11 ephemeral-regime experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .modeling import (
    CandidateModel,
    ExpirationGuardian,
    FrozenExpert,
    ResolvedOutcome,
    SimilarityModel,
    ValidationCandidate,
    fit_regime_similarity,
    make_instance_id,
    select_validation_candidate,
    temporal_block_bootstrap,
    train_candidate,
)

MODEL_VERSION = "w11-frozen-v1"
AGE_BUCKETS = ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 48))
FRAME_NAMES = (
    "candidate_evaluations",
    "instances",
    "decisions",
    "trades",
    "ttl_sensitivity",
    "half_life",
    "baselines",
    "expiration_registry",
)
EMPTY_COLUMNS = {
    "instances": ("partition", "instance_id", "model_version", "sequence", "window_hours", "horizon_minutes", "training_start", "training_end", "validation_start", "validation_end", "created_at", "expires_at", "similarity_method", "similarity_threshold", "validation_stress_edge_bps", "expected_edge_bps"),
    "decisions": ("decision_id", "decision_at", "model_family", "model_version", "model_instance_id", "training_window", "validation_window", "created_at", "expires_at", "regime_similarity_at_decision", "expected_edge_bps", "expected_direction", "horizon_minutes", "decision", "reason", "confidence", "symbol", "expiration_reason", "partition", "mode"),
    "trades": ("partition", "mode", "decision_id", "model_instance_id", "model_version", "symbol", "side", "opened_at", "resolved_at", "outcome_available_at", "horizon_minutes", "age_hours", "similarity", "gross_bps", "fee_bps", "baseline_slippage_bps", "baseline_cost_bps", "stress_cost_bps", "severe_cost_bps", "net14_bps", "net20_bps", "net30_bps", "mfe_bps", "mae_bps"),
    "ttl_sensitivity": ("partition", "instance_id", "window_hours", "horizon_minutes", "ttl_hours", "trade_count", "gross_mean_bps", "net14_mean_bps", "net20_mean_bps", "net30_mean_bps", "descriptive_only"),
    "half_life": ("partition", "instance_id", "initial_gross_bps", "initial_net14_bps", "bucket_0_1h_net14_bps", "bucket_1_3h_net14_bps", "bucket_3_6h_net14_bps", "bucket_6_12h_net14_bps", "bucket_12_24h_net14_bps", "bucket_24_48h_net14_bps", "half_life_hours", "half_life_censored", "economic_lifetime_hours", "similarity_net_spearman", "high_similarity_net14_bps", "low_similarity_net14_bps"),
    "baselines": ("strategy", "signal_count", "trade_count", "gross_per_trade_bps", "net14_per_trade_bps", "net20_per_trade_bps", "net30_per_trade_bps", "gross_per_signal_bps", "net14_per_signal_bps", "net20_per_signal_bps", "net30_per_signal_bps"),
    "expiration_registry": ("partition", "mode", "instance_id", "expired_at", "reason"),
}


@dataclass
class CreatedInstance:
    expert: FrozenExpert
    model: CandidateModel
    similarity: SimilarityModel
    validation_stress_edge: float
    expected_edge_bps: float
    partition: str


@dataclass
class ExperimentResult:
    candidate_evaluations: pd.DataFrame
    instances: pd.DataFrame
    decisions: pd.DataFrame
    trades: pd.DataFrame
    ttl_sensitivity: pd.DataFrame
    half_life: pd.DataFrame
    baselines: pd.DataFrame
    expiration_registry: pd.DataFrame
    summary: dict[str, Any]
    data_start: pd.Timestamp
    data_end: pd.Timestamp


def _utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _flat_panel(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex) or list(panel.index.names) != ["decision_at", "symbol"]:
        raise ValueError("panel must use the (decision_at, symbol) index from data.build_data_panel")
    work = panel.reset_index().copy()
    work["decision_at"] = pd.to_datetime(work["decision_at"], utc=True)
    work["outcome_available_at"] = pd.to_datetime(work["outcome_available_at"], utc=True)
    expected = work.sort_values(["decision_at", "symbol"], kind="mergesort", ignore_index=True)
    if not work[["decision_at", "symbol"]].reset_index(drop=True).equals(expected[["decision_at", "symbol"]]):
        raise ValueError("panel must be sorted by decision_at and symbol")
    missing = set(features).difference(work.columns)
    if missing or len(features) != 20:
        raise ValueError(f"panel does not contain the exact frozen feature set: {sorted(missing)}")
    return work


def _bounds(created_at: pd.Timestamp, window_hours: int) -> tuple[pd.Timestamp, ...]:
    validation_start = created_at - pd.Timedelta(hours=7)
    validation_end = created_at - pd.Timedelta(hours=1)
    training_end = validation_start - pd.Timedelta(minutes=60)
    training_start = training_end - pd.Timedelta(hours=window_hours)
    return training_start, training_end, validation_start, validation_end


def _oriented(decision: str, gross: float) -> float:
    return gross if decision == "LONG" else -gross


def _candidate_at(
    work: pd.DataFrame,
    created_at: pd.Timestamp,
    partition: str,
    cfg: Mapping[str, Any],
    sequence: int,
) -> tuple[list[dict[str, Any]], CreatedInstance | None]:
    features = list(cfg["features"]["names"])
    costs = cfg["costs_bps"]
    evaluations: list[dict[str, Any]] = []
    selectable: list[ValidationCandidate] = []
    payloads: dict[tuple[int, int], tuple[CandidateModel, SimilarityModel, float]] = {}

    for window in cfg["experts"]["training_window_hours"]:
        train_start, train_end, validation_start, validation_end = _bounds(created_at, int(window))
        train = work[
            work["decision_at"].ge(train_start)
            & work["decision_at"].lt(train_end)
            & work["outcome_available_at"].le(created_at)
        ]
        validation = work[
            work["decision_at"].ge(validation_start)
            & work["decision_at"].lt(validation_end)
            & work["outcome_available_at"].le(created_at)
        ]
        for horizon in cfg["experts"]["horizons_minutes"]:
            target = f"gross_target_{horizon}m_bps"
            usable_train = train.dropna(subset=[target])
            gross = usable_train[target].to_numpy(float)
            opportunity = np.abs(gross) >= float(cfg["targets"]["opportunity_min_gross_bps"])
            expected_edge = (
                float(np.median(np.abs(gross[opportunity]))) - float(costs["baseline"])
                if opportunity.any()
                else -float(costs["baseline"])
            )
            model = train_candidate(
                usable_train[features], opportunity, gross > 0, expected_edge_bps=expected_edge,
                seed=int(cfg["seed"]),
            )
            base = {
                "partition": partition,
                "created_at": created_at,
                "window_hours": int(window),
                "horizon_minutes": int(horizon),
                "training_start": train_start,
                "training_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "training_rows": len(usable_train),
                "validation_rows": len(validation),
                "maximum_training_outcome_available_at": usable_train["outcome_available_at"].max() if len(usable_train) else pd.NaT,
                "expected_edge_bps": expected_edge,
                "model_active": model.active,
                "failure_reason": model.failure_reason,
                "validation_trades": 0,
                "validation_symbols": 0,
                "maximum_symbol_fraction": 0.0,
                "validation_gross_mean_bps": None,
                "validation_baseline_net_bps": None,
                "validation_stress_net_bps": None,
                "bootstrap_probability_positive": None,
                "similarity_method": None,
                "eligible": False,
                "selected": False,
            }
            if not model.active:
                evaluations.append(base)
                continue
            predictions = model.predict(validation[features])
            chosen = np.array([item.decision != "SKIP" for item in predictions])
            predicted = validation.loc[chosen].copy()
            selected_predictions = [item for item, keep in zip(predictions, chosen, strict=True) if keep]
            if predicted.empty:
                base["failure_reason"] = "NO_VALIDATION_TRADES"
                evaluations.append(base)
                continue
            predicted["oriented_gross"] = [
                _oriented(item.decision, value)
                for item, value in zip(selected_predictions, predicted[target].to_numpy(float), strict=True)
            ]
            predicted = predicted.dropna(subset=["oriented_gross"])
            if predicted.empty:
                base["failure_reason"] = "NO_RESOLVED_VALIDATION_TRADES"
                evaluations.append(base)
                continue
            predicted["net14"] = predicted["oriented_gross"] - float(costs["baseline"])
            predicted["net20"] = predicted["oriented_gross"] - float(costs["stress"])
            counts = predicted["symbol"].value_counts().sort_index().to_dict()
            bootstrap = temporal_block_bootstrap(
                predicted["decision_at"], predicted["net14"],
                block_hours=int(cfg["activation"]["bootstrap_block_hours"]),
                draws=int(cfg["activation"]["bootstrap_draws"]), seed=int(cfg["seed"]),
            )
            train_predictions = model.predict(usable_train[features])
            train_trade = usable_train.loc[[item.decision != "SKIP" for item in train_predictions]]
            if len(train_trade) < 4 or len(predicted) < 2:
                base["failure_reason"] = "INSUFFICIENT_SIMILARITY_ROWS"
                evaluations.append(base)
                continue
            similarity = fit_regime_similarity(train_trade[features], predicted[features], predicted["net14"])
            candidate = ValidationCandidate(
                int(window), int(horizon), float(predicted["net20"].mean()),
                float(predicted["net14"].mean()), len(predicted), counts,
                bootstrap.probability_positive,
            )
            base.update({
                "validation_trades": len(predicted), "validation_symbols": len(counts),
                "maximum_symbol_fraction": max(counts.values()) / len(predicted),
                "validation_gross_mean_bps": float(predicted["oriented_gross"].mean()),
                "validation_baseline_net_bps": candidate.baseline_net_bps,
                "validation_stress_net_bps": candidate.stress_net_bps,
                "bootstrap_probability_positive": candidate.bootstrap_probability_positive,
                "similarity_method": similarity.method, "eligible": candidate.eligible,
                "failure_reason": None if candidate.eligible else "ACTIVATION_GATES",
            })
            evaluations.append(base)
            selectable.append(candidate)
            payloads[(int(window), int(horizon))] = (model, similarity, expected_edge)

    selected = select_validation_candidate(selectable)
    if selected is None:
        return evaluations, None
    key = (selected.window_hours, selected.horizon_minutes)
    model, similarity, expected_edge = payloads[key]
    for row in evaluations:
        if row["window_hours"] == key[0] and row["horizon_minutes"] == key[1]:
            row["selected"] = True
    train_start, train_end, validation_start, validation_end = _bounds(created_at, key[0])
    ttl = int(cfg["experts"]["primary_ttl_by_window_hours"][str(key[0])])
    expert = FrozenExpert(
        instance_id=make_instance_id(created_at, key[0], key[1], sequence),
        model_version=MODEL_VERSION, window_hours=key[0], horizon_minutes=key[1], sequence=sequence,
        training_start=train_start.to_pydatetime(), training_end=train_end.to_pydatetime(),
        validation_start=validation_start.to_pydatetime(), validation_end=validation_end.to_pydatetime(),
        created_at=created_at.to_pydatetime(), expires_at=(created_at + pd.Timedelta(hours=ttl)).to_pydatetime(),
        similarity_method=similarity.method, similarity_threshold=similarity.threshold,
    )
    return evaluations, CreatedInstance(
        expert, model, similarity, selected.stress_net_bps, expected_edge, partition
    )


def create_candidates(
    work: pd.DataFrame, cfg: Mapping[str, Any], partitions: tuple[str, ...] = ("validation", "prospective")
) -> tuple[pd.DataFrame, list[CreatedInstance]]:
    rows: list[dict[str, Any]] = []
    instances: list[CreatedInstance] = []
    sequence = 0
    cadence = int(cfg["experts"]["creation_cadence_hours"])
    for partition in partitions:
        start, end = (_utc(value) for value in cfg["partitions"][partition])
        for created_at in pd.date_range(start, end, freq=f"{cadence}h", inclusive="left"):
            evaluations, selected = _candidate_at(work, created_at, partition.upper(), cfg, sequence + 1)
            rows.extend(evaluations)
            if selected is not None:
                sequence += 1
                instances.append(selected)
    return pd.DataFrame(rows), instances


def _instance_row(instance: CreatedInstance) -> dict[str, Any]:
    expert = instance.expert
    return {
        "partition": instance.partition, "instance_id": expert.instance_id,
        "model_version": expert.model_version, "sequence": expert.sequence,
        "window_hours": expert.window_hours, "horizon_minutes": expert.horizon_minutes,
        "training_start": expert.training_start, "training_end": expert.training_end,
        "validation_start": expert.validation_start, "validation_end": expert.validation_end,
        "created_at": expert.created_at, "expires_at": expert.expires_at,
        "similarity_method": expert.similarity_method,
        "similarity_threshold": expert.similarity_threshold,
        "validation_stress_edge_bps": instance.validation_stress_edge,
        "expected_edge_bps": instance.expected_edge_bps,
    }


def _active_choice(instances: list[CreatedInstance]) -> CreatedInstance | None:
    if not instances:
        return None
    return sorted(instances, key=lambda item: (
        -item.validation_stress_edge, item.expert.window_hours, item.expert.horizon_minutes,
        item.expert.sequence,
    ))[0]


def _similarity(instance: CreatedInstance, rows: pd.DataFrame, features: list[str]) -> np.ndarray:
    return instance.similarity.score(rows[features].to_numpy(float))


def replay_partition(
    work: pd.DataFrame,
    instances: list[CreatedInstance],
    cfg: Mapping[str, Any],
    partition: str,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay selected immutable instances with FULL or TTL_ONLY expiration."""
    start, end = (_utc(value) for value in cfg["partitions"][partition])
    relevant = sorted(
        [item for item in instances if item.partition == partition.upper()],
        key=lambda item: item.expert.created_at,
    )
    features = list(cfg["features"]["names"])
    by_time = {stamp: frame for stamp, frame in work[
        work["decision_at"].ge(start) & work["decision_at"].lt(end)
    ].groupby("decision_at", sort=True)}
    guardian = ExpirationGuardian()
    available: list[CreatedInstance] = []
    added: set[str] = set()
    outcomes: dict[str, list[ResolvedOutcome]] = {}
    busy: dict[str, pd.Timestamp] = {}
    decision_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    expiration_rows: list[dict[str, Any]] = []
    creation_cursor = 0

    for now, rows in by_time.items():
        now = _utc(now)
        while creation_cursor < len(relevant) and _utc(relevant[creation_cursor].expert.created_at) <= now:
            item = relevant[creation_cursor]
            if _utc(item.expert.created_at) == now:
                guardian.add(item.expert)
                available.append(item)
                added.add(item.expert.instance_id)
                outcomes[item.expert.instance_id] = []
            creation_cursor += 1
        for symbol in list(busy):
            if busy[symbol] <= now:
                del busy[symbol]

        active: list[CreatedInstance] = []
        for item in available:
            expert = item.expert
            if expert.instance_id not in added or guardian.registry.get(expert.instance_id) is not None:
                continue
            scores = _similarity(item, rows, features)
            median_similarity = float(np.median(scores[np.isfinite(scores)])) if np.isfinite(scores).any() else -np.inf
            if mode == "FULL":
                expiration = guardian.evaluate(
                    expert, now, similarity=median_similarity,
                    resolved_outcomes=outcomes[expert.instance_id],
                )
            else:
                expiration = guardian.evaluate(expert, now, similarity=max(median_similarity, expert.similarity_threshold)) if now >= _utc(expert.expires_at) else None
            if expiration is not None:
                expiration_rows.append({
                    "partition": partition.upper(), "mode": mode, "instance_id": expert.instance_id,
                    "expired_at": expiration.expired_at, "reason": expiration.reason,
                })
            elif now < _utc(expert.expires_at):
                active.append(item)
        chosen = _active_choice(active)
        if chosen is None:
            continue
        expert = chosen.expert
        predictions = chosen.model.predict(rows[features])
        similarities = _similarity(chosen, rows, features)
        target = f"gross_target_{expert.horizon_minutes}m_bps"
        for (_, row), prediction, similarity in zip(rows.iterrows(), predictions, similarities, strict=True):
            symbol = str(row["symbol"])
            decision = prediction.decision
            reason = "MODEL_THRESHOLD" if decision != "SKIP" else "MODEL_SKIP"
            if symbol in busy and decision != "SKIP":
                decision, reason = "SKIP", "SKIP_SYMBOL_BUSY"
            decision_id = f"{mode}_{now.strftime('%Y%m%dT%H%M%SZ')}_{symbol}"
            attribution = expert.attribution(
                decision_id=decision_id, decision_at=now, similarity=float(similarity),
                expected_edge_bps=prediction.expected_edge_bps, decision=decision, reason=reason,
                symbol=symbol,
                confidence=max(
                    prediction.opportunity_probability,
                    prediction.long_probability,
                    1.0 - prediction.long_probability,
                ),
            )
            attribution.update({"partition": partition.upper(), "mode": mode})
            decision_rows.append(attribution)
            if decision == "SKIP" or pd.isna(row[target]):
                continue
            gross = _oriented(decision, float(row[target]))
            resolved_at = now + pd.Timedelta(minutes=expert.horizon_minutes)
            busy[symbol] = resolved_at
            trade = {
                "partition": partition.upper(), "mode": mode, "decision_id": decision_id,
                "model_instance_id": expert.instance_id, "model_version": expert.model_version,
                "symbol": symbol, "side": decision, "opened_at": now, "resolved_at": resolved_at,
                "outcome_available_at": resolved_at,
                "horizon_minutes": expert.horizon_minutes,
                "age_hours": (now - _utc(expert.created_at)).total_seconds() / 3600,
                "similarity": float(similarity), "gross_bps": gross, "fee_bps": 10.0,
                "baseline_slippage_bps": 4.0, "baseline_cost_bps": 14.0,
                "stress_cost_bps": 20.0, "severe_cost_bps": 30.0,
                "net14_bps": gross - 14.0, "net20_bps": gross - 20.0,
                "net30_bps": gross - 30.0, "mfe_bps": None, "mae_bps": None,
            }
            trades.append(trade)
            outcomes[expert.instance_id].append(ResolvedOutcome(now.to_pydatetime(), resolved_at.to_pydatetime(), gross - 14.0))
    return pd.DataFrame(decision_rows), pd.DataFrame(trades), pd.DataFrame(expiration_rows)


def _independent_instance_trades(
    work: pd.DataFrame, instance: CreatedInstance, cfg: Mapping[str, Any], ttl_hours: int
) -> list[dict[str, Any]]:
    expert = instance.expert
    start, end = _utc(expert.created_at), _utc(expert.created_at) + pd.Timedelta(hours=ttl_hours)
    features = list(cfg["features"]["names"])
    sample = work[work["decision_at"].ge(start) & work["decision_at"].lt(end)]
    busy: dict[str, pd.Timestamp] = {}
    output: list[dict[str, Any]] = []
    target = f"gross_target_{expert.horizon_minutes}m_bps"
    for now, rows in sample.groupby("decision_at", sort=True):
        now = _utc(now)
        for symbol in list(busy):
            if busy[symbol] <= now:
                del busy[symbol]
        predictions = instance.model.predict(rows[features])
        similarities = _similarity(instance, rows, features)
        for (_, row), prediction, similarity in zip(rows.iterrows(), predictions, similarities, strict=True):
            symbol = str(row["symbol"])
            if prediction.decision == "SKIP" or symbol in busy or pd.isna(row[target]):
                continue
            gross = _oriented(prediction.decision, float(row[target]))
            busy[symbol] = now + pd.Timedelta(minutes=expert.horizon_minutes)
            output.append({
                "instance_id": expert.instance_id, "partition": instance.partition,
                "ttl_hours": ttl_hours, "symbol": symbol, "side": prediction.decision,
                "opened_at": now, "resolved_at": busy[symbol],
                "age_hours": (now - start).total_seconds() / 3600,
                "similarity": float(similarity), "gross_bps": gross,
                "net14_bps": gross - 14.0, "net20_bps": gross - 20.0, "net30_bps": gross - 30.0,
            })
    return output


def ttl_sensitivity(
    work: pd.DataFrame, instances: list[CreatedInstance], cfg: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    max_age_trades: list[dict[str, Any]] = []
    ttls = [int(value) for value in cfg["experts"]["diagnostic_ttl_hours"]]
    for instance in instances:
        for ttl in ttls:
            trades = _independent_instance_trades(work, instance, cfg, ttl)
            if ttl == max(ttls):
                max_age_trades.extend(trades)
            rows.append({
                "partition": instance.partition, "instance_id": instance.expert.instance_id,
                "window_hours": instance.expert.window_hours,
                "horizon_minutes": instance.expert.horizon_minutes, "ttl_hours": ttl,
                "trade_count": len(trades),
                "gross_mean_bps": float(np.mean([x["gross_bps"] for x in trades])) if trades else None,
                "net14_mean_bps": float(np.mean([x["net14_bps"] for x in trades])) if trades else None,
                "net20_mean_bps": float(np.mean([x["net20_bps"] for x in trades])) if trades else None,
                "net30_mean_bps": float(np.mean([x["net30_bps"] for x in trades])) if trades else None,
                "descriptive_only": True,
            })
    return pd.DataFrame(rows), pd.DataFrame(max_age_trades)


def half_life_diagnostics(diagnostic_trades: pd.DataFrame, instances: list[CreatedInstance]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for instance in instances:
        trades = diagnostic_trades[diagnostic_trades["instance_id"].eq(instance.expert.instance_id)] if not diagnostic_trades.empty else diagnostic_trades
        bucket_edges: list[float | None] = []
        bucket_gross: list[float | None] = []
        for low, high in AGE_BUCKETS:
            bucket = trades[trades["age_hours"].ge(low) & trades["age_hours"].lt(high)]
            bucket_gross.append(float(bucket["gross_bps"].mean()) if len(bucket) else None)
            bucket_edges.append(float(bucket["net14_bps"].mean()) if len(bucket) else None)
        initial_index = next((i for i, value in enumerate(bucket_edges) if value is not None), None)
        initial_net = bucket_edges[initial_index] if initial_index is not None else None
        initial_gross = bucket_gross[initial_index] if initial_index is not None else None
        half_life = None
        if initial_index is not None and initial_net is not None and initial_net > 0:
            half_life = next((AGE_BUCKETS[i][1] for i in range(initial_index + 1, len(AGE_BUCKETS)) if bucket_edges[i] is not None and bucket_edges[i] <= initial_net * 0.5), None)
        economic_lifetime = None
        for _, high in AGE_BUCKETS:
            cumulative = trades[trades["age_hours"].lt(high)]
            if len(cumulative) and float(cumulative["net14_bps"].mean()) <= 0:
                economic_lifetime = high
                break
        valid = trades.dropna(subset=["similarity", "net14_bps"])
        relationship = valid["similarity"].corr(valid["net14_bps"], method="spearman") if len(valid) >= 2 else np.nan
        median = valid["similarity"].median() if len(valid) else np.nan
        rows.append({
            "partition": instance.partition, "instance_id": instance.expert.instance_id,
            "initial_gross_bps": initial_gross, "initial_net14_bps": initial_net,
            **{f"bucket_{low}_{high}h_net14_bps": edge for (low, high), edge in zip(AGE_BUCKETS, bucket_edges, strict=True)},
            "half_life_hours": half_life, "half_life_censored": half_life is None,
            "economic_lifetime_hours": economic_lifetime,
            "similarity_net_spearman": float(relationship) if pd.notna(relationship) else None,
            "high_similarity_net14_bps": float(valid.loc[valid["similarity"].ge(median), "net14_bps"].mean()) if len(valid) else None,
            "low_similarity_net14_bps": float(valid.loc[valid["similarity"].lt(median), "net14_bps"].mean()) if len(valid) else None,
        })
    return pd.DataFrame(rows)


def baseline_economics(work: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    primary = decisions[(decisions["partition"].eq("PROSPECTIVE")) & (decisions["mode"].eq("FULL"))]
    lookup = work.set_index(["decision_at", "symbol"])
    rows: list[dict[str, Any]] = []
    strategies = ("ALWAYS_SKIP", "ALWAYS_LONG", "ALWAYS_SHORT", "15M_MOMENTUM", "15M_MEAN_REVERSION")
    for strategy in strategies:
        signal_count = trade_count = 0
        gross_values: list[float] = []
        for record in primary.sort_values(["decision_at", "symbol"], kind="mergesort").itertuples():
            panel_row = lookup.loc[(_utc(record.decision_at), record.symbol)]
            target = panel_row[f"gross_target_{record.horizon_minutes}m_bps"]
            if pd.isna(target):
                continue
            signal_count += 1
            if strategy == "ALWAYS_SKIP":
                continue
            if strategy == "ALWAYS_LONG":
                side = 1
            elif strategy == "ALWAYS_SHORT":
                side = -1
            elif strategy == "15M_MOMENTUM":
                side = 1 if panel_row["return_15m_bps"] > 0 else -1
            else:
                side = -1 if panel_row["return_15m_bps"] > 0 else 1
            trade_count += 1
            gross_values.append(float(target) * side)
        gross_total = float(np.sum(gross_values))
        rows.append({
            "strategy": strategy, "signal_count": signal_count, "trade_count": trade_count,
            "gross_per_trade_bps": gross_total / trade_count if trade_count else 0.0,
            "net14_per_trade_bps": gross_total / trade_count - 14.0 if trade_count else 0.0,
            "net20_per_trade_bps": gross_total / trade_count - 20.0 if trade_count else 0.0,
            "net30_per_trade_bps": gross_total / trade_count - 30.0 if trade_count else 0.0,
            "gross_per_signal_bps": gross_total / signal_count if signal_count else 0.0,
            "net14_per_signal_bps": (gross_total - 14.0 * trade_count) / signal_count if signal_count else 0.0,
            "net20_per_signal_bps": (gross_total - 20.0 * trade_count) / signal_count if signal_count else 0.0,
            "net30_per_signal_bps": (gross_total - 30.0 * trade_count) / signal_count if signal_count else 0.0,
        })
    return pd.DataFrame(rows)


def _maximum_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    cumulative = values.cumsum()
    return float((cumulative.cummax() - cumulative).max())


def summarize(
    candidates: pd.DataFrame, instances: list[CreatedInstance], trades: pd.DataFrame,
    half_life: pd.DataFrame, ttl_sensitivity_frame: pd.DataFrame,
    expirations: pd.DataFrame, baselines: pd.DataFrame, decisions: pd.DataFrame,
    work: pd.DataFrame, cfg: Mapping[str, Any], valid_run: bool,
) -> dict[str, Any]:
    prospective = trades[(trades.get("partition") == "PROSPECTIVE") & (trades.get("mode") == "FULL")] if not trades.empty else trades
    ttl = trades[(trades.get("partition") == "PROSPECTIVE") & (trades.get("mode") == "TTL_ONLY")] if not trades.empty else trades
    if len(prospective):
        bootstrap = temporal_block_bootstrap(
            prospective["opened_at"], prospective["net14_bps"], block_hours=24,
            draws=10_000, seed=int(cfg["seed"]),
        )
        symbols = prospective["symbol"].value_counts()
        instance_count = prospective["model_instance_id"].nunique()
        relationship = prospective["similarity"].corr(prospective["net14_bps"], method="spearman")
        high_low = prospective.assign(high=prospective["similarity"].ge(prospective["similarity"].median())).groupby("high")["net14_bps"].mean()
        similarity_gate = pd.notna(relationship) and relationship > 0 and high_low.get(True, -np.inf) > high_low.get(False, np.inf)
    else:
        bootstrap = None
        symbols = pd.Series(dtype=int)
        instance_count = 0
        relationship = np.nan
        similarity_gate = False
    guardian_improvement = (
        len(prospective) > 0 and len(ttl) > 0
        and (prospective["net14_bps"].mean() > ttl["net14_bps"].mean()
             or _maximum_drawdown(prospective["net14_bps"]) < _maximum_drawdown(ttl["net14_bps"]))
    )
    gates = {
        "minimum_prospective_trades": len(prospective) >= int(cfg["success"]["minimum_prospective_trades"]),
        "minimum_instances": instance_count >= int(cfg["success"]["minimum_instances"]),
        "minimum_symbols": len(symbols) >= int(cfg["success"]["minimum_symbols"]),
        "symbol_concentration": (symbols.max() / len(prospective) <= float(cfg["success"]["maximum_single_symbol_fraction"])) if len(prospective) else False,
        "baseline_net_positive": bool(len(prospective) and prospective["net14_bps"].mean() > 0),
        "stress_net_positive": bool(len(prospective) and prospective["net20_bps"].mean() > 0),
        "temporal_bootstrap_ci_lower_positive": bool(bootstrap and bootstrap.ci_lower > 0),
        "guardian_improvement": bool(guardian_improvement),
        "similarity_edge_relationship": bool(similarity_gate),
    }
    validation_eligible = bool(len(candidates) and candidates[candidates["partition"].eq("VALIDATION")]["eligible"].any())
    gross_positive = bool(len(prospective) and prospective["gross_bps"].mean() > 0)
    prospective_instances = [item for item in instances if item.partition == "PROSPECTIVE"]
    instance_metadata = {
        item.expert.instance_id: (item.expert.window_hours, item.expert.horizon_minutes)
        for item in prospective_instances
    }
    window_counts: dict[str, int] = {}
    horizon_counts: dict[str, int] = {}
    for instance_id in prospective.get("model_instance_id", pd.Series(dtype=str)).unique():
        if instance_id not in instance_metadata:
            continue
        window, horizon = instance_metadata[instance_id]
        window_counts[str(window)] = window_counts.get(str(window), 0) + 1
        horizon_counts[str(horizon)] = horizon_counts.get(str(horizon), 0) + 1
    enriched = prospective.copy()
    if len(enriched):
        enriched["window_hours"] = enriched["model_instance_id"].map(
            {key: value[0] for key, value in instance_metadata.items()}
        )

    def economic_breakdown(column: str) -> list[dict[str, Any]]:
        if not len(enriched):
            return []
        output = []
        for value, rows in enriched.groupby(column, sort=True):
            output.append({
                column: int(value) if column != "side" else str(value),
                "trades": int(len(rows)),
                "gross_mean_bps": float(rows["gross_bps"].mean()),
                "net14_mean_bps": float(rows["net14_bps"].mean()),
                "net20_mean_bps": float(rows["net20_bps"].mean()),
            })
        return output

    window_economics = economic_breakdown("window_hours")
    horizon_economics = economic_breakdown("horizon_minutes")
    side_economics = economic_breakdown("side")
    prospective_slots = work[
        work["decision_at"].ge(_utc(cfg["partitions"]["prospective"][0]))
        & work["decision_at"].lt(_utc(cfg["partitions"]["prospective"][1]))
    ]
    skip_fraction = 1.0 - len(prospective) / len(prospective_slots) if len(prospective_slots) else None
    full_expirations = expirations[
        expirations.get("partition", pd.Series(dtype=str)).eq("PROSPECTIVE")
        & expirations.get("mode", pd.Series(dtype=str)).eq("FULL")
    ] if not expirations.empty else expirations
    expiration_counts = (
        full_expirations["reason"].value_counts().sort_index().astype(int).to_dict()
        if len(full_expirations) else {}
    )
    prospective_half_life = half_life[half_life.get("partition", pd.Series(dtype=str)).eq("PROSPECTIVE")] if not half_life.empty else half_life
    observed_half_lives = prospective_half_life["half_life_hours"].dropna() if len(prospective_half_life) else pd.Series(dtype=float)
    age_edges = {}
    for low, high in AGE_BUCKETS:
        column = f"bucket_{low}_{high}h_net14_bps"
        values = prospective_half_life[column].dropna() if len(prospective_half_life) and column in prospective_half_life else pd.Series(dtype=float)
        age_edges[f"{low}_{high}h"] = float(values.median()) if len(values) else None
    day_means = prospective.assign(day=pd.to_datetime(prospective["opened_at"], utc=True).dt.floor("D")).groupby("day")["net14_bps"].mean() if len(prospective) else pd.Series(dtype=float)
    baseline_records = baselines.replace({np.nan: None}).to_dict(orient="records") if len(baselines) else []
    ttl_summary = []
    if len(ttl_sensitivity_frame):
        prospective_ttl = ttl_sensitivity_frame[ttl_sensitivity_frame["partition"].eq("PROSPECTIVE")]
        for ttl_hours, rows in prospective_ttl.groupby("ttl_hours", sort=True):
            ttl_summary.append({
                "ttl_hours": int(ttl_hours),
                "instances": int(rows["instance_id"].nunique()),
                "mean_net14_bps": float(rows["net14_mean_bps"].dropna().mean()) if rows["net14_mean_bps"].notna().any() else None,
            })
    if not valid_run:
        grade, verdict = "D", "INSUFFICIENT_DATA"
    elif all(gates.values()):
        grade, verdict = "A", "EPHEMERAL_ALPHA_CONFIRMED"
    elif gross_positive or validation_eligible:
        grade, verdict = "B", "EPHEMERAL_SIGNAL_DETECTED_NOT_YET_ECONOMIC"
    else:
        grade, verdict = "C", "NO_EPHEMERAL_EDGE_FOUND"
    robust_b = grade == "B" and gates["baseline_net_positive"] and gates["stress_net_positive"] and gates["symbol_concentration"]
    return {
        "grade": grade, "verdict": verdict, "valid_run": valid_run, "gates": gates,
        "validation_had_eligible_candidates": validation_eligible,
        "prospective": {
            "trade_count": len(prospective), "instance_count": instance_count,
            "symbol_count": len(symbols),
            "gross_mean_bps": float(prospective["gross_bps"].mean()) if len(prospective) else None,
            "net14_mean_bps": float(prospective["net14_bps"].mean()) if len(prospective) else None,
            "net20_mean_bps": float(prospective["net20_bps"].mean()) if len(prospective) else None,
            "net30_mean_bps": float(prospective["net30_bps"].mean()) if len(prospective) else None,
            "maximum_drawdown_net14_bps": _maximum_drawdown(prospective["net14_bps"]) if len(prospective) else None,
            "similarity_net_spearman": float(relationship) if pd.notna(relationship) else None,
            "day_bootstrap_draws": 10_000,
            "day_bootstrap_probability_positive": bootstrap.probability_positive if bootstrap else None,
            "day_bootstrap_ci_lower": bootstrap.ci_lower if bootstrap else None,
            "day_bootstrap_ci_upper": bootstrap.ci_upper if bootstrap else None,
            "long_trades": int(prospective["side"].eq("LONG").sum()) if len(prospective) else 0,
            "short_trades": int(prospective["side"].eq("SHORT").sum()) if len(prospective) else 0,
            "skip_fraction_all_market_slots": skip_fraction,
            "window_instance_counts_used": window_counts,
            "horizon_instance_counts_used": horizon_counts,
            "economics_by_window": window_economics,
            "economics_by_horizon": horizon_economics,
            "economics_by_side": side_economics,
            "positive_days": int((day_means > 0).sum()),
            "observed_days": int(len(day_means)),
        },
        "ttl_only": {
            "trade_count": len(ttl), "net14_mean_bps": float(ttl["net14_bps"].mean()) if len(ttl) else None,
            "maximum_drawdown_net14_bps": _maximum_drawdown(ttl["net14_bps"]) if len(ttl) else None,
        },
        "selected_instances": len(instances),
        "prospective_instances_created": len(prospective_instances),
        "prospective_candidate_evaluations": int(candidates["partition"].eq("PROSPECTIVE").sum()) if len(candidates) else 0,
        "prospective_eligible_candidates": int(candidates[candidates["partition"].eq("PROSPECTIVE")]["eligible"].sum()) if len(candidates) else 0,
        "expiration_counts": {reason: int(expiration_counts.get(reason, 0)) for reason in ("TTL", "REGIME_DRIFT", "EDGE_DECAY")},
        "instances_censored_at_partition_end": len(prospective_instances) - int(full_expirations["instance_id"].nunique()) if len(full_expirations) else len(prospective_instances),
        "half_life": {
            "median_observed_hours": float(observed_half_lives.median()) if len(observed_half_lives) else None,
            "observed_instances": int(len(observed_half_lives)),
            "censored_instances": int(prospective_half_life["half_life_censored"].sum()) if len(prospective_half_life) else 0,
            "median_age_bucket_net14_bps": age_edges,
        },
        "guardian_delta_net14_bps": float(prospective["net14_bps"].mean() - ttl["net14_bps"].mean()) if len(prospective) and len(ttl) else None,
        "guardian_drawdown_reduction_bps": float(_maximum_drawdown(ttl["net14_bps"]) - _maximum_drawdown(prospective["net14_bps"])) if len(prospective) and len(ttl) else None,
        "ttl_sensitivity": ttl_summary,
        "baselines": baseline_records,
        "merits_phase_two": grade == "A" or robust_b,
        "inference_seed": int(cfg["seed"]),
    }


def run_experiment(
    panel: pd.DataFrame, config: Mapping[str, Any], *, partitions: tuple[str, ...] = ("validation", "prospective")
) -> ExperimentResult:
    """Run W11 from an already-built panel; this function performs no file access."""
    cfg = dict(config)
    features = list(cfg["features"]["names"])
    work = _flat_panel(panel, features)
    candidates, instances = create_candidates(work, cfg, partitions)
    decision_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    expiration_frames: list[pd.DataFrame] = []
    for partition in partitions:
        for mode in ("FULL", "TTL_ONLY"):
            decisions, trades, expirations = replay_partition(work, instances, cfg, partition, mode)
            decision_frames.append(decisions)
            trade_frames.append(trades)
            expiration_frames.append(expirations)
    decisions = pd.concat(decision_frames, ignore_index=True) if any(not x.empty for x in decision_frames) else pd.DataFrame()
    trades = pd.concat(trade_frames, ignore_index=True) if any(not x.empty for x in trade_frames) else pd.DataFrame()
    expirations = pd.concat(expiration_frames, ignore_index=True) if any(not x.empty for x in expiration_frames) else pd.DataFrame()
    ttl, diagnostic = ttl_sensitivity(work, instances, cfg)
    half_life = half_life_diagnostics(diagnostic, instances)
    baselines = baseline_economics(work, decisions)
    valid_run = len(work) > 0 and all(work["feature_available_at"].le(work["decision_at"]))
    summary = summarize(
        candidates, instances, trades, half_life, ttl, expirations, baselines,
        decisions, work, cfg, valid_run,
    )
    frames = {
        "instances": pd.DataFrame([_instance_row(item) for item in instances]),
        "decisions": decisions, "trades": trades, "ttl_sensitivity": ttl,
        "half_life": half_life, "baselines": baselines, "expiration_registry": expirations,
    }
    for name, columns in EMPTY_COLUMNS.items():
        if frames[name].empty and len(frames[name].columns) == 0:
            frames[name] = pd.DataFrame(columns=columns)
    return ExperimentResult(
        candidates, frames["instances"], frames["decisions"], frames["trades"],
        frames["ttl_sensitivity"], frames["half_life"], frames["baselines"],
        frames["expiration_registry"], summary,
        work["decision_at"].min(), work["decision_at"].max(),
    )
