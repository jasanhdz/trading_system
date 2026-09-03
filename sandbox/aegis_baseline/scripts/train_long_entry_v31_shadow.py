#!/usr/bin/env python3
"""Train the preregistered two-stage LONG v3.1 committee in research Shadow."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from aegis.config import CANONICAL_SYMBOLS
from aegis.features import DeterministicFeaturePipeline
from aegis.research.long_entry_v21_shadow import (
    atr_normalized_long_outcome,
    factorized_regime,
    multitimeframe_long_features,
    protected_long_utility,
)
from aegis.research.long_entry_v22_shadow import long_v22_feature_vector
from aegis.research.long_entry_v31_shadow import (
    entry_confirmation,
    family_horizon,
    global_context_gate,
    specialist_committee_score_v31,
)
from aegis.research.long_entry_v3_shadow import (
    LONG_V3_FEATURE_NAMES,
    HardNegativeType,
    LongCandidateFamily,
    classify_hard_negative,
    classify_long_v3_candidate,
    long_v3_feature_vector,
    microstructure_feature_vector,
)
from aegis.training.train import fit_platt_calibrator
from aegis.utils import sha256_file
from train_long_entry_archetypes_v2_shadow import _snapshot
from train_long_entry_v21_shadow import (
    _atr_contract,
    _classifier,
    _fold_boundaries,
    _probability,
    _protection,
    _selection_metrics,
    _source_series,
)
from train_long_entry_v22_shadow import _group_attribution, _policy_valid
from train_long_entry_v3_shadow import (
    _funding_at,
    _load_microstructure,
    _verify_protection_authority,
    _with_base_volume,
)


FAMILIES = tuple(
    value for value in LongCandidateFamily if value is not LongCandidateFamily.NONE
)


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{identity} must be a mapping")
    return value


def _feature_bundle(
    *,
    symbol: str,
    index: int,
    raw_values: Sequence[float],
    feature_names: Sequence[str],
    candles: Mapping[str, Sequence[Any]],
    micro_bars: Mapping[str, Sequence[Any]],
    funding: Mapping[str, Any],
    history_bars: int,
    maximum_funding_age: int,
    pipeline: DeterministicFeaturePipeline,
) -> Mapping[str, Any] | None:
    base = dict(zip(feature_names, raw_values))
    history = candles[symbol][index - history_bars + 1 : index + 1]
    raw_micro = [
        _with_base_volume(micro_bars[symbol][position], candles[symbol][position].volume)
        for position in range(index - 23, index + 1)
    ]
    if any(value is None for value in raw_micro):
        return None
    timestamp_ms = int(candles[symbol][index].close_time.timestamp() * 1000)
    funding_values = _funding_at(funding[symbol], timestamp_ms, maximum_funding_age)
    if funding_values is None:
        return None
    v21, context = multitimeframe_long_features(base, history, pipeline=pipeline)
    regime = factorized_regime(base, context)
    v22 = long_v22_feature_vector(v21, regime)
    micro = microstructure_feature_vector(
        tuple(value for value in raw_micro if value is not None),
        return_1=float(base["ret_1"]),
        atr_fraction=float(base["atr_12"]),
        funding_rate_last=funding_values[0],
        funding_rate_previous=funding_values[1],
    )
    micro_values = dict(zip(LONG_V3_FEATURE_NAMES[len(v22) :], micro))
    return {
        "base": base,
        "context": context,
        "regime": regime,
        "history": history,
        "micro": micro_values,
        "features": long_v3_feature_vector(v22, micro),
    }


def _label(
    *,
    signal: Any,
    history: Sequence[Any],
    future: Sequence[Any],
    atr_fraction: float,
    label_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    outcome = atr_normalized_long_outcome(
        signal=signal,
        future=future,
        atr_fraction=atr_fraction,
        contract=_atr_contract(label_config),
    )
    utility_config = _mapping(label_config["utility"], "utility")
    utility = protected_long_utility(
        history=history,
        future=future,
        outcome=outcome,
        protection=_protection(label_config),
        mae_penalty_weight=float(utility_config["mae_penalty_weight"]),
        underwater_bar_penalty_fraction=float(
            utility_config["underwater_bar_penalty_fraction"]
        ),
        catastrophic_mae_atr_multiple=float(
            utility_config["catastrophic_mae_atr_multiple"]
        ),
    )
    return {**outcome, **utility}


def build_datasets(
    root: Path,
    config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    label_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Mapping[str, Any]]:
    source = _mapping(config["source"], "source")
    sampling = _mapping(config["sampling"], "sampling")
    history_bars = int(sampling["history_bars"])
    maximum_horizon = int(sampling["maximum_horizon_bars"])
    stride = int(sampling["stride_bars"])
    independent_stride = int(sampling["independent_test_stride_bars"])
    candles, common, source_inventory = _source_series(
        root / str(source["base_database"]),
        root / str(source["public_candle_delta"]),
        lookback_days=int(source["lookback_days"]),
        history_bars=history_bars,
        horizon_bars=maximum_horizon + 1,
    )
    micro_bars, funding, micro_inventory = _load_microstructure(
        root / str(source["public_microstructure_database"]), common
    )
    pipeline = DeterministicFeaturePipeline()
    candidate_builder = _mapping(candidate_config["candidate_builder"], "candidate")
    opportunity_stage = _mapping(config["opportunity_stage"], "opportunity_stage")
    timing_stage = _mapping(config["entry_timing_stage"], "entry_timing_stage")
    context_config = _mapping(config["global_context_gate"], "global_context_gate")
    maximum_funding_age = 12 * 3600 * 1000
    independent_every = max(1, math.ceil(independent_stride / stride))
    opportunities: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    context_pass_counts: Counter[str] = Counter()
    confirmation_pass_counts: Counter[str] = Counter()
    hard_negative_counts: Counter[str] = Counter()
    evaluated_snapshots = 0
    feature_ready = 0

    for index in range(
        history_bars - 1,
        len(common) - maximum_horizon - 1,
        stride,
    ):
        current_batch = pipeline.transform(_snapshot(candles, index, 96))
        current_rows = {row.symbol: row for row in current_batch.rows}
        next_batch = None
        next_rows: Mapping[str, Any] = {}
        independent = evaluated_snapshots % independent_every == 0
        timestamp = candles[CANONICAL_SYMBOLS[0]][index].close_time
        for symbol in CANONICAL_SYMBOLS:
            row = current_rows[symbol]
            bundle = _feature_bundle(
                symbol=symbol,
                index=index,
                raw_values=row.raw_values,
                feature_names=current_batch.feature_names,
                candles=candles,
                micro_bars=micro_bars,
                funding=funding,
                history_bars=history_bars,
                maximum_funding_age=maximum_funding_age,
                pipeline=pipeline,
            )
            if bundle is None:
                continue
            feature_ready += 1
            candidate = classify_long_v3_candidate(
                base=bundle["base"],
                context=bundle["context"],
                micro=bundle["micro"],
                history=bundle["history"],
                config=candidate_builder,
            )
            family = str(candidate["family"])
            family_counts[family] += 1
            if not candidate["is_candidate"]:
                continue
            horizon = family_horizon(family, opportunity_stage)
            opportunity_future = candles[symbol][index + 1 : index + 1 + horizon]
            opportunity_outcome = _label(
                signal=candles[symbol][index],
                history=bundle["history"],
                future=opportunity_future,
                atr_fraction=float(bundle["base"]["atr_12"]),
                label_config=label_config,
            )
            opportunity_hard_negative = classify_hard_negative(
                family, opportunity_outcome
            )
            opportunity = {
                "timestamp": timestamp,
                "symbol": symbol,
                "archetype": family,
                "candidate_family": family,
                "regime": bundle["regime"]["identity"],
                "regime_axes": bundle["regime"],
                "independent": independent,
                "features": bundle["features"],
                "hard_negative": opportunity_hard_negative.value,
                **opportunity_outcome,
            }
            opportunities.append(opportunity)
            context = global_context_gate(
                base=bundle["base"],
                context=bundle["context"],
                regime=bundle["regime"],
                config=context_config,
            )
            if not context["passed"]:
                continue
            context_pass_counts[family] += 1
            trigger_name = str(timing_stage["family_trigger"][family])
            delayed = trigger_name != "NEXT_BAR_OPEN"
            confirmation_bundle = None
            if delayed:
                if next_batch is None:
                    next_batch = pipeline.transform(_snapshot(candles, index + 1, 96))
                    next_rows = {item.symbol: item for item in next_batch.rows}
                next_row = next_rows[symbol]
                confirmation_bundle = _feature_bundle(
                    symbol=symbol,
                    index=index + 1,
                    raw_values=next_row.raw_values,
                    feature_names=next_batch.feature_names,
                    candles=candles,
                    micro_bars=micro_bars,
                    funding=funding,
                    history_bars=history_bars,
                    maximum_funding_age=maximum_funding_age,
                    pipeline=pipeline,
                )
                if confirmation_bundle is None:
                    continue
            confirmation = entry_confirmation(
                family=family,
                signal=candles[symbol][index],
                confirmation=candles[symbol][index + 1] if delayed else None,
                confirmation_micro=(
                    confirmation_bundle["micro"] if confirmation_bundle else None
                ),
                config=timing_stage,
            )
            if not confirmation["passed"]:
                continue
            confirmation_pass_counts[family] += 1
            signal_index = index + (1 if delayed else 0)
            execution_bundle = confirmation_bundle if delayed else bundle
            entry_offset = int(confirmation["entry_offset_bars"])
            execution_future = candles[symbol][
                index + entry_offset : index + entry_offset + horizon
            ]
            execution_outcome = _label(
                signal=candles[symbol][signal_index],
                history=execution_bundle["history"],
                future=execution_future,
                atr_fraction=float(execution_bundle["base"]["atr_12"]),
                label_config=label_config,
            )
            hard_negative = classify_hard_negative(family, execution_outcome)
            hard_negative_counts[hard_negative.value] += 1
            executions.append(
                {
                    "timestamp": candles[symbol][signal_index].close_time,
                    "opportunity_timestamp": timestamp,
                    "symbol": symbol,
                    "archetype": family,
                    "candidate_family": family,
                    "entry_trigger": confirmation["trigger"],
                    "context_evidence": context["evidence"],
                    "confirmation_evidence": confirmation["evidence"],
                    "regime": execution_bundle["regime"]["identity"],
                    "regime_axes": execution_bundle["regime"],
                    "independent": independent,
                    "opportunity_features": bundle["features"],
                    "opportunity_target_before_stop": bool(
                        opportunity_outcome["target_before_stop"]
                    ),
                    "features": execution_bundle["features"],
                    "hard_negative": hard_negative.value,
                    "falling_knife": hard_negative is HardNegativeType.FALLING_KNIFE,
                    **execution_outcome,
                }
            )
        evaluated_snapshots += 1
    return opportunities, executions, {
        **source_inventory,
        "microstructure": micro_inventory,
        "evaluated_snapshots": evaluated_snapshots,
        "source_population_rows": evaluated_snapshots * len(CANONICAL_SYMBOLS),
        "feature_ready_rows": feature_ready,
        "opportunity_rows": len(opportunities),
        "execution_candidate_rows": len(executions),
        "execution_fraction_of_opportunities": (
            len(executions) / len(opportunities) if opportunities else 0.0
        ),
        "candidate_family_counts": dict(sorted(family_counts.items())),
        "context_pass_counts": dict(sorted(context_pass_counts.items())),
        "confirmation_pass_counts": dict(sorted(confirmation_pass_counts.items())),
        "hard_negative_counts": dict(sorted(hard_negative_counts.items())),
        "feature_count": len(LONG_V3_FEATURE_NAMES),
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }


def _fit_binary(
    train: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    *,
    target: str,
    feature_field: str,
    seed: int,
    negative_weight: float,
) -> tuple[Any, Any] | None:
    y_train = np.asarray([bool(row[target]) for row in train], dtype=int)
    y_cal = np.asarray([bool(row[target]) for row in calibration], dtype=int)
    if (
        len(train) < 200
        or len(calibration) < 80
        or len(np.unique(y_train)) != 2
        or len(np.unique(y_cal)) != 2
    ):
        return None
    weights = np.asarray(
        [
            negative_weight
            if row["hard_negative"] != HardNegativeType.NOT_HARD_NEGATIVE.value
            else 1.0
            for row in train
        ]
    )
    model = _classifier(seed).fit(
        np.asarray([row[feature_field] for row in train], dtype=np.float64),
        y_train,
        sample_weight=weights,
    )
    raw = model.predict_proba(
        np.asarray([row[feature_field] for row in calibration], dtype=np.float64)
    )[:, 1]
    return model, fit_platt_calibrator(raw, y_cal)


def _fit_bundles(
    opportunity_train: Sequence[Mapping[str, Any]],
    opportunity_calibration: Sequence[Mapping[str, Any]],
    execution_train: Sequence[Mapping[str, Any]],
    execution_calibration: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Mapping[str, tuple[Any, Any]]] | None:
    weight = float(config["hard_negatives"]["training_weight"])
    bundles: dict[str, dict[str, tuple[Any, Any]]] = {}
    for family_index, family in enumerate(FAMILIES):
        opportunity_fit = _fit_binary(
            [row for row in opportunity_train if row["archetype"] == family.value],
            [
                row
                for row in opportunity_calibration
                if row["archetype"] == family.value
            ],
            target="target_before_stop",
            feature_field="features",
            seed=seed + family_index * 10,
            negative_weight=weight,
        )
        train = [row for row in execution_train if row["archetype"] == family.value]
        calibration = [
            row for row in execution_calibration if row["archetype"] == family.value
        ]
        fits = {
            "timing_probability": _fit_binary(
                train,
                calibration,
                target="clean_fast_success",
                feature_field="features",
                seed=seed + family_index * 10 + 1,
                negative_weight=weight,
            ),
            "falling_knife_probability": _fit_binary(
                train,
                calibration,
                target="falling_knife",
                feature_field="features",
                seed=seed + family_index * 10 + 2,
                negative_weight=weight,
            ),
            "path_risk_probability": _fit_binary(
                train,
                calibration,
                target="catastrophic_path",
                feature_field="features",
                seed=seed + family_index * 10 + 3,
                negative_weight=weight,
            ),
        }
        if opportunity_fit is None or any(value is None for value in fits.values()):
            return None
        bundles[family.value] = {
            "opportunity_probability": opportunity_fit,
            **{name: value for name, value in fits.items() if value is not None},
        }
    return bundles


def _predict(
    rows: Sequence[Mapping[str, Any]],
    bundles: Mapping[str, Mapping[str, tuple[Any, Any]]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["archetype"])].append((index, row))
    ordered: list[dict[str, Any] | None] = [None] * len(rows)
    for family, indexed in groups.items():
        specialists = bundles.get(family)
        if specialists is None:
            continue
        group = [row for _, row in indexed]
        values = {}
        for output, (model, calibrator) in specialists.items():
            feature_field = (
                "opportunity_features"
                if output == "opportunity_probability"
                else "features"
            )
            proxy = [{"features": row[feature_field]} for row in group]
            values[output] = _probability(model, calibrator, proxy)
        for local_index, (source_index, row) in enumerate(indexed):
            opportunity = float(values["opportunity_probability"][local_index])
            timing = float(values["timing_probability"][local_index])
            falling = float(values["falling_knife_probability"][local_index])
            risk = float(values["path_risk_probability"][local_index])
            ordered[source_index] = {
                **row,
                "opportunity_probability": opportunity,
                "timing_probability": timing,
                "falling_knife_probability": falling,
                "path_risk_probability": risk,
                "committee_score": specialist_committee_score_v31(
                    opportunity, timing, falling, risk
                ),
            }
    return [row for row in ordered if row is not None]


def _select(rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> np.ndarray:
    eligible: dict[Any, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if (
            float(row["committee_score"]) >= float(policy["minimum_score"])
            and float(row["falling_knife_probability"])
            <= float(policy["maximum_falling_knife"])
            and float(row["path_risk_probability"])
            <= float(policy["maximum_path_risk"])
        ):
            eligible[row["timestamp"]].append((index, row))
    selected = np.zeros(len(rows), dtype=bool)
    for candidates in eligible.values():
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item[1]["committee_score"]),
                float(item[1]["falling_knife_probability"]),
                float(item[1]["path_risk_probability"]),
                str(item[1]["symbol"]),
            ),
        )
        for index, _ in ordered[: int(policy["maximum_selected_per_timestamp"])]:
            selected[index] = True
    return selected


def _derive_policy(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    ranking = config["ranking"]
    minimum = int(config["validation"]["minimum_calibration_selections"])
    scores = np.asarray([float(row["committee_score"]) for row in rows])
    falling = np.asarray([float(row["falling_knife_probability"]) for row in rows])
    risks = np.asarray([float(row["path_risk_probability"]) for row in rows])
    choices = []
    for score_quantile in ranking["score_quantiles"]:
        score = float(np.quantile(scores, float(score_quantile), method="higher"))
        for falling_quantile in ranking["maximum_falling_knife_quantiles"]:
            maximum_falling = float(
                np.quantile(falling, float(falling_quantile), method="lower")
            )
            for risk_quantile in ranking["maximum_path_risk_quantiles"]:
                maximum_risk = float(
                    np.quantile(risks, float(risk_quantile), method="lower")
                )
                for top_k in ranking["maximum_selected_per_timestamp_grid"]:
                    policy = {
                        "minimum_score": score,
                        "maximum_falling_knife": maximum_falling,
                        "maximum_path_risk": maximum_risk,
                        "maximum_selected_per_timestamp": int(top_k),
                        "score_quantile": float(score_quantile),
                        "falling_knife_quantile": float(falling_quantile),
                        "risk_quantile": float(risk_quantile),
                    }
                    metrics = _selection_metrics(rows, _select(rows, policy))
                    choices.append(
                        {
                            **policy,
                            "metrics": metrics,
                            "valid": _policy_valid(metrics, minimum),
                        }
                    )
    valid = [choice for choice in choices if choice["valid"]]
    return max(
        valid or choices,
        key=lambda choice: (
            float(choice["metrics"]["selected_protected_worst_net"] or -1.0),
            -float(choice["metrics"]["selected_mae"] or 1.0),
            int(choice["metrics"]["selected_rows"]),
        ),
    )


def _specialist_metrics(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    targets = {
        "opportunity_probability": "opportunity_target_before_stop",
        "timing_probability": "clean_fast_success",
        "falling_knife_probability": "falling_knife",
        "path_risk_probability": "catastrophic_path",
    }
    return {
        output: {
            "prevalence": float(np.mean([bool(row[target]) for row in rows])),
            "average_precision": float(
                average_precision_score(
                    [bool(row[target]) for row in rows],
                    [float(row[output]) for row in rows],
                )
            ),
        }
        for output, target in targets.items()
    }


def _evaluate_fold(
    opportunities: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    fold_id: int,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    opportunity_train = [row for row in opportunities if row["timestamp"] <= train_end]
    opportunity_calibration = [
        row
        for row in opportunities
        if train_end + embargo < row["timestamp"] <= calibration_end
    ]
    execution_train = [row for row in executions if row["timestamp"] <= train_end]
    execution_calibration = [
        row
        for row in executions
        if train_end + embargo < row["timestamp"] <= calibration_end
    ]
    test = [
        row
        for row in executions
        if row["independent"]
        and calibration_end + embargo < row["timestamp"] <= test_end
    ]
    bundles = _fit_bundles(
        opportunity_train,
        opportunity_calibration,
        execution_train,
        execution_calibration,
        config,
        20261200 + fold_id * 100,
    )
    if bundles is None or len(test) < 100:
        return {
            "fold": fold_id,
            "status": "INSUFFICIENT_CLASSES_OR_ROWS",
            "opportunity_train_rows": len(opportunity_train),
            "execution_train_rows": len(execution_train),
            "calibration_rows": len(execution_calibration),
            "test_rows": len(test),
            "passed": False,
        }
    calibration = _predict(execution_calibration, bundles)
    predicted_test = _predict(test, bundles)
    policy = _derive_policy(calibration, config)
    selected = _select(predicted_test, policy)
    metrics = _selection_metrics(predicted_test, selected)
    minimum = int(config["validation"]["minimum_scoring_selections_per_fold"])
    passed = bool(
        policy["valid"]
        and _policy_valid(metrics, minimum)
        and metrics["p95_gap_hours"] is not None
        and metrics["p95_gap_hours"]
        <= float(config["ranking"]["maximum_p95_gap_hours"])
    )
    family_metrics = {}
    for family in FAMILIES:
        indices = [
            index
            for index, row in enumerate(predicted_test)
            if row["candidate_family"] == family.value
        ]
        family_metrics[family.value] = _selection_metrics(
            [predicted_test[index] for index in indices], selected[indices]
        )
    return {
        "fold": fold_id,
        "status": "EVALUATED",
        "opportunity_train_rows": len(opportunity_train),
        "execution_train_rows": len(execution_train),
        "calibration_rows": len(execution_calibration),
        "test_rows": len(test),
        "policy": policy,
        "specialist_metrics": _specialist_metrics(predicted_test),
        "metrics": metrics,
        "family_metrics": family_metrics,
        "passed": passed,
    }


def _family_retirement(
    folds: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> Mapping[str, Any]:
    contract = config["family_retirement"]
    result = {}
    for family in FAMILIES:
        metrics = [
            fold["family_metrics"][family.value]
            for fold in folds
            if fold["status"] == "EVALUATED"
        ]
        positive = sum(
            row["selected_rows"] >= 5
            and row["selected_protected_worst_net"] is not None
            and row["selected_protected_worst_net"] > 0.0
            and row["selected_mae"] < row["baseline_mae"]
            and row["selected_underwater_bars"] < row["baseline_underwater_bars"]
            for row in metrics
        )
        worst = min(
            (
                float(row["selected_protected_worst_net"])
                for row in metrics
                if row["selected_protected_worst_net"] is not None
            ),
            default=-1.0,
        )
        passed = bool(
            len(metrics) == 4
            and positive >= int(contract["minimum_positive_folds"])
            and worst >= 0.0
        )
        result[family.value] = {
            "positive_folds": positive,
            "worst_fold_protected_net": worst,
            "status": "ELIGIBLE" if passed else "RETIRED",
            "passed": passed,
        }
    return result


def _leave_one_symbol_out(
    opportunities: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    boundaries: tuple[datetime, datetime, datetime],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    train_end, calibration_end, test_end = boundaries
    embargo = timedelta(minutes=int(config["validation"]["embargo_minutes"]))
    reports: dict[str, Any] = {}
    for symbol_index, symbol in enumerate(CANONICAL_SYMBOLS):
        opportunity_train = [
            row
            for row in opportunities
            if row["symbol"] != symbol and row["timestamp"] <= train_end
        ]
        opportunity_calibration = [
            row
            for row in opportunities
            if row["symbol"] != symbol
            and train_end + embargo < row["timestamp"] <= calibration_end
        ]
        execution_train = [
            row
            for row in executions
            if row["symbol"] != symbol and row["timestamp"] <= train_end
        ]
        execution_calibration = [
            row
            for row in executions
            if row["symbol"] != symbol
            and train_end + embargo < row["timestamp"] <= calibration_end
        ]
        test = [
            row
            for row in executions
            if row["symbol"] == symbol
            and row["independent"]
            and calibration_end + embargo < row["timestamp"] <= test_end
        ]
        bundles = _fit_bundles(
            opportunity_train,
            opportunity_calibration,
            execution_train,
            execution_calibration,
            config,
            20264000 + symbol_index * 100,
        )
        if bundles is None or len(test) < 30:
            reports[symbol] = {"status": "INSUFFICIENT", "test_rows": len(test)}
            continue
        calibration = _predict(execution_calibration, bundles)
        predicted_test = _predict(test, bundles)
        policy = _derive_policy(calibration, config)
        metrics = _selection_metrics(predicted_test, _select(predicted_test, policy))
        no_regression = bool(
            policy["valid"]
            and metrics["selected_rows"] >= 5
            and metrics["selected_protected_worst_net"] is not None
            and metrics["selected_protected_worst_net"]
            >= metrics["baseline_protected_worst_net"]
            and metrics["selected_mae"] <= metrics["baseline_mae"]
        )
        reports[symbol] = {
            "status": "EVALUATED",
            "training_excluded_symbol": True,
            "test_rows": len(test),
            "metrics": metrics,
            "generalized_without_regression": no_regression,
        }
    evaluated = [row for row in reports.values() if row["status"] == "EVALUATED"]
    passing = sum(bool(row["generalized_without_regression"]) for row in evaluated)
    required = int(config["validation"]["minimum_symbols_without_regression"])
    return {
        "method": "TWO_STAGE_COMMITTEE_REFIT_WITH_TARGET_SYMBOL_EXCLUDED",
        "symbols": reports,
        "evaluated_symbols": len(evaluated),
        "symbols_without_regression": passing,
        "required_symbols_without_regression": required,
        "passed": len(evaluated) == len(CANONICAL_SYMBOLS) and passing >= required,
    }


def train_and_validate(
    opportunities: Sequence[Mapping[str, Any]],
    executions: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    times = sorted({row["timestamp"] for row in opportunities})
    if len(times) < 1000:
        return {
            "folds": [],
            "family_retirement": {},
            "validation_pass": False,
            "verdict": "RESEARCH_ONLY_NOT_PROMOTABLE",
        }
    boundaries = _fold_boundaries(times)
    folds = [
        _evaluate_fold(opportunities, executions, fold, index + 1, config)
        for index, fold in enumerate(boundaries)
    ]
    evaluated = [fold for fold in folds if fold["status"] == "EVALUATED"]
    passing = sum(bool(fold["passed"]) for fold in evaluated)
    retirement = _family_retirement(folds, config)
    eligible_families = [
        family for family, result in retirement.items() if result["passed"]
    ]
    primary = bool(
        len(evaluated) == 4
        and passing >= int(config["validation"]["minimum_positive_folds"])
        and eligible_families
        and all(
            float(fold["metrics"]["selected_protected_worst_net"] or -1.0) >= 0.0
            for fold in evaluated
        )
    )
    loso = (
        _leave_one_symbol_out(opportunities, executions, boundaries[-1], config)
        if primary
        else {
            "status": "NOT_RUN_PRIMARY_WALK_FORWARD_GATE_FAILED",
            "passed": False,
        }
    )
    validation_pass = primary and bool(loso["passed"])
    return {
        "folds": folds,
        "evaluated_folds": len(evaluated),
        "passing_folds": passing,
        "family_retirement": retirement,
        "eligible_families": eligible_families,
        "primary_walk_forward_pass": primary,
        "leave_one_symbol_out": loso,
        "validation_pass": validation_pass,
        "verdict": (
            "ELIGIBLE_FOR_SEPARATE_PROSPECTIVE_SHADOW_REVIEW"
            if validation_pass
            else "RESEARCH_ONLY_NOT_PROMOTABLE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/aegis_long_entry_v31_shadow.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/long_entry_v31_shadow/validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    config = _mapping(yaml.safe_load(config_path.read_text()), "config")
    if (
        config.get("schema_version")
        != "aegis-long-entry-v31-shadow-preregistration-v1"
        or config.get("mode") != "SHADOW"
        or config.get("selection_effect") != "NONE"
        or config.get("automatic_live_promotion") is not False
    ):
        raise SystemExit("AEGIS_LONG_V31_CONFIG_INVALID")
    _verify_protection_authority(root, config)
    v3_config = _mapping(
        yaml.safe_load(
            (root / "config/experiments/aegis_long_entry_v3_shadow.yaml").read_text()
        ),
        "v3_config",
    )
    label_config = _mapping(
        yaml.safe_load(
            (root / "config/experiments/aegis_long_entry_v21_shadow.yaml").read_text()
        ),
        "label_config",
    )
    opportunities, executions, source = build_datasets(
        root, config, v3_config, label_config
    )
    validation = train_and_validate(opportunities, executions, config)
    report = {
        "schema_id": "aegis-long-entry-v31-shadow-validation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "preregistration": str(config_path.relative_to(root)),
        "preregistration_sha256": sha256_file(config_path),
        "source": source,
        "feature_names": list(LONG_V3_FEATURE_NAMES),
        "opportunity_attribution": {
            "by_family": _group_attribution(
                opportunities, lambda row: str(row["candidate_family"])
            )
        },
        "execution_attribution": {
            "by_family": _group_attribution(
                executions, lambda row: str(row["candidate_family"])
            ),
            "by_trigger": _group_attribution(
                executions, lambda row: str(row["entry_trigger"])
            ),
            "by_symbol": _group_attribution(executions, lambda row: str(row["symbol"])),
        },
        "validation": validation,
        "deployment": {
            "selection_effect": "NONE",
            "shadow_runtime_enabled": False,
            "live_enabled": False,
            "automatic_promotion": False,
            "model_exported": False,
            "exchange_authority": False,
        },
        "exchange_calls": 0,
        "exchange_mutations": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(output)
    print(json.dumps({"output": str(output), "verdict": validation["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
