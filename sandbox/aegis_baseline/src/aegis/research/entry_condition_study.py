"""Prospective, non-executing study of entry conditions from Shadow evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..config import CANONICAL_SYMBOLS
from ..utils import canonical_json, sha256_file


class EntryConditionStudyError(ValueError):
    pass


@dataclass(frozen=True)
class EntryConditionHypothesis:
    identifier: str
    description: str
    require_control_selected: bool
    allowed_contexts: tuple[str, ...]
    excluded_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        valid_contexts = {"ANY", "RANGE", "BEAR_TREND"}
        if (
            not self.identifier
            or not self.require_control_selected
            or not self.allowed_contexts
            or not set(self.allowed_contexts) <= valid_contexts
            or not set(self.excluded_symbols) <= set(CANONICAL_SYMBOLS)
        ):
            raise EntryConditionStudyError(
                "AEGIS_ENTRY_CONDITION_HYPOTHESIS_INVALID"
            )


@dataclass(frozen=True)
class EntryConditionStudyConfig:
    study_id: str
    evidence_start: datetime
    signal_journal: Path
    outcome_journal: Path
    v2_runtime_config: Path
    long_runtime_config: Path
    report_path: Path
    horizon_bars: int
    minimum_observation_days: int
    maximum_observation_days: int
    minimum_selected: int
    minimum_per_symbol: int
    minimum_embargo_minutes: int
    minimum_temporal_blocks: int
    maximum_symbol_concentration: float
    maximum_mean_mae: float
    minimum_profit_factor: float
    bootstrap_resamples: int
    bootstrap_seed: int
    require_positive_ci: bool
    require_positive_halves: bool
    regime_minimum_observations: int
    regime_maximum_dominant_fraction: float
    regime_minimum_direction_labels: int
    regime_minimum_structure_labels: int
    regime_minimum_transitions: int
    hypotheses: tuple[EntryConditionHypothesis, ...]
    ranking_recalibration: Mapping[str, Any]
    promotion: Mapping[str, Any]


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EntryConditionStudyError(f"{identity} must be a mapping")
    return value


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise EntryConditionStudyError(
            "AEGIS_ENTRY_CONDITION_TIMESTAMP_INVALID"
        )
    return parsed.astimezone(timezone.utc)


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def load_entry_condition_study_config(
    path: Path,
    *,
    repo_root: Path,
) -> EntryConditionStudyConfig:
    payload = _mapping(yaml.safe_load(path.read_text()), "study")
    if (
        payload.get("schema_version")
        != "aegis-entry-condition-shadow-study-v1"
        or payload.get("mode") != "SHADOW"
        or payload.get("runtime_authority") != "OBSERVATIONAL_ONLY"
    ):
        raise EntryConditionStudyError(
            "AEGIS_ENTRY_CONDITION_CONFIG_INVALID"
        )
    inputs = _mapping(payload["inputs"], "inputs")
    evidence = _mapping(payload["evidence"], "evidence")
    regime = _mapping(payload["regime_validation"], "regime_validation")
    outputs = _mapping(payload["outputs"], "outputs")
    ranking = _mapping(payload["ranking_recalibration"], "ranking")
    promotion = _mapping(payload["promotion"], "promotion")
    hypotheses = tuple(
        EntryConditionHypothesis(
            identifier=str(item["id"]),
            description=str(item["description"]),
            require_control_selected=bool(item["require_control_selected"]),
            allowed_contexts=tuple(str(value) for value in item["allowed_contexts"]),
            excluded_symbols=tuple(str(value) for value in item["excluded_symbols"]),
        )
        for item in payload["hypotheses"]
    )
    config = EntryConditionStudyConfig(
        study_id=str(payload["study_id"]),
        evidence_start=_timestamp(payload["evidence_start_utc"]),
        signal_journal=_resolve(repo_root, inputs["signal_journal"]),
        outcome_journal=_resolve(repo_root, inputs["outcome_journal"]),
        v2_runtime_config=_resolve(repo_root, inputs["v2_runtime_config"]),
        long_runtime_config=_resolve(repo_root, inputs["long_runtime_config"]),
        report_path=_resolve(repo_root, outputs["report_path"]),
        horizon_bars=int(evidence["horizon_bars"]),
        minimum_observation_days=int(evidence["minimum_observation_days"]),
        maximum_observation_days=int(evidence["maximum_observation_days"]),
        minimum_selected=int(
            evidence["minimum_non_overlapping_selected_episodes"]
        ),
        minimum_per_symbol=int(
            evidence["minimum_per_symbol_non_overlapping_selected_episodes"]
        ),
        minimum_embargo_minutes=int(evidence["minimum_embargo_minutes"]),
        minimum_temporal_blocks=int(evidence["minimum_temporal_blocks"]),
        maximum_symbol_concentration=float(
            evidence["maximum_symbol_concentration"]
        ),
        maximum_mean_mae=float(evidence["maximum_mean_mae_fraction"]),
        minimum_profit_factor=float(evidence["minimum_profit_factor"]),
        bootstrap_resamples=int(evidence["bootstrap_resamples"]),
        bootstrap_seed=int(evidence["bootstrap_seed"]),
        require_positive_ci=bool(
            evidence["require_expectancy_ci95_low_above_zero"]
        ),
        require_positive_halves=bool(
            evidence["require_positive_first_and_second_half"]
        ),
        regime_minimum_observations=int(
            regime["minimum_observations_per_symbol"]
        ),
        regime_maximum_dominant_fraction=float(
            regime["maximum_dominant_axis_fraction"]
        ),
        regime_minimum_direction_labels=int(
            regime["minimum_distinct_direction_labels"]
        ),
        regime_minimum_structure_labels=int(
            regime["minimum_distinct_structure_labels"]
        ),
        regime_minimum_transitions=int(regime["minimum_axis_transitions"]),
        hypotheses=hypotheses,
        ranking_recalibration=dict(ranking),
        promotion=dict(promotion),
    )
    _validate_config(config)
    return config


def _validate_config(config: EntryConditionStudyConfig) -> None:
    if (
        not config.study_id
        or config.horizon_bars != 12
        or config.minimum_observation_days < 14
        or config.maximum_observation_days < config.minimum_observation_days
        or min(
            config.minimum_selected,
            config.minimum_per_symbol,
            config.minimum_embargo_minutes,
            config.minimum_temporal_blocks,
            config.bootstrap_resamples,
            config.regime_minimum_observations,
            config.regime_minimum_transitions,
        )
        <= 0
        or not 0.0 < config.maximum_symbol_concentration <= 1.0
        or not 0.0 < config.regime_maximum_dominant_fraction < 1.0
        or config.maximum_mean_mae <= 0.0
        or config.minimum_profit_factor < 1.0
    ):
        raise EntryConditionStudyError(
            "AEGIS_ENTRY_CONDITION_EVIDENCE_CONTRACT_INVALID"
        )
    if (
        bool(config.ranking_recalibration.get("training_allowed_now"))
        or not bool(
            config.ranking_recalibration.get("historical_replay_required")
        )
        or not bool(
            config.ranking_recalibration.get("purged_walk_forward_required")
        )
        or not bool(
            config.ranking_recalibration.get("champion_challenger_required")
        )
        or bool(config.promotion.get("automatic_training"))
        or bool(config.promotion.get("automatic_promotion"))
        or bool(config.promotion.get("live_configuration_changes"))
        or not bool(config.promotion.get("owner_authorization_required"))
        or not bool(
            config.promotion.get(
                "same_evidence_discovery_and_validation_prohibited"
            )
        )
    ):
        raise EntryConditionStudyError(
            "AEGIS_ENTRY_CONDITION_RUNTIME_AUTHORITY_INVALID"
        )
    identifiers = [item.identifier for item in config.hypotheses]
    if len(set(identifiers)) != len(identifiers):
        raise EntryConditionStudyError(
            "AEGIS_ENTRY_CONDITION_HYPOTHESIS_DUPLICATE"
        )


def _stable_payload(path: Path, attempts: int = 3) -> bytes:
    for attempt in range(attempts):
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns:
            if payload and not payload.endswith(b"\n"):
                raise EntryConditionStudyError(
                    "AEGIS_ENTRY_CONDITION_PARTIAL_JOURNAL"
                )
            return payload
        if attempt + 1 < attempts:
            time.sleep(0.05)
    raise EntryConditionStudyError(
        "AEGIS_ENTRY_CONDITION_JOURNAL_CHANGED_DURING_READ"
    )


def _parse_rows(payload: bytes, identity: str) -> tuple[Mapping[str, Any], ...]:
    rows = []
    identities = set()
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            row = _mapping(json.loads(line), f"row:{line_number}")
            key = str(row[identity])
        except (json.JSONDecodeError, KeyError) as exc:
            raise EntryConditionStudyError(
                "AEGIS_ENTRY_CONDITION_JOURNAL_INVALID"
            ) from exc
        if key in identities:
            raise EntryConditionStudyError(
                "AEGIS_ENTRY_CONDITION_DUPLICATE_IDENTITY"
            )
        identities.add(key)
        rows.append(row)
    return tuple(rows)


def _finite(value: Any, identity: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise EntryConditionStudyError(
            f"AEGIS_ENTRY_CONDITION_NONFINITE:{identity}"
        )
    return result


def _context(regime: Mapping[str, Any]) -> str:
    direction = str(regime["direction"])
    structure = str(regime["structure"])
    if structure == "RANGE":
        return "RANGE"
    if direction == "BEARISH" and structure == "TREND":
        return "BEAR_TREND"
    return "OTHER"


def _selected(
    row: Mapping[str, Any],
    hypothesis: EntryConditionHypothesis,
) -> bool:
    if hypothesis.require_control_selected and not bool(row["control_selected"]):
        return False
    if str(row["symbol"]) in hypothesis.excluded_symbols:
        return False
    return (
        "ANY" in hypothesis.allowed_contexts
        or str(row["factorized_context"]) in hypothesis.allowed_contexts
    )


def _non_overlapping(
    rows: Sequence[Mapping[str, Any]],
    embargo_minutes: int,
) -> tuple[Mapping[str, Any], ...]:
    selected = []
    for symbol in CANONICAL_SYMBOLS:
        previous = None
        for row in sorted(
            (item for item in rows if str(item["symbol"]) == symbol),
            key=lambda item: _timestamp(item["signal_timestamp"]),
        ):
            current = _timestamp(row["signal_timestamp"])
            if (
                previous is None
                or current - previous >= timedelta(minutes=embargo_minutes)
            ):
                selected.append(row)
                previous = current
    return tuple(selected)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * (position - lower)


def _bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> tuple[float | None, float | None]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        stamp = _timestamp(row["signal_timestamp"])
        key = stamp.replace(minute=0, second=0, microsecond=0).isoformat()
        clusters[key].append(_finite(row["net"], "net"))
    values = [statistics.fmean(clusters[key]) for key in sorted(clusters)]
    if len(values) < 2:
        return None, None
    generator = random.Random(seed)
    means = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(repetitions)
    )
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        math.fsum((x - left_mean) ** 2 for x in left)
        * math.fsum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def _ranking(rows: Sequence[Mapping[str, Any]], field: str) -> Mapping[str, Any]:
    if not rows:
        return {
            "count": 0,
            "correlation_with_net_return": None,
            "top_minus_bottom_decile_mean_net_return": None,
            "higher_rank_is_better": False,
        }
    ordered = sorted(rows, key=lambda row: _finite(row[field], field))
    size = max(1, len(ordered) // 10)
    bottom = ordered[:size]
    top = ordered[-size:]
    difference = statistics.fmean(
        _finite(row["net"], "net") for row in top
    ) - statistics.fmean(_finite(row["net"], "net") for row in bottom)
    correlation = _correlation(
        [_finite(row[field], field) for row in rows],
        [_finite(row["net"], "net") for row in rows],
    )
    return {
        "count": len(rows),
        "correlation_with_net_return": correlation,
        "top_minus_bottom_decile_mean_net_return": difference,
        "higher_rank_is_better": difference > 0.0,
    }


def _axis_health(
    values: Sequence[str],
    *,
    minimum_observations: int,
    maximum_dominant_fraction: float,
    minimum_labels: int,
    minimum_transitions: int,
) -> Mapping[str, Any]:
    counts = Counter(values)
    transitions = sum(left != right for left, right in zip(values, values[1:]))
    known = {key: value for key, value in counts.items() if key != "UNKNOWN"}
    dominant = max(known.values()) / len(values) if known and values else None
    ready = len(values) >= minimum_observations
    return {
        "observations": len(values),
        "counts": dict(sorted(counts.items())),
        "dominant_fraction": dominant,
        "transitions": transitions,
        "evidence_ready": ready,
        "healthy": bool(
            ready
            and dominant is not None
            and dominant <= maximum_dominant_fraction
            and len(known) >= minimum_labels
            and transitions >= minimum_transitions
        ),
    }


def _regime_health(
    rows: Sequence[Mapping[str, Any]],
    config: EntryConditionStudyConfig,
) -> Mapping[str, Any]:
    by_symbol = {}
    for symbol in CANONICAL_SYMBOLS:
        population = sorted(
            (row for row in rows if str(row["symbol"]) == symbol),
            key=lambda row: _timestamp(row["signal_timestamp"]),
        )
        directions = [str(row["regime"]["direction"]) for row in population]
        structures = [str(row["regime"]["structure"]) for row in population]
        by_symbol[symbol] = {
            "direction": _axis_health(
                directions,
                minimum_observations=config.regime_minimum_observations,
                maximum_dominant_fraction=(
                    config.regime_maximum_dominant_fraction
                ),
                minimum_labels=config.regime_minimum_direction_labels,
                minimum_transitions=config.regime_minimum_transitions,
            ),
            "structure": _axis_health(
                structures,
                minimum_observations=config.regime_minimum_observations,
                maximum_dominant_fraction=(
                    config.regime_maximum_dominant_fraction
                ),
                minimum_labels=config.regime_minimum_structure_labels,
                minimum_transitions=config.regime_minimum_transitions,
            ),
        }
    ready = all(
        axes["direction"]["evidence_ready"]
        and axes["structure"]["evidence_ready"]
        for axes in by_symbol.values()
    )
    healthy = ready and all(
        axes["direction"]["healthy"] and axes["structure"]["healthy"]
        for axes in by_symbol.values()
    )
    return {
        "state": (
            "HEALTHY"
            if healthy
            else "COLLECTING"
            if not ready
            else "REGIME_COLLAPSE_DETECTED"
        ),
        "all_symbols_evidence_ready": ready,
        "all_symbols_healthy": healthy,
        "by_symbol": by_symbol,
    }


def _metrics(
    population: Sequence[Mapping[str, Any]],
    hypothesis: EntryConditionHypothesis,
    config: EntryConditionStudyConfig,
) -> Mapping[str, Any]:
    selected = [row for row in population if _selected(row, hypothesis)]
    independent = _non_overlapping(selected, config.minimum_embargo_minutes)
    returns = [_finite(row["net"], "net") for row in independent]
    gains = math.fsum(value for value in returns if value > 0.0)
    losses = -math.fsum(value for value in returns if value < 0.0)
    ci_low, ci_high = _bootstrap(
        independent,
        repetitions=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    symbol_counts = {
        symbol: sum(str(row["symbol"]) == symbol for row in independent)
        for symbol in CANONICAL_SYMBOLS
        if symbol not in hypothesis.excluded_symbols
    }
    concentration = (
        max(symbol_counts.values()) / len(independent)
        if independent and symbol_counts
        else 1.0
    )
    midpoint = len(independent) // 2
    ordered = sorted(
        independent, key=lambda row: _timestamp(row["signal_timestamp"])
    )
    first = ordered[:midpoint]
    second = ordered[midpoint:]
    first_mean = (
        statistics.fmean(_finite(row["net"], "net") for row in first)
        if first
        else None
    )
    second_mean = (
        statistics.fmean(_finite(row["net"], "net") for row in second)
        if second
        else None
    )
    temporal_blocks = {
        str(row["signal_timestamp"])[:10] for row in independent
    }
    profit_factor: float | str | None = (
        gains / losses
        if losses > 0.0
        else "INFINITE"
        if gains > 0.0
        else None
    )
    mean_net = statistics.fmean(returns) if returns else None
    mean_mae = (
        statistics.fmean(_finite(row["mae"], "mae") for row in independent)
        if independent
        else None
    )
    required_symbols = [
        symbol
        for symbol in CANONICAL_SYMBOLS
        if symbol not in hypothesis.excluded_symbols
    ]
    checks = {
        "minimum_selected": len(independent) >= config.minimum_selected,
        "minimum_per_symbol": all(
            symbol_counts.get(symbol, 0) >= config.minimum_per_symbol
            for symbol in required_symbols
        ),
        "minimum_temporal_blocks": (
            len(temporal_blocks) >= config.minimum_temporal_blocks
        ),
        "maximum_symbol_concentration": (
            concentration <= config.maximum_symbol_concentration
        ),
        "positive_expectancy": mean_net is not None and mean_net > 0.0,
        "expectancy_ci95_low_above_zero": (
            not config.require_positive_ci
            or (ci_low is not None and ci_low > 0.0)
        ),
        "maximum_mean_mae": (
            mean_mae is not None and mean_mae <= config.maximum_mean_mae
        ),
        "minimum_profit_factor": (
            profit_factor == "INFINITE"
            or (
                isinstance(profit_factor, float)
                and profit_factor > config.minimum_profit_factor
            )
        ),
        "positive_temporal_halves": (
            not config.require_positive_halves
            or (
                first_mean is not None
                and second_mean is not None
                and first_mean > 0.0
                and second_mean > 0.0
            )
        ),
    }
    return {
        "selected_outcomes": len(selected),
        "independent_selected_outcomes": len(independent),
        "mean_net_return": mean_net,
        "win_rate": (
            sum(value > 0.0 for value in returns) / len(returns)
            if returns
            else None
        ),
        "profit_factor": profit_factor,
        "mean_mae": mean_mae,
        "p90_mae": _percentile(
            [_finite(row["mae"], "mae") for row in independent], 0.90
        ),
        "expectancy_ci95": {"low": ci_low, "high": ci_high},
        "first_half_mean_net_return": first_mean,
        "second_half_mean_net_return": second_mean,
        "temporal_blocks": len(temporal_blocks),
        "symbol_counts": symbol_counts,
        "symbol_concentration": concentration,
        "checks": checks,
        "evidence_passed": all(checks.values()),
    }


def _runtime_modes(config: EntryConditionStudyConfig) -> Mapping[str, Any]:
    v2 = _mapping(
        yaml.safe_load(config.v2_runtime_config.read_text()), "v2 runtime"
    )
    long = _mapping(
        yaml.safe_load(config.long_runtime_config.read_text()), "long runtime"
    )
    safe = (
        v2.get("mode") == "SHADOW"
        and long.get("mode") == "SHADOW"
        and not bool(long.get("promotion", {}).get("automatic_live_activation"))
        and not bool(long.get("selection", {}).get("production_authority"))
    )
    if not safe:
        raise EntryConditionStudyError(
            "AEGIS_ENTRY_CONDITION_SHADOW_AUTHORITY_INVALID"
        )
    return {
        "entry_quality_v2": str(v2["mode"]),
        "entry_quality_v3_long": str(long["mode"]),
        "long_production_authority": bool(
            long["selection"]["production_authority"]
        ),
        "automatic_live_activation": False,
    }


def evaluate_entry_condition_study(
    config: EntryConditionStudyConfig,
    *,
    now: datetime | None = None,
) -> Mapping[str, Any]:
    signal_payload = _stable_payload(config.signal_journal)
    outcome_payload = _stable_payload(config.outcome_journal)
    signals = _parse_rows(signal_payload, "event_id")
    outcomes = _parse_rows(outcome_payload, "event_id")
    outcome_by_id = {str(row["event_id"]): row for row in outcomes}
    joined = []
    orphan_outcomes = set(outcome_by_id)
    for signal in signals:
        event_id = str(signal["event_id"])
        outcome = outcome_by_id.get(event_id)
        if outcome is None:
            continue
        orphan_outcomes.discard(event_id)
        regime = _mapping(_mapping(signal["v2"], "v2")["regime"], "regime")
        joined.append(
            {
                "event_id": event_id,
                "symbol": str(signal["symbol"]),
                "signal_timestamp": str(signal["market_timestamp"]),
                "control_selected": bool(signal["control"]["selected"]),
                "factorized_context": _context(regime),
                "regime": dict(regime),
                "score": _finite(signal["v2"]["score"], "score"),
                "opportunity_probability": _finite(
                    signal["v2"]["opportunity_probability"],
                    "opportunity_probability",
                ),
                "net": _finite(
                    outcome["net_return_fraction"], "net_return_fraction"
                ),
                "mae": _finite(outcome["mae_fraction"], "mae_fraction"),
                "mfe": _finite(outcome["mfe_fraction"], "mfe_fraction"),
            }
        )
    discovery = [
        row
        for row in joined
        if _timestamp(row["signal_timestamp"]) < config.evidence_start
    ]
    prospective = [
        row
        for row in joined
        if _timestamp(row["signal_timestamp"]) >= config.evidence_start
    ]
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    observation_days = max(
        0.0, (current - config.evidence_start).total_seconds() / 86_400.0
    )
    regime_health = _regime_health(prospective, config)
    prospective_metrics = {
        hypothesis.identifier: {
            "description": hypothesis.description,
            "excluded_symbols": list(hypothesis.excluded_symbols),
            **_metrics(prospective, hypothesis, config),
        }
        for hypothesis in config.hypotheses
    }
    discovery_metrics = {
        hypothesis.identifier: _metrics(discovery, hypothesis, config)
        for hypothesis in config.hypotheses
    }
    time_checks = {
        "minimum_observation_days": (
            observation_days >= config.minimum_observation_days
        ),
        "maximum_observation_window_exceeded": (
            observation_days > config.maximum_observation_days
        ),
    }
    ranking = {
        "score": _ranking(prospective, "score"),
        "opportunity_probability": _ranking(
            prospective, "opportunity_probability"
        ),
        "recalibration_training_allowed_now": False,
        "historical_replay_required": True,
        "purged_walk_forward_required": True,
    }
    evidence_ready = (
        time_checks["minimum_observation_days"]
        and not time_checks["maximum_observation_window_exceeded"]
        and regime_health["all_symbols_healthy"]
        and any(
            bool(result["evidence_passed"])
            for result in prospective_metrics.values()
        )
    )
    report = {
        "schema_id": "aegis-entry-condition-shadow-study-report-v1",
        "study_id": config.study_id,
        "generated_at": current.isoformat(),
        "runtime_authority": "OBSERVATIONAL_ONLY",
        "evidence_start_utc": config.evidence_start.isoformat(),
        "observation_days": observation_days,
        "source": {
            "signal_path": str(config.signal_journal),
            "signal_sha256": hashlib.sha256(signal_payload).hexdigest(),
            "signal_count": len(signals),
            "outcome_path": str(config.outcome_journal),
            "outcome_sha256": hashlib.sha256(outcome_payload).hexdigest(),
            "outcome_count": len(outcomes),
            "joined_count": len(joined),
            "orphan_outcome_count": len(orphan_outcomes),
            "discovery_rows": len(discovery),
            "prospective_rows": len(prospective),
        },
        "time_checks": time_checks,
        "runtime_modes": _runtime_modes(config),
        "regime_health": regime_health,
        "ranking_diagnostics": ranking,
        "discovery_only": {
            "promotion_use": "PROHIBITED",
            "hypotheses": discovery_metrics,
        },
        "prospective_validation": prospective_metrics,
        "readiness": {
            "state": (
                "READY_FOR_OWNER_REVIEW"
                if evidence_ready
                else "COLLECTING_PROSPECTIVE_SHADOW_EVIDENCE"
            ),
            "evidence_ready": evidence_ready,
            "automatic_training": False,
            "automatic_promotion": False,
            "live_configuration_changes": False,
            "owner_authorization_required": True,
        },
        "exchange_mutations": 0,
    }
    return report


def write_entry_condition_study_report(
    report: Mapping[str, Any],
    path: Path,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        handle.write(canonical_json(report) + "\n")
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    return sha256_file(path)
