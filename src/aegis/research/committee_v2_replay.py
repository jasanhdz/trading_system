"""Deterministic paired replay for the specialized Committee V2 observer."""

from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from ..config import CANONICAL_SYMBOLS, load_brain_config
from ..domain import (
    Candle,
    FeedQuality,
    MarketSnapshot,
    PortfolioContext,
    SymbolSeries,
)
from ..live_decision import CurrentBrainEngine
from ..training.run_state import atomic_write_json
from ..utils import sha256_file
from .committee_v2_shadow import (
    REVERSAL_FLAG_FEATURES,
    committee_v2_counterfactual,
)

REPLAY_SCHEMA = "aegis-specialized-committee-v2-replay-contract-v1"
REPORT_SCHEMA = "aegis-specialized-committee-v2-replay-report-v1"


class CommitteeV2ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommitteeV2ReplayConfig:
    replay_id: str
    config_path: Path
    committee_config: Path
    committee_config_sha256: str
    forward_signals: Path
    forward_outcomes: Path
    database: Path
    database_sha256: str
    table: str
    timeframe: str
    information_cutoff: datetime
    replay_start: datetime
    replay_end: datetime
    minimum_history_bars: int
    horizon_bars: int
    round_trip_cost_fraction: float
    symbol_embargo_minutes: int
    global_embargo_minutes: int
    fold_count: int
    bootstrap_resamples: int
    bootstrap_seed: int
    minimum_preliminary_global_episodes: int
    minimum_robust_global_episodes: int
    minimum_retained_coverage: float
    maximum_retained_coverage: float
    minimum_positive_folds: int
    output_root: Path
    fast_report: Path
    historical_report: Path
    combined_report: Path


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CommitteeV2ReplayError(f"{identity} must be a mapping")
    return value


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite(value: Any, identity: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CommitteeV2ReplayError(f"{identity} is non-finite")
    return parsed


def load_committee_v2_replay_config(
    path: Path,
    *,
    repo_root: Path,
) -> CommitteeV2ReplayConfig:
    root = repo_root.resolve()
    resolved = path.resolve()
    try:
        payload = _mapping(
            yaml.safe_load(resolved.read_text(encoding="utf-8")),
            "replay",
        )
        authority = _mapping(payload["authority"], "authority")
        committee = _mapping(payload["committee"], "committee")
        forward = _mapping(payload["forward"], "forward")
        historical = _mapping(payload["historical"], "historical")
        stats = _mapping(payload["statistics"], "statistics")
        gates = _mapping(payload["evidence_gates"], "evidence_gates")
        outputs = _mapping(payload["outputs"], "outputs")
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise CommitteeV2ReplayError(
            "AEGIS_COMMITTEE_V2_REPLAY_CONFIG_INVALID"
        ) from exc
    if (
        payload.get("schema_version") != REPLAY_SCHEMA
        or payload.get("mode") != "REPLAY_ONLY"
        or authority.get("exchange_authority") is not False
        or authority.get("network_access") is not False
        or authority.get("automatic_training") is not False
        or authority.get("automatic_promotion") is not False
        or authority.get("owner_review_required") is not True
    ):
        raise CommitteeV2ReplayError("AEGIS_COMMITTEE_V2_REPLAY_AUTHORITY_INVALID")
    committee_path = _resolve(root, committee["config_path"])
    database = _resolve(root, historical["database"])
    expected_committee_hash = str(committee["config_sha256"])
    expected_database_hash = str(historical["database_sha256"])
    if sha256_file(committee_path) != expected_committee_hash:
        raise CommitteeV2ReplayError(
            "AEGIS_COMMITTEE_V2_REPLAY_COMMITTEE_HASH_MISMATCH"
        )
    if sha256_file(database) != expected_database_hash:
        raise CommitteeV2ReplayError("AEGIS_COMMITTEE_V2_REPLAY_DATABASE_HASH_MISMATCH")
    cutoff = _timestamp(historical["information_cutoff_utc"])
    start = _timestamp(historical["replay_start_utc"])
    end = _timestamp(historical["replay_end_utc"])
    history = int(historical["minimum_history_bars"])
    horizon = int(historical["horizon_bars"])
    cost = _finite(
        historical["round_trip_cost_fraction"],
        "round_trip_cost_fraction",
    )
    symbol_embargo = int(stats["symbol_embargo_minutes"])
    global_embargo = int(stats["global_embargo_minutes"])
    fold_count = int(stats["fold_count"])
    bootstrap = int(stats["bootstrap_resamples"])
    seed = int(stats["bootstrap_seed"])
    minimum_preliminary = int(gates["minimum_preliminary_global_episodes"])
    minimum_robust = int(gates["minimum_robust_global_episodes"])
    minimum_coverage = _finite(
        gates["minimum_retained_coverage"],
        "minimum_retained_coverage",
    )
    maximum_coverage = _finite(
        gates["maximum_retained_coverage"],
        "maximum_retained_coverage",
    )
    minimum_positive_folds = int(gates["minimum_positive_folds"])
    output_root = _resolve(root, outputs["root"])
    data_root = (root / "data").resolve()
    if (
        start <= cutoff
        or end <= start
        or history < 60
        or horizon <= 0
        or not 0.0 <= cost < 0.01
        or symbol_embargo < horizon * 5
        or global_embargo < horizon * 5
        or fold_count < 2
        or bootstrap < 100
        or minimum_preliminary <= 0
        or minimum_robust < minimum_preliminary
        or not 0.0 < minimum_coverage < maximum_coverage <= 1.0
        or not 1 <= minimum_positive_folds <= fold_count
        or output_root != data_root
        and data_root not in output_root.parents
    ):
        raise CommitteeV2ReplayError("AEGIS_COMMITTEE_V2_REPLAY_CONTRACT_INVALID")
    return CommitteeV2ReplayConfig(
        replay_id=str(payload["replay_id"]),
        config_path=resolved,
        committee_config=committee_path,
        committee_config_sha256=expected_committee_hash,
        forward_signals=_resolve(root, forward["signal_journal"]),
        forward_outcomes=_resolve(root, forward["outcome_journal"]),
        database=database,
        database_sha256=expected_database_hash,
        table=str(historical["table"]),
        timeframe=str(historical["timeframe"]),
        information_cutoff=cutoff,
        replay_start=start,
        replay_end=end,
        minimum_history_bars=history,
        horizon_bars=horizon,
        round_trip_cost_fraction=cost,
        symbol_embargo_minutes=symbol_embargo,
        global_embargo_minutes=global_embargo,
        fold_count=fold_count,
        bootstrap_resamples=bootstrap,
        bootstrap_seed=seed,
        minimum_preliminary_global_episodes=minimum_preliminary,
        minimum_robust_global_episodes=minimum_robust,
        minimum_retained_coverage=minimum_coverage,
        maximum_retained_coverage=maximum_coverage,
        minimum_positive_folds=minimum_positive_folds,
        output_root=output_root,
        fast_report=output_root / str(outputs["fast_report"]),
        historical_report=output_root / str(outputs["historical_report"]),
        combined_report=output_root / str(outputs["combined_report"]),
    )


def _jsonl_snapshot(path: Path) -> tuple[tuple[Mapping[str, Any], ...], str]:
    try:
        content = path.read_bytes()
        if content and not content.endswith(b"\n"):
            raise CommitteeV2ReplayError("AEGIS_COMMITTEE_V2_REPLAY_PARTIAL_JOURNAL")
        rows = tuple(
            _mapping(json.loads(line), "journal_row")
            for line in content.decode("utf-8").splitlines()
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CommitteeV2ReplayError(
            "AEGIS_COMMITTEE_V2_REPLAY_JOURNAL_INVALID"
        ) from exc
    identities = [str(row["event_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise CommitteeV2ReplayError("AEGIS_COMMITTEE_V2_REPLAY_DUPLICATE_EVENT")
    import hashlib

    return rows, hashlib.sha256(content).hexdigest()


def _episode(
    *,
    source: str,
    symbol: str,
    timestamp: str,
    features: Mapping[str, Any],
    control_selected: bool,
    control_side: str,
    net: float,
    mae: float,
    mfe: float,
) -> Mapping[str, Any]:
    counterfactual = committee_v2_counterfactual(
        features,
        control_selected=control_selected,
        control_side=control_side,
    )
    return {
        "source": source,
        "symbol": symbol,
        "signal_timestamp": timestamp,
        "control_selected": control_selected,
        "control_side": control_side,
        "committee_action": counterfactual["paper_action"],
        "committee_reason": counterfactual["reason"],
        "risk_flag_count": counterfactual["observed_risk_count"],
        "risk_flags": counterfactual["risk_flags"],
        "net": _finite(net, "net"),
        "mae": _finite(mae, "mae"),
        "mfe": _finite(mfe, "mfe"),
        "exchange_mutations": 0,
    }


def load_forward_episodes(
    config: CommitteeV2ReplayConfig,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    signals, signal_hash = _jsonl_snapshot(config.forward_signals)
    outcomes, outcome_hash = _jsonl_snapshot(config.forward_outcomes)
    by_id = {str(row["event_id"]): row for row in signals}
    episodes = []
    for outcome in outcomes:
        signal = by_id.get(str(outcome["event_id"]))
        if signal is None:
            raise CommitteeV2ReplayError(
                "AEGIS_COMMITTEE_V2_REPLAY_OUTCOME_WITHOUT_SIGNAL"
            )
        control = _mapping(signal["control"], "control")
        episodes.append(
            _episode(
                source="FORWARD_JOURNAL_REPLAY",
                symbol=str(signal["symbol"]),
                timestamp=str(signal["market_timestamp"]),
                features=_mapping(signal["feature_values"], "feature_values"),
                control_selected=bool(control["selected"]),
                control_side=str(control["side"]),
                net=_finite(
                    outcome["net_return_fraction"],
                    "net_return_fraction",
                ),
                mae=_finite(outcome["mae_fraction"], "mae_fraction"),
                mfe=_finite(outcome["mfe_fraction"], "mfe_fraction"),
            )
        )
    return tuple(episodes), {
        "signal_journal_sha256": signal_hash,
        "outcome_journal_sha256": outcome_hash,
        "signal_records": len(signals),
        "outcome_records": len(outcomes),
    }


def _database_symbol(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT"


def _read_candles(
    config: CommitteeV2ReplayConfig,
) -> Mapping[str, Mapping[datetime, Candle]]:
    query_start = config.replay_start - timedelta(
        minutes=5 * config.minimum_history_bars
    )
    query_end = config.replay_end + timedelta(minutes=5 * config.horizon_bars)
    connection = sqlite3.connect(
        f"file:{config.database}?mode=ro",
        uri=True,
    )
    try:
        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        if config.table not in table_names:
            raise CommitteeV2ReplayError("AEGIS_COMMITTEE_V2_REPLAY_TABLE_MISSING")
        by_symbol: dict[str, dict[datetime, Candle]] = {}
        for symbol in CANONICAL_SYMBOLS:
            rows = connection.execute(
                f"SELECT timestamp, open, high, low, close, volume "
                f"FROM {config.table} "
                "WHERE symbol = ? AND timeframe = ? "
                "AND timestamp >= ? AND timestamp <= ? "
                "ORDER BY timestamp",
                (
                    _database_symbol(symbol),
                    config.timeframe,
                    query_start.replace(tzinfo=None).isoformat(sep=" "),
                    query_end.replace(tzinfo=None).isoformat(sep=" "),
                ),
            ).fetchall()
            candles: dict[datetime, Candle] = {}
            for row in rows:
                open_time = _timestamp(row[0])
                if open_time in candles:
                    raise CommitteeV2ReplayError(
                        "AEGIS_COMMITTEE_V2_REPLAY_DUPLICATE_CANDLE"
                    )
                candles[open_time] = Candle(
                    open_time,
                    open_time + timedelta(minutes=5),
                    _finite(row[1], "open"),
                    _finite(row[2], "high"),
                    _finite(row[3], "low"),
                    _finite(row[4], "close"),
                    _finite(row[5], "volume"),
                    True,
                    "LOCAL_READ_ONLY_OHLCV_REPLAY",
                    open_time.isoformat(),
                )
            by_symbol[symbol] = candles
        return by_symbol
    finally:
        connection.close()


def _snapshot(
    candle_maps: Mapping[str, Mapping[datetime, Candle]],
    *,
    closed_at: datetime,
    history_bars: int,
    symbol_set_hash: str,
) -> MarketSnapshot:
    interval = timedelta(minutes=5)
    series = []
    for symbol in CANONICAL_SYMBOLS:
        candles = tuple(
            candle_maps[symbol][closed_at - interval * offset]
            for offset in range(history_bars, 0, -1)
        )
        series.append(
            SymbolSeries(
                symbol,
                candles,
                closed_at,
                FeedQuality(),
            )
        )
    return MarketSnapshot(
        closed_at,
        "5m",
        symbol_set_hash,
        tuple(series),
        PortfolioContext(
            available_slots=1,
            operational_time=closed_at,
        ),
    )


def _outcome(
    candle_maps: Mapping[str, Mapping[datetime, Candle]],
    *,
    symbol: str,
    signal_timestamp: datetime,
    horizon_bars: int,
    cost: float,
) -> tuple[float, float, float]:
    interval = timedelta(minutes=5)
    history = candle_maps[symbol]
    entry = history[signal_timestamp - interval].close
    future = tuple(
        history[signal_timestamp + interval * index] for index in range(horizon_bars)
    )
    gross = (entry - future[-1].close) / entry
    return (
        gross - cost,
        max(0.0, (max(candle.high for candle in future) - entry) / entry),
        max(0.0, (entry - min(candle.low for candle in future)) / entry),
    )


def run_historical_replay(
    config: CommitteeV2ReplayConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    candle_maps = _read_candles(config)
    interval = timedelta(minutes=5)
    timestamps = []
    current = config.replay_start
    while current <= config.replay_end:
        valid = all(
            all(
                current - interval * offset in candle_maps[symbol]
                for offset in range(config.minimum_history_bars, 0, -1)
            )
            and all(
                current + interval * index in candle_maps[symbol]
                for index in range(config.horizon_bars)
            )
            for symbol in CANONICAL_SYMBOLS
        )
        if valid:
            timestamps.append(current)
        current += interval
    engine = CurrentBrainEngine()
    engine.initialize()
    symbol_set_hash = load_brain_config(
        Path(__file__).resolve().parents[3] / "config"
    ).universe.symbol_set_hash
    episodes = []
    selected_count = 0
    for index, closed_at in enumerate(timestamps, start=1):
        batch = engine.evaluate_replay(
            _snapshot(
                candle_maps,
                closed_at=closed_at,
                history_bars=config.minimum_history_bars,
                symbol_set_hash=symbol_set_hash,
            )
        )
        for symbol in CANONICAL_SYMBOLS:
            result = _mapping(batch["results"][symbol], symbol)
            if not bool(result["selected"]):
                continue
            selected_count += 1
            candidate = _mapping(result["candidate"], "candidate")
            net, mae, mfe = _outcome(
                candle_maps,
                symbol=symbol,
                signal_timestamp=closed_at,
                horizon_bars=config.horizon_bars,
                cost=config.round_trip_cost_fraction,
            )
            episodes.append(
                _episode(
                    source="POST_CUTOFF_CAUSAL_REPLAY",
                    symbol=symbol,
                    timestamp=closed_at.isoformat().replace("+00:00", "Z"),
                    features=_mapping(
                        result["research_features"],
                        "research_features",
                    ),
                    control_selected=True,
                    control_side=str(candidate["side"]),
                    net=net,
                    mae=mae,
                    mfe=mfe,
                )
            )
        if progress is not None and (index == len(timestamps) or index % 100 == 0):
            progress(index, len(timestamps))
    return tuple(episodes), {
        "database_sha256": config.database_sha256,
        "information_cutoff_utc": config.information_cutoff.isoformat(),
        "replay_start_utc": config.replay_start.isoformat(),
        "replay_end_utc": config.replay_end.isoformat(),
        "valid_decision_cycles": len(timestamps),
        "control_selected_episodes": selected_count,
        "network_calls": 0,
        "exchange_mutations": 0,
    }


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _performance(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    returns = [_finite(row["net"], "net") for row in rows]
    maes = [_finite(row["mae"], "mae") for row in rows]
    mfes = [_finite(row["mfe"], "mfe") for row in rows]
    worst_count = max(1, math.ceil(len(returns) * 0.10)) if returns else 0
    losses = -math.fsum(value for value in returns if value < 0.0)
    gains = math.fsum(value for value in returns if value > 0.0)
    return {
        "episodes": len(rows),
        "mean_net_return_fraction": (statistics.fmean(returns) if returns else None),
        "median_net_return_fraction": (statistics.median(returns) if returns else None),
        "win_rate": (
            sum(value > 0.0 for value in returns) / len(returns) if returns else None
        ),
        "mean_mae_fraction": statistics.fmean(maes) if maes else None,
        "p90_mae_fraction": _percentile(maes, 0.90),
        "maximum_mae_fraction": max(maes) if maes else None,
        "mean_mfe_fraction": statistics.fmean(mfes) if mfes else None,
        "worst_decile_mean_net_return_fraction": (
            statistics.fmean(sorted(returns)[:worst_count]) if returns else None
        ),
        "profit_factor": (
            gains / losses if losses > 0.0 else "INFINITE" if gains > 0.0 else None
        ),
    }


def _purge(
    rows: Sequence[Mapping[str, Any]],
    *,
    minutes: int,
    per_symbol: bool,
) -> tuple[Mapping[str, Any], ...]:
    selected = []
    previous: dict[str, datetime] = {}
    for row in sorted(rows, key=lambda item: _timestamp(item["signal_timestamp"])):
        key = str(row["symbol"]) if per_symbol else "GLOBAL"
        current = _timestamp(row["signal_timestamp"])
        prior = previous.get(key)
        if prior is None or current - prior >= timedelta(minutes=minutes):
            selected.append(row)
            previous[key] = current
    return tuple(selected)


def _paired_delta(row: Mapping[str, Any]) -> float:
    control = _finite(row["net"], "net")
    committee = control if row["committee_action"] == "ENTER_NOW" else 0.0
    return committee - control


def _bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> tuple[float | None, float | None]:
    clusters: dict[str, list[float]] = {}
    for row in rows:
        timestamp = _timestamp(row["signal_timestamp"])
        key = timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
        clusters.setdefault(key, []).append(_paired_delta(row))
    values = [statistics.fmean(clusters[key]) for key in sorted(clusters)]
    if len(values) < 2:
        return None, None
    generator = random.Random(seed)
    distribution = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(repetitions)
    )
    return (
        _percentile(distribution, 0.025),
        _percentile(distribution, 0.975),
    )


def _folds(
    rows: Sequence[Mapping[str, Any]],
    count: int,
) -> tuple[Mapping[str, Any], ...]:
    ordered = sorted(rows, key=lambda row: _timestamp(row["signal_timestamp"]))
    folds = []
    for fold in range(count):
        start = len(ordered) * fold // count
        end = len(ordered) * (fold + 1) // count
        population = ordered[start:end]
        deltas = [_paired_delta(row) for row in population]
        folds.append(
            {
                "fold": fold + 1,
                "episodes": len(population),
                "start": (population[0]["signal_timestamp"] if population else None),
                "end": (population[-1]["signal_timestamp"] if population else None),
                "mean_paired_delta": (statistics.fmean(deltas) if deltas else None),
                "positive": bool(deltas and statistics.fmean(deltas) > 0.0),
            }
        )
    return tuple(folds)


def assess_committee_v2_replay(
    episodes: Sequence[Mapping[str, Any]],
    config: CommitteeV2ReplayConfig,
    *,
    source: str,
    provenance: Mapping[str, Any],
) -> Mapping[str, Any]:
    selected = [row for row in episodes if bool(row["control_selected"])]
    symbol_purged = _purge(
        selected,
        minutes=config.symbol_embargo_minutes,
        per_symbol=True,
    )
    global_purged = _purge(
        selected,
        minutes=config.global_embargo_minutes,
        per_symbol=False,
    )
    retained = [row for row in global_purged if row["committee_action"] == "ENTER_NOW"]
    waited = [
        row for row in global_purged if row["committee_action"] == "WAIT_CONFIRMATION"
    ]
    coverage = len(retained) / len(global_purged) if global_purged else 0.0
    deltas = [_paired_delta(row) for row in global_purged]
    delta_mean = statistics.fmean(deltas) if deltas else None
    ci_low, ci_high = _bootstrap_delta(
        global_purged,
        repetitions=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    folds = _folds(global_purged, config.fold_count)
    positive_folds = sum(bool(row["positive"]) for row in folds)
    control_performance = _performance(global_purged)
    retained_performance = _performance(retained)
    waited_performance = _performance(waited)
    committee_policy_rows = [
        {
            **row,
            "net": (row["net"] if row["committee_action"] == "ENTER_NOW" else 0.0),
            "mae": (row["mae"] if row["committee_action"] == "ENTER_NOW" else 0.0),
            "mfe": (row["mfe"] if row["committee_action"] == "ENTER_NOW" else 0.0),
        }
        for row in global_purged
    ]
    committee_policy_performance = _performance(committee_policy_rows)
    preliminary_checks = {
        "minimum_sample": (
            len(global_purged) >= config.minimum_preliminary_global_episodes
        ),
        "coverage": (
            config.minimum_retained_coverage
            <= coverage
            <= config.maximum_retained_coverage
        ),
        "positive_paired_delta": bool(delta_mean is not None and delta_mean > 0.0),
        "non_increasing_retained_mae": bool(
            retained_performance["mean_mae_fraction"] is not None
            and control_performance["mean_mae_fraction"] is not None
            and retained_performance["mean_mae_fraction"]
            <= control_performance["mean_mae_fraction"]
        ),
        "non_degrading_policy_tail": bool(
            committee_policy_performance["worst_decile_mean_net_return_fraction"]
            is not None
            and control_performance["worst_decile_mean_net_return_fraction"] is not None
            and committee_policy_performance["worst_decile_mean_net_return_fraction"]
            >= control_performance["worst_decile_mean_net_return_fraction"]
        ),
        "positive_folds": (positive_folds >= config.minimum_positive_folds),
    }
    robust_checks = {
        **preliminary_checks,
        "minimum_robust_sample": (
            len(global_purged) >= config.minimum_robust_global_episodes
        ),
        "positive_ci95_lower_bound": bool(ci_low is not None and ci_low > 0.0),
    }
    preliminary_pass = all(preliminary_checks.values())
    robust_pass = all(robust_checks.values())
    verdict = (
        "ROBUST_INCREMENTAL_VALUE_SUPPORTED"
        if robust_pass
        else (
            "PRELIMINARY_INCREMENTAL_VALUE_SUPPORTED"
            if preliminary_pass
            else (
                "NO_INCREMENTAL_VALUE_DETECTED"
                if preliminary_checks["minimum_sample"]
                else "INSUFFICIENT_INDEPENDENT_EVIDENCE"
            )
        )
    )
    by_symbol = {}
    for symbol in CANONICAL_SYMBOLS:
        population = [row for row in symbol_purged if row["symbol"] == symbol]
        symbol_deltas = [_paired_delta(row) for row in population]
        by_symbol[symbol] = {
            "control": _performance(population),
            "retained": _performance(
                [row for row in population if row["committee_action"] == "ENTER_NOW"]
            ),
            "waited": _performance(
                [
                    row
                    for row in population
                    if row["committee_action"] == "WAIT_CONFIRMATION"
                ]
            ),
            "mean_paired_delta": (
                statistics.fmean(symbol_deltas) if symbol_deltas else None
            ),
        }
    flag_ablations = {}
    for flag in REVERSAL_FLAG_FEATURES:
        flagged = [row for row in global_purged if bool(row["risk_flags"][flag])]
        clear = [row for row in global_purged if not bool(row["risk_flags"][flag])]
        flag_ablations[flag] = {
            "flagged": _performance(flagged),
            "clear": _performance(clear),
        }
    return {
        "schema_id": REPORT_SCHEMA,
        "replay_id": config.replay_id,
        "source": source,
        "evaluated_through": (
            max(str(row["signal_timestamp"]) for row in episodes) if episodes else None
        ),
        "provenance": dict(provenance),
        "population": {
            "matured_episodes": len(episodes),
            "control_selected_episodes": len(selected),
            "symbol_purged_episodes": len(symbol_purged),
            "global_purged_episodes": len(global_purged),
            "retained_enter_now": len(retained),
            "wait_confirmation": len(waited),
            "retained_coverage": coverage,
        },
        "performance": {
            "control": control_performance,
            "committee_policy": committee_policy_performance,
            "retained_enter_now": retained_performance,
            "wait_confirmation": waited_performance,
            "mean_paired_delta_fraction": delta_mean,
            "paired_delta_ci95": {"low": ci_low, "high": ci_high},
        },
        "walk_forward": {
            "fold_count": config.fold_count,
            "positive_folds": positive_folds,
            "folds": list(folds),
        },
        "per_symbol": by_symbol,
        "flag_ablations": flag_ablations,
        "gates": {
            "preliminary": preliminary_checks,
            "robust": robust_checks,
        },
        "verdict": verdict,
        "limitations": {
            "wait_confirmation_entry_policy": "NOT_DEFINED",
            "wait_is_evaluated_as_abstention": True,
            "historical_replay_cannot_replace_prospective_shadow": True,
        },
        "training_performed": False,
        "automatic_promotion": False,
        "network_calls": 0,
        "exchange_mutations": 0,
    }


def write_replay_report(path: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(path, report)
    os.chmod(path, 0o600)
