"""Build hash-bound Live feedback evidence without changing the active model."""

from __future__ import annotations

import glob
import json
import math
import os
import random
import sqlite3
import statistics
import tempfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..config import CANONICAL_SYMBOLS
from ..domain import Candle
from ..features import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ..training.labels import (
    SHORT_LABEL_SCHEMA_VERSION,
    ShortLabelConfig,
    ShortPathLabel,
    build_short_path_label,
)
from ..utils import canonical_json, sha256_file


class LiveFeedbackError(ValueError):
    pass


@dataclass(frozen=True)
class LiveFeedbackConfig:
    schema_version: str
    signal_journal: Path
    outcome_journal: Path
    trade_logs_glob: str | None
    dataset_path: Path
    report_path: Path
    expected_symbols: tuple[str, ...]
    horizon_bars: int
    minimum_non_overlapping_episodes: int
    minimum_embargo_minutes: int
    minimum_challenger_selected_outcomes: int
    minimum_selected_non_overlapping_episodes: int
    maximum_symbol_concentration: float
    bootstrap_resamples: int
    bootstrap_seed: int
    automatic_training: bool
    automatic_promotion: bool
    historical_replay_required: bool
    live_only_training_allowed: bool
    purged_walk_forward_required: bool
    champion_challenger_required: bool
    owner_promotion_required: bool

    def __post_init__(self) -> None:
        if self.schema_version != "aegis-entry-quality-live-feedback-v1":
            raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_CONFIG_INVALID")
        if not self.expected_symbols or self.horizon_bars <= 0:
            raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_POPULATION_INVALID")
        if self.minimum_non_overlapping_episodes <= 0:
            raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_EVIDENCE_INVALID")
        if self.minimum_embargo_minutes < self.horizon_bars * 5:
            raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_EMBARGO_INVALID")
        if min(
            self.minimum_challenger_selected_outcomes,
            self.minimum_selected_non_overlapping_episodes,
            self.bootstrap_resamples,
        ) <= 0:
            raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_SELECTION_EVIDENCE_INVALID")
        if not 0.0 < self.maximum_symbol_concentration <= 1.0:
            raise LiveFeedbackError(
                "AEGIS_LIVE_FEEDBACK_SYMBOL_CONCENTRATION_INVALID"
            )
        if self.automatic_training or self.automatic_promotion:
            raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_AUTOMATION_PROHIBITED")
        if (
            not self.historical_replay_required
            or self.live_only_training_allowed
            or not self.purged_walk_forward_required
            or not self.champion_challenger_required
            or not self.owner_promotion_required
        ):
            raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_TRAINING_POLICY_INVALID")


def _mapping(value: Any, identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveFeedbackError(f"{identity} must be a mapping")
    return value


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def load_live_feedback_config(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> LiveFeedbackConfig:
    root = (repo_root or path.resolve().parents[2]).resolve()
    payload = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "feedback")
    inputs = _mapping(payload["inputs"], "inputs")
    outputs = _mapping(payload["outputs"], "outputs")
    evidence = _mapping(payload["evidence"], "evidence")
    automation = _mapping(payload["automation"], "automation")
    training_controls = _mapping(
        payload["training_controls"], "training_controls"
    )
    labels = _mapping(payload["labels"], "labels")
    if (
        labels.get("schema") != SHORT_LABEL_SCHEMA_VERSION
        or int(labels["horizon_bars"]) != ShortLabelConfig().horizon_bars
    ):
        raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_LABEL_AUTHORITY_MISMATCH")
    symbols = tuple(str(item) for item in payload["symbols"])
    if symbols != tuple(CANONICAL_SYMBOLS):
        raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_SYMBOL_AUTHORITY_MISMATCH")
    trade_glob = inputs.get("trade_logs_glob")
    return LiveFeedbackConfig(
        schema_version=str(payload["schema_version"]),
        signal_journal=_resolve(root, inputs["signal_journal"]),
        outcome_journal=_resolve(root, inputs["outcome_journal"]),
        trade_logs_glob=(
            str(_resolve(root, trade_glob)) if trade_glob is not None else None
        ),
        dataset_path=_resolve(root, outputs["dataset_path"]),
        report_path=_resolve(root, outputs["report_path"]),
        expected_symbols=symbols,
        horizon_bars=int(labels["horizon_bars"]),
        minimum_non_overlapping_episodes=int(
            evidence["minimum_non_overlapping_episodes"]
        ),
        minimum_embargo_minutes=int(evidence["minimum_embargo_minutes"]),
        minimum_challenger_selected_outcomes=int(
            evidence.get("minimum_challenger_selected_outcomes", 50)
        ),
        minimum_selected_non_overlapping_episodes=int(
            evidence.get("minimum_selected_non_overlapping_episodes", 30)
        ),
        maximum_symbol_concentration=float(
            evidence.get("maximum_symbol_concentration", 0.30)
        ),
        bootstrap_resamples=int(evidence.get("bootstrap_resamples", 2000)),
        bootstrap_seed=int(evidence.get("bootstrap_seed", 20260726)),
        automatic_training=bool(automation["automatic_training"]),
        automatic_promotion=bool(automation["automatic_promotion"]),
        historical_replay_required=bool(
            training_controls["historical_replay_required"]
        ),
        live_only_training_allowed=bool(
            training_controls["live_only_training_allowed"]
        ),
        purged_walk_forward_required=bool(
            training_controls["purged_walk_forward_required"]
        ),
        champion_challenger_required=bool(
            training_controls["champion_challenger_required"]
        ),
        owner_promotion_required=bool(
            training_controls["owner_promotion_required"]
        ),
    )


def _timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_TIMESTAMP_INVALID")
    return parsed.astimezone(timezone.utc)


def _rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield _mapping(json.loads(line), f"{path}:{line_number}")
            except json.JSONDecodeError as exc:
                raise LiveFeedbackError(
                    "AEGIS_LIVE_FEEDBACK_JSONL_INVALID"
                ) from exc


def _candle(row: Mapping[str, Any]) -> Candle:
    close_time = _timestamp(row["market_timestamp"])
    bar = _mapping(row["market_bar"], "market_bar")
    return Candle(
        open_time=close_time - timedelta(minutes=5),
        close_time=close_time,
        open=float(bar["open"]),
        high=float(bar["high"]),
        low=float(bar["low"]),
        close=float(bar["close"]),
        volume=0.0,
        is_closed=True,
        source="ENTRY_QUALITY_V2_SHADOW",
        sequence=str(row["event_id"]),
    )


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_scale = math.sqrt(math.fsum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(math.fsum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def _average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if not labels or positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / index
    return precision_sum / positives


def _load_trades(pattern: str | None) -> list[dict[str, Any]]:
    if pattern is None:
        return []
    records: dict[str, dict[str, Any]] = {}
    for filename in sorted(glob.glob(pattern)):
        for row in _rows(Path(filename)):
            trade_id = str(row.get("trade_id", ""))
            if not trade_id or str(row.get("side")) != "SHORT":
                continue
            record = records.setdefault(trade_id, {"trade_id": trade_id})
            if row.get("status") == "OPEN":
                record["open"] = dict(row)
            elif row.get("status") == "CLOSED":
                record["close"] = dict(row)
    result = []
    for record in records.values():
        opened = record.get("open", {}).get("opened_at")
        if opened:
            record["opened_at"] = _timestamp(opened)
            result.append(record)
    return sorted(result, key=lambda item: item["opened_at"])


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(0, index)]


def _cluster_bootstrap_interval(
    records: Sequence[Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """Bootstrap timestamp clusters so simultaneous symbols are not independent."""
    by_timestamp: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_timestamp[str(row["timestamp"])].append(float(row["net"]))
    cluster_means = [
        statistics.fmean(values)
        for _, values in sorted(by_timestamp.items())
    ]
    if len(cluster_means) < 2:
        return None, None
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(cluster_means) for _ in cluster_means)
        for _ in range(resamples)
    )
    return (
        means[max(0, int(0.025 * (len(means) - 1)))],
        means[min(len(means) - 1, int(0.975 * (len(means) - 1)))],
    )


def _selection_outcome_metrics(
    evidence: Sequence[Mapping[str, Any]],
    *,
    selection_field: str,
    config: LiveFeedbackConfig,
) -> Mapping[str, Any]:
    selected = [row for row in evidence if bool(row[selection_field])]
    independent = [row for row in selected if bool(row["non_overlapping"])]
    returns = [float(row["net"]) for row in selected]
    independent_returns = [float(row["net"]) for row in independent]
    symbol_counts = {
        symbol: sum(str(row["symbol"]) == symbol for row in selected)
        for symbol in sorted({str(row["symbol"]) for row in selected})
    }
    low, high = _cluster_bootstrap_interval(
        independent,
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
    )
    concentration = (
        max(symbol_counts.values()) / len(selected) if selected else 1.0
    )
    evidence_passed = (
        len(selected) >= config.minimum_challenger_selected_outcomes
        and len(independent)
        >= config.minimum_selected_non_overlapping_episodes
        and bool(independent_returns)
        and statistics.fmean(independent_returns) > 0.0
        and low is not None
        and low > 0.0
        and concentration <= config.maximum_symbol_concentration
    )
    return {
        "selected_outcomes": len(selected),
        "selection_rate": len(selected) / len(evidence) if evidence else 0.0,
        "non_overlapping_selected_outcomes": len(independent),
        "mean_net_return": (
            statistics.fmean(returns) if returns else None
        ),
        "mean_non_overlapping_net_return": (
            statistics.fmean(independent_returns)
            if independent_returns
            else None
        ),
        "non_overlapping_expectancy_ci95": {
            "low": low,
            "high": high,
            "cluster_unit": "SIGNAL_TIMESTAMP",
        },
        "win_rate": (
            sum(value > 0.0 for value in returns) / len(returns)
            if returns
            else None
        ),
        "bad_entry_rate": (
            sum(bool(row["bad"]) for row in selected) / len(selected)
            if selected
            else None
        ),
        "clean_entry_rate": (
            sum(bool(row["clean"]) for row in selected) / len(selected)
            if selected
            else None
        ),
        "mean_mae": (
            statistics.fmean(float(row["mae"]) for row in selected)
            if selected
            else None
        ),
        "p90_mae": _percentile(
            [float(row["mae"]) for row in selected],
            0.90,
        ),
        "symbol_counts": symbol_counts,
        "symbol_concentration": concentration,
        "delta_vs_abstention": (
            statistics.fmean(independent_returns)
            if independent_returns
            else None
        ),
        "positive_selection_evidence": evidence_passed,
        "selection_evidence_requirements": {
            "minimum_selected_outcomes": (
                config.minimum_challenger_selected_outcomes
            ),
            "minimum_non_overlapping_selected_outcomes": (
                config.minimum_selected_non_overlapping_episodes
            ),
            "maximum_symbol_concentration": (
                config.maximum_symbol_concentration
            ),
            "expectancy_ci95_low_must_exceed_zero": True,
        },
    }


def _execution_quality(trades: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    opens = [_mapping(row.get("open", {}), "trade.open") for row in trades]
    closes = [
        _mapping(row.get("close", {}), "trade.close")
        for row in trades
        if row.get("close")
    ]
    fill_prices = [float(row.get("entry_price", 0.0)) for row in opens]
    bracket_confirmed = [
        bool(row.get("brackets_confirmed")) for row in opens
    ]
    return {
        "source": "LOCAL_TRADE_JOURNALS",
        "opened_trade_records": len(opens),
        "closed_trade_records": len(closes),
        "open_records_without_close": len(opens) - len(closes),
        "fill_price_present_count": sum(value > 0.0 for value in fill_prices),
        "fill_price_capture_rate": (
            sum(value > 0.0 for value in fill_prices) / len(opens)
            if opens
            else None
        ),
        "bracket_confirmed_count": sum(bracket_confirmed),
        "bracket_confirmation_rate": (
            sum(bracket_confirmed) / len(opens) if opens else None
        ),
        "realized_pnl_available_count": sum(
            isinstance(row.get("pnl_usdt"), (int, float)) for row in closes
        ),
        "signal_quality_inference_prohibited": True,
    }


def _match_trade(
    signal: Mapping[str, Any],
    trades: Sequence[dict[str, Any]],
    used_trade_ids: set[str],
) -> Mapping[str, Any] | None:
    if not bool(_mapping(signal["control"], "control").get("selected")):
        return None
    timestamp = _timestamp(signal["market_timestamp"])
    candidates = [
        trade
        for trade in trades
        if trade["trade_id"] not in used_trade_ids
        and str(trade.get("open", {}).get("symbol")) == str(signal["symbol"])
        and timestamp <= trade["opened_at"] <= timestamp + timedelta(minutes=5)
    ]
    if not candidates:
        return None
    trade = min(candidates, key=lambda item: item["opened_at"])
    used_trade_ids.add(str(trade["trade_id"]))
    opened = _mapping(trade["open"], "trade.open")
    closed = _mapping(trade.get("close", {}), "trade.close")
    signal_price = float(_mapping(signal["market_bar"], "market_bar")["close"])
    fill_price = float(opened["entry_price"])
    return {
        "trade_id": trade["trade_id"],
        "opened_at": opened.get("opened_at"),
        "closed_at": closed.get("closed_at"),
        "entry_price": fill_price,
        "exit_price": closed.get("exit_price"),
        "position_fraction": opened.get("position_fraction"),
        "leverage": opened.get("leverage"),
        "brackets_confirmed": opened.get("brackets_confirmed"),
        "pnl_usdt": closed.get("pnl_usdt"),
        "roe": closed.get("roe"),
        "mfe_roe": closed.get("mfe_roe"),
        "mae_roe": closed.get("mae_roe"),
        "exit_type": _mapping(closed.get("metadata", {}), "trade.metadata").get(
            "exit_type"
        ),
        "execution_slippage_fraction": (signal_price - fill_price) / signal_price,
    }


def _classification(
    signal: Mapping[str, Any],
    label: ShortPathLabel,
    trade: Mapping[str, Any] | None,
) -> str:
    control_selected = bool(_mapping(signal["control"], "control").get("selected"))
    v2_selected = bool(_mapping(signal["v2"], "v2").get("selected"))
    if trade is not None and label.bad_entry:
        return "ACTUAL_BAD_ENTRY_CONFIRMED"
    if control_selected and label.bad_entry and not v2_selected:
        return "CHALLENGER_AVOIDED_BAD_CONTROL_ENTRY"
    if control_selected and label.clean_entry and not v2_selected:
        return "CHALLENGER_MISSED_CLEAN_CONTROL_ENTRY"
    if v2_selected and label.bad_entry:
        return "CHALLENGER_SELECTED_BAD_ENTRY"
    if v2_selected and label.clean_entry:
        return "CHALLENGER_SELECTED_CLEAN_ENTRY"
    if label.bad_entry:
        return "REJECTED_BAD_COUNTERFACTUAL"
    if label.clean_entry:
        return "REJECTED_CLEAN_COUNTERFACTUAL"
    return "AMBIGUOUS_COUNTERFACTUAL"


def _path_metrics(
    signal: Mapping[str, Any],
    future: Sequence[Mapping[str, Any]],
) -> Mapping[str, Mapping[str, float | int]]:
    entry = float(_mapping(signal["market_bar"], "market_bar")["close"])
    result: dict[str, Mapping[str, float | int]] = {}
    for horizon in (3, 6, 12):
        rows = future[:horizon]
        if len(rows) != horizon:
            continue
        bars = [_mapping(row["market_bar"], "market_bar") for row in rows]
        closes = [float(bar["close"]) for bar in bars]
        highs = [float(bar["high"]) for bar in bars]
        lows = [float(bar["low"]) for bar in bars]
        underwater = [close > entry for close in closes]
        longest = current = 0
        for value in underwater:
            current = current + 1 if value else 0
            longest = max(longest, current)
        result[str(horizon)] = {
            "terminal_short_return": (entry - closes[-1]) / entry,
            "mfe_fraction": max(0.0, (entry - min(lows)) / entry),
            "mae_fraction": max(0.0, (max(highs) - entry) / entry),
            "time_underwater_bars": sum(underwater),
            "maximum_consecutive_underwater_bars": longest,
        }
    return result


def _build_row(
    signal: Mapping[str, Any],
    future: Sequence[Mapping[str, Any]],
    outcome: Mapping[str, Any],
    *,
    non_overlapping: bool,
    trade: Mapping[str, Any] | None,
) -> dict[str, Any]:
    label = build_short_path_label(
        _candle(signal),
        tuple(_candle(row) for row in future),
        ShortLabelConfig(),
    )
    v2 = _mapping(signal["v2"], "v2")
    observed_mae = float(outcome["mae_fraction"])
    observed_mfe = float(outcome["mfe_fraction"])
    observed_net = float(outcome["net_return_fraction"])
    if label.valid and (
        not math.isclose(float(label.mae_fraction), observed_mae, abs_tol=1e-12)
        or not math.isclose(float(label.mfe_fraction), observed_mfe, abs_tol=1e-12)
    ):
        raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_OUTCOME_MISMATCH")
    features = _mapping(signal["feature_values"], "feature_values")
    if (
        signal.get("feature_schema") != FEATURE_SCHEMA_VERSION
        or tuple(sorted(features)) != tuple(sorted(FEATURE_NAMES))
        or not all(math.isfinite(float(features[name])) for name in FEATURE_NAMES)
    ):
        raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_FEATURE_CONTRACT_MISMATCH")
    label_payload = asdict(label)
    ratio = label_payload.get("mfe_mae_ratio")
    label_payload["mfe_mae_ratio_unbounded"] = (
        isinstance(ratio, float) and math.isinf(ratio)
    )
    if label_payload["mfe_mae_ratio_unbounded"]:
        label_payload["mfe_mae_ratio"] = None
    row = {
        "schema_id": "aegis-entry-quality-live-feedback-row-v1",
        "event_id": signal["event_id"],
        "decision_cycle_id": signal["decision_cycle_id"],
        "symbol": signal["symbol"],
        "signal_timestamp": signal["market_timestamp"],
        "maturity_timestamp": outcome["maturity_timestamp"],
        "feature_schema": signal["feature_schema"],
        "feature_hash": FEATURE_HASH,
        "feature_vector_hash": signal["feature_vector_hash"],
        "feature_values": {name: float(features[name]) for name in FEATURE_NAMES},
        "control": signal["control"],
        "challenger": v2,
        "label": label_payload,
        "observed": {
            "net_return_fraction": observed_net,
            "mfe_fraction": observed_mfe,
            "mae_fraction": observed_mae,
            "path_metrics": _path_metrics(signal, future),
        },
        "non_overlapping_episode": non_overlapping,
        "actual_trade": trade,
    }
    row["classification"] = _classification(signal, label, trade)
    return row


def build_live_feedback_evidence(config: LiveFeedbackConfig) -> Mapping[str, Any]:
    if not config.signal_journal.is_file() or not config.outcome_journal.is_file():
        raise LiveFeedbackError("AEGIS_LIVE_FEEDBACK_INPUT_MISSING")
    config.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(config.dataset_path.parent, 0o700)
    trades = _load_trades(config.trade_logs_glob)
    used_trade_ids: set[str] = set()
    windows: dict[str, deque[Mapping[str, Any]]] = {
        symbol: deque(maxlen=config.horizon_bars + 1)
        for symbol in config.expected_symbols
    }
    last_timestamp: dict[str, datetime] = {}
    last_non_overlapping: dict[str, datetime] = {}
    metrics: dict[str, list[Any]] = defaultdict(list)
    selection_evidence: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    per_symbol: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    per_regime: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    per_classification: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as temporary:
        database = sqlite3.connect(temporary.name)
        database.execute(
            "CREATE TABLE outcomes (event_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        for outcome in _rows(config.outcome_journal):
            database.execute(
                "INSERT INTO outcomes(event_id, payload) VALUES (?, ?)",
                (str(outcome["event_id"]), canonical_json(outcome)),
            )
        database.commit()

        temporary_dataset = config.dataset_path.with_suffix(".tmp")
        with temporary_dataset.open("w", encoding="utf-8", newline="\n") as output:
            for signal in _rows(config.signal_journal):
                symbol = str(signal["symbol"])
                if symbol not in windows:
                    raise LiveFeedbackError(
                        "AEGIS_LIVE_FEEDBACK_SYMBOL_POPULATION_INVALID"
                    )
                timestamp = _timestamp(signal["market_timestamp"])
                previous = last_timestamp.get(symbol)
                if previous is not None and timestamp <= previous:
                    raise LiveFeedbackError(
                        "AEGIS_LIVE_FEEDBACK_ORDERING_INVALID"
                    )
                last_timestamp[symbol] = timestamp
                window = windows[symbol]
                window.append(signal)
                if len(window) != config.horizon_bars + 1:
                    continue
                source = window[0]
                outcome_row = database.execute(
                    "SELECT payload FROM outcomes WHERE event_id = ?",
                    (str(source["event_id"]),),
                ).fetchone()
                if outcome_row is None:
                    counts["missing_outcomes"] += 1
                    continue
                outcome = _mapping(json.loads(outcome_row[0]), "outcome")
                source_timestamp = _timestamp(source["market_timestamp"])
                previous_episode = last_non_overlapping.get(symbol)
                non_overlapping = (
                    previous_episode is None
                    or source_timestamp - previous_episode
                    >= timedelta(minutes=config.minimum_embargo_minutes)
                )
                if non_overlapping:
                    last_non_overlapping[symbol] = source_timestamp
                trade = _match_trade(source, trades, used_trade_ids)
                row = _build_row(
                    source,
                    list(window)[1:],
                    outcome,
                    non_overlapping=non_overlapping,
                    trade=trade,
                )
                output.write(canonical_json(row) + "\n")
                label = _mapping(row["label"], "label")
                challenger = _mapping(row["challenger"], "challenger")
                control = _mapping(row["control"], "control")
                observed = _mapping(row["observed"], "observed")
                regime = _mapping(challenger["regime"], "regime")
                clean = bool(label["clean_entry"])
                bad = bool(label["bad_entry"])
                counts["rows"] += 1
                counts["clean"] += int(clean)
                counts["bad"] += int(bad)
                counts["non_overlapping"] += int(non_overlapping)
                counts["actual_trades"] += int(trade is not None)
                counts[str(row["classification"])] += 1
                metrics["clean"].append(clean)
                metrics["opportunity"].append(
                    float(challenger["opportunity_probability"])
                )
                metrics["score"].append(float(challenger["score"]))
                metrics["qmae"].append(float(challenger["qmae_q90"]))
                metrics["mae"].append(float(observed["mae_fraction"]))
                metrics["net"].append(float(observed["net_return_fraction"]))
                metrics["qmae_covered"].append(
                    float(observed["mae_fraction"])
                    <= float(challenger["qmae_q90"])
                )
                selection_evidence.append(
                    {
                        "timestamp": row["signal_timestamp"],
                        "symbol": symbol,
                        "net": float(observed["net_return_fraction"]),
                        "mae": float(observed["mae_fraction"]),
                        "clean": clean,
                        "bad": bad,
                        "non_overlapping": non_overlapping,
                        "control_selected": bool(control["selected"]),
                        "challenger_selected": bool(challenger["selected"]),
                    }
                )
                for bucket in (
                    per_symbol[symbol],
                    per_regime[
                        f"{symbol}|{regime['direction']}|"
                        f"{regime['volatility']}|{regime['structure']}"
                    ],
                    per_classification[str(row["classification"])],
                ):
                    bucket["rows"] += 1
                    bucket["clean"] += int(clean)
                    bucket["bad"] += int(bad)
                    bucket["net_sum"] += float(observed["net_return_fraction"])
                    bucket["mae_sum"] += float(observed["mae_fraction"])
                    bucket["control_selected"] += int(
                        bool(control["selected"])
                    )
                    bucket["challenger_selected"] += int(
                        bool(challenger["selected"])
                    )
                    bucket["control_selected_net_sum"] += (
                        float(observed["net_return_fraction"])
                        if bool(control["selected"])
                        else 0.0
                    )
                    bucket["challenger_selected_net_sum"] += (
                        float(observed["net_return_fraction"])
                        if bool(challenger["selected"])
                        else 0.0
                    )
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_dataset, 0o600)
        os.replace(temporary_dataset, config.dataset_path)

    def summarize(values: Mapping[str, float]) -> Mapping[str, float | int]:
        rows = int(values["rows"])
        return {
            "rows": rows,
            "clean_rate": values["clean"] / rows,
            "bad_rate": values["bad"] / rows,
            "mean_net_return": values["net_sum"] / rows,
            "mean_mae": values["mae_sum"] / rows,
            "control_selected": int(values["control_selected"]),
            "control_selected_mean_net_return": (
                values["control_selected_net_sum"]
                / values["control_selected"]
                if values["control_selected"]
                else None
            ),
            "challenger_selected": int(values["challenger_selected"]),
            "challenger_selected_mean_net_return": (
                values["challenger_selected_net_sum"]
                / values["challenger_selected"]
                if values["challenger_selected"]
                else None
            ),
        }

    symbols_present = sorted(symbol for symbol, values in per_symbol.items() if values["rows"])
    evidence_ready = (
        counts["non_overlapping"] >= config.minimum_non_overlapping_episodes
        and symbols_present == sorted(config.expected_symbols)
        and counts["missing_outcomes"] == 0
    )
    control_selection = _selection_outcome_metrics(
        selection_evidence,
        selection_field="control_selected",
        config=config,
    )
    challenger_selection = _selection_outcome_metrics(
        selection_evidence,
        selection_field="challenger_selected",
        config=config,
    )
    positive_selection_ready = bool(
        challenger_selection["positive_selection_evidence"]
    )
    report = {
        "schema_id": "aegis-entry-quality-live-feedback-report-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(config.dataset_path),
        "dataset_sha256": sha256_file(config.dataset_path),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "feature_count": len(FEATURE_NAMES),
        "label_schema": SHORT_LABEL_SCHEMA_VERSION,
        "horizon_bars": config.horizon_bars,
        "counts": dict(sorted(counts.items())),
        "opportunity_metrics": {
            "positive_rate": (
                sum(metrics["clean"]) / len(metrics["clean"])
                if metrics["clean"]
                else None
            ),
            "average_precision": _average_precision(
                metrics["clean"], metrics["opportunity"]
            ),
            "brier": (
                statistics.fmean(
                    (score - float(label)) ** 2
                    for score, label in zip(
                        metrics["opportunity"], metrics["clean"]
                    )
                )
                if metrics["clean"]
                else None
            ),
            "score_return_correlation": _correlation(
                metrics["score"], metrics["net"]
            ),
        },
        "risk_metrics": {
            "qmae_mae_correlation": _correlation(
                metrics["qmae"], metrics["mae"]
            ),
            "qmae_empirical_coverage": (
                sum(metrics["qmae_covered"]) / len(metrics["qmae_covered"])
                if metrics["qmae_covered"]
                else None
            ),
            "mean_realized_mae": (
                statistics.fmean(metrics["mae"]) if metrics["mae"] else None
            ),
        },
        "signal_quality": {
            "population": {
                "outcomes": counts["rows"],
                "non_overlapping_outcomes": counts["non_overlapping"],
                "clean_outcomes": counts["clean"],
                "bad_outcomes": counts["bad"],
            },
            "control_selection": control_selection,
            "challenger_selection": challenger_selection,
            "abstention_baseline": {
                "selected_outcomes": 0,
                "mean_net_return": 0.0,
                "economic_edge_claim": False,
            },
            "challenger_behavior": {
                "avoided_bad_control_entries": counts[
                    "CHALLENGER_AVOIDED_BAD_CONTROL_ENTRY"
                ]
                + counts["ACTUAL_BAD_ENTRY_CONFIRMED"],
                "missed_clean_control_entries": counts[
                    "CHALLENGER_MISSED_CLEAN_CONTROL_ENTRY"
                ],
                "all_entries_rejected": (
                    int(challenger_selection["selected_outcomes"]) == 0
                ),
                "abstention_is_not_positive_edge": True,
            },
        },
        "execution_quality": _execution_quality(trades),
        "per_symbol": {
            key: summarize(value) for key, value in sorted(per_symbol.items())
        },
        "per_regime": {
            key: summarize(value) for key, value in sorted(per_regime.items())
        },
        "per_classification": {
            key: summarize(value)
            for key, value in sorted(per_classification.items())
        },
        "training_readiness": {
            "state": (
                "EVIDENCE_READY_FOR_CONTROLLED_CHALLENGER_TRAINING"
                if evidence_ready
                else "COLLECTING_NON_OVERLAPPING_SHADOW_EVIDENCE"
            ),
            "required_non_overlapping_episodes": (
                config.minimum_non_overlapping_episodes
            ),
            "observed_non_overlapping_episodes": counts["non_overlapping"],
            "all_symbols_present": symbols_present
            == sorted(config.expected_symbols),
            "population_evidence_ready": evidence_ready,
            "positive_selection_evidence_ready": positive_selection_ready,
            "challenger_promotion_evidence_state": (
                "POSITIVE_SELECTION_EVIDENCE_READY"
                if positive_selection_ready
                else "NO_POSITIVE_SELECTION_EVIDENCE"
            ),
            "challenger_selected_outcomes": challenger_selection[
                "selected_outcomes"
            ],
            "challenger_non_overlapping_selected_outcomes": (
                challenger_selection["non_overlapping_selected_outcomes"]
            ),
            "selection_failure": (
                None
                if positive_selection_ready
                else "NO_POSITIVE_SELECTION_EVIDENCE"
            ),
            "automatic_training": False,
            "automatic_promotion": False,
            "historical_replay_required": config.historical_replay_required,
            "live_only_training_allowed": config.live_only_training_allowed,
            "purged_walk_forward_required": config.purged_walk_forward_required,
            "champion_challenger_required": config.champion_challenger_required,
            "owner_promotion_required": config.owner_promotion_required,
        },
        "exchange_mutations": 0,
    }
    temporary_report = config.report_path.with_suffix(".tmp")
    temporary_report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report.write_text(
        canonical_json(report) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary_report, 0o600)
    os.replace(temporary_report, config.report_path)
    return report
