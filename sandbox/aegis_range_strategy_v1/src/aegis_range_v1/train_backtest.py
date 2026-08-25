from __future__ import annotations

import bisect
import copy
import csv
import gzip
import hashlib
import io
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .atr import RangeAtr14V1
from .candidates import RangeCandidate, candidate_grid
from .costs import BASELINE, STRESS_20, STRESS_30, adverse_fill, fee_return, funding_return, gross_return
from .engine import RangeEngineV1
from .lifecycle import RangeLifecycleV1
from .models import Candle5m, Episode, FillEvent, Position, RegimeSnapshot
from .numeric import iso_utc_millis
from .readiness import PARTITIONS, SYMBOLS, SealedPartitionGuard, SourceIntegrityError, verify_r1_immutable
from .regime_bridge import TypeScriptRegimeEvaluator

TRAIN_START, TRAIN_END = PARTITIONS["TRAIN"]
EMBARGO_END = TRAIN_START + timedelta(hours=48)
SCENARIOS = (BASELINE, STRESS_20, STRESS_30)
MAJORS = frozenset({"BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"})
R1_MANIFEST_FILE_SHA256 = "2a55d0a5511b178b8d8c8a5b0a7259ecffe800e59b715234c75ff2eea8639d5c"
DATASET_LOGICAL_SHA256 = "587476db7f427670e0f4225ce06cd4058604d73c3ea885c95feb0492a2589ce8"
REVALIDATION_MANIFEST_FILE_SHA256 = "9fa69860198843a07d14f61116a458738e433939c4ecd7f0c6418ff87512e4d1"
SOURCE_GAP_POLICY = "MONTHLY_PRIMARY_DAILY_GAP_FILL_V1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _candidate_id(index: int) -> str:
    return f"C{index:03d}"


def _deterministic_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="ascii", newline="") as text:
                for row in rows:
                    text.write(_canonical_json(row) + "\n")
                    count += 1
    return count, _sha256_file(path)


def _deterministic_gzip_csv(path: Path, fieldnames: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="ascii", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                    count += 1
    return count, _sha256_file(path)


def _artifact_index(repo_root: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    root = repo_root / "sandbox/aegis_range_strategy_v1/artifacts/r2_data_readiness"
    manifest_path = root / "derived_dataset_manifest.json"
    if _sha256_file(manifest_path) != "55605c09e3f3de0d3f4d8b335beeac0eab4b0728a0f267512a6429ee8e2186b0":
        raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_SOURCE_INTEGRITY")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    return root, {item["path"]: item for item in manifest["artifacts"]}


def load_train_candles(repo_root: Path, symbol: str) -> list[Candle5m]:
    SealedPartitionGuard.require("TRAIN")
    if symbol not in SYMBOLS:
        raise ValueError("symbol outside frozen universe")
    root, artifacts = _artifact_index(repo_root)
    candles: list[Candle5m] = []
    for month in range(1, 13):
        relative = f"ohlcv_5m/{symbol}/2024-{month:02d}.csv.gz"
        item = artifacts[relative]
        path = root / relative
        if _sha256_file(path) != item["sha256"]:
            raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_SOURCE_INTEGRITY")
        with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
            for row in csv.DictReader(handle):
                open_time = _parse_utc(row["open_time"])
                if not TRAIN_START <= open_time < TRAIN_END:
                    raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_PARTITION_VIOLATION")
                candles.append(
                    Candle5m(
                        symbol=row["symbol"],
                        open_time=open_time,
                        available_at=_parse_utc(row["available_at"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        segment_id=int(row["segment_id"]),
                        high_source=row["high_source"],
                        low_source=row["low_source"],
                    )
                )
    expected_rows = int((TRAIN_END - TRAIN_START) / timedelta(minutes=5))
    if len(candles) != expected_rows or candles[0].open_time != TRAIN_START or candles[-1].available_at != TRAIN_END:
        raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_SOURCE_INTEGRITY")
    for previous, current in zip(candles, candles[1:]):
        if current.open_time != previous.open_time + timedelta(minutes=5) or current.segment_id != previous.segment_id:
            raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_SOURCE_INTEGRITY")
    return candles


def load_train_funding(repo_root: Path, symbol: str) -> list[tuple[datetime, float, float]]:
    SealedPartitionGuard.require("TRAIN")
    root = repo_root / "sandbox/aegis_range_strategy_v1/artifacts/r2_post_r1_fix_revalidation/build_a"
    manifest_path = root / "derived_dataset_manifest.json"
    if _sha256_file(manifest_path) != "2bd850cbb88123f870dd3f26a5865a6749aa30025cc512997ac44f44ac17a1cc":
        raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_SOURCE_INTEGRITY")
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    artifacts = {item["path"]: item for item in manifest["funding_mark_artifacts"]}
    events: list[tuple[datetime, float, float]] = []
    for month in range(1, 13):
        relative = f"funding_mark/{symbol}/2024-{month:02d}.csv.gz"
        item = artifacts[relative]
        path = root / relative
        if _sha256_file(path) != item["sha256"]:
            raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_SOURCE_INTEGRITY")
        with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
            for row in csv.DictReader(handle):
                funding_at = _parse_utc(row["funding_at"])
                if TRAIN_START < funding_at < TRAIN_END:
                    events.append((funding_at, float(row["funding_rate"]), float(row["mark_close"])))
    events.sort(key=lambda item: item[0])
    if any(current[0] <= previous[0] for previous, current in zip(events, events[1:])):
        raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_SOURCE_INTEGRITY")
    return events


REGIME_CACHE_FIELDS = (
    "open_time",
    "technical_regime",
    "transition_risk",
    "adx",
    "atr_percentile",
    "bollinger_width_percentile",
    "volume_ratio",
    "range_breakout",
    "failed_breakout_count",
    "structure",
    "chop_risk",
    "atr14_raw",
)


def _build_regime_cache_task(arguments: tuple[str, str, str]) -> dict[str, Any]:
    repo_text, symbol, destination_text = arguments
    repo_root = Path(repo_text)
    destination = Path(destination_text)
    candles = load_train_candles(repo_root, symbol)
    TypeScriptRegimeEvaluator(repo_root)
    payload = {
        "symbol": symbol,
        "timeframe": "5m",
        "candles": [
            {
                "timestamp": int(candle.open_time.timestamp() * 1000),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in candles
        ],
    }
    child = repo_root / "binance-futures-bot-ts"
    bridge = repo_root / "sandbox/aegis_range_strategy_v1/scripts/regime_v2_train_batch_bridge.cjs"
    process = subprocess.run(
        ["node", "-r", "ts-node/register", str(bridge)],
        cwd=child,
        input=_canonical_json(payload),
        capture_output=True,
        check=False,
        text=True,
        timeout=900,
    )
    if process.returncode != 0 or process.stderr:
        raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_REGIME_PARITY")
    decisions = process.stdout.splitlines()
    if len(decisions) != len(candles) - 159:
        raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_REGIME_PARITY")

    def rows() -> Iterable[dict[str, Any]]:
        for index, serialized in enumerate(decisions, start=159):
            decision = json.loads(serialized)
            candle = candles[index]
            if decision["timestamp"] != int(candle.open_time.timestamp() * 1000):
                raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_REGIME_PARITY")
            window = candles[index - 159 : index + 1]
            yield {
                "open_time": iso_utc_millis(candle.open_time),
                "technical_regime": decision["technicalRegime"],
                "transition_risk": decision["transitionRisk"],
                "adx": repr(float(decision["adx"])),
                "atr_percentile": repr(float(decision["atrPercentile"])),
                "bollinger_width_percentile": repr(float(decision["bollingerWidthPercentile"])),
                "volume_ratio": repr(float(decision["volumeRatio"])),
                "range_breakout": decision["rangeBreakout"],
                "failed_breakout_count": int(decision["failedBreakoutCount"]),
                "structure": decision["structure"],
                "chop_risk": repr(float(decision["chopRisk"])),
                "atr14_raw": repr(RangeAtr14V1.calculate(window)),
            }

    count, digest = _deterministic_gzip_csv(destination, REGIME_CACHE_FIELDS, rows())
    return {"symbol": symbol, "path": str(destination), "rows": count, "sha256": digest}


def build_regime_caches(repo_root: Path, output_root: Path, workers: int) -> dict[str, dict[str, Any]]:
    cache_root = output_root / "regime_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    tasks = [(str(repo_root), symbol, str(cache_root / f"{symbol}.csv.gz")) for symbol in SYMBOLS]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_build_regime_cache_task, tasks))
    return {item["symbol"]: item for item in results}


def load_regime_cache(path: Path, candle_count: int) -> list[RegimeSnapshot | None]:
    snapshots: list[RegimeSnapshot | None] = [None] * 159
    with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
        for row in csv.DictReader(handle):
            snapshots.append(
                RegimeSnapshot(
                    technical_regime=row["technical_regime"],
                    transition_risk=row["transition_risk"],
                    adx=float(row["adx"]),
                    atr_percentile=float(row["atr_percentile"]),
                    bollinger_width_percentile=float(row["bollinger_width_percentile"]),
                    volume_ratio=float(row["volume_ratio"]),
                    range_breakout=row["range_breakout"],
                    failed_breakout_count=int(row["failed_breakout_count"]),
                    structure=row["structure"],
                    chop_risk=float(row["chop_risk"]),
                    atr14_raw=float(row["atr14_raw"]),
                )
            )
    if len(snapshots) != candle_count:
        raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_REGIME_PARITY")
    return snapshots


class CachedRangeRegimeAdapter:
    def __init__(self, candles: list[Candle5m], snapshots: list[RegimeSnapshot | None]):
        self._origin = candles[0].open_time
        self._snapshots = snapshots

    def snapshot(self, symbol: str, history: list[Candle5m]) -> RegimeSnapshot:
        if len(history) < 160:
            raise ValueError("INSUFFICIENT_HISTORY")
        last = history[-1]
        index = int((last.open_time - self._origin) / timedelta(minutes=5))
        if index < 159 or index >= len(self._snapshots):
            raise ValueError("INSUFFICIENT_HISTORY")
        first = history[-160]
        if first.open_time != last.open_time - timedelta(minutes=5 * 159) or first.segment_id != last.segment_id:
            raise ValueError("INSUFFICIENT_HISTORY")
        snapshot = self._snapshots[index]
        if snapshot is None:
            raise ValueError("INSUFFICIENT_HISTORY")
        return snapshot


class ObservedRangeLifecycleV1(RangeLifecycleV1):
    def __init__(self, candidate: RangeCandidate):
        super().__init__(candidate)
        self.entry_events: list[tuple[Position, float]] = []
        self.exit_events: list[tuple[Position, FillEvent, float]] = []

    def consume_pending_entry(self, **kwargs: Any) -> Position | None:
        position = super().consume_pending_entry(**kwargs)
        if position is not None:
            self.entry_events.append((copy.deepcopy(position), float(kwargs["raw_open"])))
        return position

    def _exit(self, candle: Candle5m, base_price: float, reason: str) -> FillEvent:
        if self.position is None:
            raise RuntimeError("STATE_INVARIANT_VIOLATION")
        position = copy.deepcopy(self.position)
        event = super()._exit(candle, base_price, reason)
        self.exit_events.append((position, event, base_price))
        return event


class ObservedRangeEngineV1(RangeEngineV1):
    def __init__(self, symbol: str, candidate: RangeCandidate, regime_adapter: CachedRangeRegimeAdapter):
        super().__init__(symbol, candidate, regime_adapter)
        self.lifecycle = ObservedRangeLifecycleV1(candidate)
        self.ended_episodes: list[Episode] = []

    def _end_episode(self, decision_at: datetime | None, reason: str) -> None:
        episode = self.episode
        super()._end_episode(decision_at, reason)
        if episode is not None:
            self.ended_episodes.append(episode)

    def on_data_integrity(self) -> None:
        super().on_data_integrity()
        self.lifecycle = ObservedRangeLifecycleV1(self.candidate)

    def on_split_boundary(self) -> None:
        super().on_split_boundary()
        self.lifecycle = ObservedRangeLifecycleV1(self.candidate)

    def _record(self, output: dict[str, Any]) -> dict[str, Any]:
        # Bar-level structures are not retained; all strategy state transitions
        # remain in the frozen process implementation and sparse observer events.
        return output


def _funding_slice(
    funding: list[tuple[datetime, float, float]], entry_at: datetime, exit_at: datetime
) -> tuple[tuple[float, float], ...]:
    timestamps = [item[0] for item in funding]
    start = bisect.bisect_right(timestamps, entry_at)
    end = bisect.bisect_right(timestamps, exit_at)
    return tuple((rate, mark) for _, rate, mark in funding[start:end])


def _scenario_returns(
    side: str,
    entry_base: float,
    exit_base: float,
    events: tuple[tuple[float, float], ...],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    exit_side = "SHORT" if side == "LONG" else "LONG"
    for scenario in SCENARIOS:
        entry_fill = adverse_fill(entry_base, side, scenario.slippage_bps_per_side)
        exit_fill = adverse_fill(exit_base, exit_side, scenario.slippage_bps_per_side)
        gross = gross_return(side, entry_fill, exit_fill)
        fees = fee_return(entry_fill, exit_fill, scenario.fee_bps_per_side)
        funding_value = funding_return(side, entry_fill, events)
        result[scenario.name] = {
            "entry_fill": entry_fill,
            "exit_fill": exit_fill,
            "gross_return": gross,
            "fees": fees,
            "funding_return": funding_value,
            "net_return": gross - fees + funding_value,
        }
    return result


def _episode_record(candidate_id: str, candidate: RangeCandidate, episode: Episode) -> dict[str, Any]:
    pair = episode.previous_snapshot.pair
    return {
        "candidate_id": candidate_id,
        "candidate": candidate.as_dict(),
        "symbol": episode.symbol,
        "range_episode_id": episode.range_episode_id,
        "range_confirmed_at": iso_utc_millis(episode.range_confirmed_at),
        "confirmation_support": pair.support,
        "confirmation_resistance": pair.resistance,
        "episode_end_at": None,
        "episode_end_reason": None,
        "breakout_direction": None,
        "false_range": False,
        "purged": False,
        "purge_reason": None,
        "open_trade_at_boundary": False,
        "pending_entry_at_boundary": False,
    }


def _finalize_episode_end(record: dict[str, Any], episode: Episode) -> None:
    record["episode_end_at"] = None if episode.ended_at is None else iso_utc_millis(episode.ended_at)
    record["episode_end_reason"] = episode.end_reason
    record["breakout_direction"] = episode.outside_direction
    if episode.end_reason == "CONFIRMED_BREAKOUT" and episode.ended_at is not None:
        age = episode.ended_at - episode.range_confirmed_at
        record["false_range"] = timedelta(0) < age <= timedelta(minutes=60)


def _trade_record(
    candidate_id: str,
    position: Position,
    event: FillEvent,
    entry_base: float,
    exit_base: float,
    funding: list[tuple[datetime, float, float]],
) -> dict[str, Any]:
    events = _funding_slice(funding, position.entry_at, event.fill_at)
    scenarios = _scenario_returns(position.side, entry_base, exit_base, events)
    baseline = scenarios["BASELINE"]
    if baseline["entry_fill"] != position.entry_fill or baseline["exit_fill"] != event.fill_price:
        raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_ACCOUNTING_PARITY")
    thesis = json.loads(position.thesis_serialized)
    return {
        "candidate_id": candidate_id,
        "symbol": position.symbol,
        "side": position.side,
        "range_episode_id": position.range_episode_id,
        "range_id": position.range_id,
        "decision_at": thesis["decision_at"],
        "entry_at": iso_utc_millis(position.entry_at),
        "entry_base": entry_base,
        "entry_fill": position.entry_fill,
        "exit_at": iso_utc_millis(event.fill_at),
        "exit_base": exit_base,
        "exit_fill": event.fill_price,
        "exit_reason": event.reason,
        "support_at_entry": position.support_at_entry,
        "resistance_at_entry": position.resistance_at_entry,
        "midpoint_at_entry": position.midpoint_at_entry,
        "ATR_entry": position.atr_entry,
        "stop_at_entry": position.stop_at_entry,
        "target_at_entry": position.target_at_entry,
        "gross_return": baseline["gross_return"],
        "fees": baseline["fees"],
        "funding_return": baseline["funding_return"],
        "net_return": baseline["net_return"],
        "holding_bars": position.closed_bars,
        "holding_minutes": (event.fill_at - position.entry_at).total_seconds() / 60.0,
        "thesis_feature_hash": position.thesis_feature_hash,
        "funding_event_count": len(events),
        "funding_timestamps": [iso_utc_millis(item[0]) for item in funding if position.entry_at < item[0] <= event.fill_at],
        "scenarios": scenarios,
        "purged": False,
    }


def execute_candidate(
    candidate_index: int,
    candidate: RangeCandidate,
    symbol: str,
    candles: list[Candle5m],
    snapshots: list[RegimeSnapshot | None],
    funding: list[tuple[datetime, float, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    candidate_id = _candidate_id(candidate_index)
    engine = ObservedRangeEngineV1(symbol, candidate, CachedRangeRegimeAdapter(candles, snapshots))
    episodes: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    entries: dict[str, tuple[Position, float]] = {}
    replay = hashlib.sha256()

    for candle in candles:
        output = engine.process(candle, same_split=True, embargo=candle.available_at < EMBARGO_END)
        replay.update(
            (_canonical_json({key: output.get(key) for key in ("decision_at", "status", "episode_event", "signal", "entry_hash", "exit_reason")}) + "\n").encode("ascii")
        )
        lifecycle = engine.lifecycle
        if not isinstance(lifecycle, ObservedRangeLifecycleV1):
            raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_OBSERVER_INVARIANT")
        for position, raw_open in lifecycle.entry_events:
            entries[position.thesis_feature_hash] = (position, raw_open)
        lifecycle.entry_events.clear()
        for position, event, exit_base in lifecycle.exit_events:
            opened = entries.pop(position.thesis_feature_hash, None)
            if opened is None:
                raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_OBSERVER_INVARIANT")
            trades.append(_trade_record(candidate_id, position, event, opened[1], exit_base, funding))
        lifecycle.exit_events.clear()
        for ended in engine.ended_episodes:
            record = episodes.get(ended.range_episode_id)
            if record is None:
                raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_OBSERVER_INVARIANT")
            _finalize_episode_end(record, ended)
        engine.ended_episodes.clear()
        if output.get("episode_event") == "CONFIRMED":
            if engine.episode is None:
                raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_OBSERVER_INVARIANT")
            episodes[engine.episode.range_episode_id] = _episode_record(candidate_id, candidate, engine.episode)

    purge_ids: set[str] = set()
    if engine.episode is not None:
        purge_ids.add(engine.episode.range_episode_id)
    if engine.lifecycle.pending_entry is not None:
        purge_ids.add(engine.lifecycle.pending_entry.range_episode_id)
        episodes[engine.lifecycle.pending_entry.range_episode_id]["pending_entry_at_boundary"] = True
    if engine.lifecycle.position is not None:
        purge_ids.add(engine.lifecycle.position.range_episode_id)
        episodes[engine.lifecycle.position.range_episode_id]["open_trade_at_boundary"] = True
    engine.on_split_boundary()
    for ended in engine.ended_episodes:
        record = episodes[ended.range_episode_id]
        _finalize_episode_end(record, ended)
    for identifier in purge_ids:
        episodes[identifier]["purged"] = True
        episodes[identifier]["purge_reason"] = "PURGED_SPLIT_BOUNDARY"
    for trade in trades:
        if trade["range_episode_id"] in purge_ids:
            trade["purged"] = True

    trades_by_episode: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        trades_by_episode.setdefault(trade["range_episode_id"], []).append(trade)
    episode_rows = []
    for identifier, record in episodes.items():
        nested = trades_by_episode.get(identifier, [])
        record["trade_count"] = len(nested)
        record["long_trades"] = sum(item["side"] == "LONG" for item in nested)
        record["short_trades"] = sum(item["side"] == "SHORT" for item in nested)
        for scenario in SCENARIOS:
            values = [item["scenarios"][scenario.name] for item in nested]
            prefix = scenario.name.lower()
            record[f"{prefix}_gross_return"] = sum(item["gross_return"] for item in values)
            record[f"{prefix}_fees"] = sum(item["fees"] for item in values)
            record[f"{prefix}_funding_return"] = sum(item["funding_return"] for item in values)
            record[f"{prefix}_net_return"] = sum(item["net_return"] for item in values)
        episode_rows.append(record)
    episode_rows.sort(key=lambda item: (item["range_confirmed_at"], item["range_episode_id"]))
    trades.sort(key=lambda item: (item["entry_at"], item["thesis_feature_hash"]))
    return episode_rows, trades, replay.hexdigest()


def _symbol_worker(arguments: tuple[str, str, str, str]) -> dict[str, Any]:
    repo_text, symbol, cache_path_text, shard_path_text = arguments
    repo_root = Path(repo_text)
    candles = load_train_candles(repo_root, symbol)
    snapshots = load_regime_cache(Path(cache_path_text), len(candles))
    funding = load_train_funding(repo_root, symbol)
    replay_hashes: dict[str, str] = {}

    def rows() -> Iterable[dict[str, Any]]:
        for index, candidate in enumerate(candidate_grid()):
            episodes, trades, replay_hash = execute_candidate(index, candidate, symbol, candles, snapshots, funding)
            replay_hashes[_candidate_id(index)] = replay_hash
            for episode in episodes:
                yield {"type": "episode", "payload": episode}
            for trade in trades:
                yield {"type": "trade", "payload": trade}

    row_count, digest = _deterministic_gzip_jsonl(Path(shard_path_text), rows())
    return {
        "symbol": symbol,
        "path": shard_path_text,
        "rows": row_count,
        "sha256": digest,
        "replay_hashes": replay_hashes,
    }


def run_symbol_shards(repo_root: Path, output_root: Path, cache_manifest: dict[str, dict[str, Any]], workers: int) -> dict[str, dict[str, Any]]:
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(repo_root), symbol, cache_manifest[symbol]["path"], str(shard_root / f"{symbol}.jsonl.gz"))
        for symbol in SYMBOLS
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_symbol_worker, tasks))
    return {item["symbol"]: item for item in results}


def load_shards(shards: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        path = Path(shards[symbol]["path"])
        if _sha256_file(path) != shards[symbol]["sha256"]:
            raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_REPLAY_MISMATCH")
        with gzip.open(path, "rt", encoding="ascii") as handle:
            for line in handle:
                row = json.loads(line)
                (episodes if row["type"] == "episode" else trades).append(row["payload"])
    episodes.sort(key=lambda item: (item["candidate_id"], SYMBOLS.index(item["symbol"]), item["range_confirmed_at"], item["range_episode_id"]))
    trades.sort(key=lambda item: (item["candidate_id"], SYMBOLS.index(item["symbol"]), item["exit_at"], item["thesis_feature_hash"]))
    return episodes, trades


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _profit_factor(values: list[float]) -> float | str | None:
    if not values:
        return None
    positive = sum(value for value in values if value > 0)
    negative = sum(value for value in values if value < 0)
    return "Infinity" if negative == 0 else positive / abs(negative)


def _cvar95(values: list[float]) -> float | None:
    if not values:
        return None
    count = max(1, math.ceil(0.05 * len(values)))
    return sum(sorted(values)[:count]) / count


def _maximum_drawdown(trades: list[dict[str, Any]], scenario: str) -> float | None:
    if not trades:
        return None
    ordered = sorted(trades, key=lambda item: (item["exit_at"], item["symbol"], item["thesis_feature_hash"]))
    equity = peak = 1.0
    maximum = 0.0
    for trade in ordered:
        equity += trade["scenarios"][scenario]["net_return"] / 11.0
        if equity <= 0:
            return 1.0
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak)
    return maximum


def _expectancy_for_side(trades: list[dict[str, Any]], side: str, scenario: str) -> float | None:
    grouped: dict[str, float] = {}
    for trade in trades:
        if trade["side"] == side:
            grouped[trade["range_episode_id"]] = grouped.get(trade["range_episode_id"], 0.0) + trade["scenarios"][scenario]["net_return"]
    return _mean(list(grouped.values()))


def _concentration(episodes: list[dict[str, Any]], key: str, scenario: str) -> dict[str, float]:
    grouped: dict[str, float] = {}
    field = f"{scenario.lower()}_net_return"
    for episode in episodes:
        group = episode["symbol"] if key == "symbol" else episode["range_confirmed_at"][:7]
        grouped[group] = grouped.get(group, 0.0) + episode[field]
    denominator = sum(max(value, 0.0) for value in grouped.values())
    if denominator <= 0:
        return {group: 0.0 for group in sorted(grouped)}
    return {group: max(value, 0.0) / denominator for group, value in sorted(grouped.items())}


def candidate_metrics(candidate_id: str, episodes: list[dict[str, Any]], trades: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_episodes = [item for item in episodes if item["candidate_id"] == candidate_id and not item["purged"]]
    candidate_trades = [item for item in trades if item["candidate_id"] == candidate_id and not item["purged"]]
    operated = [item for item in candidate_episodes if item["trade_count"] > 0]
    result: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate": candidate_grid()[int(candidate_id[1:])].as_dict(),
        "confirmed_episodes": len(candidate_episodes),
        "operated_episodes": len(operated),
        "trades": len(candidate_trades),
        "long_trades": sum(item["side"] == "LONG" for item in candidate_trades),
        "short_trades": sum(item["side"] == "SHORT" for item in candidate_trades),
        "purged_episodes": sum(item["candidate_id"] == candidate_id and item["purged"] for item in episodes),
        "abstention_rate": None if not candidate_episodes else (len(candidate_episodes) - len(operated)) / len(candidate_episodes),
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        name = scenario.name
        field = f"{name.lower()}_net_return"
        gross_field = f"{name.lower()}_gross_return"
        episode_returns = [item[field] for item in operated]
        trade_returns = [item["scenarios"][name]["net_return"] for item in candidate_trades]
        month_returns: dict[str, float] = {}
        symbol_returns: dict[str, float] = {}
        for episode in operated:
            month = episode["range_confirmed_at"][:7]
            month_returns[month] = month_returns.get(month, 0.0) + episode[field]
            symbol_returns[episode["symbol"]] = symbol_returns.get(episode["symbol"], 0.0) + episode[field]
        majors = [item[field] for item in operated if item["symbol"] in MAJORS]
        alts = [item[field] for item in operated if item["symbol"] not in MAJORS]
        result["scenarios"][name] = {
            "episode_net_expectancy": _mean(episode_returns),
            "trade_net_expectancy": _mean(trade_returns),
            "gross_expectancy": _mean([item[gross_field] for item in operated]),
            "total_fees": sum(item["scenarios"][name]["fees"] for item in candidate_trades),
            "total_funding_contribution": sum(item["scenarios"][name]["funding_return"] for item in candidate_trades),
            "profit_factor": _profit_factor(trade_returns),
            "episode_cvar95": _cvar95(episode_returns),
            "pseudo_equity_max_drawdown": _maximum_drawdown(candidate_trades, name),
            "breakout_loss_rate": None if not candidate_trades else sum(item["exit_reason"] == "TRADE_BREAKOUT" and item["scenarios"][name]["net_return"] < 0 for item in candidate_trades) / len(candidate_trades),
            "false_range_rate": None if not candidate_episodes else sum(item["false_range"] for item in candidate_episodes) / len(candidate_episodes),
            "positive_months": sum(value > 0 for value in month_returns.values()),
            "negative_months": sum(value < 0 for value in month_returns.values()),
            "positive_symbols": sum(value > 0 for value in symbol_returns.values()),
            "negative_symbols": sum(value < 0 for value in symbol_returns.values()),
            "long_expectancy": _expectancy_for_side(candidate_trades, "LONG", name),
            "short_expectancy": _expectancy_for_side(candidate_trades, "SHORT", name),
            "majors_expectancy": _mean(majors),
            "alts_expectancy": _mean(alts),
            "symbol_returns": dict(sorted(symbol_returns.items())),
            "month_returns": dict(sorted(month_returns.items())),
            "concentration_by_symbol": _concentration(operated, "symbol", name),
            "concentration_by_month": _concentration(operated, "month", name),
        }
    return result


def materialize_results(output_root: Path, episodes: list[dict[str, Any]], trades: list[dict[str, Any]]) -> dict[str, Any]:
    episode_count, episode_hash = _deterministic_gzip_jsonl(output_root / "episodes.jsonl.gz", episodes)
    trade_count, trade_hash = _deterministic_gzip_jsonl(output_root / "trades.jsonl.gz", trades)
    metrics = [candidate_metrics(_candidate_id(index), episodes, trades) for index in range(len(candidate_grid()))]
    metrics_path = output_root / "candidate_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")
    baseline_order = sorted(
        metrics,
        key=lambda item: (
            float("-inf") if item["scenarios"]["BASELINE"]["episode_net_expectancy"] is None else item["scenarios"]["BASELINE"]["episode_net_expectancy"],
            item["candidate_id"],
        ),
    )
    descriptive = {
        "label": "TRAIN_DESCRIPTIVE_ONLY",
        "authority": "NO_SELECTION_AUTHORITY",
        "worst_candidate": baseline_order[0]["candidate_id"],
        "median_candidate": baseline_order[len(baseline_order) // 2]["candidate_id"],
        "best_candidate": baseline_order[-1]["candidate_id"],
    }
    return {
        "episodes": {"path": "episodes.jsonl.gz", "rows": episode_count, "sha256": episode_hash},
        "trades": {"path": "trades.jsonl.gz", "rows": trade_count, "sha256": trade_hash},
        "metrics": {"path": "candidate_metrics.json", "rows": len(metrics), "sha256": _sha256_file(metrics_path)},
        "descriptive": descriptive,
    }


def verify_authority(repo_root: Path, environment: dict[str, str] | None = None) -> dict[str, Any]:
    flags = SealedPartitionGuard.access_flags(environment)
    if flags != {"TRAIN": True, "CALIBRATION": False, "VALIDATION": False, "HOLDOUT": False}:
        raise PermissionError("AEGIS_RANGE_R2_TRAIN_PARTITION_VIOLATION")
    r1 = verify_r1_immutable(repo_root)
    if r1["manifest_sha256"] != R1_MANIFEST_FILE_SHA256 or r1["verified_files"] != "42":
        raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_R1_DRIFT")
    revalidation = repo_root / "docs/aegis-range-v1/r2_post_r1_fix_revalidation_manifest.json"
    if _sha256_file(revalidation) != REVALIDATION_MANIFEST_FILE_SHA256:
        raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_SOURCE_INTEGRITY")
    payload = json.loads(revalidation.read_text(encoding="ascii"))
    if payload["reproducibility"]["new_logical_sha256"] != DATASET_LOGICAL_SHA256:
        raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_SOURCE_INTEGRITY")
    if payload["source_integrity"]["replacement_policy"] != SOURCE_GAP_POLICY:
        raise SourceIntegrityError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_SOURCE_INTEGRITY")
    return {"partition_flags": flags, "r1": r1, "revalidation": payload}


def execute_train_run(repo_root: Path, output_root: Path, workers: int, cache_root: Path | None = None) -> dict[str, Any]:
    authority = verify_authority(repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if cache_root is None:
        caches = build_regime_caches(repo_root, output_root, workers)
    else:
        cache_manifest_path = cache_root / "regime_cache_manifest.json"
        caches = json.loads(cache_manifest_path.read_text(encoding="ascii"))["caches"]
        for item in caches.values():
            if _sha256_file(Path(item["path"])) != item["sha256"]:
                raise RuntimeError("AEGIS_RANGE_R2_TRAIN_BLOCKED_BY_REPLAY_MISMATCH")
    cache_manifest_path = output_root / "regime_cache_manifest.json"
    cache_manifest = {
        "schema_version": "aegis-range-r2-train-regime-cache-v1",
        "dataset_logical_sha256": DATASET_LOGICAL_SHA256,
        "market_absent": True,
        "timeframe": "5m",
        "window": 160,
        "caches": caches,
    }
    cache_manifest_path.write_text(json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    shards = run_symbol_shards(repo_root, output_root, caches, workers)
    episodes, trades = load_shards(shards)
    artifacts = materialize_results(output_root, episodes, trades)
    run_manifest = {
        "schema_version": "aegis-range-r2-train-run-v1",
        "status": "AEGIS_RANGE_R2_TRAIN_RUN_COMPLETE",
        "r1_manifest_file_sha256": R1_MANIFEST_FILE_SHA256,
        "dataset_logical_sha256": DATASET_LOGICAL_SHA256,
        "source_gap_policy": SOURCE_GAP_POLICY,
        "train": {"start_inclusive": iso_utc_millis(TRAIN_START), "end_exclusive": iso_utc_millis(TRAIN_END)},
        "embargo_end": iso_utc_millis(EMBARGO_END),
        "candidate_count": len(candidate_grid()),
        "symbols": list(SYMBOLS),
        "workers": workers,
        "regime_cache_manifest_sha256": _sha256_file(cache_manifest_path),
        "shards": shards,
        "artifacts": artifacts,
        "partition_flags": authority["partition_flags"],
        "phase_boundary": {
            "calibration_opened": False,
            "validation_opened": False,
            "holdout_opened": False,
            "candidate_selected": False,
            "pre_validation_spec_frozen": False,
        },
    }
    path = output_root / "run_manifest.json"
    path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return {**run_manifest, "run_manifest_sha256": _sha256_file(path)}
