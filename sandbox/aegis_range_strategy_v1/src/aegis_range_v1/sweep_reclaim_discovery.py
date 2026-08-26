from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import math
import platform
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidates import RangeCandidate, candidate_grid
from .costs import BASELINE, STRESS_20, STRESS_30, adverse_fill, fee_return, funding_return, gross_return
from .engine import RangeEngineV1
from .lifecycle import RangeLifecycleV1
from .models import Candle5m, FillEvent, LevelSnapshot, PendingEntry, Position
from .numeric import canonical_decimal_12dp, iso_utc_millis
from .readiness import SYMBOLS, SealedPartitionGuard, SourceIntegrityError
from .train_backtest import (
    EMBARGO_END,
    DATASET_LOGICAL_SHA256,
    TRAIN_END,
    TRAIN_START,
    CachedRangeRegimeAdapter,
    load_regime_cache,
    load_train_candles,
    load_train_funding,
)

SCHEMA = "aegis-range-v2-sweep-reclaim-opportunity-v1"
MANIFEST_SCHEMA = "aegis-range-v2-sweep-reclaim-discovery-v1"
STATUS = "AEGIS_RANGE_V2_SWEEP_RECLAIM_DISCOVERY_READY_FOR_REVIEW"
DISCOVERY_LABELS = ("DISCOVERY_ONLY", "HYPOTHESIS_GENERATION_ONLY", "NO_SELECTION_AUTHORITY", "NO_PROMOTION_AUTHORITY", "NO_WHITELIST_AUTHORITY")
HEAD_AUTHORITY = "d917602b2c7e065d18b29f87df14febb935d8ad8"
FLAGS = {"TRAIN": True, "CALIBRATION": False, "VALIDATION": False, "HOLDOUT": False}
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_DRAWS = 10_000
SWEEP_ATR = 0.10
RECLAIM_BARS = 3
HORIZONS = (15, 30, 60, 120)
FAVORABLE_BPS = (10, 20, 30, 40)
ADVERSE_BPS = (10, 20, 30, 40)
PROGRESS_PCT = (25, 50, 100)
SCENARIOS = (BASELINE, STRESS_20, STRESS_30)
ARTIFACT_NAMES = (
    "sweep_opportunities.jsonl.gz",
    "reclaim_entries.jsonl.gz",
    "reclaim_paths.jsonl.gz",
    "first_passage.jsonl.gz",
    "symbol_month_diagnostics.json",
    "diagnostic_summary.json",
    "diagnostics_manifest.json",
)
RUN_A_HASHES = {
    "run_manifest.json": "5f62022f35fb38de174e6f7c573397d1c1ceebc75d76f7d848260c35456012b8",
    "candidate_metrics.json": "12f72be45420099d7ab0a56524ca934e791dfbaa9da0c0add87277d7939b656f",
    "episodes.jsonl.gz": "82989a83a68935ed44866afb2f5904e703c81e27b50acde5fe1c2fabd6af5270",
    "trades.jsonl.gz": "125f31dcb1bf27e6f183bbbb02da901a5133847e577196a9bd6a59be42cd4537",
    "regime_cache_manifest.json": "a9699e874537bcdf14042e3d811594448e886b6811f48741b9b6ce5ad7e9c22b",
}
PRIOR_DISCOVERY_HASHES = {
    "diagnostics_manifest.json": "77b27abf29efa2cffae853ea991da4af2183dad9b1feff150b7d527bba4346e3",
    "diagnostic_summary.json": "38ff8f28bc4b6a5dc204c14fe1aa6b6bd3d57fe56df3873cf467fd6303fd56fb",
    "opportunity_paths.jsonl.gz": "796af43415a7398f04704a4ca22d60d7c7f5b9e7d54b9589ea1700c3aaaf9b00",
    "stop_recovery.jsonl.gz": "796fec88152c7cd08161691f2198b3cdd1486f7eeec8d6719fafa8bdedce86f7",
    "confirmation_counterfactuals.jsonl.gz": "aaca07c8055d96753fc56a5b83e0d57a0be69c90f67938c3f8794aab2c40044c",
    "symbol_suitability.jsonl.gz": "6061ff4b5102f1227ce85578674144e2c9be1b9d5eddff80312df1cff7b9ac34",
}
DECISION_LABELS = (
    "NO_GROSS_EDGE",
    "COST_LIMITED_SIGNAL",
    "POTENTIAL_NET_EDGE",
)


def _parse(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _json(value: object, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="ascii", newline="") as text:
                for row in rows:
                    text.write(_json(row) + "\n")
                    count += 1
    return {"rows": count, "sha256": _sha256_file(path)}


def _write_json(path: Path, value: object) -> dict[str, Any]:
    path.write_text(_json(value, pretty=True), encoding="ascii")
    return {"rows": 1, "sha256": _sha256_file(path)}


def candidate_id(index: int) -> str:
    return f"C{index:03d}"


def generate_candidate_mappings() -> dict[tuple[float, float, float, float], tuple[str, ...]]:
    """Prove the frozen 6 structures x 4 lifecycle contracts x 16 inert axes."""
    mapping: dict[tuple[float, float, float, float], list[str]] = defaultdict(list)
    grid = candidate_grid()
    for index, candidate in enumerate(grid):
        key = (
            candidate.cluster_tolerance_atr,
            candidate.min_range_amplitude_pct,
            candidate.stop_buffer_atr,
            candidate.target_buffer_atr,
        )
        mapping[key].append(candidate_id(index))
    if len(grid) != 384 or len(mapping) != 24 or any(len(ids) != 16 for ids in mapping.values()):
        raise RuntimeError("SWEEP_RECLAIM_CANDIDATE_MAPPING_INVARIANT")
    flattened = [identifier for key in sorted(mapping) for identifier in mapping[key]]
    if set(flattened) != {candidate_id(index) for index in range(384)} or len(flattened) != 384:
        raise RuntimeError("SWEEP_RECLAIM_CANDIDATE_MAPPING_INVARIANT")
    return {key: tuple(ids) for key, ids in sorted(mapping.items())}


def structural_candidates() -> tuple[RangeCandidate, ...]:
    result = {}
    for candidate in candidate_grid():
        key = (candidate.cluster_tolerance_atr, candidate.min_range_amplitude_pct)
        result.setdefault(key, candidate)
    if len(result) != 6:
        raise RuntimeError("SWEEP_RECLAIM_STRUCTURE_INVARIANT")
    return tuple(result[key] for key in sorted(result))


@dataclass(frozen=True, slots=True)
class FrozenRange:
    symbol: str
    side: str
    support: float
    resistance: float
    midpoint: float
    atr14_raw: float
    range_episode_id: str
    range_id: str
    range_confirmed_at: datetime
    snapshot_decision_at: datetime

    @classmethod
    def from_snapshot(cls, symbol: str, side: str, snapshot: LevelSnapshot, range_confirmed_at: datetime) -> "FrozenRange":
        pair = snapshot.pair
        return cls(symbol, side, pair.support, pair.resistance, pair.midpoint, snapshot.atr14_raw, snapshot.range_episode_id, snapshot.range_id, range_confirmed_at, snapshot.decision_at)


def canonical_sweep_opportunity_id(row: Mapping[str, Any]) -> str:
    value = "|".join(
        (
            SCHEMA,
            str(row["symbol"]),
            str(row["side"]),
            iso_utc_millis(_parse(row["sweep_decision_at"])),
            canonical_decimal_12dp(row["support"]),
            canonical_decimal_12dp(row["resistance"]),
            canonical_decimal_12dp(row["midpoint"]),
        )
    )
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def sweep_depth(side: str, candle: Candle5m, frozen: FrozenRange) -> tuple[float, float, str, float, float]:
    threshold = frozen.support - SWEEP_ATR * frozen.atr14_raw if side == "LONG" else frozen.resistance + SWEEP_ATR * frozen.atr14_raw
    boundary = frozen.support if side == "LONG" else frozen.resistance
    past_threshold = max(0.0, threshold - candle.low) if side == "LONG" else max(0.0, candle.high - threshold)
    total = max(0.0, boundary - candle.low) if side == "LONG" else max(0.0, candle.high - boundary)
    past_atr = 0.0 if frozen.atr14_raw == 0 else past_threshold / frozen.atr14_raw
    total_atr = 0.0 if frozen.atr14_raw == 0 else total / frozen.atr14_raw
    reference = frozen.support if side == "LONG" else frozen.resistance
    total_bps = 0.0 if reference <= 0 else total / reference * 10_000.0
    past_bps = 0.0 if reference <= 0 else past_threshold / reference * 10_000.0
    bin_name = "0.10_TO_0.25" if total_atr < 0.25 else "0.25_TO_0.50" if total_atr < 0.50 else "GE_0.50"
    return total_atr, total_bps, bin_name, past_atr, past_bps


def is_sweep(side: str, candle: Candle5m, frozen: FrozenRange) -> bool:
    if not math.isfinite(frozen.atr14_raw) or frozen.atr14_raw < 0:
        return False
    if side == "LONG":
        return candle.low <= frozen.support - SWEEP_ATR * frozen.atr14_raw
    return candle.high >= frozen.resistance + SWEEP_ATR * frozen.atr14_raw


def reclaim_matches(side: str, close: float, frozen: FrozenRange) -> bool:
    if side == "LONG":
        return frozen.support < close <= frozen.midpoint
    return frozen.midpoint <= close < frozen.resistance


def midpoint_touched(candle: Candle5m, midpoint: float) -> bool:
    return candle.low <= midpoint <= candle.high


_EPISODE_CANCEL = {
    "CONFIRMED_BREAKOUT": "CONFIRMED_BREAKOUT",
    "EXPIRED_48H": "EPISODE_EXPIRED_48H",
    "PAIR_REPLACED": "EPISODE_REPLACED",
    "AMPLITUDE": "AMPLITUDE_INVALIDATED",
    "AMPLITUDE_OUT_OF_RANGE": "AMPLITUDE_INVALIDATED",
    "STRUCTURE_LOST": "STRUCTURE_LOSS",
    "SUPPORT_CLUSTER_MISSING": "STRUCTURE_LOSS",
    "RESISTANCE_CLUSTER_MISSING": "STRUCTURE_LOSS",
}


class SweepReclaimMachine:
    """Structure-only detector. Candidate quota, position, and cooldown never alter setups."""

    def __init__(self, structure_id: str):
        self.structure_id = structure_id
        self.pending: dict[tuple[str, str], dict[str, Any]] = {}
        self.blocked: dict[tuple[str, str], tuple[FrozenRange, int]] = {}
        self.last_sweep_bar: dict[tuple[str, str], datetime] = {}
        self.bar_number = 0

    def _terminal(self, row: dict[str, Any], status: str, reason: str | None, candle: Candle5m) -> dict[str, Any]:
        row.update({"status": status, "terminal_reason": reason, "terminal_decision_at": iso_utc_millis(candle.available_at)})
        frozen = row["_frozen"]
        key = (frozen.range_episode_id, frozen.side)
        self.pending.pop(key, None)
        self.blocked[key] = (frozen, self.bar_number)
        return row

    def _cancel_pending(self, candle: Candle5m, reason: str) -> list[dict[str, Any]]:
        return [self._terminal(row, "CANCELLED", reason, candle) for row in list(self.pending.values())]

    def process_close(
        self,
        candle: Candle5m,
        prior_snapshot: LevelSnapshot | None,
        prior_range_confirmed_at: datetime | None,
        post_episode_id: str | None,
        *,
        episode_event: str | None = None,
        same_split: bool = True,
        contiguous: bool = True,
        decision_end: datetime = TRAIN_END,
    ) -> list[dict[str, Any]]:
        self.bar_number += 1
        completed: list[dict[str, Any]] = []
        if candle.available_at >= decision_end:
            return self._cancel_pending(candle, "TRAIN_BOUNDARY")
        cancellation = None
        if not contiguous:
            cancellation = "DATA_GAP"
        elif not same_split:
            cancellation = "SPLIT_BOUNDARY"
        elif episode_event in _EPISODE_CANCEL:
            cancellation = _EPISODE_CANCEL[episode_event]
        if cancellation:
            completed.extend(self._cancel_pending(candle, cancellation))

        for key, (frozen, terminal_bar) in list(self.blocked.items()):
            extreme_inside = candle.low > frozen.support if frozen.side == "LONG" else candle.high < frozen.resistance
            if self.bar_number > terminal_bar and post_episode_id == frozen.range_episode_id and frozen.support < candle.close < frozen.resistance and extreme_inside:
                del self.blocked[key]

        for key, row in list(self.pending.items()):
            frozen: FrozenRange = row["_frozen"]
            if post_episode_id != frozen.range_episode_id:
                completed.append(self._terminal(row, "CANCELLED", cancellation or "EPISODE_ENDED", candle))
                continue
            if midpoint_touched(candle, frozen.midpoint):
                completed.append(self._terminal(row, "CANCELLED", "MIDPOINT_TOUCHED", candle))
                continue
            delay = self.bar_number - row["_sweep_bar_number"]
            if 1 <= delay <= RECLAIM_BARS and reclaim_matches(frozen.side, candle.close, frozen):
                row.update({"reclaim_type": "S2", "s2_delay_bars": delay, "reclaim_decision_at": iso_utc_millis(candle.available_at), "reclaim_close": candle.close, "episode_active_at_reclaim": True, "breakout_before_entry": False})
                completed.append(self._terminal(row, "RECLAIMED", None, candle))
            elif delay == RECLAIM_BARS:
                completed.append(self._terminal(row, "NO_RECLAIM", "WINDOW_EXPIRED", candle))

        if prior_snapshot is None or prior_range_confirmed_at is None or post_episode_id != prior_snapshot.range_episode_id or cancellation:
            return completed
        for side in ("LONG", "SHORT"):
            frozen = FrozenRange.from_snapshot(candle.symbol, side, prior_snapshot, prior_range_confirmed_at)
            key = (frozen.range_episode_id, side)
            if key in self.pending or key in self.blocked or self.last_sweep_bar.get(key) == candle.open_time or not is_sweep(side, candle, frozen):
                continue
            self.last_sweep_bar[key] = candle.open_time
            total_atr, total_bps, depth_bin, past_atr, past_bps = sweep_depth(side, candle, frozen)
            row: dict[str, Any] = {
                "schema_version": SCHEMA,
                "structure_id": self.structure_id,
                "symbol": frozen.symbol,
                "side": side,
                "range_episode_id": frozen.range_episode_id,
                "range_id": frozen.range_id,
                "range_confirmed_at": iso_utc_millis(frozen.range_confirmed_at),
                "prior_snapshot_decision_at": iso_utc_millis(frozen.snapshot_decision_at),
                "sweep_bar_open_at": iso_utc_millis(candle.open_time),
                "sweep_decision_at": iso_utc_millis(candle.available_at),
                "support": frozen.support,
                "resistance": frozen.resistance,
                "midpoint": frozen.midpoint,
                "ATR14_raw": frozen.atr14_raw,
                "sweep_depth_atr": total_atr,
                "sweep_depth_bps": total_bps,
                "sweep_depth_past_threshold_atr": past_atr,
                "sweep_depth_past_threshold_bps": past_bps,
                "sweep_depth_bin": depth_bin,
                "status": "PENDING_S2",
                "terminal_reason": None,
                "reclaim_type": None,
                "s2_delay_bars": None,
                "reclaim_decision_at": None,
                "reclaim_close": None,
                "causality_snapshot_is_prior_close": prior_snapshot.decision_at == candle.open_time,
                "causality_reclaim_uses_closed_bar": True,
                "midpoint_precedence": "CANCEL_ON_HL_TOUCH_BEFORE_RECLAIM",
                "_frozen": frozen,
                "_sweep_bar_number": self.bar_number,
            }
            row["canonical_sweep_opportunity_id"] = canonical_sweep_opportunity_id(row)
            if midpoint_touched(candle, frozen.midpoint):
                completed.append(self._terminal(row, "CANCELLED", "MIDPOINT_TOUCHED", candle))
            elif reclaim_matches(side, candle.close, frozen):
                row.update({"reclaim_type": "S1", "s2_delay_bars": 0, "reclaim_decision_at": iso_utc_millis(candle.available_at), "reclaim_close": candle.close, "episode_active_at_reclaim": True, "breakout_before_entry": False})
                completed.append(self._terminal(row, "RECLAIMED", None, candle))
            else:
                self.pending[key] = row
        return completed

    def finalize(self, decision_at: datetime) -> list[dict[str, Any]]:
        synthetic = Candle5m("FINAL", decision_at - timedelta(minutes=5), decision_at, 1, 1, 1, 1, 0)
        return self._cancel_pending(synthetic, "TRAIN_BOUNDARY")


def assign_opportunity_weights(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    multiplicity: dict[str, int] = defaultdict(int)
    for row in rows:
        multiplicity[row["canonical_sweep_opportunity_id"]] += 1
    return [
        {**row, "group_multiplicity": multiplicity[row["canonical_sweep_opportunity_id"]], "row_weight": 1.0, "unique_weight": 1.0 / multiplicity[row["canonical_sweep_opportunity_id"]]}
        for row in rows
    ]


def first_passage(entry: float, side: str, midpoint: float, modeled_ranges: Sequence[tuple[float, float] | Candle5m]) -> dict[str, Any]:
    direction = 1.0 if side == "LONG" else -1.0
    favorable = {value: None for value in FAVORABLE_BPS}
    adverse = {value: None for value in ADVERSE_BPS}
    progress = {value: None for value in PROGRESS_PCT}
    distance = direction * (midpoint - entry)
    for index, observed in enumerate(modeled_ranges):
        low, high = (observed.low, observed.high) if isinstance(observed, Candle5m) else observed
        favorable_price = high if side == "LONG" else low
        adverse_price = low if side == "LONG" else high
        favorable_return = max(0.0, direction * (favorable_price - entry) / entry)
        adverse_return = max(0.0, -direction * (adverse_price - entry) / entry)
        # Recording adverse first makes a same-bar tie conservative in every matrix.
        for value in ADVERSE_BPS:
            if adverse[value] is None and adverse_return >= value / 10_000.0:
                adverse[value] = index
        for value in FAVORABLE_BPS:
            if favorable[value] is None and favorable_return >= value / 10_000.0:
                favorable[value] = index
        if distance > 0:
            made = max(0.0, direction * (favorable_price - entry) / distance * 100.0)
            for value in PROGRESS_PCT:
                if progress[value] is None and made >= value:
                    progress[value] = index
    matrix = {
        f"F{f}_A{a}": "ADVERSE_FIRST" if adverse[a] is not None and (favorable[f] is None or adverse[a] <= favorable[f]) else "FAVORABLE_FIRST" if favorable[f] is not None else "NEITHER"
        for f in FAVORABLE_BPS for a in ADVERSE_BPS
    }
    progress_matrix = {
        f"P{p}_A{a}": "ADVERSE_FIRST" if adverse[a] is not None and (progress[p] is None or adverse[a] <= progress[p]) else "PROGRESS_FIRST" if progress[p] is not None else "NEITHER"
        for p in PROGRESS_PCT for a in ADVERSE_BPS
    }
    return {
        "favorable_first_bar": {str(key): value for key, value in favorable.items()},
        "adverse_first_bar": {str(key): value for key, value in adverse.items()},
        "midpoint_progress_first_bar": {str(key): value for key, value in progress.items()},
        "favorable_first_completed_bars": {str(key): None if value is None else value + 1 for key, value in favorable.items()},
        "adverse_first_completed_bars": {str(key): None if value is None else value + 1 for key, value in adverse.items()},
        "progress_first_completed_bars": {str(key): None if value is None else value + 1 for key, value in progress.items()},
        "favorable_first_minutes": {str(key): None if value is None else (value + 1) * 5 for key, value in favorable.items()},
        "adverse_first_minutes": {str(key): None if value is None else (value + 1) * 5 for key, value in adverse.items()},
        "progress_first_minutes": {str(key): None if value is None else (value + 1) * 5 for key, value in progress.items()},
        "favorable_adverse_matrix": matrix,
        "progress_adverse_matrix": progress_matrix,
        "same_bar_policy": "ADVERSE_FIRST",
    }


def conservative_path(position: Position, event: FillEvent, bars: Sequence[Candle5m]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not bars:
        raise ValueError("RECLAIM_PATH_EMPTY")
    terminal = bars[-1]
    if event.reason == "STOP":
        terminal_prices = (terminal.open, position.stop_at_entry)
    elif event.reason == "STOP_GAP":
        terminal_prices = (terminal.open,)
    elif event.reason in {"TARGET", "TARGET_GAP"}:
        terminal_prices = (position.target_at_entry,)
    else:
        terminal_prices = (terminal.open,)
    modeled = [(bar.low, bar.high) for bar in bars[:-1]] + [(min(terminal_prices), max(terminal_prices))]
    direction = 1.0 if position.side == "LONG" else -1.0
    favorable_values = [high if position.side == "LONG" else low for low, high in modeled]
    adverse_values = [low if position.side == "LONG" else high for low, high in modeled]
    favorable_returns = [max(0.0, direction * (value - position.entry_fill) / position.entry_fill) for value in favorable_values]
    adverse_returns = [max(0.0, -direction * (value - position.entry_fill) / position.entry_fill) for value in adverse_values]
    mfe, mae = max(favorable_returns, default=0.0), max(adverse_returns, default=0.0)
    mfe_bar, mae_bar = favorable_returns.index(mfe), adverse_returns.index(mae)
    horizons = {}
    for minutes in HORIZONS:
        selected = modeled[: minutes // 5]
        favorable = [max(0.0, direction * ((high if position.side == "LONG" else low) - position.entry_fill) / position.entry_fill) for low, high in selected]
        adverse = [max(0.0, -direction * ((low if position.side == "LONG" else high) - position.entry_fill) / position.entry_fill) for low, high in selected]
        horizons[str(minutes)] = {"bars": len(selected), "mature": len(selected) == minutes // 5, "favorable": max(favorable, default=None), "adverse": max(adverse, default=None)}
    path = {
        "mfe": mfe,
        "mae": mae,
        "mfe_bar_index": mfe_bar,
        "mae_bar_index": mae_bar,
        "bars_to_MFE": mfe_bar + 1,
        "bars_to_MAE": mae_bar + 1,
        "time_to_MFE_minutes": (mfe_bar + 1) * 5,
        "time_to_MAE_minutes": (mae_bar + 1) * 5,
        "MFE_before_MAE": mfe_bar < mae_bar,
        "MAE_before_MFE": mae_bar <= mfe_bar,
        "midpoint_touched": any(low <= position.midpoint_at_entry <= high for low, high in modeled),
        "stop_touched": event.reason.startswith("STOP"),
        "target_touched": event.reason.startswith("TARGET"),
        "breakout_exit": event.reason == "TRADE_BREAKOUT",
        "maxhold_exit": event.reason == "MAX_HOLD",
        "holding_bars": position.closed_bars,
        "horizons": horizons,
        "terminal_bar_policy": "ACTUAL_LIFECYCLE_ADVERSE_FIRST_CONSERVATIVE",
    }
    return path, first_passage(position.entry_fill, position.side, position.midpoint_at_entry, modeled)


def stop_recovery(entry: float, midpoint: float, stop_fill: float, side: str, following_bars: Sequence[Candle5m]) -> dict[str, Any]:
    direction = 1.0 if side == "LONG" else -1.0
    result: dict[str, Any] = {"start_policy": "FIRST_COMPLETE_FOLLOWING_BAR"}
    for minutes in HORIZONS:
        selected = list(following_bars[: minutes // 5])
        favorable = [(bar.high if side == "LONG" else bar.low) for bar in selected]
        adverse = [(bar.low if side == "LONG" else bar.high) for bar in selected]
        result[str(minutes)] = {
            "complete_bars": len(selected),
            "mature": len(selected) == minutes // 5,
            "entry_recovered": any(direction * (value - entry) >= 0 for value in favorable),
            "midpoint_recovered": any(direction * (value - midpoint) >= 0 for value in favorable),
            "favorable_from_stop": max((max(0.0, direction * (value - stop_fill) / stop_fill) for value in favorable), default=None),
            "adverse_from_stop": max((max(0.0, -direction * (value - stop_fill) / stop_fill) for value in adverse), default=None),
        }
    return result


def scenario_economics(side: str, entry_base: float, exit_base: float, events: tuple[tuple[float, float], ...]) -> dict[str, dict[str, float]]:
    result = {}
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


def classify_edge(gross: float, net: float, profit_factor: float | str | None) -> str:
    if gross <= 0:
        return "NO_GROSS_EDGE"
    if net <= 0:
        return "COST_LIMITED_SIGNAL"
    numeric_pf = math.inf if profit_factor == "Infinity" else float(profit_factor or 0.0)
    if numeric_pf <= 1:
        raise ValueError("POSITIVE_NET_PROFIT_FACTOR_INVARIANT")
    return "POTENTIAL_NET_EDGE"


def net_edge_present(metrics: Mapping[str, Any]) -> bool:
    net = metrics.get("net_expectancy")
    profit_factor = metrics.get("profit_factor")
    numeric_pf = math.inf if profit_factor == "Infinity" else float(profit_factor or 0.0)
    return net is not None and float(net) > 0 and numeric_pf > 1


def aggregate_economics(
    rows: Sequence[dict[str, Any]],
    weight_key: str = "unique_weight",
    denominator_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = {}
    denominator = list(rows if denominator_rows is None else denominator_rows)
    eligible_weight = sum(float(row.get(weight_key, 0.0)) for row in denominator)
    for scenario in (item.name for item in SCENARIOS):
        active = [row for row in rows if row.get("scenarios") and row.get(weight_key, 0) > 0]
        weight = sum(float(row[weight_key]) for row in active)
        if not weight:
            result[scenario] = {"sample_size": 0, "eligible_rows": len(denominator), "effective_retained_opportunity_weight": 0.0, "effective_eligible_opportunity_weight": eligible_weight, "retention_rate": 0.0 if eligible_weight else None, "abstention_rate": 1.0 if eligible_weight else None, "gross_expectancy": None, "net_expectancy": None, "profit_factor": None}
            continue
        weighted = [(float(row[weight_key]), row["scenarios"][scenario]) for row in active]
        mean = lambda key: sum(w * values[key] for w, values in weighted) / weight
        gains = sum(w * max(values["net_return"], 0.0) for w, values in weighted)
        losses = sum(w * min(values["net_return"], 0.0) for w, values in weighted)
        gross, net = mean("gross_return"), mean("net_return")
        pf: float | str = "Infinity" if losses == 0 else gains / abs(losses)
        result[scenario] = {
            "sample_size": len(active), "eligible_rows": len(denominator),
            "effective_retained_opportunity_weight": weight,
            "effective_eligible_opportunity_weight": eligible_weight,
            "retention_rate": None if not eligible_weight else weight / eligible_weight,
            "abstention_rate": None if not eligible_weight else 1.0 - weight / eligible_weight,
            "gross_expectancy": gross, "net_expectancy": net, "profit_factor": pf,
            "win_rate": sum(w * (values["net_return"] > 0) for w, values in weighted) / weight,
            "stop_rate": sum(float(row[weight_key]) * row["exit_reason"].startswith("STOP") for row in active) / weight,
            "target_rate": sum(float(row[weight_key]) * row["exit_reason"].startswith("TARGET") for row in active) / weight,
            "breakout_rate": sum(float(row[weight_key]) * (row["exit_reason"] == "TRADE_BREAKOUT") for row in active) / weight,
            "maxhold_rate": sum(float(row[weight_key]) * (row["exit_reason"] == "MAX_HOLD") for row in active) / weight,
            "break_even_transaction_cost_bps": (gross + mean("funding_return")) * 10_000.0,
            "decision_label": classify_edge(gross, net, pf),
            "whitelist": False,
        }
    return result


def _localize_view(
    entries: Sequence[dict[str, Any]],
    paths: Sequence[dict[str, Any]] = (),
    passages: Sequence[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    multiplicity: dict[str, int] = defaultdict(int)
    keys: set[tuple[str, str]] = set()
    for row in entries:
        key = (row["candidate_id"], row["canonical_sweep_opportunity_id"])
        if key in keys:
            raise RuntimeError("SWEEP_RECLAIM_LOCAL_ENTRY_KEY_INVARIANT")
        keys.add(key)
        multiplicity[row["canonical_sweep_opportunity_id"]] += 1
    local_entries = [{**row, "local_group_multiplicity": multiplicity[row["canonical_sweep_opportunity_id"]], "local_unique_weight": 1.0 / multiplicity[row["canonical_sweep_opportunity_id"]]} for row in entries]
    weights = {(row["candidate_id"], row["canonical_sweep_opportunity_id"]): row["local_unique_weight"] for row in local_entries}

    def attach(rows: Sequence[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        seen = set()
        result = []
        for row in rows:
            key = (row["candidate_id"], row["canonical_sweep_opportunity_id"])
            if key in seen or key not in weights:
                raise RuntimeError(f"SWEEP_RECLAIM_LOCAL_{label}_KEY_INVARIANT")
            seen.add(key)
            result.append({**row, "local_unique_weight": weights[key]})
        return result

    return local_entries, attach(paths, "PATH"), attach(passages, "PASSAGE")


def _canonical_contract_means(rows: Sequence[dict[str, Any]], fields: Sequence[str], *, scenario: str | None = None, weight_key: str = "local_unique_weight") -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["canonical_sweep_opportunity_id"]].append(row)
    result = []
    for identifier, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: row.get("candidate_id", ""))
        weights = [float(row.get(weight_key, row.get("unique_weight", 0.0))) for row in group]
        denominator = sum(weights)
        if denominator <= 0:
            continue
        values = {}
        for field in fields:
            values[field] = sum(weight * float((row["scenarios"][scenario][field] if scenario else row[field])) for row, weight in zip(group, weights)) / denominator
        first = group[0]
        result.append({"canonical_sweep_opportunity_id": identifier, "sweep_decision_at": first["sweep_decision_at"], "symbol": first.get("symbol"), "side": first.get("side"), "reclaim_type": first.get("reclaim_type"), "active_contract_weight": denominator, **values})
    return result


def _weighted_boolean_by_opportunity(rows: Sequence[dict[str, Any]], predicate: Any, weight_key: str = "local_unique_weight") -> tuple[float | None, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["canonical_sweep_opportunity_id"]].append(row)
    means = []
    for group in groups.values():
        group = sorted(group, key=lambda row: row.get("candidate_id", ""))
        denominator = sum(float(row.get(weight_key, row.get("unique_weight", 0.0))) for row in group)
        if denominator:
            means.append(sum(float(row.get(weight_key, row.get("unique_weight", 0.0))) * bool(predicate(row)) for row in group) / denominator)
    return (None if not means else statistics.fmean(means), len(means))


def research_flags(family: str, entries: Sequence[dict[str, Any]], paths: Sequence[dict[str, Any]], passages: Sequence[dict[str, Any]], bootstrap: Mapping[str, Any]) -> dict[str, bool]:
    """Frozen conservative family-level rules; thresholds are not fitted to results."""
    selected_entries = [row for row in entries if row.get("reclaim_type") == family]
    selected_paths = [row for row in paths if row.get("reclaim_type") == family]
    selected_passages = [row for row in passages if row.get("reclaim_type") == family]
    _, selected_paths, selected_passages = _localize_view(selected_entries, selected_paths, selected_passages)
    partial_p50, passage_count = _weighted_boolean_by_opportunity(selected_passages, lambda row: row["progress_adverse_matrix"]["P50_A20"] == "PROGRESS_FIRST")
    partial_p100, _ = _weighted_boolean_by_opportunity(selected_passages, lambda row: row["progress_adverse_matrix"]["P100_A40"] == "PROGRESS_FIRST")
    side_stability = []
    for side in ("LONG", "SHORT"):
        rate, count = _weighted_boolean_by_opportunity([row for row in selected_passages if row.get("side") == side], lambda row: row["progress_adverse_matrix"]["P50_A20"] == "PROGRESS_FIRST")
        side_stability.append(count >= 30 and rate is not None and rate >= 0.50)
    stopped = [row for row in selected_paths if row["exit_reason"].startswith("STOP") and row.get("stop_recovery", {}).get("120", {}).get("mature")]
    stop_entry, stopped_count = _weighted_boolean_by_opportunity(stopped, lambda row: row["stop_recovery"]["120"]["entry_recovered"])
    stop_midpoint, _ = _weighted_boolean_by_opportunity(stopped, lambda row: row["stop_recovery"]["120"]["midpoint_recovered"])
    stop_months = {row["sweep_decision_at"][:7] for row in stopped}
    canonical = _canonical_contract_means(selected_paths, ("gross_return", "net_return"), scenario="BASELINE")
    gross = None if not canonical else statistics.fmean(row["gross_return"] for row in canonical)
    net = None if not canonical else statistics.fmean(row["net_return"] for row in canonical)
    gains = sum(max(row["net_return"], 0.0) for row in canonical)
    losses = sum(min(row["net_return"], 0.0) for row in canonical)
    profit_factor = math.inf if canonical and losses == 0 else (0.0 if not canonical else gains / abs(losses))
    positive_months = sum(statistics.fmean(row["net_return"] for row in canonical if row["sweep_decision_at"].startswith(month)) > 0 for month in {row["sweep_decision_at"][:7] for row in canonical})
    side_net = []
    for side in ("LONG", "SHORT"):
        values = [row["net_return"] for row in canonical if row["side"] == side]
        side_net.append(None if not values else statistics.fmean(values))
    return {
        "PARTIAL_REVERSAL_RESEARCH_JUSTIFIED": passage_count >= 100 and partial_p50 is not None and partial_p50 >= 0.55 and partial_p100 is not None and partial_p100 >= 0.40 and all(side_stability),
        "STOP_GEOMETRY_RESEARCH_JUSTIFIED": stopped_count >= 100 and len(stop_months) >= 6 and stop_entry is not None and stop_entry >= 0.50 and stop_midpoint is not None and stop_midpoint >= 0.25,
        "EXECUTION_COST_RESEARCH_JUSTIFIED": gross is not None and net is not None and gross > 0 >= net,
        "SWEEP_RECLAIM_HYPOTHESIS_PLAUSIBLE": len(canonical) >= 100 and net is not None and net > 0 and profit_factor > 1 and positive_months >= 6 and all(value is not None and value > 0 for value in side_net) and bootstrap.get("net") is not None and bootstrap["net"]["p5"] > 0,
    }


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def synchronized_block_bootstrap(eligible_entries: Sequence[dict[str, Any]], active_paths: Sequence[dict[str, Any]] | None = None, *, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    if active_paths is None:
        active_paths = eligible_entries
        synthetic_entries = [{"candidate_id": row.get("candidate_id", f"R{index}"), "canonical_sweep_opportunity_id": row["canonical_sweep_opportunity_id"]} for index, row in enumerate(eligible_entries)]
        normalized_paths = [{**row, "candidate_id": synthetic_entries[index]["candidate_id"]} for index, row in enumerate(active_paths)]
        _, localized_paths, _ = _localize_view(synthetic_entries, normalized_paths)
    else:
        _, localized_paths, _ = _localize_view(eligible_entries, active_paths)
    anchored = _canonical_contract_means(localized_paths, ("gross_return", "net_return"), scenario="BASELINE")
    starts = [TRAIN_START + timedelta(days=day) for day in range((TRAIN_END - TRAIN_START).days - 7 + 1)]
    blocks = [(start, [row for row in anchored if start <= _parse(row["sweep_decision_at"]) < start + timedelta(days=7)]) for start in starts]
    rng = random.Random(seed)
    gross_draws, net_draws = [], []
    for _ in range(draws):
        sample = []
        synthetic_start = 0
        while synthetic_start < 365:
            block_start, block = blocks[rng.randrange(len(blocks))]
            sample.extend(row for row in block if synthetic_start + (_parse(row["sweep_decision_at"]) - block_start).total_seconds() / 86400.0 < 365)
            synthetic_start += 7
        gross_draws.append(0.0 if not sample else statistics.fmean(row["gross_return"] for row in sample))
        net_draws.append(0.0 if not sample else statistics.fmean(row["net_return"] for row in sample))
    summarize = lambda values: {"mean": statistics.fmean(values), "median": statistics.median(values), "p5": _nearest_rank(values, 0.05), "p95": _nearest_rank(values, 0.95)}
    return {"draws": draws, "seed": seed, "block_days": 7, "synthetic_days": 365, "eligible_daily_starts": len(starts), "empty_blocks": sum(not rows for _, rows in blocks), "nonempty_blocks": sum(bool(rows) for _, rows in blocks), "anchor_memberships": sum(len(rows) for _, rows in blocks), "sample_size": len(anchored), "percentile": "NEAREST_RANK", "gross": summarize(gross_draws), "net": summarize(net_draws)}


def index_v1_trades(v1_trades: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str, str], tuple[dict[str, Any], ...]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in v1_trades:
        grouped[(row["candidate_id"], row["symbol"], row["range_episode_id"], row["side"])].append(row)
    return {key: tuple(sorted(rows, key=lambda row: row["entry_at"])) for key, rows in grouped.items()}


def match_v1(entry: Mapping[str, Any], v1_index: Mapping[tuple[str, str, str, str], Sequence[dict[str, Any]]] | Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(v1_index, Mapping):
        v1_index = index_v1_trades(v1_index)
    key = (str(entry["candidate_id"]), str(entry["symbol"]), str(entry["range_episode_id"]), str(entry["side"]))
    matches = [row for row in v1_index.get(key, ()) if _parse(row["entry_at"]) <= _parse(entry["sweep_decision_at"])]
    if len(matches) > 1:
        raise RuntimeError("SWEEP_RECLAIM_V1_MATCH_QUOTA_INVARIANT")
    original = None if not matches else matches[0]
    return {
        "matched_view": "MATCHED" if original else "UNMATCHED",
        "match_definition": "CAUSAL_SAME_CANDIDATE_SYMBOL_EPISODE_SIDE_PREDECESSOR; NOT_EXACT_SAME_SETUP",
        "matched_v1_entry_at": None if original is None else original["entry_at"],
        "matched_v1_exit_at": None if original is None else original.get("exit_at"),
        "matched_v1_exit_reason": None if original is None else original.get("exit_reason"),
        "matched_v1_scenarios": None if original is None else original.get("scenarios"),
    }


class DiscoveryLifecycle:
    def __init__(self, candidate: RangeCandidate):
        self.candidate = candidate
        self.lifecycle = RangeLifecycleV1(candidate)
        self.pending_meta: dict[str, Any] | None = None
        self.open_meta: dict[str, Any] | None = None
        self.open_position: Position | None = None
        self.path_bars: list[Candle5m] = []

    def schedule(self, opportunity: dict[str, Any], frozen: FrozenRange) -> dict[str, Any]:
        lifecycle = self.lifecycle
        reason = None
        if lifecycle.position is not None:
            reason = "OPEN_POSITION"
        elif lifecycle.pending_entry is not None:
            reason = "PENDING_ENTRY"
        elif not lifecycle.cooldown_ready():
            reason = "COOLDOWN"
        elif not lifecycle.quota_ready(frozen.range_episode_id, frozen.side):
            reason = "QUOTA"
        row = {**opportunity, "entry_status": "REJECTED" if reason else "PENDING", "entry_cancel_reason": reason, "entry_at": None, "entry_base": None, "entry_fill": None, "outcome_status": "NO_FILL" if reason else "AWAITING_ENTRY", "purged": False, "censored": False}
        if reason:
            return row
        pending = PendingEntry(frozen.symbol, frozen.side, _parse(opportunity["reclaim_decision_at"]), _parse(opportunity["reclaim_decision_at"]), frozen.range_episode_id, frozen.range_id, frozen.range_confirmed_at, frozen.support, frozen.resistance, frozen.midpoint, frozen.atr14_raw, "DISCOVERY_SIDECAR", 0.0, None)
        lifecycle.schedule_entry(pending)
        self.pending_meta = row
        return row

    def process_bar(self, candle: Candle5m, *, preceding_episode_active: bool, same_split: bool = True) -> tuple[dict[str, Any] | None, tuple[Position, FillEvent, float, list[Candle5m], dict[str, Any]] | None, dict[str, Any] | None]:
        if not same_split:
            pending, opened = self.finalize("SPLIT_BOUNDARY")
            self.lifecycle = RangeLifecycleV1(self.candidate)
            return pending, None, opened
        lifecycle = self.lifecycle
        exit_event = lifecycle.process_position_open_and_intrabar(candle)
        closed = None
        if exit_event is not None and self.open_position is not None and self.open_meta is not None:
            self.path_bars.append(candle)
            closed = (copy.deepcopy(self.open_position), exit_event, self._exit_base(exit_event, candle, self.open_position), list(self.path_bars), self.open_meta)
            self.open_position = None
            self.open_meta = None
            self.path_bars = []
        position = lifecycle.consume_pending_entry(open_at=candle.open_time, raw_open=candle.open, same_split=same_split, episode_active=preceding_episode_active)
        entry_row = None
        if self.pending_meta is not None:
            entry_row = self.pending_meta
            if position is None:
                entry_row.update({"entry_status": "REJECTED", "entry_cancel_reason": lifecycle.last_entry_cancel_reason, "outcome_status": "NO_FILL"})
            else:
                entry_row.update({"entry_status": "FILLED", "entry_at": iso_utc_millis(candle.open_time), "entry_base": candle.open, "entry_fill": position.entry_fill, "stop_at_entry": position.stop_at_entry, "target_at_entry": position.target_at_entry, "causality_entry_raw_open_only": True, "outcome_status": "OPEN"})
                self.open_position, self.open_meta, self.path_bars = copy.deepcopy(position), entry_row, [candle]
                same_bar_exit = lifecycle.process_position_open_and_intrabar(candle, include_open_gaps=False)
                if same_bar_exit is not None:
                    closed = (copy.deepcopy(position), same_bar_exit, self._exit_base(same_bar_exit, candle, position), list(self.path_bars), entry_row)
                    self.open_position = None
                    self.open_meta = None
                    self.path_bars = []
            self.pending_meta = None
        elif lifecycle.position is not None and exit_event is None:
            self.path_bars.append(candle)
        lifecycle.process_close(candle.close)
        if lifecycle.position is not None and self.open_position is not None:
            self.open_position = copy.deepcopy(lifecycle.position)
        return entry_row, closed, None

    def finalize(self, reason: str = "TRAIN_BOUNDARY") -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        pending = self.pending_meta
        opened = self.open_meta
        if pending is not None:
            pending.update({"entry_status": "PURGED", "entry_cancel_reason": reason, "purged": True, "censored": False})
        if opened is not None:
            opened.update({"outcome_status": "PURGED_OPEN_POSITION", "purge_reason": reason, "purged": True, "censored": True})
        self.lifecycle.pending_entry = None
        self.lifecycle.position = None
        self.pending_meta = None
        self.open_meta = None
        self.open_position = None
        self.path_bars = []
        return pending, opened

    @staticmethod
    def _exit_base(event: FillEvent, candle: Candle5m, position: Position) -> float:
        if event.reason == "STOP":
            return position.stop_at_entry
        if event.reason in {"TARGET", "TARGET_GAP"}:
            return position.target_at_entry
        return candle.open


def _clean_opportunity(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def replay_structure(
    symbol: str,
    candidate: RangeCandidate,
    candles: Sequence[Candle5m],
    snapshots: Sequence[Any],
    funding: Sequence[tuple[datetime, float, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    structure_key = (candidate.cluster_tolerance_atr, candidate.min_range_amplitude_pct)
    structure_id = f"CT{canonical_decimal_12dp(structure_key[0])}_AMP{canonical_decimal_12dp(structure_key[1])}"
    engine = RangeEngineV1(symbol, candidate, CachedRangeRegimeAdapter(list(candles), list(snapshots)))
    machine = SweepReclaimMachine(structure_id)
    mappings = generate_candidate_mappings()
    variants: dict[tuple[float, float], DiscoveryLifecycle] = {}
    representatives: dict[tuple[float, float], RangeCandidate] = {}
    for grid_candidate in candidate_grid():
        if (grid_candidate.cluster_tolerance_atr, grid_candidate.min_range_amplitude_pct) == structure_key:
            key = (grid_candidate.stop_buffer_atr, grid_candidate.target_buffer_atr)
            representatives.setdefault(key, grid_candidate)
    for key, representative in sorted(representatives.items()):
        variants[key] = DiscoveryLifecycle(representative)
    opportunities: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    paths: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    previous: Candle5m | None = None
    for candle_index, candle in enumerate(candles):
        prior_episode = engine.episode
        prior_snapshot = None if prior_episode is None else prior_episode.previous_snapshot
        prior_episode_id = None if prior_episode is None else prior_episode.range_episode_id
        active_preceding = prior_episode is not None and prior_episode.ended_at is None
        same_split = previous is None or previous.segment_id == candle.segment_id
        for (stop_buffer, target_buffer), variant in variants.items():
            entry_update, closed, boundary_open = variant.process_bar(candle, preceding_episode_active=active_preceding, same_split=same_split)
            if entry_update is not None:
                ids = mappings[(structure_key[0], structure_key[1], stop_buffer, target_buffer)]
                entries.extend({**entry_update, "candidate_id": identifier, "stop_buffer_atr": stop_buffer, "target_buffer_atr": target_buffer} for identifier in ids)
            if closed is not None:
                position, event, exit_base, held, metadata = closed
                selected_funding = [item for item in funding if position.entry_at < item[0] <= event.fill_at]
                economic = scenario_economics(position.side, float(metadata["entry_base"]), exit_base, tuple((rate, mark) for _, rate, mark in selected_funding))
                if not math.isclose(economic["BASELINE"]["entry_fill"], position.entry_fill, rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(economic["BASELINE"]["exit_fill"], event.fill_price, rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError("SWEEP_RECLAIM_ACCOUNTING_PARITY_INVARIANT")
                path, passage = conservative_path(position, event, held)
                base = {**metadata, "outcome_status": "EXITED", "purged": False, "censored": False, "exit_at": iso_utc_millis(event.fill_at), "exit_base": exit_base, "exit_fill": event.fill_price, "exit_reason": event.reason, "scenarios": economic, "funding_interval": "(ENTRY,EXIT]", "funding_event_count": len(selected_funding), "funding_timestamps": [iso_utc_millis(item[0]) for item in selected_funding], **path}
                if event.reason.startswith("STOP"):
                    following: list[Candle5m] = []
                    expected = candle.open_time + timedelta(minutes=5)
                    for future in candles[candle_index + 1:candle_index + 25]:
                        if future.open_time != expected or future.segment_id != candle.segment_id:
                            break
                        following.append(future)
                        expected += timedelta(minutes=5)
                    base["stop_recovery"] = stop_recovery(position.entry_fill, position.midpoint_at_entry, event.fill_price, position.side, following)
                else:
                    base["stop_recovery"] = None
                ids = mappings[(structure_key[0], structure_key[1], stop_buffer, target_buffer)]
                paths.extend({**base, "candidate_id": identifier, "stop_buffer_atr": stop_buffer, "target_buffer_atr": target_buffer} for identifier in ids)
                passages.extend({"candidate_id": identifier, "canonical_sweep_opportunity_id": metadata["canonical_sweep_opportunity_id"], "symbol": symbol, "side": position.side, "sweep_decision_at": metadata["sweep_decision_at"], "reclaim_type": metadata["reclaim_type"], "sweep_depth_bin": metadata["sweep_depth_bin"], "s2_delay_bars": metadata["s2_delay_bars"], **passage} for identifier in ids)
            if boundary_open is not None:
                keys = {(identifier, boundary_open["canonical_sweep_opportunity_id"]) for identifier in mappings[(structure_key[0], structure_key[1], stop_buffer, target_buffer)]}
                for row in entries:
                    if (row["candidate_id"], row["canonical_sweep_opportunity_id"]) in keys:
                        row.update({"outcome_status": "PURGED_OPEN_POSITION", "purge_reason": "SPLIT_BOUNDARY", "purged": True, "censored": True})
        output = engine.process(candle, same_split=same_split, embargo=candle.available_at < EMBARGO_END)
        post_id = None if engine.episode is None else engine.episode.range_episode_id
        completed = machine.process_close(candle, prior_snapshot, None if prior_episode is None else prior_episode.range_confirmed_at, post_id, episode_event=output.get("episode_event"), same_split=same_split, contiguous=previous is None or candle.open_time == previous.open_time + timedelta(minutes=5))
        for raw in completed:
            frozen: FrozenRange = raw["_frozen"]
            clean = _clean_opportunity(raw)
            opportunities.append(clean)
            if clean["status"] == "RECLAIMED":
                for (stop_buffer, target_buffer), variant in variants.items():
                    scheduled = variant.schedule(clean, frozen)
                    if scheduled["entry_status"] == "REJECTED":
                        ids = mappings[(structure_key[0], structure_key[1], stop_buffer, target_buffer)]
                        entries.extend({**scheduled, "candidate_id": identifier, "stop_buffer_atr": stop_buffer, "target_buffer_atr": target_buffer} for identifier in ids)
        previous = candle
    opportunities.extend(_clean_opportunity(row) for row in machine.finalize(TRAIN_END))
    for (stop_buffer, target_buffer), variant in variants.items():
        pending, opened = variant.finalize()
        ids = mappings[(structure_key[0], structure_key[1], stop_buffer, target_buffer)]
        if pending is not None:
            entries.extend({**pending, "candidate_id": identifier, "stop_buffer_atr": stop_buffer, "target_buffer_atr": target_buffer} for identifier in ids)
        if opened is not None:
            keys = {(identifier, opened["canonical_sweep_opportunity_id"]) for identifier in ids}
            for row in entries:
                if (row["candidate_id"], row["canonical_sweep_opportunity_id"]) in keys:
                    row.update({"outcome_status": "PURGED_OPEN_POSITION", "purge_reason": "TRAIN_BOUNDARY", "purged": True, "censored": True})
    return opportunities, entries, paths, passages


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="ascii") as handle:
        return [json.loads(line) for line in handle]


def verify_authority(repo_root: Path, output_root: Path, environment: dict[str, str] | None = None) -> dict[str, Any]:
    if SealedPartitionGuard.access_flags(environment) != FLAGS:
        raise PermissionError("AEGIS_RANGE_V2_SWEEP_RECLAIM_TRAIN_PARTITION_VIOLATION")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
    if head != HEAD_AUTHORITY:
        raise SourceIntegrityError("AEGIS_RANGE_V2_SWEEP_RECLAIM_HEAD_DRIFT")
    run_a = (repo_root / "sandbox/aegis_range_strategy_v1/artifacts/r2_train/run_a").resolve()
    prior = (repo_root / "sandbox/aegis_range_strategy_v1/artifacts/range_v2_discovery").resolve()
    output = output_root.resolve()
    immutable = (run_a, prior, (repo_root / "docs").resolve(), (repo_root / "sandbox/aegis_range_strategy_v1/src").resolve())
    if any(output == root or root in output.parents for root in immutable):
        raise PermissionError("AEGIS_RANGE_V2_SWEEP_RECLAIM_OUTPUT_INSIDE_IMMUTABLE_ROOT")
    for name, expected in RUN_A_HASHES.items():
        if _sha256_file(run_a / name) != expected:
            raise SourceIntegrityError("AEGIS_RANGE_V2_SWEEP_RECLAIM_RUN_A_DRIFT")
    for name, expected in PRIOR_DISCOVERY_HASHES.items():
        if _sha256_file(prior / name) != expected:
            raise SourceIntegrityError("AEGIS_RANGE_V2_SWEEP_RECLAIM_PRIOR_DISCOVERY_DRIFT")
    run_manifest = json.loads((run_a / "run_manifest.json").read_text(encoding="ascii"))
    if run_manifest.get("partition_flags") != FLAGS or run_manifest.get("candidate_count") != 384:
        raise SourceIntegrityError("AEGIS_RANGE_V2_SWEEP_RECLAIM_RUN_A_DRIFT")
    cache_manifest = json.loads((run_a / "regime_cache_manifest.json").read_text(encoding="ascii"))
    if set(cache_manifest.get("caches", {})) != set(SYMBOLS):
        raise SourceIntegrityError("AEGIS_RANGE_V2_SWEEP_RECLAIM_CACHE_DRIFT")
    for symbol, item in cache_manifest["caches"].items():
        cache = run_a / "regime_cache" / f"{symbol}.csv.gz"
        if item.get("symbol") != symbol or _sha256_file(cache) != item.get("sha256"):
            raise SourceIntegrityError("AEGIS_RANGE_V2_SWEEP_RECLAIM_CACHE_DRIFT")
    return {"run_a": run_a, "cache_manifest": cache_manifest, "head": head}


def _decorate_entries(entries: list[dict[str, Any]], v1_index: Mapping[tuple[str, str, str, str], Sequence[dict[str, Any]]] | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    multiplicity: dict[str, int] = defaultdict(int)
    for row in entries:
        multiplicity[row["canonical_sweep_opportunity_id"]] += 1
    result = [
        {**row, "group_multiplicity": multiplicity[row["canonical_sweep_opportunity_id"]], "row_weight": 1.0, "unique_weight": 1.0 / multiplicity[row["canonical_sweep_opportunity_id"]], **match_v1(row, v1_index)}
        for row in entries
    ]
    sums: dict[str, float] = defaultdict(float)
    for row in result:
        sums[row["canonical_sweep_opportunity_id"]] += row["unique_weight"]
    if any(not math.isclose(value, 1.0, abs_tol=1e-12) for value in sums.values()):
        raise RuntimeError("SWEEP_RECLAIM_CANDIDATE_WEIGHT_INVARIANT")
    return result


def _passage_diagnostics(rows: Sequence[dict[str, Any]], weight_key: str) -> dict[str, Any]:
    def rate(predicate: Any) -> float | None:
        denominator = sum(float(row[weight_key]) for row in rows)
        return None if not denominator else sum(float(row[weight_key]) * bool(predicate(row)) for row in rows) / denominator

    return {
        "denominator_rows": len(rows),
        "effective_denominator_weight": sum(float(row[weight_key]) for row in rows),
        "F20_before_A20": rate(lambda row: row["favorable_adverse_matrix"]["F20_A20"] == "FAVORABLE_FIRST"),
        "F30_before_A20": rate(lambda row: row["favorable_adverse_matrix"]["F30_A20"] == "FAVORABLE_FIRST"),
        "progress_25_hit": rate(lambda row: row["midpoint_progress_first_bar"]["25"] is not None),
        "progress_50_hit": rate(lambda row: row["midpoint_progress_first_bar"]["50"] is not None),
        "progress_100_hit": rate(lambda row: row["midpoint_progress_first_bar"]["100"] is not None),
    }


def _diagnostics(entries: Sequence[dict[str, Any]], paths: Sequence[dict[str, Any]], passages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    local_entries, local_paths, local_passages = _localize_view(entries, paths, passages)
    identifiers = {row["canonical_sweep_opportunity_id"] for row in local_entries}
    retained = {row["canonical_sweep_opportunity_id"] for row in local_paths}
    result = {
        "sample_sizes": {"candidate_entries": len(entries), "filled_paths": len(paths), "unique_opportunities": len(identifiers), "retained_unique_opportunities": len(retained)},
        "candidate_view": {"economics": aggregate_economics(paths, "row_weight", entries), "first_passage": _passage_diagnostics(passages, "row_weight")},
        "primary_unique_view": {"estimator": "EACH_CANONICAL_ID_WEIGHT_1_SPLIT_EQUALLY_ACROSS_ELIGIBLE_CANDIDATE_ENTRIES_IN_THIS_VIEW", "economics": aggregate_economics(local_paths, "local_unique_weight", local_entries), "first_passage": _passage_diagnostics(local_passages, "local_unique_weight")},
    }
    matched_original = [
        {**row, "scenarios": row["matched_v1_scenarios"], "exit_reason": row["matched_v1_exit_reason"]}
        for row in local_entries if row.get("matched_v1_scenarios") is not None
    ]
    if matched_original:
        result["matched_v1_predecessor"] = {
            "match_definition": "CAUSAL_SAME_CANDIDATE_SYMBOL_EPISODE_SIDE_PREDECESSOR; NOT_EXACT_SAME_SETUP",
            "sample_sizes": {"candidate_rows": len(matched_original), "unique_opportunities": len({row["canonical_sweep_opportunity_id"] for row in matched_original})},
            "candidate_view": aggregate_economics(matched_original, "row_weight", local_entries),
            "primary_unique_view": aggregate_economics(matched_original, "local_unique_weight", local_entries),
        }
    return result


def _event_diagnostics(opportunities: Sequence[dict[str, Any]], paths: Sequence[dict[str, Any]]) -> dict[str, Any]:
    unique: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in opportunities:
        unique[row["canonical_sweep_opportunity_id"]].append(row)
    statuses = {"S1": 0.0, "S2": 0.0, "NO_RECLAIM": 0.0, "CANCELLED": 0.0}
    for group in unique.values():
        labels = [(row["reclaim_type"] if row["status"] == "RECLAIMED" else row["status"]) for row in group]
        if any(label not in statuses for label in labels):
            raise RuntimeError("SWEEP_RECLAIM_EVENT_CLASS_INVARIANT")
        counts = {label: labels.count(label) for label in statuses}
        total_labels = len(labels)
        for label in statuses:
            statuses[label] += counts[label] / total_labels
    total = len(unique)
    candidate_statuses = {
        "S1": sum(row["status"] == "RECLAIMED" and row["reclaim_type"] == "S1" for row in opportunities),
        "S2": sum(row["status"] == "RECLAIMED" and row["reclaim_type"] == "S2" for row in opportunities),
        "NO_RECLAIM": sum(row["status"] == "NO_RECLAIM" for row in opportunities),
        "CANCELLED": sum(row["status"] == "CANCELLED" for row in opportunities),
    }
    order = {}
    for weight in ("row_weight", "unique_weight"):
        denominator = sum(float(row[weight]) for row in paths)
        order["candidate" if weight == "row_weight" else "primary_unique"] = {
            "denominator_rows": len(paths),
            "effective_weight": denominator,
            "MFE_before_MAE": None if not denominator else sum(float(row[weight]) * row["MFE_before_MAE"] for row in paths) / denominator,
            "MAE_before_MFE": None if not denominator else sum(float(row[weight]) * row["MAE_before_MFE"] for row in paths) / denominator,
        }
    mature = statuses["S1"] + statuses["S2"] + statuses["NO_RECLAIM"]
    mature_s2_window = statuses["S2"] + statuses["NO_RECLAIM"]
    if not math.isclose(sum(statuses.values()), total, abs_tol=1e-12):
        raise RuntimeError("SWEEP_RECLAIM_EVENT_CLASS_INVARIANT")
    return {
        "canonical_conflict_policy": "FRACTIONAL_EQUAL_WEIGHT_ACROSS_STRUCTURAL_ROWS_WITHIN_CANONICAL_ID",
        "unique_real_sweeps": total,
        "cancelled_denominator": total,
        "mature_reclaim_window_denominator": mature,
        "mature_s2_window_denominator": mature_s2_window,
        "NO_RECLAIM_among_mature_s2_window_rate": None if not mature_s2_window else statuses["NO_RECLAIM"] / mature_s2_window,
        **{f"{key}_count": value for key, value in statuses.items()},
        **{f"{key}_rate": (None if not total else value / total) for key, value in statuses.items()},
        "event_views": {
            "candidate_structural": {"denominator": len(opportunities), **{f"{key}_count": value for key, value in candidate_statuses.items()}, **{f"{key}_rate": (None if not opportunities else value / len(opportunities)) for key, value in candidate_statuses.items()}},
            "primary_unique": {"denominator": total, **{f"{key}_count": value for key, value in statuses.items()}, **{f"{key}_rate": (None if not total else value / total) for key, value in statuses.items()}},
        },
        "excursion_order": order,
    }


def _symbol_month_diagnostics(opportunities: Sequence[dict[str, Any]], entries: Sequence[dict[str, Any]], paths: Sequence[dict[str, Any]], passages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    symbols = []
    for symbol in SYMBOLS:
        symbol_opportunities = [row for row in opportunities if row["symbol"] == symbol]
        symbol_entries = [row for row in entries if row["symbol"] == symbol]
        symbol_paths = [row for row in paths if row["symbol"] == symbol]
        symbol_passages = [row for row in passages if row["symbol"] == symbol]
        events = _event_diagnostics(symbol_opportunities, symbol_paths)
        diagnostics = _diagnostics(symbol_entries, symbol_paths, symbol_passages)
        primary = diagnostics["primary_unique_view"]
        candidate = diagnostics["candidate_view"]
        baseline = primary["economics"]["BASELINE"]
        symbols.append({
            "symbol": symbol,
            "opportunities": events["unique_real_sweeps"], "S1_count": events["S1_count"], "S2_count": events["S2_count"],
            "gross_expectancy": baseline["gross_expectancy"], "net_expectancy": baseline["net_expectancy"], "profit_factor": baseline["profit_factor"],
            "stop_rate": baseline.get("stop_rate"), "target_rate": baseline.get("target_rate"),
            "F20_before_A20": primary["first_passage"]["F20_before_A20"], "F30_before_A20": primary["first_passage"]["F30_before_A20"],
            "midpoint_hit": primary["first_passage"]["progress_100_hit"], "primary_unique": primary, "candidate": candidate,
            "SYMBOL_WHITELIST_AUTHORIZED": False,
        })
    months = []
    for month in (f"2024-{value:02d}" for value in range(1, 13)):
        for family in ("S1", "S2"):
            selected_entries = [row for row in entries if row["sweep_decision_at"].startswith(month) and row["reclaim_type"] == family]
            selected_paths = [row for row in paths if row["sweep_decision_at"].startswith(month) and row["reclaim_type"] == family]
            selected_passages = [row for row in passages if row["sweep_decision_at"].startswith(month) and row["reclaim_type"] == family]
            diagnostics = _diagnostics(selected_entries, selected_paths, selected_passages)
            baseline = diagnostics["primary_unique_view"]["economics"]["BASELINE"]
            months.append({"month": month, "family": family, "opportunities": len({row["canonical_sweep_opportunity_id"] for row in selected_entries}), "gross_expectancy": baseline["gross_expectancy"], "net_expectancy": baseline["net_expectancy"], "profit_factor": baseline["profit_factor"], "stop_rate": baseline.get("stop_rate"), "target_rate": baseline.get("target_rate"), "diagnostics": diagnostics})
    return {"symbols": symbols, "months_by_family": months, "SYMBOL_WHITELIST_AUTHORIZED": False}


def execute_discovery(repo_root: Path, output_root: Path, environment: dict[str, str] | None = None) -> dict[str, Any]:
    authority = verify_authority(repo_root, output_root, environment)
    run_a = authority["run_a"]
    v1_trades = _load_jsonl(run_a / "trades.jsonl.gz")
    purged_v1_trades = sum(bool(row.get("purged")) for row in v1_trades)
    if purged_v1_trades != 0:
        raise SourceIntegrityError("SWEEP_RECLAIM_V1_PURGED_TRADE_AUTHORITY_INVARIANT")
    v1_index = index_v1_trades(v1_trades)
    all_opportunities: list[dict[str, Any]] = []
    all_entries: list[dict[str, Any]] = []
    all_paths: list[dict[str, Any]] = []
    all_passages: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        candles = load_train_candles(repo_root, symbol)
        funding = load_train_funding(repo_root, symbol)
        snapshots = load_regime_cache(run_a / "regime_cache" / f"{symbol}.csv.gz", len(candles))
        for candidate in structural_candidates():
            opportunities, entries, paths, passages = replay_structure(symbol, candidate, candles, snapshots, funding)
            all_opportunities.extend(opportunities)
            all_entries.extend(entries)
            all_paths.extend(paths)
            all_passages.extend(passages)
    all_opportunities.sort(key=lambda row: (row["sweep_decision_at"], row["symbol"], row["side"], row["structure_id"]))
    all_opportunities = assign_opportunity_weights(all_opportunities)
    all_entries = _decorate_entries(all_entries, v1_index)
    entry_by_key = {(row["candidate_id"], row["canonical_sweep_opportunity_id"]): row for row in all_entries}
    for row in all_paths:
        linked = entry_by_key.get((row["candidate_id"], row["canonical_sweep_opportunity_id"]))
        if linked is None:
            raise RuntimeError("SWEEP_RECLAIM_PATH_ENTRY_JOIN_INVARIANT")
        row.update({key: linked[key] for key in ("group_multiplicity", "row_weight", "unique_weight", "matched_view", "matched_v1_entry_at")})
        linked.update({"outcome_status": "EXITED", "purged": False, "censored": False})
    for row in all_passages:
        linked = entry_by_key.get((row["candidate_id"], row["canonical_sweep_opportunity_id"]))
        if linked is None:
            raise RuntimeError("SWEEP_RECLAIM_PASSAGE_ENTRY_JOIN_INVARIANT")
        row.update({key: linked[key] for key in ("group_multiplicity", "row_weight", "unique_weight", "matched_view", "matched_v1_entry_at")})
    all_entries.sort(key=lambda row: (row["sweep_decision_at"], row["candidate_id"]))
    all_paths.sort(key=lambda row: (row["entry_at"], row["candidate_id"]))
    all_passages.sort(key=lambda row: (row["canonical_sweep_opportunity_id"], row["candidate_id"]))
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, rows in ((ARTIFACT_NAMES[0], all_opportunities), (ARTIFACT_NAMES[1], all_entries), (ARTIFACT_NAMES[2], all_paths), (ARTIFACT_NAMES[3], all_passages)):
        artifacts[name] = deterministic_gzip_jsonl(output_root / name, rows)
    symbol_month = _symbol_month_diagnostics(all_opportunities, all_entries, all_paths, all_passages)
    artifacts[ARTIFACT_NAMES[4]] = _write_json(output_root / ARTIFACT_NAMES[4], symbol_month)
    baseline_s1 = [row for row in all_paths if row["reclaim_type"] == "S1"]
    baseline_s2 = [row for row in all_paths if row["reclaim_type"] == "S2"]
    s1_entries = [row for row in all_entries if row["reclaim_type"] == "S1"]
    s2_entries = [row for row in all_entries if row["reclaim_type"] == "S2"]
    bootstraps = {"S1_BASELINE": synchronized_block_bootstrap(s1_entries, baseline_s1), "S2_BASELINE": synchronized_block_bootstrap(s2_entries, baseline_s2)}
    family_views = {}
    family_flags = {}
    family_classifications = {}
    for family in ("S1", "S2"):
        family_views[family] = {}
        family_entries = [row for row in all_entries if row["reclaim_type"] == family]
        family_paths = [row for row in all_paths if row["reclaim_type"] == family]
        family_passages = [row for row in all_passages if row["reclaim_type"] == family]
        family_views[family]["FULL"] = _diagnostics(family_entries, family_paths, family_passages)
        family_views[family]["MATCHED"] = _diagnostics([row for row in family_entries if row["matched_view"] == "MATCHED"], [row for row in family_paths if row["matched_view"] == "MATCHED"], [row for row in family_passages if row["matched_view"] == "MATCHED"])
        family_flags[family] = research_flags(family, all_entries, all_paths, all_passages, bootstraps[f"{family}_BASELINE"])
        baseline = family_views[family]["FULL"]["primary_unique_view"]["economics"]["BASELINE"]
        family_classifications[family] = None if baseline["gross_expectancy"] is None else classify_edge(baseline["gross_expectancy"], baseline["net_expectancy"], baseline["profit_factor"])
    breakdowns = {}
    for dimension in ("side", "sweep_depth_bin", "s2_delay_bars"):
        breakdowns[dimension] = {}
        for value in sorted({row.get(dimension) for row in all_entries}, key=lambda item: (item is None, str(item))):
            selected_entries = [row for row in all_entries if row.get(dimension) == value]
            selected_paths = [row for row in all_paths if row.get(dimension) == value]
            selected_passages = [row for row in all_passages if row.get(dimension) == value]
            breakdowns[dimension][str(value)] = _diagnostics(selected_entries, selected_paths, selected_passages)
    s1_base = family_views["S1"]["FULL"]["primary_unique_view"]["economics"]["BASELINE"]
    s2_base = family_views["S2"]["FULL"]["primary_unique_view"]["economics"]["BASELINE"]
    objective = {
        "SWEEP_RECLAIM_HYPOTHESIS_PLAUSIBLE": family_flags["S1"]["SWEEP_RECLAIM_HYPOTHESIS_PLAUSIBLE"] or family_flags["S2"]["SWEEP_RECLAIM_HYPOTHESIS_PLAUSIBLE"],
        "S1_GROSS_EDGE_PRESENT": s1_base["gross_expectancy"] is not None and s1_base["gross_expectancy"] > 0,
        "S1_NET_EDGE_PRESENT": net_edge_present(s1_base),
        "S2_GROSS_EDGE_PRESENT": s2_base["gross_expectancy"] is not None and s2_base["gross_expectancy"] > 0,
        "S2_NET_EDGE_PRESENT": net_edge_present(s2_base),
        "PARTIAL_REVERSAL_RESEARCH_JUSTIFIED": any(flags["PARTIAL_REVERSAL_RESEARCH_JUSTIFIED"] for flags in family_flags.values()),
        "STOP_GEOMETRY_RESEARCH_JUSTIFIED": any(flags["STOP_GEOMETRY_RESEARCH_JUSTIFIED"] for flags in family_flags.values()),
        "EXECUTION_COST_RESEARCH_JUSTIFIED": any(flags["EXECUTION_COST_RESEARCH_JUSTIFIED"] for flags in family_flags.values()),
        "SYMBOL_WHITELIST_AUTHORIZED": False,
    }
    opportunity_levels: dict[str, set[tuple[float, float, float, str]]] = defaultdict(set)
    for row in all_opportunities:
        opportunity_levels[row["canonical_sweep_opportunity_id"]].add((row["support"], row["resistance"], row["midpoint"], row["range_id"]))
    reclaimed_structural_rows = sum(row["status"] == "RECLAIMED" for row in all_opportunities)
    path_keys = sorted((row["candidate_id"], row["canonical_sweep_opportunity_id"]) for row in all_paths)
    passage_keys = sorted((row["candidate_id"], row["canonical_sweep_opportunity_id"]) for row in all_passages)
    summary = {
        "labels": list(DISCOVERY_LABELS),
        "SYMBOL_WHITELIST_AUTHORIZED": False,
        "decision_labels": list(DECISION_LABELS),
        "family_views": family_views,
        "family_classifications": family_classifications,
        "event_diagnostics": _event_diagnostics(all_opportunities, all_paths),
        "bootstrap": bootstraps,
        "breakdowns": breakdowns,
        "family_research_flags": family_flags,
        "objective_booleans": objective,
        "research_flag_rules": {"partial_reversal": "N_UNIQUE_GE_100; P50_BEFORE_A20_GE_55%; P100_BEFORE_A40_GE_40%; EACH_SIDE_N_GE_30_AND_P50_BEFORE_A20_GE_50%", "stop_geometry": "N_MATURE_STOP_GE_100; SIX_UTC_MONTHS; ENTRY_RECOVERY_120M_GE_50%; MIDPOINT_RECOVERY_120M_GE_25%; STOP_UNCHANGED", "execution": "GROSS_POSITIVE_AND_NET_NONPOSITIVE", "sweep_plausible": "N_UNIQUE_GE_100; NET_POSITIVE; PF_GT_1; SIX_POSITIVE_UTC_MONTHS; BOTH_SIDES_NET_POSITIVE; S1_OR_S2_BOOTSTRAP_NET_P5_POSITIVE"},
    }
    artifacts[ARTIFACT_NAMES[5]] = _write_json(output_root / ARTIFACT_NAMES[5], summary)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": STATUS,
        "labels": list(DISCOVERY_LABELS),
        "head_authority": HEAD_AUTHORITY,
        "partition_flags": FLAGS,
        "train": {"start_inclusive": iso_utc_millis(TRAIN_START), "end_exclusive": iso_utc_millis(TRAIN_END)},
        "input_hashes": {"run_a": RUN_A_HASHES, "prior_discovery": PRIOR_DISCOVERY_HASHES, "regime_caches": {symbol: authority["cache_manifest"]["caches"][symbol]["sha256"] for symbol in SYMBOLS}},
        "source_hashes": {
            "discovery_module": _sha256_file(Path(__file__)),
            "cli_script": _sha256_file(Path(__file__).resolve().parents[2] / "scripts" / "run_sweep_reclaim_discovery.py"),
            "test_file": _sha256_file(Path(__file__).resolve().parents[2] / "tests" / "test_sweep_reclaim_discovery.py"),
            "python_version": platform.python_version(),
            "platform": f"{platform.system()}-{platform.machine()}",
        },
        "candidate_count": 384,
        "structural_replays_per_symbol": 6,
        "sidecar_lifecycles_per_structure": 4,
        "candidate_multiplicity_per_sidecar": 16,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "utc_block_days": 7, "synthetic_days": 365, "percentile": "NEAREST_RANK"},
        "audit": {
            "sweep_le_reclaim": all(_parse(row["sweep_decision_at"]) <= _parse(row["reclaim_decision_at"]) for row in all_opportunities if row["status"] == "RECLAIMED"),
            "reclaim_lt_entry": all(_parse(row["reclaim_decision_at"]) < _parse(row["entry_at"]) for row in all_entries if row["entry_status"] == "FILLED"),
            "no_future_setup_use": all(row["causality_snapshot_is_prior_close"] and _parse(row["prior_snapshot_decision_at"]) <= _parse(row["sweep_decision_at"]) for row in all_opportunities),
            "closed_reclaim_only": all(row["causality_reclaim_uses_closed_bar"] for row in all_opportunities),
            "raw_open_only": all(row.get("causality_entry_raw_open_only") is True for row in all_entries if row["entry_status"] == "FILLED"),
            "frozen_boundary_unchanged": all((row["support"], row["resistance"], row["midpoint"], row["range_id"]) in opportunity_levels[row["canonical_sweep_opportunity_id"]] for row in all_entries),
            "episode_active_at_reclaim": all(row.get("episode_active_at_reclaim") is True for row in all_entries),
            "no_breakout_before_entry": all(row.get("breakout_before_entry") is False for row in all_entries),
            "no_oos": all(TRAIN_START <= _parse(row["sweep_decision_at"]) <= TRAIN_END for row in all_opportunities) and all(row.get("entry_at") is None or _parse(row["entry_at"]) < TRAIN_END for row in all_entries),
            "funding_interval_entry_exclusive_exit_inclusive": all(row["funding_interval"] == "(ENTRY,EXIT]" for row in all_paths),
            "adverse_first": all(row["same_bar_policy"] == "ADVERSE_FIRST" for row in all_passages),
            "candidate_weight_sums_one": all(math.isclose(sum(row["unique_weight"] for row in all_entries if row["canonical_sweep_opportunity_id"] == identifier), 1.0, abs_tol=1e-12) for identifier in {row["canonical_sweep_opportunity_id"] for row in all_entries}),
            "candidate_entry_cardinality": len(all_entries) == reclaimed_structural_rows * 64,
            "all_entries_terminally_disposed": all(row["entry_status"] != "PENDING" and row["outcome_status"] not in {"AWAITING_ENTRY", "OPEN"} for row in all_entries),
            "path_entry_cardinality": path_keys == passage_keys and all(key in entry_by_key for key in path_keys),
            "purged_entries": sum(row.get("purged", False) for row in all_entries),
            "censored_entries": sum(row.get("censored", False) for row in all_entries),
            "rejected_entries": sum(row["entry_status"] == "REJECTED" for row in all_entries),
        },
        "determinism": {"gzip_mtime": 0, "ascii": True, "json_sorted_keys": True, "stable_row_sort": True, "bootstrap_seed": BOOTSTRAP_SEED, "percentile": "NEAREST_RANK"},
        "artifacts": artifacts,
    }
    asserted = [value for value in manifest["audit"].values() if isinstance(value, bool)]
    if not all(asserted):
        raise RuntimeError("SWEEP_RECLAIM_CAUSALITY_INVARIANT")
    manifest_path = output_root / ARTIFACT_NAMES[6]
    _write_json(manifest_path, manifest)
    if set(path.name for path in output_root.iterdir()) != set(ARTIFACT_NAMES):
        raise RuntimeError("SWEEP_RECLAIM_ARTIFACT_SET_INVARIANT")
    return {**manifest, "diagnostics_manifest_sha256": _sha256_file(manifest_path)}


def reproducible_sample() -> dict[str, Any]:
    from .models import RangePair

    origin = datetime(2024, 6, 1, tzinfo=timezone.utc)
    pair = RangePair("s", "r", 100.0, 110.0, 105.0, 10.0, 2, 2, origin, origin)
    snapshot = LevelSnapshot(origin, pair, 2.0, "episode", "range")
    machine = SweepReclaimMachine("sample")
    bars = [
        Candle5m("BTCUSDT", origin, origin + timedelta(minutes=5), 101, 104, 99.7, 101, 1),
        Candle5m("BTCUSDT", origin + timedelta(minutes=5), origin + timedelta(minutes=10), 101, 104, 100, 102, 1),
    ]
    rows = machine.process_close(bars[0], snapshot, origin - timedelta(hours=1), "episode")
    rows += machine.process_close(bars[1], snapshot, origin - timedelta(hours=1), "episode")
    cleaned = assign_opportunity_weights([_clean_opportunity(row) for row in rows])
    opportunity = cleaned[0]
    lifecycle = RangeLifecycleV1(candidate_grid()[0])
    pending = PendingEntry("BTCUSDT", "LONG", bars[0].available_at, bars[0].available_at, "episode", "range", origin - timedelta(hours=1), 100, 110, 105, 2, "SAMPLE", 0, None)
    lifecycle.schedule_entry(pending)
    entry_bar = bars[1]
    position = lifecycle.consume_pending_entry(open_at=entry_bar.open_time, raw_open=entry_bar.open, same_split=True, episode_active=True)
    if position is None:
        raise RuntimeError("SWEEP_RECLAIM_SAMPLE_ENTRY_INVARIANT")
    exit_bar = Candle5m("BTCUSDT", origin + timedelta(minutes=10), origin + timedelta(minutes=15), 102, 106, 101, 104, 1)
    event = lifecycle.process_position_open_and_intrabar(exit_bar)
    if event is None:
        raise RuntimeError("SWEEP_RECLAIM_SAMPLE_EXIT_INVARIANT")
    path, passage = conservative_path(position, event, [entry_bar, exit_bar])
    entry = {"canonical_sweep_opportunity_id": opportunity["canonical_sweep_opportunity_id"], "entry_at": iso_utc_millis(entry_bar.open_time), "entry_base": entry_bar.open, "entry_fill": position.entry_fill}
    path_row = {**entry, "side": "LONG", "sweep_decision_at": opportunity["sweep_decision_at"], "exit_reason": event.reason, "unique_weight": 1, "row_weight": 1, "scenarios": scenario_economics("LONG", entry_bar.open, position.target_at_entry, ()), **path}
    serialized_mapping = {"|".join(canonical_decimal_12dp(value) for value in key): list(ids) for key, ids in generate_candidate_mappings().items()}
    payload = {"opportunities": cleaned, "entries": [entry], "paths": [path_row], "first_passage": [passage], "metrics": aggregate_economics([path_row]), "candidate_mapping": serialized_mapping}
    return {**payload, "sha256": hashlib.sha256(_json(payload).encode("ascii")).hexdigest()}
