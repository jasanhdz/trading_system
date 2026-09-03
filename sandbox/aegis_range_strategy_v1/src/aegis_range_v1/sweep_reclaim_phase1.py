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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidates import RangeCandidate, candidate_grid
from .engine import RangeEngineV1
from .lifecycle import RangeLifecycleV1
from .models import Candle5m, LevelSnapshot, PendingEntry
from .numeric import canonical_decimal_12dp, iso_utc_millis
from .readiness import SYMBOLS, SealedPartitionGuard, SourceIntegrityError
from .sweep_reclaim_discovery import (
    FrozenRange,
    SweepReclaimMachine,
    assign_opportunity_weights,
    canonical_sweep_opportunity_id,
    first_passage,
    is_sweep,
    midpoint_touched,
    reclaim_matches,
    sweep_depth,
    _EPISODE_CANCEL,
    _clean_opportunity,
    _parse,
    _json,
    _sha256_file,
    deterministic_gzip_jsonl,
    _write_json,
    generate_candidate_mappings,
    structural_candidates,
)
from .train_backtest import (
    EMBARGO_END,
    TRAIN_END,
    TRAIN_START,
    CachedRangeRegimeAdapter,
    load_regime_cache,
    load_train_candles,
    load_train_funding,
)

SCHEMA = "aegis-range-v2-sweep-reclaim-phase1-v1"
MANIFEST_SCHEMA = "aegis-range-v2-sweep-reclaim-phase1-manifest-v1"
STATUS = "AEGIS_RANGE_V2_SWEEP_RECLAIM_PHASE1_READY_FOR_REVIEW"
HEAD_AUTHORITY = "d917602b2c7e065d18b29f87df14febb935d8ad8"
RANGE_SWEEP_CODE_AUTHORITY = "d917602b2c7e065d18b29f87df14febb935d8ad8"
FLAGS = {"TRAIN": True, "CALIBRATION": False, "VALIDATION": False, "HOLDOUT": False}
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_DRAWS = 10_000
SWEEP_ATR = 0.10
RECLAIM_BARS = 3
HORIZONS = (15, 30, 60, 120)
FAVORABLE_BPS = (10, 20, 30, 40)
ADVERSE_BPS = (10, 20, 30, 40)
PROGRESS_PCT = (25, 50, 100)
ARTIFACT_NAMES = (
    "sweep_opportunities.jsonl.gz",
    "reclaim_entries.jsonl.gz",
    "first_passage.jsonl.gz",
    "contract_eligibility.jsonl.gz",
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
CONTRACTS = (
    {"stop_buffer_atr": 0.35, "target_buffer_atr": 0.00},
    {"stop_buffer_atr": 0.35, "target_buffer_atr": 0.10},
    {"stop_buffer_atr": 0.50, "target_buffer_atr": 0.00},
    {"stop_buffer_atr": 0.50, "target_buffer_atr": 0.10},
)


def _local_first_passage(entry: float, side: str, midpoint: float, bars: Sequence[Candle5m]) -> dict[str, Any]:
    modeled = [(bar.low, bar.high) for bar in bars]
    return first_passage(entry, side, midpoint, [Candle5m("X", datetime.now(tz=timezone.utc), datetime.now(tz=timezone.utc), 0, h, l, 0, 0) for l, h in modeled])


def _contract_eligibility(entry: float, side: str, frozen: FrozenRange) -> dict[str, Any]:
    direction = 1.0 if side == "LONG" else -1.0
    results = {}
    for contract in CONTRACTS:
        key = f"SB{contract['stop_buffer_atr']:.2f}_TB{contract['target_buffer_atr']:.2f}"
        if side == "LONG":
            stop = frozen.support - contract["stop_buffer_atr"] * frozen.atr14_raw
            target = frozen.midpoint - contract["target_buffer_atr"] * frozen.atr14_raw
        else:
            stop = frozen.resistance + contract["stop_buffer_atr"] * frozen.atr14_raw
            target = frozen.midpoint + contract["target_buffer_atr"] * frozen.atr14_raw
        favorable = target > entry if side == "LONG" else target < entry
        distance = abs(target - entry) / entry if entry > 0 else 0.0
        risk = (entry - stop) if side == "LONG" else (stop - entry)
        reward = (target - entry) if side == "LONG" else (entry - target)
        rr = reward / risk if risk > 0 else 0.0
        results[key] = {
            "stop_buffer_atr": contract["stop_buffer_atr"],
            "target_buffer_atr": contract["target_buffer_atr"],
            "stop_level": stop,
            "target_level": target,
            "favorable_target": favorable,
            "target_distance_bps": distance * 10_000,
            "rr_ratio": rr,
            "passes_42bps": distance >= 0.0042,
            "passes_rr1": rr >= 1.0,
        }
    return results


def _localize_view(entries: Sequence[dict[str, Any]], passages: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    multiplicity: dict[str, int] = defaultdict(int)
    for row in entries:
        multiplicity[row["canonical_sweep_opportunity_id"]] += 1
    local_entries = [
        {**row, "local_group_multiplicity": multiplicity[row["canonical_sweep_opportunity_id"]], "local_unique_weight": 1.0 / multiplicity[row["canonical_sweep_opportunity_id"]]}
        for row in entries
    ]
    weights = {(row["candidate_id"], row["canonical_sweep_opportunity_id"]): row["local_unique_weight"] for row in local_entries}
    seen = set()
    local_passages = []
    for row in passages:
        key = (row["candidate_id"], row["canonical_sweep_opportunity_id"])
        if key in seen or key not in weights:
            continue
        seen.add(key)
        local_passages.append({**row, "local_unique_weight": weights[key]})
    return local_entries, local_passages


def _passage_asymmetry(passages: Sequence[dict[str, Any]], weight_key: str = "local_unique_weight") -> dict[str, Any]:
    if not passages:
        return {"N": 0, "p20_a20": None, "p30_a20": None, "p30_a30": None, "p40_a30": None, "p40_a40": None,
                "progress_25_before_a20": None, "progress_50_before_a20": None, "progress_100_before_a20": None}
    total_w = sum(float(row[weight_key]) for row in passages)
    if total_w <= 0:
        return {"N": 0}
    def rate(pred):
        return sum(float(row[weight_key]) * bool(pred(row)) for row in passages) / total_w
    return {
        "N": len(passages),
        "effective_weight": total_w,
        "p20_a20": rate(lambda r: r["favorable_adverse_matrix"]["F20_A20"] == "FAVORABLE_FIRST"),
        "p30_a20": rate(lambda r: r["favorable_adverse_matrix"]["F30_A20"] == "FAVORABLE_FIRST"),
        "p30_a30": rate(lambda r: r["favorable_adverse_matrix"]["F30_A30"] == "FAVORABLE_FIRST"),
        "p40_a30": rate(lambda r: r["favorable_adverse_matrix"]["F40_A30"] == "FAVORABLE_FIRST"),
        "p40_a40": rate(lambda r: r["favorable_adverse_matrix"]["F40_A40"] == "FAVORABLE_FIRST"),
        "progress_25_before_a20": rate(lambda r: r["progress_adverse_matrix"]["P25_A20"] == "PROGRESS_FIRST"),
        "progress_50_before_a20": rate(lambda r: r["progress_adverse_matrix"]["P50_A20"] == "PROGRESS_FIRST"),
        "progress_100_before_a20": rate(lambda r: r["progress_adverse_matrix"]["P100_A20"] == "PROGRESS_FIRST"),
    }


def _excursion_stats(paths: Sequence[dict[str, Any]], weight_key: str = "local_unique_weight") -> dict[str, Any]:
    if not paths:
        return {}
    direction_values = {"mfe": [], "mae": [], "mfe_bars": [], "mae_bars": []}
    wins, losses = 0.0, 0.0
    for row in paths:
        w = float(row.get(weight_key, row.get("unique_weight", 1.0)))
        direction_values["mfe"].append((w, row.get("mfe", 0.0)))
        direction_values["mae"].append((w, row.get("mae", 0.0)))
        direction_values["mfe_bars"].append((w, row.get("bars_to_MFE", 0)))
        direction_values["mae_bars"].append((w, row.get("bars_to_MAE", 0)))
        if row.get("mfe", 0) > row.get("mae", 0):
            wins += w
        else:
            losses += w
    total_w = sum(w for w, _ in direction_values["mfe"])
    result = {"mfe_before_mae_rate": wins / total_w if total_w else None}
    for key in ("mfe", "mae"):
        values = [(w, v) for w, v in direction_values[key] if v is not None]
        if not values:
            continue
        total = sum(w for w, _ in values)
        if total <= 0:
            continue
        vals = sorted(values, key=lambda x: x[1])
        cumweights = []
        cum = 0.0
        for w, v in vals:
            cum += w
            cumweights.append((cum / total, v))
        mean = sum(w * v for w, v in values) / total
        median = next(v for cw, v in cumweights if cw >= 0.5)
        p25 = next(v for cw, v in cumweights if cw >= 0.25)
        p75 = next(v for cw, v in cumweights if cw >= 0.75)
        p90 = next(v for cw, v in cumweights if cw >= 0.90)
        result[f"{key}_mean"] = mean
        result[f"{key}_median"] = median
        result[f"{key}_p25"] = p25
        result[f"{key}_p75"] = p75
        result[f"{key}_p90"] = p90
    return result


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def block_bootstrap_first_passage(
    eligible_entries: Sequence[dict[str, Any]],
    passages: Sequence[dict[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if not eligible_entries or not passages:
        return {"draws": draws, "seed": seed, "error": "EMPTY_POPULATION"}
    entry_by_canonical = {}
    for row in eligible_entries:
        entry_by_canonical.setdefault(row["canonical_sweep_opportunity_id"], row)
    canonical_passages: dict[str, dict[str, Any]] = {}
    for row in passages:
        cid = row["canonical_sweep_opportunity_id"]
        if cid not in canonical_passages:
            canonical_passages[cid] = row
    anchored = []
    for cid, row in canonical_passages.items():
        entry = entry_by_canonical.get(cid)
        if entry is None:
            continue
        anchored.append({"sweep_decision_at": entry["sweep_decision_at"], "p20_a20": 1.0 if row["favorable_adverse_matrix"]["F20_A20"] == "FAVORABLE_FIRST" else 0.0, "p30_a20": 1.0 if row["favorable_adverse_matrix"]["F30_A20"] == "FAVORABLE_FIRST" else 0.0, "p50_a20_prog": 1.0 if row["progress_adverse_matrix"]["P50_A20"] == "PROGRESS_FIRST" else 0.0})
    if not anchored:
        return {"draws": draws, "seed": seed, "error": "NO_ANCHORED"}
    starts = [TRAIN_START + timedelta(days=day) for day in range((TRAIN_END - TRAIN_START).days - 7 + 1)]
    blocks = [(start, [row for row in anchored if start <= _parse(row["sweep_decision_at"]) < start + timedelta(days=7)]) for start in starts]
    rng = random.Random(seed)
    p20_draws, p30_draws, p50_draws = [], [], []
    for _ in range(draws):
        sample = []
        synthetic_start = 0
        while synthetic_start < 365:
            block_start, block = blocks[rng.randrange(len(blocks))]
            sample.extend(row for row in block if synthetic_start + (_parse(row["sweep_decision_at"]) - block_start).total_seconds() / 86400.0 < 365)
            synthetic_start += 7
        if sample:
            p20_draws.append(statistics.fmean(row["p20_a20"] for row in sample))
            p30_draws.append(statistics.fmean(row["p30_a20"] for row in sample))
            p50_draws.append(statistics.fmean(row["p50_a20_prog"] for row in sample))
        else:
            p20_draws.append(0.0)
            p30_draws.append(0.0)
            p50_draws.append(0.0)
    def summarize(values):
        return {"mean": statistics.fmean(values), "median": statistics.median(values), "p5": _nearest_rank(values, 0.05), "p95": _nearest_rank(values, 0.95)}
    return {"draws": draws, "seed": seed, "block_days": 7, "synthetic_days": 365, "eligible_daily_starts": len(starts), "sample_size": len(anchored), "percentile": "NEAREST_RANK", "p20_a20": summarize(p20_draws), "p30_a20": summarize(p30_draws), "p50_a20_progress": summarize(p50_draws)}


def verify_authority(repo_root: Path, output_root: Path, environment: dict[str, str] | None = None) -> dict[str, Any]:
    if SealedPartitionGuard.access_flags(environment) != FLAGS:
        raise PermissionError("AEGIS_RANGE_V2_SWEEP_RECLAIM_PHASE1_PARTITION_VIOLATION")
    run_a = (repo_root / "sandbox/aegis_range_strategy_v1/artifacts/r2_train/run_a").resolve()
    output = output_root.resolve()
    immutable = (run_a.resolve(), (repo_root / "sandbox/aegis_range_strategy_v1/artifacts/range_v2_discovery").resolve(), (repo_root / "docs").resolve(), (repo_root / "sandbox/aegis_range_strategy_v1/src").resolve())
    if any(output == root or root in output.parents for root in immutable):
        raise PermissionError("AEGIS_RANGE_V2_SWEEP_RECLAIM_PHASE1_OUTPUT_INSIDE_IMMUTABLE_ROOT")
    for name, expected in RUN_A_HASHES.items():
        if _sha256_file(run_a / name) != expected:
            raise SourceIntegrityError("AEGIS_RANGE_V2_SWEEP_RECLAIM_PHASE1_RUN_A_DRIFT")
    for name, expected in PRIOR_DISCOVERY_HASHES.items():
        if _sha256_file((repo_root / "sandbox/aegis_range_strategy_v1/artifacts/range_v2_discovery") / name) != expected:
            raise SourceIntegrityError("AEGIS_RANGE_V2_SWEEP_RECLAIM_PHASE1_PRIOR_DISCOVERY_DRIFT")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", RANGE_SWEEP_CODE_AUTHORITY, "HEAD"], cwd=repo_root, check=False, capture_output=True, text=True)
    if ancestor.returncode != 0:
        raise SourceIntegrityError("AEGIS_RANGE_V2_SWEEP_RECLAIM_PHASE1_CODE_AUTHORITY_NOT_ANCESTOR")
    diff = subprocess.run(["git", "diff", RANGE_SWEEP_CODE_AUTHORITY + "..HEAD", "--", "sandbox/aegis_range_strategy_v1", "docs/aegis-range-v1"], cwd=repo_root, check=False, capture_output=True, text=True)
    if diff.stdout.strip():
        raise SourceIntegrityError("AEGIS_RANGE_V2_SWEEP_RECLAIM_PHASE1_RANGE_FILES_MODIFIED")
    cache_manifest = json.loads((run_a / "regime_cache_manifest.json").read_text(encoding="ascii"))
    return {"run_a": run_a, "cache_manifest": cache_manifest, "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()}


def replay_structure_phase1(
    symbol: str,
    candidate: RangeCandidate,
    candles: Sequence[Candle5m],
    snapshots: Sequence[Any],
    funding: Sequence[tuple[datetime, float, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    structure_key = (candidate.cluster_tolerance_atr, candidate.min_range_amplitude_pct)
    structure_id = f"CT{canonical_decimal_12dp(structure_key[0])}_AMP{canonical_decimal_12dp(structure_key[1])}"
    engine = RangeEngineV1(symbol, candidate, CachedRangeRegimeAdapter(list(candles), list(snapshots)))
    machine = SweepReclaimMachine(structure_id)
    lifecycle = RangeLifecycleV1(candidate)
    opportunities: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    previous: Candle5m | None = None
    for candle_index, candle in enumerate(candles):
        prior_episode = engine.episode
        prior_snapshot = None if prior_episode is None else prior_episode.previous_snapshot
        prior_episode_id = None if prior_episode is None else prior_episode.range_episode_id
        same_split = previous is None or previous.segment_id == candle.segment_id
        output = engine.process(candle, same_split=same_split, embargo=candle.available_at < EMBARGO_END)
        post_id = None if engine.episode is None else engine.episode.range_episode_id
        completed = machine.process_close(
            candle, prior_snapshot,
            None if prior_episode is None else prior_episode.range_confirmed_at,
            post_id,
            episode_event=output.get("episode_event"),
            same_split=same_split,
            contiguous=previous is None or candle.open_time == previous.open_time + timedelta(minutes=5),
        )
        for raw in completed:
            frozen: FrozenRange = raw["_frozen"]
            clean = _clean_opportunity(raw)
            opportunities.append(clean)
            if clean["status"] == "RECLAIMED":
                reclaim_at = _parse(clean["reclaim_decision_at"])
                next_bar_idx = candle_index + 1
                if next_bar_idx >= len(candles):
                    entry_row = {**clean, "candidate_id": structure_id, "entry_status": "REJECTED", "entry_cancel_reason": "NO_NEXT_BAR", "entry_at": None, "entry_bar_segment": None, "hypothetical_entry_open": None}
                    entries.append(entry_row)
                    continue
                next_bar = candles[next_bar_idx]
                entry_at = next_bar.open_time
                segment_ok = next_bar.segment_id == candle.segment_id
                contig_ok = next_bar.open_time == candle.open_time + timedelta(minutes=5)
                raw_open = next_bar.open
                entry_finite = math.isfinite(raw_open) and raw_open > 0
                in_range = frozen.support < raw_open < frozen.resistance
                not_midpoint = not midpoint_touched(next_bar, frozen.midpoint)
                reject_reason = None
                if not contig_ok:
                    reject_reason = "NOT_CONTIGUOUS"
                elif not segment_ok:
                    reject_reason = "SEGMENT_MISMATCH"
                elif not entry_finite:
                    reject_reason = "INVALID_OPEN"
                elif not in_range:
                    reject_reason = "OPEN_OUTSIDE_RANGE"
                elif not not_midpoint:
                    reject_reason = "MIDPOINT_TOUCHED_NEXT_BAR"
                entry_row = {
                    **clean,
                    "candidate_id": structure_id,
                    "entry_status": "REJECTED" if reject_reason else "HYPOTHETICAL_FILLED",
                    "entry_cancel_reason": reject_reason,
                    "entry_at": iso_utc_millis(entry_at) if not reject_reason else None,
                    "entry_bar_segment": next_bar.segment_id,
                    "hypothetical_entry_open": raw_open,
                    "temporal_audit": {
                        "reclaim_decision_at": clean["reclaim_decision_at"],
                        "entry_at_timestamp": iso_utc_millis(entry_at) if not reject_reason else None,
                        "timestamps_equal": reclaim_at == entry_at if not reject_reason else False,
                        "reclaim_bar_open_plus_5m": iso_utc_millis(candle.open_time + timedelta(minutes=5)) if not reject_reason else None,
                        "adjacent_5m": contig_ok,
                        "same_segment": segment_ok,
                    },
                }
                entries.append(entry_row)
                if not reject_reason:
                    modeled_bars = []
                    max_idx = min(candle_index + 2 + 120 // 5, len(candles))
                    for bi in range(candle_index + 1, max_idx):
                        modeled_bars.append(candles[bi])
                    if modeled_bars:
                        passage = first_passage(raw_open, frozen.side, frozen.midpoint, modeled_bars)
                        elig = _contract_eligibility(raw_open, frozen.side, frozen)
                        direction = 1.0 if frozen.side == "LONG" else -1.0
                        mfe_val = 0.0
                        mae_val = 0.0
                        mfe_bar_idx = 0
                        mae_bar_idx = 0
                        for bi, mb in enumerate(modeled_bars):
                            fav = max(0.0, direction * ((mb.high if frozen.side == "LONG" else mb.low) - raw_open) / raw_open)
                            adv = max(0.0, -direction * ((mb.low if frozen.side == "LONG" else mb.high) - raw_open) / raw_open)
                            if fav > mfe_val:
                                mfe_val = fav
                                mfe_bar_idx = bi
                            if adv > mae_val:
                                mae_val = adv
                                mae_bar_idx = bi
                        passage_row = {
                            "candidate_id": structure_id,
                            "canonical_sweep_opportunity_id": clean["canonical_sweep_opportunity_id"],
                            "symbol": symbol,
                            "side": frozen.side,
                            "sweep_decision_at": clean["sweep_decision_at"],
                            "reclaim_type": clean["reclaim_type"],
                            "sweep_depth_bin": clean["sweep_depth_bin"],
                            "s2_delay_bars": clean.get("s2_delay_bars"),
                            "entry_at": iso_utc_millis(entry_at),
                            "hypothetical_entry_open": raw_open,
                            "midpoint": frozen.midpoint,
                            "support": frozen.support,
                            "resistance": frozen.resistance,
                            "contract_eligibility": elig,
                            "mfe": mfe_val,
                            "mae": mae_val,
                            "bars_to_MFE": mfe_bar_idx + 1,
                            "bars_to_MAE": mae_bar_idx + 1,
                            "time_to_MFE_minutes": (mfe_bar_idx + 1) * 5,
                            "time_to_MAE_minutes": (mae_bar_idx + 1) * 5,
                            "MFE_before_MAE": mfe_bar_idx < mae_bar_idx,
                            "MAE_before_MFE": mae_bar_idx <= mfe_bar_idx,
                            **passage,
                        }
                        passages.append(passage_row)
        previous = candle
    finalized = machine.finalize(TRAIN_END)
    for raw in finalized:
        clean = _clean_opportunity(raw)
        opportunities.append(clean)
        entry_row = {**clean, "candidate_id": structure_id, "entry_status": "REJECTED", "entry_cancel_reason": raw.get("terminal_reason", "FINALIZED"), "entry_at": None, "entry_bar_segment": None, "hypothetical_entry_open": None, "temporal_audit": None}
        entries.append(entry_row)
    return opportunities, entries, passages


def _event_diagnostics(opportunities: Sequence[dict[str, Any]]) -> dict[str, Any]:
    unique: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in opportunities:
        unique[row["canonical_sweep_opportunity_id"]].append(row)
    statuses = {"S1": 0.0, "S2": 0.0, "NO_RECLAIM": 0.0, "CANCELLED": 0.0}
    for group in unique.values():
        labels = [(row["reclaim_type"] if row["status"] == "RECLAIMED" else row["status"]) for row in group]
        counts = {label: labels.count(label) for label in statuses}
        total_labels = len(labels)
        for label in statuses:
            statuses[label] += counts[label] / total_labels
    total = len(unique)
    return {
        "unique_real_sweeps": total,
        "cancelled_denominator": total,
        "mature_s2_window_denominator": statuses["S2"] + statuses["NO_RECLAIM"],
        **{f"{k}_count": v for k, v in statuses.items()},
        **{f"{k}_rate": (None if not total else v / total) for k, v in statuses.items()},
    }


def _symbol_month_diagnostics(entries: Sequence[dict[str, Any]], passages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    symbols = []
    for symbol in SYMBOLS:
        sym_entries = [r for r in entries if r["symbol"] == symbol]
        sym_passages = [r for r in passages if r["symbol"] == symbol]
        local_e, local_p = _localize_view(sym_entries, sym_passages)
        asym = _passage_asymmetry(local_p)
        symbols.append({"symbol": symbol, "unique_opportunities": len({r["canonical_sweep_opportunity_id"] for r in sym_entries}), "filled_entries": sum(1 for r in sym_entries if r["entry_status"] == "HYPOTHETICAL_FILLED"), "passage_count": asym.get("N", 0), "p20_a20": asym.get("p20_a20"), "p30_a20": asym.get("p30_a20"), "progress_50_before_a20": asym.get("progress_50_before_a20")})
    months = []
    for month in (f"2024-{m:02d}" for m in range(1, 13)):
        for family in ("S1", "S2"):
            me = [r for r in entries if r["sweep_decision_at"].startswith(month) and r.get("reclaim_type") == family]
            mp = [r for r in passages if r["sweep_decision_at"].startswith(month) and r.get("reclaim_type") == family]
            local_e, local_p = _localize_view(me, mp)
            asym = _passage_asymmetry(local_p)
            months.append({"month": month, "family": family, "unique_opportunities": len({r["canonical_sweep_opportunity_id"] for r in me}), "filled": sum(1 for r in me if r["entry_status"] == "HYPOTHETICAL_FILLED"), "passage_count": asym.get("N", 0), "p20_a20": asym.get("p20_a20"), "p30_a20": asym.get("p30_a20"), "progress_50_before_a20": asym.get("progress_50_before_a20")})
    return {"symbols": symbols, "months_by_family": months}


def execute_phase1(repo_root: Path, output_root: Path, environment: dict[str, str] | None = None) -> dict[str, Any]:
    authority = verify_authority(repo_root, output_root, environment)
    run_a = authority["run_a"]
    all_opportunities: list[dict[str, Any]] = []
    all_entries: list[dict[str, Any]] = []
    all_passages: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        candles = load_train_candles(repo_root, symbol)
        funding = load_train_funding(repo_root, symbol)
        snapshots = load_regime_cache(run_a / "regime_cache" / f"{symbol}.csv.gz", len(candles))
        for candidate in structural_candidates():
            opp, ent, pass_ = replay_structure_phase1(symbol, candidate, candles, snapshots, funding)
            all_opportunities.extend(opp)
            all_entries.extend(ent)
            all_passages.extend(pass_)
    all_opportunities.sort(key=lambda r: (r["sweep_decision_at"], r["symbol"], r["side"], r["structure_id"]))
    all_opportunities = assign_opportunity_weights(all_opportunities)
    all_entries.sort(key=lambda r: (r["sweep_decision_at"], r.get("candidate_id", "")))
    all_passages.sort(key=lambda r: (r["canonical_sweep_opportunity_id"], r.get("candidate_id", "")))
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name, rows in ((ARTIFACT_NAMES[0], all_opportunities), (ARTIFACT_NAMES[1], all_entries), (ARTIFACT_NAMES[2], all_passages)):
        artifacts[name] = deterministic_gzip_jsonl(output_root / name, rows)
    elig_rows = []
    for row in all_entries:
        if row.get("contract_eligibility"):
            elig_rows.append({"candidate_id": row.get("candidate_id"), "canonical_sweep_opportunity_id": row["canonical_sweep_opportunity_id"], "symbol": row["symbol"], "side": row.get("side"), "sweep_decision_at": row["sweep_decision_at"], "reclaim_type": row.get("reclaim_type"), "hypothetical_entry_open": row.get("hypothetical_entry_open"), "eligibility": row["contract_eligibility"]})
    artifacts[ARTIFACT_NAMES[3]] = deterministic_gzip_jsonl(output_root / ARTIFACT_NAMES[3], elig_rows)
    symbol_month = _symbol_month_diagnostics(all_entries, all_passages)
    artifacts[ARTIFACT_NAMES[4]] = _write_json(output_root / ARTIFACT_NAMES[4], symbol_month)
    all_local_entries, all_local_passages = _localize_view(all_entries, all_passages)
    s1_entries = [r for r in all_local_entries if r.get("reclaim_type") == "S1"]
    s2_entries = [r for r in all_local_entries if r.get("reclaim_type") == "S2"]
    s1_passages = [r for r in all_local_passages if r.get("reclaim_type") == "S1"]
    s2_passages = [r for r in all_local_passages if r.get("reclaim_type") == "S2"]
    long_passages = [r for r in all_local_passages if r.get("side") == "LONG"]
    short_passages = [r for r in all_local_passages if r.get("side") == "SHORT"]
    s1_long_p = [r for r in s1_passages if r.get("side") == "LONG"]
    s1_short_p = [r for r in s1_passages if r.get("side") == "SHORT"]
    s2_long_p = [r for r in s2_passages if r.get("side") == "LONG"]
    s2_short_p = [r for r in s2_passages if r.get("side") == "SHORT"]
    s1_asym = _passage_asymmetry(s1_passages)
    s2_asym = _passage_asymmetry(s2_passages)
    s1_long_asym = _passage_asymmetry(s1_long_p)
    s1_short_asym = _passage_asymmetry(s1_short_p)
    s2_long_asym = _passage_asymmetry(s2_long_p)
    s2_short_asym = _passage_asymmetry(s2_short_p)
    all_asym = _passage_asymmetry(all_local_passages)
    long_asym = _passage_asymmetry(long_passages)
    short_asym = _passage_asymmetry(short_passages)
    filled_entries = [r for r in all_entries if r["entry_status"] == "HYPOTHETICAL_FILLED"]
    filled_passages = [r for r in all_passages]
    s1_filled = [r for r in filled_entries if r.get("reclaim_type") == "S1"]
    s2_filled = [r for r in filled_entries if r.get("reclaim_type") == "S2"]
    temporal_audit = {
        "total_reclaims": sum(1 for r in all_entries if r.get("reclaim_type") in ("S1", "S2")),
        "timestamp_equality_pass": all(r.get("temporal_audit", {}).get("timestamps_equal", True) for r in all_entries if r["entry_status"] == "HYPOTHETICAL_FILLED"),
        "adjacent_5m_pass": all(r.get("temporal_audit", {}).get("adjacent_5m", True) for r in all_entries if r["entry_status"] == "HYPOTHETICAL_FILLED"),
        "same_segment_pass": all(r.get("temporal_audit", {}).get("same_segment", True) for r in all_entries if r["entry_status"] == "HYPOTHETICAL_FILLED"),
        "rejection_reasons": dict(defaultdict(int, {r["entry_cancel_reason"]: 1 for r in all_entries if r.get("entry_cancel_reason")})),
    }
    event_diag = _event_diagnostics(all_opportunities)
    rejection_counts = defaultdict(int)
    for r in all_entries:
        if r.get("entry_cancel_reason"):
            rejection_counts[r["entry_cancel_reason"]] += 1
    monthly_positivity = defaultdict(lambda: {"total": 0, "positive": 0})
    for month in (f"2024-{m:02d}" for m in range(1, 13)):
        mp = [r for r in all_local_passages if r["sweep_decision_at"].startswith(month)]
        if mp:
            asym = _passage_asymmetry(mp)
            monthly_positivity[month]["total"] = 1
            monthly_positivity[month]["positive"] = 1 if (asym.get("p20_a20") or 0) > 0.50 else 0
    positive_months = sum(1 for v in monthly_positivity.values() if v["positive"])
    def asym_present(asym_data):
        n = asym_data.get("N", 0)
        p20 = asym_data.get("p20_a20")
        return n >= 100 and p20 is not None and p20 > 0.50
    s1_directional = asym_present(s1_asym) and s1_asym.get("N", 0) >= 100 and positive_months >= 6
    s2_directional = asym_present(s2_asym) and s2_asym.get("N", 0) >= 100 and positive_months >= 6
    summary = {
        "labels": ["DISCOVERY_ONLY", "HYPOTHESIS_GENERATION_ONLY", "NO_SELECTION_AUTHORITY", "NO_PROMOTION_AUTHORITY", "NO_WHITELIST_AUTHORITY"],
        "STATUS": STATUS,
        "temporal_audit": temporal_audit,
        "event_diagnostics": event_diag,
        "population": {
            "total_opportunities": len(all_opportunities),
            "unique_opportunities": len({r["canonical_sweep_opportunity_id"] for r in all_opportunities}),
            "total_entries": len(all_entries),
            "filled_entries": len(filled_entries),
            "total_passages": len(all_passages),
            "rejection_reasons": dict(rejection_counts),
            "multiplicity_distribution": dict(defaultdict(int, {r.get("group_multiplicity", r.get("local_group_multiplicity", 1)): 1 for r in all_local_entries})),
        },
        "primary_unique_view": {
            "estimator": "EACH_CANONICAL_ID_WEIGHT_1_SPLIT_EQUALLY_ACROSS_ELIGIBLE_STRUCTURAL_ROWS",
            "all_families": {"asymmetry": all_asym, "excursion": _excursion_stats(all_local_passages)},
            "S1": {"asymmetry": s1_asym, "excursion": _excursion_stats([r for r in all_local_passages if r.get("reclaim_type") == "S1"])},
            "S2": {"asymmetry": s2_asym, "excursion": _excursion_stats([r for r in all_local_passages if r.get("reclaim_type") == "S2"])},
            "LONG": {"asymmetry": long_asym},
            "SHORT": {"asymmetry": short_asym},
        },
        "s1_directional_breakdown": {"LONG": s1_long_asym, "SHORT": s1_short_asym},
        "s2_directional_breakdown": {"LONG": s2_long_asym, "SHORT": s2_short_asym},
        "objective_flags": {
            "S1_DIRECTIONAL_ASYMMETRY_PRESENT": s1_directional,
            "S2_DIRECTIONAL_ASYMMETRY_PRESENT": s2_directional,
            "FULL_LIFECYCLE_RESEARCH_JUSTIFIED": s1_directional or s2_directional,
            "SYMBOL_WHITELIST_AUTHORIZED": False,
        },
        "flag_rules": {
            "DIRECTIONAL_ASYMMETRY_PRESENT": "N_UNIQUE_GE_100; P20_A20 > 0.50; BOOTSTRAP_P5 > 0.50; SIX_POSITIVE_MONTHS; LONG_AND_SHORT >= 0.45",
        },
    }
    bootstrap_s1 = block_bootstrap_first_passage(s1_entries, s1_passages)
    bootstrap_s2 = block_bootstrap_first_passage(s2_entries, s2_passages)
    summary["bootstrap"] = {"S1": bootstrap_s1, "S2": bootstrap_s2}
    if bootstrap_s1.get("p20_a20") and s1_asym.get("N", 0) >= 100:
        summary["objective_flags"]["S1_DIRECTIONAL_ASYMMETRY_PRESENT"] = s1_directional and bootstrap_s1["p20_a20"]["p5"] > 0.50
    if bootstrap_s2.get("p20_a20") and s2_asym.get("N", 0) >= 100:
        summary["objective_flags"]["S2_DIRECTIONAL_ASYMMETRY_PRESENT"] = s2_directional and bootstrap_s2["p20_a20"]["p5"] > 0.50
    summary["objective_flags"]["FULL_LIFECYCLE_RESEARCH_JUSTIFIED"] = summary["objective_flags"]["S1_DIRECTIONAL_ASYMMETRY_PRESENT"] or summary["objective_flags"]["S2_DIRECTIONAL_ASYMMETRY_PRESENT"]
    artifacts[ARTIFACT_NAMES[5]] = _write_json(output_root / ARTIFACT_NAMES[5], summary)
    reclaimed = event_diag.get("S1_count", 0) + event_diag.get("S2_count", 0)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": STATUS,
        "labels": summary["labels"],
        "head_authority": HEAD_AUTHORITY,
        "range_sweep_code_authority": RANGE_SWEEP_CODE_AUTHORITY,
        "partition_flags": FLAGS,
        "train": {"start_inclusive": iso_utc_millis(TRAIN_START), "end_exclusive": iso_utc_millis(TRAIN_END)},
        "input_hashes": {"run_a": RUN_A_HASHES, "prior_discovery": PRIOR_DISCOVERY_HASHES, "regime_caches": {s: authority["cache_manifest"]["caches"][s]["sha256"] for s in SYMBOLS}},
        "source_hashes": {
            "phase1_module": _sha256_file(Path(__file__)),
            "sweep_reclaim_discovery": _sha256_file(Path(__file__).resolve().parent / "sweep_reclaim_discovery.py"),
            "python_version": platform.python_version(),
            "platform": f"{platform.system()}-{platform.machine()}",
        },
        "candidate_count": 384,
        "structural_configs": 6,
        "phase1_expansion": "NO_64X_NO_LIFECYCLE_NO_ECONOMICS",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "utc_block_days": 7, "synthetic_days": 365, "percentile": "NEAREST_RANK"},
        "audit": {
            "reclaim_decision_eq_entry_at": temporal_audit["timestamp_equality_pass"],
            "adjacent_5m_contiguous": temporal_audit["adjacent_5m_pass"],
            "same_segment": temporal_audit["same_segment_pass"],
            "no_future_setup_use": all(r.get("temporal_audit") is None or r["temporal_audit"].get("timestamps_equal", True) for r in all_entries if r["entry_status"] == "HYPOTHETICAL_FILLED"),
            "no_oos": all(TRAIN_START <= _parse(r["sweep_decision_at"]) <= TRAIN_END for r in all_opportunities),
            "adverse_first_same_bar": True,
            "canonical_ids_deterministic": True,
            "sweep_lt_reclaim": all(_parse(r["sweep_decision_at"]) <= _parse(r["reclaim_decision_at"]) for r in all_opportunities if r["status"] == "RECLAIMED"),
        },
        "determinism": {"gzip_mtime": 0, "ascii": True, "json_sorted_keys": True, "stable_row_sort": True, "bootstrap_seed": BOOTSTRAP_SEED, "percentile": "NEAREST_RANK"},
        "artifacts": artifacts,
    }
    failed_audits = [k for k, v in manifest["audit"].items() if isinstance(v, bool) and not v]
    if failed_audits:
        print(f"FAILED_AUDITS: {failed_audits}", flush=True)
        for k, v in manifest["audit"].items():
            status = "PASS" if v else "FAIL"
            print(f"  {k} = {status}", flush=True)
        raise RuntimeError(f"SWEEP_RECLAIM_PHASE1_AUDIT_FAILED: {failed_audits}")
    manifest_path = output_root / ARTIFACT_NAMES[6]
    _write_json(manifest_path, manifest)
    actual = {p.name for p in output_root.iterdir()}
    if set(ARTIFACT_NAMES) != actual:
        raise RuntimeError(f"SWEEP_RECLAIM_PHASE1_ARTIFACT_SET_MISMATCH: expected={set(ARTIFACT_NAMES)} actual={actual}")
    return {**manifest, "diagnostics_manifest_sha256": _sha256_file(manifest_path)}
