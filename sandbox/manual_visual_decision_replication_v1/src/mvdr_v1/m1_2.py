from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .m1 import OTHER_STRATEGY_RECORDS, PROGRAM, SESSION_BOUNDS, Bar, _aggregate, _dt, _iso, _sha, _stable_id, build_manual_manifest
from .m1_1 import M1_AUTHORITY, M1_MARKET_HASHES, _clip, _mean, _median, _read_bars, causal_levels


PHASE = "MVDR_V1_M1_2_COMPACT_SR_CORRIDOR_GRAMMAR"
STATUS = "MVDR_V1_M1_2_COMPACT_SR_CORRIDOR_GRAMMAR_READY_FOR_REVIEW"
M1_1_AUTHORITY = "9cc915424b9be5985a4a84b851c54c6d670f550d"
FAMILY_ORDER = {"swing": 0, "cluster": 1, "mtf_extrema": 2}
ACTION_SIDE = {
    "SHORT_TO_SUPPORT": "SHORT", "LONG_FROM_SUPPORT": "LONG",
    "LONG_TO_RESISTANCE": "LONG", "SHORT_FROM_RESISTANCE": "SHORT",
    "NO_TRADE": "NO_TRADE",
}


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def causal_zone_width_bps(completed: list[Bar]) -> float:
    recent = completed[-60:]
    one_minute = [(bar.high - bar.low) / bar.close * 10_000 for bar in recent]
    three_minute = [(bar.high - bar.low) / bar.close * 10_000 / 3.0 for bar in _aggregate(recent, 3)[-20:]]
    return max(_median(one_minute) or 0.0, _median(three_minute) or 0.0, 1.0)


def make_zone(center: float, width_bps: float) -> dict[str, float]:
    half = center * width_bps / 20_000
    return {"lower_bound": center - half, "center": center, "upper_bound": center + half, "zone_width_bps": width_bps}


def level_respect_score(completed: list[Bar], zone: dict[str, float], kind: str) -> dict[str, Any]:
    history = completed[-121:-1] if len(completed) >= 121 else completed[:-1]
    recent = history[-60:]
    lower, upper, center = zone["lower_bound"], zone["upper_bound"], zone["center"]
    width = max(upper - lower, 1e-12)
    contacts = [index for index, bar in enumerate(history) if bar.low <= upper and bar.high >= lower]
    valid_closes = _mean(bar.close >= lower if kind == "support" else bar.close <= upper for bar in recent)
    penetrations = []
    returns = []
    away = []
    breaks = 0
    for index, bar in enumerate(history):
        penetrated = bar.low < lower if kind == "support" else bar.high > upper
        if penetrated:
            penetrations.append(index)
            future_closed = history[index:min(len(history), index + 3)]
            returned = any(candidate.close >= lower if kind == "support" else candidate.close <= upper for candidate in future_closed)
            returns.append(float(returned))
        broken = bar.close < lower - width if kind == "support" else bar.close > upper + width
        breaks += int(broken)
    for index in contacts:
        subsequent = history[index + 1:min(len(history), index + 4)]
        if subsequent:
            movement = max(bar.high - upper for bar in subsequent) if kind == "support" else max(lower - bar.low for bar in subsequent)
            away.append(_clip(movement / width))
    components = {
        "contact_score": _clip(len(contacts) / 3.0),
        "valid_close_score": valid_closes,
        "penetration_return_score": _mean(returns) if returns else 0.5,
        "post_touch_away_score": _mean(away),
        "age_score": _clip((len(history) - contacts[0]) / 120.0) if contacts else 0.0,
        "break_integrity_score": _clip(1.0 - breaks / max(1, len(recent)) * 5.0),
    }
    return {
        "score": _mean(components.values()), "components": components,
        "touch_count": len(contacts), "penetration_count": len(penetrations),
        "break_count": breaks, "time_since_last_touch_minutes": len(history) - 1 - contacts[-1] if contacts else None,
    }


def nearest_relevant_zones(completed: list[Bar], price: float) -> dict[str, Any]:
    levels = causal_levels(completed, price)
    width_bps = causal_zone_width_bps(completed)
    candidates = []
    for family, details in levels["families"].items():
        for kind in ("support", "resistance"):
            zone = make_zone(details[kind], width_bps)
            respect = level_respect_score(completed, zone, kind)
            candidates.append({"family": family, "kind": kind, "zone": zone, "respect": respect})
    supports = [row for row in candidates if row["zone"]["center"] <= price or row["zone"]["lower_bound"] <= price <= row["zone"]["upper_bound"]]
    resistances = [row for row in candidates if row["zone"]["center"] >= price or row["zone"]["lower_bound"] <= price <= row["zone"]["upper_bound"]]
    supports.sort(key=lambda row: (-row["respect"]["score"], abs(price - row["zone"]["center"]), FAMILY_ORDER[row["family"]]))
    resistances.sort(key=lambda row: (-row["respect"]["score"], abs(row["zone"]["center"] - price), FAMILY_ORDER[row["family"]]))
    valid_pairs = [(support, resistance) for support in supports for resistance in resistances if support["zone"]["center"] < resistance["zone"]["center"]]
    if not valid_pairs:
        raise ValueError("NO_VALID_ZONE_PAIR")
    support, resistance = min(
        valid_pairs,
        key=lambda pair: (
            -(pair[0]["respect"]["score"] + pair[1]["respect"]["score"]),
            abs(price - pair[0]["zone"]["center"]) + abs(pair[1]["zone"]["center"] - price),
            FAMILY_ORDER[pair[0]["family"]], FAMILY_ORDER[pair[1]["family"]],
        ),
    )
    return {
        "support": support, "resistance": resistance, "all_candidates": candidates,
        "selection_rule": "maximum combined respect; then minimum combined distance; then frozen family order",
    }


def corridor_clarity(completed: list[Bar], zones: dict[str, Any], price: float) -> dict[str, Any]:
    support = zones["support"]
    resistance = zones["resistance"]
    support_zone, resistance_zone = support["zone"], resistance["zone"]
    width_bps = (resistance_zone["center"] - support_zone["center"]) / price * 10_000
    recent = completed[-60:]
    contained = _mean(support_zone["lower_bound"] <= bar.close <= resistance_zone["upper_bound"] for bar in recent)
    breaks = sum(bar.close < support_zone["lower_bound"] or bar.close > resistance_zone["upper_bound"] for bar in recent)
    contradictory = 0
    for candidate in zones["all_candidates"]:
        chosen = support if candidate["kind"] == "support" else resistance
        if candidate["family"] == chosen["family"]:
            continue
        center_gap_bps = abs(candidate["zone"]["center"] - chosen["zone"]["center"]) / price * 10_000
        if center_gap_bps <= candidate["zone"]["zone_width_bps"] and abs(candidate["respect"]["score"] - chosen["respect"]["score"]) <= 0.10:
            contradictory += 1
    components = {
        "support_respect": support["respect"]["score"],
        "resistance_respect": resistance["respect"]["score"],
        "separation": _clip(width_bps / 100.0),
        "containment": contained,
        "non_ambiguity": _clip(1.0 - contradictory / 2.0),
        "break_integrity": _clip(1.0 - breaks / 60.0 * 5.0),
    }
    score = _mean(components.values())
    clear = support["respect"]["score"] >= 0.50 and resistance["respect"]["score"] >= 0.50 and width_bps >= 25.0 and contradictory == 0 and score >= 0.60
    return {
        "score": score, "components": components, "clear": clear,
        "corridor_width_bps": width_bps, "contradictory_level_count": contradictory,
        "current_position_0_1_raw": (price - support_zone["center"]) / (resistance_zone["center"] - support_zone["center"]),
        "room_to_support_bps": (price - support_zone["center"]) / price * 10_000,
        "room_to_resistance_bps": (resistance_zone["center"] - price) / price * 10_000,
    }


def movement_score(completed: list[Bar], level: float, toward_support: bool, minutes: int) -> dict[str, float]:
    seq = completed[-(minutes + 1):]
    closes = [bar.close for bar in seq]
    direction = -1.0 if toward_support else 1.0
    changes = [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    displacement = direction * (closes[-1] - closes[0]) / closes[0] * 10_000
    traveled = sum(abs(value) for value in changes) / closes[0] * 10_000
    past_distance = abs(closes[0] - level) / closes[0] * 10_000
    current_distance = abs(closes[-1] - level) / closes[-1] * 10_000
    components = {
        "directional_displacement": _clip(displacement / max(traveled, 1.0)),
        "distance_reduction": _clip((past_distance - current_distance) / max(past_distance, 10.0)),
        "advancing_closes": _mean(direction * value > 0 for value in changes),
        "path_efficiency": _clip(displacement / max(traveled, 1.0)),
    }
    return {"score": _mean(components.values()), "components": components, "signed_displacement_bps": direction * displacement, "traveled_distance_bps": traveled}


def reaction_score(completed: list[Bar], zone: dict[str, float], kind: str) -> dict[str, Any]:
    recent = completed[-4:]
    lower, upper = zone["lower_bound"], zone["upper_bound"]
    width = max(upper - lower, 1e-12)
    last = recent[-1]
    touch_indices = [index for index, bar in enumerate(recent) if bar.low <= upper and bar.high >= lower]
    in_zone = last.low <= upper and last.high >= lower
    penetration = max(0.0, lower - min(bar.low for bar in recent)) if kind == "support" else max(0.0, max(bar.high for bar in recent) - upper)
    candle_range = max(last.high - last.low, 1e-12)
    wick = (min(last.open, last.close) - last.low) / candle_range if kind == "support" else (last.high - max(last.open, last.close)) / candle_range
    close_away = _clip((last.close - upper) / width) if kind == "support" else _clip((lower - last.close) / width)
    failed = _mean(recent[index].low >= recent[index - 1].low if kind == "support" else recent[index].high <= recent[index - 1].high for index in range(1, len(recent)))
    followthrough = 0.5
    if touch_indices and touch_indices[-1] < len(recent) - 1:
        touch = recent[touch_indices[-1]]
        later = recent[touch_indices[-1] + 1:]
        followthrough = float(any(bar.close > touch.close if kind == "support" else bar.close < touch.close for bar in later))
    components = {
        "limited_penetration": _clip(1.0 - penetration / width), "wick_rejection": wick,
        "close_away": close_away, "failed_continuation": failed, "completed_followthrough": followthrough,
    }
    return {"score": _mean(components.values()), "components": components, "in_zone": in_zone, "penetration_bps": penetration / last.close * 10_000}


def btc_context(completed: list[Bar], action: str) -> dict[str, Any]:
    side = ACTION_SIDE[action]
    direction = 1.0 if side == "LONG" else -1.0 if side == "SHORT" else 0.0
    movements, thresholds = {}, {}
    for minutes in (3, 5, 15):
        current = (completed[-1].close - completed[-(minutes + 1)].close) / completed[-(minutes + 1)].close * 10_000
        historical = []
        history = completed[-181:-1]
        for index in range(minutes, len(history)):
            historical.append(abs((history[index].close - history[index - minutes].close) / history[index - minutes].close * 10_000))
        movements[str(minutes)] = current
        thresholds[str(minutes)] = _nearest_rank(historical, 0.95)
    signed = [direction * movements[str(minutes)] for minutes in (3, 5, 15)]
    strongly_opposing = direction != 0 and sum(value <= -thresholds[str(minutes)] for value, minutes in zip(signed, (3, 5, 15))) >= 2
    aligned = direction != 0 and sum(value > 0 for value in signed) >= 2
    state = "STRONGLY_OPPOSING" if strongly_opposing else "ALIGNED" if aligned else "NEUTRAL"
    return {"state": state, "movements_bps": movements, "causal_p95_thresholds_bps": thresholds, "safety_score": 0.0 if strongly_opposing else 1.0 if aligned else 0.75}


def volatility_shock(completed: list[Bar]) -> dict[str, Any]:
    prior = [(bar.high - bar.low) / bar.close * 10_000 for bar in completed[-121:-1]]
    current = (completed[-1].high - completed[-1].low) / completed[-1].close * 10_000
    threshold = _nearest_rank(prior, 0.95)
    return {"current_range_bps": current, "causal_p95_range_bps": threshold, "shock": current > threshold}


def decide_compact_action(
    corridor_clear: bool,
    support_in_zone: bool,
    resistance_in_zone: bool,
    support_reaction: float,
    resistance_reaction: float,
    travel_support: float,
    travel_resistance: float,
) -> tuple[str, float]:
    if not corridor_clear:
        return "NO_TRADE", max(support_reaction, resistance_reaction, travel_support, travel_resistance)
    reactions = []
    if support_in_zone and support_reaction >= 0.60:
        reactions.append((support_reaction, "LONG_FROM_SUPPORT"))
    if resistance_in_zone and resistance_reaction >= 0.60:
        reactions.append((resistance_reaction, "SHORT_FROM_RESISTANCE"))
    if reactions:
        return max(reactions, key=lambda item: (item[0], item[1]))[1], max(reactions)[0]
    if not support_in_zone and not resistance_in_zone:
        if travel_support >= 0.60 and travel_support >= travel_resistance + 0.05:
            return "SHORT_TO_SUPPORT", travel_support
        if travel_resistance >= 0.60 and travel_resistance >= travel_support + 0.05:
            return "LONG_TO_RESISTANCE", travel_resistance
    return "NO_TRADE", max(support_reaction, resistance_reaction, travel_support, travel_resistance)


def compact_corridor_frame(decision_at: datetime, sui: list[Bar], btc: list[Bar]) -> dict[str, Any]:
    sui_done = [bar for bar in sui if bar.open_at + timedelta(minutes=1) <= decision_at]
    btc_done = [bar for bar in btc if bar.open_at + timedelta(minutes=1) <= decision_at]
    if len(sui_done) < 181 or len(btc_done) < 181:
        raise ValueError("INSUFFICIENT_CAUSAL_CONTEXT")
    if not all(math.isfinite(value) for bar in sui_done[-181:] + btc_done[-181:] for value in (bar.open, bar.high, bar.low, bar.close, bar.volume)):
        raise ValueError("NON_FINITE_DATA")
    price = sui_done[-1].close
    zones = nearest_relevant_zones(sui_done, price)
    corridor = corridor_clarity(sui_done, zones, price)
    support_zone = zones["support"]["zone"]
    resistance_zone = zones["resistance"]["zone"]
    movement_support = {str(minutes): movement_score(sui_done, support_zone["center"], True, minutes) for minutes in (3, 6, 12)}
    movement_resistance = {str(minutes): movement_score(sui_done, resistance_zone["center"], False, minutes) for minutes in (3, 6, 12)}
    movement_support_score = _mean(value["score"] for value in movement_support.values())
    movement_resistance_score = _mean(value["score"] for value in movement_resistance.values())
    support_reaction = reaction_score(sui_done, support_zone, "support")
    resistance_reaction = reaction_score(sui_done, resistance_zone, "resistance")
    support_room = corridor["room_to_support_bps"]
    resistance_room = corridor["room_to_resistance_bps"]
    down_not_reversed = _clip(-sum(bar.close - bar.open for bar in sui_done[-3:]) / max(sum(abs(bar.close - bar.open) for bar in sui_done[-3:]), 1e-12))
    up_not_reversed = _clip(sum(bar.close - bar.open for bar in sui_done[-3:]) / max(sum(abs(bar.close - bar.open) for bar in sui_done[-3:]), 1e-12))
    travel_support = _mean((zones["support"]["respect"]["score"], corridor["score"], _clip(support_room / 100.0), movement_support_score, down_not_reversed))
    travel_resistance = _mean((zones["resistance"]["respect"]["score"], corridor["score"], _clip(resistance_room / 100.0), movement_resistance_score, up_not_reversed))

    potential_action, confirmation = decide_compact_action(
        corridor["clear"], support_reaction["in_zone"], resistance_reaction["in_zone"],
        support_reaction["score"], resistance_reaction["score"], travel_support, travel_resistance,
    )
    relevant_respect = max(zones["support"]["respect"]["score"], zones["resistance"]["respect"]["score"])
    if potential_action in ("LONG_FROM_SUPPORT", "SHORT_TO_SUPPORT"):
        relevant_respect = zones["support"]["respect"]["score"]
    elif potential_action in ("SHORT_FROM_RESISTANCE", "LONG_TO_RESISTANCE"):
        relevant_respect = zones["resistance"]["respect"]["score"]

    btc_result = btc_context(btc_done, potential_action)
    volatility = volatility_shock(sui_done)
    vetoes = []
    if not corridor["clear"]:
        vetoes.append("NO_CLEAR_CORRIDOR")
    if potential_action == "NO_TRADE":
        vetoes.append("NO_CONFIRMED_TRAVEL_OR_REACTION")
    if potential_action == "SHORT_TO_SUPPORT" and support_room < 25:
        vetoes.append("INSUFFICIENT_DESTINATION_ROOM")
    if potential_action == "LONG_TO_RESISTANCE" and resistance_room < 25:
        vetoes.append("INSUFFICIENT_DESTINATION_ROOM")
    if potential_action in ("SHORT_TO_SUPPORT", "LONG_TO_RESISTANCE") and confirmation < 0.60:
        vetoes.append("LOW_MOVEMENT_CONFIDENCE")
    if potential_action in ("LONG_FROM_SUPPORT", "SHORT_FROM_RESISTANCE") and confirmation < 0.60:
        vetoes.append("REACTION_NOT_CONFIRMED")
    if btc_result["state"] == "STRONGLY_OPPOSING":
        vetoes.append("BTC_STRONGLY_OPPOSING")
    if volatility["shock"]:
        vetoes.append("VOLATILITY_SHOCK")
    if corridor["contradictory_level_count"]:
        vetoes.append("DATA_OR_LEVEL_AMBIGUITY")
    trade_safe = not vetoes
    final_action = potential_action if trade_safe else "NO_TRADE"
    setup_clarity = _mean((corridor["score"], relevant_respect, confirmation, btc_result["safety_score"]))
    return {
        "frame_id": _stable_id(PHASE, _iso(decision_at)), "decision_at_utc": _iso(decision_at),
        "latest_completed_1m_open_at": _iso(sui_done[-1].open_at), "current_price": price,
        "support": zones["support"], "resistance": zones["resistance"],
        "zone_selection_rule": zones["selection_rule"], "corridor": corridor,
        "movement_to_support": movement_support, "movement_to_resistance": movement_resistance,
        "movement_to_support_score": movement_support_score, "movement_to_resistance_score": movement_resistance_score,
        "travel_confirmation_support": travel_support, "travel_confirmation_resistance": travel_resistance,
        "support_reaction": support_reaction, "resistance_reaction": resistance_reaction,
        "btc": btc_result, "volatility": volatility, "potential_action": potential_action,
        "trade_safe": trade_safe, "safety_veto_reason": vetoes, "final_action": final_action,
        "setup_clarity_score": setup_clarity,
    }


class ProspectiveCorridorShadowRecorder:
    """Append-only recorder. It receives causal frames and never sends orders."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, frame: dict[str, Any]) -> dict[str, Any]:
        timestamp = _dt(frame["decision_at_utc"])
        path = self.root / f"{timestamp.date().isoformat()}.jsonl"
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            if lines and _dt(json.loads(lines[-1])["timestamp"]) >= timestamp:
                raise RuntimeError("SHADOW_APPEND_ONLY_ORDER_VIOLATION")
        record = {
            "timestamp": frame["decision_at_utc"], "support_zone": frame["support"]["zone"],
            "resistance_zone": frame["resistance"]["zone"],
            "support_respect_score": frame["support"]["respect"]["score"],
            "resistance_respect_score": frame["resistance"]["respect"]["score"],
            "corridor_clarity": frame["corridor"]["score"],
            "price_position": frame["corridor"]["current_position_0_1_raw"],
            "travel_direction": frame["potential_action"],
            "travel_support_score": frame["travel_confirmation_support"],
            "travel_resistance_score": frame["travel_confirmation_resistance"],
            "support_reaction_score": frame["support_reaction"]["score"],
            "resistance_reaction_score": frame["resistance_reaction"]["score"],
            "btc_state": frame["btc"]["state"], "safety": frame["trade_safe"],
            "final_action": frame["final_action"], "setup_clarity_score": frame["setup_clarity_score"],
            "outcome_known": False, "orders_enabled": False,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.root / f"{timestamp.date().isoformat()}.sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
        return record


def _evaluate_anchors(frames: list[dict[str, Any]], manifest: list[dict[str, Any]], shift_minutes: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_minute = {frame["decision_at_utc"][:16]: frame for frame in frames}
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_session[frame["decision_at_utc"][:10]].append(frame)
    predictions = []
    for trade in manifest:
        shifted_at = _dt(trade["entry_at_utc"]) + timedelta(minutes=shift_minutes)
        session = trade["entry_at_utc"][:10]
        if shifted_at.date().isoformat() != session or _iso(shifted_at)[:16] not in by_minute:
            continue
        target = by_minute[_iso(shifted_at)[:16]]
        local = sorted(by_session[session], key=lambda row: (-row["setup_clarity_score"], row["frame_id"]))
        rank = next(index for index, row in enumerate(local, 1) if row["frame_id"] == target["frame_id"])
        emitted_side = ACTION_SIDE[target["final_action"]]
        capture = {}
        timing_errors = []
        for frame in by_session[session]:
            if ACTION_SIDE[frame["final_action"]] == trade["side"]:
                timing_errors.append(abs((_dt(frame["decision_at_utc"]) - shifted_at).total_seconds()) / 60)
        for tolerance in (1, 3, 6):
            capture[str(tolerance)] = any(error <= tolerance for error in timing_errors)
        expected_family = "SHORT_TO_SUPPORT" if trade["manual_trade_id"] == "MVDR-M1-09" else None
        predictions.append({
            "manual_trade_id": trade["manual_trade_id"], "manual_entry_at_utc": trade["entry_at_utc"],
            "evaluated_at_utc": _iso(shifted_at), "shift_minutes": shift_minutes, "manual_side": trade["side"],
            "potential_action": target["potential_action"], "final_action": target["final_action"],
            "correct_direction": emitted_side == trade["side"], "expected_action_family": expected_family,
            "correct_action_family": target["potential_action"] == expected_family if expected_family else None,
            "trade_safe": target["trade_safe"], "safety_veto_reason": target["safety_veto_reason"],
            "setup_clarity_score": target["setup_clarity_score"], "rank": rank,
            "candidate_count": len(local), "percentile_rank": 1.0 - (rank - 1) / max(1, len(local) - 1),
            "capture": capture, "nearest_timing_error_minutes": min(timing_errors) if timing_errors else None,
            "support_center": target["support"]["zone"]["center"], "resistance_center": target["resistance"]["zone"]["center"],
            "corridor_clarity": target["corridor"]["score"], "corridor_clear": target["corridor"]["clear"],
        })
    signals_by_day = {session: sum(frame["final_action"] != "NO_TRADE" for frame in local) for session, local in by_session.items()}
    n = len(predictions)
    metrics = {
        "N": n, "correct_direction_rate": sum(row["correct_direction"] for row in predictions) / n,
        "correct_action_family_available": sum(row["correct_action_family"] is not None for row in predictions),
        "correct_action_family_rate_available": _mean(row["correct_action_family"] for row in predictions if row["correct_action_family"] is not None),
        "capture_rate_1m": sum(row["capture"]["1"] for row in predictions) / n,
        "capture_rate_3m": sum(row["capture"]["3"] for row in predictions) / n,
        "capture_rate_6m": sum(row["capture"]["6"] for row in predictions) / n,
        "top_1pct": sum(row["percentile_rank"] >= 0.99 for row in predictions) / n,
        "top_5pct": sum(row["percentile_rank"] >= 0.95 for row in predictions) / n,
        "top_10pct": sum(row["percentile_rank"] >= 0.90 for row in predictions) / n,
        "median_percentile_rank": _median(row["percentile_rank"] for row in predictions),
        "signals_by_day": signals_by_day, "max_signals_per_day": max(signals_by_day.values()),
        "mean_signals_per_day": _mean(signals_by_day.values()),
    }
    return predictions, metrics


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as zipped:
        for row in rows:
            zipped.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())


def execute_m1_2(repo_root: Path, output_root: Path) -> dict[str, Any]:
    for authority in (M1_AUTHORITY, M1_1_AUTHORITY):
        subprocess.run(["git", "merge-base", "--is-ancestor", authority, "HEAD"], cwd=repo_root, check=True)
    output_root.mkdir(parents=True, exist_ok=True)
    m1_root = repo_root / "sandbox/manual_visual_decision_replication_v1/artifacts/m1"
    m1_artifact_manifest = json.loads((m1_root / "m1_manifest.json").read_text(encoding="utf-8"))
    for name, metadata in m1_artifact_manifest["artifacts"].items():
        if _sha(m1_root / name) != metadata["sha256"]:
            raise RuntimeError(f"M1_ARTIFACT_HASH_MISMATCH:{name}")
    manifest = json.loads((m1_root / "manual_entry_manifest.json").read_text(encoding="utf-8"))["trades"]
    if manifest != build_manual_manifest(repo_root / "backtest_results/real_trade_analysis.json"):
        raise RuntimeError("M1_MANUAL_MANIFEST_CHANGED")
    market = {symbol: _read_bars(m1_root / f"{symbol}_1m.jsonl.gz", expected) for symbol, expected in M1_MARKET_HASHES.items()}

    frames = []
    for start_text, end_text in SESSION_BOUNDS.values():
        cursor, end = _dt(start_text), _dt(end_text)
        while cursor < end:
            frames.append(compact_corridor_frame(cursor, market["SUIUSDT"], market["BTCUSDT"]))
            cursor += timedelta(minutes=1)
    _write_jsonl(output_root / "corridor_frames.jsonl.gz", frames)
    _write_jsonl(output_root / "level_zones.jsonl.gz", ({"frame_id": row["frame_id"], "decision_at_utc": row["decision_at_utc"], "support": row["support"], "resistance": row["resistance"], "selection_rule": row["zone_selection_rule"]} for row in frames))

    predictions, metrics = _evaluate_anchors(frames, manifest)
    shifted_plus, plus_metrics = _evaluate_anchors(frames, manifest, 60)
    shifted_minus, minus_metrics = _evaluate_anchors(frames, manifest, -60)
    _write_jsonl(output_root / "retrospective_predictions.jsonl.gz", predictions)
    _write_json(output_root / "time_shift_controls.json", {"plus_60m": {"metrics": plus_metrics, "predictions": shifted_plus}, "minus_60m": {"metrics": minus_metrics, "predictions": shifted_minus}})

    frame_by_minute = {row["decision_at_utc"][:16]: row for row in frames}
    diagnostics, dossiers = [], []
    prediction_by_trade = {row["manual_trade_id"]: row for row in predictions}
    for trade in manifest:
        frame = frame_by_minute[trade["entry_at_utc"][:16]]
        diagnostic = {
            "manual_trade_id": trade["manual_trade_id"], "entry_at_utc": trade["entry_at_utc"],
            "support_candidate": frame["support"], "resistance_candidate": frame["resistance"],
            "corridor": frame["corridor"], "current_price": frame["current_price"],
            "distance_to_support_bps": frame["corridor"]["room_to_support_bps"],
            "distance_to_resistance_bps": frame["corridor"]["room_to_resistance_bps"],
        }
        diagnostics.append(diagnostic)
        dossiers.append({
            "dossier": "COMPACT_CORRIDOR_VISUAL_DOSSIER", **diagnostic,
            "movement_to_support": frame["movement_to_support"], "movement_to_resistance": frame["movement_to_resistance"],
            "travel_confirmation_support": frame["travel_confirmation_support"],
            "travel_confirmation_resistance": frame["travel_confirmation_resistance"],
            "support_reaction": frame["support_reaction"], "resistance_reaction": frame["resistance_reaction"],
            "btc": frame["btc"], "volatility": frame["volatility"],
            "potential_action": frame["potential_action"], "final_action": frame["final_action"],
            "trade_safe": frame["trade_safe"], "safety_veto_reason": frame["safety_veto_reason"],
            "setup_clarity_score": frame["setup_clarity_score"], "ranking": prediction_by_trade[trade["manual_trade_id"]],
            "post_entry_information_used": False,
        })
    _write_jsonl(output_root / "level_respect_diagnostics.jsonl.gz", diagnostics)
    _write_jsonl(output_root / "anchor_dossiers.jsonl.gz", dossiers)

    baselines = {
        "M1_FULL": {"correct_direction_rate": 7 / 9, "capture_rate_6m": 7 / 9, "top_5pct": 3 / 9, "median_percentile_rank": 0.7363667940257034, "fold_emissions": 16143},
        "M1_SR_ONLY": {"correct_direction_rate": 4 / 9, "capture_rate_6m": 6 / 9, "top_5pct": 0.0, "median_percentile_rank": 0.7749218478638416, "fold_emissions": 12601},
        "M1_1_LEVEL_STATE": {"correct_direction_rate": 1 / 9, "capture_rate_6m": 1 / 9, "top_5pct": 2 / 9, "median_percentile_rank": 0.6344683808200139, "fold_emissions": 194},
        "M1_2_COMPACT_CORRIDOR": metrics,
        "comparability_note": "M1/M1.1 emission counts are summed folds; M1.2 reports unique deterministic session emissions.",
    }
    _write_json(output_root / "baseline_comparison.json", baselines)

    improves_m1_1 = metrics["correct_direction_rate"] > 1 / 9 and metrics["capture_rate_6m"] > 1 / 9 and metrics["median_percentile_rank"] > 0.6344683808200139
    fit = metrics["correct_direction_rate"] >= 7 / 9 and metrics["top_5pct"] >= 4 / 9 and metrics["capture_rate_6m"] >= 6 / 9 and metrics["max_signals_per_day"] <= 20 and improves_m1_1
    plus_loses = plus_metrics["correct_direction_rate"] < metrics["correct_direction_rate"] or plus_metrics["top_5pct"] < metrics["top_5pct"]
    minus_loses = minus_metrics["correct_direction_rate"] < metrics["correct_direction_rate"] or minus_metrics["top_5pct"] < metrics["top_5pct"]

    prospective_root = repo_root / "sandbox/manual_visual_decision_replication_v1/artifacts/m1_2_prospective_shadow"
    prospective_root.mkdir(parents=True, exist_ok=True)
    _write_json(prospective_root / "recorder_schema.json", {
        "mode": "READ_ONLY_SHADOW_ONLY", "append_only": True, "orders_enabled": False,
        "daily_records": "YYYY-MM-DD.jsonl", "daily_hash": "YYYY-MM-DD.sha256",
        "prospective_criterion_not_evaluated_until": {"manual_decisions": 30, "days": 5, "correct_direction": 0.70, "capture_6m": 0.60, "max_signals_day": 10},
    })

    summary = {
        "STATUS": STATUS, "program": PROGRAM, "phase": PHASE,
        "labels": ["POST_M1_1_HYPOTHESIS", "RETROSPECTIVE_SANITY_ONLY", "PROSPECTIVE_CONFIRMATION_REQUIRED", "NO_PROFITABILITY_AUTHORITY", "NO_AUTOMATION_AUTHORITY"],
        "interpretation": "COMPACT_CORRIDOR_GRAMMAR_RETROSPECTIVELY_PLAUSIBLE" if fit else "COMPACT_CORRIDOR_GRAMMAR_INSUFFICIENT",
        "flags": {"COMPACT_SR_GRAMMAR_IMPLEMENTED": True, "LEVEL_RESPECT_RECONSTRUCTION_COMPLETE": True, "COMPACT_GRAMMAR_RETROSPECTIVE_FIT": fit, "PROSPECTIVE_SHADOW_RECORDER_READY": True, "PROSPECTIVE_REPLICATION_SIGNAL": "NOT_EVALUATED", "FULL_AUTOMATION_RESEARCH_JUSTIFIED": False},
        "retrospective_metrics": metrics, "criterion_components": {"correct_direction_ge_7_of_9": metrics["correct_direction_rate"] >= 7 / 9, "top5_ge_4_of_9": metrics["top_5pct"] >= 4 / 9, "capture_6m_ge_6_of_9": metrics["capture_rate_6m"] >= 6 / 9, "signals_day_lte_20": metrics["max_signals_per_day"] <= 20, "clearly_improves_m1_1": improves_m1_1},
        "time_shift_control_loses_fit": {"plus_60m": plus_loses, "minus_60m": minus_loses},
        "manual_grammar_authority": {"trade_id": "MVDR-M1-09", "state": "TOWARD_SUPPORT", "used_for_threshold_fitting": False},
        "anti_leakage": {"m1_hashes_verified": True, "completed_1m_only": True, "outcomes_used": False, "ml_primary": False, "rules_changed_after_results": False},
        "production_modified": False, "orders_sent": False, "exits_studied": False, "economics_studied": False,
    }
    _write_json(output_root / "diagnostic_summary.json", summary)
    artifacts = sorted(path for path in output_root.iterdir() if path.name != "m1_2_manifest.json")
    _write_json(output_root / "m1_2_manifest.json", {"program": PROGRAM, "phase": PHASE, "m1_authority": M1_AUTHORITY, "m1_1_authority": M1_1_AUTHORITY, "artifacts": {path.name: {"sha256": _sha(path), "bytes": path.stat().st_size} for path in artifacts}})
    return summary
