from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .m1 import (
    FEATURE_BLOCKS,
    OTHER_STRATEGY_RECORDS,
    PROGRAM,
    SESSION_BOUNDS,
    Bar,
    _aggregate,
    _dt,
    _fit_logistic,
    _iso,
    _score,
    _sha,
    _stable_id,
    build_manual_manifest,
)


PHASE = "MVDR_V1_M1_1_LEVEL_TO_LEVEL_STATE_RECONSTRUCTION"
STATUS = "MVDR_V1_M1_1_LEVEL_TO_LEVEL_STATE_READY_FOR_REVIEW"
SEED = 20260827
M1_AUTHORITY = "522ebcd43711ca1bd75770384f5a6026fa46e467"
M1_MARKET_HASHES = {
    "SUIUSDT": "384019dff980f49f49b28e7ba37655b0f9c599124fd022141cce057a73102b08",
    "BTCUSDT": "139d8fa801f678c2aa30d0a2693b468b824d6fcc195968c66e043ba4d4b15efb",
}
STATE_ACTION = {
    "TOWARD_SUPPORT": "SHORT",
    "AT_SUPPORT": "LONG",
    "TOWARD_RESISTANCE": "LONG",
    "AT_RESISTANCE": "SHORT",
    "OTHER": "NO_TRADE",
    "UNSAFE_OR_UNCERTAIN": "NO_TRADE",
}
STATE_TO_FAMILY = {
    "TOWARD_SUPPORT": "SHORT_TO_SUPPORT",
    "AT_SUPPORT": "LONG_FROM_SUPPORT",
    "TOWARD_RESISTANCE": "LONG_TO_RESISTANCE",
    "AT_RESISTANCE": "SHORT_FROM_RESISTANCE",
    "OTHER": "NO_TRADE",
    "UNSAFE_OR_UNCERTAIN": "NO_TRADE",
}
FAMILY_BLOCKS = {
    "LEVEL_ONLY": {"level"},
    "LEVEL_APPROACH": {"level", "approach"},
    "LEVEL_APPROACH_MOMENTUM": {"level", "approach", "momentum"},
    "LEVEL_APPROACH_MOMENTUM_REACTION": {"level", "approach", "momentum", "reaction"},
    "FULL_STATE_MTF": {"level", "approach", "momentum", "reaction", "deceleration", "state", "safety", "mtf"},
    "FULL_STATE_BTC": {"level", "approach", "momentum", "reaction", "deceleration", "state", "safety", "btc"},
    "FULL_STATE": {"level", "approach", "momentum", "reaction", "deceleration", "state", "safety", "mtf", "btc"},
}


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _mean(values: Iterable[float]) -> float:
    material = list(values)
    return sum(material) / len(material) if material else 0.0


def _median(values: Iterable[float]) -> float | None:
    material = sorted(values)
    if not material:
        return None
    middle = len(material) // 2
    return material[middle] if len(material) % 2 else (material[middle - 1] + material[middle]) / 2


def _read_bars(path: Path, expected_hash: str) -> list[Bar]:
    if _sha(path) != expected_hash:
        raise RuntimeError(f"M1_MARKET_HASH_MISMATCH:{path.name}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return [Bar(_dt(row["open_at_utc"]), float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"]), float(row["taker_buy_volume"])) for row in rows]


def manual_state_annotations(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for trade in manifest:
        if trade["manual_trade_id"] == "MVDR-M1-09":
            rows.append({
                "trade_id": trade["manual_trade_id"], "annotation_status": "RESOLVED_MANUAL_ANNOTATION",
                "manual_state": "TOWARD_SUPPORT",
                "manual_note": "Trader described SHORT while price near 0.760 was seeking support near 0.746; unique matching anchor is SHORT 0.7605.",
                "source": "User preregistration for MVDR_V1_M1_1",
            })
        else:
            rows.append({"trade_id": trade["manual_trade_id"], "annotation_status": "UNANNOTATED", "manual_state": None, "manual_note": None, "source": None})
    return {"annotations": rows, "manual_annotation_count": 1, "model_inference_is_not_manual_annotation": True}


def _touch_metadata(bars: list[Bar], level: float, tolerance: float) -> tuple[int, int | None, int]:
    touches = [index for index, bar in enumerate(bars) if bar.low - tolerance <= level <= bar.high + tolerance]
    if not touches:
        return 0, None, len(bars)
    return len(touches), len(bars) - 1 - touches[-1], len(bars) - 1 - touches[0]


def causal_levels(completed: list[Bar], price: float) -> dict[str, Any]:
    if len(completed) < 120 or not math.isfinite(price) or price <= 0:
        raise ValueError("INVALID_LEVEL_CONTEXT")
    recent = completed[-120:]
    last60 = recent[-60:]
    swing_window = recent[-20:]
    swing_support = min(bar.low for bar in swing_window)
    swing_resistance = max(bar.high for bar in swing_window)
    tick = max(price * 0.0005, 1e-8)
    clusters: dict[float, int] = defaultdict(int)
    for bar in last60:
        clusters[round(bar.low / tick) * tick] += 1
        clusters[round(bar.high / tick) * tick] += 1
    ordered = sorted(clusters.items(), key=lambda item: (-item[1], abs(item[0] - price), item[0]))
    cluster_support = next((level for level, _ in ordered if level <= price), min(bar.low for bar in last60))
    cluster_resistance = next((level for level, _ in ordered if level >= price), max(bar.high for bar in last60))
    bars5 = _aggregate(recent, 5)
    bars15 = _aggregate(recent, 15)
    mtf_support = min([bar.low for bar in bars5[-12:]] + [bar.low for bar in bars15[-8:]])
    mtf_resistance = max([bar.high for bar in bars5[-12:]] + [bar.high for bar in bars15[-8:]])
    families = {
        "swing": (swing_support, swing_resistance),
        "cluster": (cluster_support, cluster_resistance),
        "mtf_extrema": (mtf_support, mtf_resistance),
    }
    details: dict[str, Any] = {}
    family_order = {"swing": 0, "cluster": 1, "mtf_extrema": 2}
    for family, (support, resistance) in families.items():
        st, sts, sage = _touch_metadata(last60, support, tick)
        rt, rts, rage = _touch_metadata(last60, resistance, tick)
        details[family] = {
            "support": support, "resistance": resistance,
            "support_touch_count": st, "resistance_touch_count": rt,
            "time_since_support_touch": sts, "time_since_resistance_touch": rts,
            "support_age": sage, "resistance_age": rage,
        }
    supports = [(abs(price - data["support"]), family_order[family], data["support"], family) for family, data in details.items() if data["support"] <= price]
    resistances = [(abs(data["resistance"] - price), family_order[family], data["resistance"], family) for family, data in details.items() if data["resistance"] >= price]
    if not supports or not resistances:
        raise ValueError("MISSING_CAUSAL_LEVEL")
    valid_pairs = [
        (support_row[0] + resistance_row[0], support_row[1], resistance_row[1], support_row, resistance_row)
        for support_row in supports for resistance_row in resistances if support_row[2] < resistance_row[2]
    ]
    if not valid_pairs:
        raise ValueError("INVALID_LEVEL_ORDER")
    _, _, _, support_row, resistance_row = min(valid_pairs)
    _, _, support, support_family = support_row
    _, _, resistance, resistance_family = resistance_row
    return {
        "families": details, "support_price": support, "resistance_price": resistance,
        "support_family": support_family, "resistance_family": resistance_family,
        "tie_break": "valid support<resistance pair with minimum total distance; then support and resistance family order swing, cluster, mtf_extrema",
    }


def _distance_bin(distance_bps: float) -> str:
    if distance_bps < 10:
        return "0_10"
    if distance_bps < 25:
        return "10_25"
    if distance_bps < 50:
        return "25_50"
    if distance_bps < 100:
        return "50_100"
    return "100_PLUS"


def approach_components(completed: list[Bar], level: float, toward_support: bool, minutes: int) -> dict[str, float]:
    seq = completed[-(minutes + 1):]
    prices = [bar.close for bar in seq]
    current = prices[-1]
    signed = (current - prices[0]) / prices[0] * 10_000
    changes = [prices[index] - prices[index - 1] for index in range(1, len(prices))]
    traveled = sum(abs(value) for value in changes) / prices[0] * 10_000
    direction = -1.0 if toward_support else 1.0
    directional_displacement = direction * signed
    efficiency = abs(signed) / traveled if traveled else 0.0
    past_distance = abs(prices[0] - level) / prices[0] * 10_000
    current_distance = abs(current - level) / current * 10_000
    distance_reduction = past_distance - current_distance
    advancing = _mean(direction * value > 0 for value in changes)
    half = max(1, len(changes) // 2)
    early_velocity = direction * sum(changes[:half]) / prices[0] * 10_000 / half
    late_velocity = direction * sum(changes[-half:]) / prices[0] * 10_000 / half
    acceleration = late_velocity - early_velocity
    progress = _clip(distance_reduction / max(past_distance, 10.0))
    direction_score = _clip(directional_displacement / max(traveled, 1.0))
    acceleration_score = _clip(0.5 + acceleration / 10.0)
    score = _mean((progress, direction_score, efficiency, advancing, acceleration_score))
    return {
        "signed_displacement_bps": signed, "traveled_distance_bps": traveled,
        "path_efficiency": efficiency, "velocity_bps_per_min": directional_displacement / minutes,
        "acceleration": acceleration, "distance_change_bps": -distance_reduction,
        "advancing_close_fraction": advancing, "distance_reduction_fraction": progress,
        "score": score,
    }


def reaction_components(completed: list[Bar], support: float, resistance: float) -> dict[str, float]:
    recent = completed[-3:]
    last = recent[-1]
    price = last.close
    candle_range = max(last.high - last.low, 1e-12)
    penetration_support = max(0.0, support - min(bar.low for bar in recent)) / price * 10_000
    penetration_resistance = max(0.0, max(bar.high for bar in recent) - resistance) / price * 10_000
    reclaim_support = max(0.0, last.close - support) / price * 10_000 if penetration_support else 0.0
    reclaim_resistance = max(0.0, resistance - last.close) / price * 10_000 if penetration_resistance else 0.0
    lower_wick = (min(last.open, last.close) - last.low) / candle_range
    upper_wick = (last.high - max(last.open, last.close)) / candle_range
    close_away_support = _clip((last.close - support) / price * 10_000 / 25.0)
    close_away_resistance = _clip((resistance - last.close) / price * 10_000 / 25.0)
    follow_support = _mean(recent[index].close > recent[index - 1].close for index in range(1, 3))
    follow_resistance = _mean(recent[index].close < recent[index - 1].close for index in range(1, 3))
    support_score = _mean((_clip(penetration_support / 10.0), _clip(reclaim_support / 25.0), lower_wick, close_away_support, follow_support))
    resistance_score = _mean((_clip(penetration_resistance / 10.0), _clip(reclaim_resistance / 25.0), upper_wick, close_away_resistance, follow_resistance))
    return {
        "penetration_support_bps": penetration_support, "reclaim_support_bps": reclaim_support,
        "lower_wick_rejection": lower_wick, "close_away_from_support": close_away_support,
        "followthrough_from_support": follow_support, "reaction_support_score": support_score,
        "penetration_resistance_bps": penetration_resistance, "reclaim_resistance_bps": reclaim_resistance,
        "upper_wick_rejection": upper_wick, "close_away_from_resistance": close_away_resistance,
        "followthrough_from_resistance": follow_resistance, "reaction_resistance_score": resistance_score,
    }


def deceleration_components(completed: list[Bar], toward_support: bool) -> dict[str, float]:
    seq = completed[-6:]
    first, second = seq[:3], seq[3:]
    body_first = _mean(abs(bar.close - bar.open) for bar in first)
    body_second = _mean(abs(bar.close - bar.open) for bar in second)
    range_first = _mean(bar.high - bar.low for bar in first)
    range_second = _mean(bar.high - bar.low for bar in second)
    velocity_first = abs(first[-1].close - first[0].open)
    velocity_second = abs(second[-1].close - second[0].open)
    overlaps = [max(0.0, min(seq[index].high, seq[index - 1].high) - max(seq[index].low, seq[index - 1].low)) / max(seq[index].high - seq[index].low, 1e-12) for index in range(1, len(seq))]
    failed_extremes = _mean(seq[index].low >= seq[index - 1].low if toward_support else seq[index].high <= seq[index - 1].high for index in range(1, len(seq)))
    last = seq[-1]
    wick = (min(last.open, last.close) - last.low) / max(last.high - last.low, 1e-12) if toward_support else (last.high - max(last.open, last.close)) / max(last.high - last.low, 1e-12)
    components = {
        "body_contraction": _clip(1.0 - body_second / max(body_first, 1e-12)),
        "range_contraction": _clip(1.0 - range_second / max(range_first, 1e-12)),
        "velocity_reduction": _clip(1.0 - velocity_second / max(velocity_first, 1e-12)),
        "overlap": _mean(overlaps), "failed_new_extremes": failed_extremes, "wick_growth": wick,
    }
    components["score"] = _mean(components.values())
    return components


def _momentum(completed: list[Bar], toward_support: bool) -> dict[str, float]:
    seq = completed[-6:]
    direction = -1.0 if toward_support else 1.0
    changes = [bar.close - bar.open for bar in seq]
    closes = [bar.close for bar in seq]
    path = sum(abs(closes[index] - closes[index - 1]) for index in range(1, len(closes)))
    net = direction * (closes[-1] - closes[0])
    ranges = [bar.high - bar.low for bar in seq]
    bodies = [abs(bar.close - bar.open) for bar in seq]
    volume_base = _mean(bar.volume for bar in completed[-12:-6])
    components = {
        "body_direction": _mean(direction * value > 0 for value in changes),
        "body_dominance": _mean(_clip(direction * value / max(abs(value), 1e-12)) for value in changes),
        "close_progression": _mean(direction * (closes[index] - closes[index - 1]) > 0 for index in range(1, len(closes))),
        "range_expansion": _clip(_mean(ranges[-3:]) / max(_mean(ranges[:3]), 1e-12) / 2.0),
        "path_efficiency": _clip(net / max(path, 1e-12)),
        "volume_expansion": _clip(_mean(bar.volume for bar in seq) / max(volume_base, 1e-12) / 2.0),
        "consecutive_movement": max((sum(1 for value in changes[-index:] if direction * value > 0) for index in range(1, 7)), default=0) / 6.0,
        "retracement_control": _clip(1.0 - max(0.0, -net) / max(path, 1e-12)),
    }
    components["score"] = _mean(components.values())
    return components


def choose_state(family_scores: dict[str, float]) -> tuple[str, str, float, float]:
    family_to_state = {"SHORT_TO_SUPPORT": "TOWARD_SUPPORT", "LONG_FROM_SUPPORT": "AT_SUPPORT", "LONG_TO_RESISTANCE": "TOWARD_RESISTANCE", "SHORT_FROM_RESISTANCE": "AT_RESISTANCE"}
    ordered = sorted(family_scores.items(), key=lambda item: (-item[1], item[0]))
    best_family, best_score = ordered[0]
    margin = best_score - ordered[1][1]
    state = family_to_state[best_family] if best_score >= 0.45 and margin >= 0.05 else "OTHER"
    return state, best_family, best_score, margin


def level_state_frame(decision_at: datetime, sui: list[Bar], btc: list[Bar]) -> dict[str, Any]:
    sui_done = [bar for bar in sui if bar.open_at + timedelta(minutes=1) <= decision_at]
    btc_done = [bar for bar in btc if bar.open_at + timedelta(minutes=1) <= decision_at]
    if len(sui_done) < 120 or len(btc_done) < 120:
        raise ValueError("INSUFFICIENT_CAUSAL_CONTEXT")
    price = sui_done[-1].close
    if not all(math.isfinite(value) for bar in sui_done[-120:] for value in (bar.open, bar.high, bar.low, bar.close, bar.volume)):
        raise ValueError("NON_FINITE_MARKET_DATA")
    levels = causal_levels(sui_done, price)
    support = levels["support_price"]
    resistance = levels["resistance_price"]
    distance_support = (price - support) / price * 10_000
    distance_resistance = (resistance - price) / price * 10_000
    width = (resistance - support) / price * 10_000
    position_raw = (price - support) / (resistance - support)
    approach_support = {str(minutes): approach_components(sui_done, support, True, minutes) for minutes in (3, 6, 9, 15)}
    approach_resistance = {str(minutes): approach_components(sui_done, resistance, False, minutes) for minutes in (3, 6, 9, 15)}
    approach_support_score = _mean(value["score"] for value in approach_support.values())
    approach_resistance_score = _mean(value["score"] for value in approach_resistance.values())
    momentum_support = _momentum(sui_done, True)
    momentum_resistance = _momentum(sui_done, False)
    decel_support = deceleration_components(sui_done, True)
    decel_resistance = deceleration_components(sui_done, False)
    reaction = reaction_components(sui_done, support, resistance)

    support_values = [data["support"] for data in levels["families"].values()]
    resistance_values = [data["resistance"] for data in levels["families"].values()]
    support_spread = (max(support_values) - min(support_values)) / price * 10_000
    resistance_spread = (max(resistance_values) - min(resistance_values)) / price * 10_000
    support_details = levels["families"][levels["support_family"]]
    resistance_details = levels["families"][levels["resistance_family"]]
    clarity_support = _mean((_clip(1.0 - support_spread / 100.0), _clip(support_details["support_touch_count"] / 3.0)))
    clarity_resistance = _mean((_clip(1.0 - resistance_spread / 100.0), _clip(resistance_details["resistance_touch_count"] / 3.0)))
    near_support = _clip(1.0 - distance_support / 50.0)
    near_resistance = _clip(1.0 - distance_resistance / 50.0)
    room_support = _clip(distance_support / 100.0)
    room_resistance = _clip(distance_resistance / 100.0)

    family_scores = {
        "SHORT_TO_SUPPORT": _mean((clarity_support, room_support, approach_support_score, momentum_support["score"], _mean(value["path_efficiency"] for value in approach_support.values()), 1.0 - reaction["reaction_support_score"], _clip(distance_support / 25.0))),
        "LONG_FROM_SUPPORT": _mean((clarity_support, near_support, reaction["reaction_support_score"], decel_support["score"], momentum_resistance["score"], room_resistance)),
        "LONG_TO_RESISTANCE": _mean((clarity_resistance, room_resistance, approach_resistance_score, momentum_resistance["score"], _mean(value["path_efficiency"] for value in approach_resistance.values()), 1.0 - reaction["reaction_resistance_score"], _clip(distance_resistance / 25.0))),
        "SHORT_FROM_RESISTANCE": _mean((clarity_resistance, near_resistance, reaction["reaction_resistance_score"], decel_resistance["score"], momentum_support["score"], room_support)),
    }
    market_state, best_family, best_score, margin = choose_state(family_scores)
    potential_action = STATE_ACTION[market_state]
    state_confidence = best_score * (0.5 + margin / 2.0)

    btc_prices = [bar.close for bar in btc_done[-4:]]
    btc_displacement = (btc_prices[-1] - btc_prices[0]) / btc_prices[0] * 10_000
    action_direction = 1.0 if potential_action == "LONG" else -1.0 if potential_action == "SHORT" else 0.0
    btc_signed = action_direction * btc_displacement
    if action_direction == 0:
        btc_state, btc_safety = "BTC_NEUTRAL", 0.5
    elif btc_signed >= 30:
        btc_state, btc_safety = "BTC_ALIGNED", 1.0
    elif btc_signed <= -30:
        btc_state, btc_safety = "BTC_STRONGLY_OPPOSING", 0.0
    elif btc_signed < 0:
        btc_state, btc_safety = "BTC_OPPOSING", 0.4
    else:
        btc_state, btc_safety = "BTC_NEUTRAL", 0.75

    recent = sui_done[-15:]
    ranges = [(bar.high - bar.low) / bar.close * 10_000 for bar in recent]
    shock = ranges[-1] / max(_median(ranges[:-1]) or 1.0, 1e-12)
    changes = [recent[index].close - recent[index - 1].close for index in range(1, len(recent))]
    alternation = _mean(changes[index] * changes[index - 1] < 0 for index in range(1, len(changes)))
    traveled = sum(abs(value) for value in changes)
    efficiency = abs(recent[-1].close - recent[0].close) / traveled if traveled else 0.0
    volatility_safety = _mean((_clip(2.0 - shock), 1.0 - alternation, efficiency))

    is_destination = market_state in ("TOWARD_SUPPORT", "TOWARD_RESISTANCE")
    is_origin = market_state in ("AT_SUPPORT", "AT_RESISTANCE")
    level_clarity = clarity_support if market_state in ("TOWARD_SUPPORT", "AT_SUPPORT") else clarity_resistance
    room_score = room_support if market_state == "TOWARD_SUPPORT" else room_resistance
    if market_state == "AT_SUPPORT":
        room_score = room_resistance
    elif market_state == "AT_RESISTANCE":
        room_score = room_support
    momentum_consistency = momentum_support["score"] if market_state in ("TOWARD_SUPPORT", "AT_RESISTANCE") else momentum_resistance["score"]
    reaction_confirmation = reaction["reaction_support_score"] if market_state == "AT_SUPPORT" else reaction["reaction_resistance_score"] if market_state == "AT_RESISTANCE" else 1.0
    safety_components = (level_clarity, room_score, momentum_consistency if is_destination else 1.0, reaction_confirmation if is_origin else 1.0, btc_safety, volatility_safety)
    trade_safety = _mean(safety_components)
    vetoes = []
    if market_state == "OTHER":
        vetoes.append("LOW_OR_AMBIGUOUS_STATE_CONFIDENCE")
    if level_clarity < 0.25:
        vetoes.append("AMBIGUOUS_LEVEL")
    if room_score < 0.10:
        vetoes.append("INSUFFICIENT_ROOM")
    if is_destination and momentum_consistency < 0.35:
        vetoes.append("MOMENTUM_INCONSISTENT")
    if is_origin and reaction_confirmation < 0.35:
        vetoes.append("REACTION_UNCONFIRMED")
    if btc_state == "BTC_STRONGLY_OPPOSING":
        vetoes.append("BTC_STRONGLY_OPPOSING")
    if volatility_safety < 0.25:
        vetoes.append("VOLATILITY_CHAOS")
    if trade_safety < 0.55:
        vetoes.append("SAFETY_SCORE_BELOW_0_55")
    final_action = potential_action if not vetoes else "NO_TRADE"
    final_state = market_state if not vetoes else "UNSAFE_OR_UNCERTAIN"

    features: dict[str, float] = {
        "level.distance_support_bps": distance_support, "level.distance_resistance_bps": distance_resistance,
        "level.range_width_bps": width, "level.position_raw": position_raw,
        "level.support_clarity": clarity_support, "level.resistance_clarity": clarity_resistance,
        "level.support_touch_count": support_details["support_touch_count"], "level.resistance_touch_count": resistance_details["resistance_touch_count"],
        "level.time_since_support_touch": support_details["time_since_support_touch"] if support_details["time_since_support_touch"] is not None else 120,
        "level.time_since_resistance_touch": resistance_details["time_since_resistance_touch"] if resistance_details["time_since_resistance_touch"] is not None else 120,
        "level.support_age": support_details["support_age"], "level.resistance_age": resistance_details["resistance_age"],
        "approach.support_score": approach_support_score, "approach.resistance_score": approach_resistance_score,
        "momentum.toward_support": momentum_support["score"], "momentum.toward_resistance": momentum_resistance["score"],
        "deceleration.near_support": decel_support["score"], "deceleration.near_resistance": decel_resistance["score"],
        "reaction.from_support": reaction["reaction_support_score"], "reaction.from_resistance": reaction["reaction_resistance_score"],
        "state.short_to_support": family_scores["SHORT_TO_SUPPORT"], "state.long_from_support": family_scores["LONG_FROM_SUPPORT"],
        "state.long_to_resistance": family_scores["LONG_TO_RESISTANCE"], "state.short_from_resistance": family_scores["SHORT_FROM_RESISTANCE"],
        "state.confidence": state_confidence, "safety.level_clarity": level_clarity, "safety.room": room_score,
        "safety.momentum_consistency": momentum_consistency, "safety.reaction_confirmation": reaction_confirmation,
        "safety.volatility": volatility_safety, "safety.trade": trade_safety,
        "btc.displacement_3m_bps": btc_displacement, "btc.safety": btc_safety,
        "mtf.sui_5m_displacement_bps": (sui_done[-1].close - sui_done[-6].close) / sui_done[-6].close * 10_000,
        "mtf.sui_15m_displacement_bps": (sui_done[-1].close - sui_done[-16].close) / sui_done[-16].close * 10_000,
    }
    for minutes, values in approach_support.items():
        for key, value in values.items():
            features[f"approach.support_{minutes}m.{key}"] = value
    for minutes, values in approach_resistance.items():
        for key, value in values.items():
            features[f"approach.resistance_{minutes}m.{key}"] = value
    for key, value in reaction.items():
        features[f"reaction.{key}"] = value
    return {
        "frame_id": _stable_id(PHASE, _iso(decision_at)), "decision_at_utc": _iso(decision_at),
        "latest_completed_1m_open_at": _iso(sui_done[-1].open_at), "current_price": price,
        "support_price": support, "resistance_price": resistance, "support_family": levels["support_family"],
        "resistance_family": levels["resistance_family"], "level_family_views": levels["families"],
        "distance_to_support_bps": distance_support, "distance_to_resistance_bps": distance_resistance,
        "room_to_support_bps": distance_support, "room_to_resistance_bps": distance_resistance,
        "range_width_bps": width, "position_between_levels_raw": position_raw,
        "position_between_levels_display": _clip(position_raw),
        "distance_support_bin": _distance_bin(distance_support), "distance_resistance_bin": _distance_bin(distance_resistance),
        "approach_support": approach_support, "approach_resistance": approach_resistance,
        "momentum_toward_support": momentum_support, "momentum_toward_resistance": momentum_resistance,
        "deceleration_near_support": decel_support, "deceleration_near_resistance": decel_resistance,
        "reaction": reaction, "family_scores": family_scores, "market_state": market_state,
        "model_inferred_state": market_state, "model_inferred_family": STATE_TO_FAMILY[market_state],
        "potential_action": potential_action, "state_confidence": state_confidence,
        "level_clarity_score": level_clarity, "room_score": room_score,
        "momentum_consistency_score": momentum_consistency, "reaction_confirmation_score": reaction_confirmation,
        "btc_state": btc_state, "btc_safety_score": btc_safety, "volatility_safety_score": volatility_safety,
        "trade_safety_score": trade_safety, "safety_veto_reason": vetoes, "final_state": final_state,
        "final_action": final_action, "features": features,
    }


def apply_entry_labels(frames: list[dict[str, Any]], manifest: list[dict[str, Any]], other_entries: list[datetime]) -> list[dict[str, Any]]:
    anchors = {trade["entry_at_utc"][:16]: trade for trade in manifest}
    manual_times = [_dt(trade["entry_at_utc"]) for trade in manifest]
    rows = []
    for frame in frames:
        row = dict(frame)
        trade = anchors.get(row["decision_at_utc"][:16])
        decision = _dt(row["decision_at_utc"])
        manual_blackout = any(abs((decision - value).total_seconds()) <= 360 for value in manual_times)
        other_blackout = any(abs((decision - value).total_seconds()) <= 360 for value in other_entries)
        row.update({
            "candidate_id": _stable_id("state-candidate", row["frame_id"]), "session": row["decision_at_utc"][:10],
            "selected": trade is not None, "manual_trade_id": trade["manual_trade_id"] if trade else None,
            "manual_side": trade["side"] if trade else None, "manual_entry_blackout": manual_blackout,
            "other_strategy_exclusion": other_blackout, "negative_eligible": not manual_blackout and not other_blackout,
            "label_semantics": "DID_TRADER_SELECT_THIS_ENTRY",
        })
        rows.append(row)
    return rows


def assign_volatility_terciles(rows: list[dict[str, Any]]) -> None:
    for session in sorted({row["session"] for row in rows}):
        local = sorted(row["volatility_safety_score"] for row in rows if row["session"] == session)
        low = local[len(local) // 3]
        high = local[(2 * len(local)) // 3]
        for row in rows:
            if row["session"] != session:
                continue
            value = row["volatility_safety_score"]
            row["volatility_tercile"] = "LOW" if value <= low else "MID" if value <= high else "HIGH"


def build_matched_controls(rows: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assign_volatility_terciles(rows)
    controls = []
    positives = {row["manual_trade_id"]: row for row in rows if row["selected"]}
    for trade in manifest:
        anchor = positives[trade["manual_trade_id"]]
        compatible_families = ("LONG_FROM_SUPPORT", "LONG_TO_RESISTANCE") if trade["side"] == "LONG" else ("SHORT_TO_SUPPORT", "SHORT_FROM_RESISTANCE")
        family = max(compatible_families, key=lambda name: (anchor["family_scores"][name], name))
        distance_bin = anchor["distance_support_bin"] if "SUPPORT" in family else anchor["distance_resistance_bin"]
        approach_key = "approach.support_score" if "SUPPORT" in family else "approach.resistance_score"
        anchor_approach = anchor["features"][approach_key]
        exact, without_volatility, family_only = [], [], []
        for row in rows:
            row_bin = row["distance_support_bin"] if "SUPPORT" in family else row["distance_resistance_bin"]
            if not (row["negative_eligible"] and row["session"] == anchor["session"] and row["potential_action"] == trade["side"] and row["model_inferred_family"] == family):
                continue
            family_only.append(row)
            if row_bin == distance_bin:
                without_volatility.append(row)
                if row["volatility_tercile"] == anchor["volatility_tercile"]:
                    exact.append(row)
        selected: list[tuple[dict[str, Any], str]] = []
        seen: set[str] = set()
        for pool, tier in ((exact, "EXACT"), (without_volatility, "IGNORE_VOLATILITY_TERCILE"), (family_only, "FAMILY_SIDE_SESSION_ONLY")):
            pool.sort(key=lambda row: (abs(row["features"][approach_key] - anchor_approach), row["candidate_id"]))
            for row in pool:
                if row["candidate_id"] not in seen and len(selected) < 20:
                    selected.append((row, tier))
                    seen.add(row["candidate_id"])
        for rank, (row, tier) in enumerate(selected, 1):
            controls.append({
                "control_id": _stable_id("matched", trade["manual_trade_id"], row["candidate_id"]),
                "matched_to_trade_id": trade["manual_trade_id"], "candidate_id": row["candidate_id"],
                "session": row["session"], "side_hypothesis": row["potential_action"],
                "state_family": row["model_inferred_family"], "distance_bin": distance_bin,
                "volatility_tercile": row["volatility_tercile"], "selected": False,
                "match_tier": tier, "anchor_model_inferred_family": anchor["model_inferred_family"],
                "anchor_side_compatible_matching_family": family,
                "label": "NOT_SELECTED_BY_TRADER", "features": row["features"],
            })
    return controls


def _feature_names(rows: list[dict[str, Any]], blocks: set[str]) -> list[str]:
    return sorted(name for name in rows[0]["features"] if name.split(".", 1)[0] in blocks)


def _training_rows(candidates: list[dict[str, Any]], controls: list[dict[str, Any]], heldout_id: str, strict_session: bool) -> list[dict[str, Any]]:
    heldout = next(row for row in candidates if row["manual_trade_id"] == heldout_id)
    positives = [row for row in candidates if row["selected"] and row["manual_trade_id"] != heldout_id and (not strict_session or row["session"] != heldout["session"])]
    allowed_ids = {row["manual_trade_id"] for row in positives}
    negatives = [{"selected": False, "features": row["features"], "session": row["session"], "candidate_id": row["candidate_id"]} for row in controls if row["matched_to_trade_id"] in allowed_ids and (not strict_session or row["session"] != heldout["session"])]
    negatives.sort(key=lambda row: row["candidate_id"])
    if not positives or not negatives:
        raise RuntimeError(f"INSUFFICIENT_FOLD_TRAINING:{heldout_id}")
    return positives + negatives


def evaluate(
    candidates: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    blocks: set[str],
    label: str,
    strict_session: bool = False,
) -> list[dict[str, Any]]:
    predictions = []
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    controls_by_trade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for control in controls:
        controls_by_trade[control["matched_to_trade_id"]].append(control)
    for trade in manifest:
        train = _training_rows(candidates, controls, trade["manual_trade_id"], strict_session)
        names = _feature_names(train, blocks)
        model = _fit_logistic(train, names)
        test = [row for row in candidates if row["session"] == trade["entry_at_utc"][:10]]
        scored = sorted(((row, _score(model, row)) for row in test), key=lambda item: (-item[1], item[0]["candidate_id"]))
        target = next(item for item in scored if item[0]["manual_trade_id"] == trade["manual_trade_id"])
        rank_all = next(index for index, item in enumerate(scored, 1) if item[0]["candidate_id"] == target[0]["candidate_id"])
        state_pool = [item for item in scored if item[0]["model_inferred_family"] == target[0]["model_inferred_family"]]
        rank_state = next(index for index, item in enumerate(state_pool, 1) if item[0]["candidate_id"] == target[0]["candidate_id"])
        negative_scores = sorted((_score(model, row) for row in train if not row["selected"]), reverse=True)
        session_count = max(1, len({row["session"] for row in train}))
        threshold = negative_scores[min(len(negative_scores) - 1, 5 * session_count - 1)]
        emitted_without_safety = [item for item in scored if item[1] >= threshold and item[0]["potential_action"] != "NO_TRADE"]
        emitted = [item for item in scored if item[1] >= threshold and item[0]["final_action"] != "NO_TRADE"]
        capture = {}
        timing_errors = []
        for item in emitted:
            if item[0]["final_action"] == trade["side"]:
                timing_errors.append(abs((_dt(item[0]["decision_at_utc"]) - _dt(trade["entry_at_utc"])).total_seconds()) / 60)
        for tolerance in (1, 3, 6):
            capture[str(tolerance)] = any(value <= tolerance for value in timing_errors)
        heldout_controls = controls_by_trade[trade["manual_trade_id"]]
        matched_scores = [_score(model, candidate_by_id[row["candidate_id"]]) for row in heldout_controls]
        matched_percentile = sum(target[1] > value for value in matched_scores) / len(matched_scores) if matched_scores else None
        predictions.append({
            "evaluation": label, "manual_trade_id": trade["manual_trade_id"],
            "heldout_entry_revealed_after_scoring": trade["entry_at_utc"], "manual_side": trade["side"],
            "model_inferred_state": target[0]["model_inferred_state"], "model_inferred_family": target[0]["model_inferred_family"],
            "potential_action": target[0]["potential_action"], "final_action": target[0]["final_action"],
            "state_confidence": target[0]["state_confidence"], "trade_safety_score": target[0]["trade_safety_score"],
            "safety_veto_reason": target[0]["safety_veto_reason"], "score": target[1],
            "rank_all": rank_all, "candidate_count": len(scored),
            "percentile_rank": 1.0 - (rank_all - 1) / max(1, len(scored) - 1),
            "rank_matched_state": rank_state, "matched_state_candidate_count": len(state_pool),
            "matched_hard_negative_count": len(matched_scores), "matched_hard_negative_percentile": matched_percentile,
            "correct_side": target[0]["potential_action"] == trade["side"],
            "threshold": threshold, "capture": capture,
            "nearest_timing_error_minutes": min(timing_errors) if timing_errors else None,
            "emitted_signals": len(emitted), "emitted_without_safety": len(emitted_without_safety),
            "safety_abstention": len(emitted_without_safety) - len(emitted),
            "normalization_scope": "TRAINING_FOLD_ONLY", "threshold_scope": "TRAINING_FOLD_ONLY",
        })
    return predictions


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(predictions)
    emitted = sum(row["emitted_signals"] for row in predictions)
    emitted_without = sum(row["emitted_without_safety"] for row in predictions)
    matched = [row["matched_hard_negative_percentile"] for row in predictions if row["matched_hard_negative_percentile"] is not None]
    return {
        "N": n,
        "top_1pct": sum(row["percentile_rank"] >= 0.99 for row in predictions) / n,
        "top_5pct": sum(row["percentile_rank"] >= 0.95 for row in predictions) / n,
        "top_10pct": sum(row["percentile_rank"] >= 0.90 for row in predictions) / n,
        "median_percentile_rank": _median(row["percentile_rank"] for row in predictions),
        "correct_side_rate": sum(row["correct_side"] for row in predictions) / n,
        "capture_rate_1m": sum(row["capture"]["1"] for row in predictions) / n,
        "capture_rate_3m": sum(row["capture"]["3"] for row in predictions) / n,
        "capture_rate_6m": sum(row["capture"]["6"] for row in predictions) / n,
        "median_timing_error_minutes": _median(row["nearest_timing_error_minutes"] for row in predictions if row["nearest_timing_error_minutes"] is not None),
        "median_matched_hard_negative_percentile": _median(matched),
        "total_emitted_signals": emitted, "signals_per_manual_trade": emitted / n,
        "total_emitted_without_safety": emitted_without,
        "safety_signal_reduction": 1.0 - emitted / max(1, emitted_without),
        "manual_capture_per_emitted_signal": sum(row["capture"]["6"] for row in predictions) / max(1, emitted),
    }


def state_confusion_diagnostic(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    confused = []
    total = 0
    for row in candidates:
        naive_action = "LONG" if row["distance_to_support_bps"] <= row["distance_to_resistance_bps"] else "SHORT"
        if naive_action != row["potential_action"] and row["potential_action"] != "NO_TRADE":
            total += 1
            if row["selected"]:
                confused.append({"manual_trade_id": row["manual_trade_id"], "m1_proximity_action": naive_action, "level_state_action": row["potential_action"], "model_inferred_state": row["model_inferred_state"]})
    return {
        "definition": "M1 proximity action buys nearest support or sells nearest resistance; compare with level-state potential action",
        "all_candidate_confusion_count": total, "manual_anchor_confusions": confused,
        "traveling_toward_support_but_proximity_buys": sum(row["model_inferred_state"] == "TOWARD_SUPPORT" and row["distance_to_support_bps"] <= row["distance_to_resistance_bps"] for row in candidates),
        "traveling_toward_resistance_but_proximity_sells": sum(row["model_inferred_state"] == "TOWARD_RESISTANCE" and row["distance_to_resistance_bps"] < row["distance_to_support_bps"] for row in candidates),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as zipped:
        for row in rows:
            zipped.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())


def _shuffle_labels(candidates: list[dict[str, Any]], manifest: list[dict[str, Any]], rng: random.Random, draw: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = [{**row, "selected": False, "manual_trade_id": None, "manual_side": None} for row in candidates]
    by_session_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in shuffled:
        by_session_action[(row["session"], row["potential_action"])].append(row)
    new_manifest = []
    for session in SESSION_BOUNDS:
        for side in ("LONG", "SHORT"):
            count = sum(trade["entry_at_utc"][:10] == session and trade["side"] == side for trade in manifest)
            pool = by_session_action[(session, side)]
            for index, row in enumerate(rng.sample(pool, count)):
                trade_id = f"LABEL-SHUFFLE-{draw:02d}-{session}-{side}-{index}"
                row.update({"selected": True, "manual_trade_id": trade_id, "manual_side": side})
                new_manifest.append({"manual_trade_id": trade_id, "entry_at_utc": row["decision_at_utc"], "side": side})
    return shuffled, new_manifest


def _shuffle_states(candidates: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    shuffled = [{**row, "features": dict(row["features"])} for row in candidates]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in shuffled:
        groups[(row["session"], row["potential_action"])].append(row)
    state_keys = ["state.short_to_support", "state.long_from_support", "state.long_to_resistance", "state.short_from_resistance", "state.confidence"]
    for group in groups.values():
        payloads = [([row["features"][key] for key in state_keys], row["model_inferred_state"], row["model_inferred_family"]) for row in group]
        rng.shuffle(payloads)
        for row, payload in zip(group, payloads):
            values, state, family = payload
            for key, value in zip(state_keys, values):
                row["features"][key] = value
            row["model_inferred_state"] = state
            row["model_inferred_family"] = family
    return shuffled


def _m1_matched_baseline(repo_root: Path, candidates: list[dict[str, Any]], controls: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> dict[str, Any]:
    path = repo_root / "sandbox/manual_visual_decision_replication_v1/artifacts/m1/candidate_decisions.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        m1_rows = [json.loads(line) for line in handle]
    m1_by_key = {(row["decision_at_utc"][:16], row["side"]): row for row in m1_rows}
    state_by_id = {row["candidate_id"]: row for row in candidates}
    percentiles = []
    for heldout in manifest:
        train = []
        allowed = {trade["manual_trade_id"] for trade in manifest if trade["manual_trade_id"] != heldout["manual_trade_id"]}
        for trade in manifest:
            if trade["manual_trade_id"] in allowed:
                source = m1_by_key[(trade["entry_at_utc"][:16], trade["side"])]
                train.append({"selected": True, "features": source["features"]})
        for control in controls:
            if control["matched_to_trade_id"] in allowed:
                state_row = state_by_id[control["candidate_id"]]
                source = m1_by_key.get((state_row["decision_at_utc"][:16], control["side_hypothesis"]))
                if source:
                    train.append({"selected": False, "features": source["features"]})
        names = sorted(name for name in train[0]["features"] if name.split(".", 1)[0] in set(FEATURE_BLOCKS))
        model = _fit_logistic(train, names)
        target = m1_by_key[(heldout["entry_at_utc"][:16], heldout["side"])]
        target_score = _score(model, target)
        matched_scores = []
        for control in controls:
            if control["matched_to_trade_id"] != heldout["manual_trade_id"]:
                continue
            state_row = state_by_id[control["candidate_id"]]
            source = m1_by_key.get((state_row["decision_at_utc"][:16], control["side_hypothesis"]))
            if source:
                matched_scores.append(_score(model, source))
        if matched_scores:
            percentiles.append(sum(target_score > value for value in matched_scores) / len(matched_scores))
    return {"median_matched_hard_negative_percentile": _median(percentiles), "folds_computable": len(percentiles), "features": "M1_FULL_FROZEN"}


def execute_m1_1(repo_root: Path, output_root: Path) -> dict[str, Any]:
    subprocess.run(["git", "merge-base", "--is-ancestor", M1_AUTHORITY, "HEAD"], cwd=repo_root, check=True)
    output_root.mkdir(parents=True, exist_ok=True)
    m1_root = repo_root / "sandbox/manual_visual_decision_replication_v1/artifacts/m1"
    m1_artifact_manifest = json.loads((m1_root / "m1_manifest.json").read_text(encoding="utf-8"))
    for name, metadata in m1_artifact_manifest["artifacts"].items():
        if _sha(m1_root / name) != metadata["sha256"]:
            raise RuntimeError(f"M1_ARTIFACT_HASH_MISMATCH:{name}")
    frozen_manifest_payload = json.loads((m1_root / "manual_entry_manifest.json").read_text(encoding="utf-8"))
    manifest = frozen_manifest_payload["trades"]
    if manifest != build_manual_manifest(repo_root / "backtest_results/real_trade_analysis.json"):
        raise RuntimeError("M1_MANUAL_ENTRY_MANIFEST_CHANGED")
    annotations = manual_state_annotations(manifest)
    _write_json(output_root / "manual_state_annotations.json", annotations)

    market = {
        symbol: _read_bars(m1_root / f"{symbol}_1m.jsonl.gz", expected)
        for symbol, expected in M1_MARKET_HASHES.items()
    }
    frames = []
    for start_text, end_text in SESSION_BOUNDS.values():
        cursor, end = _dt(start_text), _dt(end_text)
        while cursor < end:
            frames.append(level_state_frame(cursor, market["SUIUSDT"], market["BTCUSDT"]))
            cursor += timedelta(minutes=1)
    _write_jsonl(output_root / "level_state_frames.jsonl.gz", frames)

    source_records = json.loads((repo_root / "backtest_results/real_trade_analysis.json").read_text(encoding="utf-8"))
    other_entries = [_dt(source_records[index - 1]["entry_time"].replace(" ", "T") + "Z") for index in OTHER_STRATEGY_RECORDS]
    candidates = apply_entry_labels(frames, manifest, other_entries)
    controls = build_matched_controls(candidates, manifest)
    _write_jsonl(output_root / "state_candidates.jsonl.gz", candidates)
    _write_jsonl(output_root / "matched_hard_negatives.jsonl.gz", controls)

    primary = evaluate(candidates, controls, manifest, FAMILY_BLOCKS["FULL_STATE"], "LOTO_LEVEL_STATE_WITH_SAFETY")
    session_out = evaluate(candidates, controls, manifest, FAMILY_BLOCKS["FULL_STATE"], "LEAVE_SESSION_OUT_LEVEL_STATE_WITH_SAFETY", strict_session=True)
    _write_jsonl(output_root / "loto_predictions.jsonl.gz", primary)
    _write_jsonl(output_root / "session_rankings.jsonl.gz", sorted(primary + session_out, key=lambda row: (row["evaluation"], row["heldout_entry_revealed_after_scoring"])))

    ablations = {name: summarize(evaluate(candidates, controls, manifest, blocks, f"LOTO_{name}")) for name, blocks in FAMILY_BLOCKS.items()}
    ablations["LEVEL_STATE_WITHOUT_SAFETY"] = {
        **ablations["FULL_STATE"],
        "total_emitted_signals": ablations["FULL_STATE"]["total_emitted_without_safety"],
        "signals_per_manual_trade": ablations["FULL_STATE"]["total_emitted_without_safety"] / 9,
    }
    ablations["LEVEL_STATE_WITH_SAFETY"] = ablations["FULL_STATE"]
    _write_json(output_root / "ablation_results.json", ablations)

    confusion = state_confusion_diagnostic(candidates)
    _write_json(output_root / "state_confusion_diagnostic.json", confusion)
    m1_baseline = {
        "M1_FULL": {"top_5pct": 3 / 9, "correct_side_rate": 7 / 9, "median_percentile_rank": 0.7363667940257034, "total_emitted_signals": 16143},
        "M1_SR_ONLY": {"top_5pct": 0.0, "correct_side_rate": 4 / 9, "median_percentile_rank": 0.7749218478638416, "total_emitted_signals": 12601},
        "matched_hard_negative": _m1_matched_baseline(repo_root, candidates, controls, manifest),
    }

    rng = random.Random(SEED)
    label_results, state_results = [], []
    for draw in range(5):
        shuffled_candidates, shuffled_manifest = _shuffle_labels(candidates, manifest, rng, draw)
        shuffled_controls = build_matched_controls(shuffled_candidates, shuffled_manifest)
        label_predictions = evaluate(shuffled_candidates, shuffled_controls, shuffled_manifest, FAMILY_BLOCKS["FULL_STATE"], f"LABEL_SHUFFLE_{draw}")
        label_results.append({"draw": draw, **summarize(label_predictions), "pipeline_retrained": True})

        state_candidates = _shuffle_states(candidates, rng)
        state_by_id = {row["candidate_id"]: row for row in state_candidates}
        state_controls = [{**row, "features": state_by_id[row["candidate_id"]]["features"], "state_family": state_by_id[row["candidate_id"]]["model_inferred_family"]} for row in controls]
        state_predictions = evaluate(state_candidates, state_controls, manifest, FAMILY_BLOCKS["FULL_STATE"], f"STATE_SHUFFLE_{draw}")
        state_results.append({"draw": draw, **summarize(state_predictions), "pipeline_retrained": True})
    negative_controls = {"seed": SEED, "label_shuffle": label_results, "state_shuffle": state_results, "model_reused": False}
    _write_json(output_root / "negative_controls.json", negative_controls)

    prediction_by_trade = {row["manual_trade_id"]: row for row in primary}
    annotations_by_trade = {row["trade_id"]: row for row in annotations["annotations"]}
    anchor_frames = {row["manual_trade_id"]: row for row in candidates if row["selected"]}
    dossiers = []
    for trade in manifest:
        frame = anchor_frames[trade["manual_trade_id"]]
        toward = frame["approach_support"] if "SUPPORT" in frame["model_inferred_family"] else frame["approach_resistance"]
        destination = "support" if "SUPPORT" in frame["model_inferred_family"] else "resistance"
        distance = frame["distance_to_support_bps"] if destination == "support" else frame["distance_to_resistance_bps"]
        phrase = (
            f"Price was {distance:.1f} bps from {destination}; 15m distance reduction fraction={toward['15']['distance_reduction_fraction']:.3f}, "
            f"velocity={toward['15']['velocity_bps_per_min']:.2f} bps/min, path efficiency={toward['15']['path_efficiency']:.3f}, "
            f"state={frame['model_inferred_state']}, safety={frame['trade_safety_score']:.3f}."
        )
        dossiers.append({
            "dossier": "LEVEL_STATE_DOSSIER", "manual_trade_id": trade["manual_trade_id"], "entry_at_utc": trade["entry_at_utc"],
            "manual_side": trade["side"], "manual_annotation": annotations_by_trade[trade["manual_trade_id"]],
            "model_inferred_state": frame["model_inferred_state"], "model_inferred_family": frame["model_inferred_family"],
            "support_price": frame["support_price"], "resistance_price": frame["resistance_price"],
            "position_between_levels_raw": frame["position_between_levels_raw"],
            "distance_to_support_bps": frame["distance_to_support_bps"], "distance_to_resistance_bps": frame["distance_to_resistance_bps"],
            "approach_support": frame["approach_support"], "approach_resistance": frame["approach_resistance"],
            "momentum_toward_support": frame["momentum_toward_support"], "momentum_toward_resistance": frame["momentum_toward_resistance"],
            "deceleration_near_support": frame["deceleration_near_support"], "deceleration_near_resistance": frame["deceleration_near_resistance"],
            "reaction": frame["reaction"], "potential_action": frame["potential_action"], "final_action": frame["final_action"],
            "safety": {key: frame[key] for key in ("state_confidence", "level_clarity_score", "room_score", "momentum_consistency_score", "reaction_confirmation_score", "btc_safety_score", "volatility_safety_score", "trade_safety_score", "safety_veto_reason")},
            "ranking": prediction_by_trade[trade["manual_trade_id"]], "visual_language": phrase,
            "post_entry_information_used": False,
        })
    _write_jsonl(output_root / "level_state_dossiers.jsonl.gz", dossiers)

    metrics = summarize(primary)
    m1_matched = m1_baseline["matched_hard_negative"]["median_matched_hard_negative_percentile"]
    matched_improved = metrics["median_matched_hard_negative_percentile"] is not None and m1_matched is not None and metrics["median_matched_hard_negative_percentile"] > m1_matched
    shuffles_fail = max(row["top_5pct"] for row in label_results + state_results) < metrics["top_5pct"]
    improvement = (
        metrics["top_5pct"] >= 6 / 9
        and metrics["correct_side_rate"] >= 8 / 9
        and metrics["median_percentile_rank"] > m1_baseline["M1_FULL"]["median_percentile_rank"]
        and metrics["total_emitted_signals"] <= 0.10 * m1_baseline["M1_FULL"]["total_emitted_signals"]
        and matched_improved and shuffles_fail
    )
    inferred_counts = Counter(row["model_inferred_family"] for row in anchor_frames.values())
    manual_annotation = annotations_by_trade["MVDR-M1-09"]
    manual_state_match = anchor_frames["MVDR-M1-09"]["model_inferred_state"] == manual_annotation["manual_state"]
    summary = {
        "STATUS": STATUS, "program": PROGRAM, "phase": PHASE,
        "labels": ["RETROSPECTIVE_DISCOVERY_ONLY", "POST_M1_HYPOTHESIS", "NO_VALIDATION_AUTHORITY", "NO_PROFITABILITY_AUTHORITY"],
        "interpretation": "LEVEL_TO_LEVEL_STATE_STRONG_RETROSPECTIVE_REPLICATION" if improvement else "LEVEL_TO_LEVEL_STATE_INSUFFICIENT",
        "flags": {
            "CAUSAL_LEVEL_STATE_RECONSTRUCTION_COMPLETE": True,
            "MANUAL_STATE_ANNOTATION_AVAILABLE": True,
            "LEVEL_STATE_REPLICATION_IMPROVEMENT": improvement,
            "PROSPECTIVE_CAPTURE_RESEARCH_JUSTIFIED": improvement,
            "FULL_AUTOMATION_RESEARCH_JUSTIFIED": False,
        },
        "primary_metrics": metrics, "leave_session_out_metrics": summarize(session_out),
        "m1_frozen_baseline": m1_baseline, "inferred_manual_anchor_family_counts": dict(sorted(inferred_counts.items())),
        "manual_annotation_evaluation": {"trade_id": "MVDR-M1-09", "manual_state": manual_annotation["manual_state"], "model_inferred_state": anchor_frames["MVDR-M1-09"]["model_inferred_state"], "match": manual_state_match, "other_trades_manual_state_unannotated": 8},
        "safety_comparison": {"without_safety_emitted": metrics["total_emitted_without_safety"], "with_safety_emitted": metrics["total_emitted_signals"], "reduction": metrics["safety_signal_reduction"]},
        "criterion_components": {"top5_ge_6_of_9": metrics["top_5pct"] >= 6 / 9, "correct_side_ge_8_of_9": metrics["correct_side_rate"] >= 8 / 9, "median_rank_beats_m1_full": metrics["median_percentile_rank"] > m1_baseline["M1_FULL"]["median_percentile_rank"], "signals_reduced_90pct": metrics["total_emitted_signals"] <= 0.10 * m1_baseline["M1_FULL"]["total_emitted_signals"], "matched_discrimination_improved": matched_improved, "shuffles_do_not_reproduce": shuffles_fail},
        "anti_leakage": {"m1_market_hashes_verified": True, "completed_1m_only": True, "future_features": False, "outcomes_used": False, "candidate_generator_label_blind": True, "fold_normalization": True, "heldout_anchor_excluded_from_training": True},
        "production_modified": False, "exits_studied": False, "economics_studied": False,
    }
    _write_json(output_root / "diagnostic_summary.json", summary)
    artifacts = sorted(path for path in output_root.iterdir() if path.name != "m1_1_manifest.json")
    _write_json(output_root / "m1_1_manifest.json", {"program": PROGRAM, "phase": PHASE, "m1_authority": M1_AUTHORITY, "artifacts": {path.name: {"sha256": _sha(path), "bytes": path.stat().st_size} for path in artifacts}})
    return summary
