from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROGRAM = "MANUAL_VISUAL_DECISION_REPLICATION_V1"
PHASE = "MVDR_V1_M1_CAUSAL_ENTRY_RECONSTRUCTION"
SEED = 20260826
SOURCE_SHA256 = "6f7ce034ce790b3a806bcb1c0bd2c0d424700ac5eda659d0474f93edcbc9e463"
SESSION_BOUNDS = {
    "2026-08-25": ("2026-08-25T00:00:00Z", "2026-08-26T00:00:00Z"),
    "2026-08-26": ("2026-08-26T00:00:00Z", "2026-08-26T04:00:00Z"),
}
ANCHORS = (
    (1, "LONG", "2026-08-25T00:05:37Z", 0.7970, "2026-08-25T00:08:47Z", 0.8028472977781831),
    (3, "SHORT", "2026-08-25T02:19:55Z", 0.8278, "2026-08-25T07:49:42Z", 0.8067768208235438),
    (5, "SHORT", "2026-08-25T10:21:17Z", 0.8025, "2026-08-25T10:36:04Z", 0.7995),
    (9, "SHORT", "2026-08-25T13:46:38Z", 0.7879, "2026-08-25T13:51:20Z", 0.7797),
    (10, "LONG", "2026-08-25T14:02:35Z", 0.7903, "2026-08-25T14:34:00Z", 0.7979350263536177),
    (13, "LONG", "2026-08-25T21:33:30Z", 0.7577, "2026-08-25T22:07:59Z", 0.7658),
    (14, "SHORT", "2026-08-25T22:15:53Z", 0.7661, "2026-08-25T23:41:45Z", 0.7586),
    (15, "LONG", "2026-08-26T00:10:35Z", 0.7530, "2026-08-26T01:27:15Z", 0.7597),
    (16, "SHORT", "2026-08-26T01:31:04Z", 0.7605, "2026-08-26T03:32:33Z", 0.7579),
)
OTHER_STRATEGY_RECORDS = (2, 4, 6, 7, 8, 11, 12)
FEATURE_BLOCKS = ("geometry", "sequence", "momentum", "rejection", "compression", "volume", "mtf", "btc", "sr")


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class Bar:
    open_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float


def build_manual_manifest(source: Path) -> list[dict[str, Any]]:
    if _sha(source) != SOURCE_SHA256:
        raise RuntimeError("TRADE_SOURCE_SHA256_MISMATCH")
    records = json.loads(source.read_text(encoding="utf-8"))
    result = []
    for sequence, (record_no, side, entry_at, entry_price, exit_at, exit_price) in enumerate(ANCHORS, 1):
        raw = records[record_no - 1]
        if raw["side"] != side or abs(float(raw["entry_price"]) - entry_price) > 5e-7 or abs(float(raw["exit_price"]) - exit_price) > 5e-7:
            raise RuntimeError(f"TRADE_IDENTITY_MISMATCH:{record_no}")
        if raw["entry_time"].replace(" ", "T") + "Z" != entry_at or raw["exit_time"].replace(" ", "T") + "Z" != exit_at:
            raise RuntimeError(f"TRADE_TIMESTAMP_MISMATCH:{record_no}")
        result.append({
            "manual_trade_id": f"MVDR-M1-{sequence:02d}", "symbol": "SUIUSDT", "side": side,
            "entry_at_utc": entry_at, "entry_price": float(raw["entry_price"]),
            "exit_at_utc": exit_at, "exit_price": float(raw["exit_price"]),
            "source_file": str(source), "source_record_id": f"json-array-record-{record_no}",
            "source_sha256": SOURCE_SHA256,
        })
    return result


def fetch_klines(symbol: str, start: datetime, end: datetime) -> list[Bar]:
    rows: list[list[Any]] = []
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor < end_ms:
        query = urllib.parse.urlencode({"symbol": symbol, "interval": "1m", "startTime": cursor, "endTime": end_ms - 1, "limit": 1500})
        request = urllib.request.Request(f"https://fapi.binance.com/fapi/v1/klines?{query}", headers={"User-Agent": "MVDR-V1-research"})
        with urllib.request.urlopen(request, timeout=30) as response:
            batch = json.loads(response.read())
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 60_000
    unique = {int(row[0]): row for row in rows if int(row[0]) < end_ms}
    return [Bar(datetime.fromtimestamp(ms / 1000, timezone.utc), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[9])) for ms, row in sorted(unique.items())]


def audit_bars(symbol: str, bars: list[Bar], start: datetime, end: datetime) -> dict[str, Any]:
    expected = int((end - start).total_seconds() // 60)
    timestamps = [bar.open_at for bar in bars]
    gaps = [_iso(start + timedelta(minutes=i)) for i in range(expected) if start + timedelta(minutes=i) not in set(timestamps)]
    return {
        "symbol": symbol, "interval": "1m", "source": "Binance USD-M Futures REST /fapi/v1/klines",
        "start_utc": _iso(start), "end_exclusive_utc": _iso(end), "rows": len(bars), "expected_rows": expected,
        "gaps": gaps, "duplicates": len(timestamps) - len(set(timestamps)),
        "timestamp_convention": "open_at UTC; available_at=open_at+60s; only completed bars used",
    }


def _aggregate(completed: list[Bar], minutes: int) -> list[Bar]:
    buckets: dict[datetime, list[Bar]] = defaultdict(list)
    for bar in completed:
        minute = (bar.open_at.minute // minutes) * minutes
        bucket = bar.open_at.replace(minute=minute, second=0, microsecond=0)
        buckets[bucket].append(bar)
    return [Bar(key, values[0].open, max(v.high for v in values), min(v.low for v in values), values[-1].close, sum(v.volume for v in values), sum(v.buy_volume for v in values)) for key, values in sorted(buckets.items())]


def _safe(a: float, b: float) -> float:
    return a / b if b else 0.0


def _series_features(bars: list[Bar], prefix: str, block_prefix: str) -> dict[str, float]:
    if len(bars) < 8:
        return {}
    last = bars[-1]
    candle_range = max(last.high - last.low, 1e-12)
    body = last.close - last.open
    out = {
        f"{block_prefix}.body_signed": body / last.open,
        f"{block_prefix}.body_range": abs(body) / candle_range,
        f"{block_prefix}.upper_wick": (last.high - max(last.open, last.close)) / candle_range,
        f"{block_prefix}.lower_wick": (min(last.open, last.close) - last.low) / candle_range,
        f"{block_prefix}.close_position": (last.close - last.low) / candle_range,
    }
    for n in (2, 3, 5, 8):
        seq = bars[-n:]
        changes = [seq[i].close - (seq[i - 1].close if i else seq[i].open) for i in range(n)]
        traveled = sum(abs(v) for v in changes)
        displacement = seq[-1].close - seq[0].open
        ranges = [v.high - v.low for v in seq]
        bodies = [abs(v.close - v.open) for v in seq]
        overlaps = [max(0.0, min(seq[i].high, seq[i - 1].high) - max(seq[i].low, seq[i - 1].low)) / max(seq[i].high - seq[i].low, 1e-12) for i in range(1, n)]
        stem = f"{prefix}.n{n}"
        out.update({
            f"sequence.{stem}.bull_fraction": sum(v > 0 for v in changes) / n,
            f"sequence.{stem}.higher_closes": sum(seq[i].close > seq[i - 1].close for i in range(1, n)) / max(1, n - 1),
            f"sequence.{stem}.higher_highs": sum(seq[i].high > seq[i - 1].high for i in range(1, n)) / max(1, n - 1),
            f"sequence.{stem}.higher_lows": sum(seq[i].low > seq[i - 1].low for i in range(1, n)) / max(1, n - 1),
            f"sequence.{stem}.overlap": sum(overlaps) / max(1, len(overlaps)),
            f"sequence.{stem}.alternation": sum(changes[i] * changes[i - 1] < 0 for i in range(1, n)) / max(1, n - 1),
            f"momentum.{stem}.displacement": displacement / seq[0].open,
            f"momentum.{stem}.path_efficiency": abs(displacement) / traveled if traveled else 0.0,
            f"momentum.{stem}.travel": traveled / seq[0].open,
            f"momentum.{stem}.impulse_body_dominance": sum(bodies) / max(sum(ranges), 1e-12),
            f"compression.{stem}.range_ratio": _safe(sum(ranges[-max(1, n // 2):]) / max(1, math.ceil(n / 2)), sum(ranges[:max(1, n // 2)]) / max(1, n // 2)),
        })
    vols = [v.volume for v in bars[-8:]]
    out[f"volume.{prefix}.relative"] = _safe(last.volume, sum(vols[:-1]) / 7)
    out[f"volume.{prefix}.buy_ratio"] = _safe(last.buy_volume, last.volume)
    out[f"rejection.{prefix}.bull"] = out[f"{block_prefix}.lower_wick"] * (1.0 if last.close >= last.open else 0.5)
    out[f"rejection.{prefix}.bear"] = out[f"{block_prefix}.upper_wick"] * (1.0 if last.close <= last.open else 0.5)
    return out


def _sr_features(bars_1m: list[Bar], bars_5m: list[Bar], bars_15m: list[Bar], price: float) -> dict[str, float]:
    recent = bars_1m[-60:]
    lows = [v.low for v in recent]
    highs = [v.high for v in recent]
    swing_support = min(lows[-20:])
    swing_resistance = max(highs[-20:])
    rounded = defaultdict(int)
    tick = max(price * 0.0005, 1e-8)
    for value in lows + highs:
        rounded[round(value / tick) * tick] += 1
    clusters = sorted(rounded.items(), key=lambda item: (-item[1], abs(item[0] - price)))
    below = [level for level, _ in clusters if level <= price]
    above = [level for level, _ in clusters if level >= price]
    cluster_support = below[0] if below else min(lows)
    cluster_resistance = above[0] if above else max(highs)
    mtf_lows = [v.low for v in bars_5m[-12:]] + [v.low for v in bars_15m[-8:]]
    mtf_highs = [v.high for v in bars_5m[-12:]] + [v.high for v in bars_15m[-8:]]
    mtf_support, mtf_resistance = min(mtf_lows), max(mtf_highs)
    out: dict[str, float] = {}
    for family, support, resistance in (("swing", swing_support, swing_resistance), ("cluster", cluster_support, cluster_resistance), ("mtf_extrema", mtf_support, mtf_resistance)):
        out[f"sr.{family}.support_distance"] = (price - support) / price
        out[f"sr.{family}.resistance_distance"] = (resistance - price) / price
        out[f"sr.{family}.range_position"] = _safe(price - support, resistance - support)
        out[f"sr.{family}.space"] = (resistance - support) / price
        out[f"sr.{family}.support_touches"] = sum(abs(v.low - support) <= tick for v in recent) / 60
        out[f"sr.{family}.resistance_touches"] = sum(abs(v.high - resistance) <= tick for v in recent) / 60
    out["sr.agreement_support"] = max(0.0, 1.0 - (max(swing_support, cluster_support, mtf_support) - min(swing_support, cluster_support, mtf_support)) / (price * 0.005))
    out["sr.agreement_resistance"] = max(0.0, 1.0 - (max(swing_resistance, cluster_resistance, mtf_resistance) - min(swing_resistance, cluster_resistance, mtf_resistance)) / (price * 0.005))
    return out


def visual_frame(decision_at: datetime, sui: list[Bar], btc: list[Bar]) -> dict[str, Any]:
    sui_done = [v for v in sui if v.open_at + timedelta(minutes=1) <= decision_at]
    btc_done = [v for v in btc if v.open_at + timedelta(minutes=1) <= decision_at]
    if len(sui_done) < 120 or len(btc_done) < 120:
        raise ValueError("INSUFFICIENT_CAUSAL_WARMUP")
    features: dict[str, float] = {}
    aggregates: dict[str, dict[int, list[Bar]]] = {"sui": {}, "btc": {}}
    for name, source in (("sui", sui_done), ("btc", btc_done)):
        for tf in (3, 5, 15):
            bars = _aggregate(source[-240:], tf)
            aggregates[name][tf] = bars
            block = "geometry" if name == "sui" and tf == 3 else ("mtf" if name == "sui" else "btc")
            features.update(_series_features(bars, f"{name}_{tf}m", block))
    price = sui_done[-1].close
    features.update(_sr_features(sui_done, aggregates["sui"][5], aggregates["sui"][15], price))
    sui_disp = features.get("momentum.sui_3m.n3.displacement", 0.0)
    btc_disp = features.get("momentum.btc_3m.n3.displacement", 0.0)
    features["btc.relation.aligned"] = 1.0 if sui_disp * btc_disp > 0 else 0.0
    features["btc.relation.opposed"] = 1.0 if sui_disp * btc_disp < 0 else 0.0
    features["btc.relation.calm"] = 1.0 if abs(btc_disp) < 0.001 else 0.0
    return {"frame_id": _stable_id("frame", _iso(decision_at)), "decision_at_utc": _iso(decision_at), "latest_completed_1m_open_at": _iso(sui_done[-1].open_at), "current_market_price": price, "features": features}


def _side_features(features: dict[str, float], side: str) -> dict[str, float]:
    direction = 1.0 if side == "LONG" else -1.0
    result = dict(features)
    result["direction.side"] = direction
    for key in list(result):
        if key.endswith("displacement") or key.endswith("body_signed"):
            result[key] *= direction
    result["sr.nearest_level_distance"] = min(features["sr.swing.support_distance"], features["sr.cluster.support_distance"], features["sr.mtf_extrema.support_distance"]) if side == "LONG" else min(features["sr.swing.resistance_distance"], features["sr.cluster.resistance_distance"], features["sr.mtf_extrema.resistance_distance"])
    result["rejection.side"] = features.get("rejection.sui_3m.bull" if side == "LONG" else "rejection.sui_3m.bear", 0.0)
    return result


def generate_universe(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        for side in ("LONG", "SHORT"):
            features = _side_features(frame["features"], side)
            hard = features["sr.nearest_level_distance"] <= 0.0025 or features["rejection.side"] >= 0.40 or abs(features.get("momentum.sui_3m.n3.displacement", 0.0)) >= 0.003
            rows.append({"candidate_id": _stable_id(frame["frame_id"], side), "frame_id": frame["frame_id"], "decision_at_utc": frame["decision_at_utc"], "session": frame["decision_at_utc"][:10], "side": side, "hard_negative_structural": hard, "features": features})
    return rows


def apply_labels(universe: list[dict[str, Any]], manifest: list[dict[str, Any]], other_entries: list[datetime]) -> list[dict[str, Any]]:
    positives = {(item["entry_at_utc"][:16], item["side"]): item["manual_trade_id"] for item in manifest}
    manual_times = [_dt(item["entry_at_utc"]) for item in manifest]
    result = []
    for source in universe:
        row = dict(source)
        decision = _dt(row["decision_at_utc"])
        selected_id = positives.get((row["decision_at_utc"][:16], row["side"]))
        manual_blackout = any(abs((decision - value).total_seconds()) <= 6 * 60 for value in manual_times)
        other_strategy_blackout = any(abs((decision - value).total_seconds()) <= 6 * 60 for value in other_entries)
        row.update({
            "selected": bool(selected_id), "manual_trade_id": selected_id,
            "manual_entry_blackout": manual_blackout, "other_strategy_exclusion": other_strategy_blackout,
            "hard_negative_eligible": row["hard_negative_structural"] and not manual_blackout and not other_strategy_blackout,
            "label_semantics": "DID_TRADER_SELECT_THIS_ENTRY",
        })
        result.append(row)
    return result


def _feature_names(rows: list[dict[str, Any]], blocks: set[str]) -> list[str]:
    return sorted(key for key in rows[0]["features"] if key.split(".", 1)[0] in blocks)


def _fit_scaler(rows: list[dict[str, Any]], names: list[str]) -> tuple[list[float], list[float]]:
    means, scales = [], []
    for name in names:
        values = [float(row["features"].get(name, 0.0)) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        means.append(mean)
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def _vector(row: dict[str, Any], names: list[str], means: list[float], scales: list[float]) -> list[float]:
    return [(float(row["features"].get(name, 0.0)) - means[i]) / scales[i] for i, name in enumerate(names)]


def _fit_logistic(rows: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    means, scales = _fit_scaler(rows, names)
    weights = [0.0] * (len(names) + 1)
    positives = sum(row["selected"] for row in rows)
    positive_weight = max(1.0, (len(rows) - positives) / max(1, positives))
    for step in range(60):
        gradient = [0.0] * len(weights)
        for row in rows:
            x = [1.0] + _vector(row, names, means, scales)
            z = max(-30.0, min(30.0, sum(w * v for w, v in zip(weights, x))))
            prediction = 1.0 / (1.0 + math.exp(-z))
            sample_weight = positive_weight if row["selected"] else 1.0
            for i, value in enumerate(x):
                gradient[i] += sample_weight * (prediction - int(row["selected"])) * value
        rate = 0.08 / math.sqrt(step + 1)
        denominator = max(1, len(rows))
        for i in range(len(weights)):
            regularization = 0.002 * (1.0 if weights[i] > 0 else -1.0 if weights[i] < 0 else 0.0) if i else 0.0
            weights[i] -= rate * (gradient[i] / denominator + regularization)
    return {"family": "sparse_logistic", "names": names, "means": means, "scales": scales, "weights": weights}


def _score(model: dict[str, Any], row: dict[str, Any]) -> float:
    x = [1.0] + _vector(row, model["names"], model["means"], model["scales"])
    z = max(-30.0, min(30.0, sum(w * v for w, v in zip(model["weights"], x))))
    return 1.0 / (1.0 + math.exp(-z))


def _train_rows(universe: list[dict[str, Any]], heldout_id: str, strict_session: bool = False) -> list[dict[str, Any]]:
    heldout = next(row for row in universe if row["manual_trade_id"] == heldout_id)
    positives = [row for row in universe if row["manual_trade_id"] != heldout_id and row["selected"] and (not strict_session or row["session"] != heldout["session"])]
    negatives = [row for row in universe if row["hard_negative_eligible"] and row["session"] != heldout["session"]]
    negatives.sort(key=lambda row: row["candidate_id"])
    return positives + negatives[:600]


def evaluate_loto(universe: list[dict[str, Any]], manifest: list[dict[str, Any]], blocks: set[str], label: str, strict_session: bool = False) -> list[dict[str, Any]]:
    predictions = []
    for trade in manifest:
        train = _train_rows(universe, trade["manual_trade_id"], strict_session)
        names = _feature_names(train, blocks)
        model = _fit_logistic(train, names)
        test = [row for row in universe if row["session"] == trade["entry_at_utc"][:10]]
        scored = [(row, _score(model, row)) for row in test]
        scored.sort(key=lambda item: (-item[1], item[0]["candidate_id"]))
        target_minute = trade["entry_at_utc"][:16]
        target = next((item for item in scored if item[0]["manual_trade_id"] == trade["manual_trade_id"]), None)
        if target is None:
            raise RuntimeError(f"HELDOUT_TARGET_MISSING:{trade['manual_trade_id']}")
        rank = next(index for index, item in enumerate(scored, 1) if item[0]["candidate_id"] == target[0]["candidate_id"])
        same_side = [item for item in scored if item[0]["side"] == trade["side"]]
        side_rank = next(index for index, item in enumerate(same_side, 1) if item[0]["candidate_id"] == target[0]["candidate_id"])
        train_scores = sorted((_score(model, row) for row in train if row["hard_negative_eligible"]), reverse=True)
        sessions = max(1, len({row["session"] for row in train}))
        threshold_index = min(len(train_scores) - 1, 5 * sessions - 1) if train_scores else 0
        threshold = train_scores[threshold_index] if train_scores else 1.0
        emitted = [item for item in scored if item[1] >= threshold]
        opposite = next(item for item in scored if item[0]["decision_at_utc"][:16] == target_minute and item[0]["side"] != trade["side"])
        captures = {}
        emitted_correct = [item for item in emitted if item[0]["side"] == trade["side"]]
        for tolerance in (1, 3, 6):
            captures[str(tolerance)] = any(abs((_dt(item[0]["decision_at_utc"]) - _dt(trade["entry_at_utc"])).total_seconds()) <= tolerance * 60 for item in emitted_correct)
        nearest_error = min((abs((_dt(item[0]["decision_at_utc"]) - _dt(trade["entry_at_utc"])).total_seconds()) / 60 for item in emitted_correct), default=None)
        predictions.append({
            "evaluation": label, "manual_trade_id": trade["manual_trade_id"], "heldout_entry_revealed_after_scoring": trade["entry_at_utc"],
            "score": target[1], "rank_all": rank, "rank_side": side_rank, "candidate_count": len(scored),
            "percentile_rank": 1.0 - (rank - 1) / max(1, len(scored) - 1), "threshold": threshold,
            "emitted_signals": len(emitted), "signals_per_day": len(emitted), "capture": captures,
            "nearest_correct_side_signal_error_minutes": nearest_error, "correct_side": target[1] > opposite[1],
            "normalization_scope": "training_fold_only", "threshold_scope": "training_fold_only",
        })
    return predictions


def _summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(predictions)
    percentiles = sorted(row["percentile_rank"] for row in predictions)
    errors = sorted(row["nearest_correct_side_signal_error_minutes"] for row in predictions if row["nearest_correct_side_signal_error_minutes"] is not None)
    median = lambda values: values[len(values) // 2] if values else None
    emitted = sum(row["emitted_signals"] for row in predictions)
    return {
        "N": n, "capture_rate_1m": sum(row["capture"]["1"] for row in predictions) / n,
        "capture_rate_3m": sum(row["capture"]["3"] for row in predictions) / n,
        "capture_rate_6m": sum(row["capture"]["6"] for row in predictions) / n,
        "correct_side_rate": sum(row["correct_side"] for row in predictions) / n,
        "median_timing_error_minutes": median(errors), "median_percentile_rank": median(percentiles),
        "top_1pct": sum(row["percentile_rank"] >= 0.99 for row in predictions) / n,
        "top_5pct": sum(row["percentile_rank"] >= 0.95 for row in predictions) / n,
        "top_10pct": sum(row["percentile_rank"] >= 0.90 for row in predictions) / n,
        "total_emitted_signals": emitted, "signals_per_manual_trade": emitted / n,
        "replication_precision_proxy_6m": sum(row["capture"]["6"] for row in predictions) / max(1, emitted),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as zipped:
        for row in rows:
            zipped.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())


def execute_m1(repo_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    source = repo_root / "backtest_results/real_trade_analysis.json"
    manifest = build_manual_manifest(source)
    _write_json(output_root / "manual_entry_manifest.json", {"program": PROGRAM, "phase": PHASE, "timezone_assumption": "UTC corroborated by runtime logs and hour_utc analysis", "excluded_other_strategy_source_records": list(OTHER_STRATEGY_RECORDS), "trades": manifest})

    load_start = _dt("2026-08-24T20:00:00Z")
    load_end = _dt("2026-08-26T04:00:00Z")
    market: dict[str, list[Bar]] = {}
    market_manifest = {"source": "Binance USD-M Futures REST", "downloaded_at_utc": _iso(datetime.now(timezone.utc)), "coverage": [], "anti_leakage": {"base_interval": "1m", "completed_bars_only": True}}
    for symbol in ("SUIUSDT", "BTCUSDT"):
        bars = fetch_klines(symbol, load_start, load_end)
        audit = audit_bars(symbol, bars, load_start, load_end)
        if audit["gaps"] or audit["duplicates"] or audit["rows"] != audit["expected_rows"]:
            raise RuntimeError(f"MARKET_DATA_INTEGRITY_FAILED:{symbol}")
        market[symbol] = bars
        raw_path = output_root / f"{symbol}_1m.jsonl.gz"
        _write_jsonl_gz(raw_path, ({"open_at_utc": _iso(v.open_at), "open": v.open, "high": v.high, "low": v.low, "close": v.close, "volume": v.volume, "taker_buy_volume": v.buy_volume} for v in bars))
        audit["artifact"] = raw_path.name
        audit["sha256"] = _sha(raw_path)
        market_manifest["coverage"].append(audit)
    _write_json(output_root / "market_data_manifest.json", market_manifest)

    frames = []
    for _, (start_text, end_text) in SESSION_BOUNDS.items():
        cursor = _dt(start_text)
        end = _dt(end_text)
        while cursor < end:
            try:
                frames.append(visual_frame(cursor, market["SUIUSDT"], market["BTCUSDT"]))
            except ValueError:
                pass
            cursor += timedelta(minutes=1)
    unlabeled_universe = generate_universe(frames)
    all_trade_records = json.loads(source.read_text(encoding="utf-8"))
    other_entries = [_dt(all_trade_records[index - 1]["entry_time"].replace(" ", "T") + "Z") for index in OTHER_STRATEGY_RECORDS]
    universe = apply_labels(unlabeled_universe, manifest, other_entries)
    _write_jsonl_gz(output_root / "visual_frames.jsonl.gz", frames)
    _write_jsonl_gz(output_root / "candidate_decisions.jsonl.gz", universe)

    full = evaluate_loto(universe, manifest, set(FEATURE_BLOCKS), "LOTO_AUTOMATIC_CAUSAL_SR")
    leave_session_out = evaluate_loto(universe, manifest, set(FEATURE_BLOCKS), "LEAVE_SESSION_OUT_AUTOMATIC_CAUSAL_SR", strict_session=True)
    _write_jsonl_gz(output_root / "loto_predictions.jsonl.gz", full)
    _write_jsonl_gz(output_root / "session_rankings.jsonl.gz", sorted(full + leave_session_out, key=lambda row: (row["evaluation"], row["heldout_entry_revealed_after_scoring"][:10], row["rank_all"])))

    variants = {
        "SR_ONLY": {"sr"}, "REJECTION_ONLY": {"rejection"}, "VISUAL_SEQUENCE": {"geometry", "sequence", "momentum", "rejection", "compression", "volume"},
        "VISUAL_SEQUENCE_MTF": {"geometry", "sequence", "momentum", "rejection", "compression", "volume", "mtf", "sr"},
        "FULL": set(FEATURE_BLOCKS), "WITHOUT_BTC": set(FEATURE_BLOCKS) - {"btc"}, "WITHOUT_SR": set(FEATURE_BLOCKS) - {"sr"},
        "WITHOUT_SEQUENCE": set(FEATURE_BLOCKS) - {"sequence", "momentum"}, "WITHOUT_VOLUME": set(FEATURE_BLOCKS) - {"volume"}, "WITHOUT_MTF": set(FEATURE_BLOCKS) - {"mtf"},
    }
    ablations = {name: _summary(evaluate_loto(universe, manifest, blocks, name)) for name, blocks in variants.items()}
    ablations["RANDOM_RANKING_EXPECTATION"] = {"median_percentile_rank": 0.5, "top_5pct": 0.05}
    _write_json(output_root / "ablation_results.json", ablations)

    rng = random.Random(SEED)
    controls = []
    for draw in range(10):
        shuffled = []
        for session in SESSION_BOUNDS:
            candidates = [row for row in universe if row["session"] == session]
            for side in ("LONG", "SHORT"):
                count = sum(item["entry_at_utc"][:10] == session and item["side"] == side for item in manifest)
                pool = [row for row in candidates if row["side"] == side]
                for row in rng.sample(pool, count):
                    shuffled.append({**row, "selected": True, "manual_trade_id": f"SHUFFLE-{draw:02d}-{len(shuffled):02d}"})
        control_manifest = [{"manual_trade_id": row["manual_trade_id"], "entry_at_utc": row["decision_at_utc"], "side": row["side"]} for row in shuffled]
        control_universe = [{**row, "selected": False, "manual_trade_id": None} for row in universe]
        by_id = {row["candidate_id"]: row for row in control_universe}
        for selected in shuffled:
            by_id[selected["candidate_id"]]["selected"] = True
            by_id[selected["candidate_id"]]["manual_trade_id"] = selected["manual_trade_id"]
        try:
            result = evaluate_loto(control_universe, control_manifest, set(FEATURE_BLOCKS), f"LABEL_SHUFFLE_{draw:02d}")
            controls.append({"draw": draw, **_summary(result)})
        except (RuntimeError, ZeroDivisionError):
            controls.append({"draw": draw, "status": "NOT_COMPUTABLE"})
    negative_controls = {"seed": SEED, "draws": len(controls), "preserves": ["same_day", "same_side_counts"], "retrained_each_draw": True, "results": controls}
    _write_json(output_root / "negative_controls.json", negative_controls)

    dossiers = []
    prediction_by_trade = {row["manual_trade_id"]: row for row in full}
    for trade in manifest:
        entry = _dt(trade["entry_at_utc"])
        snapshots = {}
        for minutes in (60, 30, 15, 5, 3, 0):
            frame = visual_frame(entry - timedelta(minutes=minutes), market["SUIUSDT"], market["BTCUSDT"])
            snapshots[f"minus_{minutes}m" if minutes else "entry"] = {"decision_at_utc": frame["decision_at_utc"], "current_market_price": frame["current_market_price"], "features": frame["features"]}
        dossiers.append({"dossier": "TRADE_VISUAL_RECONSTRUCTION", "manual_trade_id": trade["manual_trade_id"], "side": trade["side"], "entry_at_utc": trade["entry_at_utc"], "entry_price_reference": trade["entry_price"], "post_entry_information_used": False, "snapshots": snapshots, "ranking": prediction_by_trade[trade["manual_trade_id"]]})
    _write_jsonl_gz(output_root / "manual_trade_dossiers.jsonl.gz", dossiers)

    metrics = _summary(full)
    signal = metrics["top_5pct"] >= 0.70 and metrics["correct_side_rate"] >= 0.80 and (metrics["median_timing_error_minutes"] is not None and metrics["median_timing_error_minutes"] <= 6) and max(row["signals_per_day"] for row in full) <= 5 and metrics["median_percentile_rank"] > ablations["SR_ONLY"]["median_percentile_rank"] and metrics["median_percentile_rank"] > ablations["REJECTION_ONLY"]["median_percentile_rank"]
    summary = {
        "STATUS": "MVDR_V1_M1_CAUSAL_ENTRY_RECONSTRUCTION_READY_FOR_REVIEW", "program": PROGRAM, "phase": PHASE,
        "labels": ["RETROSPECTIVE_REPLICATION_DISCOVERY_ONLY", "NOT_PROFITABILITY_EVIDENCE"],
        "flags": {"TRADE_IDENTITY_COMPLETE": True, "CAUSAL_MARKET_REPLAY_COMPLETE": True, "VISUAL_FEATURE_RECONSTRUCTION_COMPLETE": True, "MANUAL_ENTRY_REPLICATION_SIGNAL_PRESENT": signal, "FULL_AUTOMATION_RESEARCH_JUSTIFIED": signal, "MANUAL_SR_REQUIRED": None},
        "manual_sr_oracle": {"available": False, "reason": "NO_RECORDED_MANUAL_LEVELS"}, "automatic_causal_sr": {"families": ["recent_swing_zones", "repeated_touch_clusters", "5m_15m_extrema"]},
        "primary_metrics": metrics, "preregistered_rule": "top5>=70%; side>=80%; median_error<=6m; <=5 signals/day; beats SR-only and rejection-only",
        "leave_session_out_metrics": _summary(leave_session_out), "session_count": len(SESSION_BOUNDS),
        "anti_leakage": {"feature_time_lte_decision": True, "completed_1m_only": True, "exit_features": False, "outcome_labels": False, "candidate_generator_label_blind": True, "manual_blackout_minutes": 6, "other_strategy_exclusion_minutes": 6, "fold_normalization": True, "fold_threshold": True},
        "production_modified": False, "phase_2_executed": False,
    }
    _write_json(output_root / "diagnostic_summary.json", summary)
    material = sorted(path for path in output_root.iterdir() if path.name != "m1_manifest.json")
    artifact_manifest = {"program": PROGRAM, "phase": PHASE, "artifacts": {path.name: {"sha256": _sha(path), "bytes": path.stat().st_size} for path in material}}
    _write_json(output_root / "m1_manifest.json", artifact_manifest)
    return summary
