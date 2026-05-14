#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT_FALLBACK = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FALLBACK))


DEFAULT_SYMBOLS = (
    "ETHUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "LTCUSDT",
)
DEFAULT_ENTRY_QUALITY_DATASET = "aegis_alpha/data/processed/entry_quality/entry_quality_dataset_v020.npz"
DEFAULT_OUTPUT = "aegis_alpha/data/processed/decision_brain/decision_brain_dataset_v010.npz"
DEFAULT_META_OUTPUT = "aegis_alpha/data/processed/decision_brain/decision_brain_dataset_v010_meta.json"
DEFAULT_REPORT_JSON = "aegis_alpha/logs/decision_brain/decision_brain_dataset_report_v010.json"
DEFAULT_REPORT_MD = "aegis_alpha/logs/decision_brain/decision_brain_dataset_report_v010.md"
TS_AEGIS_LOG_DIR = "binance-futures-bot-ts/logs/aegis"
PY_TURBO_LOG_DIR = "aegis_alpha/logs/turbo"
EVENT_SENTIMENT_LOG_DIR = "aegis_alpha/logs/event_risk"

LABELS = ("ENTER_NOW", "WAIT_CONFIRMATION", "MANUAL_ONLY", "DO_NOT_ENTER")
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
MODE_TO_SCORE = {
    "UNKNOWN": -1.0,
    "NORMAL": 0.0,
    "CAUTION": 1.0,
    "RISK_OFF": 2.0,
    "MANUAL_ONLY": 3.0,
}
ACTION_TO_SCORE = {
    "UNKNOWN": -1.0,
    "HOLD": 0.0,
    "LONG": 1.0,
    "SHORT": -1.0,
}
SIDE_TO_SCORE = {
    "UNKNOWN": 0.0,
    "LONG": 1.0,
    "SHORT": -1.0,
}
RECOMMENDATION_TO_SCORE = {
    "UNKNOWN": -1.0,
    "ALLOW_SHADOW": 1.0,
    "BLOCK_SHADOW": -1.0,
}
SOURCE_TO_SCORE = {
    "historical_replay": 0.0,
    "live_logs": 1.0,
}


@dataclass(frozen=True)
class BuildArgs:
    symbols: list[str]
    start: datetime | None
    end: datetime | None
    output: Path
    include_live_logs: bool
    include_historical_replay: bool


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace("/", "").replace("-", "")


def parse_dt(value: str | None) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.lower() == "now":
        return datetime.now(timezone.utc)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_any_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return parse_dt(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def in_range(ts: datetime | None, start: datetime | None, end: datetime | None) -> bool:
    if ts is None:
        return False
    if start is not None and ts < start:
        return False
    if end is not None and ts > end:
        return False
    return True


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(np.asarray(value).item())
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def mode_score(mode: Any) -> float:
    return MODE_TO_SCORE.get(str(mode or "UNKNOWN").upper(), -1.0)


def action_score(action: Any) -> float:
    return ACTION_TO_SCORE.get(str(action or "UNKNOWN").upper(), -1.0)


def side_score(side: Any) -> float:
    return SIDE_TO_SCORE.get(str(side or "UNKNOWN").upper(), 0.0)


def recommendation_score(value: Any) -> float:
    return RECOMMENDATION_TO_SCORE.get(str(value or "UNKNOWN").upper(), -1.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def read_jsonl_glob(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(glob.glob(pattern)):
        rows.extend(read_jsonl(Path(item)))
    return rows


def encode_hour_day(ts: datetime | None) -> tuple[float, float, float, float]:
    if ts is None:
        return 0.0, 1.0, 0.0, 1.0
    hour = ts.hour + ts.minute / 60.0
    dow = float(ts.weekday())
    return (
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
        math.sin(2.0 * math.pi * dow / 7.0),
        math.cos(2.0 * math.pi * dow / 7.0),
    )


def session_score(ts: datetime | None) -> float:
    if ts is None:
        return -1.0
    hour = ts.hour
    if 0 <= hour < 8:
        return 0.0
    if 8 <= hour < 13:
        return 1.0
    if 13 <= hour < 21:
        return 2.0
    return 3.0


def label_from_outcome(
    *,
    future_mfe_roe: float,
    future_mae_roe: float,
    time_to_green: float,
    final_roe: float,
    hit_profit_before_loss: float,
    hit_loss_15_before_profit_5: float,
    tail_risk_score: float,
    turbo_score: float,
    mtf_conflict: float,
    event_risk_score: float,
    news_mode_score: float,
    shadow_block: float,
) -> str:
    mfe = safe_float(future_mfe_roe)
    mae = safe_float(future_mae_roe)
    ttg = safe_float(time_to_green)
    final = safe_float(final_roe)
    hit_profit_first = safe_float(hit_profit_before_loss, 0.0) >= 0.5
    hit_loss_first = safe_float(hit_loss_15_before_profit_5, 0.0) >= 0.5
    tail = safe_float(tail_risk_score)
    score = safe_float(turbo_score)

    if hit_loss_first or mae <= -0.20 or (mae <= -0.15 and mfe < 0.05) or final <= -0.08:
        return "DO_NOT_ENTER"

    contaminated = (
        event_risk_score >= 1.0
        or news_mode_score >= 1.0
        or shadow_block >= 0.5
        or mtf_conflict >= 0.5
        or tail >= 0.55
    )
    if score >= 0.75 and contaminated and mfe >= 0.05 and final > -0.04:
        return "MANUAL_ONLY"

    if hit_profit_first and mae > -0.10 and (math.isnan(ttg) or ttg <= 30.0) and tail < 0.55:
        return "ENTER_NOW"

    if mfe >= 0.08 or final > 0.02:
        return "WAIT_CONFIRMATION"

    return "DO_NOT_ENTER"


def binary_labels(label: str, tail_risk_score: float, future_mae_roe: float) -> tuple[int, int, int, int]:
    enter_now = 1 if label == "ENTER_NOW" else 0
    do_not_enter = 1 if label == "DO_NOT_ENTER" else 0
    tail_risk = 1 if safe_float(tail_risk_score, 0.0) >= 0.55 or safe_float(future_mae_roe, 0.0) <= -0.15 else 0
    wait_needed = 1 if label in {"WAIT_CONFIRMATION", "MANUAL_ONLY"} else 0
    return enter_now, do_not_enter, tail_risk, wait_needed


def build_event_sentiment_timeline() -> list[tuple[datetime, dict[str, Any]]]:
    rows = read_jsonl_glob(f"{EVENT_SENTIMENT_LOG_DIR}/event_sentiment_risk_*.jsonl")
    timeline: list[tuple[datetime, dict[str, Any]]] = []
    for row in rows:
        ts = parse_any_dt(row.get("timestamp"))
        if ts is not None:
            timeline.append((ts, row))
    timeline.sort(key=lambda item: item[0])
    return timeline


def latest_at(timeline: list[tuple[datetime, dict[str, Any]]], ts: datetime | None) -> dict[str, Any] | None:
    if ts is None or not timeline:
        return None
    latest: dict[str, Any] | None = None
    for event_ts, row in timeline:
        if event_ts <= ts:
            latest = row
        else:
            break
    return latest


def load_live_contexts() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    warnings: list[str] = []
    turbo_rows = read_jsonl_glob(f"{PY_TURBO_LOG_DIR}/turbo_shadow_*.jsonl")
    signal_rows = read_jsonl_glob(f"{TS_AEGIS_LOG_DIR}/turbo_signals_*.jsonl")
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in turbo_rows:
        symbol = normalize_symbol(row.get("symbol", ""))
        ts = parse_any_dt(row.get("timestamp"))
        if not symbol or ts is None:
            continue
        key = (symbol, ts.strftime("%Y-%m-%d %H:%M:00"))
        contexts[key] = row
    for row in signal_rows:
        symbol = normalize_symbol(row.get("symbol", ""))
        ts = parse_any_dt(row.get("timestamp"))
        freshness_ts = parse_any_dt(((row.get("freshness") or {}).get("feature_timestamp")))
        if not symbol:
            continue
        key_ts = freshness_ts or ts
        if key_ts is None:
            continue
        key = (symbol, key_ts.strftime("%Y-%m-%d %H:%M:00"))
        contexts.setdefault(key, row)

    event_rows = read_jsonl_glob(f"{TS_AEGIS_LOG_DIR}/turbo_trade_events_*.jsonl")
    trade_event_by_id: dict[str, dict[str, Any]] = {}
    for row in event_rows:
        trade_id = str(row.get("trade_id") or "")
        if not trade_id:
            continue
        trade_event_by_id.setdefault(trade_id, {})
        event = str(row.get("event") or "")
        if event.startswith("ENTRY_QUALITY_GATE"):
            trade_event_by_id[trade_id]["entry_quality_gate"] = row
        if event.startswith("EVENT_RISK"):
            trade_event_by_id[trade_id]["event_risk_overlay"] = row
    if not contexts:
        warnings.append("live_contexts_unavailable: no turbo shadow or TS signal contexts found")
    return contexts, trade_event_by_id, warnings


def load_account_snapshots() -> list[tuple[datetime, dict[str, Any]]]:
    rows = read_jsonl_glob(f"{TS_AEGIS_LOG_DIR}/account_snapshots_*.jsonl")
    timeline: list[tuple[datetime, dict[str, Any]]] = []
    for row in rows:
        ts = parse_any_dt(row.get("timestamp"))
        if ts is not None:
            timeline.append((ts, row))
    timeline.sort(key=lambda item: item[0])
    return timeline


def nearest_context(contexts: dict[tuple[str, str], dict[str, Any]], symbol: str, ts: datetime | None) -> dict[str, Any] | None:
    if ts is None:
        return None
    for delta_minutes in (0, -5, 5, -10, 10):
        shifted = ts.timestamp() + delta_minutes * 60
        key_ts = datetime.fromtimestamp(shifted, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:00")
        row = contexts.get((symbol, key_ts))
        if row is not None:
            return row
    return None


def add_mode_onehots(prefix: str, mode: Any, out: list[float]) -> None:
    normalized = str(mode or "UNKNOWN").upper()
    for value in ("NORMAL", "CAUTION", "RISK_OFF", "MANUAL_ONLY", "UNKNOWN"):
        out.append(1.0 if normalized == value else 0.0)


def common_extra_feature_names() -> list[str]:
    names = [
        "source_type_score",
        "side_score",
        "turbo_action_score",
        "entry_quality_score",
        "tail_risk_score",
        "entry_quality_recommendation_score",
        "rule_based_shadow_block",
        "event_risk_mode_score",
        "event_risk_auto_mode_score",
        "event_risk_auto_confidence",
        "event_risk_auto_reason_count",
        "news_sentiment_mode_score",
        "news_sentiment_risk_score",
        "news_sentiment_confidence",
        "fear_greed_value",
        "btc_action_score",
        "btc_score",
        "btc_tail_risk_score",
        "eth_action_score",
        "eth_score",
        "eth_tail_risk_score",
        "btc_eth_confirm_direction",
        "open_positions_count",
        "same_direction_positions",
        "wallet_balance",
        "available_balance",
        "margin_used_ratio",
        "daily_pnl_pct",
        "consecutive_losses",
        "recent_closed_pnl",
        "last_trade_outcome_same_symbol",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
        "session_score",
        "context_available",
        "event_risk_auto_available",
        "news_sentiment_available",
        "portfolio_context_available",
    ]
    for prefix in ("event_risk_mode", "event_risk_auto_mode", "news_sentiment_mode"):
        for value in ("NORMAL", "CAUTION", "RISK_OFF", "MANUAL_ONLY", "UNKNOWN"):
            names.append(f"{prefix}_{value.lower()}")
    return names


def extract_news_features(news: dict[str, Any] | None) -> tuple[float, float, float, float, str]:
    if news is None:
        return mode_score("UNKNOWN"), math.nan, math.nan, math.nan, "UNKNOWN"
    fng_value = math.nan
    for event in news.get("top_events") or []:
        title = str(event.get("title") or "")
        if "Fear & Greed Index" in title:
            parts = title.split()
            for part in reversed(parts):
                value = safe_float(part)
                if math.isfinite(value):
                    fng_value = value
                    break
            break
    mode = str(news.get("suggested_mode") or "UNKNOWN").upper()
    return (
        mode_score(mode),
        safe_float(news.get("risk_score")),
        safe_float(news.get("confidence")),
        fng_value,
        mode,
    )


def extra_features(
    *,
    source_type: str,
    ts: datetime | None,
    symbol: str,
    side: str,
    turbo_action: str,
    entry_quality_score: float,
    tail_risk_score: float,
    entry_quality_recommendation: str,
    rule_based_shadow_block: float,
    event_risk_mode: str,
    event_risk_auto: dict[str, Any] | None,
    news: dict[str, Any] | None,
    account_snapshot: dict[str, Any] | None,
    recent_closed_pnl: float,
    last_trade_outcome_same_symbol: float,
) -> list[float]:
    news_mode_score, news_risk_score, news_confidence, fng_value, news_mode = extract_news_features(news)
    event_auto_mode = str((event_risk_auto or {}).get("suggested_mode") or "UNKNOWN").upper()
    btc = (event_risk_auto or {}).get("btc_context") or {}
    eth = (event_risk_auto or {}).get("eth_context") or {}
    btc_action = btc.get("turbo_action") or btc.get("gated_action")
    eth_action = eth.get("turbo_action") or eth.get("gated_action")
    confirm_direction = 0.0
    if str(side).upper() == "LONG":
        confirm_direction = 1.0 if btc_action == "LONG" and eth_action == "LONG" else 0.0
    elif str(side).upper() == "SHORT":
        confirm_direction = 1.0 if btc_action == "SHORT" and eth_action == "SHORT" else 0.0

    snap = account_snapshot or {}
    exposure = snap.get("portfolio_exposure") or {}
    wallet_balance = safe_float(snap.get("wallet_balance"))
    available_balance = safe_float(snap.get("available_balance"))
    margin_used_ratio = math.nan
    if math.isfinite(wallet_balance) and wallet_balance > 0 and math.isfinite(available_balance):
        margin_used_ratio = max(0.0, min(1.0, (wallet_balance - available_balance) / wallet_balance))
    same_direction = exposure.get("long_symbols") if str(side).upper() == "LONG" else exposure.get("short_symbols")
    hour_sin, hour_cos, day_sin, day_cos = encode_hour_day(ts)

    out = [
        SOURCE_TO_SCORE.get(source_type, -1.0),
        side_score(side),
        action_score(turbo_action),
        entry_quality_score,
        tail_risk_score,
        recommendation_score(entry_quality_recommendation),
        rule_based_shadow_block,
        mode_score(event_risk_mode),
        mode_score(event_auto_mode),
        safe_float((event_risk_auto or {}).get("confidence")),
        float(len((event_risk_auto or {}).get("reasons") or [])),
        news_mode_score,
        news_risk_score,
        news_confidence,
        fng_value,
        action_score(btc_action),
        safe_float(btc.get("turbo_score")),
        safe_float(btc.get("tail_risk_score")),
        action_score(eth_action),
        safe_float(eth.get("turbo_score")),
        safe_float(eth.get("tail_risk_score")),
        confirm_direction,
        safe_float(snap.get("open_positions_count")),
        safe_float(same_direction),
        wallet_balance,
        available_balance,
        margin_used_ratio,
        safe_float(snap.get("daily_pnl_pct")),
        safe_float(snap.get("consecutive_losses")),
        recent_closed_pnl,
        last_trade_outcome_same_symbol,
        hour_sin,
        hour_cos,
        day_sin,
        day_cos,
        session_score(ts),
        1.0 if source_type == "live_logs" else 0.0,
        1.0 if event_risk_auto is not None else 0.0,
        1.0 if news is not None else 0.0,
        1.0 if account_snapshot is not None else 0.0,
    ]
    add_mode_onehots("event_risk_mode", event_risk_mode, out)
    add_mode_onehots("event_risk_auto_mode", event_auto_mode, out)
    add_mode_onehots("news_sentiment_mode", news_mode, out)
    return out


def build_historical_rows(
    args: BuildArgs,
    event_timeline: list[tuple[datetime, dict[str, Any]]],
    warnings: list[str],
) -> tuple[list[list[float]], dict[str, list[Any]], list[str]]:
    path = Path(DEFAULT_ENTRY_QUALITY_DATASET)
    if not path.exists():
        warnings.append(f"entry_quality_dataset_missing: {path}")
        return [], defaultdict(list), []

    data = np.load(path, allow_pickle=True)
    numeric = data["numeric"]
    numeric_columns = [str(item) for item in data["numeric_columns"].tolist()]
    symbols = np.asarray(data["str_symbol"], dtype=object)
    timestamps = np.asarray(data["str_timestamp"], dtype=object)
    sides = np.asarray(data["str_side"], dtype=object)
    turbo_actions = np.asarray(data["str_turbo_action"], dtype=object)

    col = {name: idx for idx, name in enumerate(numeric_columns)}
    outcome_cols = {
        "future_mfe_roe",
        "future_mae_roe",
        "time_to_green_minutes",
        "time_to_5pct_roe",
        "time_to_8pct_roe",
        "time_to_10pct_roe",
        "hit_profit_8_before_loss_8",
        "hit_profit_10_before_loss_10",
        "hit_loss_15_before_profit_5",
        "hit_stop_40_before_profit",
        "final_raw_return_8h",
        "label_good_entry_v1",
        "label_bad_entry_v1",
        "label_tail_risk_v1",
        "future_mfe_roe_30m",
        "future_mae_roe_30m",
        "final_roe_30m",
        "future_mfe_roe_1h",
        "future_mae_roe_1h",
        "final_roe_1h",
        "future_mfe_roe_2h",
        "future_mae_roe_2h",
        "final_roe_2h",
        "future_mfe_roe_4h",
        "future_mae_roe_4h",
        "final_roe_4h",
        "future_mfe_roe_8h",
        "future_mae_roe_8h",
    }
    base_feature_indices = [idx for idx, name in enumerate(numeric_columns) if name not in outcome_cols]
    base_feature_names = [f"eq_{numeric_columns[idx]}" for idx in base_feature_indices]
    selected_symbols = set(args.symbols)

    X: list[list[float]] = []
    meta: dict[str, list[Any]] = defaultdict(list)
    rows_seen = 0
    rows_skipped = 0
    for idx in range(len(symbols)):
        symbol = normalize_symbol(symbols[idx])
        if symbol not in selected_symbols:
            continue
        ts = parse_any_dt(timestamps[idx])
        if not in_range(ts, args.start, args.end):
            continue
        turbo_action = str(turbo_actions[idx]).upper()
        side = str(sides[idx]).upper()
        if turbo_action not in {"LONG", "SHORT", "HOLD"}:
            rows_skipped += 1
            continue
        if turbo_action == "HOLD" and safe_float(numeric[idx, col.get("turbo_score", -1)], 0.0) < 0.55:
            rows_skipped += 1
            continue

        rows_seen += 1
        base = [safe_float(numeric[idx, feature_idx]) for feature_idx in base_feature_indices]
        mfe = safe_float(numeric[idx, col["future_mfe_roe"]])
        mae = safe_float(numeric[idx, col["future_mae_roe"]])
        ttg = safe_float(numeric[idx, col["time_to_green_minutes"]])
        final_roe = safe_float(numeric[idx, col.get("final_roe_4h", col["final_raw_return_8h"])])
        hit_profit = safe_float(numeric[idx, col["hit_profit_8_before_loss_8"]], 0.0)
        hit_loss = safe_float(numeric[idx, col["hit_loss_15_before_profit_5"]], 0.0)
        tail_risk_proxy = safe_float(numeric[idx, col["label_tail_risk_v1"]], 0.0)
        turbo_score = safe_float(numeric[idx, col["turbo_score"]])
        mtf_conflict = safe_float(numeric[idx, col.get("mtf_conflict", -1)], 0.0)
        news = latest_at(event_timeline, ts)
        news_mode_score, _, _, _, news_mode = extract_news_features(news)
        label = label_from_outcome(
            future_mfe_roe=mfe,
            future_mae_roe=mae,
            time_to_green=ttg,
            final_roe=final_roe,
            hit_profit_before_loss=hit_profit,
            hit_loss_15_before_profit_5=hit_loss,
            tail_risk_score=tail_risk_proxy,
            turbo_score=turbo_score,
            mtf_conflict=mtf_conflict,
            event_risk_score=-1.0,
            news_mode_score=news_mode_score,
            shadow_block=0.0,
        )
        enter_now, do_not_enter, tail_risk, wait_needed = binary_labels(label, tail_risk_proxy, mae)
        extra = extra_features(
            source_type="historical_replay",
            ts=ts,
            symbol=symbol,
            side=side,
            turbo_action=turbo_action,
            entry_quality_score=math.nan,
            tail_risk_score=tail_risk_proxy,
            entry_quality_recommendation="UNKNOWN",
            rule_based_shadow_block=0.0,
            event_risk_mode="UNKNOWN",
            event_risk_auto=None,
            news=news,
            account_snapshot=None,
            recent_closed_pnl=math.nan,
            last_trade_outcome_same_symbol=math.nan,
        )
        X.append(base + extra)
        append_meta(
            meta,
            ts=ts,
            symbol=symbol,
            side=side,
            source_type="historical_replay",
            label=label,
            future_mfe_roe=mfe,
            future_mae_roe=mae,
            time_to_green=ttg,
            final_roe=final_roe,
            hit_profit_before_loss=hit_profit,
            enter_now=enter_now,
            do_not_enter=do_not_enter,
            tail_risk=tail_risk,
            wait_needed=wait_needed,
            event_risk_mode="UNKNOWN",
            event_risk_auto_mode="UNKNOWN",
            news_sentiment_mode=news_mode,
        )
    warnings.append(f"historical_replay_rows_selected={rows_seen} skipped_low_hold_or_invalid={rows_skipped}")
    return X, meta, base_feature_names + common_extra_feature_names()


def append_meta(
    meta: dict[str, list[Any]],
    *,
    ts: datetime | None,
    symbol: str,
    side: str,
    source_type: str,
    label: str,
    future_mfe_roe: float,
    future_mae_roe: float,
    time_to_green: float,
    final_roe: float,
    hit_profit_before_loss: float,
    enter_now: int,
    do_not_enter: int,
    tail_risk: int,
    wait_needed: int,
    event_risk_mode: str = "UNKNOWN",
    event_risk_auto_mode: str = "UNKNOWN",
    news_sentiment_mode: str = "UNKNOWN",
) -> None:
    meta["timestamp"].append(ts.isoformat() if ts else "")
    meta["symbol"].append(symbol)
    meta["side"].append(side)
    meta["source_type"].append(source_type)
    meta["label"].append(label)
    meta["y"].append(LABEL_TO_ID[label])
    meta["future_mfe_roe"].append(future_mfe_roe)
    meta["future_mae_roe"].append(future_mae_roe)
    meta["time_to_green"].append(time_to_green)
    meta["final_roe"].append(final_roe)
    meta["hit_profit_before_loss"].append(hit_profit_before_loss)
    meta["enter_now_binary"].append(enter_now)
    meta["do_not_enter_binary"].append(do_not_enter)
    meta["tail_risk"].append(tail_risk)
    meta["wait_needed"].append(wait_needed)
    meta["event_risk_mode"].append(str(event_risk_mode or "UNKNOWN").upper())
    meta["event_risk_auto_mode"].append(str(event_risk_auto_mode or "UNKNOWN").upper())
    meta["news_sentiment_mode"].append(str(news_sentiment_mode or "UNKNOWN").upper())


def build_live_rows(
    args: BuildArgs,
    event_timeline: list[tuple[datetime, dict[str, Any]]],
    feature_names: list[str],
    warnings: list[str],
) -> tuple[list[list[float]], dict[str, list[Any]]]:
    contexts, trade_event_by_id, context_warnings = load_live_contexts()
    warnings.extend(context_warnings)
    snapshots = load_account_snapshots()
    trade_rows = read_jsonl_glob(f"{TS_AEGIS_LOG_DIR}/turbo_trades_*.jsonl")
    selected_symbols = set(args.symbols)

    opened: dict[str, dict[str, Any]] = {}
    closed: dict[str, dict[str, Any]] = {}
    for row in trade_rows:
        trade_id = str(row.get("trade_id") or "")
        if not trade_id:
            continue
        status = str(row.get("status") or "").upper()
        if status == "OPEN":
            opened[trade_id] = row
        elif status == "CLOSED":
            closed[trade_id] = row

    recent_pnl = 0.0
    last_outcome_by_symbol: dict[str, float] = {}
    X: list[list[float]] = []
    meta: dict[str, list[Any]] = defaultdict(list)
    skipped_open = 0
    skipped_no_context = 0
    for trade_id, close_row in sorted(closed.items(), key=lambda item: str(item[1].get("closed_at") or item[1].get("timestamp") or "")):
        symbol = normalize_symbol(close_row.get("symbol", ""))
        if symbol not in selected_symbols:
            continue
        open_row = opened.get(trade_id)
        ts = parse_any_dt((open_row or close_row).get("opened_at") or close_row.get("timestamp"))
        if not in_range(ts, args.start, args.end):
            continue
        side = str((open_row or close_row).get("side") or "UNKNOWN").upper()
        context = nearest_context(contexts, symbol, ts)
        if context is None:
            skipped_no_context += 1
            context = {}

        event_data = trade_event_by_id.get(trade_id, {})
        eq_gate = event_data.get("entry_quality_gate") or {}
        eq_gate_meta = eq_gate.get("metadata") or {}
        eq_model = context.get("entry_quality_model") or {}
        event_auto = context.get("event_risk_auto")
        news = latest_at(event_timeline, ts)
        snapshot = latest_snapshot(snapshots, ts)
        votes = (open_row or context).get("votes") or {}
        recent_scores = context.get("recent_scores") or (open_row or {}).get("recent_scores") or {}
        base_map = live_base_feature_map(open_row or {}, context, votes, recent_scores, eq_gate_meta)
        base = [safe_float(base_map.get(name.replace("eq_", ""))) for name in feature_names[: len(feature_names) - len(common_extra_feature_names())]]

        mfe = safe_float(close_row.get("mfe_roe"))
        mae = safe_float(close_row.get("mae_roe") or close_row.get("max_drawdown_roe"))
        final_roe = safe_float(close_row.get("roe"))
        duration = safe_float(close_row.get("duration_minutes"))
        time_to_green = 0.0 if final_roe > 0 else duration
        hit_profit = 1.0 if mfe >= 0.08 and (mae > -0.08 or final_roe > 0) else 0.0
        hit_loss = 1.0 if mae <= -0.15 and mfe < 0.05 else 0.0
        tail_score = safe_float(eq_model.get("tail_risk_score"), 1.0 if mae <= -0.15 else 0.0)
        shadow_block = 1.0 if str(eq_gate.get("event") or "").startswith("ENTRY_QUALITY_GATE_SHADOW_BLOCK") else 0.0
        news_mode_score, _, _, _, news_mode = extract_news_features(news)
        event_auto_score = mode_score((event_auto or {}).get("suggested_mode"))
        overlay_mode = str(((event_data.get("event_risk_overlay") or {}).get("metadata") or {}).get("mode") or "UNKNOWN").upper()
        event_auto_mode = str((event_auto or {}).get("suggested_mode") or "UNKNOWN").upper()
        label = label_from_outcome(
            future_mfe_roe=mfe,
            future_mae_roe=mae,
            time_to_green=time_to_green,
            final_roe=final_roe,
            hit_profit_before_loss=hit_profit,
            hit_loss_15_before_profit_5=hit_loss,
            tail_risk_score=tail_score,
            turbo_score=safe_float((open_row or context).get("turbo_score")),
            mtf_conflict=safe_float(base_map.get("mtf_conflict"), 0.0),
            event_risk_score=event_auto_score,
            news_mode_score=news_mode_score,
            shadow_block=shadow_block,
        )
        enter_now, do_not_enter, tail_risk, wait_needed = binary_labels(label, tail_score, mae)
        extra = extra_features(
            source_type="live_logs",
            ts=ts,
            symbol=symbol,
            side=side,
            turbo_action=str((open_row or context).get("final_action") or context.get("action") or side),
            entry_quality_score=safe_float(eq_model.get("entry_quality_score")),
            tail_risk_score=tail_score,
            entry_quality_recommendation=str(eq_model.get("recommendation") or "UNKNOWN"),
            rule_based_shadow_block=shadow_block,
            event_risk_mode=str(((event_data.get("event_risk_overlay") or {}).get("metadata") or {}).get("mode") or "UNKNOWN"),
            event_risk_auto=event_auto,
            news=news,
            account_snapshot=snapshot,
            recent_closed_pnl=recent_pnl,
            last_trade_outcome_same_symbol=last_outcome_by_symbol.get(symbol, math.nan),
        )
        X.append(base + extra)
        append_meta(
            meta,
            ts=ts,
            symbol=symbol,
            side=side,
            source_type="live_logs",
            label=label,
            future_mfe_roe=mfe,
            future_mae_roe=mae,
            time_to_green=time_to_green,
            final_roe=final_roe,
            hit_profit_before_loss=hit_profit,
            enter_now=enter_now,
            do_not_enter=do_not_enter,
            tail_risk=tail_risk,
            wait_needed=wait_needed,
            event_risk_mode=overlay_mode,
            event_risk_auto_mode=event_auto_mode,
            news_sentiment_mode=news_mode,
        )
        recent_pnl += safe_float(close_row.get("pnl_usdt"), 0.0)
        last_outcome_by_symbol[symbol] = final_roe

    for trade_id, row in opened.items():
        if trade_id not in closed and normalize_symbol(row.get("symbol", "")) in selected_symbols:
            skipped_open += 1
    warnings.append(f"live_logs_closed_rows_selected={len(X)} skipped_open_or_unclosed={skipped_open} skipped_missing_context={skipped_no_context}")
    return X, meta


def latest_snapshot(timeline: list[tuple[datetime, dict[str, Any]]], ts: datetime | None) -> dict[str, Any] | None:
    return latest_at(timeline, ts)


def live_base_feature_map(
    open_row: dict[str, Any],
    context: dict[str, Any],
    votes: dict[str, Any],
    recent_scores: dict[str, Any],
    eq_gate_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "entry_price": open_row.get("entry_price"),
        "leverage": open_row.get("leverage"),
        "ret_1": eq_gate_meta.get("recentReturn"),
        "ret_2": math.nan,
        "ret_3": math.nan,
        "ret_6": math.nan,
        "ret_12": math.nan,
        "momentum_3": eq_gate_meta.get("microMomentum"),
        "momentum_6": math.nan,
        "momentum_12": math.nan,
        "candle_body_pct": math.nan,
        "upper_wick_pct": math.nan,
        "lower_wick_pct": math.nan,
        "green_candle_count_3": math.nan,
        "red_candle_count_3": math.nan,
        "ema_9": math.nan,
        "ema_21": math.nan,
        "ema_50": math.nan,
        "price_to_ema_9": math.nan,
        "price_to_ema_21": eq_gate_meta.get("emaDistance"),
        "ema_9_slope": math.nan,
        "ema_21_slope": math.nan,
        "atr_14": math.nan,
        "atr_pct": eq_gate_meta.get("atrPct"),
        "realized_vol_12": math.nan,
        "realized_vol_36": math.nan,
        "high_low_range_pct": math.nan,
        "atr_percentile_7d": eq_gate_meta.get("atrPercentile"),
        "volume_zscore_36": math.nan,
        "volume_ratio_12": math.nan,
        "quote_volume": math.nan,
        "mtf_15m_ret_1": math.nan,
        "mtf_15m_ret_2": math.nan,
        "mtf_15m_ema_9_slope": math.nan,
        "mtf_15m_price_to_ema_21": math.nan,
        "mtf_15m_atr_pct": math.nan,
        "mtf_15m_trend_direction": math.nan,
        "mtf_1h_ret_1": math.nan,
        "mtf_1h_ret_2": math.nan,
        "mtf_1h_ema_9_slope": math.nan,
        "mtf_1h_price_to_ema_21": math.nan,
        "mtf_1h_atr_pct": math.nan,
        "mtf_1h_trend_direction": math.nan,
        "long_mtf_agreement": math.nan,
        "short_mtf_agreement": math.nan,
        "mtf_conflict": math.nan,
        "long_score_7d": recent_scores.get("long_7d"),
        "long_score_14d": recent_scores.get("long_14d"),
        "long_score_30d": recent_scores.get("long_30d"),
        "short_score_7d": recent_scores.get("short_7d"),
        "short_score_14d": recent_scores.get("short_14d"),
        "short_score_30d": recent_scores.get("short_30d"),
        "votes_long": votes.get("long"),
        "votes_short": votes.get("short"),
        "votes_neutral": votes.get("neutral"),
        "turbo_score": open_row.get("turbo_score") or context.get("turbo_score"),
        "score_gap": math.nan,
    }


def merge_meta(left: dict[str, list[Any]], right: dict[str, list[Any]]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = defaultdict(list)
    for key in set(left.keys()) | set(right.keys()):
        out[key] = list(left.get(key, [])) + list(right.get(key, []))
    return out


def save_outputs(
    *,
    X_rows: list[list[float]],
    meta: dict[str, list[Any]],
    feature_names: list[str],
    args: BuildArgs,
    warnings: list[str],
) -> dict[str, Any]:
    if not X_rows:
        raise SystemExit("decision_brain_dataset_empty")
    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(meta["y"], dtype=np.int64)
    output = args.output
    meta_output = output.with_name(output.stem + "_meta.json")
    report_json = Path(DEFAULT_REPORT_JSON)
    report_md = Path(DEFAULT_REPORT_MD)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        X=X,
        y=y,
        feature_names=np.asarray(feature_names, dtype=object),
        label_names=np.asarray(LABELS, dtype=object),
        timestamp=np.asarray(meta["timestamp"], dtype=object),
        symbol=np.asarray(meta["symbol"], dtype=object),
        side=np.asarray(meta["side"], dtype=object),
        source_type=np.asarray(meta["source_type"], dtype=object),
        label=np.asarray(meta["label"], dtype=object),
        future_mfe_roe=np.asarray(meta["future_mfe_roe"], dtype=np.float32),
        future_mae_roe=np.asarray(meta["future_mae_roe"], dtype=np.float32),
        time_to_green=np.asarray(meta["time_to_green"], dtype=np.float32),
        final_roe=np.asarray(meta["final_roe"], dtype=np.float32),
        hit_profit_before_loss=np.asarray(meta["hit_profit_before_loss"], dtype=np.float32),
        enter_now_binary=np.asarray(meta["enter_now_binary"], dtype=np.int8),
        do_not_enter_binary=np.asarray(meta["do_not_enter_binary"], dtype=np.int8),
        tail_risk=np.asarray(meta["tail_risk"], dtype=np.int8),
        wait_needed=np.asarray(meta["wait_needed"], dtype=np.int8),
        event_risk_mode=np.asarray(meta["event_risk_mode"], dtype=object),
        event_risk_auto_mode=np.asarray(meta["event_risk_auto_mode"], dtype=object),
        news_sentiment_mode=np.asarray(meta["news_sentiment_mode"], dtype=object),
    )

    label_counts = Counter(meta["label"])
    symbol_counts = Counter(meta["symbol"])
    side_counts = Counter(meta["side"])
    source_counts = Counter(meta["source_type"])
    event_risk_mode_counts = Counter(meta["event_risk_mode"])
    event_risk_auto_mode_counts = Counter(meta["event_risk_auto_mode"])
    news_sentiment_mode_counts = Counter(meta["news_sentiment_mode"])
    total_rows = int(X.shape[0])
    if event_risk_mode_counts.get("UNKNOWN", 0) == total_rows:
        warnings.append("event_risk_overlay_coverage_missing: all rows UNKNOWN")
    if event_risk_auto_mode_counts.get("UNKNOWN", 0) == total_rows:
        warnings.append("event_risk_auto_coverage_missing: all rows UNKNOWN")
    if news_sentiment_mode_counts.get("UNKNOWN", 0) == total_rows:
        warnings.append("news_sentiment_coverage_missing: all rows UNKNOWN")
    nan_by_feature = {
        feature_names[idx]: int(np.isnan(X[:, idx]).sum())
        for idx in range(len(feature_names))
        if int(np.isnan(X[:, idx]).sum()) > 0
    }
    report = {
        "created_at": utc_now_iso(),
        "output": str(output),
        "rows_total": total_rows,
        "features_count": int(X.shape[1]),
        "symbols": args.symbols,
        "date_filter": {
            "start": args.start.isoformat() if args.start else None,
            "end": args.end.isoformat() if args.end else None,
        },
        "include_live_logs": args.include_live_logs,
        "include_historical_replay": args.include_historical_replay,
        "rows_by_symbol": dict(sorted(symbol_counts.items())),
        "rows_by_side": dict(sorted(side_counts.items())),
        "rows_by_source": dict(sorted(source_counts.items())),
        "rows_by_event_risk_mode": dict(sorted(event_risk_mode_counts.items())),
        "rows_by_event_risk_auto_mode": dict(sorted(event_risk_auto_mode_counts.items())),
        "rows_by_news_sentiment_mode": dict(sorted(news_sentiment_mode_counts.items())),
        "label_distribution": dict(label_counts),
        "features": feature_names,
        "nan_by_feature_top": dict(sorted(nan_by_feature.items(), key=lambda item: item[1], reverse=True)[:30]),
        "warnings": warnings,
        "leakage_notes": [
            "Outcome columns are saved as labels/outcome arrays and are not included in X.",
            "Historical replay uses entry_quality_dataset_v020 non-outcome columns only.",
            "Live closed trade rows use post-trade metrics only for labels/outcomes, not features.",
            "Rows with open/unclosed live trades are skipped to avoid incomplete future labels.",
        ],
    }
    meta_payload = {
        "created_at": report["created_at"],
        "dataset_version": "v010",
        "label_names": list(LABELS),
        "label_to_id": LABEL_TO_ID,
        "feature_names": feature_names,
        "rows_total": report["rows_total"],
        "features_count": report["features_count"],
        "output": str(output),
        "report_json": str(report_json),
        "report_md": str(report_md),
    }
    meta_output.write_text(json.dumps(meta_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(render_report_md(report), encoding="utf-8")
    return report


def render_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Aegis Decision Brain Dataset v010",
        "",
        f"- Created: {report['created_at']}",
        f"- Rows total: {report['rows_total']}",
        f"- Features: {report['features_count']}",
        f"- Output: `{report['output']}`",
        "",
        "## Label Distribution",
        "",
    ]
    for label, count in report["label_distribution"].items():
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Rows By Source", ""])
    for source, count in report["rows_by_source"].items():
        lines.append(f"- {source}: {count}")
    lines.extend(["", "## Rows By Event Mode", ""])
    lines.append("Event Risk Overlay:")
    for mode, count in report["rows_by_event_risk_mode"].items():
        lines.append(f"- {mode}: {count}")
    lines.append("")
    lines.append("Event Risk Auto:")
    for mode, count in report["rows_by_event_risk_auto_mode"].items():
        lines.append(f"- {mode}: {count}")
    lines.append("")
    lines.append("News/Sentiment:")
    for mode, count in report["rows_by_news_sentiment_mode"].items():
        lines.append(f"- {mode}: {count}")
    lines.extend(["", "## Rows By Symbol", ""])
    for symbol, count in report["rows_by_symbol"].items():
        lines.append(f"- {symbol}: {count}")
    lines.extend(["", "## Warnings", ""])
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    lines.extend(["", "## Leakage Notes", ""])
    for note in report["leakage_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> BuildArgs:
    parser = argparse.ArgumentParser(description="Build Aegis Decision Brain supervised dataset v010.")
    parser.add_argument("--symbols", default="all", help="'all' or comma-separated symbols")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--include-live-logs", default="true")
    parser.add_argument("--include-historical-replay", default="true")
    raw = parser.parse_args()
    symbols = list(DEFAULT_SYMBOLS) if str(raw.symbols).strip().lower() == "all" else [normalize_symbol(item) for item in str(raw.symbols).split(",") if item.strip()]
    return BuildArgs(
        symbols=symbols,
        start=parse_dt(raw.start),
        end=parse_dt(raw.end),
        output=Path(raw.output),
        include_live_logs=parse_bool(raw.include_live_logs),
        include_historical_replay=parse_bool(raw.include_historical_replay),
    )


def main() -> None:
    args = parse_args()
    warnings: list[str] = []
    event_timeline = build_event_sentiment_timeline()
    if not event_timeline:
        warnings.append("news_sentiment_history_missing: event sentiment features default to UNKNOWN/NaN")

    X_rows: list[list[float]] = []
    meta: dict[str, list[Any]] = defaultdict(list)
    feature_names: list[str] = []
    if args.include_historical_replay:
        hist_X, hist_meta, feature_names = build_historical_rows(args, event_timeline, warnings)
        X_rows.extend(hist_X)
        meta = merge_meta(meta, hist_meta)
    if args.include_live_logs:
        if not feature_names:
            _, _, feature_names = build_historical_rows(
                BuildArgs(args.symbols, args.start, args.end, args.output, False, True),
                event_timeline,
                warnings,
            )
        live_X, live_meta = build_live_rows(args, event_timeline, feature_names, warnings)
        X_rows.extend(live_X)
        meta = merge_meta(meta, live_meta)

    report = save_outputs(X_rows=X_rows, meta=meta, feature_names=feature_names, args=args, warnings=warnings)
    print(json.dumps({
        "rows_total": report["rows_total"],
        "features_count": report["features_count"],
        "label_distribution": report["label_distribution"],
        "rows_by_source": report["rows_by_source"],
        "output": report["output"],
        "warnings": report["warnings"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
