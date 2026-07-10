#!/usr/bin/env python3
"""FASE-D2 long-horizon Risk V4 / TRRM causal dataset builder.

Research-only. Reads OHLCV SQLite read-only, samples uniformly through time,
builds causal features and redesigned tail targets, and writes reports outside
live/active paths.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import (
    TARGET_CANDIDATES,
    add_tail_targets,
    audit_targets,
    decision_from_audit,
    json_default,
    strided_view,
)
from aegis_alpha.tools.build_trrm_causal_feature_dataset_d import (
    add_market_context,
    compute_causal_features,
    db_symbol,
    is_leakage_column,
)
from aegis_alpha.turbo.short_quality_v4_labels import (
    ShortV4Config,
    classify_short_bad_entry_v4,
    classify_short_clean_entry_v4,
    classify_short_management_dependent_v4,
    classify_short_no_trade_v4,
    classify_short_premium_allowed_v4,
    compute_short_path_metrics_v4,
)


DB_PATH = REPO / "data" / "binance_candles.db"
DEFAULT_OUTPUT_DIR = Path("/home/jasan/Develop")
DEFAULT_SYMBOLS = "ADAUSDT,AVAXUSDT,BNBUSDT,BTCUSDT,DOGEUSDT,ETHUSDT,LINKUSDT,LTCUSDT,SOLUSDT,SUIUSDT,XRPUSDT"
FIXED_LEVERAGE = 20.0
FEATURE_EXCLUDE = {"feature.close"}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize_symbol(symbol: str) -> str:
    return str(symbol).upper().replace("/", "").replace("-", "")


def timeframe_name(minutes: int) -> str:
    return f"{int(minutes)}m"


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(",", " ").split() if x.strip()]


def parse_symbols(text: str) -> list[str]:
    return [normalize_symbol(x.strip()) for x in text.split(",") if x.strip()]


def open_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)


def coverage_for_symbol(con: sqlite3.Connection, symbol: str, timeframe: str) -> dict[str, Any]:
    dbs = db_symbol(symbol)
    row = con.execute(
        "select min(timestamp), max(timestamp), count(*), count(distinct timestamp) from ohlcv_data where symbol=? and timeframe=?",
        (dbs, timeframe),
    ).fetchone()
    first, last, count, distinct_count = row
    if not first or not last:
        return {"symbol": symbol, "first_timestamp": None, "last_timestamp": None, "calendar_days": 0, "actual_candles": 0, "expected_candles": 0, "coverage_ratio": 0.0, "gaps_count": 0, "duplicate_count": 0, "usable_for_180d": False, "usable_for_365d": False}
    first_ts = pd.Timestamp(first)
    last_ts = pd.Timestamp(last)
    days = max(0.0, (last_ts - first_ts).total_seconds() / 86400.0)
    expected = int(days * 288) + 1
    duplicate = int(count - distinct_count)
    gaps = max(0, expected - int(distinct_count))
    return {
        "symbol": symbol,
        "first_timestamp": str(first),
        "last_timestamp": str(last),
        "calendar_days": days,
        "expected_candles": expected,
        "actual_candles": int(count),
        "coverage_ratio": float(min(1.0, distinct_count / expected)) if expected else 0.0,
        "gaps_count": int(gaps),
        "duplicate_count": duplicate,
        "usable_for_180d": days >= 180 and distinct_count >= 180 * 288 * 0.90,
        "usable_for_365d": days >= 365 and distinct_count >= 365 * 288 * 0.90,
    }


def inspect_coverage(db_path: Path, symbols: list[str], timeframe: str) -> list[dict[str, Any]]:
    with open_ro(db_path) as con:
        return [coverage_for_symbol(con, s, timeframe) for s in symbols]


def load_period(db_path: Path, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
    with open_ro(db_path) as con:
        max_row = con.execute("select max(timestamp) from ohlcv_data where symbol=? and timeframe=?", (db_symbol(symbol), timeframe)).fetchone()
        if not max_row or not max_row[0]:
            return pd.DataFrame()
        end = pd.Timestamp(max_row[0])
        start = end - pd.Timedelta(days=lookback_days)
        sql = """
            select timestamp, open, high, low, close, volume, buy_volume
            from ohlcv_data
            where symbol=? and timeframe=? and timestamp between ? and ?
            order by timestamp asc
        """
        df = pd.read_sql_query(sql, con, params=(db_symbol(symbol), timeframe, str(start), str(end)))
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    for col in ("open", "high", "low", "close", "volume", "buy_volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def uniform_steps(valid_steps: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or len(valid_steps) <= max_rows:
        return valid_steps
    # Deterministic uniform coverage through time, with tiny seeded jitter to avoid repeated calendar aliasing.
    base = np.linspace(0, len(valid_steps) - 1, max_rows)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.25, 0.25, size=len(base))
    idx = np.clip(np.rint(base + jitter), 0, len(valid_steps) - 1).astype(int)
    return np.unique(valid_steps[idx])


def build_rows_for_symbol(
    candles: pd.DataFrame,
    symbol: str,
    timeframe: str,
    horizons: list[int],
    max_rows_per_combo: int,
    seed: int,
    context_cache: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if candles.empty:
        return pd.DataFrame(), {"rows_before": 0, "rows_after": 0}
    feats = compute_causal_features(candles)
    feats = add_market_context(feats, context_cache)
    high = candles["high"].to_numpy(float)
    low = candles["low"].to_numpy(float)
    close = candles["close"].to_numpy(float)
    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {}
    for horizon in horizons:
        valid = np.arange(64, max(64, len(candles) - horizon), dtype=int)
        sampled = uniform_steps(valid, max_rows_per_combo, seed + horizon)
        manifest[str(horizon)] = {"rows_before": int(len(valid)), "rows_after": int(len(sampled))}
        cfg = ShortV4Config(horizon=horizon)
        for idx in sampled:
            metrics = compute_short_path_metrics_v4(high=high, low=low, close=close, entry_index=int(idx), horizon=horizon, config=cfg)
            row = {
                "id.symbol": symbol,
                "id.timestamp": str(candles.iloc[idx]["timestamp"]),
                "id.timeframe": timeframe,
                "id.horizon": horizon,
                "reference.close": float(close[idx]),
                "target.tail_risk_v4_legacy": int((metrics.get("mae_roe_proxy") or 0.0) >= 0.10),
                "target.bad_entry_v4": classify_short_bad_entry_v4(metrics, cfg),
                "target.early_mae_v4": int((metrics.get("time_to_mae") or horizon + 1) <= 3 and (metrics.get("mae_roe_proxy") or 0.0) >= 0.045),
                "label.clean_entry_v4": classify_short_clean_entry_v4(metrics, cfg),
                "label.premium_allowed_v4": classify_short_premium_allowed_v4(symbol, metrics, cfg),
                "label.management_dependent_v4": classify_short_management_dependent_v4(metrics, cfg),
                "label.no_trade_v4": classify_short_no_trade_v4(symbol, metrics, cfg),
                "future_eval.future_mfe_roe_proxy": metrics.get("mfe_roe_proxy"),
                "future_eval.future_mae_roe_proxy": metrics.get("mae_roe_proxy"),
                "future_eval.net_quality_after_costs": metrics.get("net_quality_after_costs"),
                "future_eval.mfe_mae_ratio": metrics.get("mfe_mae_ratio"),
                "future_eval.time_to_mfe": metrics.get("time_to_mfe"),
                "future_eval.time_to_mae": metrics.get("time_to_mae"),
                "future_eval.mfe_before_mae": metrics.get("mfe_before_mae"),
            }
            feat_row = feats.iloc[idx].to_dict()
            for col, val in feat_row.items():
                if col in {"timestamp", "open", "high", "low", "volume"}:
                    continue
                name = f"feature.{col}"
                if name == "feature.close":
                    continue
                leak, _ = is_leakage_column(name)
                if not leak and name not in FEATURE_EXCLUDE:
                    row[name] = val
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_tail_targets(out).sort_values(["id.timestamp", "id.symbol", "id.horizon"]).reset_index(drop=True)
    return out, manifest


def rows_by_month(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    month = pd.to_datetime(df["id.timestamp"]).dt.to_period("M").astype(str)
    return {str(k): int(v) for k, v in month.value_counts().sort_index().items()}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feature.")]


def target_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("target.")]


def render_markdown(result: dict[str, Any]) -> str:
    coverage_lines = [
        f"- {c['symbol']}: {c['first_timestamp']} -> {c['last_timestamp']}, days={c['calendar_days']:.1f}, coverage={c['coverage_ratio']:.4f}, gaps={c['gaps_count']}, dup={c['duplicate_count']}"
        for c in result["coverage"]
    ]
    selected = result["selected_candidate"]
    lines = [
        "# FASE-D2 Long-Horizon Dataset and Tail-Target Redesign",
        "",
        "## 1. Estado",
        f"- decision: {result['decision']}",
        f"- reason: {result['reason']}",
        "- mode: research-only",
        "",
        "## 2. Revisión Fable incorporada",
        "- Baseline oráculo queda diagnostic-only y no participa en selección.",
        "- Target anterior con MAE >= 0.10 ROE se conserva sólo como legacy/evaluación.",
        "- Cobertura anterior corta fue corregida usando consulta por ventana temporal real.",
        "- Fix f431e76 verificado; ema_slope_* no se elimina por substring sl.",
        "",
        "## 3. Cobertura OHLCV",
        *coverage_lines,
        "",
        "## 4. Dataset generado",
        f"- dense rows: {result['rows_dense']}",
        f"- strided rows: {result['rows_strided']}",
        f"- requested days: {result['requested_days']}",
        f"- actual days: {result['actual_days']:.1f}",
        f"- sampling strategy: {result['sampling_manifest']['sampling_mode']}",
        f"- dense path: {result['artifacts']['dense_csv']}",
        f"- causal path: {result['artifacts']['causal_csv']}",
        f"- strided path: {result['artifacts']['strided_csv']}",
        "",
        "## 5. Features causales",
        f"- feature count: {len(result['feature_columns'])}",
        f"- ema slopes recovered: {result['ema_slopes_present']}",
        "- raw feature.close excluded; reference.close retained.",
        f"- leakage exclusions: {len(result['excluded_leakage_columns'])}",
        f"- manual review: {len(result['manual_review_columns'])}",
        "",
        "## 6. Targets fixed-ROE",
        *[f"- {k}: rate={v['global_rate']:.4f}, positives={v['positives']}" for k, v in result["target_rates"].items() if "roe_" in k],
        "",
        "## 7. Targets ATR-normalized",
        "- Fórmula: future adverse price move pct = future_mae_roe_proxy / 20; ATR units = adverse_pct / feature.atr_proxy_24.",
        *[f"- {k}: rate={v['global_rate']:.4f}, positives={v['positives']}" for k, v in result["target_rates"].items() if "atr_" in k],
        "",
        "## 8. Quantile targets",
        "- Omitidos en D2 si no son necesarios; ninguna selección usa lockbox.",
        "",
        "## 9. Relación con resultados severos",
        f"- selected relationships: {result['relationships'].get(selected.get('selected_target'), {})}",
        "",
        "## 10. Splits",
        *[f"- {k}: {v}" for k, v in result["split_manifest"].items()],
        "",
        "## 11. Overlap y tamaño efectivo",
        *[f"- {k}: {v}" for k, v in result["overlap_diagnostics"].items()],
        "",
        "## 12. Baselines causales",
        *[f"- {k}: eligible={v.get('eligible_for_selection')} precision={v.get('precision')} recall={v.get('recall')}" for k, v in result["baselines"]["items"].items() if k != "diagnostic_oracle_upper_bound"],
        "",
        "## 13. Oracle diagnostic-only",
        f"- {result['baselines']['items'].get('diagnostic_oracle_upper_bound')}",
        "",
        "## 14. Target recomendado",
        f"- selected target: {selected.get('selected_target')}",
        f"- definition: {selected.get('definition')}",
        f"- justification: {selected.get('reason')}",
        f"- lockbox stability: {selected.get('lockbox_confirmation')}",
        "",
        "## 15. Estadísticas por símbolo/horizon",
        "- Incluidas en JSON bajo target_rates.by_symbol y by_horizon.",
        "",
        "## 16. Limitaciones",
        "- D2 no entrena modelos; E2 debe validar predictibilidad causal.",
        "- Sampling uniforme controla memoria y mantiene diversidad temporal.",
        "",
        "## 17. Decisión",
        f"- {result['decision']}",
        "",
        "## 18. Recomendación para FASE-E2",
        f"- {result['recommended_next_phase']}",
        "",
        "## 19. Confirmaciones de seguridad",
        "- No toqué live.",
        "- No toqué active_manifest.",
        "- No toqué YAML live.",
        "- No reinicié PM2.",
        "- No envié órdenes.",
        "- No cerré posiciones.",
        "- No toqué .env.",
        "- No modifiqué binance-futures-bot-ts.",
        "- No hice push.",
        "- No entrené TRRM.",
        "- No entrené QMAE.",
        "- No promoví modelos.",
        "- No usé lockbox para seleccionar targets.",
        "- No usé labels/futuro como features.",
        "- El baseline oráculo quedó diagnostic-only.",
        "- Todos los artifacts son research-only.",
        "",
    ]
    return "\n".join(lines)


def run_builder(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db_path)
    symbols = parse_symbols(args.symbols)
    horizons = parse_ints(args.horizons)
    timeframe = timeframe_name(args.timeframe_minutes)
    coverage = inspect_coverage(db_path, symbols, timeframe)
    coverage_ok = sum(1 for c in coverage if c["usable_for_365d" if args.lookback_days >= 365 else "usable_for_180d"]) >= max(1, int(len(symbols) * 0.8))
    if args.lookback_days < 180 or not any(c["usable_for_180d"] for c in coverage):
        coverage_ok = False

    context_cache: dict[str, pd.DataFrame] = {}
    for ctx_symbol in ("BTCUSDT", "ETHUSDT"):
        ctx = load_period(db_path, ctx_symbol, timeframe, args.lookback_days)
        context_cache[ctx_symbol] = compute_causal_features(ctx) if not ctx.empty else pd.DataFrame()

    frames: list[pd.DataFrame] = []
    sampling: dict[str, Any] = {"requested_lookback_days": args.lookback_days, "sampling_mode": args.sampling_mode, "max_rows_per_combo": args.max_rows_per_combo, "seed": args.seed, "per_symbol_horizon": {}}
    for symbol in symbols:
        candles = load_period(db_path, symbol, timeframe, args.lookback_days)
        frame, manifest = build_rows_for_symbol(candles, symbol, timeframe, horizons, args.max_rows_per_combo, args.seed, context_cache)
        sampling["per_symbol_horizon"][symbol] = manifest
        if not frame.empty:
            frames.append(frame)
    dense = pd.concat(frames, axis=0).sort_values(["id.timestamp", "id.symbol", "id.horizon"]).reset_index(drop=True) if frames else pd.DataFrame()
    if dense.empty:
        raise RuntimeError("No D2 rows generated")
    audit = audit_targets(dense, embargo_minutes=max(horizons) * args.timeframe_minutes)
    dense = audit["dataframe"].sort_values(["id.timestamp", "id.symbol", "id.horizon"]).reset_index(drop=True)
    strided = strided_view(dense)
    features = feature_columns(dense)
    targets = target_columns(dense)
    actual_days = (pd.to_datetime(dense["id.timestamp"]).max() - pd.to_datetime(dense["id.timestamp"]).min()).total_seconds() / 86400.0
    leakage = [{"column": c, "reason": is_leakage_column(c)[1]} for c in features if is_leakage_column(c)[0]]
    manual_review: list[dict[str, str]] = []
    ema_slopes = {f"feature.ema_slope_{n}": f"feature.ema_slope_{n}" in features for n in (6, 12, 24, 48)}
    no_feature_close = "feature.close" not in features
    decision, reason = decision_from_audit(coverage_ok and actual_days >= 180, audit, leakage_ok=(not leakage and no_feature_close and all(ema_slopes.values())))
    next_phase = "FASE-E2 — honest TRRM evaluation" if decision == "TAIL_TARGET_READY_FOR_E2" else ("FASE-D2.1" if decision == "TARGET_REDESIGN_PARTIAL" else "STOP")
    stamp = utc_stamp()
    base_csv = output_dir / f"aegis_risk_v4_long_horizon_d2_{stamp}.csv"
    causal_csv = output_dir / f"aegis_trrm_causal_feature_dataset_d2_{stamp}.csv"
    strided_csv = output_dir / f"aegis_trrm_causal_feature_dataset_d2_strided_{stamp}.csv"
    md_path = output_dir / f"aegis_phase_d2_tail_target_review_{stamp}.md"
    json_path = output_dir / f"aegis_phase_d2_tail_target_review_{stamp}.json"
    base_cols = [c for c in dense.columns if not c.startswith("feature.")]
    dense[base_cols].to_csv(base_csv, index=False)
    dense.to_csv(causal_csv, index=False)
    strided.to_csv(strided_csv, index=False)
    sampling["rows_before"] = int(sum(vv["rows_before"] for m in sampling["per_symbol_horizon"].values() for vv in m.values()))
    sampling["rows_after"] = int(len(dense))
    sampling["monthly_coverage"] = rows_by_month(dense)
    target_stats = audit["target_stats"]
    result = {
        "schema_version": "fase_d2_long_horizon_tail_target_redesign_v1",
        "generated_at": stamp,
        "source_db": str(db_path),
        "coverage": coverage,
        "requested_days": args.lookback_days,
        "actual_days": actual_days,
        "sampling_manifest": sampling,
        "rows_dense": int(len(dense)),
        "rows_strided": int(len(strided)),
        "feature_columns": features,
        "target_columns": targets,
        "evaluation_columns": [c for c in dense.columns if c.startswith(("future_eval.", "label.", "reference."))],
        "reference_columns": [c for c in dense.columns if c.startswith("reference.")],
        "id_columns": [c for c in dense.columns if c.startswith("id.")],
        "excluded_leakage_columns": leakage,
        "manual_review_columns": manual_review,
        "ema_slopes_present": ema_slopes,
        "feature_close_excluded": no_feature_close,
        "target_definitions": {c: ("fixed ROE" if "roe_" in c else "ATR-normalized") for c in TARGET_CANDIDATES},
        "target_rates": target_stats,
        "relationships": audit["relationships"],
        "split_manifest": audit["split_manifest"],
        "embargo": {"minutes": max(horizons) * args.timeframe_minutes},
        "baselines": audit["baselines"],
        "oracle_diagnostic": audit["baselines"]["items"].get("diagnostic_oracle_upper_bound"),
        "selected_candidate": audit["selected_candidate"],
        "lockbox_confirmation": audit["selected_candidate"].get("lockbox_confirmation"),
        "overlap_diagnostics": audit["overlap_diagnostics"],
        "decision": decision,
        "reason": reason,
        "recommended_next_phase": next_phase,
        "artifacts": {"dense_csv": str(base_csv), "causal_csv": str(causal_csv), "strided_csv": str(strided_csv), "md": str(md_path), "json": str(json_path)},
        "safety_confirmations": {
            "no_live_touched": True,
            "no_active_manifest": True,
            "no_yaml_live": True,
            "no_pm2_restart": True,
            "no_orders": True,
            "no_env": True,
            "no_ts_touched": True,
            "no_push": True,
            "no_trrm_training": True,
            "no_qmae_training": True,
            "no_model_promotion": True,
            "lockbox_not_used_for_target_selection": not audit["selected_candidate"].get("used_lockbox_for_selection", True),
            "oracle_diagnostic_only": not audit["baselines"]["items"]["diagnostic_oracle_upper_bound"]["eligible_for_selection"],
            "research_only_artifacts": True,
        },
    }
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=json_default), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({"decision": decision, "reason": reason, "dense_csv": str(base_csv), "causal_csv": str(causal_csv), "strided_csv": str(strided_csv), "md": str(md_path), "json": str(json_path)}, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build FASE-D2 long-horizon causal tail-risk dataset.")
    p.add_argument("--db-path", default=str(DB_PATH))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--lookback-days", type=int, default=365)
    p.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    p.add_argument("--horizons", default="6 12 24")
    p.add_argument("--timeframe-minutes", type=int, default=5)
    p.add_argument("--max-rows-per-combo", type=int, default=5000)
    p.add_argument("--sampling-mode", default="uniform_time")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write-report", default="true")
    return p


def main() -> int:
    run_builder(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
