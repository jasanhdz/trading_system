#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis_alpha.edge.common import load_model_bundle, profit_factor, safe_float
from aegis_alpha.tools.build_long_edge_candidate_dataset import compact_feature_names, load_meta_market
from aegis_alpha.tools.evaluate_long_edge_adaptive_meta import DynamicSizingConfig, _evaluate_dynamic_sizing_window
from aegis_alpha.tools.evaluate_strategy_candidate_oos import _load_candidate


DEFAULT_CANDIDATE = Path("aegis_alpha/models/strategy_candidates/aegis_long_edge_dynamic_v042.json")
DEFAULT_OOS_REPORT = Path("aegis_alpha/logs/edge/strategy_candidate_oos_20260503T090317Z.json")
DEFAULT_OUTPUT_DIR = Path("aegis_alpha/logs/edge")


def _bucket_score(score: float) -> str:
    if score < 0.55:
        return "<0.55"
    if score < 0.60:
        return "[0.55,0.60)"
    if score < 0.65:
        return "[0.60,0.65)"
    if score < 0.70:
        return "[0.65,0.70)"
    return ">=0.70"


def _bucket_vol(vol_regime: float) -> str:
    if vol_regime < -0.4:
        return "compressed"
    if vol_regime < -0.1:
        return "low"
    if vol_regime < 0.2:
        return "normal"
    if vol_regime < 0.5:
        return "elevated"
    return "high"


def _hour(ts: str) -> int:
    return int(ts[11:13])


def _day(ts: str) -> str:
    return ts[:10]


def _vol_bucket_at_entry(market: Any, entry_step: int) -> str:
    return _bucket_vol(float(market.features[entry_step, 20]))


def _select_worst_windows(oos_report: dict[str, Any], target_fee: float = 1.0) -> list[dict[str, Any]]:
    fee_reports = oos_report.get("fee_reports", [])
    chosen = None
    for report in fee_reports:
        if float(report.get("fee_multiplier", -1)) == target_fee:
            chosen = report
            break
    if chosen is None:
        ranking = oos_report.get("ranking", [])
        if not ranking:
            raise RuntimeError("OOS report has no fee_reports or ranking")
        target_fee = float(ranking[0]["fee_multiplier"])
        for report in fee_reports:
            if float(report.get("fee_multiplier", -1)) == target_fee:
                chosen = report
                break
    if chosen is None:
        raise RuntimeError("Could not locate fee report to analyze")

    windows = chosen["windows"]
    if len(windows) < 10:
        raise RuntimeError("Not enough OOS windows to analyze")
    return windows, target_fee


def _window_pf(window: dict[str, Any]) -> float:
    return float(window.get("profit_factor", 0.0))


def _window_dd(window: dict[str, Any]) -> float:
    return float(window.get("max_dd", 0.0))


def _window_balance(window: dict[str, Any]) -> float:
    return float(window.get("balance", 0.0))


def _materialize_window_trades(
    market: Any,
    classifier: Any,
    config: DynamicSizingConfig,
    gate_threshold: float,
    start_step: int,
    window_steps: int,
    source: str,
) -> list[dict[str, Any]]:
    from aegis_alpha.tools.evaluate_long_edge_adaptive_meta import _evaluate_dynamic_sizing_window as _run_window

    result = _run_window(
        market=market,
        classifier=classifier,
        meta_prob_cache={},
        config=config,
        gate_threshold=gate_threshold,
        start_step=start_step,
        window_steps=window_steps,
        source=source,
        max_hold_steps=24,
        close_edge_threshold=0.0,
        take_profit_roe=0.06,
    )
    trades = result.get("trades_detail", [])
    enriched: list[dict[str, Any]] = []
    loss_idx = 0
    for trade in trades:
        entry_step = int(trade["entry_step"])
        exit_step = int(trade["exit_step"])
        entry_ts = str(trade["entry_timestamp"])
        exit_ts = str(trade["exit_timestamp"])
        regime_path = [str(r) for r in market.regimes[entry_step : exit_step + 1].tolist()]
        regime_changes = [regime_path[0]] if regime_path else []
        for regime in regime_path[1:]:
            if regime != regime_changes[-1]:
                regime_changes.append(regime)
        is_loss = float(trade["return"]) < 0.0
        if is_loss:
            loss_idx += 1
        else:
            loss_idx = 0
        enriched.append(
            {
                "entry_timestamp": entry_ts,
                "exit_timestamp": exit_ts,
                "entry_price": safe_float(market.close[entry_step]),
                "exit_price": safe_float(market.close[exit_step]),
                "entry_regime": str(trade["entry_regime"]),
                "regime_changes": regime_changes,
                "full_size": bool(not trade["reduced_size"]),
                "reduced_size": bool(trade["reduced_size"]),
                "position_fraction": safe_float(trade["position_fraction"]),
                "meta_score": safe_float(trade["meta_filter_prob"]),
                "expected_return_long": safe_float(market.expected_long_return[entry_step]),
                "expected_return_short": safe_float(market.expected_short_return[entry_step]),
                "edge_gap": safe_float(market.expected_long_return[entry_step] - market.expected_short_return[entry_step]),
                "return": safe_float(trade["return"]),
                "mfe": safe_float(trade["mfe"]),
                "mae": safe_float(trade["mae"]),
                "exit_reason": trade["reason"],
                "consecutive_loss_index": int(loss_idx if is_loss else 0),
                "vol_bucket": _vol_bucket_at_entry(market, entry_step),
                "score_bucket": _bucket_score(float(trade["meta_filter_prob"])),
                "hour": _hour(entry_ts),
                "day": _day(entry_ts),
            }
        )
    return enriched


def _trade_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    returns = np.asarray([float(t["return"]) for t in trades], dtype=np.float32)
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    return {
        "trade_count": int(len(trades)),
        "win_count": int(np.sum(returns > 0.0)),
        "loss_count": int(np.sum(returns < 0.0)),
        "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
        "avg_return": safe_float(np.mean(returns)) if len(returns) else 0.0,
        "loss_sum": safe_float(float(-losses.sum())) if len(losses) else 0.0,
        "win_sum": safe_float(float(wins.sum())) if len(wins) else 0.0,
    }


def _loss_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [t for t in trades if float(t["return"]) < 0.0]
    by_full = Counter("full" if t["full_size"] else "reduced" for t in losses)
    by_regime = Counter(t["entry_regime"] for t in losses)
    by_reason = Counter(t["exit_reason"] for t in losses)
    by_score = Counter(t["score_bucket"] for t in losses)
    by_vol = Counter(t["vol_bucket"] for t in losses)
    by_hour = Counter(str(t["hour"]).zfill(2) for t in losses)
    by_day = Counter(t["day"] for t in losses)
    return {
        "loss_count": int(len(losses)),
        "losses_by_size": dict(by_full),
        "losses_by_regime": dict(by_regime),
        "losses_by_exit_reason": dict(by_reason),
        "losses_by_score_bucket": dict(by_score),
        "losses_by_volatility_bucket": dict(by_vol),
        "losses_by_hour": dict(by_hour),
        "losses_by_day": dict(by_day),
    }


def _recommendations(loss_summary: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    losses_by_regime = loss_summary["losses_by_regime"]
    losses_by_score = loss_summary["losses_by_score_bucket"]
    losses_by_vol = loss_summary["losses_by_volatility_bucket"]
    losses_by_reason = loss_summary["losses_by_exit_reason"]

    if losses_by_regime.get("high_vol", 0) >= max(5, sum(losses_by_regime.values()) * 0.30):
        recs.append("Bloquear o degradar size en high_vol porque concentra la cola perdida.")
    if losses_by_regime.get("trend_down", 0) > 0:
        recs.append("Mantener bloqueo total en trend_down; sigue apareciendo entre las pérdidas.")
    if losses_by_score.get("<0.55", 0) + losses_by_score.get("[0.55,0.60)", 0) >= max(5, sum(losses_by_score.values()) * 0.50):
        recs.append("Reducir size o bloquear meta_score < 0.60; ahí se concentra demasiada pérdida.")
    if losses_by_vol.get("high", 0) + losses_by_vol.get("elevated", 0) >= max(5, sum(losses_by_vol.values()) * 0.60):
        recs.append("Pausar o reducir size en volatilidad elevada porque domina la cola.")
    if losses_by_reason.get("edge_deterioration", 0) > losses_by_reason.get("take_profit", 0):
        recs.append("Apretar salida por edge deterioration o pausar antes de entrar en condiciones frágiles.")
    if not recs:
        recs.append("La cola no está concentrada en un único bucket; el problema parece distribuido entre régimen y volatilidad.")
    return recs


def analyze_failures(
    candidate_path: Path,
    oos_report_path: Path,
    config_path: str,
    output_dir: Path,
    window_steps: int,
) -> Path:
    candidate = _load_candidate(candidate_path)
    oos_report = json.loads(oos_report_path.read_text(encoding="utf-8"))
    market = load_meta_market(config_path, Path(candidate["edge_model_path"]))

    bundle = load_model_bundle(Path(candidate["meta_filter_path"]))
    expected_features = bundle.get("feature_names", [])
    if expected_features and list(expected_features) != compact_feature_names():
        raise RuntimeError("Meta-filter feature schema mismatch")
    classifier = bundle["classifier"]

    windows, chosen_fee = _select_worst_windows(oos_report, target_fee=1.0)
    sizing = candidate["policy"]["dynamic_sizing"]
    gate_threshold = float(candidate["policy"]["gate_threshold"])
    config = DynamicSizingConfig(
        full_size=float(sizing["full_size"]),
        reduced_size=float(sizing["reduced_size"]),
        meta_high_threshold=float(sizing["meta_high_threshold"]),
        meta_low_threshold=sizing["meta_low_threshold"],
        fee_multiplier=chosen_fee,
    )

    window_details: list[dict[str, Any]] = []
    for w in windows:
        start_step = int(w["start_step"])
        source = str(w["source"])
        trades = _materialize_window_trades(
            market=market,
            classifier=classifier,
            config=config,
            gate_threshold=gate_threshold,
            start_step=start_step,
            window_steps=window_steps,
            source=source,
        )
        window_details.append(
            {
                **w,
                "trade_details": trades,
                "trade_stats": _trade_stats(trades),
            }
        )

    top_balance = sorted(window_details, key=lambda w: _window_balance(w))[:10]
    top_dd = sorted(window_details, key=lambda w: _window_dd(w), reverse=True)[:10]
    top_pf = sorted(window_details, key=lambda w: _window_pf(w))[:10]

    all_trades = [trade for window in window_details for trade in window["trade_details"]]
    loss_summary = _loss_summary(all_trades)
    recommendations = _recommendations(loss_summary)

    summary = {
        "analyzed_windows": int(len(window_details)),
        "chosen_fee_multiplier": chosen_fee,
        "all_trades": int(len(all_trades)),
        "losses_full_size": int(sum(1 for t in all_trades if t["full_size"] and float(t["return"]) < 0.0)),
        "losses_reduced_size": int(sum(1 for t in all_trades if t["reduced_size"] and float(t["return"]) < 0.0)),
        "losses_by_regime": loss_summary["losses_by_regime"],
        "losses_by_exit_reason": loss_summary["losses_by_exit_reason"],
        "losses_by_score_bucket": loss_summary["losses_by_score_bucket"],
        "losses_by_volatility_bucket": loss_summary["losses_by_volatility_bucket"],
        "losses_by_hour": loss_summary["losses_by_hour"],
        "losses_by_day": loss_summary["losses_by_day"],
    }

    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "aegis_strategy_candidate_oos_failure_analysis_v1",
        "created_at": created_at,
        "candidate_path": str(candidate_path),
        "oos_report_path": str(oos_report_path),
        "candidate": candidate,
        "oos_report": {
            "created_at": oos_report.get("created_at"),
            "window_count": oos_report.get("window_count"),
            "passes_any": oos_report.get("passes_any"),
            "best_fee": oos_report.get("best_fee"),
        },
        "analysis": {
            "fee_multiplier": chosen_fee,
            "window_steps": window_steps,
            "top_10_windows_by_balance": top_balance,
            "top_10_windows_by_max_dd": top_dd,
            "top_10_windows_by_profit_factor": top_pf,
            "trade_details": all_trades,
            "summary": summary,
            "recommendations": recommendations,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"oos_failure_analysis_{created_at}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report saved -> {output_path}")
    print(f"Analyzed windows: {len(window_details)}")
    print(f"Total trades: {len(all_trades)}")
    print(f"Top balance windows: {len(top_balance)}")
    print(f"Top dd windows: {len(top_dd)}")
    print(f"Top pf windows: {len(top_pf)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--oos-report", default=str(DEFAULT_OOS_REPORT))
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window-steps", type=int, default=4032)
    args = parser.parse_args()
    analyze_failures(
        candidate_path=Path(args.candidate),
        oos_report_path=Path(args.oos_report),
        config_path=args.config,
        output_dir=Path(args.output_dir),
        window_steps=args.window_steps,
    )


if __name__ == "__main__":
    main()
