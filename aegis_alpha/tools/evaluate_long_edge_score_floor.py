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
from aegis_alpha.tools.evaluate_long_edge_adaptive_meta import _trade_mfe_mae
from aegis_alpha.tools.evaluate_strategy_candidate_oos import _load_candidate, _select_oos_windows


DEFAULT_CANDIDATE = Path("aegis_alpha/models/strategy_candidates/aegis_long_edge_dynamic_v042.json")
DEFAULT_V042_OOS = Path("aegis_alpha/logs/edge/strategy_candidate_oos_20260503T090317Z.json")
DEFAULT_V041_OOS = Path("aegis_alpha/logs/edge/long_edge_adaptive_meta_20260503T033742Z.json")
DEFAULT_OUTPUT_DIR = Path("aegis_alpha/logs/edge")
DEFAULT_SEEDS = (6101, 7331)

FULL_SIZE = 0.25
MID_SIZE_C = 0.125
MID_SIZE_D = 0.10
SCORE_FLOOR = 0.60
HIGH_SCORE = 0.70
LOW_VOL_MIXED_THRESHOLD = -0.10


@dataclass(frozen=True)
class PolicyVariant:
    name: str
    block_score_floor: bool
    block_low_vol_mixed: bool
    mid_size: float


VARIANTS: tuple[PolicyVariant, ...] = (
    PolicyVariant(name="A_score_floor", block_score_floor=True, block_low_vol_mixed=False, mid_size=MID_SIZE_C),
    PolicyVariant(name="B_low_vol_mixed_block", block_score_floor=False, block_low_vol_mixed=True, mid_size=MID_SIZE_C),
    PolicyVariant(name="C_score_floor_plus_low_vol_mixed", block_score_floor=True, block_low_vol_mixed=True, mid_size=MID_SIZE_C),
    PolicyVariant(name="D_score_floor_plus_low_vol_mixed_mid10", block_score_floor=True, block_low_vol_mixed=True, mid_size=MID_SIZE_D),
)


def _bucket_score(score: float) -> str:
    if score < 0.60:
        return "<0.60"
    if score < 0.70:
        return "[0.60,0.70)"
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


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(windows: list[dict[str, Any]], initial_balance: float) -> dict[str, Any]:
    balances = np.asarray([w["balance"] for w in windows], dtype=np.float32)
    pfs = np.asarray([w["profit_factor"] for w in windows], dtype=np.float32)
    trades = np.asarray([w["trades"] for w in windows], dtype=np.float32)
    max_dd = np.asarray([w["max_dd"] for w in windows], dtype=np.float32)
    avg_returns = np.asarray([w["avg_return_per_trade"] for w in windows], dtype=np.float32)
    exposure = np.asarray([w["exposure_time"] for w in windows], dtype=np.float32)
    trades_month = np.asarray([w["trades_per_month"] for w in windows], dtype=np.float32)
    return {
        "median_balance": safe_float(np.median(balances)),
        "p25_balance": safe_float(np.quantile(balances, 0.25)),
        "worst_balance": safe_float(np.min(balances)),
        "median_pf": safe_float(np.median(pfs)),
        "p25_pf": safe_float(np.quantile(pfs, 0.25)),
        "profitable_window_pct": safe_float(np.mean(balances > initial_balance)),
        "median_trades": safe_float(np.median(trades)),
        "median_trades_per_month": safe_float(np.median(trades_month)),
        "worst_max_dd": safe_float(np.max(max_dd)),
        "median_avg_return_per_trade": safe_float(np.median(avg_returns)),
        "median_exposure_time": safe_float(np.median(exposure)),
        "full_size_trades": int(sum(w["full_size_trades"] for w in windows)),
        "reduced_size_trades": int(sum(w["reduced_size_trades"] for w in windows)),
        "skipped_by_score_floor": int(sum(w["skipped_by_score_floor"] for w in windows)),
        "skipped_by_low_vol_mixed": int(sum(w["skipped_by_low_vol_mixed"] for w in windows)),
        "skipped_by_guard": int(sum(w["skipped_by_guard"] for w in windows)),
    }


def _rank(reports: list[dict[str, Any]], benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for report in reports:
        summary = report["summary"]
        rows.append(
            {
                "variant": report["variant"],
                "fee_multiplier": report["fee_multiplier"],
                **report["config"],
                **summary,
                "beats_v042_p25_pf": bool(summary["p25_pf"] > 0.8786),
                "beats_v041_p25_pf": bool(summary["p25_pf"] > float(benchmark["p25_pf"])),
                "passes_success_criteria": bool(
                    summary["worst_balance"] >= 19.10
                    and summary["worst_max_dd"] <= 0.07
                    and summary["profitable_window_pct"] >= 0.75
                    and summary["median_trades"] >= 5.0
                    and summary["median_balance"] >= 20.10
                    and summary["p25_pf"] > 0.8786
                ),
            }
        )

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row["passes_success_criteria"]),
            float(row["p25_pf"]),
            float(row["worst_balance"]),
            -float(row["worst_max_dd"]),
            float(row["profitable_window_pct"]),
        )

    return sorted(rows, key=score, reverse=True)


def _trade_detail(
    market: Any,
    trade: dict[str, Any],
    entry_step: int,
    exit_step: int,
    variant: PolicyVariant,
) -> dict[str, Any]:
    entry_regime = str(trade["entry_regime"])
    regime_path = [str(r) for r in market.regimes[entry_step : exit_step + 1].tolist()]
    regime_changes = [regime_path[0]] if regime_path else []
    for regime in regime_path[1:]:
        if regime != regime_changes[-1]:
            regime_changes.append(regime)
    return {
        "entry_timestamp": str(market.timestamps[entry_step]),
        "exit_timestamp": str(market.timestamps[exit_step]),
        "entry_price": safe_float(market.close[entry_step]),
        "exit_price": safe_float(market.close[exit_step]),
        "entry_regime": entry_regime,
        "regime_changes": regime_changes,
        "meta_score": safe_float(trade["meta_filter_prob"]),
        "expected_return_long": safe_float(market.expected_long_return[entry_step]),
        "expected_return_short": safe_float(market.expected_short_return[entry_step]),
        "edge_gap": safe_float(market.expected_long_return[entry_step] - market.expected_short_return[entry_step]),
        "position_fraction": safe_float(trade["position_fraction"]),
        "full_size": bool(not trade["reduced_size"]),
        "reduced_size": bool(trade["reduced_size"]),
        "return": safe_float(trade["return"]),
        "mfe": safe_float(trade["mfe"]),
        "mae": safe_float(trade["mae"]),
        "exit_reason": trade["reason"],
        "consecutive_loss_index": int(trade.get("consecutive_loss_index", 0)),
        "score_bucket": _bucket_score(float(trade["meta_filter_prob"])),
        "vol_bucket": _bucket_vol(float(market.features[entry_step, 20])),
        "variant": variant.name,
    }


def _evaluate_variant_window(
    market: Any,
    classifier: Any,
    window_start: int,
    window_steps: int,
    gate_threshold: float,
    fee_multiplier: float,
    variant: PolicyVariant,
) -> dict[str, Any]:
    risk = market.cfg.risk
    initial_balance = risk.initial_balance
    balance = initial_balance
    position = None
    open_trade: dict[str, Any] | None = None
    flat_steps = risk.min_flat_steps
    hold_steps = 0
    pause_until = -1
    consecutive_losses = 0
    exposure_steps = 0
    total_fees = 0.0
    trades: list[dict[str, Any]] = []
    skipped_by_score_floor = 0
    skipped_by_low_vol_mixed = 0
    full_size_trades = 0
    reduced_size_trades = 0
    end_limit = min(window_start + window_steps, len(market.close) - 1)

    from aegis_alpha.tools.evaluate_long_edge_adaptive_meta import _predict_meta_prob

    for step in range(window_start, end_limit):
        price = float(market.close[step])
        score = float(market.expected_long_return[step])
        regime = str(market.regimes[step])
        vol_low = float(market.features[step, 20]) < LOW_VOL_MIXED_THRESHOLD

        if position is None:
            entry_signal = flat_steps >= risk.min_flat_steps and score >= gate_threshold
            if entry_signal and step < pause_until:
                flat_steps += 1
                continue
            if entry_signal and variant.block_score_floor and _predict_meta_prob(classifier, market, step, gate_threshold) < SCORE_FLOOR:
                skipped_by_score_floor += 1
                flat_steps += 1
                continue
            if entry_signal and variant.block_low_vol_mixed and regime == "mixed" and vol_low:
                skipped_by_low_vol_mixed += 1
                flat_steps += 1
                continue
            if entry_signal:
                meta_score = _predict_meta_prob(classifier, market, step, gate_threshold)
                if meta_score >= HIGH_SCORE:
                    pos_frac = FULL_SIZE
                    reduced = False
                else:
                    pos_frac = variant.mid_size
                    reduced = True
                notional = balance * risk.leverage * pos_frac
                fee = notional * risk.total_fee * fee_multiplier
                if balance > fee * 1.5:
                    balance -= fee
                    position = {
                        "entry_step": step,
                        "entry_price": price,
                        "entry_regime": regime,
                        "meta_score": meta_score,
                        "position_fraction": pos_frac,
                        "reduced_size": reduced,
                        "entry_fee": fee,
                        "entry_notional": notional,
                        "balance_before_open": balance + fee,
                    }
                    total_fees += fee
                    full_size_trades += int(not reduced)
                    reduced_size_trades += int(reduced)
                    flat_steps = 0
                    hold_steps = 0
                else:
                    flat_steps += 1
            else:
                flat_steps += 1
        else:
            exposure_steps += 1
            entry_step = int(position["entry_step"])
            current_roe = ((price - float(position["entry_price"])) / max(float(position["entry_price"]), 1e-10)) * risk.leverage
            close_reason = ""
            if current_roe <= -risk.hard_stop_roe:
                close_reason = "hard_stop"
            elif hold_steps >= risk.min_hold_steps and score <= 0.0:
                close_reason = "edge_deterioration"
            elif hold_steps >= 24:
                close_reason = "max_hold"
            elif hold_steps >= risk.min_hold_steps and current_roe >= 0.06:
                close_reason = "take_profit"

            if close_reason:
                size = abs(float(position["entry_notional"])) / max(float(position["entry_price"]), 1e-10)
                pnl = size * (price - float(position["entry_price"]))
                close_fee = size * price * risk.total_fee * fee_multiplier
                new_balance = max(0.0, balance + pnl - close_fee)
                net = new_balance - float(position["balance_before_open"])
                trade_return = net / max(float(position["entry_notional"]), 1e-10)
                mfe, mae = _trade_mfe_mae(market.close, entry_step, step)
                trade = {
                    "entry_step": entry_step,
                    "exit_step": int(step),
                    "entry_timestamp": str(market.timestamps[entry_step]),
                    "exit_timestamp": str(market.timestamps[step]),
                    "entry_price": safe_float(position["entry_price"]),
                    "exit_price": safe_float(price),
                    "entry_regime": position["entry_regime"],
                    "meta_filter_prob": safe_float(position["meta_score"]),
                    "position_fraction": safe_float(position["position_fraction"]),
                    "reduced_size": bool(position["reduced_size"]),
                    "full_size": bool(not position["reduced_size"]),
                    "entry_fee": safe_float(position["entry_fee"]),
                    "exit_fee": safe_float(close_fee),
                    "net": safe_float(net),
                    "return": safe_float(trade_return),
                    "mfe": safe_float(mfe),
                    "mae": safe_float(mae),
                    "reason": close_reason,
                    "consecutive_loss_index": int(consecutive_losses + 1 if net < 0.0 else 0),
                    "score_bucket": _bucket_score(float(position["meta_score"])),
                    "vol_bucket": _bucket_vol(float(market.features[entry_step, 20])),
                }
                trades.append(trade)
                balance = new_balance
                if net < 0.0:
                    consecutive_losses += 1
                    pause_until = max(pause_until, step + risk.min_flat_steps)
                else:
                    consecutive_losses = 0
                position = None
                flat_steps = 0
                hold_steps = 0
            else:
                hold_steps += 1

    if position is not None:
        step = end_limit
        price = float(market.close[step])
        entry_step = int(position["entry_step"])
        size = abs(float(position["entry_notional"])) / max(float(position["entry_price"]), 1e-10)
        pnl = size * (price - float(position["entry_price"]))
        close_fee = size * price * risk.total_fee * fee_multiplier
        new_balance = max(0.0, balance + pnl - close_fee)
        net = new_balance - float(position["balance_before_open"])
        trade_return = net / max(float(position["entry_notional"]), 1e-10)
        mfe, mae = _trade_mfe_mae(market.close, entry_step, step)
        trades.append(
            {
                "entry_step": entry_step,
                "exit_step": int(step),
                "entry_timestamp": str(market.timestamps[entry_step]),
                "exit_timestamp": str(market.timestamps[step]),
                "entry_price": safe_float(position["entry_price"]),
                "exit_price": safe_float(price),
                "entry_regime": position["entry_regime"],
                "meta_filter_prob": safe_float(position["meta_score"]),
                "position_fraction": safe_float(position["position_fraction"]),
                "reduced_size": bool(position["reduced_size"]),
                "full_size": bool(not position["reduced_size"]),
                "entry_fee": safe_float(position["entry_fee"]),
                "exit_fee": safe_float(close_fee),
                "net": safe_float(net),
                "return": safe_float(trade_return),
                "mfe": safe_float(mfe),
                "mae": safe_float(mae),
                "reason": "window_end",
                "consecutive_loss_index": int(consecutive_losses + 1 if net < 0.0 else 0),
                "score_bucket": _bucket_score(float(position["meta_score"])),
                "vol_bucket": _bucket_vol(float(market.features[entry_step, 20])),
            }
        )
        balance = new_balance

    returns = np.asarray([t["return"] for t in trades], dtype=np.float32)
    equity_curve = np.asarray([initial_balance + sum(float(x["net"]) for x in trades[: idx + 1]) for idx in range(len(trades))], dtype=np.float32)
    if len(equity_curve):
        peak = np.maximum.accumulate(equity_curve)
        dd = (peak - equity_curve) / np.maximum(peak, 1e-10)
    else:
        dd = np.asarray([0.0], dtype=np.float32)
    wins = returns[returns > 0.0]
    return {
        "balance": safe_float(balance),
        "net": safe_float(balance - initial_balance),
        "p95_dd": safe_float(np.quantile(dd, 0.95)),
        "max_dd": safe_float(np.max(dd)),
        "trades": int(len(trades)),
        "win_rate": safe_float(len(wins) / max(len(returns), 1)),
        "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
        "avg_return_per_trade": safe_float(np.mean(returns)) if len(returns) else 0.0,
        "exposure_time": safe_float(exposure_steps / max(window_steps, 1)),
        "trades_per_month": safe_float(len(trades) / max(window_steps * 5.0 / 60.0 / 24.0 / 30.4375, 1e-10)),
        "skipped_by_score_floor": int(skipped_by_score_floor),
        "skipped_by_low_vol_mixed": int(skipped_by_low_vol_mixed),
        "full_size_trades": int(full_size_trades),
        "reduced_size_trades": int(reduced_size_trades),
        "trades_detail": trades,
        "trade_details": trades,
        "window_start_step": int(window_start),
        "window_end_step": int(end_limit),
    }


def _window_summary(windows: list[dict[str, Any]], initial_balance: float) -> dict[str, Any]:
    balances = np.asarray([w["balance"] for w in windows], dtype=np.float32)
    pfs = np.asarray([w["profit_factor"] for w in windows], dtype=np.float32)
    trades = np.asarray([w["trades"] for w in windows], dtype=np.float32)
    max_dd = np.asarray([w["max_dd"] for w in windows], dtype=np.float32)
    avg_returns = np.asarray([w["avg_return_per_trade"] for w in windows], dtype=np.float32)
    exposure = np.asarray([w["exposure_time"] for w in windows], dtype=np.float32)
    trades_month = np.asarray([w["trades_per_month"] for w in windows], dtype=np.float32)
    return {
        "median_balance": safe_float(np.median(balances)),
        "p25_balance": safe_float(np.quantile(balances, 0.25)),
        "worst_balance": safe_float(np.min(balances)),
        "median_pf": safe_float(np.median(pfs)),
        "p25_pf": safe_float(np.quantile(pfs, 0.25)),
        "profitable_window_pct": safe_float(np.mean(balances > initial_balance)),
        "median_trades": safe_float(np.median(trades)),
        "median_trades_per_month": safe_float(np.median(trades_month)),
        "worst_max_dd": safe_float(np.max(max_dd)),
        "median_avg_return_per_trade": safe_float(np.median(avg_returns)),
        "median_exposure_time": safe_float(np.median(exposure)),
        "full_size_trades": int(sum(w["full_size_trades"] for w in windows)),
        "reduced_size_trades": int(sum(w["reduced_size_trades"] for w in windows)),
        "skipped_by_score_floor": int(sum(w["skipped_by_score_floor"] for w in windows)),
        "skipped_by_low_vol_mixed": int(sum(w["skipped_by_low_vol_mixed"] for w in windows)),
    }


def _loss_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    losses = [t for t in trades if float(t["return"]) < 0.0]
    by_score = Counter(_bucket_score(float(t["meta_filter_prob"])) for t in losses)
    return {
        "loss_count": int(len(losses)),
        "losses_by_score_bucket": dict(by_score),
        "losses_by_size": dict(Counter("full" if t["full_size"] else "reduced" for t in losses)),
        "losses_by_regime": dict(Counter(t["entry_regime"] for t in losses)),
        "losses_by_exit_reason": dict(Counter(t["reason"] for t in losses)),
        "losses_by_volatility_bucket": dict(Counter(t["vol_bucket"] for t in losses)),
        "losses_by_hour": dict(Counter(str(t["entry_timestamp"])[11:13] for t in losses)),
        "losses_by_day": dict(Counter(str(t["entry_timestamp"])[:10] for t in losses)),
    }


def _recommendations(loss_summary: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    losses_by_score = loss_summary["losses_by_score_bucket"]
    losses_by_regime = loss_summary["losses_by_regime"]
    losses_by_reason = loss_summary["losses_by_exit_reason"]
    if losses_by_score.get("<0.60", 0) > sum(losses_by_score.values()) * 0.40:
        recs.append("Mantener bloqueo duro para meta_score < 0.60; esa zona concentra la cola.")
    if losses_by_regime.get("mixed", 0) > losses_by_regime.get("chop", 0) + losses_by_regime.get("high_vol", 0):
        recs.append("Bloquear mixed+low_vol o mantenerlo con size reducido; mixed domina las pérdidas.")
    if losses_by_reason.get("edge_deterioration", 0) > losses_by_reason.get("take_profit", 0):
        recs.append("Reducir el hold o endurecer la salida por edge deterioration.")
    if not recs:
        recs.append("No se detectó un único bucket dominante; el filtro necesita más señal de calidad.")
    return recs


def analyze_score_floor(
    candidate_path: Path,
    oos_report_path: Path,
    v041_report_path: Path,
    config_path: str,
    output_dir: Path,
    window_steps: int,
    target_max_windows: int,
    recent_windows: int,
    random_windows_per_seed: int,
    non_overlap_windows: int,
    seeds: tuple[int, ...],
) -> Path:
    candidate = _load_candidate(candidate_path)
    oos_report = _load_report(oos_report_path)
    v041_report = _load_report(v041_report_path)
    market = load_meta_market(config_path, Path(candidate["edge_model_path"]))
    windows = _select_oos_windows(
        market=market,
        window_steps=window_steps,
        seeds=seeds,
        recent_windows=recent_windows,
        random_windows_per_seed=random_windows_per_seed,
        non_overlap_windows=non_overlap_windows,
        target_max_windows=target_max_windows,
    )
    bundle = load_model_bundle(Path(candidate["meta_filter_path"]))
    expected_features = bundle.get("feature_names", [])
    if expected_features and list(expected_features) != compact_feature_names():
        raise RuntimeError("Meta-filter feature schema mismatch")
    classifier = bundle["classifier"]
    gate_threshold = float(candidate["policy"]["gate_threshold"])
    benchmark_v041 = v041_report.get("best_variant") or {}
    v042_oos = oos_report.get("best_fee") or {}

    reports: list[dict[str, Any]] = []
    for fee_multiplier in (1.0, 1.25):
        for variant in VARIANTS:
            window_results = [
                {
                    **_evaluate_variant_window(
                        market=market,
                        classifier=classifier,
                        window_start=start_step,
                        window_steps=window_steps,
                        gate_threshold=gate_threshold,
                        fee_multiplier=fee_multiplier,
                        variant=variant,
                    ),
                    "source": source,
                }
                for start_step, source in windows
            ]
            summary = _window_summary(window_results, market.cfg.risk.initial_balance)
            reports.append(
                {
                    "variant": variant.name,
                    "fee_multiplier": fee_multiplier,
                    "config": {
                        "block_score_floor": variant.block_score_floor,
                        "block_low_vol_mixed": variant.block_low_vol_mixed,
                        "mid_size": variant.mid_size,
                    },
                    "summary": summary,
                    "windows": window_results,
                    "trade_details": [trade for window in window_results for trade in window["trade_details"]],
                }
            )
            print(
                f"{variant.name} fee={fee_multiplier:.2f} p25pf={summary['p25_pf']:.2f} "
                f"worst={summary['worst_balance']:.2f} dd={summary['worst_max_dd']:.1%} "
                f"prof={summary['profitable_window_pct']:.1%} trades={summary['median_trades']:.1f} "
                f"skip_floor={summary['skipped_by_score_floor']} skip_lv={summary['skipped_by_low_vol_mixed']}"
            )

    ranking = _rank(reports, benchmark_v041)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "aegis_score_floor_low_vol_mixed_v1",
        "created_at": created_at,
        "candidate_path": str(candidate_path),
        "oos_report_path": str(oos_report_path),
        "v041_report_path": str(v041_report_path),
        "candidate": candidate,
        "benchmarks": {
            "v042_candidate_oos": v042_oos,
            "v041_dynamic_sizing": benchmark_v041,
        },
        "policy": {
            "side": "LONG_ONLY",
            "entry_gate": "top_3pct_expected_return_long",
            "allowed_regimes": ["mixed", "chop", "high_vol"],
            "risk_guard": "loss7_pause48_pause2_48_maxday3",
            "score_floor": 0.60,
            "low_vol_mixed_threshold": LOW_VOL_MIXED_THRESHOLD,
            "full_size": FULL_SIZE,
            "mid_sizes": [MID_SIZE_C, MID_SIZE_D],
            "high_score_threshold": HIGH_SCORE,
            "fee_multipliers": [1.0, 1.25],
        },
        "window_count": len(windows),
        "window_steps": window_steps,
        "target_max_windows": target_max_windows,
        "success_criteria": {
            "worst_balance": ">=19.10",
            "worst_max_dd": "<=7%",
            "profitable_window_pct": ">=75%",
            "median_trades": ">=5",
            "p25_pf": ">0.8786",
            "p25_pf_ideal": ">=1.0",
        },
        "ranking": ranking,
        "passes_any": bool(any(row["passes_success_criteria"] for row in ranking)),
        "best": ranking[0] if ranking else None,
        "reports": reports,
        "analysis": {
            "loss_summary": _loss_summary([trade for report in reports for trade in report["trade_details"]]),
            "recommendations": _recommendations(_loss_summary([trade for report in reports for trade in report["trade_details"]])),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"score_floor_analysis_{created_at}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report saved -> {output_path}")
    if ranking:
        best = ranking[0]
        print(
            f"Best {best['variant']} fee={best['fee_multiplier']:.2f} p25pf={best['p25_pf']:.2f} "
            f"worst={best['worst_balance']:.2f} dd={best['worst_max_dd']:.1%} "
            f"prof={best['profitable_window_pct']:.1%} trades={best['median_trades']:.1f} pass={best['passes_success_criteria']}"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--oos-report", default=str(DEFAULT_V042_OOS))
    parser.add_argument("--v041-report", default=str(DEFAULT_V041_OOS))
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window-steps", type=int, default=4032)
    parser.add_argument("--target-max-windows", type=int, default=144)
    parser.add_argument("--recent-windows", type=int, default=24)
    parser.add_argument("--random-windows-per-seed", type=int, default=24)
    parser.add_argument("--non-overlap-windows", type=int, default=24)
    parser.add_argument("--seeds", default="6101,7331")
    args = parser.parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    analyze_score_floor(
        candidate_path=Path(args.candidate),
        oos_report_path=Path(args.oos_report),
        v041_report_path=Path(args.v041_report),
        config_path=args.config,
        output_dir=Path(args.output_dir),
        window_steps=args.window_steps,
        target_max_windows=args.target_max_windows,
        recent_windows=args.recent_windows,
        random_windows_per_seed=args.random_windows_per_seed,
        non_overlap_windows=args.non_overlap_windows,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
