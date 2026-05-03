#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from aegis_alpha.edge.common import load_model_bundle, profit_factor, safe_float
from aegis_alpha.env.risk_engine import Position, current_roe
from aegis_alpha.tools.build_long_edge_candidate_dataset import (
    BASE_GUARD,
    compact_feature_names,
    load_meta_market,
)
from aegis_alpha.tools.evaluate_long_edge_adaptive_meta import (
    DynamicSizingConfig,
    OpenTrade,
    _close_trade_with_fee,
    _predict_meta_prob_cached,
)
from aegis_alpha.tools.evaluate_long_edge_robustness import ALLOWED_REGIMES
from aegis_alpha.tools.evaluate_strategy_candidate_oos import _load_candidate


DEFAULT_CANDIDATE = Path("aegis_alpha/models/strategy_candidates/aegis_long_edge_dynamic_v042.json")
DEFAULT_OOS_REPORT = Path("aegis_alpha/logs/edge/strategy_candidate_oos_20260503T090317Z.json")
DEFAULT_V041_REPORT = Path("aegis_alpha/logs/edge/long_edge_adaptive_meta_20260503T033742Z.json")
DEFAULT_SCORE_FLOOR_REPORT = Path("aegis_alpha/logs/edge/score_floor_analysis_20260503T092654Z.json")
DEFAULT_OUTPUT_DIR = Path("aegis_alpha/logs/edge")
DEFAULT_SEEDS = (6101, 7331)

FULL_SIZE = 0.25
REDUCED_SIZE = 0.125
META_HIGH_THRESHOLD = 0.60
FEES = (1.0, 1.25)
MAX_HOLD_STEPS = 24
TAKE_PROFIT_ROE = 0.06
DEFAULT_EDGE_CLOSE_RATIO = 0.0
LOW_VOL_MIXED_THRESHOLD = -0.10


@dataclass(frozen=True)
class DeteriorationVariant:
    name: str
    close_ratio: float | None
    pause_after_edge_loss_steps: int
    pause_after_two_edge_losses_steps: int


VARIANTS: tuple[DeteriorationVariant, ...] = (
    DeteriorationVariant(name="A_pause48_after_loss", close_ratio=None, pause_after_edge_loss_steps=48, pause_after_two_edge_losses_steps=0),
    DeteriorationVariant(name="B_pause96_after_loss", close_ratio=None, pause_after_edge_loss_steps=96, pause_after_two_edge_losses_steps=0),
    DeteriorationVariant(name="C_pause144_after_loss", close_ratio=None, pause_after_edge_loss_steps=144, pause_after_two_edge_losses_steps=0),
    DeteriorationVariant(name="D_pause288_after_2_of_3", close_ratio=None, pause_after_edge_loss_steps=0, pause_after_two_edge_losses_steps=288),
    DeteriorationVariant(name="E_close_ratio_0p50", close_ratio=0.50, pause_after_edge_loss_steps=0, pause_after_two_edge_losses_steps=0),
    DeteriorationVariant(name="F_close_ratio_0p65", close_ratio=0.65, pause_after_edge_loss_steps=0, pause_after_two_edge_losses_steps=0),
    DeteriorationVariant(name="G_B_plus_E", close_ratio=0.50, pause_after_edge_loss_steps=96, pause_after_two_edge_losses_steps=0),
    DeteriorationVariant(name="H_B_plus_F_plus_D", close_ratio=0.65, pause_after_edge_loss_steps=96, pause_after_two_edge_losses_steps=288),
)


def _window_pf(window: dict[str, Any]) -> float:
    return float(window.get("profit_factor", 0.0))


def _window_dd(window: dict[str, Any]) -> float:
    return float(window.get("max_dd", 0.0))


def _window_balance(window: dict[str, Any]) -> float:
    return float(window.get("balance", 0.0))


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


def _extract_windows_from_oos(oos_report: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    fee_reports = oos_report.get("fee_reports", [])
    chosen = None
    target_fee = 1.0
    for report in fee_reports:
        if float(report.get("fee_multiplier", -1.0)) == target_fee and report.get("windows"):
            chosen = report
            break
    if chosen is None:
        for report in fee_reports:
            if report.get("windows"):
                chosen = report
                target_fee = float(report.get("fee_multiplier", 1.0))
                break
    if chosen is None:
        raise RuntimeError("Could not locate evaluated windows in OOS report")
    windows = list(chosen["windows"])
    if len(windows) < 100:
        raise RuntimeError(f"OOS report has only {len(windows)} windows; need at least 100")
    return windows, target_fee


def _select_reference_windows(
    market: Any,
    oos_report: dict[str, Any],
    window_steps: int,
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    try:
        windows, _ = _extract_windows_from_oos(oos_report)
        return windows
    except Exception:
        from aegis_alpha.tools.evaluate_strategy_candidate_oos import _select_oos_windows

        selected = _select_oos_windows(
            market=market,
            window_steps=window_steps,
            seeds=seeds,
            recent_windows=24,
            random_windows_per_seed=24,
            non_overlap_windows=24,
            target_max_windows=144,
        )
        return [{"start_step": int(start_step), "source": source} for start_step, source in selected]


def _trade_mfe_mae(close: np.ndarray, entry_step: int, exit_step: int) -> tuple[float, float]:
    entry_price = float(close[entry_step])
    if exit_step <= entry_step or entry_price <= 0.0:
        return 0.0, 0.0
    path = close[entry_step + 1 : exit_step + 1] / entry_price - 1.0
    return float(np.max(path)), float(max(0.0, -np.min(path)))


def _edge_close_reason(entry_edge: float, current_edge: float, close_ratio: float | None) -> bool:
    if close_ratio is None:
        return current_edge <= DEFAULT_EDGE_CLOSE_RATIO
    return current_edge < entry_edge * close_ratio


def _evaluate_window(
    market: Any,
    classifier: Any,
    config: DynamicSizingConfig,
    gate_threshold: float,
    start_step: int,
    window_steps: int,
    source: str,
    variant: DeteriorationVariant,
    fee_multiplier: float,
) -> dict[str, Any]:
    risk = market.cfg.risk
    initial_balance = risk.initial_balance
    loss_floor = initial_balance * (1.0 - BASE_GUARD["max_window_loss_pct"])
    balance = initial_balance
    position = Position()
    open_trade: OpenTrade | None = None
    hold_steps = 0
    flat_steps = risk.min_flat_steps
    pause_until = -1
    deterioration_pause_until = -1
    consecutive_losses = 0
    edge_loss_history: deque[int] = deque(maxlen=3)
    trades_by_day: Counter[int] = Counter()
    guard_counts: Counter[str] = Counter()
    edge_deterioration_closes = 0
    losing_edge_deterioration_closes = 0
    skipped_after_deterioration = 0
    reduced_size_trades = 0
    full_size_trades = 0
    meta_candidate_count = 0
    exposure_steps = 0
    total_fees = 0.0
    equity_curve: list[float] = []
    trades: list[dict[str, Any]] = []
    end_limit = min(start_step + window_steps, len(market.close) - 1)
    meta_prob_cache: dict[int, float] = {}

    for step in range(start_step, end_limit):
        price = float(market.close[step])
        score = float(market.expected_long_return[step])
        regime = str(market.regimes[step])
        day = step // 288

        if position.side == 0:
            entry_signal = flat_steps >= risk.min_flat_steps and score >= gate_threshold
            if entry_signal and balance <= loss_floor:
                guard_counts["max_window_loss"] += 1
                flat_steps += 1
            elif entry_signal and step < deterioration_pause_until:
                skipped_after_deterioration += 1
                guard_counts["deterioration_pause"] += 1
                flat_steps += 1
            elif entry_signal and step < pause_until:
                guard_counts["pause"] += 1
                flat_steps += 1
            elif entry_signal and trades_by_day[day] >= BASE_GUARD["max_trades_per_day"]:
                guard_counts["max_trades_per_day"] += 1
                flat_steps += 1
            elif entry_signal and regime not in ALLOWED_REGIMES:
                guard_counts["regime"] += 1
                flat_steps += 1
            elif entry_signal:
                meta_candidate_count += 1
                meta_prob = _predict_meta_prob_cached(classifier, market, step, gate_threshold, meta_prob_cache)
                if meta_prob >= config.meta_high_threshold:
                    position_fraction = config.full_size
                    reduced_size = False
                else:
                    position_fraction = config.reduced_size
                    reduced_size = True

                before = balance
                notional = balance * risk.leverage * position_fraction
                fee = notional * risk.total_fee * fee_multiplier
                if balance > fee * 1.5:
                    balance -= fee
                    position = Position(
                        side=1,
                        size=notional / max(price, 1e-10),
                        entry_price=price,
                        entry_step=step,
                    )
                    total_fees += fee
                    trades_by_day[day] += 1
                    reduced_size_trades += int(reduced_size)
                    full_size_trades += int(not reduced_size)
                    open_trade = OpenTrade(
                        balance_before_open=before,
                        entry_notional=abs(position.size) * position.entry_price,
                        entry_step=step,
                        entry_price=price,
                        entry_score=score,
                        entry_fee=fee,
                        entry_regime=regime,
                        meta_filter_prob=meta_prob,
                        position_fraction=position_fraction,
                        reduced_size=reduced_size,
                    )
                    hold_steps = 0
                    flat_steps = 0
                else:
                    guard_counts["open_failed"] += 1
                    flat_steps += 1
            else:
                flat_steps += 1
        else:
            exposure_steps += 1
            roe = current_roe(position, price, risk)
            current_edge = score
            entry_edge = float(open_trade.entry_score) if open_trade is not None else 0.0
            close_reason = ""
            if roe <= -risk.hard_stop_roe:
                close_reason = "hard_stop"
            elif hold_steps >= risk.min_hold_steps and roe >= TAKE_PROFIT_ROE:
                close_reason = "take_profit"
            elif hold_steps >= MAX_HOLD_STEPS:
                close_reason = "max_hold"
            elif hold_steps >= risk.min_hold_steps and _edge_close_reason(entry_edge, current_edge, variant.close_ratio):
                close_reason = "edge_deterioration"

            if close_reason and open_trade is not None:
                balance, trade = _close_trade_with_fee(
                    market, position, balance, step, open_trade, close_reason, fee_multiplier
                )
                total_fees += float(trade["fees"]) - open_trade.entry_fee
                trades.append(trade)
                if close_reason == "edge_deterioration":
                    edge_deterioration_closes += 1
                    is_loss = float(trade["net"]) < 0.0
                    if is_loss:
                        losing_edge_deterioration_closes += 1
                        edge_loss_history.append(1)
                        if variant.pause_after_edge_loss_steps > 0:
                            deterioration_pause_until = max(
                                deterioration_pause_until, step + variant.pause_after_edge_loss_steps
                            )
                        if variant.pause_after_two_edge_losses_steps > 0 and sum(edge_loss_history) >= 2:
                            deterioration_pause_until = max(
                                deterioration_pause_until, step + variant.pause_after_two_edge_losses_steps
                            )
                    else:
                        edge_loss_history.append(0)
                else:
                    edge_loss_history.append(0)

                if float(trade["net"]) < 0.0:
                    consecutive_losses += 1
                    pause_until = max(pause_until, step + BASE_GUARD["pause_after_loss_steps"])
                    if consecutive_losses >= 2:
                        pause_until = max(pause_until, step + BASE_GUARD["pause_after_2_losses_steps"])
                else:
                    consecutive_losses = 0
                position = Position()
                open_trade = None
                hold_steps = 0
                flat_steps = 0
            else:
                hold_steps += 1
                flat_steps = 0

        equity = balance if position.side == 0 else balance + abs(position.size) * (price - position.entry_price)
        equity_curve.append(float(equity))

    if position.side != 0 and open_trade is not None:
        step = end_limit
        balance, trade = _close_trade_with_fee(
            market, position, balance, step, open_trade, "window_end", fee_multiplier
        )
        total_fees += float(trade["fees"]) - open_trade.entry_fee
        trades.append(trade)
        equity_curve.append(float(balance))

    equity = np.asarray(equity_curve, dtype=np.float32)
    if len(equity):
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-10)
        final_balance = float(equity[-1])
    else:
        dd = np.asarray([0.0], dtype=np.float32)
        final_balance = initial_balance

    returns = np.asarray([trade["return"] for trade in trades], dtype=np.float32)
    wins = returns[returns > 0.0]
    loss_reason_counts = Counter(trade["reason"] for trade in trades if float(trade["return"]) < 0.0)

    return {
        "source": source,
        "start_step": int(start_step),
        "end_step": int(end_limit),
        "balance": safe_float(final_balance),
        "net": safe_float(final_balance - initial_balance),
        "p95_dd": safe_float(np.quantile(dd, 0.95)),
        "max_dd": safe_float(np.max(dd)),
        "trades": int(len(trades)),
        "win_rate": safe_float(len(wins) / max(len(returns), 1)),
        "profit_factor": safe_float(profit_factor(returns)) if len(returns) else 0.0,
        "avg_return_per_trade": safe_float(np.mean(returns)) if len(returns) else 0.0,
        "fees": safe_float(total_fees),
        "exposure_time": safe_float(exposure_steps / max(window_steps, 1)),
        "trades_per_month": safe_float(len(trades) / max(window_steps * 5.0 / 60.0 / 24.0 / 30.4375, 1e-10)),
        "edge_deterioration_closes": int(edge_deterioration_closes),
        "losing_edge_deterioration_closes": int(losing_edge_deterioration_closes),
        "skipped_after_deterioration": int(skipped_after_deterioration),
        "full_size_trades": int(full_size_trades),
        "reduced_size_trades": int(reduced_size_trades),
        "meta_candidate_count": int(meta_candidate_count),
        "skipped_by_guard": int(sum(guard_counts.values())),
        "guard_counts": dict(guard_counts),
        "close_reasons": dict(loss_reason_counts),
        "avg_mfe": safe_float(np.mean([trade["mfe"] for trade in trades])) if trades else 0.0,
        "avg_mae": safe_float(np.mean([trade["mae"] for trade in trades])) if trades else 0.0,
    }


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
        "trades_per_month": safe_float(np.median(trades_month)),
        "median_trades_per_month": safe_float(np.median(trades_month)),
        "worst_max_dd": safe_float(np.max(max_dd)),
        "median_avg_return_per_trade": safe_float(np.median(avg_returns)),
        "exposure_time": safe_float(np.median(exposure)),
        "median_exposure_time": safe_float(np.median(exposure)),
        "edge_deterioration_closes": int(sum(w["edge_deterioration_closes"] for w in windows)),
        "losing_edge_deterioration_closes": int(sum(w["losing_edge_deterioration_closes"] for w in windows)),
        "skipped_after_deterioration": int(sum(w["skipped_after_deterioration"] for w in windows)),
        "full_size_trades": int(sum(w["full_size_trades"] for w in windows)),
        "reduced_size_trades": int(sum(w["reduced_size_trades"] for w in windows)),
        "skipped_by_guard": int(sum(w["skipped_by_guard"] for w in windows)),
    }


def _extract_benchmark_summary(report: dict[str, Any], primary_key: str) -> dict[str, Any]:
    value = report.get(primary_key)
    if isinstance(value, dict):
        return value
    ranking = report.get("ranking")
    if isinstance(ranking, list) and ranking:
        return ranking[0]
    variants = report.get("variants")
    if isinstance(variants, list) and variants:
        return variants[0]
    return {}


def _rank(reports: list[dict[str, Any]], benchmarks: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        summary = report["summary"]
        rows.append(
            {
                "variant": report["variant"],
                "fee_multiplier": report["fee_multiplier"],
                **report["config"],
                **summary,
                "beats_v042_p25_pf": bool(summary["p25_pf"] > float(benchmarks["v042_candidate_oos"].get("p25_pf", 0.0))),
                "beats_v041_p25_pf": bool(summary["p25_pf"] > float(benchmarks["v041_dynamic_sizing"].get("p25_pf", 0.0))),
                "beats_v046_p25_pf": bool(summary["p25_pf"] > float(benchmarks["v046_score_floor"].get("p25_pf", 0.0))),
                "passes_success_criteria": bool(
                    summary["worst_balance"] >= 19.00
                    and summary["worst_max_dd"] <= 0.08
                    and summary["profitable_window_pct"] >= 0.70
                    and summary["median_trades"] >= 5.0
                    and summary["p25_pf"] >= 0.80
                ),
            }
        )

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        return (
            float(row["passes_success_criteria"]),
            float(row["p25_pf"]),
            float(row["worst_balance"]),
            -float(row["worst_max_dd"]),
            float(row["profitable_window_pct"]),
            float(row["median_trades"]),
        )

    return sorted(rows, key=score, reverse=True)


def _recommendations(report: dict[str, Any]) -> list[str]:
    best = report.get("best") or {}
    recs: list[str] = []
    if float(best.get("losing_edge_deterioration_closes", 0)) > 0:
        recs.append("Endurecer o acortar el pause-after-loss para edge_deterioration; la cola sigue viniendo de cierres perdedores.")
    if float(best.get("skipped_after_deterioration", 0)) < 10:
        recs.append("El guard de deterioro casi no frena entradas; revisar si el close_ratio necesita ser más agresivo.")
    if float(best.get("p25_pf", 0.0)) < 0.80:
        recs.append("El filtro protege la media, pero no la cola; la señal sigue entrando en estados frágiles.")
    if not recs:
        recs.append("La cola no se concentra en una sola regla; la degradación del edge sigue siendo el problema dominante.")
    return recs


def run_deterioration_guard_eval(
    candidate_path: Path,
    edge_model_path: Path,
    meta_filter_path: Path,
    oos_report_path: Path,
    v041_report_path: Path,
    score_floor_report_path: Path,
    config_path: str,
    output_dir: Path,
    window_steps: int,
    seeds: tuple[int, ...],
) -> Path:
    candidate = _load_candidate(candidate_path)
    market = load_meta_market(config_path, edge_model_path)
    oos_report = _load_report(oos_report_path)
    v041_report = _load_report(v041_report_path)
    score_floor_report = _load_report(score_floor_report_path)
    windows = _select_reference_windows(market, oos_report, window_steps=window_steps, seeds=seeds)

    bundle = load_model_bundle(meta_filter_path)
    expected_features = bundle.get("feature_names", [])
    if expected_features and list(expected_features) != compact_feature_names():
        raise RuntimeError("Meta-filter feature schema mismatch")
    classifier = bundle["classifier"]

    gate_threshold = float(candidate["policy"]["gate_threshold"])
    config = DynamicSizingConfig(
        full_size=FULL_SIZE,
        reduced_size=REDUCED_SIZE,
        meta_high_threshold=META_HIGH_THRESHOLD,
        meta_low_threshold=None,
        fee_multiplier=1.0,
    )

    print(f"Selected windows: {len(windows)}")
    print(f"Gate threshold top 3%: {gate_threshold:.8f}")
    print(f"Variants: {len(VARIANTS)}")

    reports: list[dict[str, Any]] = []
    for fee_multiplier in FEES:
        for variant in VARIANTS:
            window_results = [
                _evaluate_window(
                    market=market,
                    classifier=classifier,
                    config=config,
                    gate_threshold=gate_threshold,
                    start_step=int(window["start_step"]),
                    window_steps=window_steps,
                    source=str(window.get("source", "oos")),
                    variant=variant,
                    fee_multiplier=fee_multiplier,
                )
                for window in windows
            ]
            summary = _summary(window_results, market.cfg.risk.initial_balance)
            reports.append(
                {
                    "variant": variant.name,
                    "fee_multiplier": fee_multiplier,
                    "config": {
                        "close_ratio": variant.close_ratio,
                        "pause_after_edge_loss_steps": variant.pause_after_edge_loss_steps,
                        "pause_after_two_edge_losses_steps": variant.pause_after_two_edge_losses_steps,
                        "full_size": FULL_SIZE,
                        "reduced_size": REDUCED_SIZE,
                        "meta_high_threshold": META_HIGH_THRESHOLD,
                        "meta_low_threshold": None,
                    },
                    "summary": summary,
                    "windows": window_results,
                }
            )
            print(
                f"{variant.name} fee={fee_multiplier:.2f} p25pf={summary['p25_pf']:.2f} "
                f"worst={summary['worst_balance']:.2f} dd={summary['worst_max_dd']:.1%} "
                f"prof={summary['profitable_window_pct']:.1%} trades={summary['median_trades']:.1f} "
                f"edge_close={summary['edge_deterioration_closes']} skip_det={summary['skipped_after_deterioration']}"
            )

    benchmarks = {
        "v042_candidate_oos": _extract_benchmark_summary(_load_report(oos_report_path), "best_fee"),
        "v041_dynamic_sizing": _extract_benchmark_summary(v041_report, "best_variant"),
        "v046_score_floor": _extract_benchmark_summary(score_floor_report, "best"),
    }
    ranking = _rank(reports, benchmarks)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "aegis_long_edge_deterioration_guard_v1",
        "created_at": created_at,
        "candidate_path": str(candidate_path),
        "edge_model_path": str(edge_model_path),
        "meta_filter_path": str(meta_filter_path),
        "oos_report_path": str(oos_report_path),
        "v041_report_path": str(v041_report_path),
        "score_floor_report_path": str(score_floor_report_path),
        "candidate": candidate,
        "policy": {
            "side": "LONG_ONLY",
            "entry_gate": "top_3pct_expected_return_long",
            "allowed_regimes": ["mixed", "chop", "high_vol"],
            "risk_guard": "loss7_pause48_pause2_48_maxday3",
            "dynamic_sizing": {
                "full_size": FULL_SIZE,
                "reduced_size": REDUCED_SIZE,
                "meta_high_threshold": META_HIGH_THRESHOLD,
                "meta_low_threshold": None,
            },
            "max_hold_steps": MAX_HOLD_STEPS,
            "take_profit_roe": TAKE_PROFIT_ROE,
            "deterioration_guard": {
                "pause_after_edge_loss_steps": [48, 96, 144],
                "pause_after_two_edge_losses_steps": [288],
                "close_ratio": [None, 0.50, 0.65],
            },
            "fee_multipliers": list(FEES),
            "short_entries": False,
        },
        "window_count": len(windows),
        "window_steps": window_steps,
        "seeds": list(seeds),
        "benchmarks": benchmarks,
        "success_criteria": {
            "worst_balance": ">=19.00",
            "worst_max_dd": "<=8%",
            "profitable_window_pct": ">=70%",
            "median_trades": ">=5",
            "p25_pf": ">=0.80",
            "p25_pf_ideal": ">=0.88",
        },
        "passes_any": bool(any(row["passes_success_criteria"] for row in ranking)),
        "best": ranking[0] if ranking else None,
        "ranking": ranking,
        "reports": reports,
        "analysis": {
            "close_reason_counts": dict(
                Counter(
                    reason
                    for report_entry in reports
                    for window in report_entry["windows"]
                    for reason, count in window.get("close_reasons", {}).items()
                    for _ in range(int(count))
                )
            ),
            "recommendations": _recommendations({"best": ranking[0] if ranking else {}}),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"long_edge_deterioration_guard_{created_at}.json"
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
    parser.add_argument("--oos-report", default=str(DEFAULT_OOS_REPORT))
    parser.add_argument("--v041-report", default=str(DEFAULT_V041_REPORT))
    parser.add_argument("--score-floor-report", default=str(DEFAULT_SCORE_FLOOR_REPORT))
    parser.add_argument("--config", default="aegis_alpha/configs/base.yaml")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window-steps", type=int, default=4032)
    parser.add_argument("--seeds", default="6101,7331")
    args = parser.parse_args()
    seeds = tuple(int(part.strip()) for part in args.seeds.split(",") if part.strip())
    candidate = _load_candidate(Path(args.candidate))
    run_deterioration_guard_eval(
        candidate_path=Path(args.candidate),
        edge_model_path=Path(candidate["edge_model_path"]),
        meta_filter_path=Path(candidate["meta_filter_path"]),
        oos_report_path=Path(args.oos_report),
        v041_report_path=Path(args.v041_report),
        score_floor_report_path=Path(args.score_floor_report),
        config_path=args.config,
        output_dir=Path(args.output_dir),
        window_steps=args.window_steps,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
