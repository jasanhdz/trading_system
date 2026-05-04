from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aegis_alpha.edge.common import load_model_bundle, profit_factor, safe_float
from aegis_alpha.env.risk_engine import Position, current_roe
from aegis_alpha.tools.build_long_edge_candidate_dataset import BASE_GUARD


@dataclass(frozen=True)
class RuleCondition:
    model_name: str
    mode: str
    value: float


@dataclass(frozen=True)
class ComboSpec:
    name: str
    edge_rules: tuple[RuleCondition, ...]
    risk_rules: tuple[RuleCondition, ...] = ()
    allowed_regimes: tuple[str, ...] = ("mixed", "chop", "high_vol")


def bucket_vol(vol_regime: float) -> str:
    if vol_regime < -0.4:
        return "compressed"
    if vol_regime < -0.1:
        return "low"
    if vol_regime < 0.2:
        return "normal"
    if vol_regime < 0.5:
        return "elevated"
    return "high"


def load_models(model_dir: Path, names: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in names:
        bundle = load_model_bundle(model_dir / f"aegis_{name}_v050.joblib")
        estimator = bundle.get("estimator") or bundle.get("classifier") or bundle.get("regressor")
        if estimator is None:
            raise RuntimeError(f"Missing estimator for {name}")
        out[name] = estimator
    return out


def predict_scores(market: Any, models: dict[str, Any]) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    x = market.signal_features.astype(np.float32)
    for name, est in models.items():
        if hasattr(est, "predict_proba"):
            scores = est.predict_proba(x)[:, 1]
        else:
            scores = est.predict(x)
        preds[name] = np.asarray(scores, dtype=np.float32)
    return preds


def threshold_for_rule(scores: np.ndarray, rule: RuleCondition) -> float:
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0
    if rule.mode == "top_pct":
        return safe_float(np.quantile(scores, 1.0 - rule.value))
    if rule.mode == "bottom_pct":
        return safe_float(np.quantile(scores, rule.value))
    if rule.mode in {"gt", "lt"}:
        return float(rule.value)
    raise ValueError(f"Unknown rule mode {rule.mode!r}")


def rule_pass(score: float, rule: RuleCondition, threshold: float) -> bool:
    if rule.mode in {"top_pct", "gt"}:
        return score >= threshold
    if rule.mode in {"bottom_pct", "lt"}:
        return score <= threshold
    raise ValueError(f"Unknown rule mode {rule.mode!r}")


def build_threshold_map(preds: dict[str, np.ndarray], rules: tuple[RuleCondition, ...]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for rule in rules:
        if rule.model_name not in thresholds:
            thresholds[rule.model_name] = threshold_for_rule(preds[rule.model_name], rule)
    return thresholds


def trade_mfe_mae(close: np.ndarray, entry_step: int, exit_step: int) -> tuple[float, float]:
    entry_price = float(close[entry_step])
    if exit_step <= entry_step or entry_price <= 0.0:
        return 0.0, 0.0
    path = close[entry_step + 1 : exit_step + 1] / entry_price - 1.0
    return float(np.max(path)), float(max(0.0, -np.min(path)))


def simulate_combo_window(
    market: Any,
    preds: dict[str, np.ndarray],
    combo: ComboSpec,
    thresholds: dict[str, float],
    start_step: int,
    window_steps: int,
    fee_multiplier: float,
    source: str,
    full_size: float = 0.25,
    take_profit_roe: float = 0.06,
    max_hold_steps: int = 24,
) -> dict[str, Any]:
    risk = market.cfg.risk
    initial_balance = risk.initial_balance
    loss_floor = initial_balance * (1.0 - BASE_GUARD["max_window_loss_pct"])
    balance = initial_balance
    position = Position()
    open_trade: dict[str, Any] | None = None
    hold_steps = 0
    flat_steps = risk.min_flat_steps
    pause_until = -1
    consecutive_losses = 0
    exposure_steps = 0
    total_fees = 0.0
    equity_curve: list[float] = []
    trades: list[dict[str, Any]] = []
    skipped_by_regime = 0
    skipped_by_signal = 0
    end_limit = min(start_step + window_steps, len(market.close) - 1)

    for step in range(start_step, end_limit):
        price = float(market.close[step])
        regime = str(market.regimes[step])
        rel_idx = step - market.cfg.model.window_size
        if rel_idx < 0 or rel_idx >= len(next(iter(preds.values()))):
            continue

        if position.side == 0:
            entry_signal = flat_steps >= risk.min_flat_steps
            if entry_signal and regime not in combo.allowed_regimes:
                skipped_by_regime += 1
                flat_steps += 1
                continue
            if entry_signal and step < pause_until:
                flat_steps += 1
                continue
            if entry_signal and balance <= loss_floor:
                flat_steps += 1
                continue
            if entry_signal:
                for rule in combo.edge_rules:
                    score = float(preds[rule.model_name][rel_idx])
                    if not rule_pass(score, rule, thresholds[rule.model_name]):
                        entry_signal = False
                        break
                if entry_signal:
                    for rule in combo.risk_rules:
                        score = float(preds[rule.model_name][rel_idx])
                        if not rule_pass(score, rule, thresholds[rule.model_name]):
                            entry_signal = False
                            break

            if entry_signal:
                primary_edge_score = float(preds[combo.edge_rules[0].model_name][rel_idx]) if combo.edge_rules else 0.0
                h12_score = float(preds.get("long_edge_h12", preds[combo.edge_rules[0].model_name])[rel_idx])
                h24_score = float(preds.get("long_edge_h24", preds[combo.edge_rules[0].model_name])[rel_idx])
                h48_score = float(preds.get("long_edge_h48", preds[combo.edge_rules[0].model_name])[rel_idx])
                notional = balance * risk.leverage * full_size
                fee = notional * risk.total_fee * fee_multiplier
                if balance > fee * 1.5:
                    before = balance
                    balance -= fee
                    size = notional / max(price, 1e-10)
                    position = Position(side=1, size=size, entry_price=price, entry_step=step)
                    total_fees += fee
                    open_trade = {
                        "entry_step": step,
                        "entry_price": price,
                        "entry_balance": before,
                        "entry_notional": notional,
                        "entry_regime": regime,
                        "entry_fee": fee,
                        "entry_primary_edge_score": primary_edge_score,
                        "h12_score": h12_score,
                        "h24_score": h24_score,
                        "h48_score": h48_score,
                        "agreement_h12_h48": bool(
                            h12_score >= thresholds.get("long_edge_h12", 0.0)
                            and h48_score >= thresholds.get("long_edge_h48", 0.0)
                        ),
                        "edge_gap": h12_score - h48_score,
                        "vol_bucket": bucket_vol(float(market.features[step, 20])),
                    }
                    hold_steps = 0
                    flat_steps = 0
                else:
                    skipped_by_signal += 1
                    flat_steps += 1
            else:
                skipped_by_signal += 1
                flat_steps += 1
        else:
            exposure_steps += 1
            edge_score = float(preds[combo.edge_rules[0].model_name][rel_idx])
            roe = current_roe(position, price, risk)
            close_reason = ""
            if roe <= -risk.hard_stop_roe:
                close_reason = "hard_stop"
            elif hold_steps >= risk.min_hold_steps and roe >= take_profit_roe:
                close_reason = "take_profit"
            elif hold_steps >= max_hold_steps:
                close_reason = "max_hold"
            elif hold_steps >= risk.min_hold_steps and open_trade is not None and edge_score <= float(open_trade["entry_primary_edge_score"]) * 0.65:
                close_reason = "edge_deterioration"

            if close_reason and open_trade is not None:
                price = float(market.close[step])
                pnl = abs(position.size) * (price - position.entry_price)
                close_fee = abs(position.size) * price * risk.total_fee * fee_multiplier
                balance = max(0.0, balance + pnl - close_fee)
                total_fees += close_fee
                net = balance - float(open_trade["entry_balance"])
                trade_return = net / max(float(open_trade["entry_notional"]), 1e-10)
                mfe, mae = trade_mfe_mae(market.close, int(open_trade["entry_step"]), step)
                regimes_path = [str(r) for r in market.regimes[int(open_trade["entry_step"]): step + 1].tolist()]
                regime_shift = bool(len(regimes_path) and len(set(regimes_path)) > 1)
                trades.append(
                    {
                        "timestamp": str(market.timestamps[int(open_trade["entry_step"])]),
                        "entry_step": int(open_trade["entry_step"]),
                        "exit_step": int(step),
                        "entry_timestamp": str(market.timestamps[int(open_trade["entry_step"])]),
                        "exit_timestamp": str(market.timestamps[step]),
                        "entry_price": safe_float(open_trade["entry_price"]),
                        "exit_price": safe_float(price),
                        "regime": str(open_trade["entry_regime"]),
                        "entry_regime": str(open_trade["entry_regime"]),
                        "regime_path": regimes_path,
                        "regime_shift": bool(regime_shift),
                        "h12_score": safe_float(open_trade["h12_score"]),
                        "h24_score": safe_float(open_trade["h24_score"]),
                        "h48_score": safe_float(open_trade["h48_score"]),
                        "agreement_h12_h48": bool(open_trade["agreement_h12_h48"]),
                        "edge_gap": safe_float(open_trade["edge_gap"]),
                        "volatility_bucket": str(open_trade["vol_bucket"]),
                        "vol_bucket": str(open_trade["vol_bucket"]),
                        "return": safe_float(trade_return),
                        "net": safe_float(net),
                        "mfe": safe_float(mfe),
                        "mae": safe_float(mae),
                        "exit_reason": close_reason,
                        "reason": close_reason,
                    }
                )
                if net < 0.0:
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
        price = float(market.close[step])
        pnl = abs(position.size) * (price - position.entry_price)
        close_fee = abs(position.size) * price * risk.total_fee * fee_multiplier
        balance = max(0.0, balance + pnl - close_fee)
        total_fees += close_fee
        net = balance - float(open_trade["entry_balance"])
        trade_return = net / max(float(open_trade["entry_notional"]), 1e-10)
        mfe, mae = trade_mfe_mae(market.close, int(open_trade["entry_step"]), step)
        regimes_path = [str(r) for r in market.regimes[int(open_trade["entry_step"]): step + 1].tolist()]
        trades.append(
            {
                "timestamp": str(market.timestamps[int(open_trade["entry_step"])]),
                "entry_step": int(open_trade["entry_step"]),
                "exit_step": int(step),
                "entry_timestamp": str(market.timestamps[int(open_trade["entry_step"])]),
                "exit_timestamp": str(market.timestamps[step]),
                "entry_price": safe_float(open_trade["entry_price"]),
                "exit_price": safe_float(price),
                "regime": str(open_trade["entry_regime"]),
                "entry_regime": str(open_trade["entry_regime"]),
                "regime_path": regimes_path,
                "regime_shift": bool(len(regimes_path) and len(set(regimes_path)) > 1),
                "h12_score": safe_float(open_trade["h12_score"]),
                "h24_score": safe_float(open_trade["h24_score"]),
                "h48_score": safe_float(open_trade["h48_score"]),
                "agreement_h12_h48": bool(open_trade["agreement_h12_h48"]),
                "edge_gap": safe_float(open_trade["edge_gap"]),
                "volatility_bucket": str(open_trade["vol_bucket"]),
                "vol_bucket": str(open_trade["vol_bucket"]),
                "return": safe_float(trade_return),
                "net": safe_float(net),
                "mfe": safe_float(mfe),
                "mae": safe_float(mae),
                "exit_reason": "window_end",
                "reason": "window_end",
            }
        )
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
        "full_size_trades": int(len(trades)),
        "reduced_size_trades": 0,
        "skipped_by_regime": int(skipped_by_regime),
        "skipped_by_signal": int(skipped_by_signal),
        "trades_detail": trades,
        "close_reasons": dict({reason: int(count) for reason, count in Counter(trade["reason"] for trade in trades).items()}),
    }
