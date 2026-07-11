#!/usr/bin/env python3
"""GEN2-ECON1 economic backtest engine (spec: GEN2_ECON1_SPEC.md v1.0).

New isolated engine. SHORT, causal next-bar-open entry, fixed-horizon H12 exit,
no management. Equal trade budget for every strategy, three pre-registered cost
scenarios, weekly block bootstrap. Emits the auditor decision.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aegis_alpha.tools.audit_tail_risk_targets_d2 import json_default  # noqa: E402
from aegis_alpha.tools.gen2_d3_build import load_series_symbol  # noqa: E402
from aegis_alpha.tools.gen2_d3_common import GEN2_ROOT, SERIES_ROOT, sha256_file, utc_stamp, validate_gen2_path  # noqa: E402
import aegis_alpha.tools.gen2_eqm1_train as eqm  # noqa: E402
import aegis_alpha.tools.gen2_rv2_train as rv2  # noqa: E402
from aegis_alpha.tools.gen2_rv2_train import MedianImputer, score_of  # noqa: E402

ECON_ROOT = GEN2_ROOT / "econ1"
NOTIONAL = 100.0
HOLD_BARS = 12
COST_SCENARIOS = {  # per-side bps on notional; funding bps per hour
    "A_optimista": {"fee": 4.0, "slip": 1.0, "funding_h": 0.5},
    "B_base": {"fee": 5.0, "slip": 2.0, "funding_h": 1.0},
    "C_pesimista": {"fee": 5.0, "slip": 5.0, "funding_h": 2.0},
}
TOPK = 0.10
BOOT_REPS = 1000
SEED = 42


def load_prices(series_version: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        df = load_series_symbol(SERIES_ROOT / series_version, s)
        df["_i"] = np.arange(len(df))
        out[s] = df.set_index("timestamp")
    return out


def trade_pnl(prices: dict[str, pd.DataFrame], symbol: str, ts: pd.Timestamp, costs: dict[str, float]) -> float | None:
    df = prices.get(symbol)
    if df is None or ts not in df.index:
        return None
    i = int(df.loc[ts, "_i"])
    if i + 1 + HOLD_BARS >= len(df):
        return None
    entry = float(df.iloc[i + 1]["open"])
    exit_ = float(df.iloc[i + 1 + HOLD_BARS]["close"])
    gross = (entry - exit_) / entry * NOTIONAL
    cost = 2 * (costs["fee"] + costs["slip"]) / 10_000 * NOTIONAL + costs["funding_h"] / 10_000 * NOTIONAL * (HOLD_BARS * 5 / 60)
    return gross - cost


def block_bootstrap_ci(pnls: np.ndarray, weeks: np.ndarray, reps: int = BOOT_REPS, seed: int = SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(weeks)
    means = []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sample = np.concatenate([pnls[weeks == w] for w in pick])
        if len(sample):
            means.append(sample.mean())
    lo, hi = np.quantile(means, [0.05, 0.95])
    return {"expectancy_ci_lo": float(lo), "expectancy_ci_hi": float(hi)}


def strategy_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    pnl = trades["net"].to_numpy()
    if not len(pnl):
        return {"trades": 0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = np.cumsum(pnl)
    dd = float((np.maximum.accumulate(equity) - equity).max()) if len(equity) else 0.0
    by_sym = trades.groupby("symbol")["net"].sum()
    by_month = trades.groupby(trades["ts"].str[:7])["net"].sum()
    net = float(pnl.sum())
    pos_total = float(wins.sum()) if len(wins) else 0.0
    return {
        "trades": int(len(pnl)), "net_pnl": net, "expectancy": float(pnl.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None,
        "win_rate": float((pnl > 0).mean()), "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0, "max_drawdown": dd,
        "worst_trade": float(pnl.min()), "p5": float(np.quantile(pnl, 0.05)), "p50": float(np.quantile(pnl, 0.50)), "p95": float(np.quantile(pnl, 0.95)),
        "cvar_5": float(pnl[pnl <= np.quantile(pnl, 0.05)].mean()),
        "sharpe_diag": float(pnl.mean() / pnl.std()) if pnl.std() else None,
        "max_symbol_share": float((by_sym.max() / net)) if net > 0 and len(by_sym) else None,
        "max_month_share": float((by_month.max() / net)) if net > 0 and len(by_month) else None,
        "max_trade_share": float((pnl.max() / net)) if net > 0 else None,
        "fees_total": float(trades["cost"].sum()),
        **block_bootstrap_ci(pnl, trades["ts"].str[:7].astype(str).to_numpy() + "-w" + (pd.to_datetime(trades["ts"]).dt.isocalendar().week.astype(str)).to_numpy()),
    }


def run_backtest(args: argparse.Namespace) -> dict[str, Any]:
    stamp = utc_stamp()
    out_dir = ECON_ROOT / stamp
    validate_gen2_path(out_dir)
    out_dir.mkdir(parents=True)
    dev_all, _, feature_cols, _ = rv2.load_development(Path(args.dataset_csv))
    dev = dev_all[pd.to_numeric(dev_all["id.horizon"], errors="coerce") == eqm.PRIMARY_HORIZON].reset_index(drop=True)
    folds = rv2.make_folds(dev)
    trrm = eqm.load_trrm(Path(args.trrm_dir))
    tail = eqm.tail_scores(trrm, dev)
    sys.modules["__main__"].MedianImputer = MedianImputer
    with (Path(args.eqm_dir) / "eqm1_candidate.pkl").open("rb") as f:
        bundle = pickle.load(f)
    x = bundle["imputer"].transform(dev[feature_cols].apply(pd.to_numeric, errors="coerce"))
    s_reg = np.asarray(bundle["reg_model"].predict(x), dtype=float)
    s_clf = score_of(bundle["clf_model"], x)
    s_eqm = s_clf * s_reg if bundle["score_kind"] == "composite_ev" else s_reg
    mom = pd.to_numeric(dev.get("feature.momentum_12"), errors="coerce").fillna(0.0).to_numpy()
    below_ema = pd.to_numeric(dev.get("feature.close_vs_ema_24"), errors="coerce").fillna(0.0).to_numpy() < 0
    atr = pd.to_numeric(dev.get("feature.atr_proxy_24"), errors="coerce").fillna(0.0).to_numpy()
    rng = np.random.default_rng(SEED)
    rand_score = rng.random(len(dev))
    rule_mom = np.where((mom < 0) & below_ema, -mom, -np.inf)
    strategies = {
        "random_plus_trrm": (rand_score, True), "rule_momentum": (rule_mom, False), "rule_momentum_plus_trrm": (rule_mom, True),
        "rule_volatility_plus_trrm": (atr, True), "trrm_only": (-tail, False), "eqm_only": (s_eqm, False), "eqm_plus_trrm": (s_eqm, True),
    }
    symbols = sorted(dev["id.symbol"].unique())
    prices = load_prices(args.series_version, symbols)
    veto_rank = {}
    results: dict[str, Any] = {}
    per_fold_expectancy: dict[str, list[float]] = {k: [] for k in strategies}
    trades_frames: dict[str, list[pd.DataFrame]] = {k: [] for k in strategies}
    for fold in folds:
        e_idx = fold["eval_idx"]
        keep_n = max(1, int(round((1 - eqm.VETO_BUDGET) * len(e_idx))))
        retained = set(e_idx[np.argsort(tail[e_idx], kind="stable")[:keep_n]])
        budget_n = max(1, int(round(TOPK * keep_n)))
        for name, (score, use_veto) in strategies.items():
            pool = np.array([i for i in e_idx if (i in retained)] if use_veto else e_idx)
            ranked = pool[np.argsort(-score[pool], kind="stable")]
            sub = dev.iloc[ranked]
            mask = eqm.correlation_limit(sub.reset_index(drop=True), score[ranked])
            chosen = ranked[mask][:budget_n]
            rows = []
            for scen, costs in COST_SCENARIOS.items():
                for i in chosen:
                    pnl = trade_pnl(prices, dev.iloc[i]["id.symbol"], dev.iloc[i]["_ts"], costs)
                    if pnl is None:
                        continue
                    cost = 2 * (costs["fee"] + costs["slip"]) / 10_000 * NOTIONAL + costs["funding_h"] / 10_000 * NOTIONAL
                    rows.append({"scenario": scen, "fold": fold["name"], "symbol": dev.iloc[i]["id.symbol"], "ts": str(dev.iloc[i]["_ts"]), "net": pnl, "cost": cost, "strategy": name})
            frame = pd.DataFrame(rows)
            trades_frames[name].append(frame)
            base = frame[frame["scenario"] == "B_base"]
            per_fold_expectancy[name].append(float(base["net"].mean()) if len(base) else 0.0)
    for name in strategies:
        allf = pd.concat(trades_frames[name], ignore_index=True) if trades_frames[name] else pd.DataFrame()
        results[name] = {
            scen: strategy_metrics(allf[allf["scenario"] == scen].reset_index(drop=True)) for scen in COST_SCENARIOS
        }
        results[name]["per_fold_expectancy_base"] = per_fold_expectancy[name]
    all_trades = pd.concat([f for fr in trades_frames.values() for f in fr], ignore_index=True)
    all_trades.to_csv(out_dir / "trades.csv", index=False)
    payload = {
        "schema": "gen2_econ1_backtest_v1", "artifact_id": stamp, "spec": "GEN2_ECON1_SPEC.md v1.0",
        "dataset_sha256": sha256_file(Path(args.dataset_csv)), "eqm_sha256": sha256_file(Path(args.eqm_dir) / "eqm1_candidate.pkl"),
        "trrm_sha256": sha256_file(Path(args.trrm_dir) / "rv2_candidate.pkl"),
        "notional": NOTIONAL, "hold_bars": HOLD_BARS, "topk": TOPK, "cost_scenarios": COST_SCENARIOS,
        "strategies": results,
        "safety": {"no_live": True, "no_management": True, "equal_budget": True, "research_only": True},
    }
    (out_dir / "econ1_report.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    print(json.dumps({"artifact": str(out_dir), "eqm_plus_trrm_base": results["eqm_plus_trrm"]["B_base"],
                      "per_fold": per_fold_expectancy["eqm_plus_trrm"]}, indent=2, default=json_default))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEN2-ECON1 backtest")
    p.add_argument("--dataset-csv", default=str(rv2.DEFAULT_DATASET))
    p.add_argument("--trrm-dir", default=str(eqm.DEFAULT_TRRM_DIR))
    p.add_argument("--eqm-dir", required=True)
    p.add_argument("--series-version", default="v1")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_backtest(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
