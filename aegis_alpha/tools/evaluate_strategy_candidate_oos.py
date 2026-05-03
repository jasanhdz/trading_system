#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aegis_alpha.edge.common import load_model_bundle, safe_float
from aegis_alpha.tools.build_long_edge_candidate_dataset import compact_feature_names, load_meta_market
from aegis_alpha.tools.evaluate_long_edge_adaptive_meta import DynamicSizingConfig, _evaluate_dynamic_sizing_window


DEFAULT_CANDIDATE = Path("aegis_alpha/models/strategy_candidates/aegis_long_edge_dynamic_v042.json")
DEFAULT_OUTPUT_DIR = Path("aegis_alpha/logs/edge")
DEFAULT_SEEDS = (6101, 7331)


def _load_candidate(path: Path) -> dict[str, Any]:
    candidate = json.loads(path.read_text(encoding="utf-8"))
    policy = candidate.get("policy", {})
    sizing = policy.get("dynamic_sizing", {})
    if candidate.get("status") != "OFFLINE_CANDIDATE":
        raise RuntimeError(f"Candidate status is not OFFLINE_CANDIDATE: {candidate.get('status')}")
    return candidate


def _add_window(windows: dict[int, set[str]], start: int, source: str, min_start: int, max_start: int) -> None:
    if min_start <= start <= max_start:
        windows[int(start)].add(source)


def _select_oos_windows(
    market: Any,
    window_steps: int,
    seeds: tuple[int, ...],
    recent_windows: int,
    random_windows_per_seed: int,
    non_overlap_windows: int,
    target_max_windows: int,
) -> list[tuple[int, str]]:
    min_start = market.cfg.model.window_size
    max_start = len(market.close) - window_steps - 1
    if max_start < min_start:
        raise RuntimeError(f"Not enough history for window_steps={window_steps}")

    windows: dict[int, set[str]] = defaultdict(set)
    timestamps = pd.to_datetime(market.timestamps)
    months = timestamps.to_period("M")
    seen_months: set[str] = set()
    for idx in range(min_start, max_start + 1):
        month = str(months[idx])
        if month in seen_months:
            continue
        seen_months.add(month)
        _add_window(windows, idx, f"monthly:{month}", min_start, max_start)

    for idx in range(recent_windows):
        _add_window(windows, max_start - idx * window_steps, "recent", min_start, max_start)

    non_overlap = np.arange(min_start, max_start + 1, window_steps, dtype=np.int64)
    if len(non_overlap) > non_overlap_windows:
        picks = np.linspace(0, len(non_overlap) - 1, non_overlap_windows).round().astype(int)
        non_overlap = non_overlap[picks]
    for start in non_overlap:
        _add_window(windows, int(start), "non_overlap", min_start, max_start)

    candidates = np.arange(min_start, max_start + 1, dtype=np.int64)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        picks = rng.choice(candidates, size=min(random_windows_per_seed, len(candidates)), replace=False)
        for start in picks:
            _add_window(windows, int(start), f"random_seed:{seed}", min_start, max_start)

    items = [(start, "+".join(sorted(sources))) for start, sources in windows.items()]
    if len(items) > target_max_windows:
        ordered = sorted(items)
        keep = np.linspace(0, len(ordered) - 1, target_max_windows).round().astype(int)
        items = [ordered[idx] for idx in keep]
    items = sorted(items)
    if len(items) < 100:
        raise RuntimeError(f"OOS selection produced only {len(items)} windows; need >=100")
    return items


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
        "skipped_by_meta": int(sum(w["skipped_by_meta"] for w in windows)),
        "skipped_by_guard": int(sum(w["skipped_by_guard"] for w in windows)),
    }


def _rank(fee_reports: list[dict[str, Any]], benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for report in fee_reports:
        summary = report["summary"]
        rows.append(
            {
                "fee_multiplier": report["fee_multiplier"],
                **summary,
                "beats_benchmark_p25_pf": bool(summary["p25_pf"] > float(benchmark["p25_pf"])),
                "passes_oos_criteria": bool(
                    summary["worst_balance"] >= 19.00
                    and summary["worst_max_dd"] <= 0.08
                    and summary["profitable_window_pct"] >= 0.70
                    and summary["median_balance"] >= 20.10
                    and summary["median_trades"] >= 5.0
                ),
                "fee_125x_floor_ok": bool(report["fee_multiplier"] != 1.25 or summary["worst_balance"] >= 18.75),
            }
        )

    def score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row["passes_oos_criteria"]),
            float(row["fee_125x_floor_ok"]),
            float(row["p25_pf"]),
            float(row["worst_balance"]),
            -float(row["worst_max_dd"]),
        )

    return sorted(rows, key=score, reverse=True)


def run_candidate_oos(
    candidate_path: Path,
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
    sizing = candidate["policy"]["dynamic_sizing"]
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
    meta_prob_cache: dict[int, float] = {}
    benchmark = candidate["freeze"]
    fee_reports: list[dict[str, Any]] = []
    for fee_multiplier in (1.0, 1.25):
        config = DynamicSizingConfig(
            full_size=float(sizing["full_size"]),
            reduced_size=float(sizing["reduced_size"]),
            meta_high_threshold=float(sizing["meta_high_threshold"]),
            meta_low_threshold=sizing["meta_low_threshold"],
            fee_multiplier=fee_multiplier,
        )
        evals = [
            _evaluate_dynamic_sizing_window(
                market=market,
                classifier=classifier,
                meta_prob_cache=meta_prob_cache,
                config=config,
                gate_threshold=float(candidate["policy"]["gate_threshold"]),
                start_step=start_step,
                window_steps=window_steps,
                source=source,
                max_hold_steps=24,
                close_edge_threshold=0.0,
                take_profit_roe=0.06,
            )
            for start_step, source in windows
        ]
        compact_windows = [
            {k: v for k, v in window.items() if k != "trades_detail"}
            for window in evals
        ]
        summary = _summary(compact_windows, market.cfg.risk.initial_balance)
        fee_reports.append(
            {
                "fee_multiplier": fee_multiplier,
                "config": {
                    "full_size": config.full_size,
                    "reduced_size": config.reduced_size,
                    "meta_high_threshold": config.meta_high_threshold,
                    "meta_low_threshold": config.meta_low_threshold,
                    "fee_multiplier": fee_multiplier,
                },
                "summary": summary,
                "windows": compact_windows,
            }
        )
        print(
            f"fee={fee_multiplier:.2f} p25pf={summary['p25_pf']:.2f} worst={summary['worst_balance']:.2f} "
            f"dd={summary['worst_max_dd']:.1%} prof={summary['profitable_window_pct']:.1%} "
            f"trades={summary['median_trades']:.1f} full={summary['full_size_trades']} reduced={summary['reduced_size_trades']}"
        )

    ranking = _rank(fee_reports, benchmark)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "aegis_strategy_candidate_oos_v1",
        "created_at": created_at,
        "candidate_path": str(candidate_path),
        "config_path": config_path,
        "candidate": candidate,
        "policy": {
            "side": "LONG_ONLY",
            "entry_gate": "top_3pct_expected_return_long",
            "allowed_regimes": ["mixed", "chop", "high_vol"],
            "risk_guard": "loss7_pause48_pause2_48_maxday3",
            "dynamic_sizing": sizing,
            "short_entries": False,
        },
        "window_count": len(windows),
        "window_steps": window_steps,
        "seeds": list(seeds),
        "selection": {
            "recent_windows": recent_windows,
            "random_windows_per_seed": random_windows_per_seed,
            "non_overlap_windows": non_overlap_windows,
            "target_max_windows": target_max_windows,
            "sources": {
                "monthly": True,
                "recent": recent_windows,
                "random_seeds": list(seeds),
                "non_overlap": non_overlap_windows,
            },
        },
        "benchmark_v042": benchmark,
        "success_criteria": {
            "worst_balance": ">=19.00",
            "worst_max_dd": "<=8%",
            "profitable_window_pct": ">=70%",
            "median_balance": ">=20.10",
            "median_trades": ">=5",
            "fee_125x_floor": "worst_balance >= 18.75",
        },
        "passes_any": bool(any(row["passes_oos_criteria"] for row in ranking)),
        "best_fee": ranking[0] if ranking else None,
        "ranking": ranking,
        "fee_reports": fee_reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"strategy_candidate_oos_{created_at}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report saved -> {output_path}")
    if ranking:
        best = ranking[0]
        print(
            f"Best fee={best['fee_multiplier']:.2f} p25pf={best['p25_pf']:.2f} worst={best['worst_balance']:.2f} "
            f"dd={best['worst_max_dd']:.1%} prof={best['profitable_window_pct']:.1%} "
            f"trades={best['median_trades']:.1f} pass={best['passes_oos_criteria']}"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
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
    run_candidate_oos(
        candidate_path=Path(args.candidate),
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
