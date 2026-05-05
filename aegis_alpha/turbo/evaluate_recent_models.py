#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.edge.common import profit_factor, safe_float  # noqa: E402
from aegis_alpha.turbo.config import DEFAULT_TURBO_CONFIG, TURBO_VERSION  # noqa: E402
from aegis_alpha.turbo.recent_dataset import build_recent_dataset  # noqa: E402


BUCKETS = (0.10, 0.05, 0.03)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _model_path(side: str, lookback_days: int) -> Path:
    return DEFAULT_TURBO_CONFIG.model_dir / f"turbo_{side}_edge_{lookback_days}d_v010.joblib"


def _load_estimator(path: Path) -> Any | None:
    if not path.exists():
        return None
    bundle = joblib.load(path)
    return bundle.get("estimator") if isinstance(bundle, dict) else bundle


def _bucket_eval(scores: np.ndarray, returns: np.ndarray, mfe: np.ndarray, mae: np.ndarray, pct: float) -> dict[str, Any]:
    if len(scores) == 0:
        return {"count": 0, "avg_return_after_fees": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "avg_mfe": 0.0, "avg_mae": 0.0}
    threshold = np.quantile(scores, 1.0 - pct)
    mask = scores >= threshold
    selected = returns[mask]
    return {
        "count": int(mask.sum()),
        "threshold": safe_float(threshold),
        "avg_return_after_fees": safe_float(np.mean(selected)) if len(selected) else 0.0,
        "win_rate": safe_float(np.mean(selected > 0.0)) if len(selected) else 0.0,
        "profit_factor": safe_float(profit_factor(selected)) if len(selected) else 0.0,
        "avg_mfe": safe_float(np.mean(mfe[mask])) if mask.any() else 0.0,
        "avg_mae": safe_float(np.mean(mae[mask])) if mask.any() else 0.0,
    }


def evaluate_recent_models(symbol: str = DEFAULT_TURBO_CONFIG.symbol) -> dict[str, Any]:
    cfg = DEFAULT_TURBO_CONFIG
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    per_model: list[dict[str, Any]] = []
    latest_votes: list[str] = []
    enable_checks: list[bool] = []

    for lookback_days in cfg.lookback_days:
        dataset = build_recent_dataset(symbol, int(lookback_days), save=False)["dataset"]
        x = np.asarray(dataset["X"], dtype=np.float32)
        split = int(len(x) * 0.75)
        x_val = x[split:] if split < len(x) else x[-1:]
        long_return = np.asarray(dataset["long_net_return_12"], dtype=np.float32)[split:]
        short_return = np.asarray(dataset["short_net_return_12"], dtype=np.float32)[split:]
        mfe = np.asarray(dataset["mfe_12"], dtype=np.float32)[split:]
        mae = np.asarray(dataset["mae_12"], dtype=np.float32)[split:]

        latest_side = "neutral"
        latest_margin = 0.0
        side_scores: dict[str, float] = {}
        for side, returns in (("long", long_return), ("short", short_return)):
            estimator = _load_estimator(_model_path(side, int(lookback_days)))
            if estimator is None or len(x_val) == 0:
                per_model.append({"lookback_days": int(lookback_days), "side": side, "model_status": "missing", "count": 0})
                continue
            scores = estimator.predict(x_val).astype(np.float32)
            side_scores[side] = float(scores[-1])
            buckets = {f"top_{int(pct * 100)}pct": _bucket_eval(scores, returns, mfe, mae, pct) for pct in BUCKETS}
            top = buckets["top_10pct"]
            per_model.append(
                {
                    "lookback_days": int(lookback_days),
                    "side": side,
                    "model_status": "evaluated",
                    "direction_bias": safe_float(np.mean(scores > 0.0)),
                    "avg_score": safe_float(np.mean(scores)),
                    "latest_score": safe_float(scores[-1]),
                    "buckets": buckets,
                }
            )
            enable_checks.append(bool(top["count"] >= 10 and top["profit_factor"] > 1.1 and top["avg_mfe"] > top["avg_mae"]))
        if side_scores:
            latest_margin = abs(side_scores.get("long", 0.0) - side_scores.get("short", 0.0))
            if latest_margin > 0 and max(side_scores.values()) > 0.0:
                latest_side = "long" if side_scores.get("long", 0.0) >= side_scores.get("short", 0.0) else "short"
        latest_votes.append(latest_side)

    vote_counts = Counter(latest_votes)
    enough_agreement = max(vote_counts.get("long", 0), vote_counts.get("short", 0)) >= cfg.thresholds.min_agreement_count
    turbo_research_enabled = bool(enough_agreement and any(enable_checks))
    report = {
        "schema_version": "aegis_turbo_recent_eval_v1",
        "created_at": _utc_stamp(),
        "turbo_version": TURBO_VERSION,
        "symbol": symbol,
        "turbo_research_enabled": turbo_research_enabled,
        "enable_reason": "passed_initial_shadow_checks" if turbo_research_enabled else "insufficient_recent_eval_quality",
        "latest_votes": dict(vote_counts),
        "models": per_model,
    }
    path = cfg.log_dir / f"turbo_recent_eval_{_utc_stamp()}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_TURBO_CONFIG.symbol)
    args = parser.parse_args()
    evaluate_recent_models(args.symbol)


if __name__ == "__main__":
    main()
