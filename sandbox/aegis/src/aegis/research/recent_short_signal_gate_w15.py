"""Meta-label recent Aegis SHORT entries using contemporary signal evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = (
    "short_probability", "quality_probability", "tail_risk_probability",
    "qmae_q50", "qmae_q90", "uncertainty", "regime_confidence",
    "econ_expected_return_bps", "econ_calibrated_score", "eqm_score",
    "qmae_quality", "trrm_compatibility", "hour_sin", "hour_cos",
)


def _nested(row: dict[str, Any], *keys: str, default: float = math.nan) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def load_evidence(journal_dir: Path) -> pd.DataFrame:
    outcomes: dict[str, dict[str, Any]] = {}
    with (journal_dir / "outcomes_v1.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("label_valid"):
                outcomes[str(row["prospective_signal_id"])] = row
    records = []
    with (journal_dir / "signal_evidence_v1.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("side") != "SHORT" or _nested(row, "final_decision", "action", default="") != "ENTER_NOW":
                continue
            outcome = outcomes.get(str(row["prospective_signal_id"]))
            if outcome is None:
                continue
            timestamp = pd.Timestamp(row["signal_timestamp_utc"])
            upstream = row.get("upstream_model") or {}
            component = row.get("component_evidence") or {}
            regime = str(_nested(component, "d3", "output", "regime", default="UNKNOWN"))
            hour = timestamp.hour + timestamp.minute / 60
            records.append({
                "signal_id": row["prospective_signal_id"], "timestamp": timestamp,
                "symbol": row["symbol"], "regime": regime,
                "short_probability": float(upstream.get("short_probability", math.nan)),
                "quality_probability": float(upstream.get("quality_probability", math.nan)),
                "tail_risk_probability": float(upstream.get("tail_risk_probability", math.nan)),
                "qmae_q50": float(upstream.get("qmae_q50", math.nan)),
                "qmae_q90": float(upstream.get("qmae_q90", math.nan)),
                "uncertainty": float(upstream.get("uncertainty", math.nan)),
                "regime_confidence": float(_nested(component, "d3", "output", "regime_confidence")),
                "econ_expected_return_bps": float(_nested(component, "econ1", "output", "expected_return")) * 10_000,
                "econ_calibrated_score": float(_nested(component, "econ1", "output", "calibrated_score")),
                "eqm_score": float(_nested(component, "eqm", "output", "score")),
                "qmae_quality": float(_nested(component, "qmae", "output", "quality")),
                "trrm_compatibility": float(_nested(component, "trrm", "output", "compatibility")),
                "hour_sin": math.sin(2 * math.pi * hour / 24),
                "hour_cos": math.cos(2 * math.pi * hour / 24),
                "net_bps": float(outcome["net_return_fraction"]) * 10_000,
                "gross_bps": float(outcome["gross_return_fraction"]) * 10_000,
                "mfe_bps": float(outcome["mfe_fraction"]) * 10_000,
                "mae_bps": float(outcome["mae_fraction"]) * 10_000,
            })
    frame = pd.DataFrame(records).set_index("timestamp").sort_index()
    categories = pd.get_dummies(frame[["symbol", "regime"]], prefix=["symbol", "regime"], dtype=float)
    return pd.concat([frame, categories], axis=1)


def _model_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns, fill_value=0.0)


def _metrics(frame: pd.DataFrame, selected: np.ndarray, extra_cost_bps: float) -> dict[str, float | int]:
    net = frame.net_bps.to_numpy()[selected] - extra_cost_bps
    loss = -net[net < 0].sum()
    return {
        "signals": len(frame), "trades": int(selected.sum()), "coverage": float(selected.mean()),
        "net_bps_per_trade": float(net.mean()) if len(net) else 0.0,
        "net_bps_per_signal": float(net.sum() / len(frame)) if len(frame) else 0.0,
        "gross_bps_per_trade": float(frame.gross_bps.to_numpy()[selected].mean()) if selected.any() else 0.0,
        "profit_factor": float(net[net > 0].sum() / loss) if loss > 0 else 0.0,
        "win_rate": float((net > 0).mean()) if len(net) else 0.0,
    }


def _bootstrap(frame: pd.DataFrame, selected: np.ndarray, extra_cost: float, repetitions: int, seed: int) -> dict[str, float]:
    work = frame[["net_bps"]].copy()
    work["selected"] = selected
    work["block"] = work.index.floor("12h")
    blocks = [group for _, group in work.groupby("block")]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sample = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        chosen = np.concatenate([(part.net_bps - extra_cost).to_numpy()[part.selected.to_numpy()] for part in sample])
        values.append(float(chosen.mean()) if len(chosen) else 0.0)
    values = np.asarray(values)
    return {"ci95_low": float(np.quantile(values, .025)), "ci95_high": float(np.quantile(values, .975)),
            "probability_positive": float((values > 0).mean())}


def evaluate(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    split = config["splits"]
    subset = lambda bounds: frame[(frame.index >= pd.Timestamp(bounds[0])) & (frame.index <= pd.Timestamp(bounds[1]))]
    train, calibration, validation = subset(split["train"]), subset(split["calibration"]), subset(split["validation"])
    feature_columns = list(NUMERIC_FEATURES) + sorted(column for column in frame if column.startswith(("symbol_", "regime_")))
    train_x, calibration_x, validation_x = (_model_matrix(part, feature_columns) for part in (train, calibration, validation))
    candidates = []
    models: dict[str, Any] = {
        "regularized_logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=.2, class_weight="balanced", max_iter=1000, random_state=20260816)),
        "shallow_hgb": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_depth=3, max_iter=120, learning_rate=.05, l2_regularization=3, random_state=20260816)),
        "shallow_hgb_regression": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(max_depth=3, max_iter=120, learning_rate=.05, l2_regularization=3, random_state=20260816)),
    }
    for name, model in models.items():
        if name.endswith("regression"):
            model.fit(train_x, train.net_bps)
            score = model.predict(calibration_x)
        else:
            model.fit(train_x, train.net_bps.gt(0).astype(int))
            score = model.predict_proba(calibration_x)[:, 1]
        for quantile in config["threshold_quantiles"]:
            threshold = float(np.quantile(score, quantile))
            selected = score >= threshold
            metrics = _metrics(calibration, selected, 0.0)
            if metrics["trades"] >= config["minimum_calibration_trades"]:
                candidates.append((metrics["net_bps_per_signal"], name, quantile, threshold, metrics))
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, name, quantile, threshold, calibration_metrics = candidates[0]
    model = models[name]
    score = model.predict(validation_x) if name.endswith("regression") else model.predict_proba(validation_x)[:, 1]
    selected = score >= threshold
    recorded = _metrics(validation, selected, 0.0)
    baseline14 = _metrics(validation, selected, 4.0)
    stress20 = _metrics(validation, selected, 10.0)
    current = _metrics(validation, np.ones(len(validation), bool), 0.0)
    per_symbol = {symbol: _metrics(group, selected[validation.symbol.to_numpy() == symbol], 4.0) for symbol, group in validation.groupby("symbol")}
    positive_symbols = sum(value["net_bps_per_trade"] > 0 for value in per_symbol.values())
    bootstrap = _bootstrap(validation, selected, 4.0, config["statistics"]["bootstrap_repetitions"], config["statistics"]["seed"])
    passed = all((recorded["trades"] >= config["minimum_validation_trades"],
                  baseline14["net_bps_per_trade"] >= config["minimum_net_expectancy_bps_at_14bps"],
                  stress20["net_bps_per_trade"] > 0, positive_symbols >= config["minimum_positive_symbols"],
                  bootstrap["ci95_low"] > 0))
    return {"selected_model": name, "threshold_quantile": quantile, "frozen_threshold": threshold,
            "train_signals": len(train), "calibration": calibration_metrics, "validation_recorded_10bps": recorded,
            "validation_14bps": baseline14, "validation_20bps": stress20, "current_policy_validation": current,
            "positive_symbols_at_14bps": positive_symbols, "per_symbol_14bps": per_symbol, "bootstrap_14bps": bootstrap,
            "W15_RECENT_SHORT_SIGNAL_EDGE_FOUND": passed, "W15_READY_FOR_LIVE": False}
