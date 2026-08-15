"""Read-only evidence audit for historical Aegis Live entry quality."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GUARD_PATHS = {
    "entry_quality": "entry_quality",
    "event_risk": "event_risk",
    "decision_brain": "decision_brain",
    "clean_entry": "clean_entry",
    "regime": "regime",
    "probe_mode": "probe_mode",
}


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _nested(record: Mapping[str, Any], *keys: str) -> Any:
    current: Any = record
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_number(*values: Any) -> float:
    for value in values:
        result = _number(value)
        if math.isfinite(result):
            return result
    return math.nan


def stable_trade_hash(trade_id: str) -> str:
    return hashlib.sha256(trade_id.encode("utf-8")).hexdigest()


def classify_entry(
    *, pnl_usdt: float, mfe_bps: float, mae_bps: float, config: Mapping[str, Any]
) -> str:
    ratio = mfe_bps / max(mae_bps, 1e-9)
    good = (
        pnl_usdt > 0
        and mfe_bps >= float(config["good_min_mfe_bps"])
        and mae_bps <= float(config["good_max_mae_bps"])
        and ratio >= float(config["good_min_mfe_mae_ratio"])
    )
    bad = (
        pnl_usdt < 0
        and (
            mfe_bps < float(config["bad_low_mfe_bps"])
            or mae_bps >= float(config["bad_high_mae_bps"])
            or ratio <= float(config["bad_max_mfe_mae_ratio"])
        )
    ) or (
        mae_bps >= float(config["bad_extreme_mae_bps"])
        and ratio <= float(config["bad_extreme_max_mfe_mae_ratio"])
    )
    if good:
        return "GOOD_CLEAN_ENTRY"
    if bad:
        return "BAD_ENTRY"
    return "MIXED_OR_EXIT_DEPENDENT"


def read_trade_pairs(logs_dir: Path, source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    opens: dict[str, dict[str, Any]] = {}
    closes: dict[str, dict[str, Any]] = {}
    invalid_json = 0
    files = sorted(logs_dir.glob(str(source["trade_glob"])))
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json += 1
                    continue
                if row.get("strategy") != source["strategy"] or row.get("mode") != source["mode"]:
                    continue
                trade_id = row.get("trade_id")
                if not isinstance(trade_id, str):
                    continue
                if row.get("status") == "OPEN":
                    opens[trade_id] = row
                elif row.get("status") == "CLOSED":
                    closes[trade_id] = row
    pairs = [{"open": opens.get(key), "close": value} for key, value in sorted(closes.items())]
    audit = {
        "files": len(files),
        "unique_opens": len(opens),
        "unique_closes": len(closes),
        "paired": sum(item["open"] is not None for item in pairs),
        "unpaired": sum(item["open"] is None for item in pairs),
        "invalid_json": invalid_json,
    }
    return pairs, audit


def flatten_pair(pair: Mapping[str, Any], classification: Mapping[str, Any], split_at: pd.Timestamp) -> dict[str, Any]:
    opened = pair.get("open") or {}
    closed = pair["close"]
    metadata = opened.get("metadata") or {}
    policy = metadata.get("entryPolicy") or {}
    guards = policy.get("guards") or {}
    clean = metadata.get("cleanEntryGuard") or {}
    regime = policy.get("regime") or {}
    context = policy.get("regimeContext") or {}
    indicators = context.get("indicators") or {}
    long_risk = policy.get("longRiskShadow") or {}
    weakness = long_risk.get("marketWeakness") or {}
    leverage = _first_number(closed.get("leverage"), opened.get("leverage"), 1.0)
    leverage = leverage if leverage > 0 else 1.0
    mfe_bps = max(0.0, _number(closed.get("mfe_roe"))) / leverage * 10_000.0
    mae_bps = max(0.0, -_number(closed.get("mae_roe"))) / leverage * 10_000.0
    pnl = _number(closed.get("pnl_usdt"))
    opened_at = pd.Timestamp(closed.get("opened_at") or opened.get("opened_at"))
    label = classify_entry(pnl_usdt=pnl, mfe_bps=mfe_bps, mae_bps=mae_bps, config=classification)
    row: dict[str, Any] = {
        "trade_id_hash": stable_trade_hash(str(closed["trade_id"])),
        "trade_id": closed["trade_id"],
        "symbol": closed.get("symbol"),
        "side": closed.get("side"),
        "opened_at": opened_at.isoformat(),
        "closed_at": closed.get("closed_at"),
        "split": "VALIDATION" if opened_at >= split_at else "DISCOVERY",
        "entry_class": label,
        "bad_entry": int(label == "BAD_ENTRY"),
        "good_entry": int(label == "GOOD_CLEAN_ENTRY"),
        "pnl_usdt": pnl,
        "roe": _number(closed.get("roe")),
        "leverage": leverage,
        "mfe_bps_underlying": mfe_bps,
        "mae_bps_underlying": mae_bps,
        "mfe_mae_ratio": mfe_bps / max(mae_bps, 1e-9),
        "duration_minutes": _number(closed.get("duration_minutes")),
        "exit_type": _nested(closed, "metadata", "exit_type"),
        "position_fraction": _first_number(closed.get("position_fraction"), opened.get("position_fraction")),
        "turbo_score": _first_number(opened.get("turbo_score"), policy.get("turboScore")),
        "entry_quality_score": _first_number(clean.get("entryQualityScore"), _nested(guards, "entry_quality", "metadata", "entryQualityScore")),
        "tail_risk_score": _first_number(clean.get("tailRiskScore"), long_risk.get("tailRiskScore")),
        "regime_confidence": _first_number(regime.get("confidence"), context.get("confidence")),
        "chop_risk": _number(context.get("chopRisk")),
        "exhaustion_risk": _number(context.get("exhaustionRisk")),
        "atr_percentile": _number(indicators.get("atrPercentile")),
        "atr_pct": _number(indicators.get("atrPct")),
        "adx": _number(indicators.get("adx")),
        "choppiness": _number(indicators.get("choppiness")),
        "return30m": _number(weakness.get("return30m")),
        "return60m": _number(weakness.get("return60m")),
        "close_location": _number(weakness.get("closeLocation")),
        "volume_ratio": _number(weakness.get("volumeRatio")),
        "trend_direction": context.get("trendDirection"),
        "regime_label": context.get("label") or regime.get("regime"),
    }
    for output_name, guard_name in GUARD_PATHS.items():
        guard = guards.get(guard_name) or {}
        row[f"{output_name}_would_block"] = bool(guard.get("wouldBlock", False))
        row[f"{output_name}_reason"] = guard.get("reason")
    return row


def guard_summary(frame: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    subset = frame.loc[frame["split"] == split].copy()
    baseline_bad = float(subset["bad_entry"].mean()) if len(subset) else math.nan
    result: list[dict[str, Any]] = []
    for name in GUARD_PATHS:
        blocked = subset[f"{name}_would_block"].fillna(False).astype(bool)
        allowed = subset.loc[~blocked]
        bad_captured = float(subset.loc[blocked, "bad_entry"].sum() / max(1, subset["bad_entry"].sum()))
        good_retained = float(allowed["good_entry"].sum() / max(1, subset["good_entry"].sum()))
        result.append({
            "guard": name,
            "split": split,
            "blocked": int(blocked.sum()),
            "allowed": int((~blocked).sum()),
            "bad_capture_rate": bad_captured,
            "good_retention_rate": good_retained,
            "allowed_bad_rate": float(allowed["bad_entry"].mean()) if len(allowed) else math.nan,
            "baseline_bad_rate": baseline_bad,
            "allowed_pnl_usdt_secondary": float(allowed["pnl_usdt"].sum()),
        })
    return result


@dataclass(frozen=True)
class ModelAudit:
    threshold: float
    coefficients: list[dict[str, Any]]
    discovery: dict[str, Any]
    validation: dict[str, Any]
    bootstrap_bad_rate_reduction_ci95: tuple[float, float]
    gate_passed: bool


def _policy_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    blocked = probability >= threshold
    allowed = frame.loc[~blocked]
    baseline_bad = float(frame["bad_entry"].mean())
    allowed_bad = float(allowed["bad_entry"].mean()) if len(allowed) else math.nan
    return {
        "episodes": int(len(frame)),
        "blocked": int(blocked.sum()),
        "allowed": int((~blocked).sum()),
        "bad_capture_rate": float(frame.loc[blocked, "bad_entry"].sum() / max(1, frame["bad_entry"].sum())),
        "good_retention_rate": float(allowed["good_entry"].sum() / max(1, frame["good_entry"].sum())),
        "baseline_bad_rate": baseline_bad,
        "allowed_bad_rate": allowed_bad,
        "relative_bad_rate_reduction": (baseline_bad - allowed_bad) / max(baseline_bad, 1e-12),
        "allowed_pnl_usdt_secondary": float(allowed["pnl_usdt"].sum()),
        "baseline_pnl_usdt_secondary": float(frame["pnl_usdt"].sum()),
    }


def fit_bad_entry_model(frame: pd.DataFrame, config: Mapping[str, Any], rng_seed: int = 20260815) -> ModelAudit:
    features = list(config["features"])
    discovery = frame.loc[frame["split"] == "DISCOVERY"].reset_index(drop=True)
    validation = frame.loc[frame["split"] == "VALIDATION"].reset_index(drop=True)
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=float(config["c"]), class_weight="balanced", max_iter=int(config["maximum_iterations"]), random_state=rng_seed)),
    ])
    pipeline.fit(discovery[features], discovery["bad_entry"])
    train_probability = pipeline.predict_proba(discovery[features])[:, 1]
    candidates = []
    for threshold in config["threshold_grid"]:
        metrics = _policy_metrics(discovery, train_probability, float(threshold))
        if metrics["blocked"] >= int(config["minimum_discovery_rejections"]) and metrics["good_retention_rate"] >= float(config["minimum_good_retention"]):
            candidates.append((metrics["relative_bad_rate_reduction"], metrics["bad_capture_rate"], -float(threshold), float(threshold), metrics))
    if not candidates:
        raise RuntimeError("no discovery threshold satisfies the preregistered retention constraints")
    _, _, _, threshold, discovery_metrics = max(candidates)
    validation_probability = pipeline.predict_proba(validation[features])[:, 1]
    validation_metrics = _policy_metrics(validation, validation_probability, threshold)

    rng = np.random.default_rng(rng_seed)
    reductions = []
    for _ in range(10_000):
        indices = rng.integers(0, len(validation), len(validation))
        sample = validation.iloc[indices].reset_index(drop=True)
        metrics = _policy_metrics(sample, validation_probability[indices], threshold)
        reductions.append(metrics["relative_bad_rate_reduction"])
    ci = tuple(float(value) for value in np.quantile(reductions, [0.025, 0.975]))

    model = pipeline.named_steps["model"]
    transformed_names = pipeline.named_steps["impute"].get_feature_names_out(features)
    coefficients = sorted(
        ({"feature": str(name), "coefficient": float(value)} for name, value in zip(transformed_names, model.coef_[0])),
        key=lambda item: abs(item["coefficient"]), reverse=True,
    )
    gate = config["validation_gate"]
    gate_passed = (
        validation_metrics["allowed"] >= int(gate["minimum_validation_allowed"])
        and validation_metrics["relative_bad_rate_reduction"] >= float(gate["minimum_relative_bad_rate_reduction"])
        and validation_metrics["good_retention_rate"] >= float(gate["minimum_good_retention"])
        and ci[0] > 0.0
    )
    return ModelAudit(threshold, coefficients, discovery_metrics, validation_metrics, ci, gate_passed)


def class_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (split, label), group in frame.groupby(["split", "entry_class"], sort=True):
        rows.append({
            "split": split,
            "entry_class": label,
            "episodes": int(len(group)),
            "pnl_usdt_secondary": float(group["pnl_usdt"].sum()),
            "median_mae_bps": float(group["mae_bps_underlying"].median()),
            "median_mfe_bps": float(group["mfe_bps_underlying"].median()),
            "median_mfe_mae_ratio": float(group["mfe_mae_ratio"].median()),
        })
    return rows
