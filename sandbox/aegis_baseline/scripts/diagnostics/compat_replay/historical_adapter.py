"""Minimal reconstruction of the frozen Gen2 ECON1 read-only evaluation path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .historical_loader import FROZEN_PICKLES, load_frozen_pickle
from .manifests import sha256_file


HOLD_BARS = 12
NOTIONAL = 100.0


def score_of(model: Any, values: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(values), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(model.predict(values), dtype=float)


def load_pickle(path: Path) -> dict[str, Any]:
    artifact_id = next((name for name, (frozen, _) in FROZEN_PICKLES.items() if frozen == path), None)
    if artifact_id is None:
        raise ValueError(f"historical pickle is not allowlisted: {path}")
    return dict(load_frozen_pickle(artifact_id).payload)


def make_folds(dev: pd.DataFrame) -> list[dict[str, np.ndarray]]:
    timestamps = dev["_ts"]; count = len(dev); embargo = pd.Timedelta(minutes=120)
    folds = []
    for index, (train_fraction, validation_fraction) in enumerate(((.50, .60), (.60, .70), (.70, .80), (.80, .90)), 1):
        train = np.arange(0, int(count * train_fraction)); raw = np.arange(int(count * train_fraction), int(count * validation_fraction))
        validation = raw[(timestamps.iloc[raw] > timestamps.iloc[train].max() + embargo).to_numpy()]
        half = timestamps.iloc[validation].quantile(.5)
        calibration = validation[(timestamps.iloc[validation] <= half).to_numpy()]
        evaluation_raw = validation[(timestamps.iloc[validation] > half).to_numpy()]
        evaluation = evaluation_raw[(timestamps.iloc[evaluation_raw] > timestamps.iloc[calibration].max() + embargo).to_numpy()]
        folds.append({"name": f"fold_{index}", "train": train, "calibration": calibration, "evaluation": evaluation})
    return folds


def correlation_limit(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    windows = pd.to_datetime(frame["id.timestamp"]).dt.floor("30min")
    ordered = pd.DataFrame({"window": windows.to_numpy(), "score": scores}).reset_index()
    retained = ordered.sort_values("score", ascending=False).drop_duplicates("window")["index"].to_numpy()
    mask = np.zeros(len(frame), dtype=bool); mask[retained] = True
    return mask


def load_canonical_prices(series_manifest_path: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    manifest = json.loads(series_manifest_path.read_text(encoding="utf-8"))
    included = manifest["included_symbols"]
    series_dir = series_manifest_path.parent
    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        path = series_dir / f"{symbol}_5m.csv"
        if symbol not in included or sha256_file(path) != included[symbol]["sha256"]:
            raise RuntimeError(f"COMPATIBILITY_REPLAY_BLOCKED: canonical series hash mismatch: {symbol}")
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        frame["_i"] = np.arange(len(frame))
        result[symbol] = frame.set_index("timestamp")
    return result


def historical_trade_pnl(
    prices: dict[str, pd.DataFrame], symbol: str, timestamp: pd.Timestamp, costs: dict[str, float]
) -> dict[str, float] | None:
    frame = prices.get(symbol)
    if frame is None or timestamp not in frame.index:
        return None
    index = int(frame.loc[timestamp, "_i"])
    if index + 1 + HOLD_BARS >= len(frame):
        return None
    entry = float(frame.iloc[index + 1]["open"])
    exit_price = float(frame.iloc[index + 1 + HOLD_BARS]["close"])
    gross = (entry - exit_price) / entry * NOTIONAL
    cost = (
        2 * (costs["fee"] + costs["slip"]) / 10_000 * NOTIONAL
        + costs["funding_h"] / 10_000 * NOTIONAL * (HOLD_BARS * 5 / 60)
    )
    return {"entry": entry, "exit": exit_price, "gross": gross, "cost": cost, "net": gross - cost}


def reproduce_historical_trades(
    dataset_path: Path,
    rv2_path: Path,
    eqm_path: Path,
    series_manifest_path: Path,
    costs: dict[str, float],
) -> pd.DataFrame:
    data = pd.read_csv(dataset_path).copy()
    data["_ts"] = pd.to_datetime(data["id.timestamp"], errors="raise")
    data = data.sort_values(["_ts", "id.symbol", "id.horizon"]).reset_index(drop=True)
    data = data[data["_ts"] <= pd.Timestamp("2026-04-26 23:59:59")]
    data = data[pd.to_numeric(data["id.horizon"], errors="coerce") == 12].reset_index(drop=True)
    features = [column for column in data.columns if column.startswith("feature.")]
    horizons = pd.to_numeric(data["id.horizon"], errors="coerce")
    horizon_columns = pd.DataFrame(
        {f"horizon_{horizon}": (horizons == horizon).astype(float) for horizon in (6, 12, 24)},
        index=data.index,
    )
    data = pd.concat([data, horizon_columns], axis=1)
    feature_columns = features + ["horizon_6", "horizon_12", "horizon_24"]
    trrm = load_pickle(rv2_path)
    tx = trrm["imputer"].transform(data[trrm["features"]].apply(pd.to_numeric, errors="coerce"))
    raw_tail = score_of(trrm["trrm_model"], tx)
    calibrator = trrm.get("calibrator")
    if calibrator is None:
        tail = raw_tail
    elif trrm.get("calibrator_kind") == "isotonic":
        tail = np.asarray(calibrator.predict(raw_tail), dtype=float)
    else:
        tail = np.asarray(calibrator.predict_proba(raw_tail.reshape(-1, 1))[:, 1], dtype=float)
    eqm = load_pickle(eqm_path)
    x = eqm["imputer"].transform(data[feature_columns].apply(pd.to_numeric, errors="coerce"))
    reg = np.asarray(eqm["reg_model"].predict(x), dtype=float)
    clean = score_of(eqm["clf_model"], x)
    scores = clean * reg if eqm["score_kind"] == "composite_ev" else reg
    selected: list[pd.DataFrame] = []
    for fold in make_folds(data):
        evaluation = fold["evaluation"]
        keep_n = max(1, int(round(.70 * len(evaluation))))
        retained = evaluation[np.argsort(tail[evaluation], kind="stable")[:keep_n]]
        budget = max(1, int(round(.10 * keep_n)))
        ranked = retained[np.argsort(-scores[retained], kind="stable")]
        subset = data.iloc[ranked].reset_index(drop=True)
        mask = correlation_limit(subset, scores[ranked])
        chosen = ranked[mask][:budget]
        frame = data.iloc[chosen][["id.symbol", "_ts"]].copy()
        frame["fold"] = fold["name"]
        frame["rv2_tail"] = tail[chosen]
        frame["eqm_score"] = scores[chosen]
        selected.append(frame)
    result = pd.concat(selected, ignore_index=True).rename(columns={"id.symbol": "symbol", "_ts": "ts"})
    prices = load_canonical_prices(series_manifest_path, sorted(result["symbol"].unique()))
    economics = [historical_trade_pnl(prices, row.symbol, row.ts, costs) for row in result.itertuples()]
    if any(item is None for item in economics):
        raise RuntimeError("COMPATIBILITY_REPLAY_BLOCKED: canonical price lookup failed")
    return pd.concat([result, pd.DataFrame(economics)], axis=1)
