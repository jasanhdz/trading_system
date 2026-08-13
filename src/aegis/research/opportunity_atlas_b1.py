"""Direction-agnostic opportunity atlas and decomposed baseline models."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EVENT_FEATURES = (
    "btc_return_1h", "btc_return_4h", "btc_return_24h", "breadth_4h",
    "return_dispersion_1h", "return_dispersion_4h", "median_volatility_24h",
    "max_volatility_24h", "median_volume_persistence", "max_volume_persistence",
    "mean_abs_taker_flow", "basis_dispersion", "funding_dispersion",
    "utc_hour_sin", "utc_hour_cos", "weekday_sin", "weekday_cos",
)
SYMBOL_FEATURES = (
    "return_1h", "return_4h", "return_24h", "relative_strength_btc_4h",
    "cross_sectional_return_rank_4h", "realized_volatility_4h",
    "realized_volatility_24h", "volume_persistence_1h", "taker_flow_1h",
    "distance_sma_4h", "distance_sma_24h", "extension_z_24h", "basis_z_7d",
    "basis_convergence_1h", "funding_rate",
)
SIDES = ("LONG", "SHORT")
PRIMARY_COST = 0.0014
OPPORTUNITY_GROSS = 0.0042


class B1ContractError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenModels:
    opportunity: Any
    direction: Any
    ranking: Mapping[str, Any]
    mae: Mapping[str, Any]
    mfe: Mapping[str, Any]
    opportunity_threshold: float


def feature_contract_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("ascii")).hexdigest()


def event_features(panel: pd.DataFrame, symbols_required: int = 11) -> pd.DataFrame:
    hourly = panel.loc[panel["timestamp_ms"].mod(4 * 3_600_000).eq(0)].copy()
    counts = hourly.groupby("timestamp_ms")["symbol"].nunique()
    valid = counts.loc[counts.eq(symbols_required)].index
    hourly = hourly.loc[hourly["timestamp_ms"].isin(valid)]
    grouped = hourly.groupby("timestamp_ms", sort=True)
    result = grouped.agg(
        btc_return_1h=("return_1h", lambda values: float(values.loc[hourly.loc[values.index, "symbol"].eq("BTCUSDT")].iloc[0])),
        btc_return_4h=("btc_return_4h", "first"),
        btc_return_24h=("return_24h", lambda values: float(values.loc[hourly.loc[values.index, "symbol"].eq("BTCUSDT")].iloc[0])),
        breadth_4h=("breadth_4h", "first"),
        return_dispersion_1h=("return_1h", "std"),
        return_dispersion_4h=("return_4h", "std"),
        median_volatility_24h=("realized_volatility_24h", "median"),
        max_volatility_24h=("realized_volatility_24h", "max"),
        median_volume_persistence=("volume_persistence_1h", "median"),
        max_volume_persistence=("volume_persistence_1h", "max"),
        mean_abs_taker_flow=("taker_flow_1h", lambda values: float(values.abs().mean())),
        basis_dispersion=("mark_spot_basis", "std"),
        funding_dispersion=("funding_rate", "std"),
        utc_hour_sin=("utc_hour_sin", "first"), utc_hour_cos=("utc_hour_cos", "first"),
        weekday_sin=("weekday_sin", "first"), weekday_cos=("weekday_cos", "first"),
    ).reset_index()
    return result.dropna(subset=list(EVENT_FEATURES)).reset_index(drop=True)


def attach_path_targets(
    states: pd.DataFrame,
    minute_frames: Mapping[str, pd.DataFrame],
    funding_frames: Mapping[str, pd.DataFrame],
    horizon_minutes: int,
) -> pd.DataFrame:
    outputs = []
    for symbol, rows in states.groupby("symbol", sort=True):
        minute = minute_frames[symbol].sort_values("open_time", ignore_index=True)
        times = minute["open_time"].to_numpy(np.int64)
        opens = minute["open"].to_numpy(float)
        highs = minute["high"].to_numpy(float)
        lows = minute["low"].to_numpy(float)
        funding = funding_frames[symbol]
        for row in rows.itertuples(index=False):
            entry_time = int(row.timestamp_ms) + 900_000
            exit_time = entry_time + horizon_minutes * 60_000
            begin, end = int(np.searchsorted(times, entry_time)), int(np.searchsorted(times, exit_time))
            if begin >= len(times) or end >= len(times) or times[begin] != entry_time or times[end] != exit_time:
                continue
            path_high, path_low = highs[begin:end], lows[begin:end]
            if len(path_high) != horizon_minutes:
                continue
            entry = float(opens[begin])
            price_return = float(opens[end]) / entry - 1.0
            paid = float(funding.loc[
                funding["funding_time"].ge(entry_time) & funding["funding_time"].lt(exit_time),
                "funding_rate",
            ].sum())
            outputs.append({
                **row._asdict(), "horizon_minutes": horizon_minutes,
                "long_gross": price_return - paid, "short_gross": -price_return + paid,
                "long_mae": max(0.0, 1.0 - float(path_low.min()) / entry),
                "short_mae": max(0.0, float(path_high.max()) / entry - 1.0),
                "long_mfe": max(0.0, float(path_high.max()) / entry - 1.0),
                "short_mfe": max(0.0, 1.0 - float(path_low.min()) / entry),
            })
    return pd.DataFrame(outputs)


def build_event_targets(events: pd.DataFrame, symbol_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for timestamp, rows in symbol_rows.groupby("timestamp_ms", sort=True):
        long_index, short_index = rows["long_gross"].idxmax(), rows["short_gross"].idxmax()
        long_best, short_best = rows.loc[long_index], rows.loc[short_index]
        best = long_best if long_best.long_gross >= short_best.short_gross else short_best
        side = "LONG" if long_best.long_gross >= short_best.short_gross else "SHORT"
        gross = float(max(long_best.long_gross, short_best.short_gross))
        records.append({
            "timestamp_ms": int(timestamp), "opportunity": gross >= OPPORTUNITY_GROSS,
            "best_side": side, "best_symbol": str(best.symbol), "best_gross": gross,
            "market_abs_return": abs(float(rows.loc[rows.symbol.eq("BTCUSDT"), "long_gross"].iloc[0])),
            "best_residual_abs_return": float((rows["long_gross"] - rows.loc[rows.symbol.eq("BTCUSDT"), "long_gross"].iloc[0]).abs().max()),
        })
    return events.merge(pd.DataFrame(records), on="timestamp_ms", validate="one_to_one")


def symbol_side_rows(symbol_rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for side in SIDES:
        lower = side.lower()
        frame = symbol_rows.copy()
        frame["side"] = side
        frame["gross_return"] = frame[f"{lower}_gross"]
        frame["mae"] = frame[f"{lower}_mae"]
        frame["mfe"] = frame[f"{lower}_mfe"]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def partition_mask(frame: pd.DataFrame, start: str, end: str) -> pd.Series:
    lower = int(pd.Timestamp(start).timestamp() * 1000)
    upper = int(pd.Timestamp(end).timestamp() * 1000)
    return frame["timestamp_ms"].ge(lower) & frame["timestamp_ms"].lt(upper)


def _hist_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=1801)


def _hist_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, random_state=seed)


def fit_models(events: pd.DataFrame, rows: pd.DataFrame, train: pd.Series, calibration: pd.Series) -> FrozenModels:
    opportunity = _hist_classifier().fit(events.loc[train, list(EVENT_FEATURES)], events.loc[train, "opportunity"])
    positive = train & events["opportunity"]
    direction = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", random_state=1802, max_iter=1000)),
    ]).fit(events.loc[positive, list(EVENT_FEATURES)], events.loc[positive, "best_side"].eq("LONG"))
    ranking, mae, mfe = {}, {}, {}
    row_train = rows["timestamp_ms"].isin(events.loc[train, "timestamp_ms"])
    for side in SIDES:
        mask = row_train & rows["side"].eq(side)
        ranking[side] = _hist_regressor(1803).fit(rows.loc[mask, list(SYMBOL_FEATURES)], rows.loc[mask, "gross_return"])
        mae[side] = _hist_regressor(1804).fit(rows.loc[mask, list(SYMBOL_FEATURES)], rows.loc[mask, "mae"])
        mfe[side] = _hist_regressor(1805).fit(rows.loc[mask, list(SYMBOL_FEATURES)], rows.loc[mask, "mfe"])
    calibration_probability = opportunity.predict_proba(events.loc[calibration, list(EVENT_FEATURES)])[:, 1]
    threshold = float(np.quantile(calibration_probability, 0.90))
    return FrozenModels(opportunity, direction, ranking, mae, mfe, threshold)


def safe_spearman(actual: Sequence[float], predicted: Sequence[float]) -> float:
    value = float(spearmanr(actual, predicted).statistic)
    return value if math.isfinite(value) else 0.0


def component_metrics(models: FrozenModels, events: pd.DataFrame, rows: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    sample = events.loc[mask]
    probability = models.opportunity.predict_proba(sample.loc[:, list(EVENT_FEATURES)])[:, 1]
    positive = sample["opportunity"]
    opportunity = {
        "events": len(sample), "base_rate": float(positive.mean()),
        "roc_auc": float(roc_auc_score(positive, probability)),
        "brier": float(brier_score_loss(positive, probability)),
        "base_rate_brier": float(brier_score_loss(positive, np.full(len(sample), positive.mean()))),
    }
    direction_sample = sample.loc[positive]
    direction_probability = models.direction.predict_proba(direction_sample.loc[:, list(EVENT_FEATURES)])[:, 1]
    direction_prediction = np.where(direction_probability >= 0.5, "LONG", "SHORT")
    direction = {
        "events": len(direction_sample),
        "balanced_accuracy": float(balanced_accuracy_score(direction_sample["best_side"], direction_prediction)),
    }
    row_mask = rows["timestamp_ms"].isin(sample["timestamp_ms"])
    ranking, path = {}, {}
    for side in SIDES:
        side_rows = rows.loc[row_mask & rows["side"].eq(side)]
        predicted_return = models.ranking[side].predict(side_rows.loc[:, list(SYMBOL_FEATURES)])
        predicted_mae = models.mae[side].predict(side_rows.loc[:, list(SYMBOL_FEATURES)])
        predicted_mfe = models.mfe[side].predict(side_rows.loc[:, list(SYMBOL_FEATURES)])
        ranked = side_rows.assign(predicted_return=predicted_return)
        top = (
            ranked.sort_values(
                ["timestamp_ms", "predicted_return", "symbol"],
                ascending=[True, False, True],
            )
            .drop_duplicates("timestamp_ms", keep="first")
        )
        random_rows = []
        for timestamp, group in side_rows.groupby("timestamp_ms", sort=True):
            digest = hashlib.sha256(
                f"b1-rank:{side}:{int(timestamp)}".encode("ascii")
            ).digest()
            ordered = group.sort_values("symbol")
            random_rows.append(
                ordered.iloc[int.from_bytes(digest[:8], "big") % len(ordered)]
            )
        random_frame = pd.DataFrame(random_rows)
        ranking[side] = {
            "events": len(side_rows),
            "spearman": safe_spearman(side_rows["gross_return"], predicted_return),
            "top_rank_gross_expectancy": float(top["gross_return"].mean()),
            "random_gross_expectancy": float(random_frame["gross_return"].mean()),
            "top_rank_outperforms_random": float(top["gross_return"].mean())
            > float(random_frame["gross_return"].mean()),
        }
        path[side] = {
            "mae_spearman": safe_spearman(side_rows["mae"], predicted_mae),
            "mfe_spearman": safe_spearman(side_rows["mfe"], predicted_mfe),
        }
    return {"opportunity": opportunity, "direction": direction, "ranking": ranking, "path_risk": path}


def combined_policy(models: FrozenModels, events: pd.DataFrame, rows: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    selected = []
    for event in events.loc[mask].itertuples(index=False):
        event_frame = pd.DataFrame([{name: getattr(event, name) for name in EVENT_FEATURES}])
        opportunity_probability = float(models.opportunity.predict_proba(event_frame)[0, 1])
        if opportunity_probability < models.opportunity_threshold:
            continue
        long_probability = float(models.direction.predict_proba(event_frame)[0, 1])
        if long_probability >= 0.55:
            side = "LONG"
        elif long_probability <= 0.45:
            side = "SHORT"
        else:
            continue
        candidates = rows.loc[rows["timestamp_ms"].eq(event.timestamp_ms) & rows["side"].eq(side)].copy()
        candidates["predicted_gross"] = models.ranking[side].predict(candidates.loc[:, list(SYMBOL_FEATURES)])
        candidates["predicted_mae"] = models.mae[side].predict(candidates.loc[:, list(SYMBOL_FEATURES)])
        candidate = candidates.sort_values(["predicted_gross", "symbol"], ascending=[False, True]).iloc[0]
        if candidate.predicted_gross < OPPORTUNITY_GROSS or candidate.predicted_mae > 0.01:
            continue
        selected.append({
            "timestamp_ms": int(event.timestamp_ms), "symbol": str(candidate.symbol), "side": side,
            "opportunity_probability": opportunity_probability, "direction_long_probability": long_probability,
            "predicted_gross": float(candidate.predicted_gross), "predicted_mae": float(candidate.predicted_mae),
            "gross_return": float(candidate.gross_return), "net_primary": float(candidate.gross_return - PRIMARY_COST),
            "net_stress": float(candidate.gross_return - 0.0020), "mae": float(candidate.mae), "mfe": float(candidate.mfe),
        })
    return pd.DataFrame(selected)


def economic_summary(rows: pd.DataFrame, column: str = "net_primary") -> dict[str, Any]:
    if rows.empty:
        return {"events": 0}
    values = rows[column].to_numpy(float)
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    ordered = rows.sort_values("timestamp_ms")
    thirds = [float(ordered.iloc[i * len(rows) // 3:(i + 1) * len(rows) // 3][column].mean()) for i in range(3)]
    return {
        "events": len(rows), "expectancy": float(values.mean()),
        "profit_factor": float(gains / losses) if losses else (math.inf if gains else 0.0),
        "win_rate": float((values > 0).mean()), "mean_mae": float(rows["mae"].mean()),
        "mean_mfe": float(rows["mfe"].mean()), "cvar_05": float(np.sort(values)[:max(1, math.ceil(len(values) * 0.05))].mean()),
        "positive_symbols": int((rows.groupby("symbol")[column].mean() > 0).sum()),
        "maximum_symbol_share": float(rows.symbol.value_counts(normalize=True).max()), "temporal_thirds": thirds,
    }
