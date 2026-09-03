"""Causal residual-alpha, regime and path-risk diagnostics for B2."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SIDES = ("LONG", "SHORT")
MECHANISMS = ("REGIME_MOMENTUM", "REGIME_REVERSAL", "RELATIVE_STRENGTH")
PRIMARY_COST = 0.0014
STRESS_COST = 0.0020
FAVORABLE_BARRIER = 0.0042
ADVERSE_BARRIER = 0.0042
RANK_FEATURES = (
    "return_1h", "return_4h", "return_24h", "relative_strength_btc_4h",
    "cross_sectional_return_rank_4h", "realized_volatility_4h",
    "realized_volatility_24h", "volume_persistence_1h", "taker_flow_1h",
    "distance_sma_4h", "distance_sma_24h", "extension_z_24h", "basis_z_7d",
    "basis_convergence_1h", "funding_rate", "beta_btc", "common_alt_state",
)


class B2ContractError(ValueError):
    pass


@dataclass(frozen=True)
class RegimeThresholds:
    trend_lower: float
    trend_upper: float
    volatility_lower: float
    volatility_upper: float


@dataclass(frozen=True)
class PairwiseRanker:
    pipeline: Any

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.pipeline.decision_function(frame.loc[:, list(RANK_FEATURES)]),
            dtype=float,
        )


def contract_hash(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("ascii")).hexdigest()


def _safe_spearman(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) < 3:
        return 0.0
    value = float(spearmanr(actual, predicted).statistic)
    return value if math.isfinite(value) else 0.0


def add_causal_residuals(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach past-only BTC beta and ex-post residual targets.

    Beta uses only state returns ending before the decision timestamp. Future
    return is used solely as the supervised target.
    """
    required = {
        "timestamp_ms", "symbol", "return_4h", "long_gross", "short_gross",
        "side", "mae", "mfe",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise B2ContractError(f"AEGIS_B2_COLUMNS_MISSING:{','.join(sorted(missing))}")
    base = rows.loc[rows["side"].eq("LONG")].copy()
    if base.duplicated(["timestamp_ms", "symbol"]).any():
        raise B2ContractError("AEGIS_B2_DUPLICATE_SYMBOL_EVENT")
    btc = base.loc[base.symbol.eq("BTCUSDT"), ["timestamp_ms", "return_4h"]].rename(
        columns={"return_4h": "btc_state_return"}
    )
    base = base.merge(btc, on="timestamp_ms", how="left", validate="many_to_one")
    pieces = []
    for _, symbol_rows in base.groupby("symbol", sort=True):
        ordered = symbol_rows.sort_values("timestamp_ms").copy()
        x = ordered["btc_state_return"].shift(1)
        y = ordered["return_4h"].shift(1)
        covariance = y.rolling(180, min_periods=84).cov(x)
        variance = x.rolling(180, min_periods=84).var()
        ordered["beta_btc"] = (covariance / variance.replace(0.0, np.nan)).clip(-3.0, 3.0)
        pieces.append(ordered[["timestamp_ms", "symbol", "beta_btc", "btc_state_return"]])
    beta = pd.concat(pieces, ignore_index=True)
    result = rows.merge(beta, on=["timestamp_ms", "symbol"], how="left", validate="many_to_one")
    result["future_price_return"] = (result["long_gross"] - result["short_gross"]) / 2.0
    btc_future = result.loc[
        result.side.eq("LONG") & result.symbol.eq("BTCUSDT"),
        ["timestamp_ms", "future_price_return"],
    ].rename(columns={"future_price_return": "btc_future_return"})
    result = result.merge(btc_future, on="timestamp_ms", how="left", validate="many_to_one")
    result["btc_neutral_future"] = (
        result["future_price_return"] - result["beta_btc"] * result["btc_future_return"]
    )
    common = (
        result.loc[result.side.eq("LONG") & result.symbol.ne("BTCUSDT")]
        .groupby("timestamp_ms")["btc_neutral_future"].median()
        .rename("common_alt_future")
    )
    result = result.merge(common, on="timestamp_ms", how="left", validate="many_to_one")
    state_common = (
        result.loc[result.side.eq("LONG") & result.symbol.ne("BTCUSDT")]
        .assign(state_neutral=lambda frame: frame.return_4h - frame.beta_btc * frame.btc_state_return)
        .groupby("timestamp_ms")["state_neutral"].median()
        .rename("common_alt_state")
    )
    result = result.merge(state_common, on="timestamp_ms", how="left", validate="many_to_one")
    sign = np.where(result.side.eq("LONG"), 1.0, -1.0)
    result["residual_return"] = sign * (
        result["btc_neutral_future"] - result["common_alt_future"]
    )
    result["residual_utility"] = result["residual_return"] - PRIMARY_COST - 0.25 * result["mae"]
    return result.dropna(subset=["beta_btc", "common_alt_future", "common_alt_state"]).reset_index(drop=True)


def fit_regime_thresholds(events: pd.DataFrame) -> RegimeThresholds:
    trend = events["btc_return_24h"].dropna()
    volatility = events["median_volatility_24h"].dropna()
    if len(trend) < 100 or len(volatility) < 100:
        raise B2ContractError("AEGIS_B2_REGIME_TRAINING_INSUFFICIENT")
    return RegimeThresholds(
        trend_lower=float(trend.quantile(0.33)),
        trend_upper=float(trend.quantile(0.67)),
        volatility_lower=float(volatility.quantile(0.33)),
        volatility_upper=float(volatility.quantile(0.67)),
    )


def assign_regimes(events: pd.DataFrame, thresholds: RegimeThresholds) -> pd.DataFrame:
    result = events.copy()
    transition = (
        result["btc_return_4h"].ne(0.0)
        & result["btc_return_24h"].ne(0.0)
        & (np.sign(result["btc_return_4h"]) != np.sign(result["btc_return_24h"]))
    )
    trend = np.select(
        [transition, result.btc_return_24h.ge(thresholds.trend_upper), result.btc_return_24h.le(thresholds.trend_lower)],
        ["TRANSITION", "TREND_UP", "TREND_DOWN"],
        default="RANGE",
    )
    volatility = np.select(
        [result.median_volatility_24h.le(thresholds.volatility_lower), result.median_volatility_24h.ge(thresholds.volatility_upper)],
        ["COMPRESSION", "EXPANSION"],
        default="NORMAL",
    )
    result["regime"] = pd.Series(trend, index=result.index) + "__" + pd.Series(volatility, index=result.index)
    return result


def mechanism_score(rows: pd.DataFrame, side: str, mechanism: str) -> pd.Series:
    sign = 1.0 if side == "LONG" else -1.0
    if mechanism == "REGIME_MOMENTUM":
        return sign * rows["return_4h"]
    if mechanism == "REGIME_REVERSAL":
        return -sign * rows["extension_z_24h"]
    if mechanism == "RELATIVE_STRENGTH":
        return sign * rows["relative_strength_btc_4h"]
    raise B2ContractError("AEGIS_B2_MECHANISM_INVALID")


def select_mechanism_rows(rows: pd.DataFrame, side: str, mechanism: str) -> pd.DataFrame:
    sample = rows.loc[rows.side.eq(side)].copy()
    sample["mechanism_score"] = mechanism_score(sample, side, mechanism)
    return (
        sample.sort_values(["timestamp_ms", "mechanism_score", "symbol"], ascending=[True, False, True])
        .drop_duplicates("timestamp_ms", keep="first")
    )


def choose_regime_mechanisms(
    rows: pd.DataFrame,
    train_timestamps: set[int],
    calibration_timestamps: set[int],
) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    choices: dict[str, tuple[str, str]] = {}
    evidence: dict[str, Any] = {}
    for regime in sorted(rows.regime.unique()):
        regime_rows = rows.loc[rows.regime.eq(regime)]
        candidates = []
        for side in SIDES:
            for mechanism in MECHANISMS:
                selected = select_mechanism_rows(regime_rows, side, mechanism)
                train = selected.loc[selected.timestamp_ms.isin(train_timestamps)]
                calibration = selected.loc[selected.timestamp_ms.isin(calibration_timestamps)]
                train_gross = float(train.gross_return.mean()) if len(train) else -math.inf
                calibration_gross = float(calibration.gross_return.mean()) if len(calibration) else -math.inf
                candidates.append({
                    "side": side, "mechanism": mechanism, "train_events": len(train),
                    "calibration_events": len(calibration), "train_gross": train_gross,
                    "calibration_gross": calibration_gross,
                })
        ordered = sorted(candidates, key=lambda item: (-item["train_gross"], item["side"], item["mechanism"]))
        winner = ordered[0]
        confirmed = winner["train_gross"] > 0.0 and winner["calibration_gross"] > 0.0
        if confirmed:
            choices[regime] = (winner["side"], winner["mechanism"])
        evidence[regime] = {"winner": winner, "confirmed": confirmed, "all_candidates": candidates}
    return choices, evidence


def _pairwise_training(rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    differences: list[np.ndarray] = []
    labels: list[int] = []
    for timestamp, group in rows.groupby("timestamp_ms", sort=True):
        ordered = group.sort_values("symbol")
        values = ordered.loc[:, list(RANK_FEATURES)].to_numpy(float)
        utility = ordered["residual_utility"].to_numpy(float)
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                if not math.isfinite(utility[left] - utility[right]) or utility[left] == utility[right]:
                    continue
                reverse = hashlib.sha256(f"b2-pair:{timestamp}:{left}:{right}".encode("ascii")).digest()[0] & 1
                first, second = (right, left) if reverse else (left, right)
                differences.append(values[first] - values[second])
                labels.append(int(utility[first] > utility[second]))
    if len(differences) < 500 or len(set(labels)) != 2:
        raise B2ContractError("AEGIS_B2_PAIRWISE_TRAINING_INSUFFICIENT")
    return pd.DataFrame(differences, columns=RANK_FEATURES), np.asarray(labels, dtype=int)


def fit_pairwise_rankers(rows: pd.DataFrame, train_timestamps: set[int]) -> dict[tuple[str, str], PairwiseRanker]:
    models: dict[tuple[str, str], PairwiseRanker] = {}
    training = rows.loc[rows.timestamp_ms.isin(train_timestamps)]
    for (side, regime), sample in training.groupby(["side", "regime"], sort=True):
        if len(sample) < 500:
            continue
        x, y = _pairwise_training(sample)
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=1807)),
        ]).fit(x, y)
        models[(str(side), str(regime))] = PairwiseRanker(pipeline)
    return models


def fit_path_models(rows: pd.DataFrame, train_timestamps: set[int]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        sample = rows.loc[rows.side.eq(side) & rows.timestamp_ms.isin(train_timestamps)]
        result[side] = {}
        for offset, target in enumerate(("mae", "mfe")):
            result[side][target] = HistGradientBoostingRegressor(
                max_iter=150, max_leaf_nodes=15, learning_rate=0.05,
                l2_regularization=1.0, random_state=1810 + offset,
            ).fit(sample.loc[:, list(RANK_FEATURES)], sample[target])
    return result


def add_barrier_outcomes(
    rows: pd.DataFrame,
    minute_frames: Mapping[str, pd.DataFrame],
    horizon_minutes: int,
) -> pd.DataFrame:
    records = []
    for symbol, sample in rows.groupby("symbol", sort=True):
        minute = minute_frames[symbol].sort_values("open_time", ignore_index=True)
        times = minute.open_time.to_numpy(np.int64)
        opens = minute.open.to_numpy(float)
        highs = minute.high.to_numpy(float)
        lows = minute.low.to_numpy(float)
        for row in sample.itertuples(index=False):
            entry_time = int(row.timestamp_ms) + 900_000
            begin = int(np.searchsorted(times, entry_time))
            end = begin + horizon_minutes
            if begin >= len(times) or end > len(times) or times[begin] != entry_time:
                continue
            entry = opens[begin]
            if row.side == "LONG":
                favorable = highs[begin:end] / entry - 1.0 >= FAVORABLE_BARRIER
                adverse = 1.0 - lows[begin:end] / entry >= ADVERSE_BARRIER
            else:
                favorable = 1.0 - lows[begin:end] / entry >= FAVORABLE_BARRIER
                adverse = highs[begin:end] / entry - 1.0 >= ADVERSE_BARRIER
            favorable_hits = np.flatnonzero(favorable)
            adverse_hits = np.flatnonzero(adverse)
            favorable_time = int(favorable_hits[0]) if len(favorable_hits) else horizon_minutes + 1
            adverse_time = int(adverse_hits[0]) if len(adverse_hits) else horizon_minutes + 1
            favorable_first = favorable_time < adverse_time
            outcome = "FAVORABLE_FIRST" if favorable_first else (
                "ADVERSE_FIRST_OR_SAME" if adverse_time <= horizon_minutes else "NO_DECISIVE_BARRIER"
            )
            records.append({
                "timestamp_ms": int(row.timestamp_ms), "symbol": symbol, "side": row.side,
                "barrier_outcome": outcome, "favorable_first": favorable_first,
                "favorable_hit_minute": favorable_time if favorable_time <= horizon_minutes else None,
                "adverse_hit_minute": adverse_time if adverse_time <= horizon_minutes else None,
            })
    barriers = pd.DataFrame(records)
    return rows.merge(barriers, on=["timestamp_ms", "symbol", "side"], how="inner", validate="one_to_one")


def grouped_rank_metrics(rows: pd.DataFrame, models: Mapping[tuple[str, str], PairwiseRanker]) -> dict[str, Any]:
    correlations, top, random = [], [], []
    scored_parts = []
    for key, sample in rows.groupby(["side", "regime"], sort=True):
        model = models.get((str(key[0]), str(key[1])))
        if model is None:
            continue
        scored_parts.append(sample.assign(rank_score=model.score(sample)))
    if not scored_parts:
        return {"events": 0, "grouped_spearman": 0.0}
    scored = pd.concat(scored_parts, ignore_index=True)
    for timestamp, group in scored.groupby("timestamp_ms", sort=True):
        correlations.append(_safe_spearman(group.residual_utility, group.rank_score))
        top.append(group.sort_values(["rank_score", "symbol"], ascending=[False, True]).iloc[0])
        digest = hashlib.sha256(f"b2-random:{timestamp}".encode("ascii")).digest()
        ordered = group.sort_values(["symbol", "side"])
        random.append(ordered.iloc[int.from_bytes(digest[:8], "big") % len(ordered)])
    top_frame, random_frame = pd.DataFrame(top), pd.DataFrame(random)
    return {
        "events": len(correlations), "grouped_spearman": float(np.mean(correlations)),
        "top_residual_utility": float(top_frame.residual_utility.mean()),
        "random_residual_utility": float(random_frame.residual_utility.mean()),
        "top_raw_net": float((top_frame.gross_return - PRIMARY_COST).mean()),
        "random_raw_net": float((random_frame.gross_return - PRIMARY_COST).mean()),
        "top_outperforms_random": float(top_frame.residual_utility.mean()) > float(random_frame.residual_utility.mean()),
    }


def select_combined(
    rows: pd.DataFrame,
    choices: Mapping[str, tuple[str, str]],
    rankers: Mapping[tuple[str, str], PairwiseRanker],
    path_models: Mapping[str, Mapping[str, Any]],
    mae_limits: Mapping[str, float],
    mfe_limits: Mapping[str, float],
) -> pd.DataFrame:
    selected = []
    for timestamp, event_rows in rows.groupby("timestamp_ms", sort=True):
        regime = str(event_rows.regime.iloc[0])
        choice = choices.get(regime)
        if choice is None:
            continue
        side, mechanism = choice
        candidates = event_rows.loc[event_rows.side.eq(side)].copy()
        ranker = rankers.get((side, regime))
        if ranker is None:
            continue
        candidates["rank_score"] = ranker.score(candidates)
        candidates["predicted_mae"] = path_models[side]["mae"].predict(candidates.loc[:, list(RANK_FEATURES)])
        candidates["predicted_mfe"] = path_models[side]["mfe"].predict(candidates.loc[:, list(RANK_FEATURES)])
        candidates["mechanism_score"] = mechanism_score(candidates, side, mechanism)
        candidate = candidates.sort_values(["rank_score", "mechanism_score", "symbol"], ascending=[False, False, True]).iloc[0]
        if candidate.predicted_mae > mae_limits[side] or candidate.predicted_mfe < mfe_limits[side]:
            continue
        selected.append({
            **candidate.to_dict(), "net_primary": float(candidate.gross_return - PRIMARY_COST),
            "net_stress": float(candidate.gross_return - STRESS_COST), "mechanism": mechanism,
        })
    return pd.DataFrame(selected)
