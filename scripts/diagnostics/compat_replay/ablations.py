"""Closed causal ablations against the frozen Gen2 Stage-0 control.

This module is diagnostic-only.  It reconstructs the two frozen scientific
paths needed by the preregistered ablations and cannot publish production
artifacts or access the semi-blind window.
"""

from __future__ import annotations

import ast
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .historical_adapter import (
    correlation_limit,
    historical_trade_pnl,
    load_canonical_prices,
    load_pickle,
    make_folds,
    score_of,
)
from .manifests import atomic_write, digest, sha256_file
from .schemas import ReplayConfig


DEV_END = pd.Timestamp("2026-04-26 23:59:59")
E2_LAST_OPEN = pd.Timestamp("2026-04-26 22:55:00")  # close_time=23:00, the frozen E2 last anchor
HISTORICAL_SOURCE_COMMIT = "1841dbf94b6e4c674917f0dbbeb2fbfa46f1181d"
HISTORICAL_FEATURE_SOURCE = "aegis_alpha/tools/build_trrm_causal_feature_dataset_d.py"
HISTORICAL_FEATURE_SOURCE_SHA256 = "3fc04e9bb0cce2266941c5edc3b26db9aa91d106546ee7d3633331fa64984b3b"
E2_ECON_SHA256 = "ec6596febe3af8b9b299d0f6eef3b8a070a2a9b15d50ea4eedafed23e3febb50"
E2_THRESHOLD = 4.047730415717134e-05
BASE_COST_FRACTION = 0.0015
SEED = 42
DETERMINISM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class StageDefinition:
    stage: str
    parent: str
    changed_axis: str
    frozen_axes: tuple[str, ...]
    retraining: str

    def validate(self) -> None:
        if self.changed_axis.count("+") or "," in self.changed_axis:
            raise ValueError("ABLATION_PROTOCOL_BLOCKED: a stage may change exactly one axis")
        if self.changed_axis in self.frozen_axes:
            raise ValueError("ABLATION_PROTOCOL_BLOCKED: changed axis is frozen")


FROZEN_AXES = (
    "sampling", "features", "model_capacity", "folds", "calibration",
    "trrm_veto", "runtime_selection", "eqm_population", "costs", "entry_exit",
)
STAGE_DEFINITIONS: Mapping[str, StageDefinition] = {
    "STAGE_1": StageDefinition("STAGE_1", "STAGE_0", "sampling", tuple(x for x in FROZEN_AXES if x != "sampling"), "FORBIDDEN"),
    "STAGE_2": StageDefinition("STAGE_2", "STAGE_0", "features", tuple(x for x in FROZEN_AXES if x != "features"), "REQUIRED_DIMENSIONAL_COMPATIBILITY"),
    "STAGE_3": StageDefinition("STAGE_3", "STAGE_0", "model_capacity", tuple(x for x in FROZEN_AXES if x != "model_capacity"), "REQUIRED_CAPACITY_ABLATION"),
    "STAGE_4": StageDefinition("STAGE_4", "STAGE_0", "runtime_selection", tuple(x for x in FROZEN_AXES if x != "runtime_selection"), "FORBIDDEN"),
    "STAGE_5": StageDefinition("STAGE_5", "STAGE_0", "eqm_population", tuple(x for x in FROZEN_AXES if x != "eqm_population"), "FOLD_DIAGNOSTIC_ONLY"),
}


class AblationProtocolError(RuntimeError):
    """A frozen input or one-axis invariant is unavailable."""


class AblationNondeterministic(RuntimeError):
    """Two clean attempts produced different scientific payloads."""


def _canonical_scientific(value: Any) -> Any:
    """Normalize sub-ulp forest inference drift at the declared 1e-12 tolerance."""
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _canonical_scientific(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_scientific(item) for item in value]
    return value


class MedianImputer:
    """Exact historical median-imputation behavior, isolated for diagnostics."""

    def fit(self, frame: pd.DataFrame) -> "MedianImputer":
        self.columns = list(frame.columns)
        self.medians = frame.median(numeric_only=True).reindex(self.columns).fillna(0.0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.reindex(columns=self.columns).fillna(self.medians).fillna(0.0)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True,
    ).stdout.strip()


def _environment() -> Mapping[str, str]:
    import sklearn

    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "executable": sys.executable,
    }


def _historical_feature_functions() -> tuple[Callable[..., pd.DataFrame], Callable[..., pd.DataFrame]]:
    source = subprocess.run(
        ["git", "show", f"{HISTORICAL_SOURCE_COMMIT}:{HISTORICAL_FEATURE_SOURCE}"],
        check=True, text=True, capture_output=True,
    ).stdout
    import hashlib

    if hashlib.sha256(source.encode()).hexdigest() != HISTORICAL_FEATURE_SOURCE_SHA256:
        raise AblationProtocolError("ABLATION_PROTOCOL_BLOCKED: historical feature source hash mismatch")
    tree = ast.parse(source)
    allowed = {"consecutive_count", "rolling_zscore", "true_range", "compute_causal_features", "add_market_context"}
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in allowed]
    if {node.name for node in body} != allowed:
        raise AblationProtocolError("ABLATION_PROTOCOL_BLOCKED: historical feature functions missing")
    namespace: dict[str, Any] = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=body, type_ignores=[]), HISTORICAL_FEATURE_SOURCE, "exec"), namespace)
    return namespace["compute_causal_features"], namespace["add_market_context"]


def _base_h12(dataset_path: Path) -> pd.DataFrame:
    data = pd.read_csv(dataset_path).copy()
    data["_ts"] = pd.to_datetime(data["id.timestamp"], errors="raise")
    data = data[(data["_ts"] <= DEV_END) & (pd.to_numeric(data["id.horizon"], errors="coerce") == 12)]
    return data.sort_values(["_ts", "id.symbol", "id.horizon"]).reset_index(drop=True)


def _historical_feature_columns(data: pd.DataFrame) -> list[str]:
    columns = [column for column in data.columns if column.startswith("feature.")]
    return columns + ["horizon_6", "horizon_12", "horizon_24"]


def _add_horizon_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    horizon = pd.to_numeric(result["id.horizon"], errors="coerce")
    for value in (6, 12, 24):
        result[f"horizon_{value}"] = (horizon == value).astype(float)
    return result


def _hourly_historical_features(series_manifest: Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    prices = load_canonical_prices(series_manifest, _canonical_symbols())
    compute, add_context = _historical_feature_functions()
    local: dict[str, pd.DataFrame] = {}
    for symbol, indexed in prices.items():
        candles = indexed.reset_index()[["timestamp", "open", "high", "low", "close", "volume"]]
        local[symbol] = compute(candles)
    context = {symbol: local[symbol] for symbol in ("BTCUSDT", "ETHUSDT")}
    rows = []
    for symbol in _canonical_symbols():
        frame = add_context(local[symbol], context)
        frame = frame[
            (frame["timestamp"].dt.minute == 55)
            & (frame["timestamp"] >= pd.Timestamp("2024-07-12 15:55:00"))
            & (frame["timestamp"] <= E2_LAST_OPEN)
        ].copy()
        frame["id.symbol"] = symbol
        frame["id.timestamp"] = frame["timestamp"]
        frame["id.horizon"] = 12
        frame["_ts"] = frame["timestamp"]
        rename = {column: f"feature.{column}" for column in frame.columns if column not in {"timestamp", "id.symbol", "id.timestamp", "id.horizon", "_ts"}}
        rows.append(frame.rename(columns=rename))
    result = _add_horizon_columns(pd.concat(rows, ignore_index=True))
    result = result.sort_values(["_ts", "id.symbol"]).reset_index(drop=True)
    missing = sorted(set(expected_columns) - set(result.columns))
    if missing:
        raise AblationProtocolError(f"ABLATION_PROTOCOL_BLOCKED: historical hourly features missing: {missing}")
    if result.groupby("_ts")["id.symbol"].nunique().min() != 11:
        raise AblationProtocolError("ABLATION_PROTOCOL_BLOCKED: hourly population is not coordinated")
    return result


def _canonical_symbols() -> list[str]:
    from aegis.config import CANONICAL_SYMBOLS

    return list(CANONICAL_SYMBOLS)


def _e2_feature_frame(base: pd.DataFrame, series_manifest: Path) -> tuple[pd.DataFrame, list[str]]:
    # Only the frozen E2 feature/domain contracts are imported. Phase E and inference are not.
    from aegis.features import DeterministicFeaturePipeline

    pipeline = DeterministicFeaturePipeline()
    prices = load_canonical_prices(series_manifest, _canonical_symbols())
    arrays: dict[str, Mapping[str, Any]] = {}
    for symbol, frame in prices.items():
        reset = frame.reset_index()
        arrays[symbol] = {
            "frame": reset,
            "positions": {timestamp: index for index, timestamp in enumerate(reset["timestamp"])},
        }
    output: dict[tuple[pd.Timestamp, str], tuple[float, ...]] = {}
    for timestamp in sorted(base["_ts"].unique()):
        positions = {
            symbol: arrays[symbol]["positions"].get(pd.Timestamp(timestamp))
            for symbol in _canonical_symbols()
        }
        if any(position is None for position in positions.values()):
            raise AblationProtocolError(f"ABLATION_PROTOCOL_BLOCKED: E2 timestamp unavailable: {timestamp}")
        if any(int(position) < 287 for position in positions.values()):
            unavailable = tuple(float("nan") for _ in pipeline.feature_names)
            for symbol in _canonical_symbols():
                output[(pd.Timestamp(timestamp), symbol)] = unavailable
            continue
        local: dict[str, dict[str, float]] = {}
        for symbol in _canonical_symbols():
            state = arrays[symbol]; position = int(positions[symbol])
            window = state["frame"].iloc[position - 287:position + 1]
            candles = tuple(
                SimpleNamespace(
                    open=float(row.open), high=float(row.high), low=float(row.low),
                    close=float(row.close), volume=float(row.volume),
                )
                for row in window.itertuples(index=False)
            )
            local[symbol] = pipeline._local_features(candles)  # frozen E2 implementation
        returns_6 = {symbol: values["ret_6"] for symbol, values in local.items()}
        returns_12 = {symbol: values["ret_12"] for symbol, values in local.items()}
        market6 = tuple(returns_6.values()); mean6 = float(np.mean(market6)); mean12 = float(np.mean(tuple(returns_12.values())))
        dispersion = float(np.std(market6)); breadth = float(np.mean([value > 0 for value in market6]))
        concentration = max(abs(value) for value in market6) / max(sum(abs(value) for value in market6), 1e-15)
        ordered = sorted(market6)
        for symbol in _canonical_symbols():
            values = local[symbol]
            value = returns_6[symbol]; lower = sum(candidate < value for candidate in market6); equal = sum(candidate == value for candidate in market6)
            values.update({
                "relative_return_6": value - mean6,
                "relative_return_12": returns_12[symbol] - mean12,
                "cross_rank_return_6": (lower + (equal - 1) / 2) / max(1, len(ordered) - 1),
                "cross_dispersion_return_6": dispersion, "market_breadth_6": breadth,
                "market_direction_6": mean6, "market_concentration_6": concentration,
                "btc_divergence_6": value - returns_6["BTCUSDT"], "eth_divergence_6": value - returns_6["ETHUSDT"],
                "btc_volatility_12": local["BTCUSDT"]["_context_volatility_12"],
                "btc_trend_proxy": local["BTCUSDT"]["trend_stack_short"],
                "eth_volatility_12": local["ETHUSDT"]["_context_volatility_12"],
                "eth_trend_proxy": local["ETHUSDT"]["trend_stack_short"],
            })
            output[(pd.Timestamp(timestamp), symbol)] = tuple(float(values[name]) for name in pipeline.feature_names)
    result = base.copy()
    matrix = np.asarray([
        output[(pd.Timestamp(timestamp), symbol)]
        for timestamp, symbol in zip(result["_ts"], result["id.symbol"])
    ], dtype=float)
    finite_rows = np.isfinite(matrix).all(axis=1)
    if not finite_rows.any():
        raise AblationProtocolError("ABLATION_PROTOCOL_BLOCKED: E2 feature matrix has no finite rows")
    columns = [f"e2.{name}" for name in pipeline.feature_names]
    result[columns] = matrix
    return result, columns


def _model_factories(capacity: str) -> Mapping[str, Callable[[], Any]]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, RandomForestClassifier

    if capacity == "historical":
        return {
            "trrm": lambda: RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
            "eqm_reg": lambda: ExtraTreesRegressor(n_estimators=300, max_depth=14, min_samples_leaf=25, random_state=SEED, n_jobs=1),
            "eqm_clf": lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED),
        }
    if capacity == "e2_smoke":
        return {
            "trrm": lambda: RandomForestClassifier(n_estimators=80, max_depth=8, min_samples_leaf=8, random_state=SEED, n_jobs=1),
            "eqm_reg": lambda: ExtraTreesRegressor(n_estimators=80, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=1),
            "eqm_clf": lambda: HistGradientBoostingClassifier(max_iter=80, max_depth=5, min_samples_leaf=10, learning_rate=.05, random_state=SEED),
        }
    raise ValueError(capacity)


def _fit_diagnostic_scores(data: pd.DataFrame, features: Sequence[str], capacity: str) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    from sklearn.isotonic import IsotonicRegression

    factories = _model_factories(capacity)
    raw = data[list(features)].apply(pd.to_numeric, errors="coerce")
    warmup_unavailable_rows = int(raw.isna().all(axis=1).sum())
    ts = data["_ts"]; split = ts.quantile(.90)
    train = np.flatnonzero((ts <= split).to_numpy())
    calibration = np.flatnonzero((ts > ts.iloc[train].max() + pd.Timedelta(minutes=120)).to_numpy())
    imputer = MedianImputer().fit(raw.iloc[train]); values = imputer.transform(raw)
    trrm = factories["trrm"]().fit(values.iloc[train], data["target.tail_risk_roe_030"].astype(int).to_numpy()[train])
    trrm_raw = score_of(trrm, values)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(trrm_raw[calibration], data["target.tail_risk_roe_030"].astype(int).to_numpy()[calibration])
    tail = np.asarray(calibrator.predict(trrm_raw), dtype=float)
    keep_n = max(1, int(round(.70 * len(train))))
    survivors = train[np.argsort(tail[train], kind="stable")[:keep_n]]
    eqm_imputer = MedianImputer().fit(raw.iloc[survivors]); eqm_values = eqm_imputer.transform(raw)
    quality = pd.to_numeric(data["future_eval.net_quality_after_costs"], errors="coerce").fillna(0.0).to_numpy()
    clean = data["label.clean_entry_v4"].astype(int).to_numpy()
    reg = factories["eqm_reg"]().fit(eqm_values.iloc[survivors], quality[survivors])
    clf = factories["eqm_clf"]().fit(eqm_values.iloc[survivors], clean[survivors])
    scores = np.asarray(reg.predict(eqm_values), dtype=float)  # historical frozen score_kind=reg_component
    return tail, scores, {
        "capacity": capacity, "train_rows": len(train), "calibration_rows": len(calibration),
        "eqm_refit_rows": len(survivors), "trrm_model": type(trrm).__name__,
        "eqm_reg_model": type(reg).__name__, "eqm_clf_model": type(clf).__name__,
        "warmup_unavailable_rows_retained": warmup_unavailable_rows,
    }


def _fold_population_diagnostic(data: pd.DataFrame, features: Sequence[str], tail: np.ndarray) -> Mapping[str, Any]:
    factories = _model_factories("historical")
    raw = data[list(features)].apply(pd.to_numeric, errors="coerce")
    quality = pd.to_numeric(data["future_eval.net_quality_after_costs"], errors="coerce").fillna(0.0).to_numpy()
    clean = data["label.clean_entry_v4"].astype(int).to_numpy()
    rows = []
    for fold in make_folds(data):
        train = fold["train"]; evaluation = fold["evaluation"]
        survivor_train = train[np.argsort(tail[train], kind="stable")[:max(1, int(round(.70 * len(train))))]]
        survivor_eval = evaluation[np.argsort(tail[evaluation], kind="stable")[:max(1, int(round(.70 * len(evaluation))))]]
        imputer = MedianImputer().fit(raw.iloc[train]); values = imputer.transform(raw)
        reg = factories["eqm_reg"]().fit(values.iloc[train], quality[train])
        clf = factories["eqm_clf"]().fit(values.iloc[train], clean[train])
        reg_score = np.asarray(reg.predict(values.iloc[survivor_eval]), dtype=float)
        clf_score = score_of(clf, values.iloc[survivor_eval])
        rows.append({
            "fold": fold["name"], "full_train_rows": len(train),
            "historical_survivor_train_rows": len(survivor_train), "scoring_survivor_rows": len(survivor_eval),
            "reg_top_decile_quality": _top_decile(reg_score, quality[survivor_eval]),
            "clf_top_decile_quality": _top_decile(clf_score, quality[survivor_eval]),
        })
    return {"folds": rows, "refit_population": "TRRM_SURVIVORS_UNCHANGED"}


def _top_decile(score: np.ndarray, quality: np.ndarray) -> float:
    count = max(1, int(round(.10 * len(score))))
    return float(quality[np.argsort(-score, kind="stable")[:count]].mean())


def _frozen_scores(data: pd.DataFrame, rv2_path: Path, eqm_path: Path) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    with_horizons = _add_horizon_columns(data)
    trrm = load_pickle(rv2_path)
    trrm_values = trrm["imputer"].transform(with_horizons[trrm["features"]].apply(pd.to_numeric, errors="coerce"))
    raw = score_of(trrm["trrm_model"], trrm_values)
    tail = np.asarray(trrm["calibrator"].predict(raw), dtype=float)
    eqm = load_pickle(eqm_path)
    eqm_values = eqm["imputer"].transform(with_horizons[eqm["features"]].apply(pd.to_numeric, errors="coerce"))
    reg = np.asarray(eqm["reg_model"].predict(eqm_values), dtype=float)
    clean = score_of(eqm["clf_model"], eqm_values)
    score = clean * reg if eqm["score_kind"] == "composite_ev" else reg
    return tail, score, {"rv2": "FROZEN_PICKLE", "eqm": "FROZEN_PICKLE", "score_kind": eqm["score_kind"]}


def _select(data: pd.DataFrame, tail: np.ndarray, scores: np.ndarray, mode: str) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    selected = []; counts = {key: 0 for key in ("total_candidates", "before_trrm", "trrm_survivors", "before_eqm", "after_eqm", "selected_before_dedup")}
    for fold in make_folds(data):
        evaluation = fold["evaluation"]
        counts["total_candidates"] += len(evaluation); counts["before_trrm"] += len(evaluation)
        if mode == "historical":
            keep_n = max(1, int(round(.70 * len(evaluation))))
            retained = evaluation[np.argsort(tail[evaluation], kind="stable")[:keep_n]]
            budget = max(1, int(round(.10 * keep_n)))
            ranked = retained[np.argsort(-scores[retained], kind="stable")]
            counts["trrm_survivors"] += len(retained); counts["before_eqm"] += len(retained)
            counts["after_eqm"] += budget; counts["selected_before_dedup"] += budget
            subset = data.iloc[ranked].reset_index(drop=True)
            chosen = ranked[correlation_limit(subset, scores[ranked])][:budget]
        elif mode == "e2_runtime":
            retained = evaluation[tail[evaluation] <= .70]
            eligible = retained[scores[retained] >= E2_THRESHOLD]
            counts["trrm_survivors"] += len(retained); counts["before_eqm"] += len(retained)
            counts["after_eqm"] += len(eligible); counts["selected_before_dedup"] += len(eligible)
            candidate = data.iloc[eligible][["_ts", "id.symbol"]].copy()
            candidate["_index"] = eligible; candidate["_score"] = scores[eligible]
            chosen = candidate.sort_values(["_ts", "_score", "id.symbol"], ascending=[True, False, True]).drop_duplicates("_ts")["_index"].to_numpy(dtype=int)
        else:
            raise ValueError(mode)
        frame = data.iloc[chosen][["id.symbol", "_ts"]].copy()
        frame["fold"] = fold["name"]; frame["rv2_tail"] = tail[chosen]; frame["eqm_score"] = scores[chosen]
        selected.append(frame)
    result = pd.concat(selected, ignore_index=True).rename(columns={"id.symbol": "symbol", "_ts": "ts"})
    counts["final_trades"] = len(result)
    return result, counts


def _attach_economics(trades: pd.DataFrame, series_manifest: Path, costs: Mapping[str, float]) -> pd.DataFrame:
    prices = load_canonical_prices(series_manifest, sorted(trades["symbol"].unique()))
    economics = [historical_trade_pnl(prices, row.symbol, row.ts, dict(costs)) for row in trades.itertuples()]
    if any(item is None for item in economics):
        raise AblationProtocolError("ABLATION_PROTOCOL_BLOCKED: canonical price lookup failed")
    return pd.concat([trades.reset_index(drop=True), pd.DataFrame(economics)], axis=1)


def _performance(frame: pd.DataFrame, counts: Mapping[str, Any]) -> Mapping[str, Any]:
    net = frame["net"].to_numpy(float); gross = frame["gross"].to_numpy(float)
    net_wins = net[net > 0]; net_losses = net[net < 0]; gross_wins = gross[gross > 0]; gross_losses = gross[gross < 0]
    equity = np.cumsum(net); drawdown = np.maximum.accumulate(np.r_[0.0, equity])[1:] - equity
    keys = frame[["fold", "symbol", "ts"]].copy(); keys["ts"] = keys["ts"].astype(str)
    def aggregate(group: pd.DataFrame) -> Mapping[str, Any]:
        values = group["net"].to_numpy(float)
        wins = values[values > 0]; losses = values[values < 0]
        return {"trades": len(values), "expectancy": float(values.mean()), "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else None, "win_rate": float((values > 0).mean())}
    month = frame.assign(month=pd.to_datetime(frame["ts"]).dt.strftime("%Y-%m"))
    total_hours = max(1.0, (pd.to_datetime(frame["ts"]).max() - pd.to_datetime(frame["ts"]).min()).total_seconds() / 3600)
    return {
        **counts, "trades": len(frame),
        "gross_profit_factor": float(gross_wins.sum() / abs(gross_losses.sum())),
        "net_profit_factor": float(net_wins.sum() / abs(net_losses.sum())),
        "gross_expectancy": float(gross.mean()), "net_expectancy": float(net.mean()),
        "win_rate": float((net > 0).mean()), "average_win": float(net_wins.mean()), "average_loss": float(net_losses.mean()),
        "p95_drawdown": float(np.quantile(drawdown, .95)), "maximum_drawdown": float(drawdown.max()),
        "turnover": len(frame), "total_costs": float(frame["cost"].sum()),
        "exposure": float(len(frame) / total_hours), "side_distribution": {"SHORT": len(frame), "LONG": 0},
        "by_fold": {str(key): aggregate(value) for key, value in frame.groupby("fold")},
        "by_symbol": {str(key): aggregate(value) for key, value in frame.groupby("symbol")},
        "by_month": {str(key): aggregate(value) for key, value in month.groupby("month")},
        "by_regime": {"status": "NOT_AVAILABLE_NO_FROZEN_HISTORICAL_REGIME_CONTRACT"},
        "trade_keys_hash": digest(keys.to_dict("records")),
    }


def _reference_keys(stage_zero: Mapping[str, Any], config: ReplayConfig) -> set[tuple[str, str, str]]:
    frozen = pd.read_csv(config.payload["inputs"]["trades"]["path"])
    frozen = frozen[(frozen["strategy"] == "eqm_plus_trrm") & (frozen["scenario"] == "B_base")]
    return {(str(row.fold), str(row.symbol), _timestamp_key(row.ts)) for row in frozen.itertuples()}


def _timestamp_key(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.isoformat()


def _e2_reference(path: Path) -> tuple[set[tuple[str, str, str]], Mapping[str, Any]]:
    if sha256_file(path) != E2_ECON_SHA256:
        raise AblationProtocolError("ABLATION_PROTOCOL_BLOCKED: E2 ECON reference hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8")); rows = [item for item in payload["report"]["trades"] if item["scenario_id"] == "B_BASE"]
    keys = {(f"fold_{item['signal']['fold']}", item["signal"]["symbol"], _timestamp_key(item["signal"]["timestamp"])) for item in rows}
    metric = payload["report"]["metrics"]["full_stack"]["B_BASE"]
    return keys, {"trades": metric["trades"], "profit_factor": metric["profit_factor"], "expectancy_historical_units": metric["expectancy"] * 100.0, "source_sha256": E2_ECON_SHA256}


def _overlap(frame: pd.DataFrame, reference: set[tuple[str, str, str]]) -> Mapping[str, Any]:
    current = {(str(row.fold), str(row.symbol), _timestamp_key(row.ts)) for row in frame.itertuples()}
    common = current & reference
    return {"ratio": len(common) / max(1, len(reference)), "common": len(common), "missing": len(reference - current), "additional": len(current - reference)}


def _manifest(definition: StageDefinition, config: ReplayConfig, attempt: int, dataset_hash: str, feature_hash: str) -> Mapping[str, Any]:
    definition.validate()
    lockbox = json.loads(Path("reports/experiments/lockbox_semi_blind_20260427_20260711.json").read_text())
    if lockbox["status"] != "NOT_CONSUMED" or lockbox["consumed_queries"]:
        raise AblationProtocolError("ABLATION_PROTOCOL_BLOCKED: lockbox is not pristine")
    scientific = {
        "stage_id": definition.stage, "parent_stage": definition.parent,
        "changed_axis": definition.changed_axis, "frozen_axes": definition.frozen_axes,
        "dataset_hash": dataset_hash, "feature_schema_hash": feature_hash,
        "configuration_hash": sha256_file(config.path), "code_commit": _git_commit(),
        "environment": _environment(), "lockbox_status": "NOT_CONSUMED",
        "semi_blind": False, "dev_end": str(DEV_END), "retraining": definition.retraining,
    }
    return {
        **scientific, "attempt": attempt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_manifest_hash": digest(scientific),
    }


def run_ablation_attempt(
    config: ReplayConfig, stage: str, attempt: int, e2_econ_path: Path,
) -> Mapping[str, Any]:
    if stage not in STAGE_DEFINITIONS:
        raise AblationProtocolError(f"ABLATION_PROTOCOL_BLOCKED: unknown stage {stage}")
    definition = STAGE_DEFINITIONS[stage]; inputs = config.payload["inputs"]
    for item in inputs.values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise AblationProtocolError(f"ABLATION_PROTOCOL_BLOCKED: input hash mismatch: {path}")
    base = _base_h12(Path(inputs["dataset"]["path"])); historical_features = _historical_feature_columns(base)
    if stage == "STAGE_2":
        from aegis.features import DeterministicFeaturePipeline

        feature_hash = digest([f"e2.{name}" for name in DeterministicFeaturePipeline().feature_names])
    else:
        feature_hash = digest(historical_features)
    manifest = _manifest(
        definition, config, attempt, sha256_file(Path(inputs["dataset"]["path"])), feature_hash,
    )
    output = Path(config.payload["output_root"]) / stage.lower() / f"attempt_{attempt}"
    atomic_write(output / "manifest.json", manifest)
    model_evidence: Mapping[str, Any]
    if stage == "STAGE_1":
        data = _hourly_historical_features(Path(inputs["series_manifest"]["path"]), historical_features)
        tail, scores, model_evidence = _frozen_scores(data, Path(inputs["rv2_pickle"]["path"]), Path(inputs["eqm_pickle"]["path"]))
        selection_mode = "historical"
    elif stage == "STAGE_2":
        data, features = _e2_feature_frame(base, Path(inputs["series_manifest"]["path"]))
        if digest(features) != feature_hash:
            raise AblationProtocolError("ABLATION_PROTOCOL_BLOCKED: E2 feature contract drift")
        tail, scores, model_evidence = _fit_diagnostic_scores(data, features, "historical")
        selection_mode = "historical"
    elif stage == "STAGE_3":
        data = _add_horizon_columns(base); tail, scores, model_evidence = _fit_diagnostic_scores(data, historical_features, "e2_smoke")
        model_evidence = {**model_evidence, "capacity_matrix": _capacity_matrix()}; selection_mode = "historical"
    else:
        data = _add_horizon_columns(base); tail, scores, model_evidence = _frozen_scores(data, Path(inputs["rv2_pickle"]["path"]), Path(inputs["eqm_pickle"]["path"]))
        selection_mode = "e2_runtime" if stage == "STAGE_4" else "historical"
        if stage == "STAGE_5":
            model_evidence = {**model_evidence, "fold_full_population_diagnostic": _fold_population_diagnostic(data, historical_features, tail)}
    trades, counts = _select(data, tail, scores, selection_mode)
    report = json.loads(Path(inputs["econ_report"]["path"]).read_text()); costs = report["cost_scenarios"]["B_base"]
    trades = _attach_economics(trades, Path(inputs["series_manifest"]["path"]), costs)
    metrics = _performance(trades, counts)
    stage0 = json.loads((Path(config.payload["output_root"]) / "stage_0.json").read_text())
    stage0_keys = _reference_keys(stage0, config); e2_keys, e2_metrics = _e2_reference(e2_econ_path)
    overlap0 = _overlap(trades, stage0_keys); overlap2 = _overlap(trades, e2_keys)
    scientific = {
        "schema_version": "aegis-causal-ablation-stage-v1", "stage": stage,
        "changed_axis": definition.changed_axis, "parent": definition.parent,
        "frozen_axes": definition.frozen_axes, "manifest_hash": manifest["scientific_manifest_hash"],
        "metrics": metrics, "model_evidence": model_evidence,
        "row_counts": {
            "dataset_rows": len(data), "coordinated_cycles": int(data["_ts"].nunique()),
            "symbols": int(data["id.symbol"].nunique()),
            "warmup_unavailable_rows": int(model_evidence.get("warmup_unavailable_rows_retained", 0)),
        },
        "input_hashes": {
            key: value["sha256"] for key, value in inputs.items()
        } | {"e2_econ_reference": E2_ECON_SHA256},
        "score_distribution": {
            "rv2_tail": {str(q): float(np.quantile(tail, q)) for q in (0, .1, .5, .7, .9, 1)},
            "eqm_score": {str(q): float(np.quantile(scores, q)) for q in (0, .1, .5, .9, 1)},
        },
        "overlap_stage_0": overlap0, "overlap_e2": overlap2, "e2_reference": e2_metrics,
        "delta_stage_0": {
            "trades": metrics["trades"] - stage0["trades"],
            "profit_factor": metrics["net_profit_factor"] - stage0["profit_factor"],
            "expectancy": metrics["net_expectancy"] - stage0["net_expectancy"],
            "win_rate": metrics["win_rate"] - stage0["win_rate"],
        },
        "trade_records": trades.assign(ts=trades["ts"].astype(str)).to_dict("records"),
        "safety_flags": {"dev_only": True, "lockbox": False, "semi_blind": False, "candidate": False, "policy": False, "freeze": False},
    }
    scientific_hash = digest(_canonical_scientific(scientific))
    payload = {**scientific, "attempt": attempt, "scientific_hash": scientific_hash}
    atomic_write(output / "result.json", payload)
    return payload


def _capacity_matrix() -> Mapping[str, Mapping[str, Any]]:
    return {
        "trrm_random_forest": {"historical": {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20, "class_weight": "balanced_subsample"}, "e2": {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 8, "class_weight": None}},
        "trrm_hgb": {"historical": {"max_iter": 300, "learning_rate": .06, "max_leaf_nodes": 31, "min_samples_leaf": 40}, "e2": {"max_iter": 80, "learning_rate": .05, "max_depth": 5, "min_samples_leaf": 10}},
        "trrm_logistic": {"historical": {"max_iter": 1000}, "e2": {"C": 1.0, "max_iter": 1000}},
        "eqm_extra_trees": {"historical": {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 25}, "e2": {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 5}},
        "eqm_hgb": {"historical": {"max_iter": 250, "learning_rate": .06, "max_leaf_nodes": 31, "min_samples_leaf": 40}, "e2": {"max_iter": 80, "learning_rate": .05, "max_depth": 5, "min_samples_leaf": 10}},
        "eqm_random_forest_clean": {"historical": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 25, "class_weight": "balanced_subsample"}, "e2": {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 8, "class_weight": None}},
        "qmae_hgb": {"historical": {"max_iter": 300, "learning_rate": .06, "max_leaf_nodes": 31, "min_samples_leaf": 40}, "e2": {"max_iter": 80, "learning_rate": .05, "max_depth": 4, "min_samples_leaf": 10}},
    }


def run_all_ablations(config: ReplayConfig, e2_econ_path: Path) -> Mapping[str, Any]:
    stage0_path = Path(config.payload["output_root"]) / "stage_0.json"
    stage0 = json.loads(stage0_path.read_text())
    if not stage0.get("passed") or stage0.get("trades") != 688 or stage0.get("trade_overlap") != 1.0:
        raise AblationProtocolError("ABLATION_PROTOCOL_BLOCKED: Stage 0 is not intact")
    results = []
    for stage in STAGE_DEFINITIONS:
        first = run_ablation_attempt(config, stage, 1, e2_econ_path)
        second = run_ablation_attempt(config, stage, 2, e2_econ_path)
        if first["scientific_hash"] != second["scientific_hash"]:
            raise AblationNondeterministic(f"ABLATION_NONDETERMINISTIC: {stage}")
        first_keys = [(item["fold"], item["symbol"], item["ts"]) for item in first["trade_records"]]
        second_keys = [(item["fold"], item["symbol"], item["ts"]) for item in second["trade_records"]]
        if first_keys != second_keys:
            raise AblationNondeterministic(f"ABLATION_NONDETERMINISTIC: {stage} trade keys")
        summary = {key: value for key, value in first.items() if key not in {"attempt", "trade_records"}}
        summary["determinism"] = {
            "attempt_1": first["scientific_hash"], "attempt_2": second["scientific_hash"],
            "canonical_identical": True, "trade_keys_identical": True,
            "float_tolerance": DETERMINISM_TOLERANCE,
        }
        atomic_write(Path(config.payload["output_root"]) / f"{stage.lower()}.json", summary)
        results.append(summary)
    table = _causal_table(stage0, results, _stage0_max_drawdown(config))
    root = Path(config.payload["output_root"])
    atomic_write(root / "causal_ablation_table.json", table)
    (root / "causal_ablation_table.md").write_text(_causal_markdown(table), encoding="utf-8")
    return table


def refresh_overlap_evidence(config: ReplayConfig, e2_econ_path: Path) -> Mapping[str, Any]:
    """Rebuild only normalized overlap fields from completed raw trade records."""
    root = Path(config.payload["output_root"])
    stage0 = json.loads((root / "stage_0.json").read_text())
    stage0_keys = _reference_keys(stage0, config)
    e2_keys, e2_metrics = _e2_reference(e2_econ_path)
    summaries = []
    for stage in STAGE_DEFINITIONS:
        attempts = []
        for attempt in (1, 2):
            path = root / stage.lower() / f"attempt_{attempt}" / "result.json"
            payload = json.loads(path.read_text())
            frame = pd.DataFrame(payload["trade_records"])
            payload["overlap_stage_0"] = _overlap(frame, stage0_keys)
            payload["overlap_e2"] = _overlap(frame, e2_keys)
            payload["e2_reference"] = e2_metrics
            scientific = {key: value for key, value in payload.items() if key not in {"attempt", "scientific_hash"}}
            payload["scientific_hash"] = digest(_canonical_scientific(scientific))
            atomic_write(path, payload); attempts.append(payload)
        if attempts[0]["scientific_hash"] != attempts[1]["scientific_hash"]:
            raise AblationNondeterministic(f"ABLATION_NONDETERMINISTIC: refreshed {stage}")
        summary = {key: value for key, value in attempts[0].items() if key not in {"attempt", "trade_records"}}
        summary["determinism"] = {
            "attempt_1": attempts[0]["scientific_hash"], "attempt_2": attempts[1]["scientific_hash"],
            "canonical_identical": True, "trade_keys_identical": True,
            "float_tolerance": DETERMINISM_TOLERANCE,
        }
        atomic_write(root / f"{stage.lower()}.json", summary); summaries.append(summary)
    table = _causal_table(stage0, summaries, _stage0_max_drawdown(config))
    atomic_write(root / "causal_ablation_table.json", table)
    (root / "causal_ablation_table.md").write_text(_causal_markdown(table), encoding="utf-8")
    return table


def _stage0_max_drawdown(config: ReplayConfig) -> float:
    report = json.loads(Path(config.payload["inputs"]["econ_report"]["path"]).read_text())
    return float(report["strategies"]["eqm_plus_trrm"]["B_base"]["max_drawdown"])


def _causal_table(
    stage0: Mapping[str, Any], results: Sequence[Mapping[str, Any]], stage0_max_drawdown: float,
) -> Mapping[str, Any]:
    rows = [{
        "stage": "STAGE_0", "changed_axis": "NONE", "parent": None, "trades": stage0["trades"],
        "PF": stage0["profit_factor"], "net_expectancy": stage0["net_expectancy"], "win_rate": stage0["win_rate"],
        "max_drawdown": stage0_max_drawdown, "turnover": stage0["trades"], "overlap_stage_0": 1.0, "overlap_e2": None,
        "delta_pf_vs_stage_0": 0.0, "delta_expectancy_vs_stage_0": 0.0, "delta_trades_vs_stage_0": 0,
        "result_status": "HISTORICAL_CONTROL", "evidence_path": "stage_0.json",
    }]
    for item in results:
        metrics = item["metrics"]
        rows.append({
            "stage": item["stage"], "changed_axis": item["changed_axis"], "parent": item["parent"],
            "trades": metrics["trades"], "PF": metrics["net_profit_factor"], "net_expectancy": metrics["net_expectancy"],
            "win_rate": metrics["win_rate"], "max_drawdown": metrics["maximum_drawdown"], "turnover": metrics["turnover"],
            "overlap_stage_0": item["overlap_stage_0"]["ratio"], "overlap_e2": item["overlap_e2"]["ratio"],
            "delta_pf_vs_stage_0": item["delta_stage_0"]["profit_factor"],
            "delta_expectancy_vs_stage_0": item["delta_stage_0"]["expectancy"],
            "delta_trades_vs_stage_0": item["delta_stage_0"]["trades"],
            "result_status": "DETERMINISTIC", "evidence_path": f"{item['stage'].lower()}.json",
        })
    payload = {"schema_version": "aegis-causal-ablation-table-v1", "rows": rows}
    return {**payload, "content_hash": digest(payload)}


def _causal_markdown(table: Mapping[str, Any]) -> str:
    lines = ["# Gen2 causal ablations", "", "| Stage | Changed axis | Trades | PF | Net expectancy | Win rate | Max DD | Overlap Stage 0 | Overlap E2 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table["rows"]:
        lines.append(f"| {row['stage']} | {row['changed_axis']} | {row['trades']} | {row['PF']:.12g} | {row['net_expectancy']:.12g} | {row['win_rate']:.12g} | {'' if row['max_drawdown'] is None else format(row['max_drawdown'], '.12g')} | {row['overlap_stage_0']} | {row['overlap_e2']} |")
    lines += ["", "Deltas are isolated diagnostics and are not additive; interactions between axes remain unidentified.", ""]
    return "\n".join(lines)
