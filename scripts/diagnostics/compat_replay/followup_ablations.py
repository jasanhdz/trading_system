"""Execution-only runners for the frozen Stage 1b and Stage 4b protocols."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .ablations import (
    DEV_END,
    DETERMINISM_TOLERANCE,
    HISTORICAL_SOURCE_COMMIT,
    _add_horizon_columns,
    _attach_economics,
    _base_h12,
    _canonical_scientific,
    _canonical_symbols,
    _historical_feature_columns,
    _hourly_historical_features,
    _performance,
    _timestamp_key,
    MedianImputer,
)
from .historical_adapter import correlation_limit, load_canonical_prices, make_folds, score_of
from .manifests import atomic_write, digest, sha256_file
from .schemas import ReplayConfig


ROOT = Path(__file__).resolve().parents[3]
STAGE_1B_PREREGISTRATION = ROOT / "reports/governance/additional_ablations/stage_1b_preregistration.yaml"
STAGE_4B_PREREGISTRATION = ROOT / "reports/governance/additional_ablations/stage_4b_preregistration.yaml"
STAGE_1B_SHA256 = "3fd8b6f59aaf0ffe538a5cc66b4cffc2223a7b78ebd03b1243b8bcbf16f1431f"
STAGE_4B_SHA256 = "330a3385109547893ed11fbb0d9c0079fa519df0db251ece338c1c050bc0865e"
HISTORICAL_LABEL_SOURCE = "aegis_alpha/turbo/short_quality_v4_labels.py"
HISTORICAL_LABEL_SOURCE_SHA256 = "5d71ed5b7ff75ae1e1382afc11cf5f7fe7614f1e5320f063f942f1c117ae791e"
E2_THRESHOLD = 4.047730415717134e-05
VETO_BUDGET = 0.30
BASE_SEED = 42


class FollowupProtocolError(RuntimeError):
    """A frozen follow-up contract cannot be executed exactly."""


class FollowupNondeterministic(RuntimeError):
    """Independent attempts produced different scientific outputs."""


@dataclass(frozen=True)
class FoldModelEvidence:
    fold: str
    seed: int
    train_rows: int
    calibration_rows: int
    scoring_rows: int
    trrm_train_survivors: int
    trrm_scoring_survivors: int
    trrm_model: str
    trrm_calibrator: str
    eqm_reg_model: str
    eqm_clean_model: str
    score_kind: str
    tail_hash: str
    score_hash: str


def _load_preregistration(path: Path, expected_hash: str) -> Mapping[str, Any]:
    if sha256_file(path) != expected_hash:
        raise FollowupProtocolError(f"PROTOCOL_VIOLATION: preregistration hash mismatch: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PRE_REGISTERED_NOT_EXECUTED":
        raise FollowupProtocolError("PROTOCOL_VIOLATION: follow-up preregistration is not executable")
    scope = payload.get("scope", {})
    if scope.get("lockbox") != "FORBIDDEN" or scope.get("semi_blind") != "FORBIDDEN":
        raise FollowupProtocolError("PROTOCOL_VIOLATION: follow-up safety scope drift")
    return payload


def _execution_identity(config: ReplayConfig) -> Mapping[str, Any]:
    import sklearn

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True,
    ).stdout.strip()
    return {
        "code_commit": commit,
        "replay_config_sha256": sha256_file(config.path),
        "input_hashes": {key: value["sha256"] for key, value in config.payload["inputs"].items()},
        "environment": {
            "python": platform.python_version(), "executable": sys.executable,
            "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
        },
    }


def _validate_stage_1b(payload: Mapping[str, Any]) -> None:
    expected = {
        "sampling": "E2_HOURLY_ANCHORS",
        "rows": 172480,
        "cycles": 15680,
        "features": "HISTORICAL_114_RECOMPUTED_ON_HOURLY_POPULATION",
        "capacities": "HISTORICAL_FULL",
        "seeds": "BASE_42_PLUS_FOLD_ID",
        "eqm_population": "TRRM_RANK_30_PERCENT_VETO_SURVIVORS",
        "selection": "HISTORICAL_POOLED_TOP_DECILE",
        "runs": 2,
        "tolerance": 1e-12,
    }
    actual = {
        "sampling": payload["data"]["sampling"],
        "rows": payload["data"]["row_count"],
        "cycles": payload["data"]["anchor_count"],
        "features": payload["features"]["schema"],
        "capacities": payload["models"]["capacities"],
        "seeds": payload["training"]["seeds"],
        "eqm_population": payload["training"]["eqm_population"],
        "selection": payload["selection"]["method"],
        "runs": payload["evaluation"]["deterministic_runs"],
        "tolerance": payload["evaluation"]["numeric_tolerance"],
    }
    if actual != expected or payload["selection"]["absolute_threshold"] != "FORBIDDEN":
        raise FollowupProtocolError(f"PROTOCOL_VIOLATION: Stage 1b contract drift: {actual}")
    if payload["folds"]["train_validation_fractions"] != [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9]]:
        raise FollowupProtocolError("PROTOCOL_VIOLATION: Stage 1b folds drift")


def _validate_stage_4b(payload: Mapping[str, Any]) -> None:
    variants = payload.get("variants", [])
    if [item.get("id") for item in variants] != ["STAGE_4B_A", "STAGE_4B_B", "STAGE_4B_C"]:
        raise FollowupProtocolError("PROTOCOL_VIOLATION: Stage 4b variants drift")
    if [item.get("changed_axis") for item in variants] != ["veto", "selection", "threshold"]:
        raise FollowupProtocolError("PROTOCOL_VIOLATION: Stage 4b changed axes drift")
    if variants[0]["veto"] != {"mechanics": "ABSOLUTE_CALIBRATED_PROBABILITY", "maximum": 0.70, "source": "E2_EXECUTED_RUNTIME"}:
        raise FollowupProtocolError("PROTOCOL_VIOLATION: Stage 4b-A drift")
    if variants[1]["selection"] != "TOP_1_PER_CYCLE" or variants[1]["threshold"] != "NONE":
        raise FollowupProtocolError("PROTOCOL_VIOLATION: Stage 4b-B drift")
    threshold = variants[2]["threshold"]
    if threshold["value"] != E2_THRESHOLD or threshold["application_order"] != "AFTER_VETO_BEFORE_RANKING":
        raise FollowupProtocolError("PROTOCOL_VIOLATION: Stage 4b-C drift")
    if payload.get("additional_variants") != "FORBIDDEN":
        raise FollowupProtocolError("PROTOCOL_VIOLATION: Stage 4b is not closed")


def _historical_label_functions() -> tuple[type[Any], Callable[..., Mapping[str, Any]], Callable[..., int]]:
    source = subprocess.run(
        ["git", "show", f"{HISTORICAL_SOURCE_COMMIT}:{HISTORICAL_LABEL_SOURCE}"],
        check=True, text=True, capture_output=True,
    ).stdout
    if hashlib.sha256(source.encode()).hexdigest() != HISTORICAL_LABEL_SOURCE_SHA256:
        raise FollowupProtocolError("PROTOCOL_VIOLATION: historical label source hash mismatch")
    names = {
        "ShortV4Config", "safe_div", "round_trip_cost_roe", "_as_float_array",
        "_future_window", "_hit_before_stop", "compute_short_path_metrics_v4",
        "classify_short_clean_entry_v4",
    }
    tree = ast.parse(source)
    body = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    if {node.name for node in body} != names:
        raise FollowupProtocolError("PROTOCOL_VIOLATION: historical label functions missing")
    from dataclasses import dataclass

    namespace: dict[str, Any] = {
        "Any": Any, "dataclass": dataclass, "math": math, "np": np,
        "DEFAULT_LEVERAGE": 20.0,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), HISTORICAL_LABEL_SOURCE, "exec"), namespace)
    return namespace["ShortV4Config"], namespace["compute_short_path_metrics_v4"], namespace["classify_short_clean_entry_v4"]


def _attach_hourly_historical_labels(data: pd.DataFrame, series_manifest: Path) -> pd.DataFrame:
    config_type, metrics_fn, clean_fn = _historical_label_functions()
    prices = load_canonical_prices(series_manifest, _canonical_symbols())
    result = data.copy()
    tail = np.empty(len(result), dtype=np.int8)
    quality = np.empty(len(result), dtype=float)
    clean = np.empty(len(result), dtype=np.int8)
    mae = np.empty(len(result), dtype=float)
    for symbol in _canonical_symbols():
        selected = result.index[result["id.symbol"] == symbol].to_numpy()
        frame = prices[symbol]
        high = frame["high"].to_numpy(float)
        low = frame["low"].to_numpy(float)
        close = frame["close"].to_numpy(float)
        positions = {timestamp: int(index) for timestamp, index in frame["_i"].items()}
        config = config_type(horizon=12)
        for row_index in selected:
            timestamp = pd.Timestamp(result.at[row_index, "_ts"])
            position = positions.get(timestamp)
            if position is None or position + 12 >= len(frame):
                raise FollowupProtocolError(f"INSUFFICIENT_DATA: missing Stage 1b label window: {symbol} {timestamp}")
            metrics = metrics_fn(high=high, low=low, close=close, entry_index=position, horizon=12, config=config)
            if not metrics.get("sample_complete"):
                raise FollowupProtocolError(f"INSUFFICIENT_DATA: incomplete Stage 1b label: {symbol} {timestamp}")
            mae_value = float(metrics["mae_roe_proxy"])
            tail[row_index] = int(mae_value >= 0.30)
            quality[row_index] = float(metrics["net_quality_after_costs"])
            clean[row_index] = clean_fn(metrics, config)
            mae[row_index] = mae_value
    result["target.tail_risk_roe_030"] = tail
    result["future_eval.net_quality_after_costs"] = quality
    result["label.clean_entry_v4"] = clean
    result["future_eval.future_mae_roe_proxy"] = mae
    return result


def _hourly_dataset(config: ReplayConfig, preregistration: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str], str]:
    inputs = config.payload["inputs"]
    base = _base_h12(Path(inputs["dataset"]["path"]))
    features = _historical_feature_columns(base)
    data = _hourly_historical_features(Path(inputs["series_manifest"]["path"]), features)
    data = _attach_hourly_historical_labels(data, Path(inputs["series_manifest"]["path"]))
    if len(data) != preregistration["data"]["row_count"] or data["_ts"].nunique() != preregistration["data"]["anchor_count"]:
        raise FollowupProtocolError("INSUFFICIENT_DATA: Stage 1b hourly population mismatch")
    if data["id.symbol"].nunique() != 11 or len(features) != 114:
        raise FollowupProtocolError("PROTOCOL_VIOLATION: Stage 1b feature/universe contract mismatch")
    if data["_ts"].max() > DEV_END:
        raise FollowupProtocolError("PROTOCOL_VIOLATION: Stage 1b crossed the dev boundary")
    columns = ["_ts", "id.symbol", *features, "target.tail_risk_roe_030", "future_eval.net_quality_after_costs", "label.clean_entry_v4"]
    row_hashes = pd.util.hash_pandas_object(data[columns], index=False, categorize=False).to_numpy(dtype=np.uint64)
    stream = hashlib.sha256()
    stream.update(json.dumps(columns, separators=(",", ":")).encode("utf-8"))
    stream.update(row_hashes.tobytes())
    scientific_hash = stream.hexdigest()
    return data, features, scientific_hash


def _rank_survivors(indices: np.ndarray, tail: np.ndarray) -> np.ndarray:
    keep = max(1, int(round((1.0 - VETO_BUDGET) * len(indices))))
    return indices[np.argsort(tail[indices], kind="stable")[:keep]]


def _fit_stage_1b_scores(data: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray, tuple[FoldModelEvidence, ...]]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.isotonic import IsotonicRegression

    raw = data[list(features)].apply(pd.to_numeric, errors="coerce")
    tail_target = data["target.tail_risk_roe_030"].astype(int).to_numpy()
    quality = data["future_eval.net_quality_after_costs"].astype(float).to_numpy()
    clean = data["label.clean_entry_v4"].astype(int).to_numpy()
    tail = np.full(len(data), np.nan, dtype=float)
    score = np.full(len(data), np.nan, dtype=float)
    evidence = []
    for fold_id, fold in enumerate(make_folds(data), start=1):
        seed = BASE_SEED + fold_id
        train, calibration, evaluation = fold["train"], fold["calibration"], fold["evaluation"]
        if not len(train) or not len(calibration) or not len(evaluation):
            raise FollowupProtocolError(f"INSUFFICIENT_DATA: empty {fold['name']}")
        trrm_imputer = MedianImputer().fit(raw.iloc[train])
        trrm_values = trrm_imputer.transform(raw)
        trrm = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight="balanced_subsample", random_state=seed, n_jobs=1,
        ).fit(trrm_values.iloc[train], tail_target[train])
        raw_tail = score_of(trrm, trrm_values)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_tail[calibration], tail_target[calibration])
        calibrated_tail = np.asarray(calibrator.predict(raw_tail), dtype=float)
        train_survivors = _rank_survivors(train, calibrated_tail)
        scoring_survivors = _rank_survivors(evaluation, calibrated_tail)
        eqm_imputer = MedianImputer().fit(raw.iloc[train_survivors])
        eqm_values = eqm_imputer.transform(raw)
        reg = ExtraTreesRegressor(
            n_estimators=300, max_depth=14, min_samples_leaf=25,
            random_state=seed, n_jobs=1,
        ).fit(eqm_values.iloc[train_survivors], quality[train_survivors])
        clean_model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
        ).fit(eqm_values.iloc[train_survivors], clean[train_survivors])
        fold_scores = np.asarray(reg.predict(eqm_values.iloc[evaluation]), dtype=float)
        tail[evaluation] = calibrated_tail[evaluation]
        score[evaluation] = fold_scores
        evidence.append(FoldModelEvidence(
            fold["name"], seed, len(train), len(calibration), len(evaluation),
            len(train_survivors), len(scoring_survivors), type(trrm).__name__,
            type(calibrator).__name__, type(reg).__name__, type(clean_model).__name__,
            "reg_component", digest(calibrated_tail[evaluation].tolist()), digest(fold_scores.tolist()),
        ))
    return tail, score, tuple(evidence)


def _select_historical_fold_scores(data: pd.DataFrame, tail: np.ndarray, scores: np.ndarray) -> tuple[pd.DataFrame, Mapping[str, int]]:
    selected = []
    counts = {key: 0 for key in ("total_candidates", "before_trrm", "trrm_survivors", "before_eqm", "after_eqm", "selected_before_dedup")}
    for fold in make_folds(data):
        evaluation = fold["evaluation"]
        if not np.isfinite(tail[evaluation]).all() or not np.isfinite(scores[evaluation]).all():
            raise FollowupProtocolError(f"PROTOCOL_VIOLATION: non-finite scoring output in {fold['name']}")
        retained = _rank_survivors(evaluation, tail)
        budget = max(1, int(round(0.10 * len(retained))))
        ranked = retained[np.argsort(-scores[retained], kind="stable")]
        subset = data.iloc[ranked].reset_index(drop=True)
        chosen = ranked[correlation_limit(subset, scores[ranked])][:budget]
        counts["total_candidates"] += len(evaluation)
        counts["before_trrm"] += len(evaluation)
        counts["trrm_survivors"] += len(retained)
        counts["before_eqm"] += len(retained)
        counts["after_eqm"] += budget
        counts["selected_before_dedup"] += budget
        frame = data.iloc[chosen][["id.symbol", "_ts"]].copy()
        frame["fold"] = fold["name"]
        frame["rv2_tail"] = tail[chosen]
        frame["eqm_score"] = scores[chosen]
        selected.append(frame)
    result = pd.concat(selected, ignore_index=True).rename(columns={"id.symbol": "symbol", "_ts": "ts"})
    counts["final_trades"] = len(result)
    return result, counts


def run_stage_1b_attempt(config: ReplayConfig, attempt: int) -> Mapping[str, Any]:
    preregistration = _load_preregistration(STAGE_1B_PREREGISTRATION, STAGE_1B_SHA256)
    _validate_stage_1b(preregistration)
    data, features, dataset_hash = _hourly_dataset(config, preregistration)
    tail, scores, model_evidence = _fit_stage_1b_scores(data, features)
    trades, counts = _select_historical_fold_scores(data, tail, scores)
    costs = json.loads(Path(config.payload["inputs"]["econ_report"]["path"]).read_text())["cost_scenarios"]["B_base"]
    trades = _attach_economics(trades, Path(config.payload["inputs"]["series_manifest"]["path"]), costs)
    metrics = _performance(trades, counts)
    by_fold = metrics["by_fold"]
    minimum = int(preregistration["folds"]["minimum_trades_each_fold"])
    insufficient = any(int(by_fold.get(f"fold_{fold}", {}).get("trades", 0)) < minimum for fold in range(1, 5))
    positive_folds = sum(float(by_fold[f"fold_{fold}"]["expectancy"]) > 0.0 for fold in range(1, 5))
    if insufficient:
        answer = "INSUFFICIENT_TRADES"
    elif metrics["net_expectancy"] > 0.0 and positive_folds >= 3:
        answer = "EDGE_PRESENT_ON_HOURLY"
    else:
        answer = "EDGE_ABSENT_ON_HOURLY"
    records = trades.assign(ts=trades["ts"].astype(str)).to_dict("records")
    scientific = {
        "schema_version": "aegis-stage-1b-result-v1",
        "stage": "STAGE_1B",
        "preregistration_sha256": STAGE_1B_SHA256,
        "dataset_hash": dataset_hash,
        "feature_count": len(features),
        "feature_hash": digest(list(features)),
        "label_source_sha256": HISTORICAL_LABEL_SOURCE_SHA256,
        "execution_identity": _execution_identity(config),
        "model_evidence": [asdict(item) for item in model_evidence],
        "metrics": metrics,
        "positive_expectancy_folds": positive_folds,
        "answer": answer,
        "trade_records": records,
        "trade_keys_hash": digest([{"fold": row["fold"], "symbol": row["symbol"], "ts": row["ts"]} for row in records]),
        "safety": {"dev_only": True, "lockbox": False, "semi_blind": False, "e3_validation": False},
    }
    payload = {**scientific, "attempt": attempt, "scientific_hash": digest(_canonical_scientific(scientific))}
    output = Path(config.payload["output_root"]) / "stage_1b" / f"attempt_{attempt}"
    atomic_write(output / "result.json", payload)
    return payload


def run_stage_1b(config: ReplayConfig) -> Mapping[str, Any]:
    first = run_stage_1b_attempt(config, 1)
    second = run_stage_1b_attempt(config, 2)
    if first["scientific_hash"] != second["scientific_hash"] or first["trade_keys_hash"] != second["trade_keys_hash"]:
        raise FollowupNondeterministic("NONDETERMINISTIC: Stage 1b attempts differ")
    summary = {key: value for key, value in first.items() if key not in {"attempt", "trade_records"}}
    summary["determinism"] = {
        "attempt_1": first["scientific_hash"], "attempt_2": second["scientific_hash"],
        "canonical_identical": True, "trade_keys_identical": True,
        "numeric_tolerance": DETERMINISM_TOLERANCE,
    }
    atomic_write(Path(config.payload["output_root"]) / "stage_1b.json", summary)
    return summary


def _select_stage_4b_variant(
    data: pd.DataFrame, tail: np.ndarray, scores: np.ndarray, variant: str,
) -> tuple[pd.DataFrame, Mapping[str, int]]:
    if variant not in {"STAGE_4B_A", "STAGE_4B_B", "STAGE_4B_C"}:
        raise FollowupProtocolError(f"PROTOCOL_VIOLATION: unknown Stage 4b variant {variant}")
    selected = []
    counts = {key: 0 for key in ("total_candidates", "before_trrm", "trrm_survivors", "before_eqm", "after_eqm", "selected_before_dedup")}
    for fold in make_folds(data):
        evaluation = fold["evaluation"]
        if variant == "STAGE_4B_A":
            retained = evaluation[tail[evaluation] <= 0.70]
            eligible = retained
        else:
            retained = _rank_survivors(evaluation, tail)
            eligible = retained if variant == "STAGE_4B_B" else retained[scores[retained] >= E2_THRESHOLD]
        ranked = eligible[np.argsort(-scores[eligible], kind="stable")]
        if variant == "STAGE_4B_B":
            candidate = data.iloc[ranked][["_ts", "id.symbol"]].copy()
            candidate["_index"] = ranked
            candidate["_score"] = scores[ranked]
            per_cycle = candidate.sort_values(
                ["_ts", "_score", "id.symbol"], ascending=[True, False, True], kind="mergesort",
            ).drop_duplicates("_ts")
            cycle_indices = per_cycle["_index"].to_numpy(dtype=int)
            subset = data.iloc[cycle_indices].reset_index(drop=True)
            chosen = cycle_indices[correlation_limit(subset, scores[cycle_indices])]
            selected_before_dedup = len(cycle_indices)
        else:
            budget = max(1, int(round(0.10 * len(retained))))
            subset = data.iloc[ranked].reset_index(drop=True)
            chosen = ranked[correlation_limit(subset, scores[ranked])][:budget]
            selected_before_dedup = min(budget, len(ranked))
        counts["total_candidates"] += len(evaluation)
        counts["before_trrm"] += len(evaluation)
        counts["trrm_survivors"] += len(retained)
        counts["before_eqm"] += len(retained)
        counts["after_eqm"] += len(eligible)
        counts["selected_before_dedup"] += selected_before_dedup
        frame = data.iloc[chosen][["id.symbol", "_ts"]].copy()
        frame["fold"] = fold["name"]
        frame["rv2_tail"] = tail[chosen]
        frame["eqm_score"] = scores[chosen]
        selected.append(frame)
    result = pd.concat(selected, ignore_index=True).rename(columns={"id.symbol": "symbol", "_ts": "ts"})
    counts["final_trades"] = len(result)
    return result, counts


def run_stage_4b_attempt(config: ReplayConfig, variant: str, attempt: int) -> Mapping[str, Any]:
    preregistration = _load_preregistration(STAGE_4B_PREREGISTRATION, STAGE_4B_SHA256)
    _validate_stage_4b(preregistration)
    inputs = config.payload["inputs"]
    data = _add_horizon_columns(_base_h12(Path(inputs["dataset"]["path"])))
    from .ablations import _frozen_scores

    tail, scores, model_evidence = _frozen_scores(data, Path(inputs["rv2_pickle"]["path"]), Path(inputs["eqm_pickle"]["path"]))
    trades, counts = _select_stage_4b_variant(data, tail, scores, variant)
    costs = json.loads(Path(inputs["econ_report"]["path"]).read_text())["cost_scenarios"]["B_base"]
    trades = _attach_economics(trades, Path(inputs["series_manifest"]["path"]), costs)
    metrics = _performance(trades, counts)
    stage0 = json.loads((Path(config.payload["output_root"]) / "stage_0.json").read_text())
    stage4 = json.loads((Path(config.payload["output_root"]) / "stage_4.json").read_text())
    records = trades.assign(ts=trades["ts"].astype(str)).to_dict("records")
    scientific = {
        "schema_version": "aegis-stage-4b-result-v1",
        "stage": variant,
        "preregistration_sha256": STAGE_4B_SHA256,
        "changed_axis": next(item["changed_axis"] for item in preregistration["variants"] if item["id"] == variant),
        "model_evidence": model_evidence,
        "execution_identity": _execution_identity(config),
        "metrics": metrics,
        "delta_stage_0": {
            "trades": metrics["trades"] - stage0["trades"],
            "profit_factor": metrics["net_profit_factor"] - stage0["profit_factor"],
            "expectancy": metrics["net_expectancy"] - stage0["net_expectancy"],
        },
        "delta_stage_4": {
            "trades": metrics["trades"] - stage4["metrics"]["trades"],
            "profit_factor": metrics["net_profit_factor"] - stage4["metrics"]["net_profit_factor"],
            "expectancy": metrics["net_expectancy"] - stage4["metrics"]["net_expectancy"],
        },
        "trade_records": records,
        "trade_keys_hash": digest([{"fold": row["fold"], "symbol": row["symbol"], "ts": row["ts"]} for row in records]),
        "safety": {"dev_only": True, "lockbox": False, "semi_blind": False, "e3_validation": False},
    }
    payload = {**scientific, "attempt": attempt, "scientific_hash": digest(_canonical_scientific(scientific))}
    output = Path(config.payload["output_root"]) / "stage_4b" / variant.lower() / f"attempt_{attempt}"
    atomic_write(output / "result.json", payload)
    return payload


def run_stage_4b(config: ReplayConfig) -> Mapping[str, Any]:
    summaries = []
    for variant in ("STAGE_4B_A", "STAGE_4B_B", "STAGE_4B_C"):
        first = run_stage_4b_attempt(config, variant, 1)
        second = run_stage_4b_attempt(config, variant, 2)
        if first["scientific_hash"] != second["scientific_hash"] or first["trade_keys_hash"] != second["trade_keys_hash"]:
            raise FollowupNondeterministic(f"NONDETERMINISTIC: {variant} attempts differ")
        summary = {key: value for key, value in first.items() if key not in {"attempt", "trade_records"}}
        summary["determinism"] = {
            "attempt_1": first["scientific_hash"], "attempt_2": second["scientific_hash"],
            "canonical_identical": True, "trade_keys_identical": True,
            "numeric_tolerance": DETERMINISM_TOLERANCE,
        }
        summaries.append(summary)
    payload = {
        "schema_version": "aegis-stage-4b-attribution-v1",
        "preregistration_sha256": STAGE_4B_SHA256,
        "variants": summaries,
        "safety": {"lockbox": False, "semi_blind": False, "e3_validation": False},
    }
    result = {**payload, "scientific_hash": digest(_canonical_scientific(payload))}
    atomic_write(Path(config.payload["output_root"]) / "stage_4b.json", result)
    return result
