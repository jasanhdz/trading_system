"""Frozen-contract helpers for the A2 external historical backcast."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from aegis.utils import sha256_file

from .economic_alpha_discovery_a1 import (
    RobustScale,
    cross_sectional_winners,
    deterministic_random_symbol,
    mechanism_rows,
)


A1_RESULT_SHA256 = "fed5675dc88afe2c1bf40c8c61eaa937aa9460bca3bb26e1c43ac3b773840fa1"
COSTS = {
    "zero": 0.0,
    "optimistic_8bps": 0.0008,
    "primary_14bps": 0.0014,
    "stress_20bps": 0.0020,
}


class A2ContractError(ValueError):
    pass


def load_frozen_a1_contract(path: Path) -> Mapping[str, Any]:
    if sha256_file(path) != A1_RESULT_SHA256:
        raise A2ContractError("AEGIS_A2_A1_AUTHORITY_HASH_MISMATCH")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "scales",
        "eligibility_thresholds",
        "calibration_score_thresholds",
        "regime_thresholds",
    }
    if not required.issubset(payload):
        raise A2ContractError("AEGIS_A2_A1_AUTHORITY_SCHEMA_MISMATCH")
    return payload


def frozen_scales(payload: Mapping[str, Any], side: str) -> dict[str, RobustScale]:
    try:
        raw = payload["scales"][side]
        return {name: RobustScale(**values) for name, values in raw.items()}
    except (KeyError, TypeError) as exc:
        raise A2ContractError("AEGIS_A2_A1_SCALE_CONTRACT_INVALID") from exc


def classify_frozen_regimes(
    panel: pd.DataFrame, thresholds: Mapping[str, float]
) -> pd.DataFrame:
    result = panel.copy()
    direction = float(thresholds["btc_abs_return_4h_q70"])
    low = float(thresholds["volatility_q30"])
    high = float(thresholds["volatility_q70"])
    result["direction_regime"] = np.select(
        [result["btc_return_4h"].gt(direction), result["btc_return_4h"].lt(-direction)],
        ["UP", "DOWN"],
        default="TRANSITION",
    )
    result["volatility_regime"] = np.select(
        [result["realized_volatility_24h"].lt(low), result["realized_volatility_24h"].gt(high)],
        ["COMPRESSED", "EXPANDING"],
        default="NORMAL",
    )
    return result


def frozen_candidates(
    panel: pd.DataFrame,
    payload: Mapping[str, Any],
    *,
    mechanism: str,
    side: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    identity = f"{mechanism}::{side}"
    rows = mechanism_rows(
        panel,
        side=side,
        mechanism=mechanism,
        scales=frozen_scales(payload, side),
        thresholds=payload["eligibility_thresholds"][side],
    )
    threshold = payload["calibration_score_thresholds"].get(identity)
    if threshold is None or not math.isfinite(float(threshold)):
        raise A2ContractError("AEGIS_A2_A1_SCORE_THRESHOLD_INVALID")
    context = panel[
        ["timestamp_ms", "symbol", "direction_regime", "volatility_regime"]
    ]
    rows = rows.merge(
        context,
        on=["timestamp_ms", "symbol"],
        how="left",
        validate="one_to_one",
    )
    return rows.loc[
        rows["timestamp_ms"].ge(start_ms)
        & rows["timestamp_ms"].lt(end_ms)
        & rows["score"].ge(float(threshold))
    ].copy()


def fixed_horizon_outcomes(
    events: pd.DataFrame,
    minute_frames: Mapping[str, pd.DataFrame],
    funding_frames: Mapping[str, pd.DataFrame],
    horizon_minutes: int,
) -> pd.DataFrame:
    outputs: list[dict[str, Any]] = []
    for symbol, group in events.groupby("symbol", sort=True):
        minute = minute_frames[symbol].sort_values("open_time", ignore_index=True)
        times = minute["open_time"].to_numpy(dtype=np.int64)
        opens = minute["open"].to_numpy(dtype=float)
        highs = minute["high"].to_numpy(dtype=float)
        lows = minute["low"].to_numpy(dtype=float)
        funding = funding_frames[symbol]
        for row in group.itertuples(index=False):
            entry_time = int(row.timestamp_ms) + 900_000
            entry_index = int(np.searchsorted(times, entry_time))
            exit_time = entry_time + horizon_minutes * 60_000
            exit_index = int(np.searchsorted(times, exit_time))
            if (
                entry_index >= len(times)
                or exit_index >= len(times)
                or times[entry_index] != entry_time
                or times[exit_index] != exit_time
            ):
                continue
            path_high = highs[entry_index:exit_index]
            path_low = lows[entry_index:exit_index]
            if len(path_high) != horizon_minutes:
                continue
            entry = float(opens[entry_index])
            sign = 1.0 if row.side == "LONG" else -1.0
            gross = sign * (float(opens[exit_index]) / entry - 1.0)
            if row.side == "LONG":
                mae = max(0.0, 1.0 - float(path_low.min()) / entry)
                mfe = max(0.0, float(path_high.max()) / entry - 1.0)
                favorable = path_high / entry - 1.0
            else:
                mae = max(0.0, float(path_high.max()) / entry - 1.0)
                mfe = max(0.0, 1.0 - float(path_low.min()) / entry)
                favorable = 1.0 - path_low / entry
            paid = funding.loc[
                funding["funding_time"].ge(entry_time)
                & funding["funding_time"].lt(exit_time),
                "funding_rate",
            ].sum()
            funding_return = -sign * float(paid)
            economic_gross = gross + funding_return
            crossing = np.flatnonzero(favorable > COSTS["primary_14bps"])
            output = {
                **row._asdict(),
                "horizon_minutes": horizon_minutes,
                "entry_time": entry_time,
                "entry_price": entry,
                "exit_price": float(opens[exit_index]),
                "gross_return": economic_gross,
                "funding_return": funding_return,
                "mae": mae,
                "mfe": mfe,
                "time_to_first_positive_minutes": int(crossing[0] + 1)
                if len(crossing)
                else None,
            }
            for name, cost in COSTS.items():
                output[f"net_{name}"] = economic_gross - cost
            outputs.append(output)
    return pd.DataFrame(outputs)


def economic_summary(
    rows: pd.DataFrame, column: str = "net_primary_14bps"
) -> dict[str, Any]:
    if rows.empty:
        return {"events": 0}
    values = rows[column].to_numpy(dtype=float)
    gains = float(values[values > 0.0].sum())
    losses = float(-values[values < 0.0].sum())
    ordered = rows.sort_values(["timestamp_ms", "symbol"])
    thirds = [
        float(
            ordered.iloc[
                index * len(ordered) // 3 : (index + 1) * len(ordered) // 3
            ][column].mean()
        )
        for index in range(3)
    ]
    symbols = rows.groupby("symbol")[column].mean()
    return {
        "events": len(rows),
        "expectancy": float(values.mean()),
        "profit_factor": gains / losses if losses else (math.inf if gains else 0.0),
        "win_rate": float((values > 0.0).mean()),
        "mean_mae": float(rows["mae"].mean()),
        "mean_mfe": float(rows["mfe"].mean()),
        "maximum_symbol_share": float(rows["symbol"].value_counts(normalize=True).max()),
        "positive_symbols": int((symbols > 0.0).sum()),
        "temporal_thirds": thirds,
    }


def clustered_bootstrap(
    rows: pd.DataFrame, *, repetitions: int = 2000, seed: int = 182023
) -> dict[str, float]:
    if rows.empty:
        raise A2ContractError("AEGIS_A2_BOOTSTRAP_EMPTY")
    clustered = (
        rows.groupby("timestamp_ms", as_index=False)["net_primary_14bps"].mean()
        .assign(
            day=lambda frame: pd.to_datetime(
                frame["timestamp_ms"], unit="ms", utc=True
            ).dt.floor("1D")
        )
    )
    days = [
        group["net_primary_14bps"].to_numpy(dtype=float)
        for _, group in clustered.groupby("day", sort=True)
    ]
    random = np.random.default_rng(seed)
    means: list[float] = []
    factors: list[float] = []
    for _ in range(repetitions):
        sample = np.concatenate(
            [days[index] for index in random.integers(0, len(days), len(days))]
        )
        gains = float(sample[sample > 0.0].sum())
        losses = float(-sample[sample < 0.0].sum())
        means.append(float(sample.mean()))
        factors.append(gains / losses if losses else math.inf)
    return {
        "expectancy_lower_95": float(np.quantile(means, 0.025)),
        "expectancy_upper_95": float(np.quantile(means, 0.975)),
        "profit_factor_lower_95": float(np.quantile(factors, 0.025)),
    }


def control_events(
    selected: pd.DataFrame, eligible: pd.DataFrame, panel: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    panel_index = panel.set_index(["timestamp_ms", "symbol"], drop=False)
    eligible_groups = {
        timestamp: group for timestamp, group in eligible.groupby("timestamp_ms")
    }
    names = (
        "random_eligible",
        "regime_matched_random",
        "simple_momentum",
        "simple_reversal",
        "btc_reference",
    )
    outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for event in selected.itertuples(index=False):
        pool = eligible_groups.get(int(event.timestamp_ms))
        if pool is None or pool.empty:
            continue
        random_symbol = deterministic_random_symbol(
            pool, f"a2:{event.mechanism}:{event.side}:{int(event.timestamp_ms)}"
        )
        regime_pool = pool.loc[
            pool["direction_regime"].eq(event.direction_regime)
            & pool["volatility_regime"].eq(event.volatility_regime)
        ]
        if regime_pool.empty:
            regime_pool = pool
        regime_symbol = deterministic_random_symbol(
            regime_pool,
            f"a2:regime:{event.mechanism}:{event.side}:{int(event.timestamp_ms)}",
        )
        snapshot = panel.loc[panel["timestamp_ms"].eq(event.timestamp_ms)]
        ascending = event.side == "SHORT"
        momentum = str(
            snapshot.sort_values(
                ["return_4h", "symbol"], ascending=[ascending, True]
            ).iloc[0]["symbol"]
        )
        reversal = str(
            snapshot.sort_values(
                ["return_4h", "symbol"], ascending=[not ascending, True]
            ).iloc[0]["symbol"]
        )
        choices = {
            "random_eligible": random_symbol,
            "regime_matched_random": regime_symbol,
            "simple_momentum": momentum,
            "simple_reversal": reversal,
            "btc_reference": "BTCUSDT",
        }
        for name, symbol in choices.items():
            source = panel_index.loc[(event.timestamp_ms, symbol)]
            outputs[name].append(
                {
                    "timestamp_ms": int(event.timestamp_ms),
                    "state_close_ms": int(source.state_close_ms),
                    "symbol": symbol,
                    "side": event.side,
                    "mechanism": event.mechanism,
                    "score": float("nan"),
                    "return_4h": float(source.return_4h),
                    "realized_volatility_24h": float(source.realized_volatility_24h),
                    "cross_sectional_return_rank_4h": float(
                        source.cross_sectional_return_rank_4h
                    ),
                    "direction_regime": source.direction_regime,
                    "volatility_regime": source.volatility_regime,
                }
            )
    return {name: pd.DataFrame(values) for name, values in outputs.items()}


def event_identity_hash(rows: pd.DataFrame) -> str:
    identities = rows[["timestamp_ms", "symbol", "side", "mechanism"]].sort_values(
        ["timestamp_ms", "symbol", "side", "mechanism"]
    )
    return hashlib.sha256(identities.to_csv(index=False).encode("ascii")).hexdigest()


def top_rank(rows: pd.DataFrame) -> pd.DataFrame:
    return cross_sectional_winners(rows)
