"""Dataset preparation helpers for the neural trading model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.storage.database_manager import db_manager
from .features import build_feature_frame


@dataclass
class DatasetConfig:
    """Configuration for slicing OHLCV history into supervised samples."""

    symbol: str = "XRP/USDT"
    timeframe: str = "5m"
    prediction_horizon: int = 12  # bars ahead
    target_return: float = 0.002  # 0.2%
    target_return_long: Optional[float] = None
    target_return_short: Optional[float] = None
    min_records: int = 2000
    max_history_days: Optional[int] = None
    max_samples: Optional[int] = None  # rolling window (most recent N rows)


class PatternTensorDataset(Dataset):
    """Simple torch Dataset holding feature and label tensors."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        label_dtype: Optional[torch.dtype] = None,
    ):
        self.features = torch.from_numpy(features.astype(np.float32))

        if label_dtype is None:
            if np.issubdtype(labels.dtype, np.integer):
                tensor = torch.from_numpy(labels.astype(np.int64))
            else:
                tensor = torch.from_numpy(labels.astype(np.float32))
        else:
            if label_dtype in (torch.int64, torch.long, torch.int32, torch.int16):
                tensor = torch.from_numpy(labels.astype(np.int64))
                tensor = tensor.to(label_dtype)
            else:
                tensor = torch.from_numpy(labels.astype(np.float32)).to(label_dtype)

        self.labels = tensor

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def _symbol_variants(symbol: str) -> List[str]:
    """Return possible DB symbol keys covering legacy and futures formats."""
    variants: List[str] = []
    seen: set[str] = set()

    def add(value: Optional[str]) -> None:
        if not value:
            return
        if value in seen:
            return
        seen.add(value)
        variants.append(value)

    add(symbol)

    if ":" in symbol:
        add(symbol.split(":", 1)[0])
    elif "/" in symbol:
        base, quote = symbol.split("/", 1)
        add(f"{base}/{quote}:{quote}")

    return variants


def load_feature_matrix(cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Load OHLCV data from the DB, engineer features, and derive classification labels.

    Returns:
        features_df: Feature matrix indexed by timestamp.
        labels_df: DataFrame with columns ['long', 'short'] of 0/1 targets.
        feature_cols: Ordered list of feature names.
    """
    effective_min = cfg.min_records
    if cfg.max_samples is not None:
        effective_min = min(effective_min, cfg.max_samples)
    required = effective_min + cfg.prediction_horizon
    raw_df: Optional[pd.DataFrame] = None
    resolved_symbol: Optional[str] = None

    for candidate in _symbol_variants(cfg.symbol):
        candidate_df = db_manager.get_ohlcv_data(candidate, cfg.timeframe)
        if candidate_df.empty:
            continue

        candidate_df = candidate_df.sort_index()
        if cfg.max_history_days:
            latest_ts = candidate_df.index.max()
            if pd.isna(latest_ts):
                continue
            cutoff = latest_ts - pd.Timedelta(days=cfg.max_history_days)
            candidate_df = candidate_df[candidate_df.index >= cutoff]

        if cfg.max_samples is not None and len(candidate_df) > cfg.max_samples:
            candidate_df = candidate_df.tail(cfg.max_samples + cfg.prediction_horizon)

        if len(candidate_df) >= required:
            raw_df = candidate_df
            resolved_symbol = candidate
            break

        if raw_df is None or len(candidate_df) > len(raw_df):
            raw_df = candidate_df
            resolved_symbol = candidate

    if raw_df is None or resolved_symbol is None or len(raw_df) < required:
        available = len(raw_df) if raw_df is not None else 0
        tried = ", ".join(_symbol_variants(cfg.symbol))
        raise RuntimeError(
            f"Not enough {cfg.timeframe} data for {cfg.symbol} (tried: {tried}). "
            f"Need at least {required} rows, found {available}."
        )

    features_df, feature_cols = build_feature_frame(raw_df)
    features_df.attrs["source_symbol"] = resolved_symbol

    aligned_close = raw_df.loc[features_df.index, "close"]
    future_close = aligned_close.shift(-cfg.prediction_horizon)
    future_return = (future_close / aligned_close) - 1.0

    labels_df = pd.DataFrame(index=features_df.index)
    long_threshold = cfg.target_return_long if cfg.target_return_long is not None else cfg.target_return
    short_threshold = cfg.target_return_short if cfg.target_return_short is not None else cfg.target_return
    labels_df["long"] = (future_return >= long_threshold).astype(int)
    labels_df["short"] = (future_return <= -short_threshold).astype(int)

    mask = future_return.notna()
    features_df = features_df[mask]
    labels_df = labels_df[mask]
    labels_df.attrs["source_symbol"] = resolved_symbol

    return features_df, labels_df, feature_cols
