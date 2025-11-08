#!/usr/bin/env python3
"""
Train the neural probability model with stronger validation and metrics.
"""
from __future__ import annotations

import copy
import json
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ml.nn_pattern.dataset import DatasetConfig, PatternTensorDataset, load_feature_matrix
from ml.nn_pattern.model import PatternNet
from utils.logger import setup_logger

logger = setup_logger("nn_trainer")

MODEL_DIR = (REPO_ROOT / "models" / "trained").resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HISTORY_DAYS = {
    "5m": 180,
    "15m": 270,
}

DEFAULT_CLASS_LABELS = ["neutral", "long", "short"]


def _to_ccxt_symbol(binance_symbol: str) -> str:
    clean = binance_symbol.replace("/", "").replace(" ", "").upper()
    quote_tokens = ("USDT", "BUSD", "USDC", "BTC", "ETH")
    for quote in quote_tokens:
        if clean.endswith(quote):
            base = clean[: -len(quote)]
            return f"{base}/{quote}:USDT" if quote == "USDT" else f"{base}/{quote}"
    return clean


def _symbol_key(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "").replace("-", "").upper()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _chronological_split(
    total: int,
    train_ratio: float,
    valid_ratio: float,
) -> Tuple[slice, slice, slice]:
    train_end = max(int(total * train_ratio), 1)
    valid_end = train_end + max(int(total * valid_ratio), 0)
    train_slice = slice(0, train_end)
    valid_slice = slice(train_end, min(valid_end, total))
    test_slice = slice(valid_end, total)
    return train_slice, valid_slice, test_slice


def _labels_to_classes(labels: np.ndarray) -> np.ndarray:
    long_mask = labels[:, 0] >= 0.5
    short_mask = labels[:, 1] >= 0.5
    classes = np.zeros(len(labels), dtype=np.int64)
    classes[long_mask] = 1
    classes[short_mask] = 2
    return classes


def _prepare_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = PatternTensorDataset(features, targets, label_dtype=torch.long)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _evaluate_model(
    model: PatternNet,
    loader: DataLoader,
    device: torch.device,
    class_labels: List[str],
    criterion: Optional[nn.Module] = None,
    return_details: bool = False,
) -> Dict[str, float | Dict[str, Dict[str, float]]]:
    if loader.dataset is None or len(loader.dataset) == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "macro_f1": float("nan")}

    model.eval()
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_X)
            all_logits.append(logits.cpu())
            all_targets.append(batch_y.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    predictions = probs.argmax(axis=1)
    y_true = targets.numpy()

    loss = float("nan")
    if criterion is not None:
        with torch.no_grad():
            loss = float(criterion(logits, targets).item())

    accuracy = float((predictions == y_true).mean())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=list(range(len(class_labels))),
        zero_division=0,
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(class_labels):
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": float(support[idx]),
        }

    macro_f1 = float(np.mean(f1)) if len(f1) > 0 else float("nan")

    # Compute average precision for long/short when possible
    diagnostics: Dict[str, float] = {}
    for label in ("long", "short"):
        if label in class_labels:
            idx = class_labels.index(label)
            positives = (y_true == idx).astype(int)
            if positives.any():
                diagnostics[f"ap_{label}"] = float(
                    average_precision_score(positives, probs[:, idx])
                )
            else:
                diagnostics[f"ap_{label}"] = float("nan")

    metrics: Dict[str, float | Dict[str, Dict[str, float]]] = {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
    metrics.update(diagnostics)
    if return_details:
        metrics["details"] = {
            "predictions": predictions,
            "targets": y_true,
            "probabilities": probs,
        }
    return metrics


def _class_weights(
    targets: np.ndarray,
    num_classes: int,
    multipliers: Optional[Tuple[float, ...]] = None,
) -> torch.Tensor:
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    if (counts == 0).any():
        logger.warning("Class imbalance detected; falling back to uniform weights.")
        weights = np.ones(num_classes, dtype=np.float32)
    else:
        inv = counts.sum() / counts
        weights = (inv / inv.mean()).astype(np.float32)
    if multipliers:
        factors = np.array(multipliers, dtype=np.float32)
        if len(factors) != num_classes:
            raise ValueError("Class weight multipliers must match number of classes.")
        weights = weights * factors
    return torch.from_numpy(weights.astype(np.float32))


def _parse_prefix_option(option: Optional[str]) -> Optional[List[str]]:
    if not option:
        return None
    tokens = [token.strip() for token in option.split(",") if token.strip()]
    return tokens or None


@dataclass
class TrainingParams:
    epochs: int
    batch_size: int
    lr: float
    hidden_layers: Tuple[int, ...]
    dropout: float
    patience: int
    min_delta: float
    seed: int
    device: torch.device
    scheduler: str = "none"
    class_weight_multipliers: Optional[Tuple[float, ...]] = None


def _filter_feature_columns(
    df: pd.DataFrame,
    keep_prefixes: Optional[List[str]] = None,
    drop_prefixes: Optional[List[str]] = None,
) -> pd.DataFrame:
    result = df
    if keep_prefixes:
        tokens = [token.strip() for token in keep_prefixes if token.strip()]
        if tokens:
            mask = [
                any(col.startswith(token) or token in col for token in tokens) for col in result.columns
            ]
            selected = [col for col, ok in zip(result.columns, mask) if ok]
            if selected:
                result = result[selected]
            else:
                warnings.warn("Feature keep prefixes filtered out all columns; keeping original set.")
    if drop_prefixes:
        tokens = [token.strip() for token in drop_prefixes if token.strip()]
        if tokens:
            selected = [
                col for col in result.columns if not any(col.startswith(token) or token in col for token in tokens)
            ]
            if selected:
                result = result[selected]
    return result


def _filter_outliers(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    return_feature: Optional[str],
    return_cut: Optional[float],
    volume_feature: Optional[str],
    volume_cut: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    mask = pd.Series(True, index=features_df.index)
    stats: Dict[str, int] = {}

    if return_feature and return_cut is not None:
        if return_feature in features_df.columns:
            series = features_df[return_feature].astype(float)
            zscores = ((series - series.mean()) / (series.std(ddof=0) or 1.0)).abs()
            keep = (zscores <= return_cut) | series.isna()
            stats["return_filtered"] = int((~keep).sum())
            mask &= keep.fillna(True)
        else:
            logger.warning("Return feature '%s' not found; skipping return filter.", return_feature)

    if volume_feature and volume_cut is not None:
        if volume_feature in features_df.columns:
            series = features_df[volume_feature].astype(float)
            zscores = ((series - series.mean()) / (series.std(ddof=0) or 1.0)).abs()
            keep = (zscores <= volume_cut) | series.isna()
            stats["volume_filtered"] = int((~keep).sum())
            mask &= keep.fillna(True)
        else:
            logger.warning("Volume feature '%s' not found; skipping volume filter.", volume_feature)

    filtered_features = features_df[mask].dropna()
    filtered_labels = labels_df.loc[filtered_features.index]
    stats["remaining"] = len(filtered_features)
    return filtered_features, filtered_labels, stats


def _augment_training_data(
    train_X: np.ndarray,
    train_y: np.ndarray,
    noise_std: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if noise_std <= 0.0:
        return train_X, train_y
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=train_X.shape).astype(np.float32)
    augmented_X = np.concatenate([train_X, train_X + noise], axis=0)
    augmented_y = np.concatenate([train_y, train_y], axis=0)
    return augmented_X, augmented_y


def _compute_regime_metrics(
    details: Dict[str, np.ndarray],
    regime_series: Optional[pd.Series],
    class_labels: List[str],
) -> Dict[str, Dict[str, float]]:
    if regime_series is None or regime_series.empty:
        return {}
    predictions = details["predictions"]
    targets = details["targets"]
    metrics: Dict[str, Dict[str, float]] = {}
    for regime in sorted(regime_series.dropna().unique()):
        mask = regime_series == regime
        idx = np.where(mask.to_numpy())[0]
        if len(idx) < 20:
            continue
        subset_preds = predictions[idx]
        subset_true = targets[idx]
        precision, recall, f1, support = precision_recall_fscore_support(
            subset_true,
            subset_preds,
            labels=list(range(len(class_labels))),
            zero_division=0,
        )
        metrics[str(regime)] = {
            "macro_f1": float(np.mean(f1)),
            "accuracy": float((subset_preds == subset_true).mean()),
            "support": int(len(idx)),
        }
    return metrics


def _prepare_scaler(kind: str) -> StandardScaler | RobustScaler:
    if kind == "robust":
        return RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
    return StandardScaler()


def _maybe_step_scheduler(scheduler, value: float) -> None:
    if scheduler is None:
        return
    try:
        scheduler.step(value)
    except Exception as exc:
        logger.warning("scheduler_step_failed", {"error": str(exc)})


def train_pattern_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    valid_X: np.ndarray,
    valid_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    class_labels: List[str],
    params: TrainingParams,
) -> Tuple[PatternNet, int, Dict[str, any], Dict[str, any], Dict[str, any]]:
    num_classes = len(class_labels)
    train_loader = _prepare_loader(train_X, train_y, params.batch_size, shuffle=True, seed=params.seed)
    valid_loader = (
        _prepare_loader(valid_X, valid_y, params.batch_size, shuffle=False, seed=params.seed)
        if len(valid_y)
        else None
    )
    test_loader = (
        _prepare_loader(test_X, test_y, params.batch_size, shuffle=False, seed=params.seed)
        if len(test_y)
        else None
    )

    model = PatternNet(
        input_dim=train_X.shape[1],
        hidden_dims=params.hidden_layers or (128, 64),
        dropout=params.dropout,
        num_classes=num_classes,
        output_activation="softmax",
    ).to(params.device)

    class_weight_tensor = _class_weights(train_y, num_classes, params.class_weight_multipliers).to(params.device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    scheduler = None
    if params.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=max(2, params.patience // 2),
            threshold=params.min_delta / 2,
            min_lr=params.lr * 0.1,
        )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_metric = -float("inf")
    epochs_without_improvement = 0

    logger.info(
        "Starting training",
        {
            "epochs": params.epochs,
            "train_samples": len(train_loader.dataset),
            "valid_samples": len(valid_loader.dataset) if valid_loader else 0,
            "test_samples": len(test_loader.dataset) if test_loader else 0,
        },
    )

    for epoch in range(1, params.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0

        for batch_X, batch_y in train_loader:
            if batch_X.size(0) <= 1:
                continue
            batch_X = batch_X.to(params.device)
            batch_y = batch_y.to(params.device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            seen += batch_X.size(0)

        train_loss = epoch_loss / max(seen, 1)

        valid_metrics = (
            _evaluate_model(model, valid_loader, params.device, class_labels, criterion)
            if valid_loader
            else {"loss": float("nan"), "accuracy": float("nan"), "macro_f1": float("nan")}
        )

        metric = valid_metrics.get("macro_f1", float("nan"))
        if np.isnan(metric):
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metric = metric
        else:
            if metric - best_metric > params.min_delta:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            _maybe_step_scheduler(scheduler, metric)

        logger.info(
            "epoch_metrics",
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "valid_loss": round(valid_metrics.get("loss", float("nan")), 6),
                "valid_macro_f1": round(valid_metrics.get("macro_f1", float("nan")), 6),
                "valid_accuracy": round(valid_metrics.get("accuracy", float("nan")), 6),
            },
        )

        if params.patience and epochs_without_improvement >= params.patience:
            logger.info("Early stopping triggered", {"epoch": epoch})
            break

    model.load_state_dict(best_state)

    train_metrics = _evaluate_model(model, train_loader, params.device, class_labels, criterion)
    valid_metrics = (
        _evaluate_model(model, valid_loader, params.device, class_labels, criterion) if valid_loader else {}
    )
    test_metrics = (
        _evaluate_model(model, test_loader, params.device, class_labels, criterion, return_details=True)
        if test_loader
        else {}
    )

    return model, best_epoch, train_metrics, valid_metrics, test_metrics, best_metric


def _walk_forward_evaluation(
    features: np.ndarray,
    targets: np.ndarray,
    class_labels: List[str],
    params: TrainingParams,
    scaler_kind: str,
    augmentation_noise: float,
    folds: int,
    valid_ratio: float,
) -> List[Dict[str, float]]:
    if folds <= 0:
        return []

    total = len(features)
    fold_metrics: List[Dict[str, float]] = []
    fold_size = total // (folds + 1)
    indices = np.arange(total)
    for fold in range(folds):
        train_end = fold_size * (fold + 1)
        test_end = fold_size * (fold + 2) if fold < folds - 1 else total
        if test_end - train_end < 200 or train_end < 500:
            logger.warning("Skipping walk-forward fold %s due to insufficient data.", fold + 1)
            continue

        train_idx = indices[:train_end]
        test_idx = indices[train_end:test_end]

        train_X_raw = features[train_idx]
        train_y_raw = targets[train_idx]
        test_X_raw = features[test_idx]
        test_y_raw = targets[test_idx]

        if valid_ratio > 0.0 and len(train_X_raw) > 200:
            valid_size = int(len(train_X_raw) * valid_ratio)
            valid_size = max(valid_size, 64)
            valid_size = min(valid_size, len(train_X_raw) // 3)
        else:
            valid_size = 0

        if valid_size > 0:
            fold_train_X = train_X_raw[:-valid_size]
            fold_train_y = train_y_raw[:-valid_size]
            fold_valid_X = train_X_raw[-valid_size:]
            fold_valid_y = train_y_raw[-valid_size:]
        else:
            fold_train_X = train_X_raw
            fold_train_y = train_y_raw
            fold_valid_X = np.empty((0, train_X_raw.shape[1]), dtype=train_X_raw.dtype)
            fold_valid_y = np.empty((0,), dtype=train_y_raw.dtype)

        scaler = _prepare_scaler(scaler_kind)
        fold_train_X = scaler.fit_transform(fold_train_X)
        fold_valid_X = scaler.transform(fold_valid_X) if len(fold_valid_X) else fold_valid_X
        fold_test_X = scaler.transform(test_X_raw)

        fold_train_X, fold_train_y = _augment_training_data(
            fold_train_X,
            fold_train_y,
            augmentation_noise,
            params.seed + fold,
        )

        fold_params = TrainingParams(
            epochs=params.epochs,
            batch_size=params.batch_size,
            lr=params.lr,
            hidden_layers=params.hidden_layers,
            dropout=params.dropout,
            patience=params.patience,
            min_delta=params.min_delta,
            seed=params.seed + fold,
            device=params.device,
            scheduler=params.scheduler,
            class_weight_multipliers=params.class_weight_multipliers,
        )

        _, best_epoch, _, _, fold_test_metrics, _ = train_pattern_model(
            fold_train_X,
            fold_train_y,
            fold_valid_X,
            fold_valid_y,
            fold_test_X,
            test_y_raw,
            class_labels,
            fold_params,
        )

        if isinstance(fold_test_metrics, dict):
            fold_test_metrics.pop("details", None)

        metrics_copy = {
            "fold": fold + 1,
            "train_end": int(train_end),
            "test_size": int(len(test_idx)),
            "best_epoch": best_epoch,
            "accuracy": float(fold_test_metrics.get("accuracy", float("nan"))),
            "macro_f1": float(fold_test_metrics.get("macro_f1", float("nan"))),
            "ap_long": float(fold_test_metrics.get("ap_long", float("nan"))),
            "ap_short": float(fold_test_metrics.get("ap_short", float("nan"))),
        }
        fold_metrics.append(metrics_copy)

    return fold_metrics


@click.command()
@click.option("--symbol", default="XRPUSDT", show_default=True, help="Binance futures symbol.")
@click.option("--timeframe", default="5m", show_default=True, help="Timeframe to train on.")
@click.option("--horizon", default=12, show_default=True, help="Bars ahead for the label.")
@click.option(
    "--target-return",
    default=0.002,
    show_default=True,
    help="Return threshold (e.g., 0.002 = 0.2%).",
)
@click.option(
    "--short-target-return",
    type=float,
    default=None,
    help="Return threshold for short labels (defaults to --target-return).",
)
@click.option("--epochs", default=60, show_default=True, help="Maximum training epochs.")
@click.option("--batch-size", default=512, show_default=True, help="Mini-batch size.")
@click.option("--lr", default=1e-3, show_default=True, help="Learning rate.")
@click.option(
    "--hidden-dims",
    default="128,64",
    show_default=True,
    help="Comma separated hidden layer sizes.",
)
@click.option("--dropout", default=0.2, show_default=True, help="Dropout probability.")
@click.option(
    "--history-days",
    type=int,
    default=None,
    help="Limit training to the most recent N days (defaults depend on timeframe).",
)
@click.option(
    "--max-samples",
    type=int,
    default=30000,
    show_default=True,
    help="Limit to the latest N samples (after indicators warm-up).",
)
@click.option("--train-ratio", default=0.7, show_default=True, help="Share for training set.")
@click.option("--valid-ratio", default=0.2, show_default=True, help="Share for validation set.")
@click.option("--test-ratio", default=0.1, show_default=True, help="Share for test set.")
@click.option("--patience", default=8, show_default=True, help="Early stopping patience.")
@click.option(
    "--min-delta",
    default=5e-4,
    show_default=True,
    help="Minimum macro-F1 improvement to reset patience.",
)
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "cuda"]),
    show_default=True,
    help="Torch device.",
)
@click.option(
    "--feature-keep-prefixes",
    default=None,
    help="Comma separated prefixes to keep (e.g., 'ema_,rsi_'). Keeps all if omitted.",
)
@click.option(
    "--feature-drop-prefixes",
    default=None,
    help="Comma separated prefixes to drop (e.g., 'volume_,mfi').",
)
@click.option(
    "--return-feature",
    default="roc_10",
    show_default=True,
    help="Feature used to detect price return outliers.",
)
@click.option(
    "--return-zscore-cut",
    type=float,
    default=None,
    help="Z-score threshold to drop extreme return observations.",
)
@click.option(
    "--volume-feature",
    default="volume_ratio_20",
    show_default=True,
    help="Feature used to detect abnormal volume.",
)
@click.option(
    "--volume-zscore-cut",
    type=float,
    default=None,
    help="Z-score threshold to drop abnormal volume observations.",
)
@click.option(
    "--scaler-kind",
    default="standard",
    type=click.Choice(["standard", "robust"]),
    show_default=True,
    help="Scaling strategy for features.",
)
@click.option(
    "--augmentation-noise",
    type=float,
    default=0.0,
    show_default=True,
    help="Std-dev of Gaussian noise added for data augmentation (0 disables).",
)
@click.option(
    "--lr-scheduler",
    default="none",
    type=click.Choice(["none", "plateau"]),
    show_default=True,
    help="Optional learning rate scheduler.",
)
@click.option(
    "--walk-forward-folds",
    default=0,
    show_default=True,
    type=int,
    help="Number of walk-forward folds for evaluation (0 disables).",
)
@click.option(
    "--walk-forward-valid-ratio",
    default=0.1,
    show_default=True,
    type=float,
    help="Validation share inside each walk-forward training window.",
)
@click.option(
    "--class-weight-neutral",
    default=1.0,
    show_default=True,
    type=float,
    help="Multiplier applied to neutral class weight.",
)
@click.option(
    "--class-weight-long",
    default=1.0,
    show_default=True,
    type=float,
    help="Multiplier applied to long class weight.",
)
@click.option(
    "--class-weight-short",
    default=1.0,
    show_default=True,
    type=float,
    help="Multiplier applied to short class weight.",
)
def main(
    symbol: str,
    timeframe: str,
    horizon: int,
    target_return: float,
    short_target_return: Optional[float],
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dims: str,
    dropout: float,
    history_days: Optional[int],
    max_samples: Optional[int],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    patience: int,
    min_delta: float,
    seed: int,
    device: str,
    feature_keep_prefixes: Optional[str],
    feature_drop_prefixes: Optional[str],
    return_feature: str,
    return_zscore_cut: Optional[float],
    volume_feature: str,
    volume_zscore_cut: Optional[float],
    scaler_kind: str,
    augmentation_noise: float,
    lr_scheduler: str,
    walk_forward_folds: int,
    walk_forward_valid_ratio: float,
    class_weight_neutral: float,
    class_weight_long: float,
    class_weight_short: float,
) -> None:
    """Train PatternNet with chronological splits and richer diagnostics."""
    total_ratio = train_ratio + valid_ratio + test_ratio
    if total_ratio > 1.0 + 1e-6:
        raise click.BadParameter("train + valid + test ratios must be <= 1.")
    if total_ratio < 1.0:
        train_ratio += 1.0 - total_ratio

    history_window = history_days or DEFAULT_HISTORY_DAYS.get(timeframe)
    ccxt_symbol = _to_ccxt_symbol(symbol)

    logger.info(
        "Preparing dataset",
        {
            "symbol": symbol,
            "ccxt_symbol": ccxt_symbol,
            "timeframe": timeframe,
            "history_days": history_window,
            "max_samples": max_samples,
        },
    )

    short_threshold = short_target_return if short_target_return is not None else target_return

    cfg = DatasetConfig(
        symbol=ccxt_symbol,
        timeframe=timeframe,
        prediction_horizon=horizon,
        target_return=target_return,
        target_return_long=target_return,
        target_return_short=short_threshold,
        max_history_days=history_window,
        max_samples=max_samples,
    )

    features_df, labels_df, feature_cols = load_feature_matrix(cfg)
    resolved_symbol = features_df.attrs.get("source_symbol", ccxt_symbol)
    logger.info("Resolved dataset symbol %s", resolved_symbol)

    keep_tokens = _parse_prefix_option(feature_keep_prefixes)
    drop_tokens = _parse_prefix_option(feature_drop_prefixes)
    if keep_tokens or drop_tokens:
        before_cols = len(features_df.columns)
        features_df = _filter_feature_columns(features_df, keep_tokens, drop_tokens)
        feature_cols = list(features_df.columns)
        logger.info(
            "Feature selection applied",
            {"before": before_cols, "after": len(feature_cols)},
        )

    features_df, labels_df, outlier_stats = _filter_outliers(
        features_df,
        labels_df,
        return_feature,
        return_zscore_cut,
        volume_feature,
        volume_zscore_cut,
    )
    feature_cols = list(features_df.columns)
    logger.info("Outlier filtering", outlier_stats)

    features = features_df.values.astype(np.float32)
    labels = labels_df.values.astype(np.float32)

    if len(features) < 512:
        raise RuntimeError(f"Insufficient samples for training ({len(features)} rows).")

    class_targets = _labels_to_classes(labels)
    class_labels = DEFAULT_CLASS_LABELS
    num_classes = len(class_labels)
    class_counts = np.bincount(class_targets, minlength=num_classes).astype(int)
    base_class_weight_tuple = (
        class_weight_neutral,
        class_weight_long,
        class_weight_short,
    )
    class_weight_multipliers = base_class_weight_tuple
    if all(abs(w - 1.0) < 1e-6 for w in class_weight_multipliers):
        class_weight_multipliers = None

    _set_seed(seed)
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    if device == "cuda" and torch_device.type != "cuda":
        logger.warning("CUDA requested but unavailable, using CPU.")

    train_slice, valid_slice, test_slice = _chronological_split(len(features), train_ratio, valid_ratio)

    train_X_raw = features[train_slice]
    train_y = class_targets[train_slice]
    valid_X_raw = features[valid_slice]
    valid_y = class_targets[valid_slice]
    test_X_raw = features[test_slice]
    test_y = class_targets[test_slice]

    scaler = _prepare_scaler(scaler_kind)
    train_X = scaler.fit_transform(train_X_raw)
    valid_X = scaler.transform(valid_X_raw) if len(valid_X_raw) else np.empty((0, train_X.shape[1]))
    test_X = scaler.transform(test_X_raw) if len(test_X_raw) else np.empty((0, train_X.shape[1]))

    train_X, train_y = _augment_training_data(train_X, train_y, augmentation_noise, seed)

    hidden_layers = tuple(int(x.strip()) for x in hidden_dims.split(",") if x.strip())
    params = TrainingParams(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_layers=hidden_layers or (128, 64),
        dropout=dropout,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
        device=torch_device,
        scheduler=lr_scheduler,
        class_weight_multipliers=class_weight_multipliers,
    )

    model, best_epoch, train_metrics, valid_metrics, test_metrics, best_metric_training = train_pattern_model(
        train_X,
        train_y,
        valid_X,
        valid_y,
        test_X,
        test_y,
        class_labels,
        params,
    )

    test_details = test_metrics.pop("details", None) if isinstance(test_metrics, dict) else None
    regime_metrics: Dict[str, Dict[str, float]] = {}
    if test_details and "vol_regime" in features_df.columns:
        regime_series = features_df.iloc[test_slice]["vol_regime"]
        regime_metrics = _compute_regime_metrics(test_details, regime_series, class_labels)

    walk_forward_results = _walk_forward_evaluation(
        features,
        class_targets,
        class_labels,
        params,
        scaler_kind,
        augmentation_noise,
        walk_forward_folds,
        walk_forward_valid_ratio,
    )

    symbol_dir = (MODEL_DIR / _symbol_key(symbol) / timeframe).resolve()
    symbol_dir.mkdir(parents=True, exist_ok=True)

    model_path = symbol_dir / "model.pt"
    scaler_path = symbol_dir / "scaler.pkl"
    meta_path = symbol_dir / "meta.json"

    torch.save(model.state_dict(), model_path)
    logger.info("Saved model", {"path": str(model_path)})

    import joblib

    joblib.dump(scaler, scaler_path)
    logger.info("Saved scaler", {"path": str(scaler_path)})

    class_weight_multipliers_meta = list(class_weight_multipliers or base_class_weight_tuple)

    preprocessing_info = {
        "feature_keep_prefixes": keep_tokens or [],
        "feature_drop_prefixes": drop_tokens or [],
        "return_feature": return_feature,
        "return_zscore_cut": return_zscore_cut,
        "volume_feature": volume_feature,
        "volume_zscore_cut": volume_zscore_cut,
        "outlier_stats": outlier_stats,
        "scaler": scaler_kind,
        "augmentation_noise": augmentation_noise,
        "class_weight_multipliers": class_weight_multipliers_meta,
    }

    class_distribution = {label: int(class_counts[idx]) for idx, label in enumerate(class_labels)}

    walk_forward_summary: Dict[str, Any] = {}
    if walk_forward_results:
        macro_values = [r["macro_f1"] for r in walk_forward_results if not np.isnan(r.get("macro_f1", np.nan))]
        acc_values = [r["accuracy"] for r in walk_forward_results if not np.isnan(r.get("accuracy", np.nan))]
        walk_forward_summary = {
            "folds": len(walk_forward_results),
            "average_macro_f1": float(np.mean(macro_values)) if macro_values else float("nan"),
            "average_accuracy": float(np.mean(acc_values)) if acc_values else float("nan"),
            "results": walk_forward_results,
        }

    best_metric = float(best_metric_training) if not np.isnan(best_metric_training) else float(
        valid_metrics.get("macro_f1", train_metrics.get("macro_f1", float("nan")))
        if valid_metrics
        else train_metrics.get("macro_f1", float("nan"))
    )

    meta = {
        "symbol": resolved_symbol,
        "symbol_key": _symbol_key(symbol),
        "timeframe": timeframe,
        "prediction_horizon": horizon,
        "target_return": target_return,
        "target_return_short": short_threshold,
        "features": feature_cols,
        "input_dim": train_X.shape[1],
        "hidden_dims": list(params.hidden_layers),
        "dropout": dropout,
        "train_samples": int(len(train_X)),
        "valid_samples": int(len(valid_X)),
        "test_samples": int(len(test_X)),
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "max_history_days": history_window,
        "max_samples": max_samples,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_valid_macro_f1": best_metric,
        "class_labels": class_labels,
        "class_distribution": class_distribution,
        "class_weight_multipliers": class_weight_multipliers_meta,
        "loss": "cross_entropy",
        "output_activation": "softmax",
        "preprocessing": preprocessing_info,
        "metrics": {
            "train": train_metrics,
            "valid": valid_metrics,
            "test": test_metrics,
            "regime": regime_metrics,
            "walk_forward": walk_forward_summary,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Saved metadata", {"path": str(meta_path)})

    click.echo("Entrenamiento completado")
    click.echo(f"  Mejor época: {best_epoch} | Macro-F1 valid: {best_metric:.4f}")
    click.echo(f"  Artefactos: {model_path}, {scaler_path}, {meta_path}")


if __name__ == "__main__":
    main()
