"""Prediction helper that loads the trained neural model and scaler."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch

from .features import build_feature_frame
from .model import PatternNet


class PatternPredictor:
    """Loads model artifacts and produces probability scores for new candles."""

    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        meta_path: Path,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.meta_path = Path(meta_path)
        self.device = torch.device(device)

        if not self.model_path.exists() or not self.scaler_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                "Missing model artifacts. "
                f"Expected {self.model_path}, {self.scaler_path}, {self.meta_path}"
            )

        self.meta = json.loads(self.meta_path.read_text())
        self.scaler = joblib.load(self.scaler_path)

        self.class_labels = self.meta.get("class_labels", ["long", "short"])
        self.output_activation = self.meta.get("output_activation", "sigmoid")
        self.loss_type = self.meta.get("loss", "bce")

        self.model = PatternNet(
            input_dim=self.meta["input_dim"],
            hidden_dims=self.meta["hidden_dims"],
            dropout=self.meta["dropout"],
            num_classes=len(self.class_labels),
            output_activation=self.output_activation,
        )
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.features = self.meta["features"]

    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute long/short probabilities from a fresh chunk of OHLCV data.

        Args:
            df: DataFrame with OHLCV columns, already sorted ascending.
        """
        feature_frame, _ = build_feature_frame(df)
        if feature_frame.empty:
            raise ValueError("Not enough rows to compute features for prediction.")

        latest_vector = feature_frame.iloc[-1][self.features].values.astype(np.float32)
        scaled = self.scaler.transform(latest_vector.reshape(1, -1))
        tensor = torch.from_numpy(scaled).to(self.device)

        prob_tensor = self.model.predict_proba(tensor)
        probs = prob_tensor.detach().cpu().numpy().flatten()

        prob_map = {
            label.lower(): float(probs[idx]) for idx, label in enumerate(self.class_labels)
        }

        result = {
            "long": prob_map.get("long", 0.0),
            "short": prob_map.get("short", 0.0),
        }
        if "neutral" in prob_map:
            result["neutral"] = prob_map["neutral"]
        return result
