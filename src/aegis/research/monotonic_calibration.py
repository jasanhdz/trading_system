"""Monotonic probability calibration primitives for preregistered research."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class MonotonicCalibrationError(ValueError):
    """Raised when monotonic calibration cannot be established."""


@dataclass(frozen=True)
class MonotonicPlattCalibrator:
    slope: float
    intercept: float

    @classmethod
    def fit(
        cls, scores: Sequence[float], labels: Sequence[int]
    ) -> "MonotonicPlattCalibrator":
        x = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if (
            x.ndim != 1
            or y.shape != x.shape
            or len(x) < 20
            or not np.isfinite(x).all()
            or not np.isin(y, [0.0, 1.0]).all()
            or len(np.unique(y)) != 2
        ):
            raise MonotonicCalibrationError("invalid monotonic calibration evidence")

        prevalence = float(np.mean(y))
        initial_intercept = math.log(prevalence / (1.0 - prevalence))

        def loss(parameters: np.ndarray) -> float:
            slope, intercept = parameters
            logits = slope * x + intercept
            return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

        fitted = minimize(
            loss,
            x0=np.asarray([1.0, initial_intercept]),
            method="L-BFGS-B",
            bounds=((0.0, None), (None, None)),
        )
        if not fitted.success or not np.isfinite(fitted.x).all():
            raise MonotonicCalibrationError("monotonic calibration fit failed")
        slope, intercept = (float(value) for value in fitted.x)
        if slope <= 1e-8:
            raise MonotonicCalibrationError("calibration evidence would erase or invert ranking")
        return cls(slope=slope, intercept=intercept)

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise MonotonicCalibrationError("invalid calibration scores")
        result = expit(self.slope * values + self.intercept)
        if not np.isfinite(result).all():
            raise MonotonicCalibrationError("non-finite calibrated probabilities")
        return result
