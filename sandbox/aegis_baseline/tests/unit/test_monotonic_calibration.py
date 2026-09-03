import numpy as np
import pytest

from aegis.research.monotonic_calibration import (
    MonotonicCalibrationError,
    MonotonicPlattCalibrator,
)


def test_monotonic_platt_preserves_score_order():
    scores = np.linspace(-3.0, 3.0, 200)
    labels = (scores + np.sin(scores * 2.0) * 0.2 > 0.0).astype(int)
    calibrator = MonotonicPlattCalibrator.fit(scores, labels)
    probabilities = calibrator.predict(scores)
    assert calibrator.slope > 0.0
    assert np.all(np.diff(probabilities) >= 0.0)


def test_monotonic_platt_fails_instead_of_inverting_an_inverse_calibration_sample():
    scores = np.linspace(-3.0, 3.0, 200)
    labels = (scores < 0.0).astype(int)
    with pytest.raises(
        MonotonicCalibrationError,
        match="erase or invert ranking",
    ):
        MonotonicPlattCalibrator.fit(scores, labels)


def test_monotonic_platt_rejects_non_finite_evidence():
    with pytest.raises(MonotonicCalibrationError):
        MonotonicPlattCalibrator.fit([0.0, float("nan")] * 10, [0, 1] * 10)
