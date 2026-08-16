import numpy as np
import pandas as pd

from aegis.research.recent_short_signal_gate_w15 import _metrics


def test_metrics_preserve_original_signal_denominator():
    frame = pd.DataFrame({"net_bps": [10.0, -5.0, 20.0], "gross_bps": [20.0, 5.0, 30.0]})
    metrics = _metrics(frame, np.array([True, False, True]), 4.0)
    assert metrics["trades"] == 2
    assert metrics["net_bps_per_trade"] == 11.0
    assert metrics["net_bps_per_signal"] == 22.0 / 3.0
