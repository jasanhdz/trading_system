from pathlib import Path

import numpy as np
import pandas as pd

from aegis.research.recent_short_w14 import _economic_metrics, _rsi, _volume_trade_return


def test_rsi_and_economic_metrics_are_finite():
    rsi = _rsi(pd.Series(np.arange(30, dtype=float)), 6)
    assert np.isfinite(rsi).all()
    metrics = _economic_metrics(np.array([30.0, -10.0, 20.0]), np.array([True, False, True]), 14.0)
    assert metrics["trades"] == 2
    assert metrics["net_bps_per_signal"] == (16.0 + 6.0) / 3.0


def test_volume_trade_return_uses_next_open_and_direction():
    frame = pd.DataFrame({"open": [100, 100, 99, 98], "close": [100, 99, 98, 97]})
    short = _volume_trade_return(frame, 0, -1, "fixed_15m")
    assert short > 0
