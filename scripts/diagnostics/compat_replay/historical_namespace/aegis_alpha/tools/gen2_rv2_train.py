"""DIAGNOSTIC HISTORICAL COMPATIBILITY ONLY
NOT FOR PRODUCTION IMPORT
SOURCE_BRANCH: feature/wraith-phantom-v8
SOURCE_COMMIT: 4984a0473e6080f07181f7450b5b98a3aa454637
SOURCE_PATH: aegis_alpha/tools/gen2_rv2_train.py
SOURCE_SHA256: 5ddc49e7ad69b0fa38e7e928ae4128703f6d293fbce7cdadaa96fac381ac99b6
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MedianImputer:
    """Exact historical state/transform contract required by frozen pickles."""

    def fit(self, x: pd.DataFrame) -> "MedianImputer":
        self.medians = x.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.replace([np.inf, -np.inf], np.nan).fillna(self.medians).fillna(0.0)
