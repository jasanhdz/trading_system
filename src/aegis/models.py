"""Approved model-bundle runtime boundary.

TODO migration reference: study the bundle and feature compatibility contracts
in ``feature/wraith-phantom-v8:aegis_alpha/tools/gen2_rv2_train.py`` without
copying its estimator or training implementation into the online runtime.
"""

from typing import Protocol

from .domain import FeatureBatch, ModelPredictions


class ModelRuntime(Protocol):
    def predict(self, features: FeatureBatch) -> ModelPredictions:
        """TODO: run an immutable approved bundle and expose all predictions."""
        ...
