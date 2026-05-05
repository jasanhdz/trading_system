from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aegis_alpha.config import AegisConfig, REPO_ROOT


@dataclass
class ModelPrediction:
    raw_action: str
    probs: dict[str, float]
    model_loaded: bool
    reason: str


class AegisModelLoader:
    """Minimal model wrapper. It is defensive until a champion exists."""

    def __init__(self, cfg: AegisConfig):
        self.cfg = cfg
        self.model = None
        self.model_path = REPO_ROOT / cfg.model.champion_path
        self._load_attempted = False

    def reload(self) -> bool:
        self._load_attempted = True
        if not Path(self.model_path).exists():
            self.model = None
            return False
        try:
            from stable_baselines3 import PPO

            self.model = PPO.load(str(self.model_path), device="cpu")
            return True
        except Exception:
            self.model = None
            return False

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, market_window: np.ndarray | None = None, account: np.ndarray | None = None) -> ModelPrediction:
        # Do not auto-load PPO in request paths. The inference API must remain
        # responsive even when champion artifacts are absent, huge, or invalid.
        if self.model is None or market_window is None or account is None:
            return ModelPrediction(
                raw_action="IDLE",
                probs={"idle": 0.99, "long": 0.005, "short": 0.005, "close": 0.0},
                model_loaded=False,
                reason="aegis_champion_not_loaded_lazy_safe_mode",
            )

        obs = {
            "market": market_window[None, :, :].astype(np.float32),
            "account": account[None, :].astype(np.float32),
        }
        action, _ = self.model.predict(obs, deterministic=True)
        action_id = int(action[0] if hasattr(action, "__len__") else action)
        names = {0: "IDLE", 1: "LONG", 2: "SHORT", 3: "CLOSE"}
        raw_action = names.get(action_id, "IDLE")
        probs = {"idle": 0.25, "long": 0.25, "short": 0.25, "close": 0.25}
        probs[raw_action.lower()] = 0.70
        return ModelPrediction(raw_action=raw_action, probs=probs, model_loaded=True, reason="model_prediction")
