from __future__ import annotations

from dataclasses import dataclass

import numpy as np


IDLE = 0
LONG = 1
SHORT = 2
CLOSE = 3


@dataclass(frozen=True)
class ActionMaskConfig:
    min_hold_steps: int = 6
    min_flat_steps: int = 12
    allow_flips: bool = False


def valid_action_mask(position: float, hold_steps: int, flat_steps: int, cfg: ActionMaskConfig) -> np.ndarray:
    mask = np.zeros(4, dtype=bool)
    mask[IDLE] = True
    if position == 0:
        can_enter = flat_steps >= cfg.min_flat_steps
        mask[LONG] = can_enter
        mask[SHORT] = can_enter
        mask[CLOSE] = False
    else:
        mask[CLOSE] = hold_steps >= cfg.min_hold_steps
        if cfg.allow_flips:
            mask[LONG] = position < 0 and hold_steps >= cfg.min_hold_steps
            mask[SHORT] = position > 0 and hold_steps >= cfg.min_hold_steps
    return mask


def coerce_action(action: int, position: float, hold_steps: int, flat_steps: int, cfg: ActionMaskConfig) -> tuple[int, bool]:
    mask = valid_action_mask(position, hold_steps, flat_steps, cfg)
    if 0 <= action < len(mask) and mask[action]:
        return action, False
    return IDLE, True
