"""Aegis model training and retraining entry points."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep the CLIs runnable both as modules and as direct script files.
_TRAINING_DIR = str(Path(__file__).resolve().parent)
if _TRAINING_DIR not in sys.path:
    sys.path.insert(0, _TRAINING_DIR)
