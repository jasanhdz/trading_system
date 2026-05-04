#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.signals.evaluate_horizon_agreement import main  # noqa: E402


if __name__ == "__main__":
    main()
