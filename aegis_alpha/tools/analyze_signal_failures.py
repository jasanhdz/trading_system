#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from aegis_alpha.signals.analyze_signal_failures import main  # noqa: E402


if __name__ == "__main__":
    main()
