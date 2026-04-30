#!/usr/bin/env python3
"""Compatibility entrypoint for the PM2 watchdog process."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.phantom_v30.utils.watchdog_trainer import main


if __name__ == "__main__":
    main()
