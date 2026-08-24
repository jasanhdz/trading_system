from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .models import Candle5m

FROZEN_HEAD = "bb034431e0ce05c8e0f978453c46dcff6efb981c"
FROZEN_HASHES = {
    "RegimeEngineV2.ts": "3726e28badfdba5acc81d87ccd3202fc43310a04d4b3cff2597f38acb2913134",
    "RegimeEngineV2.types.ts": "3b3972153f7c977d50ec864a5d8a4c4b3d8d2e73822453eaed1a25391211d10c",
    "RegimeEngineV2.test.ts": "80aa2619efdcb74fa8722f79ce62a01a4028bb213856f3a5b7fc0ea32e091cf1",
}


class RegimeParityError(RuntimeError):
    pass


class TypeScriptRegimeEvaluator:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.child = self.repo_root / "binance-futures-bot-ts"
        self.bridge = self.repo_root / "sandbox/aegis_range_strategy_v1/scripts/regime_v2_bridge.cjs"
        source = self.child / "src/domain/services/regime-v2"
        for name, expected in FROZEN_HASHES.items():
            actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
            if actual != expected:
                raise RegimeParityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_PARITY")

    def evaluate(self, *, symbol: str, candles: tuple[Candle5m, ...], timeframe: str) -> dict[str, Any]:
        if timeframe != "5m" or len(candles) != 160:
            raise RegimeParityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_PARITY")
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": [
                {
                    "timestamp": int(candle.open_time.timestamp() * 1000),
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
                for candle in candles
            ],
        }
        process = subprocess.run(
            ["node", "-r", "ts-node/register", str(self.bridge)],
            cwd=self.child,
            input=json.dumps(payload, separators=(",", ":")),
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        if process.returncode != 0 or process.stderr or not process.stdout:
            raise RegimeParityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_PARITY")
        try:
            return json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RegimeParityError("AEGIS_RANGE_R2_DATA_READINESS_BLOCKED_BY_PARITY") from exc
