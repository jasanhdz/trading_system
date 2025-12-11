"""Strategy that gates entries using the neural probability model."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ..core.types import Candle, Signal, BotState
from ..core.ports.exchange import Exchange
from ..core.ports.logger import Logger
from .base import Strategy
from ml.nn_pattern.predictor import PatternPredictor
from ml.advanced_models.predictor import AdvancedPredictor
from ..core.utils.features import compute_features


def _candles_to_frame(candles: List[Candle]) -> pd.DataFrame:
    """Convert Candle objects to a pandas DataFrame suitable for features."""
    if not candles:
        raise ValueError("No candles received for ML strategy.")
    data = {
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
        "volume": [c.volume for c in candles],
    }
    index = pd.to_datetime([c.close_time for c in candles], unit="ms")
    return pd.DataFrame(data, index=index)


class MLProbabilityStrategy(Strategy):
    """Wraps neural predictions and only fires when probability confidence is high."""

    def __init__(self, history_bars: int = 512):
        super().__init__(name="ml_probability", timeframe="5m")
        self.history_bars = history_bars
        self.predictors: Dict[Tuple[str, str], PatternPredictor] = {}

    @staticmethod
    def _normalize_extras(extra_timeframes: Any) -> List[str]:
        if not extra_timeframes:
            return []
        if isinstance(extra_timeframes, str):
            tokens = [token.strip() for token in extra_timeframes.split(",")]
            return [token for token in tokens if token]
        return [token.strip() for token in extra_timeframes if isinstance(token, str) and token.strip()]

    def _resolve_timeframes(self, symbol_cfg: Any, config: Any) -> List[str]:
        """Return ordered list of timeframes (fast first) to evaluate."""
        timeframes: List[str] = []
        seen: set[str] = set()

        def add(tf: Optional[str]) -> None:
            if not tf:
                return
            norm = tf.strip()
            if not norm:
                return
            key = norm.lower()
            if key not in seen:
                seen.add(key)
                timeframes.append(norm)

        add(getattr(symbol_cfg, "timeframe", "5m"))
        extras = self._normalize_extras(getattr(config, "ML_EXTRA_TIMEFRAMES", None))
        if not extras:
            extras = ["15m"]
        for tf in extras:
            add(tf)
        return timeframes

    def _artifact_paths(
        self,
        symbol: str,
        timeframe: str,
        symbol_cfg: Any,
        config: Any,
    ) -> Tuple[Path, Path, Path]:
        """Resolve model/scaler/meta paths for the requested timeframe."""
        if timeframe == getattr(symbol_cfg, "timeframe", None):
            return (
                Path(symbol_cfg.model_path),
                Path(symbol_cfg.scaler_path),
                Path(symbol_cfg.meta_path),
            )

        models_root = Path(getattr(config, "ML_MODELS_ROOT", "models/trained"))
        model_filename = getattr(config, "ML_MODEL_FILENAME", None) or "model.pt"
        scaler_filename = getattr(config, "ML_SCALER_FILENAME", None) or "scaler.pkl"
        meta_filename = getattr(config, "ML_META_FILENAME", None) or "meta.json"

        base_dir = (models_root / symbol / timeframe).resolve()
        return (
            base_dir / model_filename,
            base_dir / scaler_filename,
            base_dir / meta_filename,
        )

    def _ensure_predictor(
        self,
        symbol: str,
        timeframe: str,
        config: Any,
        logger: Logger,
    ) -> PatternPredictor:
        key = (symbol, timeframe)
        if key in self.predictors:
            return self.predictors[key]

        if not hasattr(config, "get_symbol_config"):
            raise AttributeError("Config object must expose get_symbol_config for multi-symbol ML.")

        symbol_cfg = config.get_symbol_config(symbol)
        model_path, scaler_path, meta_path = self._artifact_paths(symbol, timeframe, symbol_cfg, config)

        # Support for directory-based models (ensembles of folds)
        # If model.pt doesn't exist, but we have fold models in the directory, use the directory as model_path
        if not model_path.exists() and model_path.parent.exists():
            if list(model_path.parent.glob("best_model_fold*.pt")):
                model_path = model_path.parent

        if not model_path.exists() or not scaler_path.exists() or not meta_path.exists():
            missing = {
                "model": model_path.exists(),
                "scaler": scaler_path.exists(),
                "meta": meta_path.exists(),
            }
            raise FileNotFoundError(f"Missing artifacts for {symbol} [{timeframe}]: {missing}")

        logger.info(
            "loading_ml_model",
            {
                "symbol": symbol,
                "model": model_path.name,
                "timeframe": timeframe,
            },
        )

        predictor: PatternPredictor | AdvancedPredictor

        # Decide which predictor implementation to use based on metadata schema
        try:
            meta_data = json.loads(meta_path.read_text())
        except Exception:
            meta_data = {}

        is_advanced = isinstance(meta_data, dict) and {
            "sequence_length",
            "model_config",
            "selected_features",
        }.issubset(meta_data.keys())

        device = getattr(config, "ML_MODEL_DEVICE", "cpu")

        if is_advanced:
            predictor = AdvancedPredictor(
                model_path=model_path,
                scaler_path=scaler_path,
                meta_path=meta_path,
                device=device,
            )
        else:
            predictor = PatternPredictor(
                model_path=model_path,
                scaler_path=scaler_path,
                meta_path=meta_path,
                device=device,
            )
        self.predictors[key] = predictor
        return predictor

    async def evaluate(
        self,
        symbol: str,
        exchange: Exchange,
        config: Any,
        state: BotState | None,
        now: int,
        logger: Logger,
    ) -> Signal:
        """Fetch recent candles, run inference, and gate decisions."""
        symbol_cfg = config.get_symbol_config(symbol)
        timeframes = self._resolve_timeframes(symbol_cfg, config)
        fast_timeframe = timeframes[0]
        self.timeframe = fast_timeframe

        limit = max(self.history_bars, getattr(config, "ML_HISTORY_BARS", self.history_bars))
        min_multi_bars = min(limit, max(getattr(config, "ML_MULTI_TF_MIN_BARS", 256), 64))

        candles_by_tf: Dict[str, List[Candle]] = {}
        probs_by_tf: Dict[str, Dict[str, float]] = {}

        for timeframe in timeframes:
            try:
                predictor = self._ensure_predictor(symbol, timeframe, config, logger)
            except Exception as exc:
                logger.error(
                    "ml_model_load_fail",
                    {"err": str(exc), "symbol": symbol, "timeframe": timeframe},
                )
                return {"action": "IDLE", "reason": f"ml_model_missing_{timeframe}"}

            candles = await exchange.get_candles(symbol, timeframe, limit)
            if len(candles) < min_multi_bars:
                return {"action": "IDLE", "reason": f"few_candles_{timeframe}"}

            candles_by_tf[timeframe] = candles

            try:
                frame = _candles_to_frame(candles)
                probs = predictor.predict(frame)
            except Exception as exc:
                logger.warn(
                    "ml_prediction_fail",
                    {"err": str(exc), "symbol": symbol, "timeframe": timeframe},
                )
                return {"action": "IDLE", "reason": f"ml_prediction_fail_{timeframe}"}

            probs_by_tf[timeframe] = {
                "long": float(probs["long"]),
                "short": float(probs["short"]),
            }

        logger.debug("ml_probs_multi", {"symbol": symbol, "probs": probs_by_tf})

        margin = getattr(config, "ML_MARGIN", 0.12)
        long_threshold = getattr(config, "ML_THRESHOLD_LONG", 0.50)
        short_threshold = getattr(config, "ML_THRESHOLD_SHORT", 0.50)

        confirm_long_threshold = getattr(
            config,
            "ML_CONFIRM_THRESHOLD_LONG",
            max(long_threshold - 0.05, 0.55),
        )
        confirm_short_threshold = getattr(
            config,
            "ML_CONFIRM_THRESHOLD_SHORT",
            max(short_threshold - 0.05, 0.55),
        )
        confirm_margin = getattr(
            config,
            "ML_CONFIRM_MARGIN",
            max(margin * 0.5, 0.05),
        )
        require_confirmation = bool(timeframes[1:]) and getattr(
            config, "ML_REQUIRE_CONFIRMATION", True
        )

        fast_candles = candles_by_tf[fast_timeframe]
        feature_lb = min(len(fast_candles), max(getattr(config, "ML_FILTER_LOOKBACK", 60), 20))
        try:
            features = compute_features(fast_candles[-feature_lb:], lookback=feature_lb)
        except ValueError:
            features = {}

        close_price = float(features.get("close", fast_candles[-1].close))
        ema_base = float(features.get("ema_21", close_price))
        atr_val = float(features.get("atr_14", 0.0))
        rsi_val = float(features.get("rsi_14", 50.0))
        max_ext_pct = getattr(config, "ML_MAX_EXT_PCT", 0.015)
        max_rsi = getattr(config, "ML_MAX_RSI", 68.0)
        min_rsi = getattr(config, "ML_MIN_RSI", 32.0)
        max_body_atr = getattr(config, "ML_MAX_BODY_ATR", 2.5)

        last_candle = fast_candles[-1]
        body = abs(last_candle.close - last_candle.open)

        filter_reason_long = None
        filter_reason_short = None

        if ema_base and ema_base > 0 and math.isfinite(ema_base) and math.isfinite(close_price):
            ext_long = (close_price - ema_base) / ema_base
            ext_short = (ema_base - close_price) / ema_base
            if ext_long > max_ext_pct:
                filter_reason_long = f"ml_filter_ext_long={ext_long:.3f}"
            if ext_short > max_ext_pct:
                filter_reason_short = f"ml_filter_ext_short={ext_short:.3f}"

        if math.isfinite(rsi_val):
            if rsi_val > max_rsi:
                filter_reason_long = filter_reason_long or f"ml_filter_rsi_high={rsi_val:.1f}"
            if rsi_val < min_rsi:
                filter_reason_short = filter_reason_short or f"ml_filter_rsi_low={rsi_val:.1f}"

        if atr_val and atr_val > 0 and math.isfinite(atr_val):
            body_ratio = body / atr_val
            if body_ratio > max_body_atr:
                if last_candle.close >= last_candle.open:
                    filter_reason_long = filter_reason_long or f"ml_filter_body_ratio={body_ratio:.2f}"
                if last_candle.close <= last_candle.open:
                    filter_reason_short = filter_reason_short or f"ml_filter_body_ratio={body_ratio:.2f}"

        def is_long_ready(probs: Dict[str, float], threshold: float, min_margin: float) -> bool:
            return probs["long"] >= threshold and (probs["long"] - probs["short"]) >= min_margin

        def is_short_ready(probs: Dict[str, float], threshold: float, min_margin: float) -> bool:
            return probs["short"] >= threshold and (probs["short"] - probs["long"]) >= min_margin

        fast_probs = probs_by_tf[fast_timeframe]
        fast_long_ready = is_long_ready(fast_probs, long_threshold, margin)
        fast_short_ready = is_short_ready(fast_probs, short_threshold, margin)

        confirm_long_tfs: List[str] = []
        confirm_short_tfs: List[str] = []
        oppose_long_tfs: List[str] = []
        oppose_short_tfs: List[str] = []

        for timeframe in timeframes[1:]:
            tf_probs = probs_by_tf[timeframe]
            if is_long_ready(tf_probs, confirm_long_threshold, confirm_margin):
                confirm_long_tfs.append(timeframe)
                oppose_short_tfs.append(timeframe)
            if is_short_ready(tf_probs, confirm_short_threshold, confirm_margin):
                confirm_short_tfs.append(timeframe)
                oppose_long_tfs.append(timeframe)

        if (
            getattr(config, "ALLOW_LONGS", True)
            and fast_long_ready
        ):
            if oppose_long_tfs:
                logger.debug(
                    "ml_pre_filter_block",
                    {"symbol": symbol, "reason": "oppose_long", "timeframes": oppose_long_tfs},
                )
                return {
                    "action": "IDLE",
                    "reason": f"ml_block_long={','.join(oppose_long_tfs)}",
                }
            if require_confirmation and not confirm_long_tfs:
                return {
                    "action": "IDLE",
                    "reason": "ml_wait_long_confirmation",
                }
            if filter_reason_long:
                logger.debug("ml_pre_filter_block", {"symbol": symbol, "reason": filter_reason_long})
                return {"action": "IDLE", "reason": filter_reason_long}
            confirm_str = ",".join(confirm_long_tfs) if confirm_long_tfs else ""
            gap = fast_probs["long"] - fast_probs["short"]
            return {
                "action": "ENTER_LONG",
                "reason": (
                    f"ml_long {fast_timeframe}={fast_probs['long']:.2f} Δ={gap:.2f}"
                    + (f" confirm={confirm_str}" if confirm_str else "")
                    + f" short={fast_probs['short']:.2f}"
                ),
            }

        if (
            getattr(config, "ALLOW_SHORTS", True)
            and fast_short_ready
        ):
            if oppose_short_tfs:
                logger.debug(
                    "ml_pre_filter_block",
                    {"symbol": symbol, "reason": "oppose_short", "timeframes": oppose_short_tfs},
                )
                return {
                    "action": "IDLE",
                    "reason": f"ml_block_short={','.join(oppose_short_tfs)}",
                }
            if require_confirmation and not confirm_short_tfs:
                return {
                    "action": "IDLE",
                    "reason": "ml_wait_short_confirmation",
                }
            if filter_reason_short:
                logger.debug("ml_pre_filter_block", {"symbol": symbol, "reason": filter_reason_short})
                return {"action": "IDLE", "reason": filter_reason_short}
            confirm_str = ",".join(confirm_short_tfs) if confirm_short_tfs else ""
            gap = fast_probs["short"] - fast_probs["long"]
            return {
                "action": "ENTER_SHORT",
                "reason": (
                    f"ml_short {fast_timeframe}={fast_probs['short']:.2f} Δ={gap:.2f}"
                    + (f" confirm={confirm_str}" if confirm_str else "")
                    + f" long={fast_probs['long']:.2f}"
                ),
            }

        summary = "; ".join(
            f"{tf}=L{probs['long']:.2f}/S{probs['short']:.2f}"
            for tf, probs in ((tf, probs_by_tf[tf]) for tf in timeframes)
        )
        return {
            "action": "IDLE",
            "reason": f"ml_idle {summary}",
        }
