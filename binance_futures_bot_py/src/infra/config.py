"""Configuration module for the trading bot."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


def _upper_symbol(value: Optional[str]) -> str:
    return (value or "").strip().upper()


def _parse_share(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    cleaned = value.replace("%", "").strip()
    if not cleaned:
        return None
    try:
        num = float(cleaned)
    except ValueError:
        return None
    if num > 1:
        num /= 100.0
    if num < 0:
        return None
    return min(max(num, 0.0), 1.0)


def _parse_positive_number(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        num = float(cleaned)
    except ValueError:
        return None
    if num <= 0:
        return None
    return num


@dataclass
class SymbolDescriptor:
    """User supplied hints for a symbol entry."""

    symbol: str
    leverage: Optional[int]
    capital_usage: Optional[float]


@dataclass
class SymbolSettings:
    """Resolved runtime configuration for a single symbol."""

    symbol: str
    leverage: int
    capital_usage_pct: float
    timeframe: str
    model_path: Path
    scaler_path: Path
    meta_path: Path


class Config:
    """Trading bot configuration with multi-symbol support."""

    def __init__(self) -> None:
        # Project roots
        self.PROJECT_ROOT = Path(__file__).resolve().parents[3]
        self.BOT_ROOT = Path(__file__).resolve().parents[2]

        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
        self.LOG_PRETTY_SIGNALS = os.getenv("LOG_PRETTY_SIGNALS", "1") == "1"
        self.LOG_FILE = os.getenv("LOG_FILE", "")

        # Credentials / endpoints
        self.API_KEY = os.getenv("BINANCE_API_KEY", "")
        self.API_SECRET = os.getenv("BINANCE_API_SECRET", "")
        self.IS_TESTNET = os.getenv("IS_TESTNET") == "1"
        self.HTTP_FUTURES = (
            "https://testnet.binancefuture.com"
            if self.IS_TESTNET
            else "https://fapi.binance.com"
        )
        self.WS_FUTURES = (
            "wss://fstream.binancefuture.com"
            if self.IS_TESTNET
            else "wss://fstream.binance.com"
        )

        # Bot timing
        self.BOT_INTERVAL_SEC = int(os.getenv("BOT_INTERVAL_SEC", "5"))
        self.BOT_STAGGER_MS = int(os.getenv("BOT_STAGGER_MS", "2000"))

        # Global sizing defaults (symbol entries can override)
        base_symbol = _upper_symbol(os.getenv("SYMBOL", "XRPUSDT"))
        base_leverage = int(round(float(os.getenv("LEVERAGE", "50"))))
        base_capital_usage = float(os.getenv("CAPITAL_USAGE_PCT", "0.85"))
        self.MIN_WALLET_RESERVE_USDT = float(os.getenv("MIN_WALLET_RESERVE_USDT", "0.2"))
        self.FEE_BUFFER_PCT = float(os.getenv("FEE_BUFFER_PCT", "0.0015"))

        # Entry / study parameters
        self.TP_ROE = float(os.getenv("TP_ROE", "1.0"))
        self.ENTRY_TIMEFRAME = os.getenv("ENTRY_TIMEFRAME", "5m")
        self.VOL_AVG_LEN = int(os.getenv("VOL_AVG_LEN", "20"))
        self.VOL_FACTOR_ENTRY = float(os.getenv("VOL_FACTOR_ENTRY", "1.1"))
        self.GREEN_STREAK_MIN = int(os.getenv("GREEN_STREAK_MIN", "3"))
        self.RED_STREAK_MIN = int(os.getenv("RED_STREAK_MIN", "3"))
        self.ENTRY_EMA_PERIOD = int(os.getenv("ENTRY_EMA_PERIOD", "20"))
        self.ENTRY_MAX_EMA_EXTENSION = float(os.getenv("ENTRY_MAX_EMA_EXTENSION", "9"))

        # ML gate + re-entries
        self.ML_THRESHOLD = float(os.getenv("ML_THRESHOLD", "0.7"))
        self.REENTER_ON_TP = os.getenv("REENTER_ON_TP", "1") == "1"
        self.REENTER_COOLDOWN_MS = int(os.getenv("REENTER_COOLDOWN_MS", "5000"))
        self.VOL_FACTOR_REENTER = float(os.getenv("VOL_FACTOR_REENTER", "1.5"))
        self.GREEN_STREAK_REENTER_MIN = int(os.getenv("GREEN_STREAK_REENTER_MIN", "2"))
        self.RED_STREAK_REENTER_MIN = int(os.getenv("RED_STREAK_REENTER_MIN", "2"))

        # Stop / guard defaults
        self.SL_TICKS_ABOVE_LIQ_MAP: Dict[str, int] = {
            "XRPUSDT": 69,
            "ETHUSDT": 8,
            "BTCUSDT": 50,
        }
        self.SL_TICKS_ABOVE_LIQ_DEFAULT = int(os.getenv("SL_TICKS_ABOVE_LIQ_DEFAULT", "69"))
        self.PROFIT_LOCK_BE_AT_ROE = float(os.getenv("PROFIT_LOCK_BE_AT_ROE", "0.2"))
        self.PROFIT_GIVEBACK_ARM_ROE = float(os.getenv("PROFIT_GIVEBACK_ARM_ROE", "0.4"))
        self.PROFIT_GIVEBACK_DROP_REL = float(os.getenv("PROFIT_GIVEBACK_DROP_REL", "0.3"))
        self.PROFIT_GIVEBACK_DROP_MIN = float(os.getenv("PROFIT_GIVEBACK_DROP_MIN", "0.1"))
        self.EARLY_FAIL_WINDOW_MS = int(os.getenv("EARLY_FAIL_WINDOW_MS", str(12 * 60_000)))
        self.EARLY_FAIL_VOL_FACTOR = float(os.getenv("EARLY_FAIL_VOL_FACTOR", "1.5"))
        self.SHARP_BODY_PCT = float(os.getenv("SHARP_BODY_PCT", "0.6"))

        # Pyramid / trailing (optional)
        self.PYRAMID_MAX_UNITS = int(os.getenv("PYRAMID_MAX_UNITS", "3"))
        self.PYRAMID_STEP_ATR = float(os.getenv("PYRAMID_STEP_ATR", "0.5"))
        self.PYRAMID_UNIT_PCT_OF_ENTRY = float(os.getenv("PYRAMID_UNIT_PCT_OF_ENTRY", "0.5"))
        self.ATR_LEN = int(os.getenv("ATR_LEN", "14"))
        self.TRAIL_ATR_MULT_BASE = float(os.getenv("TRAIL_ATR_MULT_BASE", "2.5"))
        self.TRAIL_ATR_MULT_MIN = float(os.getenv("TRAIL_ATR_MULT_MIN", "1.2"))
        self.TRAIL_ATR_STEP_ROE = float(os.getenv("TRAIL_ATR_STEP_ROE", "0.5"))
        self.STOP_MIN_IMPROVE_TICKS = int(os.getenv("STOP_MIN_IMPROVE_TICKS", "2"))
        self.TIME_STOP_MINUTES = int(os.getenv("TIME_STOP_MINUTES", "0"))
        self.TIME_STOP_MIN_ROE = float(os.getenv("TIME_STOP_MIN_ROE", "0.05"))
        self.TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "2.5"))
        self.TRAIL_THROTTLE_MS = int(os.getenv("TRAIL_THROTTLE_MS", "15000"))
        self.MAX_RISK_PCT = float(os.getenv("MAX_RISK_PCT", "0.0"))

        # Trend filters
        self.TREND_TIMEFRAMES = os.getenv("TREND_TIMEFRAMES", "5m,15m").split(",")
        self.EMA_FAST = int(os.getenv("EMA_FAST", "7"))
        self.EMA_MID = int(os.getenv("EMA_MID", "25"))
        self.EMA_SLOW = int(os.getenv("EMA_SLOW", "99"))
        self.ADX_LEN = int(os.getenv("ADX_LEN", "14"))
        self.ADX_MIN = int(os.getenv("ADX_MIN", "20"))
        self.MAX_EXT_FROM_EMA_FAST = float(os.getenv("MAX_EXT_FROM_EMA_FAST", "0.015"))
        self.NO_TRADE_BAND_AROUND_EMA_SLOW = float(os.getenv("NO_TRADE_BAND_AROUND_EMA_SLOW", "0.003"))

        # ML thresholds
        self.ML_MIN_PROB = float(os.getenv("ML_MIN_PROB", "0.7"))
        self.ML_MARGIN = float(os.getenv("ML_MARGIN", "0.15"))
        self.ALLOW_LONGS = os.getenv("ALLOW_LONGS", "1").lower() not in {"0", "false"}
        self.ALLOW_SHORTS = os.getenv("ALLOW_SHORTS", "1").lower() not in {"0", "false"}
        self.ML_THRESHOLD_LONG = float(os.getenv("ML_THRESHOLD_LONG", "0.65"))
        self.ML_THRESHOLD_SHORT = float(os.getenv("ML_THRESHOLD_SHORT", "0.7"))
        self.ADX_MIN_FOR_SHORT = int(os.getenv("ADX_MIN_FOR_SHORT", "25"))
        self.REQUIRE_BEAR_MA_FOR_SHORT = os.getenv("REQUIRE_BEAR_MA_FOR_SHORT", "1") == "1"

        # Intelligent take-profit guard
        self.INT_TP_MIN_ROE = float(os.getenv("INT_TP_MIN_ROE", "0.20"))
        self.INT_TP_TRAIL_DROP = float(os.getenv("INT_TP_TRAIL_DROP", "0.35"))
        self.INT_TP_TREND_ADX = float(os.getenv("INT_TP_TREND_ADX", "18"))
        self.INT_TP_LOOKBACK = int(os.getenv("INT_TP_LOOKBACK", "40"))
        self.INT_TP_COOLDOWN_MS = int(os.getenv("INT_TP_COOLDOWN_MS", "15000"))
        self.INT_TP_PANIC_DROP = float(os.getenv("INT_TP_PANIC_DROP", "0.8"))
        default_hold_threshold = max(self.ML_THRESHOLD_LONG, self.ML_THRESHOLD_SHORT, 0.5)
        self.INT_TP_ML_HOLD_THRESHOLD = float(
            os.getenv("INT_TP_ML_HOLD_THRESHOLD", str(default_hold_threshold))
        )
        self.INT_TP_ML_HOLD_MARGIN = float(
            os.getenv("INT_TP_ML_HOLD_MARGIN", str(self.ML_MARGIN))
        )
        exit_threshold_default = max(0.55, self.INT_TP_ML_HOLD_THRESHOLD + 0.05)
        self.INT_TP_ML_EXIT_THRESHOLD = float(
            os.getenv("INT_TP_ML_EXIT_THRESHOLD", str(exit_threshold_default))
        )
        exit_margin_default = max(0.05, self.INT_TP_ML_HOLD_MARGIN + 0.05)
        self.INT_TP_ML_EXIT_MARGIN = float(
            os.getenv("INT_TP_ML_EXIT_MARGIN", str(exit_margin_default))
        )

        # ML entry sanity filters
        self.ML_FILTER_LOOKBACK = int(os.getenv("ML_FILTER_LOOKBACK", "60"))
        self.ML_MAX_EXT_PCT = float(os.getenv("ML_MAX_EXT_PCT", "0.015"))
        self.ML_MAX_RSI = float(os.getenv("ML_MAX_RSI", "68"))
        self.ML_MIN_RSI = float(os.getenv("ML_MIN_RSI", "32"))
        self.ML_MAX_BODY_ATR = float(os.getenv("ML_MAX_BODY_ATR", "2.5"))

        # Legacy strategy toggles (kept for config parity)
        self.STACKC_RANGE_FALLBACK = os.getenv("STACKC_RANGE_FALLBACK", "MR")
        self.STACKC_VOL_FACTOR = float(os.getenv("STACKC_VOL_FACTOR", "1.6"))
        self.STACKC_VOL_FACTOR_SHORT = float(os.getenv("STACKC_VOL_FACTOR_SHORT", "2.1"))
        self.STACKC_GREEN_STREAK = int(os.getenv("STACKC_GREEN_STREAK", "3"))
        self.STACKC_RED_STREAK = int(os.getenv("STACKC_RED_STREAK", "4"))
        self.STACKC_BLOCK_TOP = os.getenv("STACKC_BLOCK_TOP", "0") == "1"
        self.STACKC_USE_ML = os.getenv("STACKC_USE_ML", "0") == "1"
        self.STACKC_TREND_ADX_MIN = int(os.getenv("STACKC_TREND_ADX_MIN", "22"))
        self.STACKC_RANGE_ADX_MAX = int(os.getenv("STACKC_RANGE_ADX_MAX", "18"))
        self.STACKC_BB_WIDTH_MAX = float(os.getenv("STACKC_BB_WIDTH_MAX", "0.025"))
        self.SHORT_CONFIRM_1H = os.getenv("SHORT_CONFIRM_1H", "1") == "1"
        self.SHORT_1H_ADX_MIN = int(os.getenv("SHORT_1H_ADX_MIN", "20"))

        # Mean reversion legacy params (not used but preserved)
        self.MR_ADX_MAX = int(os.getenv("MR_ADX_MAX", "20"))
        self.MR_BB_WIDTH_MAX = float(os.getenv("MR_BB_WIDTH_MAX", "0.025"))
        self.MR_RSI_LOW = int(os.getenv("MR_RSI_LOW", "32"))
        self.MR_RSI_HIGH = int(os.getenv("MR_RSI_HIGH", "68"))
        self.MR_TOUCH_EPS = float(os.getenv("MR_TOUCH_EPS", "0.001"))
        self.MR_SPIKE_VOL_FACTOR = float(os.getenv("MR_SPIKE_VOL_FACTOR", "2.5"))
        self.MR_MIN_STREAK = int(os.getenv("MR_MIN_STREAK", "2"))
        self.MR_STRICT_SHORTS = os.getenv("MR_STRICT_SHORTS", "1") == "1"
        self.MR_SHORT_CONFIRM_1H = os.getenv("MR_SHORT_CONFIRM_1H", "0") == "1"
        self.MR_SHORT_1H_ADX_MIN = int(os.getenv("MR_SHORT_1H_ADX_MIN", "18"))

        # Misc toggles
        self.ANTI_LOSS_ON = os.getenv("ANTI_LOSS_ON", "1") == "1"
        self.ANTI_LOSS_THR_LONG = float(os.getenv("ANTI_LOSS_THR_LONG", "0.9"))
        self.ANTI_LOSS_THR_SHORT = float(os.getenv("ANTI_LOSS_THR_SHORT", "0.82"))
        self.ALLOW_REVERSE = os.getenv("ALLOW_REVERSE", "0") == "1"
        self.DAILY_DD_MAX_PCT = float(os.getenv("DAILY_DD_MAX_PCT", "0.0"))

        # ML artifact roots
        self.ML_MODELS_ROOT = self._make_absolute_path(os.getenv("ML_MODELS_ROOT", "models/trained"))
        self.ML_DEFAULT_TIMEFRAME = os.getenv("ML_DEFAULT_TIMEFRAME", self.ENTRY_TIMEFRAME)
        self.ML_MODEL_FILENAME = os.getenv("ML_MODEL_FILENAME")
        self.ML_SCALER_FILENAME = os.getenv("ML_SCALER_FILENAME")
        self.ML_META_FILENAME = os.getenv("ML_META_FILENAME")
        self.ML_MODEL_DEVICE = os.getenv("ML_MODEL_DEVICE", "cpu").lower()
        self.ML_HISTORY_BARS = max(1, int(os.getenv("ML_HISTORY_BARS", "512")))
        self.ML_HISTORY_DAYS = int(os.getenv("ML_HISTORY_DAYS", "180"))
        self.ML_TRAIN_HORIZON = int(os.getenv("ML_TRAIN_HORIZON", "12"))
        self.ML_TRAIN_TARGET_RETURN = float(os.getenv("ML_TRAIN_TARGET_RETURN", "0.002"))
        extra_tf_raw = os.getenv("ML_EXTRA_TIMEFRAMES", os.getenv("ML_ADDITIONAL_TIMEFRAMES", ""))
        self.ML_EXTRA_TIMEFRAMES = tuple(
            tf.strip() for tf in extra_tf_raw.split(",") if tf and tf.strip()
        )

        # Symbol resolution
        descriptors = self._parse_symbol_descriptors(base_symbol)
        if not descriptors:
            descriptors = [SymbolDescriptor(base_symbol, None, None)]
        symbols = [d.symbol for d in descriptors]
        primary_symbol = symbols[0]
        self.SYMBOL_DESCRIPTORS = descriptors
        self.SYMBOLS = symbols
        self.SYMBOL = primary_symbol
        self.SYMBOL_SETTINGS = self._build_symbol_settings(
            descriptors=descriptors,
            base_leverage=base_leverage,
            base_capital_usage=base_capital_usage,
            primary_symbol=primary_symbol,
        )
        self.SYMBOL_LEVERAGE = {k: v.leverage for k, v in self.SYMBOL_SETTINGS.items()}
        self.SYMBOL_ALLOCATIONS = {k: v.capital_usage_pct for k, v in self.SYMBOL_SETTINGS.items()}

        primary_settings = self.SYMBOL_SETTINGS[primary_symbol]
        self.LEVERAGE = primary_settings.leverage
        self.CAPITAL_USAGE_PCT = primary_settings.capital_usage_pct
        self.ML_MODEL_TIMEFRAME = primary_settings.timeframe
        self.ML_MODEL_PATH = str(primary_settings.model_path)
        self.ML_SCALER_PATH = str(primary_settings.scaler_path)
        self.ML_META_PATH = str(primary_settings.meta_path)

    # ------------------------------------------------------------------ #
    # Symbol helpers
    # ------------------------------------------------------------------ #

    def get_symbol_config(self, symbol: str) -> SymbolSettings:
        key = _upper_symbol(symbol)
        if key in self.SYMBOL_SETTINGS:
            return self.SYMBOL_SETTINGS[key]
        dynamic = self._build_dynamic_symbol_settings(key)
        self.SYMBOL_SETTINGS[key] = dynamic
        self.SYMBOL_LEVERAGE[key] = dynamic.leverage
        self.SYMBOL_ALLOCATIONS[key] = dynamic.capital_usage_pct
        if key not in self.SYMBOLS:
            self.SYMBOLS.append(key)
        return dynamic

    # ------------------------------------------------------------------ #
    # Internal parsing helpers
    # ------------------------------------------------------------------ #

    def _parse_symbol_descriptors(self, base_symbol: str) -> List[SymbolDescriptor]:
        raw = os.getenv("SYMBOLS", "") or ""
        tokens = [t.strip() for t in re.split(r"[,\n]+", raw) if t.strip()]
        descriptors: List[SymbolDescriptor] = []
        index: Dict[str, int] = {}

        for token in tokens:
            parts = token.split(":")
            symbol = _upper_symbol(parts[0] if parts else "")
            if not symbol:
                continue

            leverage: Optional[int] = None
            capital_usage: Optional[float] = None

            if len(parts) >= 3:
                lev_candidate = _parse_positive_number(parts[1])
                share_candidate = _parse_share(parts[2])
                if lev_candidate is not None:
                    leverage = int(round(lev_candidate))
                if share_candidate is not None:
                    capital_usage = share_candidate
            elif len(parts) == 2:
                share_candidate = _parse_share(parts[1])
                lev_candidate = _parse_positive_number(parts[1])
                if share_candidate is not None and share_candidate <= 1:
                    capital_usage = share_candidate
                elif lev_candidate is not None:
                    leverage = int(round(lev_candidate))

            if symbol in index:
                idx = index[symbol]
                current = descriptors[idx]
                descriptors[idx] = SymbolDescriptor(
                    symbol=symbol,
                    leverage=leverage if leverage is not None else current.leverage,
                    capital_usage=capital_usage if capital_usage is not None else current.capital_usage,
                )
            else:
                index[symbol] = len(descriptors)
                descriptors.append(SymbolDescriptor(symbol, leverage, capital_usage))

        if base_symbol not in index:
            descriptors.insert(0, SymbolDescriptor(base_symbol, None, None))
        return descriptors

    def _build_symbol_settings(
        self,
        descriptors: List[SymbolDescriptor],
        base_leverage: float,
        base_capital_usage: float,
        primary_symbol: str,
    ) -> Dict[str, SymbolSettings]:
        symbol_list = [d.symbol for d in descriptors]
        model_map = self._parse_symbol_models(symbol_list)
        settings: Dict[str, SymbolSettings] = {}

        for desc in descriptors:
            symbol = desc.symbol
            leverage = int(round(desc.leverage if desc.leverage is not None else base_leverage))
            capital_usage = desc.capital_usage if desc.capital_usage is not None else base_capital_usage
            capital_usage = float(min(max(capital_usage, 0.0), 1.0))

            timeframe, model_dir = model_map.get(symbol, (self.ML_DEFAULT_TIMEFRAME, self.ML_MODELS_ROOT))

            model_path = self._resolve_artifact_path(
                base_dir=model_dir,
                override=self.ML_MODEL_FILENAME,
                candidates=self._model_name_candidates(symbol, timeframe),
            )
            scaler_path = self._resolve_artifact_path(
                base_dir=model_dir,
                override=self.ML_SCALER_FILENAME,
                candidates=self._scaler_name_candidates(symbol, timeframe),
            )
            meta_path = self._resolve_artifact_path(
                base_dir=model_dir,
                override=self.ML_META_FILENAME,
                candidates=self._meta_name_candidates(symbol, timeframe),
            )

            settings[symbol] = SymbolSettings(
                symbol=symbol,
                leverage=leverage,
                capital_usage_pct=capital_usage,
                timeframe=timeframe,
                model_path=model_path,
                scaler_path=scaler_path,
                meta_path=meta_path,
            )

        model_override = os.getenv("ML_MODEL_PATH")
        scaler_override = os.getenv("ML_SCALER_PATH")
        meta_override = os.getenv("ML_META_PATH")
        timeframe_override = os.getenv("ML_MODEL_TIMEFRAME")

        if primary_symbol in settings:
            base_settings = settings[primary_symbol]
            settings[primary_symbol] = SymbolSettings(
                symbol=base_settings.symbol,
                leverage=base_settings.leverage,
                capital_usage_pct=base_settings.capital_usage_pct,
                timeframe=timeframe_override or base_settings.timeframe,
                model_path=self._explicit_or(base_settings.model_path, model_override),
                scaler_path=self._explicit_or(base_settings.scaler_path, scaler_override),
                meta_path=self._explicit_or(base_settings.meta_path, meta_override),
            )

        return settings

    def _parse_symbol_models(self, symbols: List[str]) -> Dict[str, Tuple[str, Path]]:
        mapping: Dict[str, Tuple[str, Path]] = {}
        for symbol in symbols:
            mapping[symbol] = (
                self.ML_DEFAULT_TIMEFRAME,
                (self.ML_MODELS_ROOT / symbol / self.ML_DEFAULT_TIMEFRAME).resolve(),
            )

        raw = os.getenv("ML_SYMBOL_MODELS", "") or ""
        entries = [part.strip() for part in re.split(r"[;,]+", raw) if part.strip()]

        for entry in entries:
            delimiter = "=" if "=" in entry else ":"
            if delimiter not in entry:
                continue
            left, right = entry.split(delimiter, 1)
            symbol_token = left.strip()
            if not symbol_token:
                continue
            if "@" in symbol_token:
                sym_part, tf_part = symbol_token.split("@", 1)
                symbol = _upper_symbol(sym_part)
                timeframe = tf_part.strip() or self.ML_DEFAULT_TIMEFRAME
            else:
                symbol = _upper_symbol(symbol_token)
                timeframe = self.ML_DEFAULT_TIMEFRAME

            if not symbol:
                continue

            clean_path = right.strip()
            if not clean_path:
                resolved_dir = self.ML_MODELS_ROOT
            else:
                raw_path = Path(clean_path)
                resolved_dir = raw_path if raw_path.is_absolute() else (self.ML_MODELS_ROOT / raw_path)
                resolved_dir = resolved_dir.resolve()

            mapping[symbol] = (timeframe, resolved_dir)

        return mapping

    def _resolve_artifact_path(
        self,
        base_dir: Path,
        override: Optional[str],
        candidates: List[str],
    ) -> Path:
        names: List[str] = []
        if override:
            names.append(override)
        names.extend(candidates)

        absolute_candidates: List[Path] = []
        for name in names:
            path = Path(name)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            else:
                path = path.resolve()
            absolute_candidates.append(path)
            if path.exists():
                return path

        return absolute_candidates[0] if absolute_candidates else base_dir.resolve()

    def _model_name_candidates(self, symbol: str, timeframe: str) -> List[str]:
        aliases = self._symbol_aliases(symbol)
        names = [
            "model.pt",
            "pattern_model.pt",
            f"model_{timeframe}.pt",
            f"pattern_model_{timeframe}.pt",
        ]
        for alias in aliases:
            names.append(f"{alias}_pattern_model_{timeframe}.pt")
            names.append(f"{alias}_pattern_model.pt")
            names.append(f"{alias}_model_{timeframe}.pt")
            names.append(f"{alias}_model.pt")
        return self._dedupe(names)

    def _scaler_name_candidates(self, symbol: str, timeframe: str) -> List[str]:
        aliases = self._symbol_aliases(symbol)
        names = [
            "scaler.pkl",
            "pattern_scaler.pkl",
            f"scaler_{timeframe}.pkl",
            f"pattern_scaler_{timeframe}.pkl",
            "scaler.joblib",
            "pattern_scaler.joblib",
        ]
        for alias in aliases:
            names.append(f"{alias}_pattern_scaler_{timeframe}.pkl")
            names.append(f"{alias}_pattern_scaler.pkl")
            names.append(f"{alias}_scaler_{timeframe}.pkl")
            names.append(f"{alias}_scaler.pkl")
        return self._dedupe(names)

    def _meta_name_candidates(self, symbol: str, timeframe: str) -> List[str]:
        aliases = self._symbol_aliases(symbol)
        names = [
            "meta.json",
            "pattern_meta.json",
            f"meta_{timeframe}.json",
            f"pattern_meta_{timeframe}.json",
        ]
        for alias in aliases:
            names.append(f"{alias}_pattern_meta_{timeframe}.json")
            names.append(f"{alias}_pattern_meta.json")
            names.append(f"{alias}_meta_{timeframe}.json")
            names.append(f"{alias}_meta.json")
        return self._dedupe(names)

    def _symbol_aliases(self, symbol: str) -> List[str]:
        base = symbol.lower()
        aliases = {base, base.replace("_", "")}
        if base.endswith("usdt"):
            aliases.add(base[:-4])
        if base.endswith("perp"):
            aliases.add(base[:-4])
        aliases.add(base.replace("-", ""))
        return [a for a in aliases if a]

    def _dedupe(self, items: List[str]) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def _build_dynamic_symbol_settings(self, symbol: str) -> SymbolSettings:
        leverage = int(round(float(getattr(self, "LEVERAGE", 0)) or 0))
        if leverage <= 0:
            leverage = 1

        capital_usage = float(getattr(self, "CAPITAL_USAGE_PCT", 0.0) or 0.0)
        capital_usage = float(min(max(capital_usage, 0.0), 1.0))

        timeframe = getattr(self, "ML_MODEL_TIMEFRAME", None) or self.ML_DEFAULT_TIMEFRAME

        model_dir = (self.ML_MODELS_ROOT / symbol / timeframe).resolve()
        model_path = self._resolve_artifact_path(
            base_dir=model_dir,
            override=self.ML_MODEL_FILENAME,
            candidates=self._model_name_candidates(symbol, timeframe),
        )
        scaler_path = self._resolve_artifact_path(
            base_dir=model_dir,
            override=self.ML_SCALER_FILENAME,
            candidates=self._scaler_name_candidates(symbol, timeframe),
        )
        meta_path = self._resolve_artifact_path(
            base_dir=model_dir,
            override=self.ML_META_FILENAME,
            candidates=self._meta_name_candidates(symbol, timeframe),
        )

        return SymbolSettings(
            symbol=symbol,
            leverage=leverage,
            capital_usage_pct=capital_usage,
            timeframe=timeframe,
            model_path=model_path,
            scaler_path=scaler_path,
            meta_path=meta_path,
        )

    def _explicit_or(self, fallback: Path, override: Optional[str]) -> Path:
        if not override:
            return fallback
        override_path = Path(override)
        if override_path.is_absolute():
            return override_path.resolve()
        if override_path.parent == Path("."):
            return (fallback.parent / override_path).resolve()
        return (self.PROJECT_ROOT / override_path).resolve()

    def _make_absolute_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = (self.PROJECT_ROOT / path).resolve()
        else:
            path = path.resolve()
        return path


CONFIG = Config()
