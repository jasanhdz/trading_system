# src/infra/fs/logger.py
import os
import re
import json
import errno
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Callable, List

from ...core.ports.logger import Logger
from ..config import CONFIG

# =========================
#  Config (estilo TS)
# =========================

LOG_DIR = Path(os.getenv("LOG_DIR", "logs")).resolve()
LEGACY_PATH = LOG_DIR / "history.log"

LEVELS = ("debug", "info", "warn", "error")
ORDER = {lvl: i for i, lvl in enumerate(LEVELS)}
CURRENT_LEVEL = os.getenv("LOG_LEVEL", "info").lower()
if CURRENT_LEVEL not in ORDER:
    CURRENT_LEVEL = "info"

LOG_TO_FILE = os.getenv("LOG_TO_FILE", "1") != "0"
LOG_RETAIN_DAYS = int(os.getenv("LOG_RETAIN_DAYS", "7"))
PRETTY = os.getenv("LOG_PRETTY", "1") == "1"  # por defecto “bonito”
FILE_LOGGING_DISABLED = False

# =========================
#  ANSI colores (sin deps)
# =========================

def _code(n: int) -> Callable[[str], str]:
    return lambda s: f"\x1b[{n}m{s}\x1b[0m"

color = {
    "dim": _code(2),
    "gray": _code(90),
    "info": _code(36),   # cyan
    "ok": _code(32),     # green
    "warn": _code(33),   # yellow
    "error": _code(31),  # red
    "bold": _code(1),
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)

def _pad_cell(text: str, width: int) -> str:
    pad = width - len(_strip_ansi(text))
    return text + (" " * pad if pad > 0 else "")

def _format_usd(value: Any) -> str:
    if not isinstance(value, (int, float)) or not value == value:
        return color["gray"]("—")
    formatted = f"${value:,.2f}"
    if value > 0:
        return color["ok"](formatted)
    if value < 0:
        return color["error"](formatted)
    return color["gray"](formatted)

def _format_roi_pct(value: Any) -> str:
    if not isinstance(value, (int, float)) or not value == value:
        return color["gray"]("—")
    formatted = f"{'+' if value >= 0 else ''}{value:.2f}%"
    if value > 0:
        return color["ok"](formatted)
    if value < 0:
        return color["error"](formatted)
    return color["gray"](formatted)

def _render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(_strip_ansi(h)) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(_strip_ansi(cell)))

    def border(left: str, cross: str, right: str) -> str:
        segments = ["─" * (w + 2) for w in widths]
        return left + cross.join(segments) + right

    def format_row(row_values: list[str]) -> str:
        cells = [_pad_cell(val, widths[idx]) for idx, val in enumerate(row_values)]
        return "│ " + " │ ".join(cells) + " │"

    lines = [border("┌", "┬", "┐"), format_row(headers), border("├", "┼", "┤")]
    for row in rows:
        lines.append(format_row(row))
    lines.append(border("└", "┴", "┘"))
    return "\n".join(lines)

def zone_badge(z: Optional[str]) -> str:
    if z == "SUPPORT":
        return color["ok"]("🛡️ SUPPORT")
    if z == "RESISTANCE":
        return color["error"]("🧱 RESISTANCE")
    if z == "MIDDLE":
        return color["gray"]("⚖️ MIDDLE")
    if z == "LOWER_RANGE":
        return color["info"]("⬇️ LOWER")
    if z == "UPPER_RANGE":
        return color["info"]("⬆️ UPPER")
    return color["gray"]("—")

def trend_badge(dir_: Optional[str]) -> str:
    if dir_ == "STRONG_BULL":
        return color["ok"]("⚡ STRONG BULL") + " 📈"
    if dir_ == "BULL":
        return color["ok"]("🟢 BULL")
    if dir_ == "NEUTRAL":
        return color["warn"]("🟡 NEUTRAL")
    if dir_ == "BEAR":
        return color["error"]("🔴 BEAR")
    if dir_ == "STRONG_BEAR":
        return color["error"]("⚡ STRONG BEAR") + " 📉"
    return color["gray"]("⚪ UNKNOWN")

# =========================
#  Helpers FS
# =========================

def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def _should(level: str) -> bool:
    return ORDER[level] >= ORDER[CURRENT_LEVEL]

def _today_path() -> Path:
    date = datetime.utcnow().strftime("%Y-%m-%d")
    return LOG_DIR / f"history-{date}.log"

def _prune_old_logs(retain_days: int) -> None:
    try:
        _ensure_log_dir()
        now = int(time.time() * 1000)
        cutoff = now - retain_days * 86_400_000
        for f in LOG_DIR.iterdir():
            m = re.match(r"^history-(\d{4}-\d{2}-\d{2})\.log(\.gz)?$", f.name)
            if not m:
                continue
            try:
                t = int(datetime.fromisoformat(m.group(1) + "T00:00:00").timestamp()) * 1000
                if t < cutoff:
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass

def _append(file_path: Path, line: str) -> None:
    global FILE_LOGGING_DISABLED
    if not LOG_TO_FILE or FILE_LOGGING_DISABLED:
        return
    try:
        _ensure_log_dir()
        with file_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        if e.errno in (errno.ENOSPC, errno.EFBIG):
            print(color["warn"](f"[Logger] {e.strerror or e.errno}: deshabilitando archivo; quedará solo consola."))
            _prune_old_logs(max(3, min(LOG_RETAIN_DAYS, 30)))
            FILE_LOGGING_DISABLED = True
            return
        print(color["error"]("❌ Error escribiendo log: " + str(e)))

# =========================
#  Pretty printer
# =========================

def _local_time_str() -> str:
    # Hora local estilo 10:48:33 PM
    return datetime.now().strftime("%-I:%M:%S %p") if os.name != "nt" else datetime.now().strftime("%I:%M:%S %p")

def _n(x: Any, d: int = 2) -> str:
    return f"{x:.{d}f}" if isinstance(x, (int, float)) and (x == x) else str(x)

def _pct(x: Optional[float], d: int = 1) -> str:
    if isinstance(x, (int, float)) and (x == x):
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.{d}f}%"
    return "—"

def _sgn(x: Optional[float], s: Optional[str] = None) -> str:
    if isinstance(x, (int, float)) and (x == x):
        painter = color["ok"] if x > 0 else color["error"] if x < 0 else color["gray"]
        return painter(s if s is not None else str(x))
    return color["gray"]("—")

def _level_color(level: str) -> Callable[[str], str]:
    return color["error"] if level == "error" else color["warn"] if level == "warn" else color["info"] if level == "info" else color["gray"]

def _pretty_line(level: str, msg: str, ctx: Optional[Dict[str, Any]]) -> str:
    t = color["gray"](_local_time_str())
    L = _level_color(level)

    # Plantillas clave (paridad con TS)
    if msg == "sync_attach_to_open_position":
        side = ctx.get("side") if ctx else "?"
        entry = _n(ctx.get("entry"), 4) if ctx else "—"
        lev = _n(ctx.get("lev"), 0) if ctx else "—"
        qty = _n(ctx.get("qtyAbs"), 4) if ctx else "—"
        emoji = "🟢" if side == "LONG" else "🔻"
        return f"{t} {emoji} {color['bold']('Attach')} {side} @ {entry} ×{lev} | qty {qty}"

    if msg == "raw_open_orders":
        count = (ctx or {}).get("count", 0)
        sample = (ctx or {}).get("sample", [])
        stops = [o for o in sample if str(o.get("type", "")).upper().find("STOP") >= 0]
        tps = [o for o in sample if str(o.get("type", "")).upper().find("TAKE_PROFIT") >= 0]
        bestStop = max(stops, key=lambda o: float(o.get("stopPrice", 0)))["stopPrice"] if stops else "-"
        bestTP = tps[0]["stopPrice"] if tps else "-"
        return f"{t} 📜 Órdenes abiertas: {count} | ⛔ stop* {bestStop} | 🎯 tp* {bestTP}"

    if msg == "tp_watch":
        side = (ctx or {}).get("side", "?")
        hit = color["ok"]("HIT") if (ctx or {}).get("hit") else color["dim"]("…")
        return f"{t} 🎯 TP watch {side}: mark {_n((ctx or {}).get('mark'))} vs target {_n((ctx or {}).get('target'))} {hit}"

    if msg == "market_opened":
        side = (ctx or {}).get("side", "?")
        qty = _n((ctx or {}).get("qty"), 4)
        avgp = _n((ctx or {}).get("avgPrice"), 4)
        return f"{t} 🚀 Abierto {side} qty {qty} @ ~{avgp}"

    if msg == "signal":
        action = (ctx or {}).get("action", "IDLE")
        reason = (ctx or {}).get("reason", "")
        symbol = (ctx or {}).get("symbol") or "?"
        emoji = "🟢" if action == "ENTER_LONG" else "🔻" if action == "ENTER_SHORT" else "⏸️"
        return f"{t} {emoji} {action} [{symbol}] · {reason}"

    if msg == "position_snapshot":
        symbol = (ctx or {}).get("symbol", "?")
        side = (ctx or {}).get("side", "?")
        entry = _n((ctx or {}).get("entry"), 4)
        mark = _n((ctx or {}).get("mark"), 4)
        roi = _format_roi_pct((ctx or {}).get("roiPct"))
        pnl = _format_usd((ctx or {}).get("pnlUsd"))
        qty = _n((ctx or {}).get("qtyAbs"), 4)
        lev = _n((ctx or {}).get("leverage"), 0)
        open_ms = (ctx or {}).get("openMs")
        open_secs = (
            str(int(round(open_ms / 1000)))
            if isinstance(open_ms, (int, float)) and open_ms == open_ms
            else "—"
        )

        headers = [
            color["gray"]("Time"),
            color["gray"]("Symbol"),
            color["gray"]("Side"),
            color["gray"]("Entry"),
            color["gray"]("Mark"),
            color["gray"]("ROI %"),
            color["gray"]("PnL (USDT)"),
            color["gray"]("Qty"),
            color["gray"]("Lev"),
            color["gray"]("Open (s)"),
        ]

        row = [
            t,
            color["info"](str(symbol)),
            str(side),
            entry,
            mark,
            roi,
            pnl,
            qty,
            lev,
            open_secs,
        ]

        return _render_table(headers, [row])

    if msg in ("stop_upserted", "ensure_stop_created"):
        side = (ctx or {}).get("side", "?")
        return f"{t} ⛔ Stop {side} @ {_n((ctx or {}).get('stop'))}"

    if msg in ("tp_upserted", "ensure_tp_created"):
        side = (ctx or {}).get("side", "?")
        return f"{t} 🎯 TP {side} @ {_n((ctx or {}).get('tp'))}"

    if msg == "profit_guard_status":
        roe = (ctx or {}).get("roe", 0)
        peak = (ctx or {}).get("peak", 0)
        return f"{t} 🛡️ ROE {_pct(roe)} (peak {_pct(peak)})"

    if msg in ("BE_protect_close", "Time_stop_close", "Giveback_close", "Early_fail_close"):
        flat = color["dim"](json.dumps(ctx, ensure_ascii=False)) if ctx else ""
        tag = msg.replace("_", " ")
        return f"{t} 🔒 {color['warn'](tag)} {flat}"

    if msg == "market_snapshot":
        p = (ctx or {}).get("price")
        res = (ctx or {}).get("resistance")
        sup = (ctx or {}).get("support")
        z = (ctx or {}).get("zone")
        trend = (ctx or {}).get("trend")
        t10 = (ctx or {}).get("t10")
        t5 = (ctx or {}).get("t5")
        rsi = (ctx or {}).get("rsi")
        adx = (ctx or {}).get("adx")
        bbw = (ctx or {}).get("bbw")
        ema7 = (ctx or {}).get("ema7")
        ema25 = (ctx or {}).get("ema25")
        ema99 = (ctx or {}).get("ema99")

        line1 = f"{t} {color['info']('💹 market')} {color['gray']('•')} P:{color['bold'](_n(p,4))} {color['gray']('│')} R:{color['error'](_n(res,4))} S:{color['ok'](_n(sup,4))} {color['gray']('│')} {zone_badge(z)}"
        line2 = f" {trend_badge(trend)} {color['gray']('│')} T10:{_sgn(t10, _pct(t10,2))} {color['gray']('·')} T5:{_sgn(t5, _pct(t5,2))} {color['gray']('│')} RSI:{_n(rsi,1)} {color['gray']('·')} ADX:{_n(adx,1)} {color['gray']('·')} BBW:{_pct(bbw,2)}"
        line3 = f"{color['gray']('   └ ')}EMA7:{_n(ema7,4)}  EMA25:{_n(ema25,4)}  EMA99:{_n(ema99,4)}" if all(isinstance(x, (int, float)) for x in (ema7, ema25, ema99)) else ""
        return line1 + "\n" + line2 + (("\n" + line3) if line3 else "")

    # Fallback compacto: msg + pares plano
    flat = ""
    if ctx and isinstance(ctx, dict):
        flat = " ".join(f"{k}={v}" for k, v in ctx.items() if not isinstance(v, (dict, list)))
    return f"{t} {L(msg)}" + ((" " + color["dim"](flat)) if flat else "")

# =========================
#  Núcleo de escritura
# =========================

def _write(level: str, msg: str, ctx: Optional[Dict[str, Any]], static_ctx: Optional[Dict[str, Any]]) -> None:
    if not _should(level):
        return

    merged_ctx = {}
    if static_ctx:
        merged_ctx.update(static_ctx)
    if ctx:
        merged_ctx.update(ctx)

    symbol_tag = (merged_ctx or {}).get("symbol")

    payload = {
        "ts": datetime.utcnow().isoformat(),
        "level": level,
        "msg": msg,
        "ctx": merged_ctx if merged_ctx else None,
        # metadatos útiles (como en tu versión previa)
        "symbol": symbol_tag or CONFIG.SYMBOL,
        "is_testnet": CONFIG.IS_TESTNET,
    }

    # 1) Consola
    if PRETTY:
        line = _pretty_line(level, msg, merged_ctx)
        if level == "error":
            print(line, file=os.sys.stderr)
        elif level == "warn":
            print(line)
        elif level == "info":
            print(line)
        else:
            print(color["gray"](line))
    else:
        j = json.dumps(payload, ensure_ascii=False)
        if level == "error":
            print(j, file=os.sys.stderr)
        else:
            print(j)

    # 2) Archivos (si habilitado)
    if LOG_TO_FILE and not FILE_LOGGING_DISABLED:
        j = json.dumps(payload, ensure_ascii=False)
        _append(_today_path(), j)
        # compat línea plana
        flat = f"[{payload['ts']}] {msg}" + (f" {json.dumps(merged_ctx, ensure_ascii=False)}" if merged_ctx else "")
        _append(LEGACY_PATH, flat)

# =========================
#  API Logger (Protocol)
# =========================

def _merge_args_kwargs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # Permite logger.info("msg", {"a":1}) o logger.info("msg", a=1)
    merged = dict(kwargs)
    if args and isinstance(args[0], dict):
        merged = {**args[0], **merged}
    return merged

class FsLogger(Logger):
    """
    Logger sin dependencias externas, compatible con tu TS:
    - LOG_PRETTY=1 (consola bonita) o 0 (JSON)
    - LOG_TO_FILE=1/0
    - LOG_RETAIN_DAYS=N
    - LOG_LEVEL=debug|info|warn|error
    """
    def __init__(self, static_ctx: Optional[Dict[str, Any]] = None):
        self._static_ctx = dict(static_ctx or {})

        # limpieza opcional al arranque
        try:
            _prune_old_logs(LOG_RETAIN_DAYS)
        except Exception:
            pass

    # Niveles
    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        _write("debug", message, _merge_args_kwargs(args, kwargs), self._static_ctx)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        _write("info", message, _merge_args_kwargs(args, kwargs), self._static_ctx)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        _write("warn", message, _merge_args_kwargs(args, kwargs), self._static_ctx)

    def warn(self, message: str, *args: Any, **kwargs: Any) -> None:
        _write("warn", message, _merge_args_kwargs(args, kwargs), self._static_ctx)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        _write("error", message, _merge_args_kwargs(args, kwargs), self._static_ctx)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        _write("error", message, _merge_args_kwargs(args, kwargs), self._static_ctx)

    # bind() para compatibilidad con Protocol
    def bind(self, **kwargs: Any) -> "Logger":
        new_ctx = dict(self._static_ctx)
        new_ctx.update(kwargs)
        return FsLogger(static_ctx=new_ctx)

    # Opcionales: archivos especializados (como tenías)
    def log_trade(self, trade: Dict[str, Any]) -> None:
        try:
            _ensure_log_dir()
            path = LOG_DIR / "trades.jsonl"
            _append(path, json.dumps({"timestamp": datetime.utcnow().isoformat(), **trade}, ensure_ascii=False))
        except Exception:
            pass

    def log_signal(self, signal: Dict[str, Any]) -> None:
        try:
            _ensure_log_dir()
            path = LOG_DIR / "signals.jsonl"
            _append(path, json.dumps({"timestamp": datetime.utcnow().isoformat(), **signal}, ensure_ascii=False))
        except Exception:
            pass

    def log_performance(self, stats: Dict[str, Any]) -> None:
        try:
            _ensure_log_dir()
            path = LOG_DIR / "performance.jsonl"
            _append(path, json.dumps({"timestamp": datetime.utcnow().isoformat(), **stats}, ensure_ascii=False))
        except Exception:
            pass
