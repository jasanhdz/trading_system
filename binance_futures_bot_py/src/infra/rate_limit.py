"""Basic rate-limit tracking to avoid Binance REST bans."""
from __future__ import annotations

import re
import time
from typing import Optional, Union

_ban_until_ms = 0.0

_BAN_REGEX = re.compile(r"banned until (\d+)", re.IGNORECASE)


def _extract_ban_until(message: str) -> Optional[int]:
    match = _BAN_REGEX.search(message)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    if "way too many requests" in message.lower():
        return int(time.time() * 1000 + 60_000)
    return None


def note_rate_limit_until(ts_ms: int) -> None:
    global _ban_until_ms
    if ts_ms > _ban_until_ms:
        _ban_until_ms = float(ts_ms)


def note_rate_limit_from_error(err: Union[BaseException, str]) -> Optional[int]:
    if isinstance(err, BaseException):
        msg = getattr(err, "message", None) or getattr(err, "msg", None) or str(err)
    else:
        msg = str(err)
    ban = _extract_ban_until(msg)
    if ban is not None:
        note_rate_limit_until(ban)
    return ban


def is_rate_limited() -> bool:
    return time.time() * 1000 <= _ban_until_ms


def ms_until_reset() -> int:
    return max(0, int(_ban_until_ms - time.time() * 1000))
