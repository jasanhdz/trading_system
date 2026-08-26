from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .core import causal_bars, normalize_panel

BG = (12, 16, 22)
GRID = (37, 45, 55)
UP = (54, 205, 146)
DOWN = (244, 91, 105)
TEXT = (205, 212, 220)
VOL_UP = (42, 112, 91)
VOL_DOWN = (128, 57, 66)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_panel(canvas: Image.Image, box: tuple[int, int, int, int], bars: pd.DataFrame, title: str) -> None:
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=BG, outline=GRID, width=1)
    draw.text((x0 + 9, y0 + 7), title, font=_font(16), fill=TEXT)
    plot = (x0 + 9, y0 + 31, x1 - 9, y1 - 9)
    px0, py0, px1, py1 = plot
    split = int(py0 + (py1 - py0) * .77)
    for frac in (.25, .5, .75):
        yy = int(py0 + (split - py0) * frac)
        draw.line((px0, yy, px1, yy), fill=GRID, width=1)
    if bars.empty:
        return
    b = normalize_panel(bars)
    n = len(b)
    slot = (px1 - px0) / max(n, 1)
    body_w = max(1, int(slot * .62))
    for i, row in b.iterrows():
        cx = int(px0 + (i + .5) * slot)
        yp = lambda v: int(split - 2 - float(v) * max(split - py0 - 4, 1))
        color = UP if row.close >= row.open else DOWN
        draw.line((cx, yp(row.low), cx, yp(row.high)), fill=color, width=max(1, body_w // 3))
        top, bot = sorted((yp(row.open), yp(row.close)))
        draw.rectangle((cx - body_w // 2, top, cx + body_w // 2, max(top + 1, bot)), fill=color)
        vtop = int(py1 - float(row.volume) * max(py1 - split - 3, 1))
        draw.rectangle((cx - body_w // 2, vtop, cx + body_w // 2, py1), fill=VOL_UP if color == UP else VOL_DOWN)


def render_frame(dfs: dict[str, pd.DataFrame], decision_at_ms: int, resolution: tuple[int, int] = (1200, 900)) -> Image.Image:
    """Canonical five-panel RAW_VIEW. No absolute price or time is rendered."""
    w, h = resolution
    image = Image.new("RGB", resolution, BG)
    gap = 8
    top_h = int(h * .42)
    row_h = (h - top_h - 3 * gap) // 2
    half = (w - 3 * gap) // 2
    specs = [
        ((gap, gap, w-gap, top_h), "SUIUSDT", 15, 40, "SUIUSDT 15m"),
        ((gap, top_h+gap, gap+half, top_h+gap+row_h), "SUIUSDT", 5, 60, "SUIUSDT 5m"),
        ((2*gap+half, top_h+gap, w-gap, top_h+gap+row_h), "BTCUSDT", 5, 60, "BTCUSDT 5m"),
        ((gap, top_h+2*gap+row_h, gap+half, h-gap), "SUIUSDT", 3, 80, "SUIUSDT 3m"),
        ((2*gap+half, top_h+2*gap+row_h, w-gap, h-gap), "BTCUSDT", 3, 80, "BTCUSDT 3m"),
    ]
    for box, symbol, tf, count, title in specs:
        draw_panel(image, box, causal_bars(dfs[symbol], decision_at_ms, tf, count), title)
    return image


def image_sha256(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def layout_fingerprint(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.resize((32, 24)).convert("L"), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    gy = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    return np.concatenate([gx, gy]).astype(np.float32)


def layout_distance(a: Image.Image, b: Image.Image) -> float:
    return float(np.linalg.norm(layout_fingerprint(a) - layout_fingerprint(b)))

