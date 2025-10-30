"""Package init for object_protection.

Adds a drop-in replacement for cv2.putText so Chinese text renders correctly.
"""

from __future__ import annotations

import os
import platform
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - Pillow missing
    Image = None
    ImageDraw = None
    ImageFont = None


def _iter_font_candidates() -> Iterable[Path]:
    env_font = os.environ.get("CV_SAFETY_FONT")
    if env_font:
        yield Path(env_font).expanduser()

    package_fonts = Path(__file__).resolve().parent / "fonts"
    if package_fonts.exists():
        for font_file in package_fonts.glob("*.ttf"):
            yield font_file
        for font_file in package_fonts.glob("*.ttc"):
            yield font_file

    system = platform.system()
    candidates: Tuple[str, ...]
    if system == "Darwin":
        candidates = (
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB W3.otf",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        )
    elif system == "Windows":
        candidates = (
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/Microsoft YaHei UI.ttf",
        )
    else:  # Linux/BSD
        candidates = (
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        )

    for path in candidates:
        yield Path(path)


def _find_font_path() -> Optional[Path]:
    for candidate in _iter_font_candidates():
        if candidate.is_file():
            return candidate
    return None


def _patch_cv2_puttext() -> None:
    if Image is None or ImageFont is None or ImageDraw is None:
        return

    font_path = _find_font_path()
    if font_path is None:
        return

    original_puttext = cv2.putText

    @lru_cache(maxsize=32)
    def _get_font(pixel_size: int) -> ImageFont.FreeTypeFont:
        size = max(12, pixel_size)
        return ImageFont.truetype(str(font_path), size=size)

    def _has_non_ascii(text: str) -> bool:
        return any(ord(char) > 127 for char in text)

    def _put_text_with_pillow(
        image: np.ndarray,
        text: str,
        org: Tuple[int, int],
        font_scale: float,
        color: Tuple[int, int, int],
        thickness: int,
        bottom_left_origin: bool,
    ) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            return original_puttext(
                image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
            )

        divisor = 0.9 if font_scale > 1.2 else 0.85
        pixel_size = int(font_scale * 42 * divisor)
        font = _get_font(pixel_size)

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        x, y = org
        ascent, descent = font.getmetrics()
        if bottom_left_origin:
            y = image.shape[0] - y
        y -= ascent

        fill = (int(color[2]), int(color[1]), int(color[0]))
        stroke_width = max(0, thickness - 1)
        stroke_fill = (0, 0, 0) if stroke_width > 0 else None
        draw.text(
            (x, y),
            text,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

        np.copyto(image, cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        return image

    def patched_puttext(
        image,
        text,
        org,
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1.0,
        color=(255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
        bottomLeftOrigin=False,
    ):
        if not text or not _has_non_ascii(text):
            return original_puttext(
                image,
                text,
                org,
                font_face,
                font_scale,
                color,
                thickness,
                lineType,
                bottomLeftOrigin,
            )
        return _put_text_with_pillow(
            image, text, org, font_scale, color, thickness, bottomLeftOrigin
        )

    cv2.putText = patched_puttext


_patch_cv2_puttext()
