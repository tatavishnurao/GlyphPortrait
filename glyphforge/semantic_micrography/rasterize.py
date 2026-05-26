from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from glyphforge.semantic_micrography.config import MicrographyStyleConfig
from glyphforge.semantic_micrography.text_layout import TextLayoutResult


def rasterize_svg(svg_path: Path, png_path: Path) -> Path:
    if importlib.util.find_spec("cairosvg") is None:
        raise RuntimeError("cairosvg is not installed; use rasterize_layout_preview fallback or install cairosvg")
    import cairosvg

    png_path.parent.mkdir(parents=True, exist_ok=True)
    cairosvg.svg2png(url=str(svg_path), write_to=str(png_path))
    return png_path


def _font(size: int, bold: bool) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _draw_rotated_text(
    canvas: Image.Image,
    text: str,
    x: float,
    y: float,
    angle: float,
    font_size: int,
    fill: tuple[int, int, int],
    bold: bool,
) -> None:
    font = _font(font_size, bold)
    probe = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    patch = Image.new("RGBA", (tw + 8, th + 8), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=(*fill, 235))
    rotated = patch.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    canvas.alpha_composite(rotated, (int(x - rotated.width / 2), int(y - rotated.height / 2)))


def _samples(points: list[tuple[float, float]], spacing: float) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    if len(points) < 2:
        return out
    carry = 0.0
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1:
            continue
        dist = carry
        while dist <= length:
            t = dist / length
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, math.degrees(math.atan2(y1 - y0, x1 - x0))))
            dist += spacing
        carry = max(0.0, dist - length)
    return out


def rasterize_layout_preview(
    layout: TextLayoutResult,
    canvas_size: tuple[int, int],
    style: MicrographyStyleConfig,
    png_path: Path,
) -> Path:
    w, h = canvas_size
    canvas = Image.new("RGBA", (w, h), (*style.background_color, 255))
    for item in layout.text_paths:
        words = item.text.split()
        if not words:
            continue
        spacing = max(22.0, item.font_size * 4.8)
        for idx, (x, y, angle) in enumerate(_samples(item.lane.points, spacing)):
            token = words[idx % len(words)]
            if idx + 1 < len(words) and len(token) < 5:
                token = f"{token} {words[(idx + 1) % len(words)]}"
            _draw_rotated_text(canvas, token, x, y, angle, item.font_size, _hex_to_rgb(item.fill), item.is_hero)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(png_path)
    return png_path
