from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AnchorSpec:
    text: str
    x: float
    y: float
    size: int
    region: str
    color: tuple[int, int, int]


JORDAN_REFERENCE_ANCHORS: Sequence[AnchorSpec] = [
    AnchorSpec("MICHAEL JORDAN", 0.74, 0.18, 42, "face", (230, 230, 230)),
    AnchorSpec("MVP", 0.74, 0.27, 58, "face", (235, 235, 235)),
    AnchorSpec("CHAMPION", 0.72, 0.45, 36, "face", (210, 210, 210)),
    AnchorSpec("BULLS", 0.72, 0.70, 72, "jersey", (220, 25, 35)),
    AnchorSpec("23", 0.79, 0.82, 116, "jersey", (245, 235, 210)),
    AnchorSpec("SIX RINGS", 0.70, 0.86, 34, "jersey", (225, 32, 42)),
    AnchorSpec("AIR JORDAN", 0.70, 0.34, 34, "face", (220, 220, 220)),
    AnchorSpec("CLUTCH", 0.80, 0.59, 34, "jersey", (235, 235, 235)),
]


def scale_anchor(anchor: AnchorSpec, width: int, height: int) -> tuple[int, int, int]:
    x = int(anchor.x * width)
    y = int(anchor.y * height)
    size = max(8, int(anchor.size * (width / 1920.0)))
    return x, y, size


def anchor_bbox(
    anchor: AnchorSpec,
    width: int,
    height: int,
    font_loader,
    pad: int = 8,
) -> tuple[int, int, int, int]:
    x, y, size = scale_anchor(anchor, width=width, height=height)
    font = font_loader(size)
    left, top, right, bottom = font.getbbox(anchor.text)
    w = max(1, right - left)
    h = max(1, bottom - top)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)
    return x0, y0, x1, y1
