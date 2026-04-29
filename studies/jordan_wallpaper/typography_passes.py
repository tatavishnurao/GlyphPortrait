from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from glyphforge.keywords.parser import parse_weighted_words
from glyphforge.typography.layout import LayoutWord, generate_layout

from .anchors import AnchorSpec, anchor_bbox, scale_anchor


@dataclass
class PassStats:
    placed: int
    attempts: int


def _masked_importance(
    base_importance: np.ndarray, region_mask: np.ndarray, floor: float = 0.0
) -> np.ndarray:
    imp = base_importance.astype(np.float32).copy()
    imp *= (region_mask > 0).astype(np.float32)
    if floor > 0:
        imp += (region_mask > 0).astype(np.float32) * floor
    return imp


def render_text_pass(
    image: Image.Image,
    occupancy_mask: np.ndarray,
    region_mask: np.ndarray,
    importance_map: np.ndarray,
    words_text: str,
    font_loader,
    color_picker,
    min_size: int,
    max_size: int,
    density: float,
    attempts_per_word: int,
    seed: int,
) -> tuple[list[LayoutWord], PassStats]:
    words = parse_weighted_words(words_text)
    imp = _masked_importance(importance_map, region_mask, floor=0.01)
    layout, stats = generate_layout(
        words=words,
        mask=((region_mask > 0) & (occupancy_mask == 0)).astype(np.uint8) * 255,
        min_size=min_size,
        max_size=max_size,
        density=density,
        attempts_per_word=attempts_per_word,
        seed=seed,
        font_loader=font_loader,
        importance_map=imp,
    )
    draw = ImageDraw.Draw(image)
    for item in layout:
        draw.text(
            (item.x, item.y),
            item.word,
            font=font_loader(item.size),
            fill=color_picker(item),
        )
        occupancy_mask[item.y : item.y + item.height, item.x : item.x + item.width] = (
            255
        )
    return layout, PassStats(
        placed=stats.words_placed, attempts=stats.placement_attempts
    )


def render_anchor_pass(
    image: Image.Image,
    anchors: Sequence[AnchorSpec],
    font_loader,
    width: int,
    height: int,
) -> int:
    draw = ImageDraw.Draw(image)
    placed = 0
    for anchor in anchors:
        x, y, size = scale_anchor(anchor, width, height)
        draw.text((x, y), anchor.text, fill=anchor.color, font=font_loader(size))
        placed += 1
    return placed


def render_slogan(
    image: Image.Image,
    text: str,
    font_loader,
    canvas_w: int,
    canvas_h: int,
    color: tuple[int, int, int] = (190, 190, 200),
) -> None:
    draw = ImageDraw.Draw(image)
    font = font_loader(max(14, int(canvas_w * 0.02)))
    x = int(canvas_w * 0.33)
    y = int(canvas_h * 0.50)
    draw.text((x, y), text, fill=color, font=font)


def reserve_anchor_regions(
    occupancy_mask: np.ndarray,
    anchors: Sequence[AnchorSpec],
    font_loader,
    width: int,
    height: int,
    pad: int = 8,
) -> int:
    reserved = 0
    for anchor in anchors:
        x0, y0, x1, y1 = anchor_bbox(
            anchor=anchor,
            width=width,
            height=height,
            font_loader=font_loader,
            pad=pad,
        )
        if x1 > x0 and y1 > y0:
            occupancy_mask[y0:y1, x0:x1] = 255
            reserved += 1
    return reserved
