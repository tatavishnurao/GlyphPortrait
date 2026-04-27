from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from PIL import ImageDraw, ImageFont


@dataclass
class LayoutWord:
    word: str
    x: int
    y: int
    size: int
    weight: float


@dataclass
class LayoutStats:
    words_input: int
    words_placed: int
    placement_attempts: int
    fill_ratio: float


def _bbox_size(font: ImageFont.ImageFont, word: str) -> Tuple[int, int]:
    left, top, right, bottom = font.getbbox(word)
    return max(1, right - left), max(1, bottom - top)


def _inside_mask(mask: np.ndarray, x: int, y: int, w: int, h: int, threshold: float = 0.86) -> bool:
    mh, mw = mask.shape[:2]
    if x < 0 or y < 0 or x + w > mw or y + h > mh:
        return False
    region = mask[y : y + h, x : x + w]
    if region.size == 0:
        return False
    return float((region > 0).mean()) >= threshold


def generate_layout(
    words: Sequence[Tuple[str, float]],
    mask: np.ndarray,
    min_size: int,
    max_size: int,
    density: float,
    attempts_per_word: int,
    seed: int,
    font_loader,
) -> Tuple[List[LayoutWord], LayoutStats]:
    rng = np.random.default_rng(seed)
    h, w = mask.shape[:2]
    occupancy = np.zeros((h, w), dtype=np.uint8)

    ordered = sorted(words, key=lambda x: x[1], reverse=True)
    placements: List[LayoutWord] = []
    attempts = 0

    target_fill = np.clip(density, 0.15, 0.98)
    max_fill_pixels = int((mask > 0).sum() * target_fill)
    used_pixels = 0

    for word, weight in ordered:
        if used_pixels >= max_fill_pixels:
            break
        size = int(min_size + (max_size - min_size) * np.clip(weight, 0.0, 1.0))
        size = max(min_size, min(max_size, size))

        placed = False
        for _ in range(attempts_per_word):
            attempts += 1
            jitter = int(rng.integers(-3, 4))
            try_size = max(min_size, size + jitter)
            font = font_loader(try_size)
            tw, th = _bbox_size(font, word)
            x = int(rng.integers(0, max(1, w - tw)))
            y = int(rng.integers(0, max(1, h - th)))

            if not _inside_mask(mask, x, y, tw, th):
                continue
            roi = occupancy[y : y + th, x : x + tw]
            if roi.size == 0 or roi.any():
                continue

            occupancy[y : y + th, x : x + tw] = 1
            used_pixels += tw * th
            placements.append(LayoutWord(word=word, x=x, y=y, size=try_size, weight=weight))
            placed = True
            break

        if not placed:
            continue

    fill_ratio = float(used_pixels / max(1, (mask > 0).sum()))
    stats = LayoutStats(
        words_input=len(words),
        words_placed=len(placements),
        placement_attempts=attempts,
        fill_ratio=fill_ratio,
    )
    return placements, stats
