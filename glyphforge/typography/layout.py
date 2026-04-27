from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
from PIL import ImageFont


@dataclass
class LayoutWord:
    word: str
    x: int
    y: int
    size: int
    width: int
    height: int
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


def _inside_mask(
    mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    threshold: float = 0.86,
) -> bool:
    mh, mw = mask.shape[:2]
    if x < 0 or y < 0 or x + w > mw or y + h > mh:
        return False
    region = mask[y : y + h, x : x + w]
    if region.size == 0:
        return False
    return float((region > 0).mean()) >= threshold


def _normalize_importance_map(
    importance_map: np.ndarray | None, mask: np.ndarray
) -> np.ndarray | None:
    if importance_map is None:
        return None
    if importance_map.shape != mask.shape:
        raise ValueError("importance_map shape must match mask shape")

    weighted = np.asarray(importance_map, dtype=np.float64).copy()
    weighted = np.clip(weighted, 0.0, None)
    weighted *= (mask > 0).astype(np.float64)
    total = float(weighted.sum())
    if total <= 0.0:
        return None
    return weighted / total


def _build_importance_cdf(normalized_map: np.ndarray | None) -> np.ndarray | None:
    if normalized_map is None:
        return None
    flat = normalized_map.ravel()
    cdf = np.cumsum(flat)
    if cdf.size > 0:
        cdf[-1] = 1.0
    return cdf


def _sample_candidate_top_left(
    rng: np.random.Generator,
    canvas_w: int,
    canvas_h: int,
    box_w: int,
    box_h: int,
    importance_cdf: np.ndarray | None,
) -> tuple[int, int]:
    max_x = max(0, canvas_w - box_w)
    max_y = max(0, canvas_h - box_h)

    if importance_cdf is None:
        x = int(rng.integers(0, max_x + 1))
        y = int(rng.integers(0, max_y + 1))
        return x, y

    rand = float(rng.random())
    idx = int(np.searchsorted(importance_cdf, rand, side="right"))
    idx = min(idx, importance_cdf.size - 1)
    center_y = idx // canvas_w
    center_x = idx % canvas_w

    x = center_x - (box_w // 2) + int(rng.integers(-2, 3))
    y = center_y - (box_h // 2) + int(rng.integers(-2, 3))
    return int(np.clip(x, 0, max_x)), int(np.clip(y, 0, max_y))


def generate_layout(
    words: Sequence[Tuple[str, float]],
    mask: np.ndarray,
    min_size: int,
    max_size: int,
    density: float,
    attempts_per_word: int,
    seed: int,
    font_loader: Callable[[int], ImageFont.ImageFont],
    importance_map: np.ndarray | None = None,
) -> Tuple[List[LayoutWord], LayoutStats]:
    rng = np.random.default_rng(seed)
    h, w = mask.shape[:2]
    occupancy = np.zeros((h, w), dtype=np.uint8)
    importance_cdf = _build_importance_cdf(
        _normalize_importance_map(importance_map=importance_map, mask=mask)
    )

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
            x, y = _sample_candidate_top_left(
                rng=rng,
                canvas_w=w,
                canvas_h=h,
                box_w=tw,
                box_h=th,
                importance_cdf=importance_cdf,
            )

            if not _inside_mask(mask, x, y, tw, th):
                continue
            roi = occupancy[y : y + th, x : x + tw]
            if roi.size == 0 or roi.any():
                continue

            occupancy[y : y + th, x : x + tw] = 1
            used_pixels += tw * th
            placements.append(
                LayoutWord(
                    word=word,
                    x=x,
                    y=y,
                    size=try_size,
                    width=tw,
                    height=th,
                    weight=weight,
                )
            )
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
