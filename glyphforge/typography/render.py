from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from glyphforge.config import AppConfig
from glyphforge.image.masks import cleanup_mask
from glyphforge.image.preprocess import preprocess_portrait
from glyphforge.image.segment import segment_subject
from glyphforge.keywords.parser import parse_weighted_words
from glyphforge.typography.fonts import find_font, load_font
from glyphforge.typography.layout import LayoutStats, LayoutWord, generate_layout
from glyphforge.typography.themes import get_theme
from glyphforge.utils.seed import normalize_seed


@dataclass
class RenderResult:
    image: Image.Image
    layout: List[LayoutWord]
    layout_stats: LayoutStats
    metrics: Dict[str, float | int | str]


def _pick_text_color(gray_value: int, theme_name: str, subject_ratio_y: float, weight: float) -> Tuple[int, int, int]:
    theme = get_theme(theme_name)
    if theme_name == "sports_red_black":
        if subject_ratio_y > 0.58 and weight > 0.4:
            return theme.accent
    if theme_name == "gold_black_tribute":
        if weight > 0.72:
            return theme.accent

    if gray_value < 85:
        return theme.text_light
    if gray_value < 170:
        return theme.text_mid
    return theme.text_dark


def render_typographic_portrait(
    image_input,
    words_text: str,
    theme_name: str,
    ratio_label: str,
    density: float,
    seed: int | None,
    long_edge: int,
    min_font_size: int = 12,
    max_font_size: int = 64,
    attempts_per_word: int = 180,
    config: AppConfig | None = None,
) -> RenderResult:
    cfg = config or AppConfig()
    s = normalize_seed(seed, cfg.default_seed)
    start = perf_counter()

    prep = preprocess_portrait(image_input, ratio_label=ratio_label, long_edge=long_edge)
    raw_mask = segment_subject(prep.image_rgb, prep.gray)
    mask = cleanup_mask(raw_mask)

    words = parse_weighted_words(words_text)
    theme = get_theme(theme_name)
    canvas = Image.new("RGB", prep.canvas_size, color=theme.background)
    draw = ImageDraw.Draw(canvas)

    font_path = find_font(cfg.fonts_dir)

    def font_loader(size: int):
        return load_font(font_path, size)

    layout, stats = generate_layout(
        words=words,
        mask=mask,
        min_size=min_font_size,
        max_size=max_font_size,
        density=density,
        attempts_per_word=attempts_per_word,
        seed=s,
        font_loader=font_loader,
    )

    h, w = prep.gray.shape
    for item in layout:
        yy = min(h - 1, max(0, item.y))
        xx = min(w - 1, max(0, item.x))
        gv = int(prep.gray[yy, xx])
        color = _pick_text_color(gv, theme_name, yy / max(1, h), item.weight)
        font = font_loader(item.size)
        draw.text((item.x, item.y), item.word, fill=color, font=font)

    elapsed_ms = (perf_counter() - start) * 1000.0
    metrics: Dict[str, float | int | str] = {
        "theme": theme_name,
        "ratio": ratio_label,
        "seed": s,
        "render_ms": round(elapsed_ms, 2),
        "words_input": stats.words_input,
        "words_placed": stats.words_placed,
        "placement_attempts": stats.placement_attempts,
        "fill_ratio": round(stats.fill_ratio, 4),
        "resolution": f"{prep.canvas_size[0]}x{prep.canvas_size[1]}",
    }
    try:
        import torch

        if torch.cuda.is_available():
            metrics["gpu"] = torch.cuda.get_device_name(0)
            metrics["gpu_mem_alloc_mb"] = round(
                torch.cuda.memory_allocated(0) / (1024 * 1024), 2
            )
    except Exception:
        pass
    return RenderResult(image=canvas, layout=layout, layout_stats=stats, metrics=metrics)
