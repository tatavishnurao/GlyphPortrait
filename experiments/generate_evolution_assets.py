from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from glyphforge.config import AppConfig
from glyphforge.image.masks import cleanup_mask
from glyphforge.image.preprocess import preprocess_portrait
from glyphforge.image.segment import segment_subject
from glyphforge.keywords.parser import parse_weighted_words
from glyphforge.typography.fonts import find_font, load_font
from glyphforge.typography.layout import generate_layout
from glyphforge.typography.render import _build_importance_map, _pick_text_color
from glyphforge.typography.themes import get_theme


WORDS_TEXT = (
    "MVP, GOAT, ICON, GRIT, FOCUS, DRIVE, LEAD, CLUTCH, LEGACY, WINNER, "
    "CHAMPION, DISCIPLINE, RESILIENCE, VISION, POWER, AIR, FINALS, DEFENSE, "
    "RINGS, PRESSURE, KILLER, BULLS, MVP, GOAT, ICON, GRIT, FOCUS, DRIVE"
)


def _render_layout(prep, mask, importance_map, theme_name: str, output_path: Path) -> None:
    cfg = AppConfig()
    words = parse_weighted_words(WORDS_TEXT)
    theme = get_theme(theme_name)
    canvas = Image.new("RGB", prep.canvas_size, color=theme.background)
    draw = ImageDraw.Draw(canvas)

    font_path = find_font(cfg.fonts_dir)

    def font_loader(size: int):
        return load_font(font_path, size)

    layout, _stats = generate_layout(
        words=words,
        mask=mask,
        min_size=8,
        max_size=24,
        density=0.62,
        attempts_per_word=260,
        seed=42,
        font_loader=font_loader,
        importance_map=importance_map,
    )

    h, w = prep.gray.shape
    for item in layout:
        yy = min(h - 1, max(0, item.y))
        xx = min(w - 1, max(0, item.x))
        gray_value = int(prep.gray[yy, xx])
        color = _pick_text_color(gray_value, theme_name, yy / max(1, h), item.weight)
        draw.text((item.x, item.y), item.word, fill=color, font=font_loader(item.size))

    canvas.save(output_path)


def main() -> None:
    out_dir = Path("examples/evolution")
    out_dir.mkdir(parents=True, exist_ok=True)

    source = Image.open("reference_img/Michael-Jordan-Wallpaper-Desktop-1.jpg").convert(
        "RGB"
    )
    prep = preprocess_portrait(source, ratio_label="16:9", long_edge=960)
    raw_mask = segment_subject(prep.image_rgb, prep.gray)
    mask = cleanup_mask(raw_mask)
    importance_map = _build_importance_map(prep.gray, mask)

    Image.fromarray(prep.image_rgb, mode="RGB").save(out_dir / "00_original_crop.png")
    Image.fromarray(mask, mode="L").save(out_dir / "01_mask.png")

    uniform_map = np.where(mask > 0, 1.0, 0.0).astype(np.float32)
    _render_layout(
        prep=prep,
        mask=mask,
        importance_map=uniform_map,
        theme_name="monochrome_dark",
        output_path=out_dir / "02_uniform_sampling_bad.png",
    )
    _render_layout(
        prep=prep,
        mask=mask,
        importance_map=importance_map,
        theme_name="monochrome_dark",
        output_path=out_dir / "03_importance_sampling_better.png",
    )
    _render_layout(
        prep=prep,
        mask=mask,
        importance_map=importance_map,
        theme_name="sports_red_black",
        output_path=out_dir / "04_reference_sports_attempt.png",
    )
    norm = (importance_map / max(1e-6, float(importance_map.max())) * 255.0).astype(
        np.uint8
    )
    Image.fromarray(norm, mode="L").save(out_dir / "05_importance_map.png")


if __name__ == "__main__":
    main()
