from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from glyphforge.config import AppConfig
from glyphforge.image.export import save_png
from glyphforge.typography.render import render_typographic_portrait


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GlyphForge CLI")
    p.add_argument("--input", type=Path, required=True, help="Input portrait image")
    p.add_argument(
        "--words",
        type=str,
        required=True,
        help="Comma/newline separated words",
    )
    p.add_argument("--theme", type=str, default="monochrome_dark")
    p.add_argument(
        "--ratio",
        type=str,
        default="4:5",
        choices=["1:1", "4:5", "16:9", "9:16"],
    )
    p.add_argument("--density", type=float, default=0.65)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--long-edge", type=int, default=1600)
    p.add_argument("--min-font-size", type=int, default=12)
    p.add_argument("--max-font-size", type=int, default=64)
    p.add_argument("--attempts-per-word", type=int, default=180)
    p.add_argument("--output", type=Path, required=True, help="Output PNG path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    img = Image.open(args.input).convert("RGB")
    result = render_typographic_portrait(
        image_input=img,
        words_text=args.words,
        theme_name=args.theme,
        ratio_label=args.ratio,
        density=args.density,
        seed=args.seed,
        long_edge=args.long_edge,
        min_font_size=args.min_font_size,
        max_font_size=args.max_font_size,
        attempts_per_word=args.attempts_per_word,
        config=AppConfig(),
    )
    save_png(result.image, args.output, dpi=300)
    print(json.dumps(result.metrics, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
