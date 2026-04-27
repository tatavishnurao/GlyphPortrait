from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import ImageFont


def find_font(fonts_dir: Path, preferred_name: Optional[str] = None) -> Optional[Path]:
    if preferred_name:
        p = fonts_dir / preferred_name
        if p.exists():
            return p
    if not fonts_dir.exists():
        return None
    for ext in ("*.ttf", "*.otf", "*.ttc"):
        matches = sorted(fonts_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def load_font(font_path: Optional[Path], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except Exception:
            pass
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()
