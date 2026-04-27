from __future__ import annotations

from PIL import Image


def upscale_image(image: Image.Image, scale: int = 1) -> Image.Image:
    if scale <= 1:
        return image
    w, h = image.size
    return image.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
