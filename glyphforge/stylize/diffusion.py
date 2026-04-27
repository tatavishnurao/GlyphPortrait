from __future__ import annotations

from PIL import Image


def stylize_with_diffusion(image: Image.Image, enabled: bool = False) -> Image.Image:
    """
    Placeholder for optional local SD/ControlNet integration.
    Returns input unchanged when disabled or unavailable.
    """
    _ = enabled
    return image
