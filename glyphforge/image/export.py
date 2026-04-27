from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image


def save_png(
    image: Image.Image,
    output_path: Path | str,
    dpi: Optional[int] = 300,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"format": "PNG"}
    if dpi:
        kwargs["dpi"] = (dpi, dpi)
    image.save(path, **kwargs)
    return path
