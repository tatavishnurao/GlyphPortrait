from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:5": (4, 5),
    "16:9": (16, 9),
    "9:16": (9, 16),
}


@dataclass
class PreprocessResult:
    image_rgb: np.ndarray
    gray: np.ndarray
    canvas_size: Tuple[int, int]


def _fit_to_ratio(
    image_rgb: np.ndarray, ratio_label: str, long_edge: int = 1600
) -> np.ndarray:
    if ratio_label not in ASPECT_RATIOS:
        ratio_label = "4:5"
    rw, rh = ASPECT_RATIOS[ratio_label]

    src_h, src_w = image_rgb.shape[:2]
    if src_w >= src_h:
        target_w = long_edge
        target_h = int(round(target_w * rh / rw))
    else:
        target_h = long_edge
        target_w = int(round(target_h * rw / rh))

    scale = max(target_w / src_w, target_h / src_h)
    resized = cv2.resize(
        image_rgb,
        (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    h, w = resized.shape[:2]
    y0 = max(0, (h - target_h) // 2)
    x0 = max(0, (w - target_w) // 2)
    return resized[y0 : y0 + target_h, x0 : x0 + target_w]


def preprocess_portrait(
    image: Image.Image | np.ndarray, ratio_label: str, long_edge: int
) -> PreprocessResult:
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = image[..., :3].copy()

    rgb = _fit_to_ratio(rgb, ratio_label=ratio_label, long_edge=long_edge)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = rgb.shape[:2]
    return PreprocessResult(image_rgb=rgb, gray=gray, canvas_size=(w, h))
