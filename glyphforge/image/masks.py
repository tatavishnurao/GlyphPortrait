from __future__ import annotations

import cv2
import numpy as np


def cleanup_mask(
    mask: np.ndarray, kernel_size: int = 5, blur_size: int = 7
) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    if blur_size > 0 and blur_size % 2 == 1:
        cleaned = cv2.GaussianBlur(cleaned, (blur_size, blur_size), 0)
    return (cleaned > 120).astype(np.uint8) * 255
