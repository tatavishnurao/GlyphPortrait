from __future__ import annotations

import importlib.util

import cv2
import numpy as np


def _segment_with_rembg(image_rgb: np.ndarray) -> np.ndarray | None:
    if importlib.util.find_spec("onnxruntime") is None:
        return None
    try:
        from rembg import remove
    except BaseException:
        return None

    try:
        rgba = remove(image_rgb)
        if rgba is None:
            return None
        alpha = rgba[..., 3] if rgba.ndim == 3 and rgba.shape[2] == 4 else None
        if alpha is None:
            return None
        return (alpha > 10).astype(np.uint8) * 255
    except BaseException:
        return None


def _segment_with_grabcut(image_rgb: np.ndarray) -> np.ndarray | None:
    try:
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        rect = (
            max(1, w // 12),
            max(1, h // 12),
            max(2, w - (w // 6)),
            max(2, h - (h // 6)),
        )
        cv2.grabCut(image_rgb, mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
        out = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        return out
    except Exception:
        return None


def _segment_with_threshold(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m1 if m1.sum() < m2.sum() else m2


def segment_subject(image_rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    for fn in (_segment_with_rembg, _segment_with_grabcut):
        mask = fn(image_rgb)
        if mask is not None and mask.sum() > 0:
            return mask
    return _segment_with_threshold(gray)
