from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from glyphforge.image.masks import cleanup_mask


def extract_nonblack_subject_mask(
    rgb: np.ndarray, black_threshold: int = 22
) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > black_threshold).astype(np.uint8) * 255
    return cleanup_mask(mask, kernel_size=5, blur_size=5)


def extract_red_jersey_mask(
    rgb: np.ndarray, subject_mask: np.ndarray, min_delta: int = 30, min_red: int = 80
) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    red_dominant = (r > g + min_delta) & (r > b + min_delta) & (r > min_red)
    mask = red_dominant & (subject_mask > 0)
    return cleanup_mask(mask.astype(np.uint8) * 255, kernel_size=5, blur_size=3)


def extract_face_body_mask(
    subject_mask: np.ndarray, jersey_mask: np.ndarray
) -> np.ndarray:
    face = ((subject_mask > 0) & (jersey_mask == 0)).astype(np.uint8) * 255
    return cleanup_mask(face, kernel_size=3, blur_size=3)


def build_target_luminance_map(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def build_target_edge_map(gray_u8: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray_u8, 60, 140).astype(np.float32) / 255.0
    return cv2.GaussianBlur(edges, (0, 0), sigmaX=1.2)


def find_subject_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 + 1, y1 + 1)
