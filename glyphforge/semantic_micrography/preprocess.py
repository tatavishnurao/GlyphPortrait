from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from glyphforge.image.masks import cleanup_mask
from glyphforge.image.segment import segment_subject
from glyphforge.semantic_micrography.config import BackgroundMode, CanvasConfig


@dataclass(frozen=True)
class PreprocessResult:
    image_rgb: np.ndarray
    subject_mask: np.ndarray
    luminance: np.ndarray
    edge_map: np.ndarray
    background_mode: BackgroundMode
    canvas_size: tuple[int, int]


def _resize_to_canvas(image: np.ndarray, config: CanvasConfig) -> np.ndarray:
    h, w = image.shape[:2]
    if config.width and config.height:
        return cv2.resize(image, (config.width, config.height), interpolation=cv2.INTER_AREA)
    if not config.preserve_aspect:
        return cv2.resize(image, (config.long_edge, config.long_edge), interpolation=cv2.INTER_AREA)
    long_edge = max(1, int(config.long_edge))
    scale = min(1.0, long_edge / max(h, w))
    if scale == 1.0:
        return image.copy()
    return cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L").resize((shape[1], shape[0]), Image.Resampling.NEAREST))
    return (mask > 127).astype(np.uint8) * 255


def _largest_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    areas: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            areas.append((area, label))
    for _, label in sorted(areas, reverse=True)[:4]:
        out[labels == label] = 255
    return out


def _dominant_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary * 255
    best_label = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for label in range(2, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label
    out = np.zeros_like(binary)
    out[labels == best_label] = 255
    return out


def _black_background_subject_hint(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    nonblack = ((gray > 14) | ((val > 26) & (sat > 28))).astype(np.uint8) * 255
    close_size = max(9, int(round(min(rgb.shape[:2]) * 0.028)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    nonblack = cv2.morphologyEx(nonblack, cv2.MORPH_CLOSE, kernel, iterations=2)
    nonblack = cv2.morphologyEx(nonblack, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    min_area = max(80, int(rgb.shape[0] * rgb.shape[1] * 0.001))
    components = _largest_components(nonblack, min_area=min_area)
    contours, _ = cv2.findContours(components, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(components)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filled, [contour], contourIdx=-1, color=255, thickness=-1)
    return filled if int((filled > 0).sum()) else components


def load_and_preprocess(
    input_path: Path,
    canvas: CanvasConfig,
    background: BackgroundMode = "black",
    mask_path: Path | None = None,
) -> PreprocessResult:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    rgb = np.array(Image.open(input_path).convert("RGB"), dtype=np.uint8)
    rgb = _resize_to_canvas(rgb, canvas)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_equalized = cv2.equalizeHist(gray)
    if mask_path is None:
        raw_mask = segment_subject(rgb, gray_equalized)
        if background == "black":
            bg_hint = _black_background_subject_hint(rgb, gray)
            raw_area = int((raw_mask > 0).sum())
            hint_area = int((bg_hint > 0).sum())
            if hint_area > raw_area * 1.20:
                raw_mask = bg_hint
    else:
        raw_mask = _load_mask(mask_path, rgb.shape[:2])
    subject_mask = cleanup_mask(raw_mask, kernel_size=5, blur_size=7)
    subject_mask = _dominant_component(subject_mask)
    edges = cv2.Canny(gray_equalized, threshold1=70, threshold2=150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    h, w = rgb.shape[:2]
    return PreprocessResult(
        image_rgb=rgb,
        subject_mask=subject_mask,
        luminance=gray,
        edge_map=edges,
        background_mode=background,
        canvas_size=(w, h),
    )
