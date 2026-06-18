from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from glyphforge.semantic_micrography.config import RegionConfig
from glyphforge.semantic_micrography.preprocess import PreprocessResult

REGION_KEYS = [
    "subject",
    "dark_hair_or_shadow",
    "skin_or_warm",
    "clothing_primary",
    "clothing_secondary",
    "highlight",
    "outline_or_edge",
]


@dataclass(frozen=True)
class RegionResult:
    masks: dict[str, np.ndarray]
    diagnostics: dict[str, float | int]


def _binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def _clean(mask: np.ndarray, min_area: int, kernel_px: int) -> np.ndarray:
    binary = _binary(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, kernel_px), max(1, kernel_px)))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((cleaned > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(cleaned)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 255
    return out


def extract_regions(prep: PreprocessResult, config: RegionConfig | None = None) -> RegionResult:
    cfg = config or RegionConfig()
    rgb = prep.image_rgb
    subject = _binary(prep.subject_mask)
    subject_bool = subject > 0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[..., 0] / 179.0
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    lum = prep.luminance.astype(np.float32)
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    yy = np.indices(lum.shape)[0]
    ynorm = yy / max(lum.shape[0] - 1, 1)

    dark = subject_bool & ((lum < 82) | ((val < 0.32) & (sat > 0.10)))
    warm = subject_bool & (hue > 0.015) & (hue < 0.14) & (sat > 0.10) & (val > 0.28) & (r > b * 1.08)
    primary = subject_bool & (ynorm > 0.32) & (sat > 0.34) & (r > g * 1.10) & (r > b * 1.28)
    secondary = subject_bool & (ynorm > 0.38) & (sat > 0.20) & ~primary & (((b > r * 0.86) & (hue > 0.52)) | ((g > r * 0.86) & (hue > 0.20)))
    if np.any(subject_bool):
        subject_lum = lum[subject_bool]
        low_cut = float(np.percentile(subject_lum, 34))
        mid_cut = float(np.percentile(subject_lum, 48))
        high_cut = float(np.percentile(subject_lum, 78))
    else:
        low_cut = 70.0
        mid_cut = 110.0
        high_cut = 200.0

    subject_pixels = max(1, int(subject_bool.sum()))
    # Generic grayscale fallback: if cleaned color evidence is too fragmented,
    # use coarse silhouette zones so text lanes can fill regions instead of
    # tracing only sparse source marks.
    warm_fallback = int((_clean(warm, cfg.min_component_area_px, cfg.morphology_kernel_px) > 0).sum()) < int(subject_pixels * 0.035)
    primary_fallback = int((_clean(primary, cfg.min_component_area_px, cfg.morphology_kernel_px) > 0).sum()) < int(subject_pixels * 0.035)
    secondary_fallback = int((_clean(secondary, cfg.min_component_area_px, cfg.morphology_kernel_px) > 0).sum()) < int(subject_pixels * 0.015)
    if warm_fallback:
        warm = subject_bool & (ynorm < 0.64)
    if primary_fallback:
        primary = subject_bool & (ynorm >= 0.58)
    if secondary_fallback:
        secondary = subject_bool & (ynorm >= 0.42) & (ynorm < 0.78) & ~primary

    highlight = subject_bool & (lum >= high_cut) & ~primary
    edge = subject_bool & (prep.edge_map > 0)
    edge = cv2.dilate(edge.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.edge_dilate_px, cfg.edge_dilate_px)), iterations=1) > 0
    outline = edge | (dark & (prep.edge_map > 0))

    if not primary_fallback:
        primary = primary & ~dark
    if not secondary_fallback:
        secondary = secondary & ~dark
    if int((_clean(primary, cfg.min_component_area_px, cfg.morphology_kernel_px) > 0).sum()) < int(subject_pixels * 0.035):
        primary = subject_bool & (ynorm >= 0.58)
        primary_fallback = True
    secondary = secondary & ~primary
    warm = warm & ~primary & ~secondary
    highlight = highlight & ~dark

    masks = {
        "subject": subject,
        "dark_hair_or_shadow": _clean(dark, cfg.min_component_area_px, cfg.morphology_kernel_px),
        "skin_or_warm": _clean(warm, cfg.min_component_area_px, cfg.morphology_kernel_px),
        "clothing_primary": _clean(primary, cfg.min_component_area_px, cfg.morphology_kernel_px),
        "clothing_secondary": _clean(secondary, cfg.min_component_area_px, cfg.morphology_kernel_px),
        "highlight": _clean(highlight, max(30, cfg.min_component_area_px // 2), 3),
        "outline_or_edge": _clean(outline, max(20, cfg.min_component_area_px // 3), 3),
    }
    diagnostics: dict[str, float | int] = {
        f"{key}_pixels": int((mask > 0).sum()) for key, mask in masks.items()
    }
    diagnostics["subject_coverage"] = float((masks["subject"] > 0).mean())
    return RegionResult(masks=masks, diagnostics=diagnostics)

def save_regions_panel(
    region_result: RegionResult,
    edge_map: np.ndarray,
    out_path: Path,
    mask_source: str = "auto",
    mask_quality: dict[str, object] | None = None,
) -> None:
    masks = region_result.masks
    names = REGION_KEYS + ["edge_map"]
    arrays = {**masks, "edge_map": _binary(edge_map)}
    h, w = next(iter(masks.values())).shape
    thumb_w = 260
    thumb_h = max(1, int(round(h * thumb_w / w)))
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    panel = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 30)), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    coverage = None
    if mask_quality:
        raw_coverage = mask_quality.get("subject_coverage")
        if isinstance(raw_coverage, (int, float)):
            coverage = float(raw_coverage)
    for idx, name in enumerate(names):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + 30)
        img = Image.fromarray(arrays[name].astype(np.uint8), "L").resize((thumb_w, thumb_h), Image.Resampling.NEAREST).convert("RGB")
        panel.paste(img, (x, y + 30))
        label = name
        if name == "subject":
            coverage_text = f" coverage={coverage:.3f}" if coverage is not None else ""
            label = f"subject ({mask_source}){coverage_text}"
        draw.text((x + 8, y + 8), label, fill=(245, 245, 245))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)
