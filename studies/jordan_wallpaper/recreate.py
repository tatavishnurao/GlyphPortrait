from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import cv2
import numpy as np
from PIL import Image

from glyphforge.config import AppConfig
from glyphforge.typography.fonts import find_font, load_font

from .anchors import JORDAN_REFERENCE_ANCHORS
from .composition import compose_right_subject_canvas, to_pil
from .metrics import (
    build_metrics,
    edge_overlap_score,
    luminance_mae_inside_mask,
    mask_iou,
)
from .regions import ReferenceRegions, build_reference_regions
from .target_analysis import extract_nonblack_subject_mask, extract_red_jersey_mask
from .typography_passes import render_anchor_pass, render_slogan, render_text_pass


@dataclass
class ReferenceRenderResult:
    image: Image.Image
    regions: ReferenceRegions
    target_rgb: np.ndarray
    metrics: Dict[str, float | int | str]


def _build_importance(
    luminance: np.ndarray, edge_map: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    darkness = 1.0 - np.clip(luminance, 0.0, 1.0)
    importance = (0.65 * darkness) + (0.35 * np.clip(edge_map, 0.0, 1.0)) + 0.02
    importance *= (mask > 0).astype(np.float32)
    return cv2.GaussianBlur(importance.astype(np.float32), (0, 0), sigmaX=1.3)


def render_reference_jordan_poster(
    target_image: Image.Image,
    words_text: str,
    output_size: tuple[int, int] = (1920, 1080),
    seed: int = 23,
    slogan: str = "change the game.",
    include_structure_pass: bool = True,
    include_jersey_pass: bool = True,
    include_anchor_pass: bool = True,
    include_slogan_pass: bool = True,
    config: AppConfig | None = None,
) -> ReferenceRenderResult:
    cfg = config or AppConfig()
    t0 = perf_counter()

    target_rgb = np.array(target_image.convert("RGB"))
    target_regions = build_reference_regions(target_rgb)
    _, regions = compose_right_subject_canvas(
        target_rgb, target_regions, output_size=output_size
    )
    canvas = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    output = to_pil(canvas)

    font_path = find_font(cfg.fonts_dir)

    def font_loader(size: int):
        return load_font(font_path, size)

    occupancy = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)
    base_importance = _build_importance(
        regions.luminance_map,
        regions.edge_map,
        regions.subject_mask,
    )

    def face_color(item):
        y = min(regions.luminance_map.shape[0] - 1, max(0, item.y))
        x = min(regions.luminance_map.shape[1] - 1, max(0, item.x))
        lum = float(regions.luminance_map[y, x])
        v = int(145 + (1.0 - lum) * 105)
        return (v, v, v)

    face_words, face_stats = render_text_pass(
        image=output,
        occupancy_mask=occupancy,
        region_mask=regions.face_mask,
        importance_map=base_importance,
        words_text=words_text,
        font_loader=font_loader,
        color_picker=face_color,
        min_size=6,
        max_size=13,
        density=0.58,
        attempts_per_word=220,
        seed=seed,
    )
    if include_structure_pass:
        _structure_words, _ = render_text_pass(
            image=output,
            occupancy_mask=occupancy,
            region_mask=regions.face_mask,
            importance_map=base_importance,
            words_text="MVP,CLUTCH,LEGEND,GOAT,DEFENSE,CHAMPION,WINNER,FOCUS",
            font_loader=font_loader,
            color_picker=lambda _item: (225, 225, 225),
            min_size=14,
            max_size=32,
            density=0.42,
            attempts_per_word=160,
            seed=seed + 77,
        )

    def jersey_color(item):
        y = min(regions.luminance_map.shape[0] - 1, max(0, item.y))
        x = min(regions.luminance_map.shape[1] - 1, max(0, item.x))
        lum = float(regions.luminance_map[y, x])
        if lum < 0.45:
            return (145, 25, 30)
        if lum < 0.65:
            return (200, 26, 36)
        return (235, 235, 235)

    jersey_stats = type(face_stats)(placed=0, attempts=0)
    if include_jersey_pass:
        _jersey_words, jersey_stats = render_text_pass(
            image=output,
            occupancy_mask=occupancy,
            region_mask=regions.jersey_mask,
            importance_map=base_importance,
            words_text="BULLS,23,CHICAGO,FINALS,DYNASTY,SIX RINGS,AIR,WIN",
            font_loader=font_loader,
            color_picker=jersey_color,
            min_size=12,
            max_size=36,
            density=0.55,
            attempts_per_word=200,
            seed=seed + 33,
        )

    anchors_placed = 0
    if include_anchor_pass:
        anchors_placed = render_anchor_pass(
            image=output,
            anchors=JORDAN_REFERENCE_ANCHORS,
            font_loader=font_loader,
            width=output_size[0],
            height=output_size[1],
        )
    if include_slogan_pass:
        render_slogan(
            output,
            text=slogan,
            font_loader=font_loader,
            canvas_w=output_size[0],
            canvas_h=output_size[1],
        )

    out_rgb = np.array(output.convert("RGB"))
    out_lum = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    out_edge = (
        cv2.Canny(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2GRAY), 60, 140).astype(np.float32)
        / 255.0
    )

    output_subject = extract_nonblack_subject_mask(out_rgb)
    output_jersey = extract_red_jersey_mask(out_rgb, output_subject)

    metrics = build_metrics(
        {
            "subject_iou": round(mask_iou(regions.subject_mask, output_subject), 4),
            "jersey_iou": round(mask_iou(regions.jersey_mask, output_jersey), 4),
            "luminance_mae_subject": round(
                luminance_mae_inside_mask(
                    regions.luminance_map, out_lum, regions.subject_mask
                ),
                4,
            ),
            "edge_overlap": round(edge_overlap_score(regions.edge_map, out_edge), 4),
            "face_words_placed": face_stats.placed,
            "jersey_words_placed": jersey_stats.placed,
            "anchors_placed": anchors_placed,
            "placement_attempts_face": face_stats.attempts,
            "placement_attempts_jersey": jersey_stats.attempts,
            "render_ms": round((perf_counter() - t0) * 1000.0, 2),
            "output_resolution": f"{output_size[0]}x{output_size[1]}",
        }
    )

    return ReferenceRenderResult(
        image=output,
        regions=regions,
        target_rgb=target_rgb,
        metrics=metrics,
    )
