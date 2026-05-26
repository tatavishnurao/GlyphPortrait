from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from glyphforge.semantic_micrography.config import PipelineConfig, RenderConfig, style_config
from glyphforge.semantic_micrography.lanes import generate_lanes, gate_lanes_to_subject
from glyphforge.semantic_micrography.metrics import compute_micrography_metrics
from glyphforge.semantic_micrography.ordering import order_lanes
from glyphforge.semantic_micrography.preprocess import load_and_preprocess
from glyphforge.semantic_micrography.profiles import WordProfile
from glyphforge.semantic_micrography.rasterize import (
    rasterize_lane_overlay_preview,
    rasterize_layout_preview,
    rasterize_svg,
)
from glyphforge.semantic_micrography.regions import extract_regions, save_regions_panel
from glyphforge.semantic_micrography.render_svg import render_lane_overlay_svg, render_master_svg
from glyphforge.semantic_micrography.text_layout import assign_text_to_lanes
from glyphforge.semantic_micrography.vectorize import vectorize_regions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _dominant_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)
    best_label = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for label in range(2, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label
    return labels == best_label


def _cleanup_text_islands(layout, subject_mask: np.ndarray):
    dominant = _dominant_component(subject_mask)
    kept = []
    dropped = 0
    for item in layout.text_paths:
        points = item.lane.points
        if not points:
            dropped += 1
            continue
        inside = 0
        total = 0
        for x, y in points:
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= iy < dominant.shape[0] and 0 <= ix < dominant.shape[1]:
                total += 1
                if dominant[iy, ix]:
                    inside += 1
        if (inside / max(total, 1)) < 0.5:
            dropped += 1
            continue
        kept.append(item)
    return type(layout)(
        text_paths=kept,
        used_words=layout.used_words,
        coverage={
            **layout.coverage,
            "text_path_count": len(kept),
            "dropped_text_island_groups": dropped,
        },
    )


def run_pipeline(
    input_path: Path,
    profile: WordProfile,
    out_dir: Path,
    config: PipelineConfig | None = None,
    mask_path: Path | None = None,
) -> dict[str, Any]:
    start = perf_counter()
    cfg = config or PipelineConfig(output_dir=out_dir)
    style = style_config(cfg.render.style)
    if cfg.style.name == style.name:
        style = cfg.style
    render_cfg = cfg.render
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = load_and_preprocess(input_path, cfg.canvas, background=render_cfg.background, mask_path=mask_path)
    region_result = extract_regions(prep, cfg.regions)
    contours = vectorize_regions(region_result.masks)
    lanes, lane_diagnostics = generate_lanes(region_result.masks, contours, style)
    gated_lanes, gate_stats = gate_lanes_to_subject(lanes, region_result.masks["subject"])
    ordered_lanes = order_lanes(gated_lanes)
    layout = assign_text_to_lanes(ordered_lanes, profile, style)
    layout = _cleanup_text_islands(layout, region_result.masks["subject"])

    svg_path = render_master_svg(layout, contours, prep.canvas_size, style, out_dir / "current_best.svg")
    overlay_svg_path = render_lane_overlay_svg(layout, prep.canvas_size, out_dir / "lane_overlay.svg")
    save_regions_panel(region_result, prep.edge_map, out_dir / "regions_panel.png")

    rasterizer = "cairosvg"
    try:
        rasterize_svg(svg_path, out_dir / "current_best.png")
    except RuntimeError:
        rasterizer = "pillow_fallback"
        rasterize_layout_preview(layout, prep.canvas_size, style, out_dir / "current_best.png")
    try:
        rasterize_svg(overlay_svg_path, out_dir / "lane_overlay.png")
    except RuntimeError:
        rasterize_lane_overlay_preview(layout, prep.canvas_size, out_dir / "lane_overlay.png")

    metrics = compute_micrography_metrics(
        region_result.masks,
        ordered_lanes,
        layout,
        perf_counter() - start,
        prep.canvas_size,
        lane_diagnostics,
    )
    metrics["rasterizer"] = rasterizer
    metrics["style"] = style.name
    metrics["background"] = render_cfg.background
    metrics["subject"] = profile.subject_name
    _write_json(out_dir / "current_best_metrics.json", metrics)

    debug_summary = {
        "input": str(input_path),
        "subject": profile.subject_name,
        "canvas_size": list(prep.canvas_size),
        "regions": region_result.diagnostics,
        "lanes": lane_diagnostics,
        "lane_gating": gate_stats,
        "text_layout": layout.coverage,
        "rasterizer": rasterizer,
        "outputs": {
            "svg": "current_best.svg",
            "png": "current_best.png",
            "metrics": "current_best_metrics.json",
            "regions_panel": "regions_panel.png",
            "lane_overlay_svg": "lane_overlay.svg",
            "lane_overlay_png": "lane_overlay.png",
        },
        "note": "Practical raster approximation of Digital Micrography; SVG textPath is the master representation.",
    }
    _write_json(out_dir / "debug_summary.json", debug_summary)
    return metrics
