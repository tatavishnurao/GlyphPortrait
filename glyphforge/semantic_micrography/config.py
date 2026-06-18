from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RGB = tuple[int, int, int]
BackgroundMode = Literal["black", "original", "transparent"]


@dataclass(frozen=True)
class CanvasConfig:
    width: int | None = None
    height: int | None = None
    long_edge: int = 1400
    preserve_aspect: bool = True


@dataclass(frozen=True)
class MicrographyStyleConfig:
    name: str = "tribute_dark"
    background_color: RGB = (0, 0, 0)
    font_family: str = "DejaVu Sans, Liberation Sans, Arial, sans-serif"
    default_font_size: int = 15
    hero_font_size: int = 26
    min_font_size: int = 9
    lane_spacing_px: float = 24.0
    min_lane_length_px: float = 80.0
    max_lane_curvature: float = 0.18
    filler_density: float = 1.0
    feature_lane_boost: float = 1.0
    edge_stroke: bool = True
    anchor_lane_count: int = 8
    anchor_min_length_px: float = 220.0
    anchor_font_scale: float = 1.55
    anchor_font_weight: int = 800
    anchor_letter_spacing: float = 0.9
    anchor_opacity: float = 0.98
    anchor_regions: tuple[str, ...] = (
        "feature_detail",
        "clothing_primary",
        "outline_or_edge",
        "skin_or_warm",
    )
    region_lane_styles: dict[str, dict[str, float | int]] = field(
        default_factory=lambda: {
            "dark_hair_or_shadow": {
                "spacing_scale": 0.78,
                "font_scale": 0.92,
                "letter_spacing": 0.14,
                "opacity": 0.92,
                "font_weight": 520,
            },
            "skin_or_warm": {
                "spacing_scale": 0.90,
                "font_scale": 1.00,
                "letter_spacing": 0.20,
                "opacity": 0.93,
                "font_weight": 540,
            },
            "clothing_primary": {
                "spacing_scale": 1.00,
                "font_scale": 1.20,
                "letter_spacing": 0.36,
                "opacity": 0.96,
                "font_weight": 660,
            },
            "clothing_secondary": {
                "spacing_scale": 0.94,
                "font_scale": 1.03,
                "letter_spacing": 0.24,
                "opacity": 0.92,
                "font_weight": 560,
            },
            "highlight": {
                "spacing_scale": 0.85,
                "font_scale": 0.92,
                "letter_spacing": 0.12,
                "opacity": 0.88,
                "font_weight": 500,
            },
            "outline_or_edge": {
                "spacing_scale": 0.60,
                "font_scale": 0.86,
                "letter_spacing": 0.10,
                "opacity": 0.90,
                "font_weight": 500,
            },
            "feature_detail": {
                "spacing_scale": 0.55,
                "font_scale": 1.22,
                "letter_spacing": 0.34,
                "opacity": 0.98,
                "font_weight": 720,
            },
        }
    )
    region_palettes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "subject": ["#d8d8d8"],
            "dark_hair_or_shadow": ["#f2f2f2", "#bfc4ca", "#8d949d"],
            "skin_or_warm": ["#ffe8c2", "#f3c894", "#d79a66"],
            "clothing_primary": ["#ff4a2d", "#d82024", "#ff9b44"],
            "clothing_secondary": ["#cfd6e8", "#9ea9bf", "#757f93"],
            "highlight": ["#fff5da", "#ffffff", "#e8e8e8"],
            "outline_or_edge": ["#ffffff", "#d0d6de", "#8c929a"],
            "feature_detail": ["#ffffff", "#f1d7ac", "#ff5b35"],
        }
    )


@dataclass(frozen=True)
class RegionConfig:
    min_component_area_px: int = 80
    morphology_kernel_px: int = 5
    edge_dilate_px: int = 3


@dataclass(frozen=True)
class RenderConfig:
    background: BackgroundMode = "black"
    style: str = "tribute_dark"
    write_debug_layers: bool = True
    rasterize_png: bool = True
    strict_mask_quality: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    style: MicrographyStyleConfig = field(default_factory=MicrographyStyleConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    seed: int = 20260527
    output_dir: Path = Path("outputs/semantic_micrography")


def style_config(name: str) -> MicrographyStyleConfig:
    if name != "tribute_dark":
        raise ValueError(f"Unknown semantic micrography style: {name}")
    return MicrographyStyleConfig(name=name)
