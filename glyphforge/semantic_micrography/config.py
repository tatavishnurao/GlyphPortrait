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
    edge_stroke: bool = True
    region_palettes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "subject": ["#d8d8d8"],
            "dark_hair_or_shadow": ["#f2f2f2", "#bfc4ca", "#8d949d"],
            "skin_or_warm": ["#ffe8c2", "#f3c894", "#d79a66"],
            "clothing_primary": ["#ff4a2d", "#d82024", "#ff9b44"],
            "clothing_secondary": ["#cfd6e8", "#9ea9bf", "#757f93"],
            "highlight": ["#fff5da", "#ffffff", "#e8e8e8"],
            "outline_or_edge": ["#ffffff", "#d0d6de", "#8c929a"],
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
