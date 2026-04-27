from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    name: str
    background: RGB
    text_dark: RGB
    text_mid: RGB
    text_light: RGB
    accent: RGB


THEMES: Dict[str, Theme] = {
    "monochrome_dark": Theme(
        name="monochrome_dark",
        background=(8, 10, 14),
        text_dark=(90, 96, 110),
        text_mid=(170, 176, 190),
        text_light=(230, 235, 240),
        accent=(190, 195, 210),
    ),
    "minimal_grayscale": Theme(
        name="minimal_grayscale",
        background=(242, 242, 242),
        text_dark=(35, 35, 35),
        text_mid=(95, 95, 95),
        text_light=(180, 180, 180),
        accent=(50, 50, 50),
    ),
    "sports_red_black": Theme(
        name="sports_red_black",
        background=(5, 5, 7),
        text_dark=(120, 18, 24),
        text_mid=(180, 28, 35),
        text_light=(235, 235, 235),
        accent=(220, 28, 36),
    ),
    "gold_black_tribute": Theme(
        name="gold_black_tribute",
        background=(10, 8, 4),
        text_dark=(120, 92, 28),
        text_mid=(180, 145, 45),
        text_light=(244, 226, 160),
        accent=(212, 175, 55),
    ),
}


def get_theme(name: str) -> Theme:
    return THEMES.get(name, THEMES["monochrome_dark"])
