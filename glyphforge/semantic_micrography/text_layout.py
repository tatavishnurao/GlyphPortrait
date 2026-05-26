from __future__ import annotations

from dataclasses import dataclass

from glyphforge.semantic_micrography.config import MicrographyStyleConfig
from glyphforge.semantic_micrography.lanes import TextLane
from glyphforge.semantic_micrography.profiles import WordProfile


@dataclass(frozen=True)
class TextPathLayout:
    lane: TextLane
    text: str
    font_size: int
    fill: str
    is_hero: bool = False


@dataclass(frozen=True)
class TextLayoutResult:
    text_paths: list[TextPathLayout]
    used_words: dict[str, int]
    coverage: dict[str, float | int]


def _palette_color(style: MicrographyStyleConfig, region: str, index: int) -> str:
    palette = style.region_palettes.get(region) or style.region_palettes["subject"]
    return palette[index % len(palette)]


def _font_size(style: MicrographyStyleConfig, lane: TextLane, profile: WordProfile) -> tuple[int, bool]:
    hero_threshold = max(180.0, style.min_lane_length_px * 2.2)
    if lane.length_px >= hero_threshold and lane.order_index % 7 == 0 and profile.hero_words:
        return style.hero_font_size, True
    size = style.default_font_size
    if lane.region == "outline_or_edge":
        size = max(style.min_font_size, size - 3)
    elif lane.region == "highlight":
        size += 1
    elif lane.region == "clothing_primary":
        size += 2
    return max(style.min_font_size, size), False


def _text_for_lane(words: list[str], length_px: float, font_size: int, start: int) -> tuple[str, dict[str, int]]:
    if not words:
        words = ["Legacy"]
    avg_char_px = max(4.0, font_size * 0.55)
    target_chars = max(1, int(length_px / avg_char_px))
    pieces: list[str] = []
    counts: dict[str, int] = {}
    idx = start
    while len("  ".join(pieces)) < target_chars:
        word = words[idx % len(words)]
        pieces.append(word)
        counts[word] = counts.get(word, 0) + 1
        idx += 1
        if len(pieces) > 80:
            break
    return "  ".join(pieces), counts


def assign_text_to_lanes(
    lanes: list[TextLane],
    profile: WordProfile,
    style: MicrographyStyleConfig,
) -> TextLayoutResult:
    text_paths: list[TextPathLayout] = []
    used_words: dict[str, int] = {}
    region_cursors: dict[str, int] = {}
    for lane in lanes:
        font_size, is_hero = _font_size(style, lane, profile)
        words = profile.hero_words if is_hero else profile.words_for_region(lane.region)
        cursor = region_cursors.get(lane.region, lane.order_index)
        text, counts = _text_for_lane(words, lane.length_px, font_size, cursor)
        region_cursors[lane.region] = cursor + max(1, len(counts))
        for word, count in counts.items():
            used_words[word] = used_words.get(word, 0) + count
        text_paths.append(
            TextPathLayout(
                lane=lane,
                text=text,
                font_size=font_size,
                fill=_palette_color(style, lane.region, lane.order_index),
                is_hero=is_hero,
            )
        )
    total_chars = sum(len(item.text) for item in text_paths)
    return TextLayoutResult(
        text_paths=text_paths,
        used_words=used_words,
        coverage={
            "text_path_count": len(text_paths),
            "used_word_count": sum(used_words.values()),
            "total_text_chars": total_chars,
            "hero_path_count": sum(1 for item in text_paths if item.is_hero),
        },
    )
