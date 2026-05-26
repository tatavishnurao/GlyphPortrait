from __future__ import annotations

from dataclasses import dataclass
import math

from glyphforge.semantic_micrography.config import MicrographyStyleConfig
from glyphforge.semantic_micrography.lanes import TextLane
from glyphforge.semantic_micrography.profiles import WordProfile


@dataclass(frozen=True)
class TextPathLayout:
    lane: TextLane
    text: str
    font_size: int
    fill: str
    font_weight: int
    opacity: float
    letter_spacing: float
    start_offset: str
    is_hero: bool = False
    is_anchor: bool = False


@dataclass(frozen=True)
class TextLayoutResult:
    text_paths: list[TextPathLayout]
    used_words: dict[str, int]
    coverage: dict[str, float | int]


def _palette_color(style: MicrographyStyleConfig, region: str, index: int) -> str:
    palette = style.region_palettes.get(region) or style.region_palettes["subject"]
    return palette[index % len(palette)]


def _region_style(style: MicrographyStyleConfig, region: str) -> dict[str, float | int]:
    return style.region_lane_styles.get(region, {})


def _lane_centroid(lane: TextLane) -> tuple[float, float]:
    if not lane.points:
        return 0.0, 0.0
    xs = [p[0] for p in lane.points]
    ys = [p[1] for p in lane.points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _select_anchor_lanes(lanes: list[TextLane], style: MicrographyStyleConfig) -> set[str]:
    if not lanes or style.anchor_lane_count <= 0:
        return set()
    xs = [point[0] for lane in lanes for point in lane.points]
    ys = [point[1] for lane in lanes for point in lane.points]
    if not xs or not ys:
        return set()
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_diag = math.hypot(max_x - min_x, max_y - min_y)
    min_sep = max(42.0, bbox_diag * 0.11)
    region_priority = {
        "clothing_primary": 1.6,
        "outline_or_edge": 1.35,
        "skin_or_warm": 1.25,
        "dark_hair_or_shadow": 1.0,
    }
    ranked: list[tuple[float, TextLane, tuple[float, float]]] = []
    for lane in lanes:
        if lane.region not in style.anchor_regions:
            continue
        if lane.length_px < style.anchor_min_length_px:
            continue
        cx, cy = _lane_centroid(lane)
        y_norm = (cy - min_y) / max(max_y - min_y, 1.0)
        centrality = 1.0 - min(1.0, abs(y_norm - 0.58))
        score = lane.length_px * region_priority.get(lane.region, 1.0) * (1.0 + centrality * 0.45)
        ranked.append((score, lane, (cx, cy)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[float, float]] = []
    out: set[str] = set()
    for _, lane, center in ranked:
        if any(math.hypot(center[0] - sx, center[1] - sy) < min_sep for sx, sy in selected):
            continue
        out.add(lane.id)
        selected.append(center)
        if len(out) >= style.anchor_lane_count:
            break
    return out


def _fit_text_to_lane(words: list[str], length_px: float, font_size: int, start: int) -> tuple[str, dict[str, int]]:
    if not words:
        words = ["Legacy"]
    avg_char_px = max(4.0, font_size * 0.57)
    target_chars = max(8, int(length_px / avg_char_px))
    pieces: list[str] = []
    counts: dict[str, int] = {}
    idx = start
    max_pieces = max(4, min(26, int(target_chars / 4)))
    while len("  ".join(pieces)) < target_chars and len(pieces) < max_pieces:
        word = words[idx % len(words)]
        pieces.append(word)
        counts[word] = counts.get(word, 0) + 1
        idx += 1
    if not pieces:
        pieces = [words[start % len(words)]]
    return "  ".join(pieces), counts


def _anchor_text(anchor_words: list[str], lane: TextLane, cursor: int) -> tuple[str, dict[str, int]]:
    if not anchor_words:
        anchor_words = ["CHAMPION"]
    phrase = anchor_words[cursor % len(anchor_words)]
    counts = {phrase: 1}
    if lane.length_px > 440:
        text = f"{phrase}   {phrase}"
        counts[phrase] = 2
    else:
        text = phrase
    return text, counts


def assign_text_to_lanes(
    lanes: list[TextLane],
    profile: WordProfile,
    style: MicrographyStyleConfig,
) -> TextLayoutResult:
    text_paths: list[TextPathLayout] = []
    used_words: dict[str, int] = {}
    region_cursors: dict[str, int] = {}
    anchor_cursor = 0
    anchor_lane_ids = _select_anchor_lanes(lanes, style)
    for lane in lanes:
        is_anchor = lane.id in anchor_lane_ids and bool(profile.anchor_words)
        region_style = _region_style(style, lane.region)
        base_scale = float(region_style.get("font_scale", 1.0))
        font_size = int(round(style.default_font_size * base_scale))
        font_size = max(style.min_font_size, font_size)
        font_weight = int(region_style.get("font_weight", 520))
        opacity = float(region_style.get("opacity", 0.92))
        letter_spacing = float(region_style.get("letter_spacing", 0.18))
        start_offset = "0%"

        hero_threshold = max(230.0, style.min_lane_length_px * 2.6)
        is_hero = lane.length_px >= hero_threshold and (lane.order_index % 10 == 0)
        if lane.region == "outline_or_edge":
            font_size = max(style.min_font_size, font_size - 1)
        if is_anchor:
            font_size = max(
                style.hero_font_size,
                int(round(font_size * style.anchor_font_scale)),
            )
            font_weight = max(font_weight, style.anchor_font_weight)
            opacity = max(opacity, style.anchor_opacity)
            letter_spacing = max(letter_spacing, style.anchor_letter_spacing)
            start_offset = "50%"
            is_hero = True
        elif is_hero:
            font_size = max(font_size, style.hero_font_size)
            font_weight = max(font_weight, 700)
            opacity = max(opacity, 0.95)
            letter_spacing = max(letter_spacing, 0.42)
            start_offset = f"{(lane.order_index % 3) * 2}%"

        cursor = region_cursors.get(lane.region, lane.order_index)
        if is_anchor:
            text, counts = _anchor_text(profile.anchor_words, lane, anchor_cursor)
            anchor_cursor += 1
        else:
            words = profile.hero_words if is_hero else profile.words_for_region(lane.region)
            text, counts = _fit_text_to_lane(words, lane.length_px, font_size, cursor)
        region_cursors[lane.region] = cursor + max(1, len(counts))
        for word, count in counts.items():
            used_words[word] = used_words.get(word, 0) + count
        text_paths.append(
            TextPathLayout(
                lane=lane,
                text=text,
                font_size=font_size,
                fill=_palette_color(style, lane.region, lane.order_index),
                font_weight=font_weight,
                opacity=opacity,
                letter_spacing=letter_spacing,
                start_offset=start_offset,
                is_hero=is_hero,
                is_anchor=is_anchor,
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
            "anchor_path_count": sum(1 for item in text_paths if item.is_anchor),
        },
    )
