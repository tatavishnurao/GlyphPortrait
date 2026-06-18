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


def _lane_angle(lane: TextLane) -> float:
    if len(lane.points) < 2:
        return 0.0
    x0, y0 = lane.points[0]
    x1, y1 = lane.points[-1]
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return float(angle)


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
        "feature_detail": 3.6,
        "clothing_primary": 1.6,
        "outline_or_edge": 1.35,
        "skin_or_warm": 1.25,
        "dark_hair_or_shadow": 1.0,
    }
    ranked: list[tuple[float, TextLane, tuple[float, float]]] = []
    for lane in lanes:
        if lane.region not in style.anchor_regions:
            continue
        min_anchor_length = style.anchor_min_length_px
        if lane.region == "feature_detail":
            min_anchor_length = max(style.min_lane_length_px * 1.15, style.anchor_min_length_px * 0.48)
        if lane.length_px < min_anchor_length:
            continue
        if lane.region == "feature_detail" and lane.length_px > max(190.0, bbox_diag * 0.24):
            continue
        if lane.region == "feature_detail":
            abs_angle = abs(_lane_angle(lane))
            if 34.0 < abs_angle < 60.0:
                continue
        cx, cy = _lane_centroid(lane)
        y_norm = (cy - min_y) / max(max_y - min_y, 1.0)
        if lane.region == "feature_detail" and y_norm > 0.82:
            continue
        if lane.region == "feature_detail":
            zone_weight = 1.35 if y_norm <= 0.62 else 0.82
            centrality = 1.0 - min(1.0, abs(y_norm - 0.42))
        else:
            zone_weight = 1.0
            centrality = 1.0 - min(1.0, abs(y_norm - 0.58))
        effective_length = min(lane.length_px, 260.0) if lane.region == "feature_detail" else lane.length_px
        score = effective_length * region_priority.get(lane.region, 1.0) * zone_weight * (1.0 + centrality * 0.45)
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
    word = words[start % len(words)]
    return word, {word: 1}


def _anchor_text(anchor_words: list[str], lane: TextLane, cursor: int) -> tuple[str, dict[str, int]]:
    if not anchor_words:
        anchor_words = ["CHAMPION"]
    phrase = anchor_words[cursor % len(anchor_words)]
    return phrase, {phrase: 1}


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
    lane_ys = [point[1] for lane in lanes for point in lane.points]
    global_min_y = min(lane_ys) if lane_ys else 0.0
    global_max_y = max(lane_ys) if lane_ys else 1.0
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
        is_hero = lane.source != "feature" and lane.length_px >= hero_threshold and (lane.order_index % 10 == 0)
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
        elif lane.source == "feature":
            _, lane_cy = _lane_centroid(lane)
            y_norm = (lane_cy - global_min_y) / max(global_max_y - global_min_y, 1.0)
            if y_norm <= 0.76:
                font_size = max(font_size, style.hero_font_size - 7)
                font_weight = max(font_weight, 660)
                opacity = max(opacity, 0.96)
                letter_spacing = max(letter_spacing, 0.34)
            else:
                font_size = max(font_size, style.default_font_size + 1)
                font_weight = max(font_weight, 620)
                opacity = min(max(opacity, 0.88), 0.93)
                letter_spacing = max(letter_spacing, 0.28)

        cursor = region_cursors.get(lane.region, lane.order_index)
        if is_anchor:
            text, counts = _anchor_text(profile.anchor_words, lane, anchor_cursor)
            anchor_cursor += 1
        else:
            if lane.source == "feature":
                words = profile.anchor_words or profile.hero_words or profile.texture_words
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
