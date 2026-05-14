from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch import recreate_jordan_face_study_v4 as base

OUT_DIR = ROOT / "examples" / "reference_recreation" / "face_optimization"
TOP_DIR = OUT_DIR / "top_10"
SEED = 5301
GATE_THRESHOLDS = {
    "protected_dark_zone_fill_ratio": 0.095,
    "lane_words_min": 120,
    "lane_words_max": 260,
    "face_luma_mae": 37.0,
    "gray_slab_penalty": 0.10,
    "mouth_banding_penalty": 0.55,
    "anchor_dominance_penalty": 0.12,
}


EXTRA_LANES = [
    {
        "id": "cheek_upper_plane_opt",
        "region": "cheek",
        "points_norm": [[0.715, 0.405], [0.765, 0.42], [0.835, 0.455]],
        "words": ["Air Jordan", "Dedication", "Champion"],
        "size_range": [10, 17],
        "alpha_range": [125, 200],
        "color_family": "mid_gray",
    },
    {
        "id": "cheek_mid_plane_opt",
        "region": "cheek",
        "points_norm": [[0.705, 0.465], [0.765, 0.49], [0.84, 0.525]],
        "words": ["Dedication", "Love of the Game", "Air Jordan"],
        "size_range": [10, 18],
        "alpha_range": [120, 195],
        "color_family": "mid_gray",
    },
    {
        "id": "nose_bridge_vertical_opt",
        "region": "nose_bridge",
        "points_norm": [[0.815, 0.345], [0.825, 0.42], [0.815, 0.505], [0.805, 0.555]],
        "words": ["Scoring", "Finals MVP", "Champion"],
        "size_range": [8, 15],
        "alpha_range": [135, 210],
        "color_family": "mid_gray_to_white",
    },
    {
        "id": "nose_side_highlight_opt",
        "region": "nose_bridge",
        "points_norm": [[0.845, 0.39], [0.86, 0.455], [0.875, 0.525]],
        "words": ["Scoring", "MVP", "Finals MVP"],
        "size_range": [8, 15],
        "alpha_range": [140, 215],
        "color_family": "mid_gray_to_white",
    },
]


@dataclass(frozen=True)
class TrialConfig:
    anchor_size_scale: float
    anchor_alpha_scale: float
    lane_word_spacing: float
    lane_word_count_multiplier: float
    contour_alpha_min: int
    contour_alpha_max: int
    tone_alpha_min: int
    tone_alpha_max: int
    shadow_alpha_min: int
    shadow_alpha_max: int
    dark_zone_suppression: float
    luma_modulation_strength: float
    cheek_lane_multiplier: float
    nose_lane_multiplier: float
    jitter_strength: float
    structure_layer_opacity: float


@dataclass
class TrialResult:
    index: int
    score: float
    params: TrialConfig
    metrics: dict[str, object]
    image: Image.Image


def sample_config(rng: random.Random) -> TrialConfig:
    tone_min = rng.randint(72, 105)
    tone_max = rng.randint(max(tone_min + 35, 130), 185)
    contour_min = rng.randint(120, 165)
    contour_max = rng.randint(max(contour_min + 40, 185), 240)
    shadow_min = rng.randint(26, 44)
    shadow_max = rng.randint(max(shadow_min + 28, 62), 96)
    return TrialConfig(
        anchor_size_scale=rng.uniform(0.58, 0.86),
        anchor_alpha_scale=rng.uniform(0.62, 0.9),
        lane_word_spacing=rng.uniform(6.0, 14.0),
        lane_word_count_multiplier=rng.uniform(1.0, 2.35),
        contour_alpha_min=contour_min,
        contour_alpha_max=contour_max,
        tone_alpha_min=tone_min,
        tone_alpha_max=tone_max,
        shadow_alpha_min=shadow_min,
        shadow_alpha_max=shadow_max,
        dark_zone_suppression=rng.uniform(0.38, 0.74),
        luma_modulation_strength=rng.uniform(0.20, 0.46),
        cheek_lane_multiplier=rng.uniform(1.0, 2.8),
        nose_lane_multiplier=rng.uniform(1.0, 2.6),
        jitter_strength=rng.uniform(1.2, 5.0),
        structure_layer_opacity=rng.uniform(0.62, 1.0),
    )


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clamp_int(value: float, lo: int, hi: int) -> int:
    return int(round(clamp(value, lo, hi)))


def refine_config(parent: TrialConfig, rng: random.Random) -> TrialConfig:
    tone_min = clamp_int(parent.tone_alpha_min + rng.gauss(0, 8), 64, 112)
    tone_max = clamp_int(parent.tone_alpha_max + rng.gauss(0, 12), max(tone_min + 28, 118), 190)
    contour_min = clamp_int(parent.contour_alpha_min + rng.gauss(0, 10), 105, 175)
    contour_max = clamp_int(parent.contour_alpha_max + rng.gauss(0, 14), max(contour_min + 30, 168), 245)
    shadow_min = clamp_int(parent.shadow_alpha_min + rng.gauss(0, 5), 22, 50)
    shadow_max = clamp_int(parent.shadow_alpha_max + rng.gauss(0, 8), max(shadow_min + 20, 54), 102)
    return TrialConfig(
        anchor_size_scale=clamp(parent.anchor_size_scale + rng.gauss(0, 0.055), 0.50, 0.9),
        anchor_alpha_scale=clamp(parent.anchor_alpha_scale + rng.gauss(0, 0.06), 0.50, 0.95),
        lane_word_spacing=clamp(parent.lane_word_spacing + rng.gauss(0, 1.35), 5.5, 16.0),
        lane_word_count_multiplier=clamp(parent.lane_word_count_multiplier + rng.gauss(0, 0.25), 0.8, 2.5),
        contour_alpha_min=contour_min,
        contour_alpha_max=contour_max,
        tone_alpha_min=tone_min,
        tone_alpha_max=tone_max,
        shadow_alpha_min=shadow_min,
        shadow_alpha_max=shadow_max,
        dark_zone_suppression=clamp(parent.dark_zone_suppression + rng.gauss(0, 0.055), 0.30, 0.7),
        luma_modulation_strength=clamp(parent.luma_modulation_strength + rng.gauss(0, 0.045), 0.16, 0.52),
        cheek_lane_multiplier=clamp(parent.cheek_lane_multiplier + rng.gauss(0, 0.35), 0.8, 3.2),
        nose_lane_multiplier=clamp(parent.nose_lane_multiplier + rng.gauss(0, 0.30), 0.8, 3.0),
        jitter_strength=clamp(parent.jitter_strength + rng.gauss(0, 0.55), 0.8, 5.4),
        structure_layer_opacity=clamp(parent.structure_layer_opacity + rng.gauss(0, 0.06), 0.50, 1.0),
    )


def side_by_side(ref: Image.Image, rec: Image.Image) -> Image.Image:
    out = Image.new("RGB", (ref.width + rec.width, max(ref.height, rec.height)), (5, 5, 6))
    out.paste(ref, (0, 0))
    out.paste(rec, (ref.width, 0))
    return out


def build_spec_with_extra_lanes(spec: dict) -> dict:
    spec = json.loads(json.dumps(spec))
    existing = {lane["id"] for lane in spec.get("contour_lanes", [])}
    spec.setdefault("contour_lanes", [])
    spec["contour_lanes"].extend([lane for lane in EXTRA_LANES if lane["id"] not in existing])
    return spec


def lane_repetitions(lane: dict, cfg: TrialConfig) -> int:
    mult = cfg.lane_word_count_multiplier
    if lane["region"] == "cheek":
        mult *= cfg.cheek_lane_multiplier
    if lane["region"] == "nose_bridge":
        mult *= cfg.nose_lane_multiplier
    return max(1, int(round(mult)))


def alpha_scaled_layer(layer: Image.Image, scale: float) -> Image.Image:
    if scale >= 0.995:
        return layer
    arr = np.array(layer, dtype=np.uint8)
    arr[..., 3] = np.clip(arr[..., 3].astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def layer_alpha_fraction(layer: Image.Image, mask: np.ndarray, threshold: int = 20) -> float:
    alpha = np.array(layer, dtype=np.uint8)[..., 3]
    region = mask > 0
    if not np.any(region):
        return 0.0
    return float(((alpha > threshold) & region).sum() / region.sum())


def lower_face_mask(region_masks: dict[str, np.ndarray]) -> np.ndarray:
    cheek = region_masks["cheek"] > 0
    cheek_ys = np.where(cheek)[0]
    lower_cheek = np.zeros_like(region_masks["cheek"], dtype=bool)
    if cheek_ys.size:
        lower_cheek = cheek & (np.arange(cheek.shape[0])[:, None] >= np.percentile(cheek_ys, 45))
    return np.maximum.reduce(
        [
            lower_cheek.astype(np.uint8) * 255,
            region_masks["jaw"],
            region_masks["mouth_chin"],
        ]
    )


def lower_face_brightness_penalty(ref: np.ndarray, rec: np.ndarray, region_masks: dict[str, np.ndarray]) -> float:
    mask = lower_face_mask(region_masks) > 0
    if not np.any(mask):
        return 0.0
    excess = float(base.luma(rec)[mask].mean() - base.luma(ref)[mask].mean())
    return max(0.0, excess / 70.0)


def mouth_banding_penalty(ref: np.ndarray, rec: np.ndarray, mouth_mask: np.ndarray) -> float:
    mask = mouth_mask > 0
    if not np.any(mask):
        return 0.0
    rec_luma = base.luma(rec)
    ref_luma = base.luma(ref)
    ys = np.where(mask)[0]
    if ys.size == 0:
        return 0.0
    row_excesses = []
    row_means = []
    for y in range(int(ys.min()), int(ys.max()) + 1):
        row_mask = mask[y]
        if row_mask.sum() < 8:
            continue
        row_rec = float(rec_luma[y, row_mask].mean())
        row_ref = float(ref_luma[y, row_mask].mean())
        row_means.append(row_rec)
        row_excesses.append(max(0.0, row_rec - row_ref - 10.0))
    if not row_excesses:
        return 0.0
    row_excess = max(row_excesses) / 55.0
    smooth_band = 0.0
    if len(row_means) >= 3:
        smooth_band = max(0.0, (max(row_means) - float(np.median(row_means)) - 12.0) / 50.0)
    return float(np.clip(row_excess * 0.72 + smooth_band * 0.28, 0.0, 1.0))


def gray_slab_penalty(ref: np.ndarray, rec: np.ndarray, region_masks: dict[str, np.ndarray]) -> float:
    roi = lower_face_mask(region_masks) > 0
    if not np.any(roi):
        return 0.0
    overbright = (base.luma(rec) > base.luma(ref) + 35.0) & roi
    labels, count = ndi.label(overbright, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return 0.0
    largest = max(int((labels == label).sum()) for label in range(1, count + 1))
    return float(largest / roi.sum())


def lane_clutter_penalty(lane_layer: Image.Image, lane_words: int, face_mask: np.ndarray) -> float:
    alpha = np.array(lane_layer, dtype=np.uint8)[..., 3]
    region = face_mask > 0
    if not np.any(region):
        return 0.0
    lane_area = float(((alpha > 20) & region).sum() / region.sum())
    bright_lane_area = float(((alpha > 145) & region).sum() / region.sum())
    return max(0.0, (lane_words - 220) / 260.0) + max(0.0, (lane_area - 0.16) / 0.22) + max(0.0, (bright_lane_area - 0.055) / 0.12)


def render_shadow_layer(layer: Image.Image, spec: dict, guides: dict[str, np.ndarray], face_mask: np.ndarray, protected: np.ndarray, rng: random.Random, stats: base.RenderStats, cfg: TrialConfig) -> None:
    h, w = face_mask.shape
    words = base.choose_words(spec, "shadow")
    for y in range(rng.randrange(0, 5), h, 6):
        x = rng.randrange(0, 12)
        while x < w:
            jx = x + rng.randint(-4, 4)
            jy = y + rng.randint(-3, 3)
            x += rng.randint(13, 22)
            if not (0 <= jx < w and 0 <= jy < h and face_mask[jy, jx] > 0):
                continue
            tone = float(guides["tone"][jy, jx])
            is_protected = protected[jy, jx] > 0
            alpha = int(cfg.shadow_alpha_min + (1 - tone) * (cfg.shadow_alpha_max - cfg.shadow_alpha_min))
            if is_protected:
                alpha = int(alpha * cfg.dark_zone_suppression)
            col = base.color_from_tone(tone, protected=is_protected)
            base.draw_text(layer, jx, jy, rng.choice(words), rng.randint(5, 8), col, alpha, rng.uniform(-11, 11), False)
            stats.shadow_words += 1
            stats.alpha_sum += alpha


def render_tone_layer(layer: Image.Image, spec: dict, guides: dict[str, np.ndarray], region_masks: dict[str, np.ndarray], protected: np.ndarray, rng: random.Random, stats: base.RenderStats, cfg: TrialConfig) -> None:
    words = base.choose_words(spec, "texture") + base.choose_words(spec, "structure")
    for region, mask in region_masks.items():
        if region in {"shoulder", "jersey_chest", "jersey_trim"}:
            continue
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            continue
        attempts = int(xs.size * 0.5 / 42)
        for _ in range(attempts):
            i = rng.randrange(xs.size)
            x, y = int(xs[i]), int(ys[i])
            if rng.random() > float(guides["density"][y, x]):
                continue
            tone = float(guides["tone"][y, x])
            is_protected = protected[y, x] > 0
            alpha = int(cfg.tone_alpha_min + (1 - tone) * (cfg.tone_alpha_max - cfg.tone_alpha_min))
            if is_protected:
                alpha = int(alpha * cfg.dark_zone_suppression)
            angle = base.constrained_angle(region, guides["sx"], guides["sy"], x, y, rng)
            base.draw_text(layer, x, y, rng.choice(words), rng.randint(8, 15), base.color_from_tone(tone, protected=is_protected), alpha, angle, rng.random() < 0.36)
            stats.tone_words += 1
            stats.alpha_sum += alpha


def render_contour_layer(layer: Image.Image, spec: dict, guides: dict[str, np.ndarray], region_masks: dict[str, np.ndarray], face_mask: np.ndarray, protected: np.ndarray, rng: random.Random, stats: base.RenderStats, cfg: TrialConfig) -> None:
    words = base.choose_words(spec, "contour") + base.choose_words(spec, "highlight")
    for region, quota in base.CONTOUR_QUOTAS.items():
        region_mask = region_masks.get(region)
        if region_mask is None:
            continue
        region_edge = base.region_edge_candidates(guides, region_mask, 62.0 if region in {"eye_sockets", "mouth_chin", "brow"} else 68.0)
        edge_band = base.protected_edge_band(protected, region_mask) & region_edge
        if region in {"eye_sockets", "mouth_chin", "brow"}:
            candidates = (region_edge | edge_band) & (region_mask > 0) & ~((protected > 0) & ~edge_band)
        else:
            candidates = (region_edge | edge_band) & (region_mask > 0)
        ys, xs = np.where(candidates)
        if xs.size == 0:
            continue
        order = list(range(xs.size))
        rng.shuffle(order)
        drawn = 0
        for i in order:
            if drawn >= quota:
                break
            x, y = int(xs[i]), int(ys[i])
            if protected[y, x] > 0 and not edge_band[y, x]:
                continue
            tone = float(guides["tone"][y, x])
            alpha = int(cfg.contour_alpha_min + tone * (cfg.contour_alpha_max - cfg.contour_alpha_min))
            if protected[y, x] > 0:
                alpha = int(alpha * cfg.dark_zone_suppression)
            angle = base.constrained_angle(region, guides["sx"], guides["sy"], x, y, rng)
            col = base.color_from_tone(tone, protected=False, highlight=True)
            size = rng.randint(9, 17)
            base.draw_text(layer, x, y, rng.choice(words), size, col, alpha, angle, True)
            stats.contour_words += 1
            stats.alpha_sum += alpha
            drawn += 1


def render_lane_layer(layer: Image.Image, spec: dict, guides: dict[str, np.ndarray], face_mask: np.ndarray, protected: np.ndarray, crop_box: tuple[int, int, int, int], rng: random.Random, stats: base.RenderStats, cfg: TrialConfig) -> None:
    h, w = face_mask.shape
    for lane in spec.get("contour_lanes", []):
        points = base.lane_points(lane, crop_box)
        if len(points) < 2:
            continue
        size_lo, size_hi = lane["size_range"]
        alpha_lo, alpha_hi = lane["alpha_range"]
        spacing = max(5.5, cfg.lane_word_spacing * (size_hi / 18.0))
        samples = base.interpolate_polyline(points, spacing)
        for _ in range(lane_repetitions(lane, cfg)):
            for x, y, angle in samples:
                px = int(round(x))
                py = int(round(y))
                if not (0 <= px < w and 0 <= py < h):
                    continue
                theta = np.deg2rad(angle + 90)
                offset = rng.uniform(-cfg.jitter_strength, cfg.jitter_strength)
                px = int(round(px + np.cos(theta) * offset))
                py = int(round(py + np.sin(theta) * offset))
                if not (0 <= px < w and 0 <= py < h and face_mask[py, px] > 0):
                    continue
                in_dark = protected[py, px] > 0
                if in_dark and lane["id"] not in {"upper_eye_socket", "mouth_crease"}:
                    continue
                tone = float(guides["tone"][py, px])
                alpha = rng.randint(alpha_lo, alpha_hi)
                if in_dark:
                    alpha = int(alpha * cfg.dark_zone_suppression)
                size = rng.randint(size_lo, size_hi)
                base.draw_text(layer, px, py, rng.choice(lane["words"]), size, base.lane_color(lane["color_family"], tone), alpha, angle, True)
                stats.alpha_sum += alpha
                stats.add_lane(lane["id"], lane["region"], alpha)


def render_anchor_layer(layer: Image.Image, spec: dict, crop_box: tuple[int, int, int, int], face_mask: np.ndarray, protected: np.ndarray, stats: base.RenderStats, cfg: TrialConfig) -> None:
    cx0, cy0, _, _ = crop_box
    for anchor in spec["anchor_placements"]:
        if anchor["region"] in {"jersey_chest", "jersey_trim", "shoulder"}:
            continue
        x = int(anchor["pos_norm"][0] * base.CANVAS_W) - cx0
        y = int(anchor["pos_norm"][1] * base.CANVAS_H) - cy0
        if not (0 <= x < face_mask.shape[1] and 0 <= y < face_mask.shape[0]):
            continue
        is_protected = protected[y, x] > 0
        size = max(9, int(anchor["size_px"] * cfg.anchor_size_scale))
        alpha = int(np.clip(int(anchor.get("alpha", 200)) * cfg.anchor_alpha_scale, 90, 230))
        if is_protected:
            alpha = int(alpha * cfg.dark_zone_suppression)
        base.draw_text(layer, x, y, anchor["text"], size, base.anchor_color(anchor["color"]), alpha, float(anchor["angle"]), True)
        stats.manual_anchor_words += 1
        stats.alpha_sum += alpha


def final_composite(texture: Image.Image, structure: Image.Image, crop: np.ndarray, face_mask: np.ndarray, protected: np.ndarray, cfg: TrialConfig) -> Image.Image:
    texture_rgb = np.array(texture.convert("RGB"), dtype=np.float32)
    gray = base.luma(crop)
    target = np.repeat(gray[..., None], 3, axis=2)
    mask = face_mask > 0
    strength = cfg.luma_modulation_strength
    texture_rgb[mask] = texture_rgb[mask] * (1 - strength) + target[mask] * strength
    dark = protected > 0
    texture_rgb[dark] = np.minimum(texture_rgb[dark], target[dark] * cfg.dark_zone_suppression + 14)
    texture_rgb[~mask] = 5
    base_img = Image.fromarray(np.clip(texture_rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    base_img.alpha_composite(alpha_scaled_layer(base.clip_layer(structure, face_mask), cfg.structure_layer_opacity))
    arr = np.array(base_img.convert("RGB"), dtype=np.float32)
    arr[dark] = np.minimum(arr[dark], target[dark] * cfg.dark_zone_suppression + 16)
    arr[~mask] = 5
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def score_metrics(metrics: dict[str, object]) -> float:
    return (
        3.0 * float(metrics["edge_overlap_face"])
        + 2.5 * float(metrics["cheek_edge_overlap"])
        + 2.5 * float(metrics["nose_bridge_edge_overlap"])
        - 0.025 * float(metrics["face_luma_mae"])
        - 5.0 * abs(float(metrics["protected_dark_zone_fill_ratio"]) - 0.08)
        - 2.5 * float(metrics["lower_face_brightness_penalty"])
        - 3.0 * float(metrics["gray_slab_penalty"])
        - 1.5 * float(metrics["anchor_dominance_penalty"])
        - 2.25 * float(metrics["mouth_banding_penalty"])
        - 1.25 * float(metrics["lane_clutter_penalty"])
        + 0.0008 * min(int(metrics["lane_words"]), 220)
    )


def failed_gates(metrics: dict[str, object]) -> list[str]:
    failures: list[str] = []
    if float(metrics["protected_dark_zone_fill_ratio"]) > GATE_THRESHOLDS["protected_dark_zone_fill_ratio"]:
        failures.append("protected_dark_zone_fill_ratio")
    lane_words = int(metrics["lane_words"])
    if lane_words < GATE_THRESHOLDS["lane_words_min"] or lane_words > GATE_THRESHOLDS["lane_words_max"]:
        failures.append("lane_words")
    if float(metrics["face_luma_mae"]) > GATE_THRESHOLDS["face_luma_mae"]:
        failures.append("face_luma_mae")
    if float(metrics["gray_slab_penalty"]) > GATE_THRESHOLDS["gray_slab_penalty"]:
        failures.append("gray_slab_penalty")
    if float(metrics["mouth_banding_penalty"]) > GATE_THRESHOLDS["mouth_banding_penalty"]:
        failures.append("mouth_banding_penalty")
    if float(metrics["anchor_dominance_penalty"]) > GATE_THRESHOLDS["anchor_dominance_penalty"]:
        failures.append("anchor_dominance_penalty")
    return failures


def attach_eligibility(metrics: dict[str, object]) -> None:
    failures = failed_gates(metrics)
    metrics["eligible"] = not failures
    metrics["failed_gates"] = ",".join(failures)


def render_trial(context: dict, cfg: TrialConfig, trial_index: int) -> TrialResult:
    rng = random.Random(SEED + trial_index * 7919)
    crop = context["crop"]
    crop_box = context["crop_box"]
    face_mask = context["face_mask"]
    protected = context["protected"]
    region_masks = context["region_masks"]
    guides = context["guides"]
    spec = context["spec"]
    h, w = face_mask.shape
    stats = base.RenderStats()

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    tone = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    contour = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    lanes = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    anchors = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    render_shadow_layer(shadow, spec, guides, face_mask, protected, rng, stats, cfg)
    render_tone_layer(tone, spec, guides, region_masks, protected, rng, stats, cfg)
    render_contour_layer(contour, spec, guides, region_masks, face_mask, protected, rng, stats, cfg)
    render_lane_layer(lanes, spec, guides, face_mask, protected, crop_box, rng, stats, cfg)
    render_anchor_layer(anchors, spec, crop_box, face_mask, protected, stats, cfg)

    texture = Image.new("RGBA", (w, h), (5, 5, 6, 255))
    for layer in (shadow, tone):
        texture.alpha_composite(base.clip_layer(layer, face_mask))
    structure = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for layer in (contour, lanes, anchors):
        structure.alpha_composite(base.clip_layer(layer, face_mask))
    image = final_composite(texture, structure, crop, face_mask, protected, cfg)
    rec = np.array(image)
    anchor_area = layer_alpha_fraction(anchors, face_mask, threshold=18)

    metrics: dict[str, object] = {
        "face_luma_mae": float(np.abs(base.luma(crop) - base.luma(rec))[face_mask > 0].mean()),
        "edge_overlap_face": base.edge_overlap(crop, rec, face_mask),
        "protected_dark_zone_fill_ratio": base.protected_fill_ratio(rec, protected),
        "cheek_edge_overlap": base.edge_overlap(crop, rec, region_masks["cheek"]),
        "nose_bridge_edge_overlap": base.edge_overlap(crop, rec, region_masks["nose_bridge"]),
        "lower_face_brightness_penalty": lower_face_brightness_penalty(crop, rec, region_masks),
        "gray_slab_penalty": gray_slab_penalty(crop, rec, region_masks),
        "anchor_dominance_penalty": max(0.0, (anchor_area - 0.035) / 0.12),
        "mouth_banding_penalty": mouth_banding_penalty(crop, rec, region_masks["mouth_chin"]),
        "lane_clutter_penalty": lane_clutter_penalty(lanes, stats.lane_words, face_mask),
        "lane_words": stats.lane_words,
        "average_alpha": stats.average_alpha,
        "total_words_drawn": stats.total_words,
    }
    attach_eligibility(metrics)
    return TrialResult(trial_index, score_metrics(metrics), cfg, metrics, image)


def build_context() -> dict:
    spec = build_spec_with_extra_lanes(base.load_spec())
    crop, crop_box = base.load_face_crop(spec)
    h, w = crop.shape[:2]
    face_mask = base.build_face_mask(crop)
    return {
        "spec": spec,
        "crop": crop,
        "crop_box": crop_box,
        "face_mask": face_mask,
        "region_masks": base.build_region_masks(spec, (h, w), crop_box, face_mask),
        "protected": base.protected_mask(spec, crop, (h, w), crop_box, face_mask),
        "guides": base.build_guides(crop, face_mask),
    }


def write_csv(path: Path, results: list[TrialResult]) -> None:
    if not results:
        return
    metric_keys = sorted({key for result in results for key in result.metrics})
    param_keys = list(asdict(results[0].params).keys())
    fieldnames = ["trial", "score", *metric_keys, *param_keys]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({"trial": result.index, "score": result.score, **result.metrics, **asdict(result.params)})


def write_top_results(results: list[TrialResult]) -> None:
    TOP_DIR.mkdir(parents=True, exist_ok=True)
    for rank, result in enumerate(sorted(results, key=lambda item: item.score, reverse=True)[:10], start=1):
        result.image.save(TOP_DIR / f"rank_{rank}_trial_{result.index}.png")
        (TOP_DIR / f"rank_{rank}_trial_{result.index}.json").write_text(
            json.dumps({"rank": rank, "trial": result.index, "score": result.score, "params": asdict(result.params), "metrics": result.metrics}, indent=2) + "\n"
        )


def eligible_results(results: list[TrialResult]) -> list[TrialResult]:
    return [result for result in results if bool(result.metrics.get("eligible"))]


def select_eligible_best(results: list[TrialResult]) -> TrialResult | None:
    eligible = eligible_results(results)
    return max(eligible, key=lambda result: result.score) if eligible else None


def select_near_miss(results: list[TrialResult]) -> TrialResult:
    def key(result: TrialResult) -> tuple[int, float, float]:
        failed = str(result.metrics.get("failed_gates", ""))
        failure_count = 0 if not failed else len(failed.split(","))
        return (-failure_count, result.score, -float(result.metrics["gray_slab_penalty"]))

    return max(results, key=key)


def write_result_bundle(prefix: str, result: TrialResult, context: dict) -> None:
    ref_img = Image.fromarray(context["crop"], "RGB")
    result.image.save(OUT_DIR / f"{prefix}_face.png")
    side_by_side(ref_img, result.image).save(OUT_DIR / f"{prefix}_side_by_side.png")
    (OUT_DIR / f"{prefix}_metrics.json").write_text(
        json.dumps({"score": result.score, "params": asdict(result.params), "metrics": result.metrics}, indent=2) + "\n"
    )


def write_outputs(results: list[TrialResult], refined_results: list[TrialResult], context: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best = select_eligible_best(results) or select_near_miss(results)
    write_result_bundle("best", best, context)
    eligible_best = select_eligible_best([*results, *refined_results])
    if eligible_best:
        write_result_bundle("eligible_best", eligible_best, context)
    else:
        write_result_bundle("near_miss_best", select_near_miss([*results, *refined_results]), context)
    write_csv(OUT_DIR / "trials.csv", results)
    write_top_results(results)

    write_csv(OUT_DIR / "refine_trials.csv", refined_results)
    if refined_results:
        refined_best = select_eligible_best(refined_results) or select_near_miss(refined_results)
        write_result_bundle("refined_best", refined_best, context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop optimizer for the Jordan face study renderer.")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--refine-top", type=int, default=5)
    parser.add_argument("--refine-trials", type=int, default=100)
    args = parser.parse_args()
    start = perf_counter()
    rng = random.Random(SEED)
    context = build_context()
    results: list[TrialResult] = []
    refined_results: list[TrialResult] = []
    best: TrialResult | None = None

    for idx in range(args.trials):
        cfg = sample_config(rng)
        result = render_trial(context, cfg, idx)
        results.append(result)
        if idx % 5 == 0:
            print(f"stage 1 progress {idx + 1}/{args.trials}", flush=True)
        if best is None or result.score > best.score:
            best = result
            print(f"stage 1 trial {idx:03d}: best score {best.score:.4f}")
            print(json.dumps({"params": asdict(best.params), "metrics": best.metrics}, indent=2))

    ranked = sorted(results, key=lambda result: result.score, reverse=True)
    parents = ranked[: max(1, min(args.refine_top, len(ranked)))]
    refined_best: TrialResult | None = None
    for idx in range(args.refine_trials):
        parent = parents[idx % len(parents)]
        cfg = refine_config(parent.params, rng)
        result = render_trial(context, cfg, args.trials + idx)
        refined_results.append(result)
        if idx % 5 == 0:
            print(f"stage 2 progress {idx + 1}/{args.refine_trials}", flush=True)
        if refined_best is None or result.score > refined_best.score:
            refined_best = result
            print(f"stage 2 trial {idx:03d}: refined best score {refined_best.score:.4f}")
            print(json.dumps({"params": asdict(refined_best.params), "metrics": refined_best.metrics}, indent=2))

    write_outputs(results, refined_results, context)
    assert best is not None
    final_best = select_eligible_best([*results, *refined_results]) or select_near_miss([*results, *refined_results])
    selected_refined_best = (select_eligible_best(refined_results) or select_near_miss(refined_results)) if refined_results else None
    print("optimization complete")
    print(
        json.dumps(
            {
                "trials": args.trials,
                "refine_top": args.refine_top,
                "refine_trials": args.refine_trials,
                "elapsed_seconds": perf_counter() - start,
                "best_score": final_best.score,
                "best_params": asdict(final_best.params),
                "best_metrics": final_best.metrics,
                "eligible_results": len(eligible_results([*results, *refined_results])),
                "refined_best_params": asdict(selected_refined_best.params) if selected_refined_best else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
