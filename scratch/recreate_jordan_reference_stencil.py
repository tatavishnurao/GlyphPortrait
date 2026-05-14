from __future__ import annotations

import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glyphforge.typography.fonts import find_font

TARGET = ROOT / "reference_img" / "Michael-Jordan-Wallpaper-Desktop-1.jpg"
OUT = ROOT / "examples" / "reference_recreation"
W, H = 1920, 1080
SEED = 20260514
RIGHT_CUTOFF = 0.52

FACE_WORDS = [
    "Air Jordan",
    "Michael Jordan",
    "MVP",
    "Champion",
    "NBA Rookie of the Year",
    "Dedication",
    "Dominance",
    "Scoring",
    "Defense",
    "Finals MVP",
    "Six Rings",
    "Chicago",
    "Flight",
    "Clutch",
    "Greatness",
    "Love of the Game",
    "1984",
    "1985",
    "Slam Dunk",
    "Olympic Gold",
]
JERSEY_WORDS = [
    "BULLS",
    "23",
    "Chicago Bulls",
    "red and black",
    "NBA Champion",
    "Finals MVP",
    "Dynasty",
    "Game Winner",
    "All-Star",
    "Six Rings",
    "Rookie",
    "MVP",
]
MICRO_WORDS = FACE_WORDS + JERSEY_WORDS + ["focus", "prime", "drive", "work", "rise", "air"]


@lru_cache(maxsize=160)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[Path | None] = []
    if bold:
        candidates.extend(
            [
                Path("/usr/share/fonts/TTF/OpenSans-CondensedExtraBold.ttf"),
                Path("/usr/share/fonts/TTF/OpenSans-ExtraBold.ttf"),
                Path("/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf"),
            ]
        )
    candidates.extend(
        [
            Path("/usr/share/fonts/TTF/OpenSans-CondensedBold.ttf"),
            Path("/usr/share/fonts/TTF/OpenSans-Bold.ttf"),
            Path("/usr/share/fonts/TTF/OpenSans-Regular.ttf"),
            Path("/usr/share/fonts/TTF/DejaVuSansCondensed.ttf"),
            find_font(ROOT / "assets" / "fonts"),
        ]
    )
    for candidate in candidates:
        if candidate and candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def load_reference() -> np.ndarray:
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)
    return np.array(Image.open(TARGET).convert("RGB").resize((W, H), Image.Resampling.LANCZOS), dtype=np.uint8)


def luma(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def largest_component(mask: np.ndarray, min_cx: int = 0) -> np.ndarray:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    xs_grid = np.broadcast_to(np.arange(mask.shape[1]), mask.shape)
    best_label = 0
    best_score = -1.0
    for label in range(1, count + 1):
        region = labels == label
        area = int(region.sum())
        if area < 1000:
            continue
        cx = float(xs_grid[region].mean())
        if cx < min_cx:
            continue
        score = area * (1.0 + cx / mask.shape[1])
        if score > best_score:
            best_label = label
            best_score = score
    return labels == best_label if best_label else np.zeros(mask.shape, dtype=bool)


def smooth_mask(mask: np.ndarray, radius: float = 2.0, threshold: int = 80) -> np.ndarray:
    blurred = Image.fromarray(mask.astype(np.uint8) * 255, "L").filter(ImageFilter.GaussianBlur(radius))
    return np.array(blurred) > threshold


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, 0, W, H
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def contrast_stretch(values: np.ndarray, mask: np.ndarray, lo_q: float = 2.0, hi_q: float = 98.5) -> np.ndarray:
    inside = values[mask]
    if inside.size == 0:
        return np.clip(values, 0.0, 1.0).astype(np.float32)
    lo, hi = np.percentile(inside, [lo_q, hi_q])
    return np.clip((values - lo) / max(1e-6, hi - lo), 0.0, 1.0).astype(np.float32)


def clahe_like_luma(tone: np.ndarray, subject: np.ndarray) -> np.ndarray:
    stretched = contrast_stretch(tone, subject)
    local_mean = ndi.gaussian_filter(stretched, sigma=16.0)
    local_detail = stretched - local_mean
    enhanced = np.clip(stretched + local_detail * 0.85, 0.0, 1.0)
    return np.power(enhanced, 0.86).astype(np.float32)


def build_target_maps(rgb: np.ndarray) -> dict[str, np.ndarray]:
    gray = luma(rgb)
    xs = np.broadcast_to(np.arange(W), gray.shape)
    # Keep the black field black: the reference has faint background type below
    # luma 18, so a lower threshold admits large ghost slabs into the subject.
    raw_subject = (gray > 18) & (xs > int(W * RIGHT_CUTOFF))
    raw_subject = ndi.binary_closing(raw_subject, structure=np.ones((15, 15), dtype=bool), iterations=1)
    raw_subject = ndi.binary_dilation(raw_subject, structure=np.ones((5, 5), dtype=bool), iterations=1)
    subject = largest_component(raw_subject, int(W * RIGHT_CUTOFF))
    subject = ndi.binary_fill_holes(subject)
    subject = ndi.binary_closing(subject, structure=np.ones((13, 13), dtype=bool), iterations=1)
    subject = smooth_mask(subject, 1.4, 72)
    subject = largest_component(subject, int(W * RIGHT_CUTOFF))

    x0, y0, x1, y1 = bbox(subject)
    yy = np.broadcast_to(np.arange(H)[:, None], gray.shape)
    xx = np.broadcast_to(np.arange(W), gray.shape)
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    red_strength = np.clip((r - np.maximum(g, b)) / 130.0, 0.0, 1.0)
    lower_subject = yy > y0 + int((y1 - y0) * 0.43)
    jersey = (red_strength > 0.115) & (r > 28) & subject & lower_subject
    jersey = ndi.binary_closing(jersey, structure=np.ones((19, 19), dtype=bool), iterations=2)
    jersey = ndi.binary_dilation(jersey, structure=np.ones((9, 9), dtype=bool), iterations=2)
    jersey = ndi.binary_fill_holes(jersey) & subject
    jersey = smooth_mask(jersey, 1.2, 78)

    upper = yy < y0 + int((y1 - y0) * 0.59)
    neck_band = (
        (yy < y0 + int((y1 - y0) * 0.72))
        & (xx > x0 + int((x1 - x0) * 0.18))
        & (xx < x0 + int((x1 - x0) * 0.68))
    )
    face = subject & ~jersey & (upper | neck_band)
    face = ndi.binary_closing(face, structure=np.ones((11, 11), dtype=bool), iterations=1)
    face = smooth_mask(face, 1.4, 75) & subject & ~jersey
    shoulder = subject & ~face & ~jersey

    tone = np.clip(gray / 255.0, 0.0, 1.0).astype(np.float32)
    enhanced_luma = clahe_like_luma(tone, subject)
    blur = ndi.gaussian_filter(gray.astype(np.float32), sigma=1.05)
    sx = ndi.sobel(blur, axis=1)
    sy = ndi.sobel(blur, axis=0)
    edge = np.hypot(sx, sy)
    edge = edge / max(1e-6, float(edge.max()))
    dog = np.abs(ndi.gaussian_filter(gray, sigma=0.7) - ndi.gaussian_filter(gray, sigma=2.1))
    dog = dog / max(1e-6, float(dog.max()))
    edge_strength = np.maximum(edge, dog * 0.78) * subject
    edge_inside = edge_strength[subject]
    edge_cut = np.percentile(edge_inside, 84.0) if edge_inside.size else 1.0
    edge_map = (edge_strength >= edge_cut) & subject
    edge_map = ndi.binary_dilation(edge_map, structure=np.ones((2, 2), dtype=bool), iterations=1)

    fx0, fy0, fx1, fy1 = bbox(face)
    fh = max(1, fy1 - fy0)
    rel_y = (yy - fy0) / fh
    dark_candidate = face & (
        (tone < 0.18)
        | ((enhanced_luma < 0.25) & (rel_y > 0.18))
        | ((tone < 0.28) & edge_map & ((rel_y > 0.22) & (rel_y < 0.88)))
    )
    dark_zone = dark_candidate & (
        ((rel_y > 0.23) & (rel_y < 0.50))
        | ((rel_y > 0.58) & (rel_y < 0.88))
        | ((rel_y > 0.79) & (rel_y < 0.98))
    )
    dark_zone = ndi.binary_dilation(dark_zone, structure=np.ones((5, 5), dtype=bool), iterations=1)
    dark_zone = smooth_mask(dark_zone, 1.1, 54) & face

    highlight = ((enhanced_luma > 0.66) | ((tone > 0.47) & edge_map)) & subject
    highlight = smooth_mask(highlight, 0.8, 76) & subject

    return {
        "subject": subject,
        "face": face,
        "jersey": jersey,
        "shoulder": shoulder,
        "tone": tone,
        "enhanced_luma": enhanced_luma,
        "edge_strength": edge_strength.astype(np.float32),
        "edge_map": edge_map,
        "dark_zone": dark_zone,
        "highlight": highlight,
        "red_strength": red_strength.astype(np.float32),
        "sx": sx.astype(np.float32),
        "sy": sy.astype(np.float32),
    }


def draw_text(
    layer: Image.Image,
    x: int,
    y: int,
    text: str,
    size: int,
    fill: tuple[int, int, int],
    alpha: int,
    angle: float = 0.0,
    bold: bool = False,
) -> None:
    fnt = font(size, bold)
    box = fnt.getbbox(text)
    patch = Image.new("RGBA", (max(1, box[2] - box[0] + 12), max(1, box[3] - box[1] + 12)), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((6 - box[0], 6 - box[1]), text, font=fnt, fill=(*fill, int(alpha)))
    if abs(angle) > 0.2:
        patch = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    layer.alpha_composite(patch, (int(x), int(y)))


def clip_layer(layer: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(layer, dtype=np.uint8)
    arr[..., 3] = np.minimum(arr[..., 3], mask.astype(np.uint8) * 255)
    return Image.fromarray(arr, "RGBA")


def alpha_to_rgb(layer: Image.Image) -> np.ndarray:
    return np.array(Image.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, 255)), layer).convert("RGB"), dtype=np.float32)


def tangent_angle(maps: dict[str, np.ndarray], x: int, y: int, clamp: float = 70.0) -> float:
    gx = float(maps["sx"][y, x])
    gy = float(maps["sy"][y, x])
    if abs(gx) + abs(gy) < 0.01:
        return 0.0
    angle = math.degrees(math.atan2(gy, gx)) + 90.0
    while angle < -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return float(np.clip(angle, -clamp, clamp))


def draw_weighted_texture(
    mask: np.ndarray,
    maps: dict[str, np.ndarray],
    words: list[str],
    rng: random.Random,
    *,
    jersey: bool = False,
    count: int = 1000,
    size_range: tuple[int, int] = (8, 18),
    alpha_range: tuple[int, int] = (70, 165),
) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ys, xs = np.where(mask)
    if xs.size == 0:
        return layer
    weights = (
        0.30
        + maps["edge_strength"][ys, xs] * 0.36
        + (1.0 - maps["tone"][ys, xs]) * (0.28 if not jersey else 0.10)
        + maps["red_strength"][ys, xs] * (0.0 if not jersey else 0.46)
    )
    weights = weights / max(1e-6, float(weights.sum()))
    picks = rng.choices(range(xs.size), weights=weights.tolist(), k=count)
    for idx in picks:
        x = int(xs[idx] + rng.randint(-10, 10))
        y = int(ys[idx] + rng.randint(-8, 8))
        if not (0 <= x < W and 0 <= y < H and mask[y, x]):
            continue
        tone = float(maps["tone"][y, x])
        edge = float(maps["edge_strength"][y, x])
        size = rng.randint(*size_range)
        alpha = int(np.clip(rng.randint(*alpha_range) + edge * 45, 35, 225))
        if jersey:
            red = float(maps["red_strength"][y, x])
            if tone > 0.62 and rng.random() < 0.20:
                v = int(np.clip(160 + tone * 95, 170, 255))
                color = (v, v, v)
            else:
                color = (
                    int(np.clip(95 + red * 170 + tone * 70, 90, 255)),
                    int(np.clip(6 + tone * 28, 4, 54)),
                    int(np.clip(8 + tone * 24, 4, 52)),
                )
        else:
            v = int(np.clip(32 + maps["enhanced_luma"][y, x] * 205 + edge * 45 + rng.randint(-10, 10), 26, 248))
            color = (v, v, min(255, v + 5))
        angle = tangent_angle(maps, x, y, 64.0) + rng.uniform(-8, 8) if rng.random() < 0.48 else rng.uniform(-18, 18)
        draw_text(layer, x, y, rng.choice(words), size, color, alpha, angle, rng.random() < 0.36)
    return clip_layer(layer, mask)


def draw_micro_texture(mask: np.ndarray, maps: dict[str, np.ndarray], rng: random.Random, jersey: bool = False) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ys, xs = np.where(mask)
    if xs.size == 0:
        return layer
    x0, y0, x1, y1 = bbox(mask)
    row_step = 5 if not jersey else 8
    for y in range(y0 + rng.randrange(0, row_step), y1, row_step):
        x = x0 + rng.randrange(0, 14)
        while x < x1:
            jx = x + rng.randint(-5, 5)
            jy = y + rng.randint(-4, 4)
            x += rng.randint(10, 22) if not jersey else rng.randint(18, 38)
            if not (0 <= jx < W and 0 <= jy < H and mask[jy, jx]) or rng.random() > (0.93 if not jersey else 0.78):
                continue
            tone = float(maps["tone"][jy, jx])
            if jersey:
                red = float(maps["red_strength"][jy, jx])
                color = (
                    int(np.clip(75 + red * 120 + tone * 75, 70, 230)),
                    int(np.clip(4 + tone * 18, 3, 36)),
                    int(np.clip(5 + tone * 16, 3, 34)),
                )
                size = rng.randint(6, 11)
                alpha = int(np.clip(42 + red * 55 + tone * 28, 38, 120))
            else:
                v = int(np.clip(42 + tone * 148 + rng.randint(-6, 7), 28, 205))
                color = (v, v, min(220, v + 4))
                size = rng.randint(5, 9)
                alpha = int(np.clip(50 + (1.0 - tone) * 38 + rng.randint(-5, 12), 42, 112))
            draw_text(layer, jx, jy, rng.choice(MICRO_WORDS), size, color, alpha, rng.uniform(-24, 24))
    return clip_layer(layer, mask)


def draw_contour_texture(mask: np.ndarray, maps: dict[str, np.ndarray], rng: random.Random, jersey: bool = False) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    candidates = np.argwhere(maps["edge_map"] & mask)
    if candidates.size == 0:
        return layer
    order = list(range(candidates.shape[0]))
    rng.shuffle(order)
    limit = 720 if not jersey else 560
    for idx in order[:limit]:
        y, x = [int(v) for v in candidates[idx]]
        tone = float(maps["tone"][y, x])
        edge = float(maps["edge_strength"][y, x])
        if jersey:
            red = float(maps["red_strength"][y, x])
            color = (
                int(np.clip(130 + red * 120 + tone * 60, 120, 255)),
                int(np.clip(8 + tone * 34, 5, 62)),
                int(np.clip(8 + tone * 30, 5, 58)),
            )
            word = rng.choice(JERSEY_WORDS)
            size = rng.randint(10, 22)
        else:
            v = int(np.clip(72 + maps["enhanced_luma"][y, x] * 160 + edge * 42, 66, 252))
            color = (v, v, min(255, v + 5))
            word = rng.choice(FACE_WORDS)
            size = rng.randint(8, 19)
        alpha = int(np.clip(118 + edge * 88 + tone * 28 + rng.randint(-12, 18), 110, 230))
        draw_text(layer, x + rng.randint(-7, 7), y + rng.randint(-6, 6), word, size, color, alpha, tangent_angle(maps, x, y), True)
    return clip_layer(layer, mask)


def combine_layers(layers: list[Image.Image]) -> Image.Image:
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for layer in layers:
        out.alpha_composite(layer)
    return out


def apply_luminance_stencil(
    raw_layer: Image.Image,
    region_mask: np.ndarray,
    maps: dict[str, np.ndarray],
    *,
    jersey: bool = False,
) -> np.ndarray:
    raw = alpha_to_rgb(clip_layer(raw_layer, region_mask))
    luma_curve = maps["enhanced_luma"]
    if jersey:
        curve = np.clip(0.20 + 1.22 * np.power(luma_curve, 0.92) + maps["red_strength"] * 0.34, 0.16, 1.72)
    else:
        curve = np.clip(0.24 + 1.50 * np.power(luma_curve, 1.00) + maps["edge_strength"] * 0.18, 0.14, 1.78)
    out = raw * curve[..., None]
    dark = maps["dark_zone"] & region_mask
    out[dark] *= 0.20
    highlight = maps["highlight"] & region_mask
    out[highlight] = np.maximum(out[highlight], raw[highlight] * 1.12)
    out[~region_mask] = 0
    return np.clip(out, 0, 255)


def lower_face_slab_correction(face_rgb: np.ndarray, ref_rgb: np.ndarray, face: np.ndarray, maps: dict[str, np.ndarray]) -> np.ndarray:
    out = face_rgb.copy()
    ref_l = luma(ref_rgb)
    out_l = luma(out)
    _, fy0, _, fy1 = bbox(face)
    yy = np.broadcast_to(np.arange(H)[:, None], face.shape)
    rel_y = (yy - fy0) / max(1, fy1 - fy0)
    lower = face & (rel_y > 0.55) & (rel_y < 0.92)
    overbright = lower & (out_l > ref_l + 25.0) & (out_l > 55.0)
    if np.any(overbright):
        target_ratio = (ref_l[overbright] + 8.0) / np.maximum(out_l[overbright], 1.0)
        out[overbright] *= np.clip(target_ratio, 0.22, 0.92)[:, None]

    slabs = lower & (luma(out) > ref_l + 18.0) & (maps["enhanced_luma"] < 0.42)
    labels, count = ndi.label(slabs, structure=np.ones((3, 3), dtype=bool))
    for label in range(1, count + 1):
        region = labels == label
        if int(region.sum()) > 900:
            out[region] *= 0.58
    return np.clip(out, 0, 255)


def render_anchor_layer(maps: dict[str, np.ndarray], subject: np.ndarray) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    anchors = [
        ("NBA Rookie of the Year", 1400, 218, 30, (225, 225, 232), 215, 5),
        ("MVP", 1410, 284, 48, (235, 235, 240), 218, -5),
        ("Air Jordan", 1368, 365, 32, (220, 220, 226), 210, 4),
        ("Dedication", 1370, 454, 33, (160, 160, 168), 168, 25),
        ("Dominance", 1340, 590, 31, (220, 220, 226), 202, 70),
        ("Scoring", 1565, 535, 28, (218, 218, 224), 200, -8),
        ("BULLS", 1462, 950, 94, (238, 224, 203), 220, -12),
        ("23", 1548, 846, 116, (245, 235, 214), 220, -8),
    ]
    for text, x, y, size, color, alpha, angle in anchors:
        draw_text(layer, x, y, text, size, color, alpha, angle, True)
    clipped = clip_layer(layer, subject)
    arr = np.array(clipped, dtype=np.uint8)
    tone = maps["tone"]
    mod = np.clip(0.55 + maps["enhanced_luma"] * 0.65, 0.35, 1.08)
    arr[..., 3] = (arr[..., 3].astype(np.float32) * mod).astype(np.uint8)
    dark = maps["dark_zone"]
    arr[..., 3][dark] = (arr[..., 3][dark].astype(np.float32) * 0.38).astype(np.uint8)
    bright = (tone > 0.55) & subject
    arr[..., 3][bright] = np.maximum(arr[..., 3][bright], (arr[..., 3][bright].astype(np.float32) * 1.08).astype(np.uint8))
    return Image.fromarray(arr, "RGBA")


def edge_mask(rgb: np.ndarray, region: np.ndarray) -> np.ndarray:
    gray = luma(rgb).astype(np.float32)
    blur = ndi.gaussian_filter(gray, sigma=1.1)
    mag = np.hypot(ndi.sobel(blur, axis=1), ndi.sobel(blur, axis=0))
    inside = mag[region]
    if inside.size == 0:
        return np.zeros(region.shape, dtype=bool)
    local = mag >= ndi.maximum_filter(mag, size=5)
    return (mag >= np.percentile(inside, 82.0)) & local & region


def red_mask(rgb: np.ndarray, subject: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    red = (r - np.maximum(g, b) > 18) & (r > 40) & subject
    return ndi.binary_opening(red, structure=np.ones((3, 3), dtype=bool), iterations=1)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    union = a | b
    if not np.any(union):
        return 0.0
    return float((a & b).sum() / union.sum())


def compute_metrics(ref: np.ndarray, rec: np.ndarray, maps: dict[str, np.ndarray]) -> dict[str, float | int | list[int]]:
    diff = np.abs(ref.astype(np.float32) - rec.astype(np.float32))
    ref_l = luma(ref)
    rec_l = luma(rec)
    ldiff = np.abs(ref_l - rec_l)
    subject = maps["subject"]
    face = maps["face"]
    jersey = maps["jersey"]
    dark = maps["dark_zone"]
    ref_edges_face = edge_mask(ref, face)
    rec_edges_face = edge_mask(rec, face)
    ref_red = red_mask(ref, subject)
    rec_red = red_mask(rec, subject)
    red_union = ref_red | rec_red
    red_iou = float((ref_red & rec_red).sum() / red_union.sum()) if np.any(red_union) else 0.0

    _, fy0, _, fy1 = bbox(face)
    yy = np.broadcast_to(np.arange(H)[:, None], face.shape)
    rel_y = (yy - fy0) / max(1, fy1 - fy0)
    lower = face & (rel_y > 0.55) & (rel_y < 0.92)
    slab = lower & (rec_l > ref_l + 25.0) & (rec_l > 58.0)
    labels, count = ndi.label(slab, structure=np.ones((3, 3), dtype=bool))
    max_slab = 0
    for label in range(1, count + 1):
        max_slab = max(max_slab, int((labels == label).sum()))
    gray_slab_penalty = float(max_slab / max(1, int(lower.sum())))

    mouth = face & (rel_y > 0.58) & (rel_y < 0.80)
    if np.any(mouth):
        mouth_rows = np.array([rec_l[y][mouth[y]].mean() if np.any(mouth[y]) else np.nan for y in range(H)])
        valid = mouth_rows[np.isfinite(mouth_rows)]
        mouth_banding_penalty = float(np.mean(np.abs(np.diff(valid))) / 32.0) if valid.size > 2 else 0.0
    else:
        mouth_banding_penalty = 0.0

    protected_dark_zone_fill_ratio = float(((rec_l > 46.0) & dark).sum() / max(1, int(dark.sum())))
    return {
        "mae_full_rgb": float(diff.mean()),
        "mae_subject_rgb": float(diff[subject].mean()),
        "mae_face_rgb": float(diff[face].mean()),
        "mae_jersey_rgb": float(diff[jersey].mean()),
        "face_luma_mae": float(ldiff[face].mean()),
        "jersey_luma_mae": float(ldiff[jersey].mean()),
        "edge_overlap_face": overlap(ref_edges_face, rec_edges_face),
        "red_mask_iou": red_iou,
        "gray_slab_penalty": gray_slab_penalty,
        "mouth_banding_penalty": mouth_banding_penalty,
        "protected_dark_zone_fill_ratio": protected_dark_zone_fill_ratio,
        "subject_mask_coverage": float(subject.mean()),
        "face_mask_coverage": float(face.mean()),
        "jersey_mask_coverage": float(jersey.mean()),
        "output_resolution": [W, H],
        "seed": SEED,
    }


def save_l(path: Path, data: np.ndarray) -> None:
    Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), "L").save(path)


def side_by_side(ref: np.ndarray, rec: Image.Image) -> Image.Image:
    out = Image.new("RGB", (W * 2, H), (0, 0, 0))
    out.paste(Image.fromarray(ref, "RGB"), (0, 0))
    out.paste(rec.convert("RGB"), (W, 0))
    d = ImageDraw.Draw(out)
    d.text((18, 18), "reference", font=font(24, True), fill=(245, 245, 245))
    d.text((W + 18, 18), "reference-stencil reconstruction", font=font(24, True), fill=(245, 245, 245))
    return out


def main() -> None:
    t0 = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    ref = load_reference()
    maps = build_target_maps(ref)

    face_micro = draw_micro_texture(maps["face"], maps, rng)
    face_structure = draw_weighted_texture(maps["face"], maps, FACE_WORDS, rng, count=2400, size_range=(9, 20), alpha_range=(66, 168))
    face_contour = draw_contour_texture(maps["face"], maps, rng)
    face_raw = combine_layers([face_micro, face_structure, face_contour])

    jersey_micro = draw_micro_texture(maps["jersey"], maps, rng, jersey=True)
    jersey_structure = draw_weighted_texture(
        maps["jersey"], maps, JERSEY_WORDS, rng, jersey=True, count=1900, size_range=(10, 24), alpha_range=(72, 176)
    )
    jersey_contour = draw_contour_texture(maps["jersey"], maps, rng, jersey=True)
    jersey_raw = combine_layers([jersey_micro, jersey_structure, jersey_contour])

    face_after_luma = lower_face_slab_correction(
        apply_luminance_stencil(face_raw, maps["face"], maps),
        ref,
        maps["face"],
        maps,
    )
    jersey_after_luma = apply_luminance_stencil(jersey_raw, maps["jersey"], maps, jersey=True)

    shoulder_raw = draw_weighted_texture(
        maps["shoulder"], maps, MICRO_WORDS, rng, count=650, size_range=(8, 19), alpha_range=(48, 130)
    )
    shoulder_after_luma = apply_luminance_stencil(shoulder_raw, maps["shoulder"], maps)

    final_arr = np.zeros((H, W, 3), dtype=np.float32)
    final_arr += shoulder_after_luma
    final_arr += jersey_after_luma
    final_arr += face_after_luma
    final_arr[~maps["subject"]] = 0
    final_arr = np.clip(final_arr, 0, 255)
    final_img = Image.fromarray(final_arr.astype(np.uint8), "RGB").convert("RGBA")
    final_img.alpha_composite(render_anchor_layer(maps, maps["subject"]))
    draw_text(final_img, 795, 514, "change the game.", 28, (188, 188, 202), 220)
    final_rgb = final_img.convert("RGB")

    rec = np.array(final_rgb, dtype=np.uint8)
    metrics = compute_metrics(ref, rec, maps)
    metrics["render_time_seconds"] = round(perf_counter() - t0, 3)

    save_l(OUT / "stencil_subject_mask.png", maps["subject"] * 255)
    save_l(OUT / "stencil_face_mask.png", maps["face"] * 255)
    save_l(OUT / "stencil_jersey_mask.png", maps["jersey"] * 255)
    save_l(OUT / "stencil_luminance_map.png", maps["enhanced_luma"] * 255)
    save_l(OUT / "stencil_dark_zone_mask.png", maps["dark_zone"] * 255)
    Image.fromarray(alpha_to_rgb(face_raw).astype(np.uint8), "RGB").save(OUT / "stencil_face_texture_raw.png")
    Image.fromarray(face_after_luma.astype(np.uint8), "RGB").save(OUT / "stencil_face_after_luma.png")
    Image.fromarray(alpha_to_rgb(jersey_raw).astype(np.uint8), "RGB").save(OUT / "stencil_jersey_texture_raw.png")
    final_rgb.save(OUT / "stencil_final.png")
    side_by_side(ref, final_rgb).save(OUT / "stencil_side_by_side.png")
    (OUT / "stencil_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
