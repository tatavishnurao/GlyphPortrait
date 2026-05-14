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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glyphforge.typography.fonts import find_font

OUT_W = 1920
OUT_H = 1080
SEED = 2301
RIGHT_CUTOFF = 0.54
SUBJECT_SHIFT_X = -60

FACE_WORDS = [
    "Air Jordan",
    "Michael",
    "Jordan",
    "MVP",
    "Champion",
    "clutch",
    "defense",
    "finals",
    "legend",
    "greatness",
    "dedication",
    "rookie",
    "scoring",
    "six rings",
    "flight",
    "winner",
    "Chicago",
    "intensity",
]
JERSEY_WORDS = [
    "BULLS",
    "23",
    "Chicago",
    "red",
    "dynasty",
    "finals MVP",
    "champion",
    "six rings",
    "black red",
    "game winner",
]
SHADOW_WORDS = FACE_WORDS + JERSEY_WORDS + ["NBA", "drive", "focus", "prime"]
ANCHORS = [
    ("MICHAEL JORDAN", 1284, 178, 38, (210, 210, 214), 160, -4.0, "head"),
    ("MVP", 1368, 266, 58, (225, 225, 228), 172, -2.0, "head"),
    ("CHAMPION", 1280, 434, 34, (202, 202, 208), 154, -5.0, "head"),
    ("BULLS", 1238, 700, 130, (204, 18, 28), 192, -10.0, "jersey"),
    ("23", 1440, 826, 165, (224, 216, 198), 196, -7.0, "jersey"),
]


def _target_path() -> Path:
    path = REPO_ROOT / "reference_img" / "Michael-Jordan-Wallpaper-Desktop-1.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_target() -> np.ndarray:
    rgb = np.array(
        Image.open(_target_path()).convert("RGB").resize((OUT_W, OUT_H), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    shifted = np.zeros_like(rgb)
    dx = abs(SUBJECT_SHIFT_X)
    shifted[:, : OUT_W - dx] = rgb[:, dx:]
    return shifted


def _load_reference_target() -> np.ndarray:
    return np.array(
        Image.open(_target_path()).convert("RGB").resize((OUT_W, OUT_H), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


@lru_cache(maxsize=96)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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
            Path("/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf"),
            find_font(REPO_ROOT / "assets" / "fonts"),
        ]
    )
    for candidate in candidates:
        if candidate and candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0].astype(np.float32) * 0.299
        + rgb[..., 1].astype(np.float32) * 0.587
        + rgb[..., 2].astype(np.float32) * 0.114
    )


def _largest_right_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)

    xs_grid = np.broadcast_to(np.arange(mask.shape[1]), mask.shape)
    best_label = 0
    best_score = -1.0
    for label in range(1, count + 1):
        region = labels == label
        area = int(region.sum())
        if area < 500:
            continue
        cx = float(xs_grid[region].mean())
        if cx < OUT_W * RIGHT_CUTOFF:
            continue
        score = area * (1.0 + (cx / OUT_W - RIGHT_CUTOFF))
        if score > best_score:
            best_score = score
            best_label = label
    return labels == best_label if best_label else np.zeros(mask.shape, dtype=bool)


def _smooth_mask(mask: np.ndarray, radius: float = 2.0) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), "L").filter(ImageFilter.GaussianBlur(radius))
    return np.array(img) > 88


def _build_shape_mask(gray: np.ndarray) -> np.ndarray:
    xs = np.broadcast_to(np.arange(OUT_W), gray.shape)
    raw = (gray > 12) & (xs > int(OUT_W * RIGHT_CUTOFF))
    raw = ndi.binary_closing(raw, structure=np.ones((23, 23), dtype=bool), iterations=2)
    raw = ndi.binary_dilation(raw, structure=np.ones((9, 9), dtype=bool), iterations=2)
    raw = _largest_right_component(raw)
    filled = ndi.binary_fill_holes(raw)
    filled = ndi.binary_closing(filled, structure=np.ones((27, 27), dtype=bool), iterations=1)
    filled = ndi.binary_dilation(filled, structure=np.ones((5, 5), dtype=bool), iterations=1)
    filled = _smooth_mask(filled, radius=2.4)
    filled = _largest_right_component(filled)
    return (filled.astype(np.uint8) * 255)


def _build_jersey_mask(rgb: np.ndarray, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    red_strength = np.clip((r - np.maximum(g, b)) / 130.0, 0.0, 1.0)
    jersey = (red_strength > 0.13) & (r > 22) & (shape > 0)
    jersey = ndi.binary_closing(jersey, structure=np.ones((17, 17), dtype=bool), iterations=2)
    jersey = ndi.binary_dilation(jersey, structure=np.ones((11, 11), dtype=bool), iterations=2)
    jersey = ndi.binary_fill_holes(jersey) & (shape > 0)
    return jersey.astype(np.uint8) * 255, (red_strength * (shape > 0)).astype(np.float32)


def _build_maps(rgb: np.ndarray, shape: np.ndarray) -> dict[str, np.ndarray]:
    gray = _luma(rgb)
    tone = gray / 255.0
    blurred = ndi.gaussian_filter(gray, sigma=1.1)
    sx = ndi.sobel(blurred, axis=1)
    sy = ndi.sobel(blurred, axis=0)
    mag = np.hypot(sx, sy)
    mag = mag / max(1e-6, float(mag.max()))
    canny_like = mag > np.percentile(mag[shape > 0], 76)
    canny_like = ndi.binary_dilation(canny_like, structure=np.ones((3, 3), dtype=bool), iterations=1)
    edge = np.maximum(mag, canny_like.astype(np.float32))
    upper_boost = np.linspace(1.45, 0.82, OUT_H, dtype=np.float32)[:, None]
    feature = ndi.gaussian_filter(edge * upper_boost, sigma=0.7) * (shape > 0)
    feature = np.clip(feature / max(1e-6, np.percentile(feature[shape > 0], 99)), 0.0, 1.0)
    highlight = np.clip((tone - 0.34) / 0.48, 0.0, 1.0) * (shape > 0)
    shadow = np.clip((0.25 - tone) / 0.25, 0.0, 1.0) * (shape > 0)
    return {
        "gray": gray.astype(np.float32),
        "tone": tone.astype(np.float32),
        "feature": feature.astype(np.float32),
        "highlight": highlight.astype(np.float32),
        "shadow": shadow.astype(np.float32),
        "sx": sx.astype(np.float32),
        "sy": sy.astype(np.float32),
    }


def _draw_text(
    layer: Image.Image,
    x: int,
    y: int,
    text: str,
    size: int,
    fill: tuple[int, int, int],
    alpha: int,
    angle: float = 0.0,
    bold: bool = False,
) -> tuple[int, int, int, int]:
    font = _font(size, bold)
    bbox = font.getbbox(text)
    w = max(1, bbox[2] - bbox[0] + 10)
    h = max(1, bbox[3] - bbox[1] + 10)
    patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((5 - bbox[0], 5 - bbox[1]), text, font=font, fill=(*fill, alpha))
    if abs(angle) > 0.2:
        patch = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    layer.alpha_composite(patch, (x, y))
    return (x, y, x + patch.size[0], y + patch.size[1])


def _clip_layer_to_mask(layer: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(layer, dtype=np.uint8)
    arr[..., 3] = np.minimum(arr[..., 3], mask)
    return Image.fromarray(arr, "RGBA")


def _tangent_angle(maps: dict[str, np.ndarray], x: int, y: int) -> float:
    gx = float(maps["sx"][y, x])
    gy = float(maps["sy"][y, x])
    if abs(gx) + abs(gy) < 0.01:
        return 0.0
    angle = math.degrees(math.atan2(gy, gx)) + 90.0
    while angle < -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return float(np.clip(angle, -23, 23))


def _local_rgb(target: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    x0, y0 = max(0, x - radius), max(0, y - radius)
    x1, y1 = min(OUT_W, x + radius + 1), min(OUT_H, y + radius + 1)
    return target[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


def _gray_color(target: np.ndarray, tone: float, x: int, y: int) -> tuple[int, int, int]:
    avg = _local_rgb(target, x, y, 4)
    v = int(np.clip(avg.mean() * 1.84 + tone * 60 + 8, 34, 238))
    return (v, v, min(255, v + 4))


def _face_focus_mask(shape: np.ndarray, jersey: np.ndarray) -> np.ndarray:
    yy = np.broadcast_to(np.arange(OUT_H)[:, None], shape.shape)
    xx = np.broadcast_to(np.arange(OUT_W), shape.shape)
    non_jersey = (shape > 0) & (jersey == 0)
    head_and_neck = yy < 720
    torso_overlap = (yy < 850) & (xx > 1040) & (xx < 1530)
    return (non_jersey & (head_and_neck | torso_overlap)).astype(np.uint8) * 255


def _jersey_color(target: np.ndarray, red_strength: float, tone: float, x: int, y: int, rng: random.Random) -> tuple[int, int, int]:
    avg = _local_rgb(target, x, y, 5)
    if tone > 0.58 and rng.random() < 0.22:
        v = int(np.clip(avg.mean() * 1.7, 184, 255))
        return (v, v, v)
    red = int(np.clip(avg[0] * (1.55 + red_strength), 90, 255))
    return (red, int(np.clip(avg[1] * 0.30, 4, 42)), int(np.clip(avg[2] * 0.30, 4, 42)))


def _valid_point(mask: np.ndarray, x: int, y: int) -> bool:
    return 0 <= x < OUT_W and 0 <= y < OUT_H and mask[y, x] > 0


def _render_shadow(target: np.ndarray, shape: np.ndarray, rng: random.Random) -> tuple[Image.Image, int, int]:
    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    count = 0
    alpha_sum = 0
    for y in range(rng.randrange(0, 5), OUT_H, 6):
        step = rng.choice((7, 8, 9))
        for x in range(int(OUT_W * RIGHT_CUTOFF) + rng.randrange(0, 9), OUT_W, step):
            jx = x + rng.randint(-5, 5)
            jy = y + rng.randint(-4, 4)
            if not _valid_point(shape, jx, jy) or rng.random() > 0.72:
                continue
            tone = float(_luma(target[jy : jy + 1, jx : jx + 1])[0, 0] / 255.0)
            alpha = int(np.clip(35 + (1.0 - tone) * 38 + rng.randint(-5, 8), 35, 75))
            v = int(np.clip(18 + tone * 42, 16, 70))
            _draw_text(layer, jx, jy, rng.choice(SHADOW_WORDS), rng.randint(5, 8), (v, v, v + 2), alpha)
            count += 1
            alpha_sum += alpha
    return _clip_layer_to_mask(layer, shape), count, alpha_sum


def _render_tone(target: np.ndarray, shape: np.ndarray, maps: dict[str, np.ndarray], rng: random.Random) -> tuple[Image.Image, int, int]:
    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    count = 0
    alpha_sum = 0
    for y in range(rng.randrange(0, 9), OUT_H - 10, 9):
        x = int(OUT_W * RIGHT_CUTOFF) + rng.randrange(0, 13)
        while x < OUT_W - 8:
            jx = x + rng.randint(-8, 8)
            jy = y + rng.randint(-7, 7)
            x += rng.choice((10, 12, 14, 16))
            if not _valid_point(shape, jx, jy):
                continue
            tone = float(maps["tone"][jy, jx])
            edge = float(maps["feature"][jy, jx])
            p = 0.24 + tone * 0.38 + edge * 0.28
            if rng.random() > p:
                continue
            alpha = int(np.clip(60 + tone * 74 + edge * 35 + rng.randint(-12, 14), 60, 150))
            _draw_text(
                layer,
                jx,
                jy,
                rng.choice(FACE_WORDS),
                rng.randint(8, 13),
                _gray_color(target, tone, jx, jy),
                alpha,
                rng.uniform(-4.0, 4.0) if rng.random() < 0.18 else 0.0,
            )
            count += 1
            alpha_sum += alpha
    return _clip_layer_to_mask(layer, shape), count, alpha_sum


def _render_face_texture(target: np.ndarray, face: np.ndarray, maps: dict[str, np.ndarray], rng: random.Random) -> tuple[Image.Image, int, int]:
    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    count = 0
    alpha_sum = 0
    ys, xs = np.where(face > 0)
    if xs.size == 0:
        return layer, 0, 0

    y_min, y_max = max(0, int(ys.min()) - 8), min(OUT_H, int(ys.max()) + 8)
    x_min, x_max = max(0, int(xs.min()) - 8), min(OUT_W, int(xs.max()) + 8)
    for y in range(y_min + rng.randrange(0, 8), y_max, 8):
        x = x_min + rng.randrange(0, 10)
        while x < x_max:
            jx = x + rng.randint(-8, 8)
            jy = y + rng.randint(-6, 6)
            x += rng.choice((10, 11, 13, 15))
            if not _valid_point(face, jx, jy):
                continue
            tone = float(maps["tone"][jy, jx])
            feature = float(maps["feature"][jy, jx])
            # Dark face regions need visible typography, not empty black.
            p = 0.46 + feature * 0.38 + max(0.0, 0.24 - tone) * 1.25
            if rng.random() > min(0.88, p):
                continue
            word = rng.choice(FACE_WORDS)
            size = rng.randint(9, 17)
            base = 76 + int(feature * 38) + int((1.0 - tone) * 28)
            alpha = int(np.clip(base + rng.randint(-10, 16), 68, 150))
            v = int(np.clip(62 + tone * 108 + feature * 44 + rng.randint(-8, 14), 58, 194))
            angle = _tangent_angle(maps, jx, jy) + rng.uniform(-6, 6) if feature > 0.16 else rng.uniform(-5, 5)
            _draw_text(layer, jx, jy, word, size, (v, v, min(255, v + 5)), alpha, angle, bold=feature > 0.22)
            count += 1
            alpha_sum += alpha
    return _clip_layer_to_mask(layer, face), count, alpha_sum


def _render_contour(target: np.ndarray, shape: np.ndarray, jersey: np.ndarray, red_strength: np.ndarray, maps: dict[str, np.ndarray], rng: random.Random) -> tuple[Image.Image, int, int, int]:
    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    feature = maps["feature"]
    weighted = np.argwhere((feature > 0.16) & (shape > 0))
    rng.shuffle(weighted)
    count = 0
    alpha_sum = 0
    sampled = 0
    for y, x in weighted[:7600]:
        sampled += 1
        if rng.random() > min(0.78, 0.23 + float(feature[y, x]) * 0.84):
            continue
        x = int(x + rng.randint(-9, 9))
        y = int(y + rng.randint(-9, 9))
        if not _valid_point(shape, x, y):
            continue
        tone = float(maps["tone"][y, x])
        red = float(red_strength[y, x])
        is_jersey = jersey[y, x] > 0
        word = rng.choice(JERSEY_WORDS if is_jersey else FACE_WORDS)
        size = rng.randint(12, 24 if is_jersey else 21)
        alpha = int(np.clip(120 + float(feature[y, x]) * 86 + tone * 34 + rng.randint(-15, 18), 120, 220))
        color = _jersey_color(target, red, tone, x, y, rng) if is_jersey else _gray_color(target, tone, x, y)
        _draw_text(layer, x, y, word, size, color, alpha, _tangent_angle(maps, x, y) + rng.uniform(-5, 5), bold=True)
        count += 1
        alpha_sum += alpha
    return _clip_layer_to_mask(layer, shape), count, sampled, alpha_sum


def _render_jersey(target: np.ndarray, jersey: np.ndarray, red_strength: np.ndarray, maps: dict[str, np.ndarray], rng: random.Random) -> tuple[Image.Image, int, int]:
    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    count = 0
    alpha_sum = 0
    ys, xs = np.where(jersey > 0)
    if xs.size == 0:
        return layer, 0, 0
    for y in range(max(0, int(ys.min()) - 4) + rng.randrange(0, 11), min(OUT_H, int(ys.max()) + 1), 12):
        for x in range(max(0, int(xs.min()) - 10) + rng.randrange(0, 17), OUT_W - 10, rng.choice((14, 16, 18, 20))):
            jx = x + rng.randint(-12, 12)
            jy = y + rng.randint(-9, 9)
            if not _valid_point(jersey, jx, jy) or rng.random() > 0.62:
                continue
            tone = float(maps["tone"][jy, jx])
            red = float(red_strength[jy, jx])
            alpha = int(np.clip(92 + red * 78 + tone * 40 + rng.randint(-14, 18), 72, 180))
            _draw_text(
                layer,
                jx,
                jy,
                rng.choice(JERSEY_WORDS),
                rng.randint(11, 22),
                _jersey_color(target, red, tone, jx, jy, rng),
                alpha,
                rng.uniform(-8, 8),
                bold=rng.random() < 0.35,
            )
            count += 1
            alpha_sum += alpha
    return _clip_layer_to_mask(layer, jersey), count, alpha_sum


def _render_anchors(shape: np.ndarray) -> tuple[Image.Image, int, int]:
    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    alpha_sum = 0
    for text, x, y, size, color, alpha, angle, _kind in ANCHORS:
        _draw_text(layer, x, y, text, size, color, alpha, angle, bold=True)
        alpha_sum += alpha
    return _clip_layer_to_mask(layer, shape), len(ANCHORS), alpha_sum


def _anchor_reserve_mask(kind: str | None = None) -> np.ndarray:
    reserve = np.zeros((OUT_H, OUT_W), dtype=np.uint8)
    for text, x, y, size, _color, _alpha, angle, anchor_kind in ANCHORS:
        if kind is not None and anchor_kind != kind:
            continue
        scratch = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
        _draw_text(scratch, x, y, text, size, (255, 255, 255), 255, angle, bold=True)
        alpha = np.array(scratch)[..., 3] > 0
        alpha = ndi.binary_dilation(alpha, structure=np.ones((15, 15), dtype=bool), iterations=1)
        reserve[alpha] = 255
    return reserve


def _reduce_under_anchor(layer: Image.Image, reserve: np.ndarray, multiplier: float) -> Image.Image:
    arr = np.array(layer, dtype=np.uint8)
    reserved = reserve > 0
    arr[..., 3][reserved] = (arr[..., 3][reserved].astype(np.float32) * multiplier).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _render_jersey_anchor_grain(
    target: np.ndarray,
    jersey_anchor_mask: np.ndarray,
    jersey: np.ndarray,
    red_strength: np.ndarray,
    maps: dict[str, np.ndarray],
    rng: random.Random,
) -> tuple[Image.Image, int, int]:
    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    mask = ((jersey_anchor_mask > 0) & (jersey > 0)).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return layer, 0, 0
    count = 0
    alpha_sum = 0
    for y in range(max(0, int(ys.min()) - 4) + rng.randrange(0, 7), min(OUT_H, int(ys.max()) + 5), 7):
        for x in range(max(0, int(xs.min()) - 4) + rng.randrange(0, 11), min(OUT_W, int(xs.max()) + 5), rng.choice((9, 11, 13))):
            jx = x + rng.randint(-6, 6)
            jy = y + rng.randint(-5, 5)
            if not _valid_point(mask, jx, jy) or rng.random() > 0.62:
                continue
            tone = float(maps["tone"][jy, jx])
            red = float(red_strength[jy, jx])
            alpha = int(np.clip(60 + red * 58 + tone * 38 + rng.randint(-10, 15), 52, 138))
            color = _jersey_color(target, red, tone, jx, jy, rng)
            if rng.random() < 0.28:
                v = int(np.clip(95 + tone * 130, 95, 218))
                color = (v, v, v)
            _draw_text(layer, jx, jy, rng.choice(JERSEY_WORDS), rng.randint(6, 12), color, alpha, rng.uniform(-8, 8))
            count += 1
            alpha_sum += alpha
    return _clip_layer_to_mask(layer, mask), count, alpha_sum


def _composite_on_black(layers: list[Image.Image]) -> Image.Image:
    out = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 255))
    for layer in layers:
        out.alpha_composite(layer)
    return out


def _final_blend(rendered: Image.Image, target: np.ndarray, shape: np.ndarray, face: np.ndarray, maps: dict[str, np.ndarray]) -> Image.Image:
    arr = np.array(rendered.convert("RGB"), dtype=np.float32)
    tone = maps["tone"][..., None]
    mask = (shape > 0)[..., None]
    face_mask = (face > 0)[..., None]
    boost = 0.54 + tone * 1.16
    boost += maps["feature"][..., None] * 0.24
    face_boost = 0.76 + tone * 0.76 + maps["feature"][..., None] * 0.17
    boost = np.where(face_mask, face_boost, boost)
    boost = np.clip(boost, 0.32, 1.55)
    arr = np.where(mask, arr * boost, arr)
    target_f = target.astype(np.float32)
    highlight = ((tone > 0.58) & mask & ~face_mask).astype(np.float32)
    face_highlight = ((tone > 0.62) & face_mask).astype(np.float32)
    arr = arr * (1.0 - highlight * 0.16) + target_f * (highlight * 0.16)
    arr = arr * (1.0 - face_highlight * 0.05) + target_f * (face_highlight * 0.05)
    deep = ((tone < 0.095) & mask & ~face_mask).astype(np.float32)
    face_deep = ((tone < 0.15) & face_mask).astype(np.float32)
    arr *= 1.0 - deep * 0.28
    arr *= 1.0 - face_deep * 0.10
    arr[~mask.repeat(3, axis=2)] = 0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _save_map(path: Path, data: np.ndarray) -> None:
    Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), "L").save(path)


def main() -> None:
    t0 = perf_counter()
    rng = random.Random(SEED)
    out_dir = REPO_ROOT / "examples" / "reference_recreation"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = _load_target()
    gray = _luma(target)
    shape = _build_shape_mask(gray)
    jersey, red_strength = _build_jersey_mask(target, shape)
    face = _face_focus_mask(shape, jersey)
    maps = _build_maps(target, shape)

    pass_a, shadow_words, shadow_alpha = _render_shadow(target, shape, rng)
    pass_b, tone_words, tone_alpha = _render_tone(target, shape, maps, rng)
    pass_face, face_words, face_alpha = _render_face_texture(target, face, maps, rng)
    pass_c, contour_words, edge_pixels_sampled, contour_alpha = _render_contour(
        target, shape, jersey, red_strength, maps, rng
    )
    pass_d, jersey_words, jersey_alpha = _render_jersey(target, jersey, red_strength, maps, rng)
    pass_e, anchor_words, anchor_alpha = _render_anchors(shape)
    head_reserve = _anchor_reserve_mask("head")
    jersey_reserve = _anchor_reserve_mask("jersey")
    pass_anchor_grain, anchor_grain_words, anchor_grain_alpha = _render_jersey_anchor_grain(
        target, jersey_reserve, jersey, red_strength, maps, rng
    )
    pass_a = _reduce_under_anchor(pass_a, head_reserve, 0.68)
    pass_b = _reduce_under_anchor(pass_b, head_reserve, 0.72)
    pass_c = _reduce_under_anchor(pass_c, head_reserve, 0.78)

    slogan_layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    _draw_text(slogan_layer, 795, 514, "change the game.", 28, (188, 188, 202), 225)

    pre_final = _composite_on_black([pass_a, pass_b, pass_face, pass_c, pass_d, pass_e, pass_anchor_grain, slogan_layer])
    final = _final_blend(pre_final, target, shape, face, maps)
    final.alpha_composite(slogan_layer) if final.mode == "RGBA" else None

    final_rgba = final.convert("RGBA")
    final_rgba.alpha_composite(slogan_layer)
    final = final_rgba.convert("RGB")

    side = Image.new("RGB", (OUT_W * 2, OUT_H), (0, 0, 0))
    side.paste(Image.fromarray(_load_reference_target(), "RGB"), (0, 0))
    side.paste(final, (OUT_W, 0))

    final.save(out_dir / "one_shot_v2_final.png")
    side.save(out_dir / "one_shot_v2_side_by_side.png")

    alpha_sum = shadow_alpha + tone_alpha + face_alpha + contour_alpha + jersey_alpha + anchor_alpha + anchor_grain_alpha
    jersey_words += anchor_grain_words
    total_words = shadow_words + tone_words + face_words + contour_words + jersey_words + anchor_words
    metrics = {
        "target_path": str(_target_path()),
        "total_words_drawn": total_words,
        "shadow_words": shadow_words,
        "tone_words": tone_words,
        "face_texture_words": face_words,
        "contour_words": contour_words,
        "jersey_words": jersey_words,
        "jersey_anchor_grain_words": anchor_grain_words,
        "anchor_words": anchor_words,
        "shape_mask_coverage": round(float(np.mean(shape > 0)), 5),
        "jersey_mask_coverage": round(float(np.mean(jersey > 0)), 5),
        "face_mask_coverage": round(float(np.mean(face > 0)), 5),
        "average_alpha": round(alpha_sum / max(1, total_words), 2),
        "render_time": round(perf_counter() - t0, 3),
        "edge_pixels_sampled": edge_pixels_sampled,
        "seed": SEED,
        "subject_shift_x": SUBJECT_SHIFT_X,
        "output_resolution": [OUT_W, OUT_H],
    }
    (out_dir / "one_shot_v2_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
