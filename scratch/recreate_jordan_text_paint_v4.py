from __future__ import annotations

import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glyphforge.typography.fonts import find_font

OUT_W = 1920
OUT_H = 1080
SEED = 23
RIGHT_SUBJECT_CUTOFF = 0.55

FACE_WORDS = [
    "MVP",
    "CHAMPION",
    "CLUTCH",
    "LEGEND",
    "GOAT",
    "GREATNESS",
    "DEDICATION",
    "DEFENSE",
    "AIR JORDAN",
]
JERSEY_WORDS = ["BULLS", "23", "CHICAGO", "DYNASTY", "SIX RINGS", "FINALS MVP"]


@lru_cache(maxsize=96)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(str(find_font("assets/fonts")), size=size)
    except Exception:
        return ImageFont.load_default()


def _resolve_target() -> Path:
    requested = REPO_ROOT / "reference_img" / "Michael-Jordan-Wallpaper-Desktop-1.jpg"
    if requested.exists():
        return requested
    fallback = REPO_ROOT / "examples" / "reference_recreation" / "side_by_side_final.png"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No target found at reference_img/... or side_by_side_final.png")


def _load_target_rgb(path: Path) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"))
    if path.name == "side_by_side_final.png":
        _h, w = rgb.shape[:2]
        rgb = rgb[:, : w // 2, :]
    return np.array(
        Image.fromarray(rgb).resize((OUT_W, OUT_H), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _right_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx = float(centroids[label][0])
        if area >= min_area and cx > OUT_W * RIGHT_SUBJECT_CUTOFF:
            kept[labels == label] = 255
    return kept


def _build_shape_mask(gray: np.ndarray) -> np.ndarray:
    xs = np.indices(gray.shape)[1]
    raw = ((gray > 3) & (xs > OUT_W * RIGHT_SUBJECT_CUTOFF)).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8), iterations=2)
    raw = cv2.dilate(raw, np.ones((9, 9), np.uint8), iterations=3)
    raw = _right_components(raw, min_area=700)

    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    for contour in contours:
        if cv2.contourArea(contour) >= 1200:
            epsilon = 0.004 * cv2.arcLength(contour, closed=True)
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            cv2.drawContours(filled, [approx], contourIdx=-1, color=255, thickness=cv2.FILLED)

    filled = cv2.morphologyEx(
        filled, cv2.MORPH_CLOSE, np.ones((23, 23), np.uint8), iterations=1
    )
    filled = cv2.dilate(filled, np.ones((5, 5), np.uint8), iterations=1)
    return _right_components(filled, min_area=2200)


def _build_jersey_mask(rgb: np.ndarray, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    red_strength = np.clip((r - np.maximum(g, b)) / 120.0, 0.0, 1.0)
    red = (red_strength > 0.16) & (r > 25) & (shape > 0)
    jersey = red.astype(np.uint8) * 255
    jersey = cv2.morphologyEx(
        jersey, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2
    )
    jersey = cv2.dilate(jersey, np.ones((9, 9), np.uint8), iterations=2)
    jersey = ((jersey > 0) & (shape > 0)).astype(np.uint8) * 255
    red_strength *= (shape > 0).astype(np.float32)
    return jersey, red_strength.astype(np.float32)


def _build_feature_maps(
    rgb: np.ndarray, gray: np.ndarray, shape: np.ndarray, red_strength: np.ndarray
) -> dict[str, np.ndarray]:
    shape_f = (shape > 0).astype(np.float32)
    luma = gray.astype(np.float32) / 255.0
    canny = cv2.Canny(gray, 35, 115).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    if float(np.max(sobel_mag)) > 0:
        sobel_mag = sobel_mag / float(np.max(sobel_mag))
    edge = np.maximum(canny, sobel_mag)
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.0) * shape_f
    highlight = np.clip((luma - 0.34) / 0.42, 0.0, 1.0) * shape_f
    shadow = np.clip((0.24 - luma) / 0.24, 0.0, 1.0) * shape_f
    red_strength = cv2.GaussianBlur(red_strength, (0, 0), sigmaX=1.0) * shape_f
    return {
        "luma": luma,
        "edge": edge.astype(np.float32),
        "highlight": highlight.astype(np.float32),
        "shadow": shadow.astype(np.float32),
        "red_strength": red_strength.astype(np.float32),
        "sobel_x": sobel_x,
        "sobel_y": sobel_y,
    }


def _face_color(avg: np.ndarray, luma: float, edge: float, highlight: float) -> tuple[int, int, int]:
    r, g, b = avg.astype(np.float32)
    v = 0.299 * r + 0.587 * g + 0.114 * b
    if highlight > 0.35 or edge > 0.16:
        v = v * 2.05 + 20
    elif luma < 0.18:
        v = v * 1.15 + 18
    else:
        v = v * 1.55 + 12
    v_i = min(248, max(24, int(v)))
    return (v_i, v_i, v_i)


def _jersey_color(avg: np.ndarray, red_strength: float, highlight: float, rng: random.Random) -> tuple[int, int, int]:
    r, g, b = avg.astype(np.float32)
    if highlight > 0.45 and rng.random() < 0.35:
        v = min(248, max(178, int((r + g + b) / 3 * 1.45)))
        return (v, v, v)
    if red_strength < 0.16:
        return (105, 12, 14)
    return (
        min(255, max(120, int(r * 2.25))),
        min(50, max(4, int(g * 0.3))),
        min(50, max(4, int(b * 0.3))),
    )


def _angle_at(sobel_x: np.ndarray, sobel_y: np.ndarray, x: int, y: int) -> float:
    gx = float(sobel_x[y, x])
    gy = float(sobel_y[y, x])
    if abs(gx) + abs(gy) < 0.01:
        return 0.0
    angle = math.degrees(math.atan2(gy, gx)) + 90.0
    while angle < -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return float(np.clip(angle, -18.0, 18.0))


def _draw_rotated_text(
    base: Image.Image,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    alpha: int,
    angle: float,
) -> None:
    left, top, right, bottom = font.getbbox(text)
    w = max(1, right - left + 8)
    h = max(1, bottom - top + 8)
    patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((4 - left, 4 - top), text, font=font, fill=(*fill, alpha))
    rotated = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    base.paste(rotated, pos, rotated)


def _text_alpha(luma: float, edge: float, highlight: float, shadow: float, red_strength: float, base: int) -> int:
    alpha = base + int(highlight * 80) + int(edge * 55) + int(red_strength * 45) - int(shadow * 70)
    if luma < 0.12 and edge < 0.08 and red_strength < 0.12:
        alpha -= 45
    return min(245, max(35, alpha))


def _placement_probability(
    luma: float, edge: float, highlight: float, shadow: float, red_strength: float, base: float
) -> float:
    p = base + 0.36 * edge + 0.26 * highlight + 0.34 * red_strength - 0.26 * shadow
    if luma < 0.13 and edge < 0.08 and red_strength < 0.1:
        p -= 0.22
    return min(0.96, max(0.08, p))


def _paint_feature_layer(
    out: Image.Image,
    target: np.ndarray,
    shape: np.ndarray,
    jersey: np.ndarray,
    maps: dict[str, np.ndarray],
    rng: random.Random,
    config: dict[str, int | float | tuple[int, int] | bool],
) -> tuple[int, int, int, int, float, float, int]:
    cell_w = int(config["cell_w"])
    cell_h = int(config["cell_h"])
    size_min, size_max = config["font_range"]  # type: ignore[misc]
    base_alpha = int(config["alpha"])
    base_p = float(config["base_p"])
    jitter = int(config["jitter"])
    rotate = bool(config["rotate"])

    total = micro = jersey_count = face_count = 0
    alpha_sum = 0.0
    p_sum = 0.0
    samples = 0

    y_offset = rng.randint(0, cell_h - 1)
    x_offset = rng.randint(0, cell_w - 1)
    for y in range(y_offset, OUT_H - cell_h, cell_h):
        for x in range(x_offset, OUT_W - cell_w, cell_w):
            patch_shape = shape[y : y + cell_h, x : x + cell_w]
            if patch_shape.size == 0 or float(np.mean(patch_shape)) < 8.0:
                continue

            patch_luma = float(np.mean(maps["luma"][y : y + cell_h, x : x + cell_w]))
            patch_edge = float(np.mean(maps["edge"][y : y + cell_h, x : x + cell_w]))
            patch_high = float(np.mean(maps["highlight"][y : y + cell_h, x : x + cell_w]))
            patch_shadow = float(np.mean(maps["shadow"][y : y + cell_h, x : x + cell_w]))
            patch_red = float(np.mean(maps["red_strength"][y : y + cell_h, x : x + cell_w]))
            p = _placement_probability(patch_luma, patch_edge, patch_high, patch_shadow, patch_red, base_p)
            p_sum += p
            samples += 1
            if rng.random() > p:
                continue

            patch = target[y : y + cell_h, x : x + cell_w]
            avg = np.mean(patch.reshape(-1, 3), axis=0)
            is_jersey = float(np.mean(jersey[y : y + cell_h, x : x + cell_w])) > 18.0
            if is_jersey:
                word = rng.choice(JERSEY_WORDS)
                color = _jersey_color(avg, patch_red, patch_high, rng)
                jersey_count += 1
            else:
                word = rng.choice(FACE_WORDS)
                color = _face_color(avg, patch_luma, patch_edge, patch_high)
                face_count += 1

            size = rng.randint(int(size_min), int(size_max))
            alpha = _text_alpha(patch_luma, patch_edge, patch_high, patch_shadow, patch_red, base_alpha)
            font = _load_font(size)
            px = x + rng.randint(-jitter, jitter)
            py = y + rng.randint(-jitter, jitter)
            if rotate and rng.random() < 0.28:
                angle = _angle_at(maps["sobel_x"], maps["sobel_y"], min(OUT_W - 1, x), min(OUT_H - 1, y))
                angle = float(np.clip(angle + rng.uniform(-5.0, 5.0), -18.0, 18.0))
                _draw_rotated_text(out, (px, py), word, font, color, alpha, angle)
            else:
                ImageDraw.Draw(out).text((px, py), word, font=font, fill=(*color, alpha))
            total += 1
            micro += 1
            alpha_sum += alpha

    return total, micro, face_count, jersey_count, alpha_sum, p_sum, samples if samples else 1


def _paint_contour_pass(
    out: Image.Image,
    target: np.ndarray,
    shape: np.ndarray,
    jersey: np.ndarray,
    maps: dict[str, np.ndarray],
    rng: random.Random,
) -> tuple[int, int, int, float]:
    strong = (
        (maps["edge"] > 0.13)
        & (shape > 0)
        & ((maps["highlight"] > 0.08) | (maps["red_strength"] > 0.08) | (maps["shadow"] > 0.18))
    )
    ys, xs = np.where(strong)
    coords = list(zip(xs.tolist(), ys.tolist()))
    rng.shuffle(coords)
    coords = coords[:2200]

    contour_words = 0
    jersey_words = 0
    alpha_sum = 0.0
    for x, y in coords:
        if rng.random() > 0.58:
            continue
        x0 = max(0, x - 5)
        y0 = max(0, y - 5)
        x1 = min(OUT_W, x + 6)
        y1 = min(OUT_H, y + 6)
        avg = np.mean(target[y0:y1, x0:x1].reshape(-1, 3), axis=0)
        luma = float(maps["luma"][y, x])
        edge = float(maps["edge"][y, x])
        high = float(maps["highlight"][y, x])
        shadow = float(maps["shadow"][y, x])
        red = float(maps["red_strength"][y, x])
        is_jersey = bool(jersey[y, x] > 0)
        if is_jersey:
            word = rng.choice(JERSEY_WORDS)
            color = _jersey_color(avg, red, high, rng)
            size = rng.randint(11, 20)
            jersey_words += 1
        else:
            word = rng.choice(FACE_WORDS)
            color = _face_color(avg, luma, edge, high)
            size = rng.randint(10, 18)
        alpha = _text_alpha(luma, edge, high, shadow, red, 165)
        angle = _angle_at(maps["sobel_x"], maps["sobel_y"], x, y)
        angle = float(np.clip(angle + rng.uniform(-4.0, 4.0), -18.0, 18.0))
        _draw_rotated_text(
            out,
            (x + rng.randint(-8, 8), y + rng.randint(-8, 8)),
            word,
            _load_font(size),
            color,
            alpha,
            angle,
        )
        contour_words += 1
        alpha_sum += alpha

    return contour_words, jersey_words, len(coords), alpha_sum


def _draw_anchors(out: Image.Image) -> int:
    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((1350, 150), "MICHAEL JORDAN", fill=(232, 232, 234, 220), font=_load_font(58))
    draw.text((1390, 248), "MVP", fill=(245, 245, 245, 225), font=_load_font(88))
    draw.text((1352, 408), "CHAMPION", fill=(220, 220, 224, 210), font=_load_font(50))
    draw.text((1328, 676), "BULLS", fill=(244, 22, 34, 235), font=_load_font(136))
    draw.text((1508, 800), "23", fill=(250, 242, 222, 235), font=_load_font(174))
    draw.text((610, 505), "change the game.", fill=(178, 178, 188, 215), font=_load_font(56))
    out.alpha_composite(layer)
    return 6


def _main() -> None:
    t0 = perf_counter()
    rng = random.Random(SEED)
    target_path = _resolve_target()
    target = _load_target_rgb(target_path)
    gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    shape = _build_shape_mask(gray)
    jersey, red_strength = _build_jersey_mask(target, shape)
    maps = _build_feature_maps(target, gray, shape, red_strength)

    out = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 255))
    layer_configs = [
        {"cell_w": 7, "cell_h": 5, "font_range": (5, 8), "alpha": 115, "base_p": 0.34, "jitter": 7, "rotate": False},
        {"cell_w": 10, "cell_h": 7, "font_range": (8, 13), "alpha": 145, "base_p": 0.28, "jitter": 10, "rotate": False},
        {"cell_w": 18, "cell_h": 13, "font_range": (14, 24), "alpha": 175, "base_p": 0.16, "jitter": 13, "rotate": True},
    ]

    total_words = 0
    micro_words = 0
    jersey_words = 0
    alpha_sum = 0.0
    p_sum = 0.0
    p_samples = 0
    for config in layer_configs:
        total, micro, _face, jersey_count, alpha, p_total, samples = _paint_feature_layer(
            out, target, shape, jersey, maps, rng, config
        )
        total_words += total
        micro_words += micro
        jersey_words += jersey_count
        alpha_sum += alpha
        p_sum += p_total
        p_samples += samples

    contour_words, contour_jersey, edge_pixels_sampled, contour_alpha = _paint_contour_pass(
        out, target, shape, jersey, maps, rng
    )
    total_words += contour_words
    jersey_words += contour_jersey
    alpha_sum += contour_alpha

    anchor_words = _draw_anchors(out)
    total_words += anchor_words
    alpha_sum += anchor_words * 225

    out_dir = REPO_ROOT / "examples" / "reference_recreation"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(shape, mode="L").save(out_dir / "shape_mask_v4.png")
    Image.fromarray(jersey, mode="L").save(out_dir / "jersey_mask_v4.png")
    edge_u8 = np.clip(maps["edge"] * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(edge_u8, mode="L").save(out_dir / "edge_map_v4.png")

    output_rgb = out.convert("RGB")
    output_path = out_dir / "bruteforce_text_paint_v4.png"
    output_rgb.save(output_path)

    side = Image.new("RGB", (OUT_W * 2, OUT_H), color=(0, 0, 0))
    side.paste(Image.fromarray(target, mode="RGB"), (0, 0))
    side.paste(output_rgb, (OUT_W, 0))
    side.save(out_dir / "side_by_side_bruteforce_v4.png")

    metrics = {
        "target_path": str(target_path),
        "total_words_drawn": total_words,
        "micro_words_drawn": micro_words,
        "contour_words_drawn": contour_words,
        "jersey_words_drawn": jersey_words,
        "anchor_words_drawn": anchor_words,
        "average_alpha": round(alpha_sum / max(1, total_words), 2),
        "edge_pixels_sampled": edge_pixels_sampled,
        "average_placement_probability": round(p_sum / max(1, p_samples), 4),
        "shape_mask_coverage": round(float(np.mean(shape > 0)), 4),
        "jersey_mask_coverage": round(float(np.mean(jersey > 0)), 4),
        "render_time_ms": round((perf_counter() - t0) * 1000.0, 2),
        "seed": SEED,
        "output_resolution": f"{OUT_W}x{OUT_H}",
    }
    (out_dir / "metrics_bruteforce_v4.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    _main()
