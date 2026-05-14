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

# Allow direct execution from repo root.
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
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((23, 23), np.uint8), iterations=2)
    raw = cv2.dilate(raw, np.ones((11, 11), np.uint8), iterations=4)
    raw = _right_components(raw, min_area=900)

    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    for contour in contours:
        if cv2.contourArea(contour) >= 1500:
            hull = cv2.convexHull(contour)
            cv2.drawContours(filled, [hull], contourIdx=-1, color=255, thickness=cv2.FILLED)

    filled = cv2.morphologyEx(
        filled, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8), iterations=2
    )
    filled = cv2.dilate(filled, np.ones((7, 7), np.uint8), iterations=2)
    return _right_components(filled, min_area=2500)


def _build_jersey_mask(rgb: np.ndarray, shape: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    red = (r > g * 1.25) & (r > b * 1.25) & (r > 25) & (shape > 0)
    jersey = red.astype(np.uint8) * 255
    jersey = cv2.morphologyEx(
        jersey, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2
    )
    jersey = cv2.dilate(jersey, np.ones((9, 9), np.uint8), iterations=2)
    return ((jersey > 0) & (shape > 0)).astype(np.uint8) * 255


def _face_color(avg: np.ndarray) -> tuple[int, int, int]:
    r, g, b = avg.astype(np.float32)
    v = int(0.299 * r + 0.587 * g + 0.114 * b)
    if v < 18:
        v = int(v * 1.25) + 20
    else:
        v = int(v * 1.9)
    v = min(245, max(28, v))
    return (v, v, v)


def _jersey_color(avg: np.ndarray, rng: random.Random) -> tuple[int, int, int]:
    r, g, b = avg.astype(np.float32)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    if luma > 135 and rng.random() < 0.4:
        v = min(245, max(170, int(luma * 1.35)))
        return (v, v, v)
    if r < 35:
        return (85, 10, 12)
    return (
        min(255, max(100, int(r * 2.1))),
        min(55, max(6, int(g * 0.35))),
        min(55, max(6, int(b * 0.35))),
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
    angle: float,
) -> None:
    left, top, right, bottom = font.getbbox(text)
    w = max(1, right - left + 8)
    h = max(1, bottom - top + 8)
    patch = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    patch_draw = ImageDraw.Draw(patch)
    patch_draw.text((4 - left, 4 - top), text, font=font, fill=(*fill, 210))
    rotated = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    base.paste(rotated, pos, rotated)


def _draw_alpha_text(
    layer: Image.Image,
    pos: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    alpha: int,
) -> None:
    ImageDraw.Draw(layer).text(pos, text, font=font, fill=(*fill, alpha))


def _draw_anchors(out: Image.Image) -> None:
    layer = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((1350, 150), "MICHAEL JORDAN", fill=(230, 230, 232, 185), font=_load_font(58))
    draw.text((1395, 250), "MVP", fill=(242, 242, 242, 190), font=_load_font(84))
    draw.text((1358, 410), "CHAMPION", fill=(212, 212, 216, 175), font=_load_font(48))
    draw.text((1355, 720), "BULLS", fill=(240, 24, 34, 205), font=_load_font(96))
    draw.text((1515, 835), "23", fill=(248, 238, 215, 205), font=_load_font(126))
    draw.text((610, 505), "change the game.", fill=(178, 178, 188, 205), font=_load_font(56))
    out.alpha_composite(layer)


def _paint_layer(
    out: Image.Image,
    target: np.ndarray,
    shape: np.ndarray,
    jersey: np.ndarray,
    luma: np.ndarray,
    edge: np.ndarray,
    sobel_x: np.ndarray,
    sobel_y: np.ndarray,
    rng: random.Random,
    config: dict[str, int | float | tuple[int, int] | bool],
) -> tuple[int, int, int, float, int]:
    cell_w = int(config["cell_w"])
    cell_h = int(config["cell_h"])
    size_min, size_max = config["font_range"]  # type: ignore[misc]
    alpha = int(config["alpha"])
    rotate = bool(config["rotate"])
    base_p = float(config["base_p"])
    jitter = int(config["jitter"])

    words_drawn = 0
    face_words = 0
    jersey_words = 0
    total_p = 0.0
    samples = 0

    y_offset = rng.randint(0, cell_h - 1)
    x_offset = rng.randint(0, cell_w - 1)
    for y in range(y_offset, OUT_H - cell_h, cell_h):
        for x in range(x_offset, OUT_W - cell_w, cell_w):
            patch_shape = shape[y : y + cell_h, x : x + cell_w]
            if patch_shape.size == 0 or float(np.mean(patch_shape)) < 8.0:
                continue

            patch_luma = float(np.mean(luma[y : y + cell_h, x : x + cell_w]))
            patch_edge = float(np.mean(edge[y : y + cell_h, x : x + cell_w]))
            p = base_p + 0.25 * patch_edge + 0.18 * patch_luma
            p = min(0.98, max(0.42, p))
            total_p += p
            samples += 1
            if rng.random() > p:
                continue

            patch = target[y : y + cell_h, x : x + cell_w]
            avg = np.mean(patch.reshape(-1, 3), axis=0)
            is_jersey = float(np.mean(jersey[y : y + cell_h, x : x + cell_w])) > 18.0
            if is_jersey:
                word = rng.choice(JERSEY_WORDS)
                color = _jersey_color(avg, rng)
                jersey_words += 1
            else:
                word = rng.choice(FACE_WORDS)
                color = _face_color(avg)
                face_words += 1

            size = rng.randint(int(size_min), int(size_max))
            font = _load_font(size)
            px = x + rng.randint(-jitter, jitter)
            py = y + rng.randint(-jitter, jitter)
            if rotate and rng.random() < 0.42:
                angle = _angle_at(sobel_x, sobel_y, min(OUT_W - 1, x), min(OUT_H - 1, y))
                angle += rng.uniform(-6.0, 6.0)
                angle = float(np.clip(angle, -18.0, 18.0))
                _draw_rotated_text(out, (px, py), word, font, color, angle)
            else:
                _draw_alpha_text(out, (px, py), word, font, color, alpha)
            words_drawn += 1

    return words_drawn, face_words, jersey_words, total_p, samples


def _main() -> None:
    t0 = perf_counter()
    rng = random.Random(SEED)
    target_path = _resolve_target()
    target = _load_target_rgb(target_path)
    gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    shape = _build_shape_mask(gray)
    jersey = _build_jersey_mask(target, shape)
    edge = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    luma = gray.astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=5)

    out = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 255))
    layers = [
        {
            "cell_w": 7,
            "cell_h": 5,
            "font_range": (5, 8),
            "alpha": 145,
            "rotate": False,
            "base_p": 0.64,
            "jitter": 5,
        },
        {
            "cell_w": 9,
            "cell_h": 7,
            "font_range": (8, 13),
            "alpha": 170,
            "rotate": False,
            "base_p": 0.56,
            "jitter": 7,
        },
        {
            "cell_w": 18,
            "cell_h": 14,
            "font_range": (14, 24),
            "alpha": 205,
            "rotate": True,
            "base_p": 0.36,
            "jitter": 10,
        },
    ]

    total_words = 0
    face_words = 0
    jersey_words = 0
    total_probability = 0.0
    probability_samples = 0
    for layer_config in layers:
        drawn, face, jersey_count, p_sum, samples = _paint_layer(
            out,
            target,
            shape,
            jersey,
            luma,
            edge,
            sobel_x,
            sobel_y,
            rng,
            layer_config,
        )
        total_words += drawn
        face_words += face
        jersey_words += jersey_count
        total_probability += p_sum
        probability_samples += samples

    _draw_anchors(out)

    out_dir = REPO_ROOT / "examples" / "reference_recreation"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(shape, mode="L").save(out_dir / "shape_mask_v3.png")
    Image.fromarray(jersey, mode="L").save(out_dir / "jersey_mask_v3.png")

    output_rgb = out.convert("RGB")
    output_path = out_dir / "bruteforce_text_paint_v3.png"
    output_rgb.save(output_path)

    side = Image.new("RGB", (OUT_W * 2, OUT_H), color=(0, 0, 0))
    side.paste(Image.fromarray(target, mode="RGB"), (0, 0))
    side.paste(output_rgb, (OUT_W, 0))
    side.save(out_dir / "side_by_side_bruteforce_v3.png")

    metrics = {
        "target_path": str(target_path),
        "total_words_drawn": total_words,
        "face_words_drawn": face_words,
        "jersey_words_drawn": jersey_words,
        "shape_mask_coverage": round(float(np.mean(shape > 0)), 4),
        "jersey_mask_coverage": round(float(np.mean(jersey > 0)), 4),
        "average_placement_probability": round(
            total_probability / max(1, probability_samples), 4
        ),
        "render_time_ms": round((perf_counter() - t0) * 1000.0, 2),
        "seed": SEED,
        "output_resolution": f"{OUT_W}x{OUT_H}",
    }
    (out_dir / "metrics_bruteforce_v3.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    _main()
