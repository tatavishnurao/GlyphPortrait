from __future__ import annotations

import json
import random
import sys
from functools import lru_cache
from pathlib import Path

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


@lru_cache(maxsize=64)
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


def _keep_right_subject_components(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    kept = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx = float(centroids[label][0])
        if cx > OUT_W * RIGHT_SUBJECT_CUTOFF and area > 1000:
            kept[labels == label] = 255
    return kept


def _build_subject_mask(gray: np.ndarray) -> np.ndarray:
    xs = np.indices(gray.shape)[1]
    subject = ((gray > 3) & (xs > OUT_W * RIGHT_SUBJECT_CUTOFF)).astype(np.uint8) * 255
    subject = cv2.morphologyEx(
        subject, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8), iterations=2
    )
    subject = cv2.dilate(subject, np.ones((9, 9), np.uint8), iterations=6)
    subject = _keep_right_subject_components(subject)
    subject = cv2.morphologyEx(
        subject, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=1
    )
    subject = cv2.dilate(subject, np.ones((5, 5), np.uint8), iterations=1)
    return subject


def _build_jersey_mask(rgb: np.ndarray, subject: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    red = (r > g * 1.25) & (r > b * 1.25) & (r > 25) & (subject > 0)
    jersey = red.astype(np.uint8) * 255
    jersey = cv2.morphologyEx(
        jersey, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2
    )
    jersey = cv2.dilate(jersey, np.ones((7, 7), np.uint8), iterations=2)
    jersey = ((jersey > 0) & (subject > 0)).astype(np.uint8) * 255
    return jersey


def _coverage_debug(rgb: np.ndarray, subject: np.ndarray, jersey: np.ndarray) -> Image.Image:
    debug = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
    debug[subject > 0] = np.maximum(debug[subject > 0], np.array([55, 95, 55], dtype=np.uint8))
    debug[jersey > 0] = np.array([185, 20, 25], dtype=np.uint8)
    return Image.fromarray(debug, mode="RGB")


def _face_color(avg: np.ndarray) -> tuple[int, int, int]:
    r, g, b = avg.astype(np.float32)
    v = int(0.299 * r + 0.587 * g + 0.114 * b)
    v = min(245, max(35, int(v * 1.8)))
    return (v, v, v)


def _jersey_color(avg: np.ndarray) -> tuple[int, int, int]:
    r, g, b = avg.astype(np.float32)
    if r + g + b > 480:
        v = min(245, max(180, int((r + g + b) / 3 * 1.15)))
        return (v, v, v)
    return (
        min(255, max(95, int(r * 2.0))),
        min(70, max(8, int(g * 0.4))),
        min(70, max(8, int(b * 0.4))),
    )


def _draw_anchors(draw: ImageDraw.ImageDraw) -> None:
    draw.text((1350, 150), "MICHAEL JORDAN", fill=(230, 230, 232), font=_load_font(58))
    draw.text((1395, 250), "MVP", fill=(242, 242, 242), font=_load_font(86))
    draw.text((1358, 410), "CHAMPION", fill=(212, 212, 216), font=_load_font(48))
    draw.text((1355, 720), "BULLS", fill=(240, 24, 34), font=_load_font(98))
    draw.text((1515, 835), "23", fill=(248, 238, 215), font=_load_font(128))
    draw.text((610, 505), "change the game.", fill=(178, 178, 188), font=_load_font(56))


def _main() -> None:
    rng = random.Random(SEED)
    target_path = _resolve_target()
    target = _load_target_rgb(target_path)
    gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    subject = _build_subject_mask(gray)
    jersey = _build_jersey_mask(target, subject)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    luma = gray.astype(np.float32) / 255.0

    out = Image.new("RGB", (OUT_W, OUT_H), color=(0, 0, 0))
    draw = ImageDraw.Draw(out)

    cell_w = 6
    cell_h = 5
    total_probability = 0.0
    probability_samples = 0
    cells_in_subject = 0
    words_drawn = 0
    face_words = 0
    jersey_words = 0

    for y in range(0, OUT_H - cell_h, cell_h):
        for x in range(0, OUT_W - cell_w, cell_w):
            patch_subject = subject[y : y + cell_h, x : x + cell_w]
            if patch_subject.size == 0 or float(np.mean(patch_subject)) < 10.0:
                continue

            cells_in_subject += 1
            patch_luma = float(np.mean(luma[y : y + cell_h, x : x + cell_w]))
            patch_edge = float(np.mean(edges[y : y + cell_h, x : x + cell_w]))
            p = 0.58 + 0.35 * patch_edge + 0.22 * patch_luma
            p = min(0.97, max(0.48, p))
            total_probability += p
            probability_samples += 1
            if rng.random() > p:
                continue

            patch = target[y : y + cell_h, x : x + cell_w]
            avg = np.mean(patch.reshape(-1, 3), axis=0)
            is_jersey = float(np.mean(jersey[y : y + cell_h, x : x + cell_w])) > 20.0
            if is_jersey:
                word = rng.choice(JERSEY_WORDS)
                color = _jersey_color(avg)
                size = rng.randint(6, 18)
                jersey_words += 1
            else:
                word = rng.choice(FACE_WORDS)
                color = _face_color(avg)
                size = rng.randint(5, 14)
                face_words += 1

            px = x + rng.randint(-2, 2)
            py = y + rng.randint(-2, 2)
            draw.text((px, py), word, font=_load_font(size), fill=color)
            words_drawn += 1

    _draw_anchors(draw)

    out_dir = REPO_ROOT / "examples" / "reference_recreation"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(subject, mode="L").save(out_dir / "subject_mask_v2.png")
    Image.fromarray(jersey, mode="L").save(out_dir / "jersey_mask_v2.png")
    _coverage_debug(target, subject, jersey).save(out_dir / "coverage_debug_v2.png")

    out_path = out_dir / "bruteforce_text_paint_v2.png"
    out.save(out_path)

    side = Image.new("RGB", (OUT_W * 2, OUT_H), color=(0, 0, 0))
    side.paste(Image.fromarray(target, mode="RGB"), (0, 0))
    side.paste(out, (OUT_W, 0))
    side.save(out_dir / "side_by_side_bruteforce_v2.png")

    metrics = {
        "target_path": str(target_path),
        "subject_pixel_coverage": round(float(np.mean(subject > 0)), 4),
        "jersey_pixel_coverage": round(float(np.mean(jersey > 0)), 4),
        "cells_in_subject": cells_in_subject,
        "total_words_drawn": words_drawn,
        "face_words_drawn": face_words,
        "jersey_words_drawn": jersey_words,
        "average_placement_probability": round(
            total_probability / max(1, probability_samples), 4
        ),
        "cell_size": f"{cell_w}x{cell_h}",
        "seed": SEED,
        "output_resolution": f"{OUT_W}x{OUT_H}",
    }
    (out_dir / "metrics_bruteforce_v2.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    _main()
