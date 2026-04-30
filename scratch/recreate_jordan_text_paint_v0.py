from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Allow direct execution from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glyphforge.typography.fonts import find_font
from studies.jordan_wallpaper.target_analysis import (
    build_target_edge_map,
    build_target_luminance_map,
    extract_nonblack_subject_mask,
    extract_red_jersey_mask,
)

OUT_W = 1920
OUT_H = 1080
SEED = 23

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


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        font_path = find_font("assets/fonts")
        return ImageFont.truetype(str(font_path), size=size)
    except Exception:
        return ImageFont.load_default()


def _resolve_target(repo_root: Path) -> Path:
    requested = repo_root / "reference_img" / "Michael-Jordan-Wallpaper-Desktop-1.jpg"
    if requested.exists():
        return requested
    fallback = repo_root / "examples" / "reference_recreation" / "side_by_side_final.png"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No target found at reference_img/... or side_by_side_final.png")


def _load_target_rgb(path: Path) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"))
    if "side_by_side_final.png" in str(path):
        h, w = rgb.shape[:2]
        rgb = rgb[:, : (w // 2), :]
    return np.array(
        Image.fromarray(rgb).resize((OUT_W, OUT_H), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _boost_face_color(c: np.ndarray, lum: float) -> tuple[int, int, int]:
    base = c.astype(np.float32)
    lift = np.array([16, 16, 18], dtype=np.float32)
    if lum < 0.45:
        lift += 20
    out = np.clip(base + lift, 140, 245).astype(np.uint8)
    return int(out[0]), int(out[1]), int(out[2])


def _boost_jersey_color(c: np.ndarray) -> tuple[int, int, int]:
    r, g, b = [int(v) for v in c]
    r = min(255, max(r + 45, g + 40, b + 40))
    g = max(10, g - 25)
    b = max(10, b - 25)
    return (r, g, b)


def _main() -> None:
    rng = random.Random(SEED)
    repo_root = REPO_ROOT
    target_path = _resolve_target(repo_root)
    target_rgb = _load_target_rgb(target_path)

    subject_mask = extract_nonblack_subject_mask(target_rgb, black_threshold=18)
    jersey_mask = extract_red_jersey_mask(target_rgb, subject_mask, min_delta=20, min_red=70)
    lum = build_target_luminance_map(target_rgb)
    gray_u8 = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
    edge = build_target_edge_map(gray_u8)

    out = Image.new("RGB", (OUT_W, OUT_H), color=(0, 0, 0))
    draw = ImageDraw.Draw(out)

    cell_w = 18
    cell_h = 14
    jitter_x = 3
    jitter_y = 2
    words_drawn = 0
    face_words = 0
    jersey_words = 0
    cells_used = 0

    for y in range(0, OUT_H - cell_h, cell_h):
        for x in range(0, OUT_W - cell_w, cell_w):
            patch_subject = subject_mask[y : y + cell_h, x : x + cell_w]
            if patch_subject.size == 0 or float(np.mean(patch_subject)) < 20.0:
                continue
            patch = target_rgb[y : y + cell_h, x : x + cell_w]
            avg = np.mean(patch.reshape(-1, 3), axis=0)
            patch_lum = float(np.mean(lum[y : y + cell_h, x : x + cell_w]))
            patch_edge = float(np.mean(edge[y : y + cell_h, x : x + cell_w]))
            is_jersey = float(np.mean(jersey_mask[y : y + cell_h, x : x + cell_w])) > 80.0
            if is_jersey:
                word = rng.choice(JERSEY_WORDS)
                color = _boost_jersey_color(avg)
                base_size = 14 + int((1.0 - patch_lum) * 14 + patch_edge * 10)
                jersey_words += 1
            else:
                word = rng.choice(FACE_WORDS)
                color = _boost_face_color(avg, patch_lum)
                base_size = 10 + int((1.0 - patch_lum) * 12 + patch_edge * 9)
                face_words += 1

            size = max(8, min(28, base_size))
            px = x + rng.randint(-jitter_x, jitter_x)
            py = y + rng.randint(-jitter_y, jitter_y)
            draw.text((px, py), word, font=_load_font(size), fill=color)
            words_drawn += 1
            cells_used += 1

    anchor_font = _load_font(62)
    draw.text((1380, 160), "MICHAEL JORDAN", fill=(228, 228, 230), font=_load_font(56))
    draw.text((1410, 260), "MVP", fill=(240, 240, 240), font=_load_font(78))
    draw.text((1370, 730), "BULLS", fill=(235, 24, 34), font=_load_font(92))
    draw.text((1520, 840), "23", fill=(245, 236, 215), font=anchor_font)
    draw.text((620, 515), "change the game.", fill=(176, 176, 186), font=_load_font(56))

    out_dir = repo_root / "examples" / "reference_recreation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bruteforce_text_paint_v1.png"
    out.save(out_path)

    target_img = Image.fromarray(target_rgb).convert("RGB")
    merged = Image.new("RGB", (OUT_W * 2, OUT_H), color=(0, 0, 0))
    merged.paste(target_img, (0, 0))
    merged.paste(out.convert("RGB"), (OUT_W, 0))
    merged.save(out_dir / "side_by_side_bruteforce_v1.png")

    metrics = {
        "target_path": str(target_path),
        "cells_used": cells_used,
        "words_drawn": words_drawn,
        "jersey_words_drawn": jersey_words,
        "face_words_drawn": face_words,
        "cell_size": f"{cell_w}x{cell_h}",
        "seed": SEED,
        "output_resolution": f"{OUT_W}x{OUT_H}",
    }
    (out_dir / "bruteforce_text_paint_v1_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    _main()
