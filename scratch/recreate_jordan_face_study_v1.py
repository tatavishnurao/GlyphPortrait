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

OUT_DIR = REPO_ROOT / "examples" / "reference_recreation"
TARGET = REPO_ROOT / "reference_img" / "Michael-Jordan-Wallpaper-Desktop-1.jpg"
SEED = 4101
CANVAS_W = 750
CANVAS_H = 595

WORDS = [
    "Air",
    "Jordan",
    "Michael",
    "MVP",
    "focus",
    "clutch",
    "flight",
    "legend",
    "defense",
    "finals",
    "champion",
    "drive",
    "Chicago",
    "winner",
    "greatness",
    "intensity",
    "six",
    "rings",
]


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[Path | None] = []
    if bold:
        candidates.extend(
            [
                Path("/usr/share/fonts/TTF/OpenSans-CondensedBold.ttf"),
                Path("/usr/share/fonts/TTF/OpenSans-Bold.ttf"),
                Path("/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf"),
            ]
        )
    candidates.extend(
        [
            Path("/usr/share/fonts/TTF/OpenSans-CondensedRegular.ttf"),
            Path("/usr/share/fonts/TTF/OpenSans-Regular.ttf"),
            Path("/usr/share/fonts/TTF/DejaVuSansCondensed.ttf"),
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


def _load_reference() -> np.ndarray:
    return np.array(Image.open(TARGET).convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS), dtype=np.uint8)


def _subject_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    gray = _luma(rgb)
    xs_grid = np.broadcast_to(np.arange(rgb.shape[1]), gray.shape)
    raw = (gray > 12) & (xs_grid > rgb.shape[1] * 0.54)
    raw = ndi.binary_closing(raw, structure=np.ones((23, 23), dtype=bool), iterations=2)
    raw = ndi.binary_dilation(raw, structure=np.ones((9, 9), dtype=bool), iterations=2)
    labels, count = ndi.label(raw, structure=np.ones((3, 3), dtype=bool))
    best_label = 0
    best_area = 0
    for label in range(1, count + 1):
        region = labels == label
        area = int(region.sum())
        if area > best_area:
            best_area = area
            best_label = label
    mask = ndi.binary_fill_holes(labels == best_label)
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _crop_face(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = _subject_bbox(rgb)
    w = x1 - x0
    crop = (
        max(0, x0 - int(w * 0.05)),
        max(0, y0 - 35),
        min(rgb.shape[1], x0 + int(w * 0.82)),
        min(rgb.shape[0], y0 + 585),
    )
    return rgb[crop[1] : crop[3], crop[0] : crop[2]].copy(), crop


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    best = max(range(1, count + 1), key=lambda label: int((labels == label).sum()))
    return labels == best


def _build_face_mask(crop: np.ndarray) -> np.ndarray:
    gray = _luma(crop)
    raw = gray > 11
    raw = ndi.binary_closing(raw, structure=np.ones((13, 13), dtype=bool), iterations=1)
    raw = ndi.binary_dilation(raw, structure=np.ones((5, 5), dtype=bool), iterations=1)
    raw = _largest_component(raw)
    filled = ndi.binary_fill_holes(raw)
    filled = ndi.binary_closing(filled, structure=np.ones((15, 15), dtype=bool), iterations=1)
    smooth = Image.fromarray((filled.astype(np.uint8) * 255), "L").filter(ImageFilter.GaussianBlur(1.2))
    return (np.array(smooth) > 82).astype(np.uint8) * 255


def _equalize_tile(tile: np.ndarray, clip_limit: float = 0.018) -> np.ndarray:
    hist, _ = np.histogram(tile.ravel(), bins=256, range=(0, 255))
    clip_at = max(1, int(tile.size * clip_limit))
    excess = np.maximum(hist - clip_at, 0).sum()
    hist = np.minimum(hist, clip_at)
    hist += excess // 256
    cdf = hist.cumsum().astype(np.float32)
    cdf = (cdf - cdf.min()) / max(1e-6, cdf.max() - cdf.min())
    return cdf[np.clip(tile.astype(np.uint8), 0, 255)]


def _clahe_like(gray: np.ndarray, mask: np.ndarray, tile_size: int = 96) -> np.ndarray:
    h, w = gray.shape
    out = np.zeros_like(gray, dtype=np.float32)
    weight = np.zeros_like(gray, dtype=np.float32)
    for y in range(0, h, tile_size // 2):
        for x in range(0, w, tile_size // 2):
            y0, y1 = max(0, y - tile_size // 2), min(h, y + tile_size)
            x0, x1 = max(0, x - tile_size // 2), min(w, x + tile_size)
            tile = gray[y0:y1, x0:x1]
            eq = _equalize_tile(tile)
            yy = np.hanning(max(3, y1 - y0))[:, None]
            xx = np.hanning(max(3, x1 - x0))[None, :]
            win = np.maximum(yy * xx, 0.05).astype(np.float32)
            out[y0:y1, x0:x1] += eq * win
            weight[y0:y1, x0:x1] += win
    clahe = out / np.maximum(weight, 1e-6)
    inside = gray[mask > 0]
    lo, hi = np.percentile(inside, [2, 98]) if inside.size else (0, 255)
    raw = np.clip((gray - lo) / max(1e-6, hi - lo), 0, 1)
    inside_eq = clahe[mask > 0]
    eq_lo, eq_hi = np.percentile(inside_eq, [4, 98]) if inside_eq.size else (0, 1)
    clahe = np.clip((clahe - eq_lo) / max(1e-6, eq_hi - eq_lo), 0, 1)
    # CLAHE reveals texture; raw contrast keeps likeness-defining shadows.
    out = raw * 0.62 + clahe * 0.38
    out = np.power(out, 0.88)
    return (out * (mask > 0)).astype(np.float32)


def _build_contours(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blurred = ndi.gaussian_filter(gray, sigma=1.0)
    sx = ndi.sobel(blurred, axis=1)
    sy = ndi.sobel(blurred, axis=0)
    mag = np.hypot(sx, sy)
    mag = mag / max(1e-6, float(mag.max()))
    inside = mag[mask > 0]
    high = np.percentile(inside, 84) if inside.size else 0.3
    low = np.percentile(inside, 64) if inside.size else 0.16
    strong = (mag >= high) & (mask > 0)
    weak = (mag >= low) & (mask > 0)
    edges = strong.copy()
    for _ in range(3):
        edges = weak & ndi.binary_dilation(edges, structure=np.ones((3, 3), dtype=bool))
    local = mag >= ndi.maximum_filter(mag, size=5)
    contours = ((edges & local) | ((mag > np.percentile(inside, 91)) & (mask > 0))).astype(np.float32)
    contours = ndi.binary_dilation(contours, structure=np.ones((2, 2), dtype=bool), iterations=1).astype(np.float32)
    return contours * mag, sx.astype(np.float32), sy.astype(np.float32)


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
) -> None:
    font = _font(size, bold)
    bbox = font.getbbox(text)
    patch = Image.new("RGBA", (max(1, bbox[2] - bbox[0] + 8), max(1, bbox[3] - bbox[1] + 8)), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=(*fill, alpha))
    if abs(angle) > 0.2:
        patch = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    layer.alpha_composite(patch, (x, y))


def _angle(sx: np.ndarray, sy: np.ndarray, x: int, y: int) -> float:
    gx = float(sx[y, x])
    gy = float(sy[y, x])
    if abs(gx) + abs(gy) < 0.01:
        return 0.0
    angle = math.degrees(math.atan2(gy, gx)) + 90.0
    while angle < -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return float(np.clip(angle, -22, 22))


def _clip(layer: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(layer, dtype=np.uint8)
    arr[..., 3] = np.minimum(arr[..., 3], mask)
    return Image.fromarray(arr, "RGBA")


def _render_face(crop: np.ndarray, mask: np.ndarray, luminance: np.ndarray, contours: np.ndarray, sx: np.ndarray, sy: np.ndarray, rng: random.Random) -> tuple[Image.Image, dict[str, int | float]]:
    h, w = mask.shape
    shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    tone_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    contour_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow_words = tone_words = contour_words = 0
    alpha_sum = 0

    for y in range(rng.randrange(0, 5), h, 5):
        x = rng.randrange(0, 8)
        while x < w:
            jx = x + rng.randint(-5, 5)
            jy = y + rng.randint(-4, 4)
            x += rng.choice((7, 8, 9, 10))
            if not (0 <= jx < w and 0 <= jy < h and mask[jy, jx] > 0):
                continue
            lum = float(luminance[jy, jx])
            alpha = int(np.clip(44 + (1.0 - lum) * 36 + rng.randint(-5, 8), 42, 86))
            v = int(np.clip(28 + lum * 54, 28, 92))
            _draw_text(shadow_layer, jx, jy, rng.choice(WORDS), rng.randint(5, 8), (v, v, v + 2), alpha)
            shadow_words += 1
            alpha_sum += alpha

    for y in range(rng.randrange(0, 8), h, 8):
        x = rng.randrange(0, 11)
        while x < w:
            jx = x + rng.randint(-7, 7)
            jy = y + rng.randint(-6, 6)
            x += rng.choice((10, 12, 14))
            if not (0 <= jx < w and 0 <= jy < h and mask[jy, jx] > 0):
                continue
            lum = float(luminance[jy, jx])
            edge = float(contours[jy, jx])
            p = 0.36 + lum * 0.20 + edge * 0.34 + max(0.0, 0.30 - lum) * 0.42
            if rng.random() > min(0.88, p):
                continue
            alpha = int(np.clip(76 + lum * 48 + edge * 62 + max(0.0, 0.28 - lum) * 54 + rng.randint(-12, 14), 68, 166))
            v = int(np.clip(48 + lum * 145 + edge * 42 + rng.randint(-8, 12), 44, 226))
            _draw_text(
                tone_layer,
                jx,
                jy,
                rng.choice(WORDS),
                rng.randint(7, 13),
                (v, v, min(255, v + 4)),
                alpha,
                rng.uniform(-4, 4),
                bold=edge > 0.20,
            )
            tone_words += 1
            alpha_sum += alpha

    coords = np.argwhere((contours > 0.08) & (mask > 0))
    rng.shuffle(coords)
    for y, x in coords[:3600]:
        if rng.random() > min(0.76, 0.24 + float(contours[y, x]) * 0.95):
            continue
        x = int(x + rng.randint(-6, 6))
        y = int(y + rng.randint(-5, 5))
        if not (0 <= x < w and 0 <= y < h and mask[y, x] > 0):
            continue
        lum = float(luminance[y, x])
        edge = float(contours[y, x])
        alpha = int(np.clip(118 + edge * 92 + lum * 28 + rng.randint(-14, 16), 104, 218))
        v = int(np.clip(88 + lum * 142 + edge * 42, 82, 248))
        _draw_text(
            contour_layer,
            x,
            y,
            rng.choice(WORDS),
            rng.randint(9, 16),
            (v, v, min(255, v + 4)),
            alpha,
            _angle(sx, sy, x, y) + rng.uniform(-4, 4),
            bold=True,
        )
        contour_words += 1
        alpha_sum += alpha

    out = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    for layer in (_clip(shadow_layer, mask), _clip(tone_layer, mask), _clip(contour_layer, mask)):
        out.alpha_composite(layer)

    arr = np.array(out.convert("RGB"), dtype=np.float32)
    lum = luminance[..., None]
    mask3 = (mask > 0)[..., None]
    sculpt = np.clip(0.46 + lum * 1.32 + contours[..., None] * 0.22, 0.36, 1.54)
    arr = np.where(mask3, arr * sculpt, 0)
    deep = ((lum < 0.10) & mask3).astype(np.float32)
    arr *= 1.0 - deep * 0.24
    result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    return result, {
        "shadow_words": shadow_words,
        "tone_words": tone_words,
        "contour_words": contour_words,
        "total_words_drawn": shadow_words + tone_words + contour_words,
        "average_alpha": round(alpha_sum / max(1, shadow_words + tone_words + contour_words), 2),
    }


def main() -> None:
    t0 = perf_counter()
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = _load_reference()
    crop, crop_box = _crop_face(ref)
    reference_crop = Image.fromarray(crop, "RGB")
    reference_crop.save(OUT_DIR / "face_reference_crop.png")

    mask = _build_face_mask(crop)
    gray = _luma(crop)
    luminance = _clahe_like(gray, mask)
    contours, sx, sy = _build_contours(gray, mask)
    study, stats = _render_face(crop, mask, luminance, contours, sx, sy, rng)
    study.save(OUT_DIR / "face_study_v1.png")

    side = Image.new("RGB", (crop.shape[1] * 2, crop.shape[0]), (0, 0, 0))
    side.paste(reference_crop, (0, 0))
    side.paste(study, (crop.shape[1], 0))
    side.save(OUT_DIR / "face_study_side_by_side.png")

    metrics = {
        "target_path": str(TARGET),
        "crop_box": crop_box,
        "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
        "face_mask_coverage": round(float(np.mean(mask > 0)), 5),
        "contour_coverage": round(float(np.mean(contours > 0)), 5),
        "render_time": round(perf_counter() - t0, 3),
        "seed": SEED,
        **stats,
    }
    (OUT_DIR / "face_study_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
