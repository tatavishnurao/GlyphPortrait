from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
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
SPEC_PATH = ROOT / "reference_analysis" / "reference_layout_spec.json"
OUT = ROOT / "examples" / "reference_recreation"
SEED = 9142
CANVAS_W = 1920
CANVAS_H = 1080


@dataclass
class RenderStats:
    shadow_words: int = 0
    tone_words: int = 0
    contour_words: int = 0
    manual_anchor_words: int = 0
    alpha_sum: int = 0

    @property
    def total_words(self) -> int:
        return self.shadow_words + self.tone_words + self.contour_words + self.manual_anchor_words

    @property
    def average_alpha(self) -> float:
        return float(self.alpha_sum / max(1, self.total_words))


@lru_cache(maxsize=96)
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
            Path("/usr/share/fonts/TTF/DejaVuSansCondensed-Bold.ttf"),
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


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0].astype(np.float32) * 0.299 + rgb[..., 1].astype(np.float32) * 0.587 + rgb[..., 2].astype(np.float32) * 0.114


def bbox_px(norm: list[float], w: int = CANVAS_W, h: int = CANVAS_H) -> tuple[int, int, int, int]:
    return (int(norm[0] * w), int(norm[1] * h), int(norm[2] * w), int(norm[3] * h))


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


def load_face_crop(spec: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ref = Image.open(TARGET).convert("RGB").resize((CANVAS_W, CANVAS_H), Image.Resampling.LANCZOS)
    crop_box = bbox_px(spec["canvas"]["face_crop_norm"])
    return np.array(ref.crop(crop_box), dtype=np.uint8), crop_box


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    best = max(range(1, count + 1), key=lambda label: int((labels == label).sum()))
    return labels == best


def build_face_mask(crop: np.ndarray) -> np.ndarray:
    gray = luma(crop)
    raw = gray > 13
    raw = ndi.binary_closing(raw, structure=np.ones((13, 13), dtype=bool), iterations=1)
    raw = ndi.binary_dilation(raw, structure=np.ones((5, 5), dtype=bool), iterations=1)
    raw = largest_component(raw)
    raw = ndi.binary_fill_holes(raw)
    raw = ndi.binary_closing(raw, structure=np.ones((13, 13), dtype=bool), iterations=1)
    smooth = Image.fromarray(raw.astype(np.uint8) * 255, "L").filter(ImageFilter.GaussianBlur(1.4))
    return (np.array(smooth) > 72).astype(np.uint8) * 255


def crop_region_mask(crop_shape: tuple[int, int], crop_box: tuple[int, int, int, int], bbox_norm_values: list[float]) -> np.ndarray:
    h, w = crop_shape
    x0, y0, x1, y1 = bbox_px(bbox_norm_values)
    cx0, cy0, _, _ = crop_box
    rx0 = max(0, x0 - cx0)
    ry0 = max(0, y0 - cy0)
    rx1 = min(w, x1 - cx0)
    ry1 = min(h, y1 - cy0)
    mask = np.zeros((h, w), dtype=np.uint8)
    if rx1 > rx0 and ry1 > ry0:
        mask[ry0:ry1, rx0:rx1] = 255
    return mask


def build_region_masks(spec: dict, crop_shape: tuple[int, int], crop_box: tuple[int, int, int, int], face_mask: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for region in spec["regions"]:
        mask = crop_region_mask(crop_shape, crop_box, region["bbox_norm"])
        masks[region["id"]] = np.minimum(mask, face_mask)
    return masks


def protected_mask(spec: dict, crop: np.ndarray, crop_shape: tuple[int, int], crop_box: tuple[int, int, int, int], face_mask: np.ndarray) -> np.ndarray:
    mask = np.zeros(crop_shape, dtype=np.uint8)
    gray = luma(crop)
    for zone in spec["protected_dark_zones"]:
        box = crop_region_mask(crop_shape, crop_box, zone["bbox_norm"])
        dark_pixels = ((box > 0) & (face_mask > 0) & (gray <= float(zone.get("max_luma", 60)))).astype(np.uint8) * 255
        dark_pixels = ndi.binary_dilation(dark_pixels > 0, structure=np.ones((5, 5), dtype=bool), iterations=1).astype(np.uint8) * 255
        mask = np.maximum(mask, dark_pixels)
    return np.minimum(mask, face_mask)


def build_guides(crop: np.ndarray, face_mask: np.ndarray) -> dict[str, np.ndarray]:
    gray = luma(crop)
    inside = gray[face_mask > 0]
    lo, hi = np.percentile(inside, [2, 98]) if inside.size else (0, 255)
    tone = np.clip((gray - lo) / max(1e-6, hi - lo), 0, 1)
    blur = ndi.gaussian_filter(gray, 1.0)
    sx = ndi.sobel(blur, axis=1)
    sy = ndi.sobel(blur, axis=0)
    mag = np.hypot(sx, sy)
    edge_inside = mag[face_mask > 0]
    local = mag >= ndi.maximum_filter(mag, size=5)
    contours = (mag >= np.percentile(edge_inside, 84)) & local & (face_mask > 0)
    contours = ndi.binary_dilation(contours, structure=np.ones((2, 2), dtype=bool), iterations=1)
    density = np.clip(0.22 + (1.0 - tone) * 0.48 + contours.astype(np.float32) * 0.22, 0, 1) * (face_mask > 0)
    return {"gray": gray, "tone": tone, "sx": sx, "sy": sy, "contours": contours, "density": density}


def tangent_angle(sx: np.ndarray, sy: np.ndarray, x: int, y: int, clamp: float = 72.0) -> float:
    angle = math.degrees(math.atan2(float(sy[y, x]), float(sx[y, x]))) + 90
    while angle < -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return float(np.clip(angle, -clamp, clamp))


def draw_text(layer: Image.Image, x: int, y: int, text: str, size: int, fill: tuple[int, int, int], alpha: int, angle: float, bold: bool) -> None:
    f = font(size, bold)
    box = f.getbbox(text)
    patch = Image.new("RGBA", (max(1, box[2] - box[0] + 10), max(1, box[3] - box[1] + 10)), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((5 - box[0], 5 - box[1]), text, font=f, fill=(*fill, alpha))
    if abs(angle) > 0.2:
        patch = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    layer.alpha_composite(patch, (int(x), int(y)))


def clip_layer(layer: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(layer, dtype=np.uint8)
    arr[..., 3] = np.minimum(arr[..., 3], mask)
    return Image.fromarray(arr, "RGBA")


def color_from_tone(tone: float, protected: bool = False, highlight: bool = False) -> tuple[int, int, int]:
    if protected:
        v = int(np.clip(26 + tone * 42, 24, 72))
    elif highlight:
        v = int(np.clip(118 + tone * 120, 120, 238))
    else:
        v = int(np.clip(42 + tone * 128, 38, 188))
    return (v, v, min(245, v + 6))


def darken_protected_alpha(alpha: int, protected: bool) -> int:
    return int(alpha * 0.48) if protected else alpha


def choose_words(spec: dict, role: str) -> list[str]:
    return spec["word_inventory"].get(role, spec["word_inventory"]["texture"])


def render_shadow_layer(layer: Image.Image, spec: dict, guides: dict[str, np.ndarray], face_mask: np.ndarray, protected: np.ndarray, rng: random.Random, stats: RenderStats) -> None:
    h, w = face_mask.shape
    words = choose_words(spec, "shadow")
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
            alpha = darken_protected_alpha(int(40 + (1 - tone) * 42 + rng.randint(-5, 7)), is_protected)
            col = color_from_tone(tone, protected=is_protected)
            draw_text(layer, jx, jy, rng.choice(words), rng.randint(5, 8), col, alpha, rng.uniform(-11, 11), False)
            stats.shadow_words += 1
            stats.alpha_sum += alpha


def render_tone_layer(layer: Image.Image, spec: dict, guides: dict[str, np.ndarray], region_masks: dict[str, np.ndarray], protected: np.ndarray, rng: random.Random, stats: RenderStats) -> None:
    region_density = {
        "forehead": 0.72,
        "brow": 0.44,
        "eye_sockets": 0.30,
        "nose_bridge": 0.58,
        "cheek": 0.62,
        "jaw": 0.54,
        "mouth_chin": 0.48,
        "neck": 0.54
    }
    words = choose_words(spec, "texture") + choose_words(spec, "structure")
    for region, mask in region_masks.items():
        if region in {"shoulder", "jersey_chest", "jersey_trim"}:
            continue
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            continue
        attempts = int(xs.size * region_density.get(region, 0.42) / 42)
        for _ in range(attempts):
            i = rng.randrange(xs.size)
            x, y = int(xs[i]), int(ys[i])
            tone = float(guides["tone"][y, x])
            if rng.random() > float(guides["density"][y, x]):
                continue
            is_protected = protected[y, x] > 0
            size = rng.randint(8, 15)
            alpha = darken_protected_alpha(int(np.clip(70 + (1 - tone) * 55 + rng.randint(-12, 16), 54, 135)), is_protected)
            angle = rng.uniform(-14, 14)
            if region in {"jaw", "neck"}:
                angle = rng.uniform(38, 78)
            elif region == "nose_bridge":
                angle = rng.uniform(34, 68)
            elif region == "forehead":
                angle = rng.uniform(-10, 14)
            col = color_from_tone(tone, protected=is_protected)
            draw_text(layer, x, y, rng.choice(words), size, col, alpha, angle, rng.random() < 0.36)
            stats.tone_words += 1
            stats.alpha_sum += alpha


def render_contour_layer(layer: Image.Image, spec: dict, guides: dict[str, np.ndarray], face_mask: np.ndarray, protected: np.ndarray, rng: random.Random, stats: RenderStats) -> Image.Image:
    contour_vis = Image.new("RGB", face_mask.shape[::-1], (5, 5, 6))
    d = ImageDraw.Draw(contour_vis)
    words = choose_words(spec, "contour") + choose_words(spec, "highlight")
    ys, xs = np.where(guides["contours"] & (face_mask > 0))
    order = list(range(xs.size))
    rng.shuffle(order)
    for i in order[:430]:
        x, y = int(xs[i]), int(ys[i])
        if rng.random() < 0.58 and protected[y, x] > 0:
            continue
        tone = float(guides["tone"][y, x])
        angle = tangent_angle(guides["sx"], guides["sy"], x, y)
        highlight = tone > 0.42 or rng.random() < 0.22
        alpha = int(np.clip(108 + tone * 82 + rng.randint(-12, 18), 92, 205))
        if protected[y, x] > 0:
            alpha = darken_protected_alpha(alpha, True)
        size = rng.randint(9, 17)
        col = color_from_tone(tone, protected=protected[y, x] > 0, highlight=highlight)
        text = rng.choice(words)
        draw_text(layer, x, y, text, size, col, alpha, angle, True)
        d.line((x - 6, y, x + 6, y), fill=(220, 220, 230))
        stats.contour_words += 1
        stats.alpha_sum += alpha
    return contour_vis


def anchor_color(name: str) -> tuple[int, int, int]:
    if name == "bright_white":
        return (218, 218, 226)
    if name == "mid_gray":
        return (142, 142, 152)
    return (92, 92, 102)


def render_anchor_layer(layer: Image.Image, spec: dict, crop_box: tuple[int, int, int, int], face_mask: np.ndarray, protected: np.ndarray, stats: RenderStats) -> Image.Image:
    overlay = Image.new("RGBA", face_mask.shape[::-1], (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    cx0, cy0, _, _ = crop_box
    for anchor in spec["anchor_placements"]:
        if anchor["region"] in {"jersey_chest", "jersey_trim", "shoulder"}:
            continue
        x = int(anchor["pos_norm"][0] * CANVAS_W) - cx0
        y = int(anchor["pos_norm"][1] * CANVAS_H) - cy0
        if not (0 <= x < face_mask.shape[1] and 0 <= y < face_mask.shape[0]):
            continue
        is_protected = protected[y, x] > 0
        size = max(12, int(anchor["size_px"] * 0.88))
        alpha = darken_protected_alpha(int(anchor.get("alpha", 180)), is_protected)
        color = anchor_color(anchor["color"])
        if is_protected:
            color = tuple(int(c * 0.54) for c in color)
        draw_text(layer, x, y, anchor["text"], size, color, alpha, float(anchor["angle"]), True)
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 210, 70, 230))
        d.text((x + 6, y + 5), anchor["text"], font=font(13, True), fill=(255, 230, 90, 230))
        stats.manual_anchor_words += 1
        stats.alpha_sum += alpha
    arr = np.array(overlay, dtype=np.uint8)
    arr[..., 3] = np.minimum(arr[..., 3], face_mask)
    return Image.fromarray(arr, "RGBA").convert("RGB")


def final_modulation(rendered: Image.Image, crop: np.ndarray, face_mask: np.ndarray, protected: np.ndarray) -> Image.Image:
    arr = np.array(rendered.convert("RGB"), dtype=np.float32)
    gray = luma(crop)
    target = np.repeat(gray[..., None], 3, axis=2)
    mask = face_mask > 0
    arr[mask] = arr[mask] * 0.78 + target[mask] * 0.22
    dark = protected > 0
    arr[dark] = np.minimum(arr[dark], target[dark] * 0.72 + 24)
    arr[~mask] = 5
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def edge_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = luma(rgb)
    mag = np.hypot(ndi.sobel(ndi.gaussian_filter(gray, 1.1), axis=1), ndi.sobel(ndi.gaussian_filter(gray, 1.1), axis=0))
    inside = mag[mask > 0]
    if inside.size == 0:
        return np.zeros(mask.shape, dtype=bool)
    local = mag >= ndi.maximum_filter(mag, size=5)
    return (mag >= np.percentile(inside, 82)) & local & (mask > 0)


def edge_overlap(ref: np.ndarray, rec: np.ndarray, mask: np.ndarray) -> float:
    a = edge_mask(ref, mask)
    b = edge_mask(rec, mask)
    union = a | b
    return float((a & b).sum() / union.sum()) if np.any(union) else 0.0


def protected_fill_ratio(rec: np.ndarray, protected: np.ndarray) -> float:
    zone = protected > 0
    if not np.any(zone):
        return 0.0
    return float((luma(rec)[zone] > 68).mean())


def side_by_side(ref: Image.Image, rec: Image.Image) -> Image.Image:
    out = Image.new("RGB", (ref.width + rec.width, max(ref.height, rec.height)), (5, 5, 6))
    out.paste(ref, (0, 0))
    out.paste(rec, (ref.width, 0))
    d = ImageDraw.Draw(out)
    d.text((12, 12), "reference face crop", font=font(18, True), fill=(245, 245, 245))
    d.text((ref.width + 12, 12), "face study v2", font=font(18, True), fill=(245, 245, 245))
    return out


def render() -> None:
    start = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = load_spec()
    crop, crop_box = load_face_crop(spec)
    h, w = crop.shape[:2]
    face_mask = build_face_mask(crop)
    region_masks = build_region_masks(spec, (h, w), crop_box, face_mask)
    protected = protected_mask(spec, crop, (h, w), crop_box, face_mask)
    guides = build_guides(crop, face_mask)
    rng = random.Random(SEED)
    stats = RenderStats()

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    tone = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    contour = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    anchors = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    render_shadow_layer(shadow, spec, guides, face_mask, protected, rng, stats)
    render_tone_layer(tone, spec, guides, region_masks, protected, rng, stats)
    contours_vis = render_contour_layer(contour, spec, guides, face_mask, protected, rng, stats)
    anchor_overlay = render_anchor_layer(anchors, spec, crop_box, face_mask, protected, stats)

    canvas = Image.new("RGBA", (w, h), (5, 5, 6, 255))
    for layer in (shadow, tone, contour, anchors):
        canvas.alpha_composite(clip_layer(layer, face_mask))
    final = final_modulation(canvas.convert("RGB"), crop, face_mask, protected)

    ref_img = Image.fromarray(crop, "RGB")
    final.save(OUT / "face_study_v2.png")
    side_by_side(ref_img, final).save(OUT / "face_study_v2_side_by_side.png")
    contours_vis.save(OUT / "face_study_v2_contours.png")
    anchor_overlay.save(OUT / "face_study_v2_anchor_overlay.png")

    rec = np.array(final)
    metrics = {
        "total_words_drawn": stats.total_words,
        "shadow_words": stats.shadow_words,
        "tone_words": stats.tone_words,
        "contour_words": stats.contour_words,
        "manual_anchor_words": stats.manual_anchor_words,
        "average_alpha": stats.average_alpha,
        "face_luma_mae": float(np.abs(luma(crop) - luma(rec))[face_mask > 0].mean()),
        "edge_overlap_face": edge_overlap(crop, rec, face_mask),
        "protected_dark_zone_fill_ratio": protected_fill_ratio(rec, protected),
        "render_time": perf_counter() - start
    }
    (OUT / "face_study_v2_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    render()
