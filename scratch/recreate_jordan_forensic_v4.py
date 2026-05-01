from __future__ import annotations

import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glyphforge.typography.fonts import find_font

TARGET = ROOT / "reference_img" / "Michael-Jordan-Wallpaper-Desktop-1.jpg"
ANALYSIS = ROOT / "reference_analysis"
OUT = ROOT / "examples" / "reference_recreation"
W, H = 1920, 1080
SEED = 8204

FACE_CROP = (1280, 135, 1715, 680)
JERSEY_CROP = (1210, 575, 1910, 1070)

FACE_WORDS = [
    "Air Jordan",
    "MVP",
    "Chicago Bulls",
    "Dedication",
    "Love of the Game",
    "Dominance",
    "Finals MVP",
    "NBA Champion",
    "Rookie of the Year",
    "All American Game",
    "Scoring",
    "Defense",
    "Slam Dunk",
    "Olympic Gold",
    "1984",
    "1985",
    "1986",
    "63 points",
    "six-time NBA Champion",
    "Hang Time",
    "King of Clutch",
]

JERSEY_WORDS = [
    "BULLS",
    "23",
    "Named to the All-Star Team",
    "Chicago Bulls",
    "NBA Most Valuable Player",
    "NBA Champion",
    "game winner",
    "All-Defensive First Team",
    "Finals MVP",
    "Rookie",
    "red and black",
    "Give Up",
    "I can't accept",
    "air",
    "competitive",
]


@lru_cache(maxsize=128)
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
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def load_target() -> np.ndarray:
    return np.array(Image.open(TARGET).convert("RGB").resize((W, H), Image.Resampling.LANCZOS), dtype=np.uint8)


def largest_component(mask: np.ndarray, x_min: int = 0) -> np.ndarray:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(mask)
    xs_grid = np.broadcast_to(np.arange(mask.shape[1]), mask.shape)
    best_label, best_score = 0, -1.0
    for label in range(1, count + 1):
        region = labels == label
        area = int(region.sum())
        if area < 500:
            continue
        cx = float(xs_grid[region].mean())
        if cx < x_min:
            continue
        score = area * (1.0 + cx / mask.shape[1])
        if score > best_score:
            best_label, best_score = label, score
    return labels == best_label


def build_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    gray = luma(rgb)
    xs = np.broadcast_to(np.arange(W), gray.shape)
    raw = (gray > 10) & (xs > int(W * 0.52))
    raw = ndi.binary_closing(raw, structure=np.ones((27, 27), bool), iterations=2)
    raw = ndi.binary_dilation(raw, structure=np.ones((7, 7), bool), iterations=2)
    shape = largest_component(raw, int(W * 0.52))
    shape = ndi.binary_fill_holes(shape)
    shape = ndi.binary_closing(shape, structure=np.ones((19, 19), bool), iterations=1)
    shape = np.array(Image.fromarray(shape.astype(np.uint8) * 255, "L").filter(ImageFilter.GaussianBlur(2.0))) > 70

    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    red_strength = (r - np.maximum(g, b)) / 130.0
    jersey = (red_strength > 0.12) & (r > 25) & shape
    jersey = ndi.binary_closing(jersey, structure=np.ones((19, 19), bool), iterations=2)
    jersey = ndi.binary_dilation(jersey, structure=np.ones((9, 9), bool), iterations=2)
    jersey = ndi.binary_fill_holes(jersey) & shape

    yy = np.broadcast_to(np.arange(H)[:, None], shape.shape)
    head = shape & ~jersey & (yy < 640)
    neck = shape & ~jersey & (yy >= 520) & (yy < 790)
    shoulder = shape & ~jersey & ~head & ~neck
    return {
        "shape": shape.astype(np.uint8) * 255,
        "jersey": jersey.astype(np.uint8) * 255,
        "head": head.astype(np.uint8) * 255,
        "neck": neck.astype(np.uint8) * 255,
        "shoulder": shoulder.astype(np.uint8) * 255,
        "negative": (~shape).astype(np.uint8) * 255,
    }


def maps(rgb: np.ndarray, shape: np.ndarray) -> dict[str, np.ndarray]:
    gray = luma(rgb).astype(np.float32)
    blur = ndi.gaussian_filter(gray, 1.2)
    sx = ndi.sobel(blur, axis=1)
    sy = ndi.sobel(blur, axis=0)
    edge = np.hypot(sx, sy)
    edge = edge / max(1e-6, float(edge.max()))
    local = edge >= ndi.maximum_filter(edge, size=5)
    inside = edge[shape > 0]
    contours = ((edge > np.percentile(inside, 84)) & local & (shape > 0)).astype(np.uint8) * 255
    tone = np.clip(gray / 255.0, 0, 1)
    density = np.clip((0.18 + 0.72 * (1 - tone)) * (shape > 0) + 0.18 * (contours > 0), 0, 1)
    return {"gray": gray, "tone": tone, "edge": edge, "contours": contours, "sx": sx, "sy": sy, "density": density}


def draw_text(layer: Image.Image, x: int, y: int, text: str, size: int, fill: tuple[int, int, int], alpha: int, angle: float = 0, bold: bool = False) -> None:
    f = font(size, bold)
    box = f.getbbox(text)
    patch = Image.new("RGBA", (max(1, box[2] - box[0] + 12), max(1, box[3] - box[1] + 12)), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((6 - box[0], 6 - box[1]), text, font=f, fill=(*fill, alpha))
    if abs(angle) > 0.2:
        patch = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    layer.alpha_composite(patch, (int(x), int(y)))


def clip(layer: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(layer, dtype=np.uint8)
    arr[..., 3] = np.minimum(arr[..., 3], mask)
    return Image.fromarray(arr, "RGBA")


def tangent_angle(sx: np.ndarray, sy: np.ndarray, x: int, y: int, clamp: float = 80.0) -> float:
    angle = math.degrees(math.atan2(float(sy[y, x]), float(sx[y, x]))) + 90
    while angle < -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return float(np.clip(angle, -clamp, clamp))


def color_for(rgb: np.ndarray, x: int, y: int, jersey: bool = False) -> tuple[int, int, int]:
    r, g, b = [int(v) for v in rgb[y, x]]
    if jersey:
        return (max(55, r), min(42, g + 8), min(48, b + 8))
    v = int(np.clip(luma(rgb[y : y + 1, x : x + 1])[0, 0] * 1.35 + 20, 28, 232))
    return (v, v, min(240, v + 6))


def render_region(
    rgb: np.ndarray,
    mask: np.ndarray,
    guide: dict[str, np.ndarray],
    words: list[str],
    rng: random.Random,
    jersey: bool = False,
) -> Image.Image:
    h, w = mask.shape
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return layer

    # Shadow microtext pass: high count, low alpha, short row cadence.
    for y in range(int(ys.min()), int(ys.max()), 7 if jersey else 6):
        x = int(xs.min()) + rng.randint(0, 12)
        while x < int(xs.max()):
            if 0 <= y < h and 0 <= x < w and mask[y, x] > 0:
                tone = float(guide["tone"][y, x])
                p = 0.48 + float(guide["density"][y, x]) * 0.45
                if rng.random() < p:
                    size = rng.randint(6, 11) if jersey else rng.randint(5, 9)
                    col = color_for(rgb, x, y, jersey)
                    if not jersey:
                        col = tuple(int(c * (0.55 + tone * 0.75)) for c in col)
                    draw_text(layer, x + rng.randint(-4, 4), y + rng.randint(-3, 3), rng.choice(words), size, col, int(54 + 70 * (1 - tone)), rng.uniform(-12, 12), False)
            x += rng.randint(17, 34)

    # Surface structure pass: readable words, orientation biased by local contours.
    for _ in range(1450 if jersey else 1050):
        i = rng.randrange(xs.size)
        x, y = int(xs[i]), int(ys[i])
        if rng.random() > float(guide["density"][y, x]):
            continue
        size = rng.randint(10, 22) if jersey else rng.randint(9, 20)
        angle = tangent_angle(guide["sx"], guide["sy"], x, y, 42) if rng.random() < 0.42 else rng.uniform(-18, 18)
        tone = float(guide["tone"][y, x])
        col = color_for(rgb, x, y, jersey)
        alpha = int(np.clip(78 + (1 - tone) * 78 + rng.randint(-10, 30), 68, 180))
        draw_text(layer, x, y, rng.choice(words), size, col, alpha, angle, rng.random() < 0.45)

    # Contour pass: words placed on local edge tangents.
    cys, cxs = np.where((guide["contours"] > 0) & (mask > 0))
    order = list(range(cxs.size))
    rng.shuffle(order)
    for i in order[:520 if jersey else 460]:
        x, y = int(cxs[i]), int(cys[i])
        size = rng.randint(9, 18) if jersey else rng.randint(8, 17)
        tone = float(guide["tone"][y, x])
        col = color_for(rgb, x, y, jersey)
        if not jersey and tone > 0.34:
            col = tuple(min(245, int(c * 1.22)) for c in col)
        draw_text(layer, x, y, rng.choice(words), size, col, int(132 + 74 * tone), tangent_angle(guide["sx"], guide["sy"], x, y, 78), True)

    return clip(layer, mask)


def render_anchors(canvas: Image.Image, scope: str = "full") -> None:
    anchors = [
        ("All American Game", 1420, 182, 20, (212, 212, 218), 205, -8),
        ("NBA Rookie of the Year", 1405, 220, 31, (230, 230, 236), 230, 6),
        ("MVP", 1410, 285, 46, (230, 230, 238), 232, -5),
        ("Chicago Bulls", 1490, 295, 30, (200, 200, 210), 220, 50),
        ("Air Jordan", 1367, 365, 32, (218, 218, 224), 225, 4),
        ("Dedication", 1370, 455, 34, (150, 150, 158), 185, 26),
        ("Love of the Game", 1395, 501, 25, (158, 158, 166), 188, 12),
        ("Dominance", 1340, 590, 31, (218, 218, 226), 225, 70),
        ("Finals MVP", 1450, 637, 27, (166, 166, 174), 175, 45),
        ("Scoring", 1565, 535, 29, (218, 218, 226), 220, -8),
        ("Named to the All-Star Team", 1350, 775, 30, (230, 32, 35), 220, 13),
        ("NBA Most Valuable Player", 1450, 760, 31, (228, 30, 34), 220, 82),
        ("BULLS", 1465, 953, 93, (244, 232, 210), 235, -12),
        ("23", 1544, 847, 114, (245, 237, 214), 230, -8),
        ("Chicago Bulls", 1358, 855, 52, (182, 18, 24), 210, 11),
        ("I can't accept", 1760, 740, 24, (242, 242, 236), 218, 82),
        ("Give Up", 1770, 700, 48, (215, 215, 218), 225, 6),
        ("Jordan", 1718, 914, 66, (188, 188, 194), 210, 78),
    ]
    for text, x, y, size, col, alpha, angle in anchors:
        if scope == "face" and y > 690:
            continue
        if scope == "jersey" and y < 570:
            continue
        draw_text(canvas, x, y, text, size, col, alpha, angle, True)


def composite_full(rgb: np.ndarray, masks: dict[str, np.ndarray], guide: dict[str, np.ndarray]) -> Image.Image:
    rng = random.Random(SEED)
    canvas = Image.new("RGBA", (W, H), (5, 5, 6, 255))
    draw_text(canvas, 800, 500, "change the game.", 28, (218, 218, 232), 220, 0, False)
    head = render_region(rgb, np.maximum(masks["head"], masks["neck"]), guide, FACE_WORDS, rng, False)
    jersey = render_region(rgb, masks["jersey"], guide, JERSEY_WORDS, rng, True)
    shoulder = render_region(rgb, masks["shoulder"], guide, FACE_WORDS + JERSEY_WORDS, rng, False)
    canvas.alpha_composite(shoulder)
    canvas.alpha_composite(jersey)
    canvas.alpha_composite(head)
    anchor_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    render_anchors(anchor_layer)
    anchor_layer = clip(anchor_layer, masks["shape"])
    canvas.alpha_composite(anchor_layer)
    return canvas.convert("RGB")


def crop_from_full(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return img.crop(box)


def side_by_side(ref: Image.Image, study: Image.Image) -> Image.Image:
    h = max(ref.height, study.height)
    out = Image.new("RGB", (ref.width + study.width, h), (5, 5, 6))
    out.paste(ref, (0, 0))
    out.paste(study, (ref.width, 0))
    d = ImageDraw.Draw(out)
    d.text((12, 12), "reference", fill=(245, 245, 245), font=font(18, True))
    d.text((ref.width + 12, 12), "reconstruction", fill=(245, 245, 245), font=font(18, True))
    return out


def annotated_regions(rgb: np.ndarray) -> Image.Image:
    img = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    regions = [
        ("forehead / crown", (1355, 160, 1698, 315), (90, 180, 255, 72)),
        ("eye socket", (1385, 306, 1682, 428), (80, 80, 120, 95)),
        ("nose / cheek", (1510, 390, 1698, 565), (255, 230, 90, 70)),
        ("mouth / chin", (1468, 535, 1675, 675), (210, 210, 210, 70)),
        ("neck", (1390, 625, 1590, 790), (130, 220, 150, 70)),
        ("left shoulder", (1215, 675, 1435, 1070), (230, 230, 230, 70)),
        ("red jersey chest", (1320, 690, 1790, 1070), (255, 40, 45, 80)),
        ("right shoulder", (1700, 640, 1915, 1080), (180, 180, 210, 70)),
        ("negative space", (0, 0, 1180, 1080), (0, 0, 0, 0)),
    ]
    for label, box, fill in regions:
        d.rectangle(box, outline=fill[:3] + (220,), width=3, fill=fill)
        d.text((box[0] + 8, box[1] + 8), label, font=font(22, True), fill=(255, 255, 255, 235))
    d.text((800, 500), "background / negative space", font=font(28, True), fill=(210, 210, 225, 235))
    return Image.alpha_composite(img, overlay).convert("RGB")


def density_map(rgb: np.ndarray, masks: dict[str, np.ndarray], guide: dict[str, np.ndarray]) -> Image.Image:
    base = Image.fromarray(rgb).convert("RGBA").filter(ImageFilter.GaussianBlur(1.0))
    heat = np.zeros((H, W, 4), dtype=np.uint8)
    dens = (guide["density"] * (masks["shape"] > 0))
    heat[..., 0] = np.clip(dens * 255, 0, 255)
    heat[..., 1] = np.clip((1 - np.abs(dens - 0.55) * 2) * 150, 0, 150)
    heat[..., 3] = np.clip(dens * 175, 0, 175)
    out = Image.alpha_composite(base, Image.fromarray(heat, "RGBA"))
    d = ImageDraw.Draw(out)
    for text, xy in [
        ("dense: forehead/temple rings", (1335, 190)),
        ("dark dense: eye sockets", (1430, 360)),
        ("bright contour: nose/mouth", (1550, 495)),
        ("dense red structure: jersey", (1360, 810)),
        ("sparse/silent: black field", (220, 520)),
    ]:
        d.text(xy, text, font=font(24, True), fill=(255, 255, 255, 235))
    return out.convert("RGB")


def orientation_map(rgb: np.ndarray) -> Image.Image:
    img = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    strokes = [
        ((1365, 205), (1665, 155), "crown arcs / diagonal bands", (80, 180, 255, 230)),
        ((1335, 360), (1590, 345), "horizontal face fillers", (240, 240, 240, 230)),
        ((1640, 255), (1675, 520), "vertical temple contour", (255, 210, 80, 230)),
        ((1490, 395), (1630, 565), "nose bridge / cheek diagonal", (255, 210, 80, 230)),
        ((1340, 520), (1425, 695), "jaw/neck verticals", (130, 240, 150, 230)),
        ((1240, 700), (1770, 655), "jersey trim sweep", (255, 255, 255, 230)),
        ((1340, 815), (1790, 1005), "red jersey diagonals", (255, 45, 45, 235)),
        ((1755, 665), (1875, 1040), "right shoulder verticals", (210, 210, 230, 230)),
    ]
    for start, end, label, color in strokes:
        d.line((start, end), fill=color, width=6)
        ang = math.atan2(end[1] - start[1], end[0] - start[0])
        head = (end[0] - 18 * math.cos(ang - 0.45), end[1] - 18 * math.sin(ang - 0.45))
        head2 = (end[0] - 18 * math.cos(ang + 0.45), end[1] - 18 * math.sin(ang + 0.45))
        d.polygon([end, head, head2], fill=color)
        d.text((start[0] + 8, start[1] + 8), label, font=font(20, True), fill=color)
    return Image.alpha_composite(img, overlay).convert("RGB")


def red_mask(rgb: np.ndarray, shape: np.ndarray | None = None) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    red = (r - np.maximum(g, b) > 18) & (r > 38)
    if shape is not None:
        red &= shape > 0
    red = ndi.binary_opening(red, structure=np.ones((3, 3), bool), iterations=1)
    return red


def edge_mask(rgb: np.ndarray, region: np.ndarray) -> np.ndarray:
    gray = luma(rgb).astype(np.float32)
    blur = ndi.gaussian_filter(gray, 1.1)
    mag = np.hypot(ndi.sobel(blur, axis=1), ndi.sobel(blur, axis=0))
    inside = mag[region > 0]
    if inside.size == 0:
        return np.zeros(region.shape, dtype=bool)
    local = mag >= ndi.maximum_filter(mag, size=5)
    return (mag >= np.percentile(inside, 82)) & local & (region > 0)


def overlap_score(a: np.ndarray, b: np.ndarray) -> float:
    union = a | b
    if not np.any(union):
        return 0.0
    return float((a & b).sum() / union.sum())


def rgb_error(ref: np.ndarray, rec: np.ndarray) -> np.ndarray:
    return np.abs(ref.astype(np.float32) - rec.astype(np.float32)).mean(axis=2)


def error_heatmap(error: np.ndarray, mask: np.ndarray | None = None, label: str = "") -> Image.Image:
    visible = error if mask is None else error * (mask > 0)
    nonzero = visible[visible > 0]
    scale = float(np.percentile(nonzero, 98)) if nonzero.size else 1.0
    norm = np.clip(visible / max(scale, 1e-6), 0, 1)
    arr = np.zeros((H, W, 4), dtype=np.uint8)
    arr[..., 0] = np.clip(norm * 255, 0, 255)
    arr[..., 1] = np.clip((1 - np.abs(norm - 0.5) * 2) * 210, 0, 210)
    arr[..., 2] = np.clip((1 - norm) * 70, 0, 70)
    arr[..., 3] = np.where(mask > 0 if mask is not None else norm > 0, np.clip(70 + norm * 185, 0, 255), 0)
    out = Image.alpha_composite(Image.new("RGBA", (W, H), (5, 5, 6, 255)), Image.fromarray(arr, "RGBA"))
    if label:
        ImageDraw.Draw(out).text((24, 24), label, font=font(28, True), fill=(255, 255, 255, 240))
    return out.convert("RGB")


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], label: str) -> Image.Image:
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    overlay[..., :3] = color
    overlay[..., 3] = (mask > 0).astype(np.uint8) * 112
    out = Image.alpha_composite(base, Image.fromarray(overlay, "RGBA"))
    d = ImageDraw.Draw(out)
    d.text((24, 24), label, font=font(30, True), fill=(255, 255, 255, 245))
    return out.convert("RGB")


def mask_tile(mask: np.ndarray, label: str, color: tuple[int, int, int]) -> Image.Image:
    small = Image.new("RGB", (480, 270), (8, 8, 10))
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    arr[mask > 0] = color
    tile = Image.fromarray(arr, "RGB").resize((480, 270), Image.Resampling.NEAREST)
    small.paste(tile)
    ImageDraw.Draw(small).text((14, 14), label, font=font(20, True), fill=(255, 255, 255))
    return small


def mask_panel(masks: dict[str, np.ndarray], ref: np.ndarray, rec: np.ndarray) -> Image.Image:
    ref_red = red_mask(ref, masks["shape"])
    rec_red = red_mask(rec, masks["shape"])
    tiles = [
        mask_tile(masks["shape"], "subject mask", (220, 220, 230)),
        mask_tile(np.maximum(masks["head"], masks["neck"]), "head + neck mask", (120, 210, 255)),
        mask_tile(masks["jersey"], "reference jersey mask", (245, 40, 42)),
        mask_tile(masks["shoulder"], "shoulder/body mask", (190, 190, 210)),
        mask_tile(ref_red.astype(np.uint8) * 255, "reference red mask", (255, 35, 35)),
        mask_tile(rec_red.astype(np.uint8) * 255, "reconstruction red mask", (255, 140, 30)),
    ]
    panel = Image.new("RGB", (1440, 540), (5, 5, 6))
    for i, tile in enumerate(tiles):
        panel.paste(tile, ((i % 3) * 480, (i // 3) * 270))
    return panel


def warn_for_metrics(m: dict[str, float]) -> list[str]:
    warnings: list[str] = []
    if m["subject_coverage"] > 0.35:
        warnings.append("subject_coverage > 0.35: subject mask may be overfilled")
    if m["mae_jersey_rgb"] > 45:
        warnings.append("mae_jersey_rgb > 45: jersey/color distribution is still poor")
    if m["red_mask_iou"] < 0.35:
        warnings.append("red_mask_iou < 0.35: jersey mask extraction or red typography placement is bad")
    return warnings


def metrics(ref: np.ndarray, rec: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float | list[str]]:
    diff = np.abs(ref.astype(np.float32) - rec.astype(np.float32))
    luma_diff = np.abs(luma(ref).astype(np.float32) - luma(rec).astype(np.float32))
    shape = masks["shape"] > 0
    jersey = masks["jersey"] > 0
    face = masks["head"] > 0
    head_neck = (masks["head"] > 0) | (masks["neck"] > 0)
    ref_subject_edges = edge_mask(ref, masks["shape"])
    rec_subject_edges = edge_mask(rec, masks["shape"])
    ref_face_edges = edge_mask(ref, masks["head"])
    rec_face_edges = edge_mask(rec, masks["head"])
    ref_red = red_mask(ref, masks["shape"])
    rec_red = red_mask(rec, masks["shape"])
    red_union = ref_red | rec_red
    red_iou = float((ref_red & rec_red).sum() / red_union.sum()) if np.any(red_union) else 0.0

    m: dict[str, float | list[str]] = {
        "mae_full_rgb": float(diff.mean()),
        "mae_face_rgb": float(diff[face].mean()),
        "mae_head_neck_rgb": float(diff[head_neck].mean()),
        "mae_subject_rgb": float(diff[shape].mean()),
        "mae_jersey_rgb": float(diff[jersey].mean()),
        "mae_subject_luma": float(luma_diff[shape].mean()),
        "mae_jersey_luma": float(luma_diff[jersey].mean()),
        "mae_jersey_red_channel": float(diff[..., 0][jersey].mean()),
        "edge_overlap_subject": overlap_score(ref_subject_edges, rec_subject_edges),
        "edge_overlap_face": overlap_score(ref_face_edges, rec_face_edges),
        "red_mask_iou": red_iou,
        "subject_coverage": float(shape.mean()),
        "jersey_coverage": float(jersey.mean()),
    }
    m["warnings"] = warn_for_metrics(m)  # type: ignore[arg-type]
    return m


def save_forensic_diagnostics(ref: np.ndarray, rec: np.ndarray, masks: dict[str, np.ndarray]) -> None:
    mask_overlay(ref, masks["shape"], (80, 170, 255), "forensic v4 subject mask overlay").save(
        OUT / "forensic_v4_subject_overlay.png"
    )
    mask_overlay(ref, masks["jersey"], (255, 40, 42), "forensic v4 jersey mask overlay").save(
        OUT / "forensic_v4_jersey_overlay.png"
    )
    err = rgb_error(ref, rec)
    error_heatmap(err, masks["shape"], "forensic v4 RGB error heatmap: subject region").save(
        OUT / "forensic_v4_rgb_error_heatmap.png"
    )
    error_heatmap(err, masks["jersey"], "forensic v4 RGB error heatmap: jersey region").save(
        OUT / "forensic_v4_jersey_error_heatmap.png"
    )
    mask_panel(masks, ref, rec).save(OUT / "forensic_v4_mask_panel.png")


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rgb = load_target()
    masks = build_masks(rgb)
    guide = maps(rgb, masks["shape"])
    ref = Image.fromarray(rgb)

    ref.crop(FACE_CROP).save(ANALYSIS / "reference_crop_face.png")
    ref.crop(JERSEY_CROP).save(ANALYSIS / "reference_crop_jersey.png")
    annotated_regions(rgb).save(ANALYSIS / "reference_regions_annotated.png")
    density_map(rgb, masks, guide).save(ANALYSIS / "reference_density_map.png")
    orientation_map(rgb).save(ANALYSIS / "reference_orientation_map.png")

    full = composite_full(rgb, masks, guide)
    full.save(OUT / "one_shot_v4_final.png")
    side_by_side(ref, full).save(OUT / "one_shot_v4_side_by_side.png")

    face_ref = ref.crop(FACE_CROP)
    face = full.crop(FACE_CROP)
    face.save(OUT / "face_study_v2.png")
    side_by_side(face_ref, face).save(OUT / "face_study_v2_side_by_side.png")

    jersey_ref = ref.crop(JERSEY_CROP)
    jersey = full.crop(JERSEY_CROP)
    jersey.save(OUT / "jersey_study_v1.png")
    side_by_side(jersey_ref, jersey).save(OUT / "jersey_study_v1_side_by_side.png")

    rec = np.array(full)
    save_forensic_diagnostics(rgb, rec, masks)
    m = metrics(rgb, rec, masks)
    (OUT / "metrics_one_shot_v4.json").write_text(json.dumps(m, indent=2) + "\n")
    (OUT / "forensic_v4_metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
