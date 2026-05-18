from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage as ndi


DEFAULT_WORDS = {
    "hair": ["LEGEND", "POWER", "FOCUS", "STRENGTH"],
    "skin": ["COURAGE", "HOPE", "RESOLVE", "HEART"],
    "neck": ["TRAINING", "DISCIPLINE", "BATTLE"],
    "orange_gi": ["WARRIOR", "PROTECTOR", "BATTLE"],
    "blue_undershirt": ["STRENGTH", "RESOLVE", "TRAINING"],
    "outline": ["POWER", "LEGEND", "BATTLE"],
}


def build_parser(
    default_input: Path | None = None,
    default_out: Path | None = None,
    default_profile: Path | None = None,
    default_background: str = "black",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a one-shot semantic word-tribute portrait with canonical outputs."
    )
    parser.add_argument("--input", type=Path, default=default_input, required=default_input is None)
    parser.add_argument("--out-dir", type=Path, default=default_out or Path("outputs/word_tribute"))
    parser.add_argument("--profile", type=Path, default=default_profile)
    parser.add_argument("--subject", default=None, help="Optional subject label stored in the output profile.")
    parser.add_argument("--words", default=None, help="Comma-separated words appended to every semantic region.")
    parser.add_argument("--words-file", type=Path, default=None, help="Plain text file with one word or phrase per line.")
    parser.add_argument("--mask", type=Path, default=None, help="Optional binary subject mask override.")
    parser.add_argument("--background", choices=("black", "preserve"), default=default_background)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    metrics = render_from_paths(
        input_path=args.input,
        out_dir=args.out_dir,
        profile_path=args.profile,
        subject=args.subject,
        extra_words=words_from_args(args.words, args.words_file),
        mask_path=args.mask,
        background=args.background,
    )
    print(json.dumps(metrics, indent=2))


def run_cli(
    default_input: Path,
    default_out: Path,
    default_profile: Path,
    default_background: str = "black",
    argv: list[str] | None = None,
) -> None:
    args = build_parser(default_input, default_out, default_profile, default_background).parse_args(argv)
    metrics = render_from_paths(
        input_path=args.input,
        out_dir=args.out_dir,
        profile_path=args.profile,
        subject=args.subject,
        extra_words=words_from_args(args.words, args.words_file),
        mask_path=args.mask,
        background=args.background,
    )
    print(json.dumps(metrics, indent=2))


def words_from_args(words: str | None, words_file: Path | None) -> list[str]:
    out: list[str] = []
    if words:
        out.extend(word.strip() for word in words.split(",") if word.strip())
    if words_file:
        out.extend(line.strip() for line in words_file.read_text(encoding="utf-8").splitlines() if line.strip())
    return out


def load_profile(path: Path | None, subject: str | None, extra_words: list[str]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    if path:
        with path.open("r", encoding="utf-8") as handle:
            profile = json.load(handle)
    profile.setdefault("seed", 20260514)
    profile.setdefault("subject", subject or "subject")
    profile.setdefault("words", {key: list(value) for key, value in DEFAULT_WORDS.items()})
    profile.setdefault("anchors", [])
    profile.setdefault("lanes", [])
    profile.setdefault("foreground_hint", {})
    profile["background"] = profile.get("background", "black")
    if subject:
        profile["subject"] = subject
    if extra_words:
        for key in DEFAULT_WORDS:
            profile["words"].setdefault(key, [])
            profile["words"][key].extend(extra_words)
    for key, fallback in DEFAULT_WORDS.items():
        profile["words"].setdefault(key, list(fallback))
        if not profile["words"][key]:
            profile["words"][key] = list(fallback)
    return profile


def render_from_paths(
    input_path: Path,
    out_dir: Path,
    profile_path: Path | None = None,
    subject: str | None = None,
    extra_words: list[str] | None = None,
    mask_path: Path | None = None,
    background: str = "black",
) -> dict[str, float | int | list[int] | str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input reference: {input_path}")
    ref = np.array(Image.open(input_path).convert("RGB"), dtype=np.float32)
    profile = load_profile(profile_path, subject, extra_words or [])
    if background == "black":
        profile["background"] = "black"
    elif background == "preserve":
        profile["background"] = "preserve"
    subject_override = load_mask(mask_path, ref.shape[:2]) if mask_path else None
    metrics = render_word_tribute(ref, profile, out_dir, subject_override)
    (out_dir / "word_profile_used.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    return metrics


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.array(Image.open(path).convert("L").resize((shape[1], shape[0]), Image.Resampling.NEAREST))
    return mask > 127


def rgb_to_hsv(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.clip(arr / 255.0, 0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    diff = mx - mn
    hue = np.zeros_like(mx)
    mask = diff > 1e-6
    red = mask & (mx == r)
    green = mask & (mx == g)
    blue = mask & (mx == b)
    hue[red] = ((g[red] - b[red]) / diff[red]) % 6.0
    hue[green] = ((b[green] - r[green]) / diff[green]) + 2.0
    hue[blue] = ((r[blue] - g[blue]) / diff[blue]) + 4.0
    hue /= 6.0
    sat = np.zeros_like(mx)
    sat[mx > 1e-6] = diff[mx > 1e-6] / mx[mx > 1e-6]
    return hue, sat, mx


def luma(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722


def gradient_strength(gray: np.ndarray) -> np.ndarray:
    gx = ndi.sobel(gray, axis=1)
    gy = ndi.sobel(gray, axis=0)
    grad = np.hypot(gx, gy)
    max_val = float(np.percentile(grad, 99.4))
    return np.clip(grad / max(max_val, 1.0), 0.0, 1.0)


def keep_large_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.where(sizes >= min_pixels)[0]] = True
    keep[0] = False
    return keep[labels]


def keep_largest_component(mask: np.ndarray, x_bias: float = 0.0) -> np.ndarray:
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return mask
    _, w = mask.shape
    best_label = 0
    best_score = -1.0
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        score = float(xs.size) * (1.0 + x_bias * float(xs.mean() / max(w, 1)))
        if score > best_score:
            best_score = score
            best_label = label
    return labels == best_label


def clean_region(mask: np.ndarray, min_pixels: int = 80, close_size: int = 3) -> np.ndarray:
    out = ndi.binary_closing(mask, structure=np.ones((close_size, close_size), dtype=bool), iterations=1)
    out = ndi.binary_opening(out, structure=np.ones((2, 2), dtype=bool), iterations=1)
    return keep_large_components(out, min_pixels)


def build_masks(ref: np.ndarray, profile: dict[str, Any], subject_override: np.ndarray | None = None) -> dict[str, np.ndarray]:
    h, w = ref.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    xnorm = xx / max(w - 1, 1)
    ynorm = yy / max(h - 1, 1)
    hue, sat, val = rgb_to_hsv(ref)
    lum = luma(ref)
    edge = gradient_strength(lum)
    r, g, b = ref[..., 0], ref[..., 1], ref[..., 2]
    hint = profile.get("foreground_hint", {})
    allowed = xnorm >= float(hint.get("min_x_norm", 0.0))
    allowed &= xnorm <= float(hint.get("max_x_norm", 1.0))
    allowed &= ynorm >= float(hint.get("min_y_norm", 0.0))
    allowed &= ynorm <= float(hint.get("max_y_norm", 1.0))

    primary = (
        allowed
        & (r > 145)
        & (g > 38)
        & (g < 150)
        & (b < 105)
        & (r > g * 1.24)
        & (r > b * 1.85)
        & (sat > 0.48)
        & (val > 0.38)
    )
    primary = clean_region(primary, 240, 5)

    secondary = (
        allowed
        & (hue > 0.60)
        & (hue < 0.76)
        & (sat > 0.24)
        & (val < 0.55)
        & (b > r * 0.72)
    )
    secondary = clean_region(secondary, 120, 3)

    warm = (
        allowed
        & (hue > 0.015)
        & (hue < 0.105)
        & (sat > 0.16)
        & (sat < 0.72)
        & (val > 0.46)
        & (r > 135)
        & (g > 78)
        & (b > 50)
        & ~(primary & (sat > 0.55))
    )
    warm = clean_region(warm, 150, 5)

    dark = allowed & ((lum < 58) | ((val < 0.27) & (sat > 0.12)))
    dark &= ~(secondary & (ynorm > 0.76))
    dark_region = dark & (ynorm < float(hint.get("dark_max_y_norm", 1.0)))
    dark_region |= allowed & (lum < 85) & (sat < 0.45) & (ynorm < float(hint.get("dark_soft_max_y_norm", 1.0)))
    dark_region = ndi.binary_closing(dark_region, structure=np.ones((5, 5), dtype=bool), iterations=1)
    dark_region = keep_large_components(dark_region, 120)

    outline = allowed & ((lum < 72) | ((edge > 0.34) & (lum < 130)))
    outline |= ndi.binary_dilation(dark_region, structure=np.ones((3, 3), dtype=bool), iterations=1) & (edge > 0.16)
    outline = keep_large_components(outline, 30)

    if subject_override is None:
        subject_seed = primary | secondary | warm | dark_region | outline
        subject_seed = ndi.binary_dilation(subject_seed, structure=np.ones((9, 9), dtype=bool), iterations=1)
        subject_seed = ndi.binary_closing(subject_seed, structure=np.ones((19, 19), dtype=bool), iterations=1)
        subject = keep_largest_component(subject_seed, float(hint.get("component_x_bias", 0.7)))
        subject = ndi.binary_fill_holes(subject)
        subject = ndi.binary_dilation(subject, structure=np.ones((3, 3), dtype=bool), iterations=1)
        subject |= subject_seed & ndi.binary_dilation(subject, structure=np.ones((45, 45), dtype=bool), iterations=1)
        subject = ndi.binary_closing(subject, structure=np.ones((9, 9), dtype=bool), iterations=1)
        subject = ndi.binary_fill_holes(subject)
    else:
        subject = ndi.binary_fill_holes(subject_override)
        subject = ndi.binary_closing(subject, structure=np.ones((5, 5), dtype=bool), iterations=1)

    primary &= subject
    secondary &= subject
    warm &= subject & ~primary & ~secondary
    dark_region &= subject & ~primary & ~secondary & ~warm
    outline &= subject
    eye_feature = outline & warm & (ynorm > 0.30) & (ynorm < 0.62)
    outline_shadow = outline | (dark_region & (edge > 0.12))
    light_subject = subject & ~(dark_region | warm | primary | secondary)

    return {
        "subject": subject.astype(bool),
        "hair": dark_region.astype(bool),
        "skin": warm.astype(bool),
        "eye_feature": eye_feature.astype(bool),
        "orange_gi": primary.astype(bool),
        "blue_undershirt": secondary.astype(bool),
        "outline": outline_shadow.astype(bool),
        "background": (~subject).astype(bool),
        "light_subject": light_subject.astype(bool),
        "luma": lum.astype(np.float32),
        "edge_strength": edge.astype(np.float32),
    }


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_text(
    layer: Image.Image,
    x: float,
    y: float,
    text: str,
    size: int,
    color: tuple[int, int, int],
    alpha: int,
    angle: float,
    bold: bool = False,
) -> None:
    fnt = font(size, bold)
    probe = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=fnt)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])
    pad = max(8, size // 2)
    patch = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(patch).text((pad - bbox[0], pad - bbox[1]), text, font=fnt, fill=(*color, int(alpha)))
    rotated = patch.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    layer.alpha_composite(rotated, (int(x - rotated.width / 2), int(y - rotated.height / 2)))


def clip_layer(layer: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(layer, dtype=np.uint8)
    arr[..., 3] = np.minimum(arr[..., 3], mask.astype(np.uint8) * 255)
    return Image.fromarray(arr, "RGBA")


def sample_points(mask: np.ndarray, count: int, rng: random.Random) -> list[tuple[int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return []
    points: list[tuple[int, int]] = []
    for _ in range(count):
        idx = rng.randrange(xs.size)
        points.append((int(xs[idx]), int(ys[idx])))
    return points


def local_tangent(edge: np.ndarray, x: int, y: int, rng: random.Random) -> float:
    y0, y1 = max(0, y - 2), min(edge.shape[0], y + 3)
    x0, x1 = max(0, x - 2), min(edge.shape[1], x + 3)
    patch = edge[y0:y1, x0:x1]
    if patch.size < 4 or float(patch.max()) < 0.08:
        return rng.uniform(-28, 28)
    gy, gx = np.gradient(patch.astype(np.float32))
    return math.degrees(math.atan2(float(gx.mean()), -float(gy.mean()))) + rng.uniform(-9, 9)


def region_color(ref: np.ndarray, maps: dict[str, np.ndarray], region: str, x: int, y: int, rng: random.Random) -> tuple[int, int, int]:
    edge = float(maps["edge_strength"][y, x])
    base = ref[y, x]
    if region == "hair":
        if rng.random() < 0.34 or edge > 0.18:
            value = int(np.clip(82 + edge * 126 + rng.randrange(-18, 24), 46, 220))
            return value, value + 2, min(255, value + 14)
        value = rng.randrange(8, 45)
        return value, value, min(56, value + 8)
    if region == "skin":
        if rng.random() < 0.45:
            return 92, 46, 30
        return (
            int(np.clip(base[0] * 1.12 + 18, 120, 255)),
            int(np.clip(base[1] * 1.06 + 12, 92, 235)),
            int(np.clip(base[2] * 0.92 + 8, 58, 205)),
        )
    if region == "orange_gi":
        return random_choice(rng, [(245, 78, 24), (204, 43, 20), (255, 162, 42), (118, 24, 16), (255, 210, 128)])
    if region == "blue_undershirt":
        return random_choice(rng, [(20, 18, 75), (35, 38, 126), (70, 74, 180), (185, 198, 255), (12, 12, 42)])
    if region == "outline":
        value = rng.randrange(0, 34)
        return value, value, min(48, value + 6)
    value = int(np.clip(luma(ref[y : y + 1, x : x + 1])[0, 0] + 28, 40, 245))
    return value, value, value


def random_choice(rng: random.Random, values: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    return values[rng.randrange(len(values))]


def render_region_texture(
    ref: np.ndarray,
    maps: dict[str, np.ndarray],
    region: str,
    words: list[str],
    count: int,
    size_range: tuple[int, int],
    alpha_range: tuple[int, int],
    rng: random.Random,
) -> Image.Image:
    h, w = ref.shape[:2]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for x, y in sample_points(maps[region], count, rng):
        word = words[rng.randrange(len(words))]
        size = rng.randrange(size_range[0], size_range[1] + 1)
        alpha = rng.randrange(alpha_range[0], alpha_range[1] + 1)
        if region in {"hair", "outline"}:
            angle = rng.choice([-58, -34, -18, 0, 18, 34, 58]) + rng.uniform(-8, 8)
        elif region == "skin":
            angle = local_tangent(maps["edge_strength"], x, y, rng)
        else:
            angle = rng.uniform(-25, 25)
        draw_text(layer, x, y, word, size, region_color(ref, maps, region, x, y, rng), alpha, angle, size >= 20)
    return clip_layer(layer, maps[region])


def interpolate_lane(points: list[list[float]], width: int, height: int, spacing: float) -> list[tuple[float, float, float]]:
    scaled = [(x * width, y * height) for x, y in points]
    segments: list[tuple[float, float, float, float, float]] = []
    total = 0.0
    for (x0, y0), (x1, y1) in zip(scaled[:-1], scaled[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1:
            continue
        segments.append((x0, y0, x1, y1, length))
        total += length
    samples: list[tuple[float, float, float]] = []
    dist = 0.0
    while dist <= total:
        cursor = dist
        for x0, y0, x1, y1, length in segments:
            if cursor <= length:
                ratio = cursor / length
                samples.append((x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio, math.degrees(math.atan2(y1 - y0, x1 - x0))))
                break
            cursor -= length
        dist += spacing
    return samples


def nearest_allowed(x: int, y: int, mask: np.ndarray, radius: int) -> tuple[int, int] | None:
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]:
        return x, y
    x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
    y0, y1 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
    ys, xs = np.where(mask[y0:y1, x0:x1])
    if xs.size == 0:
        return None
    dist = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
    idx = int(np.argmin(dist))
    return int(xs[idx] + x0), int(ys[idx] + y0)


def resolve_words(spec: Any, profile: dict[str, Any]) -> list[str]:
    if isinstance(spec, str):
        return list(profile["words"][spec])
    return [str(item) for item in spec]


def render_lanes(ref: np.ndarray, maps: dict[str, np.ndarray], profile: dict[str, Any], rng: random.Random) -> tuple[Image.Image, Image.Image, int]:
    h, w = ref.shape[:2]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    placed = 0
    for lane in profile.get("lanes", []):
        region = str(lane["region"])
        words = resolve_words(lane["words"], profile)
        samples = interpolate_lane(lane["points"], w, h, float(lane["spacing"]))
        points_px = [(int(x), int(y)) for x, y, _ in samples]
        if len(points_px) > 1:
            draw_overlay.line(points_px, fill=(0, 255, 255, 165), width=3)
        region_mask = maps["subject"] if region == "outline" else maps[region]
        for idx, (x, y, angle) in enumerate(samples):
            ix, iy = int(round(x)), int(round(y))
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                continue
            if not bool(region_mask[iy, ix]):
                near = nearest_allowed(ix, iy, region_mask, 22)
                if near is None:
                    continue
                ix, iy = near
            word = words[idx % len(words)]
            draw_text(layer, ix, iy, word, int(lane["size"]), region_color(ref, maps, region, ix, iy, rng), int(lane["alpha"]), angle + rng.uniform(-5, 5))
            placed += 1
    return clip_layer(layer, maps["subject"]), overlay, placed


def render_anchors(ref: np.ndarray, maps: dict[str, np.ndarray], profile: dict[str, Any], rng: random.Random) -> tuple[Image.Image, Image.Image, int]:
    h, w = ref.shape[:2]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    placed = 0
    for anchor in profile.get("anchors", []):
        region = str(anchor["region"])
        x = int(float(anchor["pos"][0]) * w)
        y = int(float(anchor["pos"][1]) * h)
        point = nearest_allowed(x, y, maps[region], 55)
        if point is None:
            continue
        px, py = point
        color = region_color(ref, maps, region, px, py, rng)
        if region == "hair":
            color = tuple(max(c, 118) for c in color)
        draw_text(layer, px, py, str(anchor["text"]), int(anchor["size"]), color, int(anchor["alpha"]), float(anchor["angle"]), bool(anchor.get("bold", False)))
        draw_overlay.ellipse((px - 8, py - 8, px + 8, py + 8), fill=(255, 230, 0, 210))
        draw_overlay.text((px + 10, py - 10), str(anchor["text"]), font=font(18, True), fill=(255, 255, 255, 225))
        placed += 1
    return clip_layer(layer, maps["subject"]), overlay, placed


def subject_base(ref: np.ndarray, maps: dict[str, np.ndarray], background: str) -> np.ndarray:
    base = np.zeros_like(ref) if background == "black" else ref.copy()
    subject = maps["subject"]
    recon = np.zeros_like(ref)
    recon[subject] = ref[subject] * 0.24 + 10
    recon[maps["hair"]] = ref[maps["hair"]] * 0.18
    recon[maps["skin"]] = ref[maps["skin"]] * 0.60 + np.array([10, 6, 3], dtype=np.float32)
    recon[maps["orange_gi"]] = ref[maps["orange_gi"]] * 0.50
    recon[maps["blue_undershirt"]] = ref[maps["blue_undershirt"]] * 0.42
    recon[maps["light_subject"]] = ref[maps["light_subject"]] * 0.38 + 16
    recon[maps["outline"]] = ref[maps["outline"]] * 0.70
    alpha = np.array(Image.fromarray(subject.astype(np.uint8) * 255, "L").filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32) / 255.0
    return np.clip(base * (1.0 - alpha[..., None]) + recon * alpha[..., None], 0, 255)


def composite_layers(ref: np.ndarray, maps: dict[str, np.ndarray], layers: list[Image.Image], background: str) -> np.ndarray:
    out = subject_base(ref, maps, background)
    canvas = Image.fromarray(out.astype(np.uint8), "RGB").convert("RGBA")
    for layer in layers:
        canvas = Image.alpha_composite(canvas, layer)
    arr = np.array(canvas.convert("RGB"), dtype=np.float32)
    outline = maps["outline"]
    arr[outline] = arr[outline] * 0.28 + ref[outline] * 0.72
    if background == "preserve":
        arr[~maps["subject"]] = ref[~maps["subject"]]
    else:
        arr[~maps["subject"]] = 0
        rim = ndi.binary_dilation(maps["subject"], structure=np.ones((5, 5), dtype=bool), iterations=1) & ~maps["subject"]
        edge = maps["edge_strength"]
        arr[rim] = np.maximum(arr[rim], np.clip(edge[rim, None] * 95.0, 0, 95))
    return np.clip(arr, 0, 255)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path)


def make_mask_panel(maps: dict[str, np.ndarray]) -> Image.Image:
    names = [
        ("subject", "subject"),
        ("hair", "dark / hair"),
        ("skin", "warm / skin"),
        ("orange_gi", "primary warm"),
        ("blue_undershirt", "secondary blue"),
        ("outline", "outline / shadow"),
        ("background", "background"),
        ("eye_feature", "eye features"),
    ]
    h, w = maps["subject"].shape
    thumb_w = 480
    thumb_h = int(round(h * thumb_w / w))
    panel = Image.new("RGB", (thumb_w * 4, (thumb_h + 44) * 2), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    for idx, (key, label) in enumerate(names):
        x = (idx % 4) * thumb_w
        y = (idx // 4) * (thumb_h + 44)
        mask_img = Image.fromarray(maps[key].astype(np.uint8) * 255, "L").resize((thumb_w, thumb_h), Image.Resampling.NEAREST).convert("RGB")
        panel.paste(mask_img, (x, y + 34))
        draw.text((x + 12, y + 8), label, font=font(20, True), fill=(255, 255, 255))
    return panel


def side_by_side(ref: np.ndarray, rec: np.ndarray, background: str) -> Image.Image:
    left = Image.fromarray(ref.astype(np.uint8), "RGB")
    right = Image.fromarray(rec.astype(np.uint8), "RGB")
    out = Image.new("RGB", (left.width + right.width, left.height), (0, 0, 0))
    out.paste(left, (0, 0))
    out.paste(right, (left.width, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle((18, 16, 170, 52), fill=(0, 0, 0))
    draw.rectangle((left.width + 18, 16, left.width + 330, 52), fill=(0, 0, 0))
    draw.text((28, 22), "reference", font=font(22, True), fill=(255, 255, 255))
    draw.text((left.width + 28, 22), f"current best / {background}", font=font(22, True), fill=(255, 255, 255))
    return out


def threshold_orange(arr: np.ndarray) -> np.ndarray:
    _, sat, val = rgb_to_hsv(arr)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (r > 135) & (g > 34) & (g < 170) & (b < 120) & (r > g * 1.15) & (r > b * 1.55) & (sat > 0.38) & (val > 0.28)


def threshold_blue(arr: np.ndarray) -> np.ndarray:
    hue, sat, val = rgb_to_hsv(arr)
    return (hue > 0.58) & (hue < 0.78) & (sat > 0.20) & (val < 0.62)


def threshold_hair(arr: np.ndarray) -> np.ndarray:
    hue, sat, val = rgb_to_hsv(arr)
    lum = luma(arr)
    return (lum < 75) | ((val < 0.32) & (sat < 0.55))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b)
    if not np.any(union):
        return 1.0
    return float(np.logical_and(a, b).sum() / union.sum())


def edge_overlap(ref: np.ndarray, rec: np.ndarray, mask: np.ndarray) -> float:
    ref_g = gradient_strength(luma(ref))
    rec_g = gradient_strength(luma(rec))
    if not np.any(mask):
        return 0.0
    ref_thresh = float(np.percentile(ref_g[mask], 76))
    rec_thresh = float(np.percentile(rec_g[mask], 70))
    ref_edges = (ref_g > ref_thresh) & mask
    rec_edges = ndi.binary_dilation((rec_g > rec_thresh) & mask, structure=np.ones((5, 5), dtype=bool), iterations=1)
    if not np.any(ref_edges):
        return 0.0
    return float(np.logical_and(ref_edges, rec_edges).sum() / ref_edges.sum())


def floating_fragment_count(rec: np.ndarray, subject: np.ndarray) -> int:
    visible = (luma(rec) > 12) & ~ndi.binary_dilation(subject, structure=np.ones((7, 7), dtype=bool), iterations=1)
    labels, count = ndi.label(visible, structure=np.ones((3, 3), dtype=bool))
    fragments = 0
    for label in range(1, count + 1):
        area = int((labels == label).sum())
        if 8 <= area < 240:
            fragments += 1
    return fragments


def compute_metrics(ref: np.ndarray, rec: np.ndarray, maps: dict[str, np.ndarray], render_time: float, background: str) -> dict[str, float | int | list[int] | str]:
    diff = np.abs(ref.astype(np.float32) - rec.astype(np.float32))
    subject = maps["subject"]

    def region_mae(mask_name: str) -> float:
        mask = maps[mask_name]
        if not np.any(mask):
            return 0.0
        return float(diff[mask].mean())

    orange_out = threshold_orange(rec) & subject
    blue_out = threshold_blue(rec) & subject
    hair_out = threshold_hair(rec) & subject
    bg = ~subject
    bg_luma = luma(rec)[bg] if np.any(bg) else np.array([0.0], dtype=np.float32)
    return {
        "mae_full_rgb": float(diff.mean()),
        "mae_subject_rgb": float(diff[subject].mean()) if np.any(subject) else 0.0,
        "mae_hair_rgb": region_mae("hair"),
        "mae_skin_rgb": region_mae("skin"),
        "mae_gi_rgb": region_mae("orange_gi"),
        "mae_undershirt_rgb": region_mae("blue_undershirt"),
        "edge_overlap_subject": edge_overlap(ref, rec, subject),
        "hair_mask_iou": iou(maps["hair"], hair_out),
        "orange_mask_iou": iou(maps["orange_gi"], orange_out),
        "blue_mask_iou": iou(maps["blue_undershirt"], blue_out),
        "floating_fragment_count": floating_fragment_count(rec, subject),
        "background_cleanliness": float(np.mean(bg_luma < 6.0)) if background == "black" else 0.0,
        "render_time_seconds": round(render_time, 3),
        "output_resolution": [int(ref.shape[1]), int(ref.shape[0])],
        "subject_coverage": float(subject.mean()),
        "background": background,
    }


def write_diagnostics(out_dir: Path, maps: dict[str, np.ndarray], lane_overlay: Image.Image, anchor_overlay: Image.Image) -> None:
    diagnostics = out_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    save_mask(maps["subject"], diagnostics / "subject_mask.png")
    save_mask(maps["hair"], diagnostics / "dark_region_mask.png")
    save_mask(maps["skin"], diagnostics / "warm_region_mask.png")
    save_mask(maps["eye_feature"], diagnostics / "eye_feature_mask.png")
    save_mask(maps["orange_gi"], diagnostics / "primary_color_region_mask.png")
    save_mask(maps["blue_undershirt"], diagnostics / "secondary_color_region_mask.png")
    save_mask(maps["outline"], diagnostics / "outline_shadow_mask.png")
    save_mask(maps["background"], diagnostics / "background_mask.png")
    lane_overlay.convert("RGB").save(diagnostics / "lane_overlay.png")
    anchor_overlay.convert("RGB").save(diagnostics / "anchor_overlay.png")


def render_word_tribute(
    ref: np.ndarray,
    profile: dict[str, Any],
    out_dir: Path,
    subject_override: np.ndarray | None = None,
) -> dict[str, float | int | list[int] | str]:
    start = perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(profile.get("seed", 20260514)))
    background = str(profile.get("background", "black"))
    maps = build_masks(ref, profile, subject_override)
    words = profile["words"]

    hair_count = max(260, int(maps["hair"].sum() / 520))
    skin_count = max(240, int(maps["skin"].sum() / 470))
    primary_count = max(420, int(maps["orange_gi"].sum() / 620))
    secondary_count = max(240, int(maps["blue_undershirt"].sum() / 430))
    outline_count = max(260, int(maps["outline"].sum() / 180))
    light_count = max(140, int(maps["light_subject"].sum() / 900))

    hair_layer = render_region_texture(ref, maps, "hair", words["hair"], hair_count, (9, 27), (92, 226), rng)
    skin_layer = render_region_texture(ref, maps, "skin", words["skin"] + words.get("neck", []), skin_count, (8, 20), (96, 214), rng)
    primary_layer = render_region_texture(ref, maps, "orange_gi", words["orange_gi"], primary_count, (10, 26), (98, 222), rng)
    secondary_layer = render_region_texture(ref, maps, "blue_undershirt", words["blue_undershirt"], secondary_count, (9, 22), (105, 225), rng)
    outline_layer = render_region_texture(ref, maps, "outline", words["outline"], outline_count, (6, 14), (135, 240), rng)
    light_layer = render_region_texture(ref, maps, "light_subject", words["skin"] + words["orange_gi"], light_count, (7, 15), (75, 170), rng)
    lane_layer, lane_overlay, lane_words = render_lanes(ref, maps, profile, rng)
    anchor_layer, anchor_overlay, anchor_words = render_anchors(ref, maps, profile, rng)

    rec = composite_layers(
        ref,
        maps,
        [hair_layer, skin_layer, primary_layer, secondary_layer, light_layer, lane_layer, anchor_layer, outline_layer],
        background,
    ).astype(np.uint8)
    metrics = compute_metrics(ref, rec.astype(np.float32), maps, perf_counter() - start, background)
    metrics["subject_pixels"] = int(maps["subject"].sum())
    metrics["anchor_words"] = anchor_words
    metrics["contour_lane_words"] = lane_words
    metrics["texture_words"] = hair_count + skin_count + primary_count + secondary_count + outline_count + light_count

    Image.fromarray(rec, "RGB").save(out_dir / "current_best.png")
    side_by_side(ref, rec.astype(np.float32), background).save(out_dir / "current_best_side_by_side.png")
    make_mask_panel(maps).save(out_dir / "mask_panel.png")
    lane_overlay.convert("RGB").save(out_dir / "lane_overlay.png")
    anchor_overlay.convert("RGB").save(out_dir / "anchor_overlay.png")
    write_diagnostics(out_dir, maps, lane_overlay, anchor_overlay)
    (out_dir / "current_best_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    main()
