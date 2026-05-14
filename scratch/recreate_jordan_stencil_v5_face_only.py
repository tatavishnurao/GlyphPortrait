from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch import recreate_jordan_reference_stencil as v1
from scratch import recreate_jordan_reference_stencil_v2 as v2
from scratch import recreate_jordan_reference_stencil_v3 as v3
from scratch import recreate_jordan_reference_stencil_v4 as v4

OUT = ROOT / "examples" / "reference_recreation"
W, H = v1.W, v1.H
SEED = 20260518

TIERED_FACE_ANCHORS = [
    {"tier": 1, "text": "NBA Rookie of the Year", "mode": "path", "points": [(0.18, 0.10), (0.48, 0.035), (0.82, 0.12)], "size": 24, "alpha": 218},
    {"tier": 1, "text": "MVP", "mode": "pos", "pos": (0.40, 0.245), "size": 42, "alpha": 224, "angle": -6},
    {"tier": 1, "text": "Air Jordan", "mode": "pos", "pos": (0.22, 0.355), "size": 29, "alpha": 214, "angle": 4},
    {"tier": 1, "text": "Dedication", "mode": "path", "points": [(0.20, 0.405), (0.31, 0.510), (0.47, 0.595)], "size": 29, "alpha": 194},
    {"tier": 1, "text": "Dominance", "mode": "path", "points": [(0.20, 0.535), (0.245, 0.690), (0.335, 0.900)], "size": 30, "alpha": 216},
    {"tier": 2, "text": "Love of the Game", "mode": "pos", "pos": (0.29, 0.475), "size": 21, "alpha": 166, "angle": 14},
    {"tier": 2, "text": "Finals MVP", "mode": "pos", "pos": (0.48, 0.660), "size": 22, "alpha": 170, "angle": 34},
    {"tier": 2, "text": "Scoring", "mode": "pos", "pos": (0.69, 0.510), "size": 25, "alpha": 188, "angle": -8},
    {"tier": 2, "text": "Champion", "mode": "pos", "pos": (0.34, 0.165), "size": 20, "alpha": 156, "angle": 8},
    {"tier": 2, "text": "Defense", "mode": "pos", "pos": (0.22, 0.300), "size": 18, "alpha": 146, "angle": -3},
    {"tier": 2, "text": "1984", "mode": "pos", "pos": (0.30, 0.610), "size": 17, "alpha": 140, "angle": 55},
    {"tier": 2, "text": "1985", "mode": "pos", "pos": (0.66, 0.370), "size": 17, "alpha": 142, "angle": -12},
]

FACE_LANES = [
    {"id": "forehead_arc", "points": [(0.10, 0.145), (0.34, 0.045), (0.70, 0.085), (0.88, 0.245)], "words": ["Rookie of the Year", "MVP", "Champion", "1984"], "spacing": 19, "size": (7, 13), "alpha": (116, 202)},
    {"id": "skull_curve_left", "points": [(0.09, 0.190), (0.065, 0.330), (0.098, 0.505), (0.180, 0.640)], "words": ["Air Jordan", "Defense", "Dedication"], "spacing": 18, "size": (7, 13), "alpha": (116, 196)},
    {"id": "skull_curve_right", "points": [(0.790, 0.150), (0.865, 0.310), (0.875, 0.505), (0.790, 0.650)], "words": ["Scoring", "Finals MVP", "Champion"], "spacing": 18, "size": (7, 13), "alpha": (112, 194)},
    {"id": "brow_ridge", "points": [(0.135, 0.305), (0.315, 0.288), (0.570, 0.318), (0.820, 0.352)], "words": ["MVP", "Defense", "Focus", "Champion"], "spacing": 17, "size": (7, 13), "alpha": (112, 196)},
    {"id": "upper_eye_socket", "points": [(0.135, 0.352), (0.330, 0.334), (0.610, 0.358)], "words": ["Air Jordan", "Finals MVP", "Dedication"], "spacing": 16, "size": (6, 12), "alpha": (108, 184)},
    {"id": "lower_eye_socket", "points": [(0.165, 0.397), (0.375, 0.428), (0.660, 0.405)], "words": ["Clutch", "Scoring", "Champion"], "spacing": 16, "size": (6, 12), "alpha": (106, 180)},
    {"id": "nose_bridge", "points": [(0.555, 0.315), (0.600, 0.445), (0.565, 0.615)], "words": ["Scoring", "Finals MVP", "MVP"], "spacing": 15, "size": (6, 12), "alpha": (124, 210)},
    {"id": "nose_side_highlight", "points": [(0.675, 0.380), (0.730, 0.505), (0.800, 0.640)], "words": ["Scoring", "Air", "Jordan"], "spacing": 16, "size": (6, 12), "alpha": (120, 204)},
    {"id": "left_cheek_plane", "points": [(0.150, 0.440), (0.285, 0.505), (0.505, 0.565)], "words": ["Dedication", "Love of the Game", "Champion"], "spacing": 18, "size": (7, 13), "alpha": (104, 180)},
    {"id": "right_cheek_plane", "points": [(0.590, 0.455), (0.755, 0.505), (0.845, 0.625)], "words": ["Scoring", "Finals MVP", "Clutch"], "spacing": 17, "size": (7, 13), "alpha": (114, 190)},
    {"id": "mouth_crease", "points": [(0.365, 0.610), (0.560, 0.608), (0.785, 0.638)], "words": ["Clutch", "Drive", "Focus", "MVP"], "spacing": 14, "size": (6, 11), "alpha": (108, 178)},
    {"id": "chin_curve", "points": [(0.345, 0.705), (0.540, 0.785), (0.780, 0.720)], "words": ["Finals MVP", "Dominance", "Champion"], "spacing": 17, "size": (7, 13), "alpha": (102, 172)},
    {"id": "jawline", "points": [(0.105, 0.420), (0.140, 0.580), (0.240, 0.760), (0.405, 0.910)], "words": ["Dominance", "Defense", "Champion"], "spacing": 17, "size": (7, 14), "alpha": (116, 202)},
    {"id": "neck_column", "points": [(0.205, 0.640), (0.260, 0.790), (0.355, 0.985)], "words": ["Dominance", "Dedication", "Air Jordan"], "spacing": 17, "size": (7, 14), "alpha": (112, 196)},
    {"id": "neck_shadow_edge", "points": [(0.445, 0.650), (0.520, 0.810), (0.650, 0.985)], "words": ["Defense", "Focus", "MVP"], "spacing": 16, "size": (6, 12), "alpha": (96, 172)},
]


def face_crop_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = v1.bbox(mask)
    return max(0, x0 - 84), max(0, y0 - 58), min(W, x1 + 72), min(H, y1 + 54)


def crop_to_full(point: tuple[float, float], box: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return x0 + point[0] * (x1 - x0), y0 + point[1] * (y1 - y0)


def sample_crop_path(points: list[tuple[float, float]], box: tuple[int, int, int, int], spacing: float, offset: float = 0.0) -> list[tuple[float, float, float]]:
    full = [crop_to_full(p, box) for p in points]
    segments: list[tuple[float, float, float, float, float]] = []
    total = 0.0
    for (x0, y0), (x1, y1) in zip(full[:-1], full[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1.0:
            continue
        segments.append((x0, y0, x1, y1, length))
        total += length
    samples: list[tuple[float, float, float]] = []
    d = offset
    while d <= total:
        cursor = d
        for x0, y0, x1, y1, length in segments:
            if cursor <= length:
                t = cursor / length
                samples.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, math.degrees(math.atan2(y1 - y0, x1 - x0))))
                break
            cursor -= length
        d += spacing
    return samples


def local_face_color(ref: np.ndarray, maps: dict[str, np.ndarray], x: int, y: int, lift: float = 1.0) -> tuple[int, int, int]:
    ref_l = float(v1.luma(ref[y : y + 1, x : x + 1])[0, 0])
    edge = float(maps["edge_strength"][y, x])
    value = int(np.clip(ref_l * lift + maps["enhanced_luma"][y, x] * 70.0 + edge * 82.0, 54, 246))
    return value, value, min(255, value + 5)


def nearest_allowed(x: int, y: int, allowed: np.ndarray, radius: int = 16) -> tuple[int, int] | None:
    if 0 <= x < W and 0 <= y < H and allowed[y, x]:
        return x, y
    x0, x1 = max(0, x - radius), min(W, x + radius + 1)
    y0, y1 = max(0, y - radius), min(H, y + radius + 1)
    ys, xs = np.where(allowed[y0:y1, x0:x1])
    if xs.size == 0:
        return None
    dist = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
    idx = int(np.argmin(dist))
    return int(xs[idx] + x0), int(ys[idx] + y0)


def protected_interior(maps: dict[str, np.ndarray]) -> np.ndarray:
    return ndi.binary_erosion(maps["dark_zone"], structure=np.ones((5, 5), dtype=bool), iterations=1)


def render_tiered_anchors(
    ref: np.ndarray,
    maps: dict[str, np.ndarray],
    visibility: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[np.ndarray, Image.Image, int]:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    allowed = maps["face"] & ~protected_interior(maps)
    words = 0
    for spec in TIERED_FACE_ANCHORS:
        if spec["mode"] == "pos":
            x, y = crop_to_full(spec["pos"], box)  # type: ignore[arg-type]
            angle = float(spec.get("angle", 0.0))
        else:
            samples = sample_crop_path(spec["points"], box, 42.0, 0.0)  # type: ignore[arg-type]
            if not samples:
                continue
            x, y, angle = samples[len(samples) // 2]
        point = nearest_allowed(int(x), int(y), allowed, 30)
        if point is None:
            continue
        px, py = point
        lift = 1.14 if int(spec["tier"]) == 1 else 1.02
        v1.draw_text(
            layer,
            px,
            py,
            str(spec["text"]),
            int(spec["size"]),
            local_face_color(ref, maps, px, py, lift),
            int(spec["alpha"]),
            angle,
            True,
        )
        words += 1
    clipped = v1.clip_layer(layer, maps["face"])
    arr = v1.alpha_to_rgb(clipped)
    arr *= np.clip(0.24 + visibility * 0.78 + maps["edge_strength"] * 0.42, 0.0, 1.18)[..., None]
    arr[maps["dark_zone"]] *= v3.dark_zone_soft_visibility(maps)[maps["dark_zone"], None]
    arr[~maps["face"]] = 0
    return np.clip(arr, 0, 255), clipped, words


def lane_corridor(points: list[tuple[float, float]], box: tuple[int, int, int, int], radius: int = 17) -> np.ndarray:
    trace = np.zeros((H, W), dtype=bool)
    for x, y, _ in sample_crop_path(points, box, 5.0, 0.0):
        ix, iy = int(x), int(y)
        if 0 <= ix < W and 0 <= iy < H:
            trace[iy, ix] = True
    return ndi.binary_dilation(trace, structure=np.ones((radius, radius), dtype=bool), iterations=1)


def render_face_lanes(
    ref: np.ndarray,
    maps: dict[str, np.ndarray],
    visibility: np.ndarray,
    box: tuple[int, int, int, int],
    rng: random.Random,
) -> tuple[np.ndarray, Image.Image, int, np.ndarray]:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    allowed_base = maps["face"] & ~protected_interior(maps)
    used = np.zeros((H, W), dtype=bool)
    total_words = 0
    for lane in FACE_LANES:
        corridor = lane_corridor(lane["points"], box, 19)  # type: ignore[arg-type]
        allowed = allowed_base & corridor
        spacing = float(lane["spacing"])
        for pass_i in range(3):
            offset = rng.uniform(0.0, spacing * 0.55) + pass_i * spacing * 0.36
            for x, y, angle in sample_crop_path(lane["points"], box, spacing * 0.70, offset):  # type: ignore[arg-type]
                point = nearest_allowed(int(x), int(y), allowed, 20)
                if point is None:
                    continue
                px, py = point
                size_min, size_max = lane["size"]  # type: ignore[misc]
                alpha_min, alpha_max = lane["alpha"]  # type: ignore[misc]
                edge = float(maps["edge_strength"][py, px])
                alpha = int(np.clip(rng.randint(int(alpha_min), int(alpha_max)) + edge * 54, 70, 235))
                words = lane["words"]  # type: ignore[assignment]
                v1.draw_text(
                    layer,
                    px + rng.randint(-2, 2),
                    py + rng.randint(-2, 2),
                    rng.choice(words),  # type: ignore[arg-type]
                    rng.randint(int(size_min), int(size_max)),
                    local_face_color(ref, maps, px, py, 1.12),
                    alpha,
                    angle + rng.uniform(-3.5, 3.5),
                    True,
                )
                used[py, px] = True
                total_words += 1

        candidates = np.argwhere(allowed & maps["edge_map"] & (maps["tone"] > 0.055))
        if candidates.size:
            rng.shuffle(candidates)
            for y, x in candidates[: min(28, candidates.shape[0])]:
                px, py = int(x + rng.randint(-3, 3)), int(y + rng.randint(-3, 3))
                if not (0 <= px < W and 0 <= py < H and allowed[py, px]):
                    continue
                size_min, size_max = lane["size"]  # type: ignore[misc]
                alpha_min, alpha_max = lane["alpha"]  # type: ignore[misc]
                edge = float(maps["edge_strength"][py, px])
                words = lane["words"]  # type: ignore[assignment]
                v1.draw_text(
                    layer,
                    px,
                    py,
                    rng.choice(words),  # type: ignore[arg-type]
                    rng.randint(int(size_min), int(size_max)),
                    local_face_color(ref, maps, px, py, 1.25),
                    int(np.clip(int(alpha_max) + edge * 70 + rng.randint(-8, 16), 118, 240)),
                    v1.tangent_angle(maps, px, py, 78) + rng.uniform(-3.5, 3.5),
                    True,
                )
                used[py, px] = True
                total_words += 1
    clipped = v1.clip_layer(layer, maps["face"] & ~protected_interior(maps))
    arr = v1.alpha_to_rgb(clipped)
    arr *= np.clip(0.28 + visibility * 0.88 + maps["edge_strength"] * 0.82, 0.0, 1.42)[..., None]
    arr[maps["dark_zone"]] *= v3.dark_zone_soft_visibility(maps)[maps["dark_zone"], None]
    arr[~maps["face"]] = 0
    lane_mask = ndi.binary_dilation(used, structure=np.ones((5, 5), dtype=bool), iterations=1)
    return np.clip(arr, 0, 255), clipped, total_words, lane_mask


def render_too_dark_face_microtext(
    ref: np.ndarray,
    current: np.ndarray,
    maps: dict[str, np.ndarray],
    lane_mask: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    ref_l = v1.luma(ref)
    out_l = v1.luma(current)
    regions = v3.face_recovery_regions(maps)
    side_highlight = maps["face"] & (maps["edge_strength"] > 0.12) & (maps["tone"] > 0.10)
    target = regions["cheek"] | regions["jaw"] | regions["neck"] | regions["mouth_chin"] | side_highlight | lane_mask
    mask = (ref_l - out_l > 14) & target & ~protected_interior(maps) & (maps["tone"] > 0.075)
    mask = ndi.binary_opening(mask, structure=np.ones((2, 2), dtype=bool), iterations=1)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    words = ["Air Jordan", "Dedication", "Dominance", "Scoring", "Champion", "Finals MVP", "Defense", "MVP"]
    ys, xs = np.where(mask)
    drawn = 0
    if xs.size:
        weights = 0.25 + maps["edge_strength"][ys, xs] * 1.1 + np.clip((ref_l[ys, xs] - out_l[ys, xs]) / 72, 0, 1)
        weights = weights / max(1e-6, float(weights.sum()))
        for idx in rng.choices(range(xs.size), weights=weights.tolist(), k=min(850, max(320, xs.size // 58))):
            x = int(xs[idx] + rng.randint(-4, 4))
            y = int(ys[idx] + rng.randint(-4, 4))
            if not (0 <= x < W and 0 <= y < H and mask[y, x]):
                continue
            edge = float(maps["edge_strength"][y, x])
            angle = v1.tangent_angle(maps, x, y, 74) + rng.uniform(-4, 4) if edge > 0.08 else rng.uniform(-14, 14)
            v1.draw_text(
                layer,
                x,
                y,
                rng.choice(words),
                rng.randint(5, 9),
                local_face_color(ref, maps, x, y, 0.88),
                int(np.clip(50 + edge * 82 + rng.randint(-6, 18), 46, 142)),
                angle,
                edge > 0.09,
            )
            drawn += 1
    clipped = v1.clip_layer(layer, mask)
    arr = v1.alpha_to_rgb(clipped)
    arr *= np.clip(0.26 + maps["enhanced_luma"] * 0.56 + maps["edge_strength"] * 0.30, 0.0, 0.94)[..., None]
    arr[maps["dark_zone"]] *= v3.dark_zone_soft_visibility(maps)[maps["dark_zone"], None]
    arr[~mask] = 0
    return np.clip(arr, 0, 255), mask, drawn, float(mask.sum() / max(1, int(maps["face"].sum())))


def cleanup_fragments(arr: np.ndarray, maps: dict[str, np.ndarray], useful_mask: np.ndarray) -> tuple[np.ndarray, int]:
    visible = (v1.luma(arr) > 10) & maps["face"]
    labels, count = ndi.label(visible, structure=np.ones((3, 3), dtype=bool))
    remove = np.zeros((H, W), dtype=bool)
    fragments = 0
    useful = ndi.binary_dilation(useful_mask | maps["edge_map"], structure=np.ones((7, 7), dtype=bool), iterations=1)
    for label in range(1, count + 1):
        region = labels == label
        area = int(region.sum())
        if area < 18 and not np.any(region & useful):
            remove |= region
            fragments += 1
    out = arr.copy()
    out[remove] = 0
    out[~maps["face"]] = 0
    return out, fragments


def final_face_correction(arr: np.ndarray, ref: np.ndarray, maps: dict[str, np.ndarray], lane_mask: np.ndarray) -> np.ndarray:
    out = arr.copy()
    ref_l = v1.luma(ref)
    ref_gray = np.repeat(ref_l[..., None], 3, axis=2)
    edge_midtone = maps["face"] & ~maps["dark_zone"] & (maps["edge_strength"] > 0.035)
    out[edge_midtone] = out[edge_midtone] * 0.65 + ref_gray[edge_midtone] * 0.35

    out[maps["dark_zone"]] *= 0.45
    dark_edge = maps["dark_zone"] & lane_mask & (maps["edge_strength"] > 0.13)
    if np.any(dark_edge):
        glimmer = ref_gray * np.clip(0.18 + maps["edge_strength"][..., None] * 0.80, 0.16, 0.44)
        out[dark_edge] = np.maximum(out[dark_edge], glimmer[dark_edge])

    out_l = v1.luma(out)
    rel_y = v2.rel_y_mask(maps["face"])
    lower = maps["face"] & (rel_y > 0.55) & (rel_y < 0.92)
    overbright = lower & (out_l > ref_l + 22) & (out_l > 54)
    if np.any(overbright):
        ratio = (ref_l[overbright] + 8.0) / np.maximum(out_l[overbright], 1.0)
        out[overbright] *= np.clip(ratio, 0.22, 0.90)[:, None]

    interior = maps["dark_zone"] & ~lane_mask
    leak = interior & (v1.luma(out) > 48)
    out[leak] *= 0.60
    out[~maps["face"]] = 0
    return np.clip(out, 0, 255)


def post_cleanup_face_balance(arr: np.ndarray, ref: np.ndarray, maps: dict[str, np.ndarray]) -> np.ndarray:
    out = arr.copy()
    ref_l = v1.luma(ref)
    ref_gray = np.repeat(ref_l[..., None], 3, axis=2)
    safe_edges = maps["face"] & ~maps["dark_zone"] & (maps["edge_strength"] > 0.035)
    out[safe_edges] = out[safe_edges] * 0.50 + ref_gray[safe_edges] * 0.50

    rel_y = v2.rel_y_mask(maps["face"])
    lower = maps["face"] & (rel_y > 0.55) & (rel_y < 0.92)
    out_l = v1.luma(out)
    overbright = lower & (out_l > ref_l + 22) & (out_l > 54)
    if np.any(overbright):
        ratio = (ref_l[overbright] + 8.0) / np.maximum(out_l[overbright], 1.0)
        out[overbright] *= np.clip(ratio, 0.22, 0.90)[:, None]
    out[~maps["face"]] = 0
    return np.clip(out, 0, 255)


def crop(img: Image.Image | np.ndarray, box: tuple[int, int, int, int]) -> Image.Image:
    image = Image.fromarray(img.astype(np.uint8), "RGB") if isinstance(img, np.ndarray) else img
    return image.crop(box)


def side_by_side(images: list[tuple[str, Image.Image]]) -> Image.Image:
    widths = [im.width for _, im in images]
    heights = [im.height for _, im in images]
    out = Image.new("RGB", (sum(widths), max(heights)), (0, 0, 0))
    x = 0
    for label, im in images:
        out.paste(im.convert("RGB"), (x, 0))
        ImageDraw.Draw(out).text((x + 12, 12), label, font=v1.font(20, True), fill=(245, 245, 245))
        x += im.width
    return out


def metrics(ref: np.ndarray, rec: np.ndarray, maps: dict[str, np.ndarray], stats: dict[str, int | float]) -> dict[str, float | int]:
    base = v2.compute_metrics(ref, rec, maps)
    out: dict[str, float | int] = {
        "face_luma_mae": float(base["face_luma_mae"]),
        "edge_overlap_face": float(base["edge_overlap_face"]),
        "protected_dark_zone_fill_ratio": float(base["protected_dark_zone_fill_ratio"]),
        "gray_slab_penalty": float(base["gray_slab_penalty"]),
        "mouth_banding_penalty": float(base["mouth_banding_penalty"]),
    }
    out.update(stats)
    return out


def main() -> None:
    start = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    ref = v1.load_reference()
    maps = v1.build_target_maps(ref)
    maps["dark_zone"] = v3.refine_protected_dark_zone_v3(maps)
    visibility = v2.build_visibility_mask(ref, maps)
    box = face_crop_box(maps["face"])

    v4_path = OUT / "stencil_v4_final.png"
    v4_img = Image.open(v4_path).convert("RGB") if v4_path.exists() else Image.fromarray(ref, "RGB")
    ref_crop = crop(ref, box)
    v4_crop = v4_img.crop(box)
    ref_crop.save(OUT / "stencil_v5_face_reference_crop.png")
    v4_crop.save(OUT / "stencil_v5_face_v4_crop.png")

    face_raw = v1.combine_layers(
        [
            v1.draw_micro_texture(maps["face"], maps, rng),
            v1.draw_weighted_texture(maps["face"], maps, v1.FACE_WORDS, rng, count=3150, size_range=(8, 19), alpha_range=(70, 170)),
            v1.draw_contour_texture(maps["face"], maps, rng),
        ]
    )
    base_face = v2.apply_face_stencil(face_raw, ref, maps, visibility)
    detail, _detail_layer, detail_stats, edge_ring = v3.render_face_detail_recovery(ref, maps, visibility, rng)
    face = v3.face_after_recovery(base_face, detail, maps, visibility)
    anchor_arr, anchor_overlay, anchor_words = render_tiered_anchors(ref, maps, visibility, box)
    lane_arr, lane_overlay, lane_words, lane_mask = render_face_lanes(ref, maps, visibility, box, rng)
    current = np.clip(face + anchor_arr + lane_arr, 0, 255)
    too_dark_arr, too_dark_mask, too_dark_words, too_dark_ratio = render_too_dark_face_microtext(ref, current, maps, lane_mask | edge_ring, rng)
    face = np.clip(current + too_dark_arr, 0, 255)
    face = final_face_correction(face, ref, maps, lane_mask | edge_ring)
    face, floating_fragments = cleanup_fragments(face, maps, lane_mask | edge_ring | too_dark_mask)
    face = post_cleanup_face_balance(face, ref, maps)

    full_rec = np.zeros((H, W, 3), dtype=np.uint8)
    full_rec[maps["face"]] = face[maps["face"]].astype(np.uint8)
    final_crop = crop(full_rec, box)
    final_crop.save(OUT / "stencil_v5_face_only.png")
    side_by_side(
        [
            ("reference", ref_crop),
            ("v4 face crop", v4_crop),
            ("v5 face only", final_crop),
        ]
    ).save(OUT / "stencil_v5_face_side_by_side.png")
    crop(anchor_overlay.convert("RGB"), box).save(OUT / "stencil_v5_face_anchor_overlay.png")
    crop(lane_overlay.convert("RGB"), box).save(OUT / "stencil_v5_face_lane_overlay.png")
    Image.fromarray((too_dark_mask.astype(np.uint8) * 255), "L").crop(box).save(
        OUT / "stencil_v5_face_too_dark_recovery_mask.png"
    )

    stats: dict[str, int | float] = {
        "manual_anchor_words": anchor_words,
        "contour_lane_words": lane_words,
        "too_dark_recovery_words": too_dark_words,
        "floating_fragment_count": floating_fragments,
        "face_too_dark_pixel_ratio": too_dark_ratio,
        "face_detail_words": int(detail_stats["face_detail_words"]),
        "edge_ring_words": int(detail_stats["edge_ring_words"]),
        "render_time_seconds": round(perf_counter() - start, 3),
        "seed": SEED,
        "crop_box": list(box),  # type: ignore[dict-item]
    }
    m = metrics(ref, full_rec, maps, stats)
    (OUT / "stencil_v5_face_metrics.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
