from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch import recreate_jordan_reference_stencil as v1
from scratch import recreate_jordan_reference_stencil_v2 as v2
from scratch import recreate_jordan_reference_stencil_v3 as v3

OUT = ROOT / "examples" / "reference_recreation"
W, H = v1.W, v1.H
BASE_SEED = v3.SEED
V4_SEED = 20260517

FACE_ANCHORS_V4 = [
    {
        "text": "NBA Rookie of the Year",
        "mode": "path",
        "points": [(0.690, 0.168), (0.758, 0.135), (0.842, 0.168)],
        "size": 21,
        "alpha": 205,
    },
    {"text": "MVP", "mode": "pos", "pos": (0.733, 0.255), "size": 36, "alpha": 210, "angle": -6.0},
    {"text": "Air Jordan", "mode": "pos", "pos": (0.700, 0.340), "size": 26, "alpha": 205, "angle": 4.0},
    {
        "text": "Dedication",
        "mode": "path",
        "points": [(0.700, 0.405), (0.731, 0.475), (0.770, 0.545)],
        "size": 25,
        "alpha": 180,
    },
    {"text": "Love of the Game", "mode": "pos", "pos": (0.716, 0.472), "size": 21, "alpha": 160, "angle": 12.0},
    {
        "text": "Dominance",
        "mode": "path",
        "points": [(0.692, 0.505), (0.704, 0.604), (0.724, 0.704)],
        "size": 27,
        "alpha": 204,
    },
    {"text": "Scoring", "mode": "pos", "pos": (0.812, 0.496), "size": 25, "alpha": 185, "angle": -8.0},
    {"text": "Finals MVP", "mode": "pos", "pos": (0.742, 0.592), "size": 23, "alpha": 165, "angle": 43.0},
]

CONTOUR_LANES_V4 = [
    {"id": "forehead_arc", "points": [(0.680, 0.185), (0.735, 0.126), (0.840, 0.170), (0.866, 0.240)], "words": ["Rookie of the Year", "MVP", "Champion"], "spacing": 34, "size": (8, 13), "alpha": (118, 188), "mask": "face"},
    {"id": "brow_ridge", "points": [(0.685, 0.304), (0.735, 0.296), (0.805, 0.318), (0.860, 0.342)], "words": ["MVP", "Defense", "Focus"], "spacing": 30, "size": (8, 14), "alpha": (116, 194), "mask": "face"},
    {"id": "upper_eye_socket", "points": [(0.690, 0.342), (0.742, 0.336), (0.815, 0.354)], "words": ["Air Jordan", "Finals MVP", "Dedication"], "spacing": 29, "size": (7, 13), "alpha": (112, 188), "mask": "face"},
    {"id": "lower_eye_socket", "points": [(0.700, 0.382), (0.752, 0.404), (0.830, 0.400)], "words": ["Clutch", "Scoring", "Champion"], "spacing": 28, "size": (7, 13), "alpha": (110, 180), "mask": "face"},
    {"id": "nose_bridge", "points": [(0.792, 0.328), (0.807, 0.415), (0.800, 0.535)], "words": ["Scoring", "Finals MVP", "MVP"], "spacing": 27, "size": (7, 13), "alpha": (126, 208), "mask": "face"},
    {"id": "nose_side_highlight", "points": [(0.832, 0.382), (0.852, 0.460), (0.870, 0.548)], "words": ["Scoring", "Air", "Jordan"], "spacing": 27, "size": (7, 13), "alpha": (120, 200), "mask": "face"},
    {"id": "left_cheek_plane", "points": [(0.695, 0.420), (0.735, 0.470), (0.790, 0.515)], "words": ["Dedication", "Love of the Game", "Champion"], "spacing": 32, "size": (8, 14), "alpha": (102, 172), "mask": "face"},
    {"id": "right_cheek_plane", "points": [(0.805, 0.430), (0.852, 0.470), (0.878, 0.542)], "words": ["Scoring", "Finals MVP", "Clutch"], "spacing": 30, "size": (8, 14), "alpha": (112, 184), "mask": "face"},
    {"id": "mouth_crease", "points": [(0.744, 0.548), (0.805, 0.556), (0.858, 0.574)], "words": ["Clutch", "Drive", "Focus"], "spacing": 25, "size": (7, 12), "alpha": (112, 178), "mask": "face"},
    {"id": "chin_curve", "points": [(0.737, 0.600), (0.790, 0.655), (0.850, 0.628)], "words": ["Finals MVP", "Dominance", "Champion"], "spacing": 30, "size": (8, 14), "alpha": (100, 168), "mask": "face"},
    {"id": "jawline", "points": [(0.674, 0.410), (0.688, 0.520), (0.715, 0.660), (0.760, 0.730)], "words": ["Dominance", "Defense", "Champion"], "spacing": 32, "size": (8, 15), "alpha": (118, 202), "mask": "face"},
    {"id": "neck_column", "points": [(0.692, 0.570), (0.707, 0.680), (0.732, 0.790)], "words": ["Dominance", "Dedication", "Air Jordan"], "spacing": 31, "size": (8, 15), "alpha": (112, 190), "mask": "face"},
    {"id": "neck_shadow_edge", "points": [(0.754, 0.610), (0.775, 0.705), (0.810, 0.780)], "words": ["Defense", "Focus", "MVP"], "spacing": 30, "size": (7, 13), "alpha": (96, 168), "mask": "face"},
    {"id": "collar_curve", "points": [(0.664, 0.720), (0.725, 0.775), (0.845, 0.760), (0.927, 0.706)], "words": ["Chicago Bulls", "Finals MVP", "Six Rings"], "spacing": 34, "size": (8, 14), "alpha": (88, 150), "mask": "subject_non_jersey"},
    {"id": "jersey_trim_curve", "points": [(0.635, 0.681), (0.720, 0.715), (0.840, 0.700), (0.935, 0.648)], "words": ["BULLS", "23", "Chicago"], "spacing": 36, "size": (8, 14), "alpha": (64, 112), "mask": "trim"},
]


def norm_xy(point: tuple[float, float]) -> tuple[float, float]:
    return point[0] * W, point[1] * H


def sample_path(points: list[tuple[float, float]], spacing: float, offset: float = 0.0) -> list[tuple[float, float, float]]:
    pts = [norm_xy(p) for p in points]
    if len(pts) < 2:
        return []
    segments: list[tuple[float, float, float, float, float]] = []
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
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
                angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
                samples.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, angle))
                break
            cursor -= length
        d += spacing
    return samples


def local_gray_color(ref: np.ndarray, maps: dict[str, np.ndarray], x: int, y: int, lift: float = 1.0) -> tuple[int, int, int]:
    ref_l = float(v1.luma(ref[y : y + 1, x : x + 1])[0, 0])
    edge = float(maps["edge_strength"][y, x])
    value = int(np.clip(ref_l * lift + maps["enhanced_luma"][y, x] * 72.0 + edge * 74.0, 58, 244))
    return value, value, min(255, value + 5)


def nearest_allowed(x: int, y: int, allowed: np.ndarray, radius: int = 10) -> tuple[int, int] | None:
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


def render_manual_face_anchors(
    ref: np.ndarray,
    maps: dict[str, np.ndarray],
    visibility: np.ndarray,
) -> tuple[np.ndarray, Image.Image, int]:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    allowed = maps["face"] & ~ndi.binary_erosion(maps["dark_zone"], structure=np.ones((5, 5), dtype=bool), iterations=1)
    count = 0
    for anchor in FACE_ANCHORS_V4:
        if anchor["mode"] == "pos":
            x, y = norm_xy(anchor["pos"])  # type: ignore[arg-type]
            point = nearest_allowed(int(x), int(y), allowed, 14)
            if point is None:
                continue
            px, py = point
            v1.draw_text(
                layer,
                px,
                py,
                str(anchor["text"]),
                int(anchor["size"]),
                local_gray_color(ref, maps, px, py, 1.10),
                int(anchor["alpha"]),
                float(anchor.get("angle", 0.0)),
                True,
            )
            count += 1
        else:
            samples = sample_path(anchor["points"], 36.0, 0.0)  # type: ignore[arg-type]
            if not samples:
                continue
            mx, my, angle = samples[len(samples) // 2]
            point = nearest_allowed(int(mx), int(my), allowed, 28)
            if point is None:
                continue
            px, py = point
            v1.draw_text(
                layer,
                px,
                py,
                str(anchor["text"]),
                int(anchor["size"]),
                local_gray_color(ref, maps, px, py, 1.08),
                int(anchor["alpha"]),
                angle,
                True,
            )
            count += 1
    clipped = v1.clip_layer(layer, maps["face"])
    arr = v1.alpha_to_rgb(clipped)
    soft = v3.dark_zone_soft_visibility(maps)
    arr *= np.clip(0.22 + visibility * 0.72 + maps["edge_strength"] * 0.36, 0.0, 1.10)[..., None]
    arr[maps["dark_zone"]] *= soft[maps["dark_zone"], None]
    arr[~maps["face"]] = 0
    return np.clip(arr, 0, 255), clipped, count


def lane_allowed_mask(lane: dict[str, object], maps: dict[str, np.ndarray], edge_ring: np.ndarray) -> np.ndarray:
    mask_name = str(lane.get("mask", "face"))
    if mask_name == "subject_non_jersey":
        base = maps["subject"] & ~maps["jersey"]
    elif mask_name == "trim":
        trim = ndi.binary_dilation(maps["jersey"], structure=np.ones((7, 7), dtype=bool), iterations=1) & ~ndi.binary_erosion(
            maps["jersey"], structure=np.ones((7, 7), dtype=bool), iterations=1
        )
        base = trim & maps["subject"]
    else:
        base = maps["face"]
    protected_interior = ndi.binary_erosion(maps["dark_zone"], structure=np.ones((5, 5), dtype=bool), iterations=1)
    return base & (~protected_interior | edge_ring)


def render_contour_lanes(
    ref: np.ndarray,
    maps: dict[str, np.ndarray],
    visibility: np.ndarray,
    edge_ring: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, Image.Image, int, np.ndarray]:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    used = np.zeros((H, W), dtype=bool)
    count = 0
    for lane in CONTOUR_LANES_V4:
        allowed = lane_allowed_mask(lane, maps, edge_ring)
        path_trace = np.zeros((H, W), dtype=bool)
        trace_samples = sample_path(lane["points"], 8.0, 0.0)  # type: ignore[arg-type]
        for tx, ty, _ in trace_samples:
            ix, iy = int(tx), int(ty)
            if 0 <= ix < W and 0 <= iy < H:
                path_trace[iy, ix] = True
        corridor = ndi.binary_dilation(path_trace, structure=np.ones((23, 23), dtype=bool), iterations=1)
        for pass_i in range(2):
            offset = rng.uniform(0.0, float(lane["spacing"]) * 0.40) + pass_i * float(lane["spacing"]) * 0.48
            for x, y, angle in sample_path(lane["points"], float(lane["spacing"]) * 0.72, offset):  # type: ignore[arg-type]
                point = nearest_allowed(int(x), int(y), allowed, 18)
                if point is None:
                    continue
                px, py = point
                words = lane["words"]  # type: ignore[assignment]
                size_min, size_max = lane["size"]  # type: ignore[misc]
                alpha_min, alpha_max = lane["alpha"]  # type: ignore[misc]
                alpha = int(np.clip(rng.randint(int(alpha_min), int(alpha_max)) + maps["edge_strength"][py, px] * 48.0, 58, 230))
                lift = 1.16 if str(lane["mask"]) == "face" else 0.96
                v1.draw_text(
                    layer,
                    px + rng.randint(-3, 3),
                    py + rng.randint(-3, 3),
                    rng.choice(words),  # type: ignore[arg-type]
                    rng.randint(int(size_min), int(size_max)),
                    local_gray_color(ref, maps, px, py, lift),
                    alpha,
                    angle + rng.uniform(-4.0, 4.0),
                    True,
                )
                used[py, px] = True
                count += 1

        candidates = np.argwhere(corridor & allowed & maps["edge_map"] & (maps["tone"] > 0.055))
        if candidates.size:
            rng.shuffle(candidates)
            extra_limit = min(18 if str(lane["mask"]) == "face" else 8, candidates.shape[0])
            for y, x in candidates[:extra_limit]:
                px, py = int(x + rng.randint(-4, 4)), int(y + rng.randint(-4, 4))
                if not (0 <= px < W and 0 <= py < H and allowed[py, px]):
                    continue
                words = lane["words"]  # type: ignore[assignment]
                size_min, size_max = lane["size"]  # type: ignore[misc]
                alpha_min, alpha_max = lane["alpha"]  # type: ignore[misc]
                edge = float(maps["edge_strength"][py, px])
                alpha = int(np.clip(int(alpha_max) + edge * 64.0 + rng.randint(-8, 14), 116, 235))
                v1.draw_text(
                    layer,
                    px,
                    py,
                    rng.choice(words),  # type: ignore[arg-type]
                    rng.randint(int(size_min), int(size_max)),
                    local_gray_color(ref, maps, px, py, 1.22),
                    alpha,
                    v1.tangent_angle(maps, px, py, 76.0) + rng.uniform(-4.0, 4.0),
                    True,
                )
                used[py, px] = True
                count += 1
    mask = maps["subject"] & ~ndi.binary_erosion(maps["dark_zone"], structure=np.ones((5, 5), dtype=bool), iterations=1)
    clipped = v1.clip_layer(layer, mask)
    arr = v1.alpha_to_rgb(clipped)
    soft = v3.dark_zone_soft_visibility(maps)
    arr *= np.clip(0.26 + visibility * 0.82 + maps["edge_strength"] * 0.72, 0.0, 1.36)[..., None]
    arr[maps["dark_zone"]] *= soft[maps["dark_zone"], None]
    arr[~maps["subject"]] = 0
    return np.clip(arr, 0, 255), clipped, count, ndi.binary_dilation(used, structure=np.ones((5, 5), dtype=bool), iterations=1)


def render_too_dark_recovery(
    ref: np.ndarray,
    current: np.ndarray,
    maps: dict[str, np.ndarray],
    edge_ring: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    ref_l = v1.luma(ref)
    out_l = v1.luma(current)
    regions = v3.face_recovery_regions(maps)
    target_regions = regions["cheek"] | regions["jaw"] | regions["neck"] | regions["mouth_chin"] | edge_ring
    too_dark = (ref_l - out_l > 18.0) & target_regions & ~ndi.binary_erosion(
        maps["dark_zone"], structure=np.ones((5, 5), dtype=bool), iterations=1
    )
    too_dark &= maps["tone"] > 0.075
    too_dark = ndi.binary_opening(too_dark, structure=np.ones((2, 2), dtype=bool), iterations=1)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    words = ["Air Jordan", "Dedication", "Dominance", "Scoring", "Champion", "Finals MVP", "MVP"]
    ys, xs = np.where(too_dark)
    drawn = 0
    if xs.size:
        weights = 0.30 + maps["edge_strength"][ys, xs] * 0.90 + np.clip((ref_l[ys, xs] - out_l[ys, xs]) / 80.0, 0.0, 1.0)
        weights = weights / max(1e-6, float(weights.sum()))
        for idx in rng.choices(range(xs.size), weights=weights.tolist(), k=min(720, max(260, xs.size // 70))):
            x = int(xs[idx] + rng.randint(-5, 5))
            y = int(ys[idx] + rng.randint(-4, 4))
            if not (0 <= x < W and 0 <= y < H and too_dark[y, x]):
                continue
            edge = float(maps["edge_strength"][y, x])
            alpha = int(np.clip(54 + edge * 76 + rng.randint(-6, 18), 48, 138))
            angle = v1.tangent_angle(maps, x, y, 70.0) + rng.uniform(-5, 5) if edge > 0.08 else rng.uniform(-15, 15)
            v1.draw_text(
                layer,
                x,
                y,
                rng.choice(words),
                rng.randint(5, 9),
                local_gray_color(ref, maps, x, y, 0.86),
                alpha,
                angle,
                edge > 0.09,
            )
            drawn += 1
    clipped = v1.clip_layer(layer, too_dark)
    arr = v1.alpha_to_rgb(clipped)
    arr *= np.clip(0.28 + maps["enhanced_luma"] * 0.54 + maps["edge_strength"] * 0.26, 0.0, 0.92)[..., None]
    arr[maps["dark_zone"]] *= v3.dark_zone_soft_visibility(maps)[maps["dark_zone"], None]
    arr[~too_dark] = 0
    ratio = float(too_dark.sum() / max(1, int(maps["face"].sum())))
    return np.clip(arr, 0, 255), too_dark, drawn, ratio


def final_corrections_v4(arr: np.ndarray, ref: np.ndarray, maps: dict[str, np.ndarray], edge_ring: np.ndarray) -> np.ndarray:
    out = v3.final_corrections_v3(arr, ref, maps, edge_ring)
    ref_l = v1.luma(ref)
    ref_gray = np.repeat(ref_l[..., None], 3, axis=2)
    edge_midtone = maps["face"] & ~maps["dark_zone"] & (maps["edge_strength"] > 0.06)
    out[edge_midtone] = out[edge_midtone] * 0.63 + ref_gray[edge_midtone] * 0.37
    out_l = v1.luma(out)
    interior = maps["dark_zone"] & ~edge_ring
    leak = interior & (out_l > 48.0)
    out[leak] *= 0.62
    edge_leak = maps["dark_zone"] & edge_ring & (out_l > 66.0)
    out[edge_leak] *= 0.78

    rel_y = v2.rel_y_mask(maps["face"])
    lower = maps["face"] & (rel_y > 0.55) & (rel_y < 0.92)
    overbright = lower & (v1.luma(out) > ref_l + 23.0) & (v1.luma(out) > 54.0)
    if np.any(overbright):
        ratio = (ref_l[overbright] + 8.0) / np.maximum(v1.luma(out)[overbright], 1.0)
        out[overbright] *= np.clip(ratio, 0.22, 0.90)[:, None]
    out[~maps["subject"]] = 0
    return np.clip(out, 0, 255)


def compute_metrics(
    ref: np.ndarray,
    rec: np.ndarray,
    maps: dict[str, np.ndarray],
    stats: dict[str, int | float],
) -> dict[str, float | int | list[int]]:
    out = v2.compute_metrics(ref, rec, maps)
    out.update(stats)
    return out


def main() -> None:
    t0 = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    base_rng = random.Random(BASE_SEED)
    v4_rng = random.Random(V4_SEED)
    ref = v1.load_reference()
    maps = v1.build_target_maps(ref)
    maps["dark_zone"] = v3.refine_protected_dark_zone_v3(maps)
    visibility = v2.build_visibility_mask(ref, maps)

    face_raw = v1.combine_layers(
        [
            v1.draw_micro_texture(maps["face"], maps, base_rng),
            v1.draw_weighted_texture(maps["face"], maps, v1.FACE_WORDS, base_rng, count=2850, size_range=(9, 20), alpha_range=(72, 176)),
            v1.draw_contour_texture(maps["face"], maps, base_rng),
        ]
    )
    jersey_raw = v1.combine_layers(
        [
            v1.draw_micro_texture(maps["jersey"], maps, base_rng, jersey=True),
            v1.draw_weighted_texture(
                maps["jersey"], maps, v1.JERSEY_WORDS, base_rng, jersey=True, count=2050, size_range=(9, 23), alpha_range=(62, 152)
            ),
            v1.draw_contour_texture(maps["jersey"], maps, base_rng, jersey=True),
        ]
    )
    shoulder_raw = v1.draw_weighted_texture(maps["shoulder"], maps, v1.MICRO_WORDS, base_rng, count=620, size_range=(8, 18), alpha_range=(42, 118))

    base_face = v2.apply_face_stencil(face_raw, ref, maps, visibility)
    detail, detail_layer, v3_stats, edge_ring = v3.render_face_detail_recovery(ref, maps, visibility, base_rng)
    recovered_face = v3.face_after_recovery(base_face, detail, maps, visibility)
    jersey_after = v2.apply_jersey_stencil(jersey_raw, ref, maps, visibility)
    body_after = v2.apply_body_stencil(shoulder_raw, maps, visibility)
    jersey_anchors = v2.render_integrated_jersey_anchors(ref, maps, visibility)
    face_anchors = v2.render_face_anchors(maps, visibility)

    manual_anchor_arr, anchor_overlay, manual_anchor_words = render_manual_face_anchors(ref, maps, visibility)
    lane_arr, lane_overlay, contour_lane_words, lane_mask = render_contour_lanes(ref, maps, visibility, edge_ring, v4_rng)
    combined_edge_ring = edge_ring | lane_mask

    face_before_too_dark = np.clip(recovered_face + face_anchors + manual_anchor_arr + lane_arr, 0, 255)
    temp_arr = np.clip(body_after + face_before_too_dark + jersey_after + jersey_anchors, 0, 255)
    too_dark_arr, too_dark_mask, too_dark_words, too_dark_ratio = render_too_dark_recovery(
        ref, temp_arr, maps, combined_edge_ring, v4_rng
    )
    face_after = np.clip(face_before_too_dark + too_dark_arr, 0, 255)
    final_arr = np.clip(body_after + face_after + jersey_after + jersey_anchors, 0, 255)
    final_arr = final_corrections_v4(final_arr, ref, maps, combined_edge_ring)

    final_img = Image.fromarray(final_arr.astype(np.uint8), "RGB").convert("RGBA")
    v1.draw_text(final_img, 795, 514, "change the game.", 28, (188, 188, 202), 220)
    final_rgb = final_img.convert("RGB")
    rec = np.array(final_rgb, dtype=np.uint8)

    stats: dict[str, int | float] = dict(v3_stats)
    stats.update(
        {
            "manual_anchor_words": manual_anchor_words,
            "contour_lane_words": contour_lane_words,
            "too_dark_recovery_words": too_dark_words,
            "face_too_dark_pixel_ratio": too_dark_ratio,
            "base_seed": BASE_SEED,
            "v4_seed": V4_SEED,
        }
    )
    metrics = compute_metrics(ref, rec, maps, stats)
    metrics["seed"] = V4_SEED
    metrics["render_time_seconds"] = round(perf_counter() - t0, 3)

    final_rgb.save(OUT / "stencil_v4_final.png")
    v1.side_by_side(ref, final_rgb).save(OUT / "stencil_v4_side_by_side.png")
    anchor_overlay.convert("RGB").save(OUT / "stencil_v4_anchor_overlay.png")
    lane_overlay.convert("RGB").save(OUT / "stencil_v4_lane_overlay.png")
    Image.fromarray((too_dark_mask.astype(np.uint8) * 255), "L").save(OUT / "stencil_v4_too_dark_recovery_mask.png")
    Image.fromarray(face_after.astype(np.uint8), "RGB").save(OUT / "stencil_v4_face_after_recovery.png")
    (OUT / "stencil_v4_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
