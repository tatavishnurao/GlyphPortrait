from __future__ import annotations

import json
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

OUT = ROOT / "examples" / "reference_recreation"
W, H = v1.W, v1.H
SEED = 20260516


def refine_protected_dark_zone_v3(maps: dict[str, np.ndarray]) -> np.ndarray:
    base = v2.refine_protected_dark_zone(maps)
    face = maps["face"]
    tone = maps["tone"]
    rel_y = v2.rel_y_mask(face)
    interior = ndi.binary_erosion(base, structure=np.ones((5, 5), dtype=bool), iterations=1)
    deepest = base & (tone < 0.13)
    eye_core = base & (rel_y > 0.30) & (rel_y < 0.49) & (tone < 0.20)
    mouth_core = base & (rel_y > 0.62) & (rel_y < 0.75) & (tone < 0.18)
    protected = interior | deepest | eye_core | mouth_core
    protected = ndi.binary_dilation(protected, structure=np.ones((2, 2), dtype=bool), iterations=1)
    return protected & face


def dark_zone_soft_visibility(maps: dict[str, np.ndarray]) -> np.ndarray:
    dark = maps["dark_zone"]
    edge = maps["edge_strength"]
    interior = ndi.binary_erosion(dark, structure=np.ones((5, 5), dtype=bool), iterations=1)
    boundary = dark & ~interior
    visibility = np.ones((H, W), dtype=np.float32)
    visibility[dark] = np.clip(0.20 + edge[dark] * 0.34, 0.18, 0.32)
    visibility[boundary] = np.clip(0.50 + edge[boundary] * 1.45, 0.50, 0.84)
    return visibility


def build_edge_ring_mask(maps: dict[str, np.ndarray]) -> np.ndarray:
    face = maps["face"]
    dark = maps["dark_zone"]
    edges = ndi.binary_dilation(maps["edge_map"] & face, structure=np.ones((3, 3), dtype=bool), iterations=1)
    dark_boundary = ndi.binary_dilation(dark, structure=np.ones((5, 5), dtype=bool), iterations=1) & ~ndi.binary_erosion(
        dark, structure=np.ones((5, 5), dtype=bool), iterations=1
    )
    edge_ring = (edges | dark_boundary) & face
    edge_ring &= ~ndi.binary_erosion(dark, structure=np.ones((7, 7), dtype=bool), iterations=1)
    edge_ring &= maps["tone"] > 0.055
    return edge_ring


def face_recovery_regions(maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    face = maps["face"]
    rel_y = v2.rel_y_mask(face)
    x0, _, x1, _ = v1.bbox(face)
    xx = np.broadcast_to(np.arange(W), face.shape)
    rel_x = (xx - x0) / max(1, x1 - x0)
    tone = maps["tone"]
    safe = face & ~ndi.binary_erosion(maps["dark_zone"], structure=np.ones((5, 5), dtype=bool), iterations=1)
    cheek = safe & (rel_y > 0.34) & (rel_y < 0.66) & (rel_x > 0.40) & (rel_x < 0.95) & (tone > 0.09)
    jaw = safe & (rel_y > 0.56) & (rel_y < 0.82) & (rel_x > 0.22) & (rel_x < 0.82) & (tone > 0.08)
    neck = safe & (rel_y > 0.72) & (rel_y < 1.03) & (rel_x > 0.18) & (rel_x < 0.74) & (tone > 0.075)
    mouth_chin = safe & (rel_y > 0.56) & (rel_y < 0.76) & (rel_x > 0.46) & (rel_x < 0.92) & (maps["edge_strength"] > 0.08)
    return {"cheek": cheek, "jaw": jaw, "neck": neck, "mouth_chin": mouth_chin}


def draw_recovery_words(
    layer: Image.Image,
    ref: np.ndarray,
    maps: dict[str, np.ndarray],
    mask: np.ndarray,
    rng: random.Random,
    count: int,
    *,
    alpha_range: tuple[int, int],
    size_range: tuple[int, int],
    edge_biased: bool,
) -> int:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0
    weights = 0.25 + maps["edge_strength"][ys, xs] * (1.35 if edge_biased else 0.48) + maps["enhanced_luma"][ys, xs] * 0.34
    weights = weights / max(1e-6, float(weights.sum()))
    drawn = 0
    for idx in rng.choices(range(xs.size), weights=weights.tolist(), k=count):
        x = int(xs[idx] + rng.randint(-7, 7))
        y = int(ys[idx] + rng.randint(-6, 6))
        if not (0 <= x < W and 0 <= y < H and mask[y, x]):
            continue
        ref_l = float(v1.luma(ref[y : y + 1, x : x + 1])[0, 0])
        edge = float(maps["edge_strength"][y, x])
        v = int(np.clip(ref_l * 1.20 + maps["enhanced_luma"][y, x] * 78.0 + edge * 62.0, 54, 238))
        color = (v, v, min(255, v + 5))
        alpha = int(np.clip(rng.randint(*alpha_range) + edge * 42, 70, 230))
        angle = v1.tangent_angle(maps, x, y, 72.0) + rng.uniform(-5, 5) if edge_biased else rng.uniform(-16, 16)
        v1.draw_text(
            layer,
            x,
            y,
            rng.choice(v1.FACE_WORDS),
            rng.randint(*size_range),
            color,
            alpha,
            angle,
            edge_biased or rng.random() < 0.25,
        )
        drawn += 1
    return drawn


def render_face_detail_recovery(
    ref: np.ndarray,
    maps: dict[str, np.ndarray],
    visibility: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, Image.Image, dict[str, int], np.ndarray]:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    edge_ring = build_edge_ring_mask(maps)
    regions = face_recovery_regions(maps)
    stats = {
        "edge_ring_words": draw_recovery_words(
            layer,
            ref,
            maps,
            edge_ring,
            rng,
            980,
            alpha_range=(120, 204),
            size_range=(7, 15),
            edge_biased=True,
        )
    }
    stats["cheek_recovery_words"] = draw_recovery_words(
        layer,
        ref,
        maps,
        regions["cheek"] & ~edge_ring,
        rng,
        390,
        alpha_range=(82, 146),
        size_range=(7, 13),
        edge_biased=False,
    )
    stats["neck_recovery_words"] = draw_recovery_words(
        layer,
        ref,
        maps,
        (regions["jaw"] | regions["neck"] | regions["mouth_chin"]) & ~edge_ring,
        rng,
        440,
        alpha_range=(78, 152),
        size_range=(7, 14),
        edge_biased=False,
    )
    stats["face_detail_words"] = stats["edge_ring_words"] + stats["cheek_recovery_words"] + stats["neck_recovery_words"]

    clipped = v1.clip_layer(layer, maps["face"])
    arr = v1.alpha_to_rgb(clipped)
    soft = dark_zone_soft_visibility(maps)
    arr *= np.clip(0.34 + visibility * 0.84 + maps["edge_strength"] * 0.40, 0.0, 1.20)[..., None]
    arr[maps["dark_zone"]] *= soft[maps["dark_zone"], None]
    arr[~maps["face"]] = 0
    return np.clip(arr, 0, 255), clipped, stats, edge_ring


def face_after_recovery(base_face: np.ndarray, detail: np.ndarray, maps: dict[str, np.ndarray], visibility: np.ndarray) -> np.ndarray:
    out = base_face.copy()
    regions = face_recovery_regions(maps)
    boost = np.zeros((H, W), dtype=np.float32)
    boost[regions["cheek"]] = 0.20
    boost[regions["jaw"]] = np.maximum(boost[regions["jaw"]], 0.16)
    boost[regions["neck"]] = np.maximum(boost[regions["neck"]], 0.18)
    boost[regions["mouth_chin"]] = np.maximum(boost[regions["mouth_chin"]], 0.12)
    safe = maps["face"] & ~maps["dark_zone"]
    out[safe] *= (1.0 + boost[safe])[:, None]
    out += detail
    soft = dark_zone_soft_visibility(maps)
    out[maps["dark_zone"]] *= soft[maps["dark_zone"], None]
    out[~maps["face"]] = 0
    return np.clip(out, 0, 255)


def final_corrections_v3(arr: np.ndarray, ref: np.ndarray, maps: dict[str, np.ndarray], edge_ring: np.ndarray) -> np.ndarray:
    out = arr.copy()
    face = maps["face"]
    dark = maps["dark_zone"]
    ref_l = v1.luma(ref)
    ref_gray = np.repeat(ref_l[..., None], 3, axis=2)
    face_recover = face & ~dark & (maps["tone"] > 0.065)
    out[face_recover] = out[face_recover] * 0.55 + ref_gray[face_recover] * 0.45

    dark_edge = dark & edge_ring & (maps["edge_strength"] > 0.12)
    if np.any(dark_edge):
        glimmer = ref_gray * np.clip(0.20 + maps["edge_strength"][..., None] * 1.00, 0.18, 0.48)
        out[dark_edge] = np.maximum(out[dark_edge], glimmer[dark_edge])

    ref_red = v1.red_mask(ref, maps["subject"])
    red_recover = ref_red & maps["jersey"] & (maps["red_strength"] > 0.12) & (maps["tone"] > 0.075)
    if np.any(red_recover):
        recovered_r = np.maximum(out[..., 0][red_recover], ref[..., 0][red_recover].astype(np.float32) * 0.72 + 20.0)
        channel_cap = np.maximum(0.0, recovered_r - 24.0)
        out[..., 0][red_recover] = recovered_r
        out[..., 1][red_recover] = np.minimum(out[..., 1][red_recover], channel_cap)
        out[..., 2][red_recover] = np.minimum(out[..., 2][red_recover], channel_cap)

    out_l = v1.luma(out)
    rel_y = v2.rel_y_mask(face)
    lower = face & (rel_y > 0.55) & (rel_y < 0.92)
    overbright = lower & (out_l > ref_l + 23.0) & (out_l > 54.0)
    if np.any(overbright):
        ratio = (ref_l[overbright] + 8.0) / np.maximum(out_l[overbright], 1.0)
        out[overbright] *= np.clip(ratio, 0.22, 0.90)[:, None]

    out_l = v1.luma(out)
    interior = dark & ~edge_ring
    leaking = interior & (out_l > 42.0)
    out[leaking] *= 0.38
    edge_leaking = dark & edge_ring & (out_l > 54.0)
    out[edge_leaking] *= 0.52
    out[~maps["subject"]] = 0
    return np.clip(out, 0, 255)


def compute_metrics(ref: np.ndarray, rec: np.ndarray, maps: dict[str, np.ndarray], stats: dict[str, int]) -> dict[str, float | int | list[int]]:
    out = v2.compute_metrics(ref, rec, maps)
    out.update(stats)
    return out


def save_l(path: Path, data: np.ndarray) -> None:
    Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), "L").save(path)


def main() -> None:
    t0 = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    ref = v1.load_reference()
    maps = v1.build_target_maps(ref)
    maps["dark_zone"] = refine_protected_dark_zone_v3(maps)
    visibility = v2.build_visibility_mask(ref, maps)

    face_raw = v1.combine_layers(
        [
            v1.draw_micro_texture(maps["face"], maps, rng),
            v1.draw_weighted_texture(maps["face"], maps, v1.FACE_WORDS, rng, count=2850, size_range=(9, 20), alpha_range=(72, 176)),
            v1.draw_contour_texture(maps["face"], maps, rng),
        ]
    )
    jersey_raw = v1.combine_layers(
        [
            v1.draw_micro_texture(maps["jersey"], maps, rng, jersey=True),
            v1.draw_weighted_texture(
                maps["jersey"], maps, v1.JERSEY_WORDS, rng, jersey=True, count=2050, size_range=(9, 23), alpha_range=(62, 152)
            ),
            v1.draw_contour_texture(maps["jersey"], maps, rng, jersey=True),
        ]
    )
    shoulder_raw = v1.draw_weighted_texture(maps["shoulder"], maps, v1.MICRO_WORDS, rng, count=620, size_range=(8, 18), alpha_range=(42, 118))

    base_face = v2.apply_face_stencil(face_raw, ref, maps, visibility)
    detail, detail_layer, stats, edge_ring = render_face_detail_recovery(ref, maps, visibility, rng)
    recovered_face = face_after_recovery(base_face, detail, maps, visibility)
    jersey_after = v2.apply_jersey_stencil(jersey_raw, ref, maps, visibility)
    body_after = v2.apply_body_stencil(shoulder_raw, maps, visibility)
    jersey_anchors = v2.render_integrated_jersey_anchors(ref, maps, visibility)
    face_anchors = v2.render_face_anchors(maps, visibility)

    final_arr = np.clip(body_after + recovered_face + jersey_after + jersey_anchors + face_anchors, 0, 255)
    final_arr = final_corrections_v3(final_arr, ref, maps, edge_ring)
    final_img = Image.fromarray(final_arr.astype(np.uint8), "RGB").convert("RGBA")
    v1.draw_text(final_img, 795, 514, "change the game.", 28, (188, 188, 202), 220)
    final_rgb = final_img.convert("RGB")
    rec = np.array(final_rgb, dtype=np.uint8)

    metrics = compute_metrics(ref, rec, maps, stats)
    metrics["render_time_seconds"] = round(perf_counter() - t0, 3)
    metrics["seed"] = SEED

    final_rgb.save(OUT / "stencil_v3_final.png")
    v1.side_by_side(ref, final_rgb).save(OUT / "stencil_v3_side_by_side.png")
    detail_layer.convert("RGB").save(OUT / "stencil_v3_face_detail_layer.png")
    save_l(OUT / "stencil_v3_dark_zone_mask.png", maps["dark_zone"] * 255)
    save_l(OUT / "stencil_v3_edge_ring_mask.png", edge_ring * 255)
    Image.fromarray(recovered_face.astype(np.uint8), "RGB").save(OUT / "stencil_v3_face_after_recovery.png")
    (OUT / "stencil_v3_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
