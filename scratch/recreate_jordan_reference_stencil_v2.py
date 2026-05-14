from __future__ import annotations

import json
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

OUT = ROOT / "examples" / "reference_recreation"
W, H = v1.W, v1.H
SEED = 20260515


def rel_y_mask(mask: np.ndarray) -> np.ndarray:
    _, y0, _, y1 = v1.bbox(mask)
    yy = np.broadcast_to(np.arange(H)[:, None], mask.shape)
    return (yy - y0) / max(1, y1 - y0)


def build_visibility_mask(ref: np.ndarray, maps: dict[str, np.ndarray]) -> np.ndarray:
    tone = maps["tone"]
    shape = maps["subject"]
    local = v1.contrast_stretch(tone, shape, 1.0, 99.0)
    detail = np.clip(local * 0.74 + maps["enhanced_luma"] * 0.38 + maps["edge_strength"] * 0.28, 0.0, 1.0)
    visibility = np.power(detail, 0.82)
    visibility *= np.clip((tone - 0.035) / 0.36, 0.0, 1.0) ** 0.34
    visibility = np.clip(visibility + maps["edge_strength"] * 0.16, 0.0, 1.0)
    visibility[maps["dark_zone"]] *= 0.22
    visibility[~shape] = 0.0
    return visibility.astype(np.float32)


def refine_protected_dark_zone(maps: dict[str, np.ndarray]) -> np.ndarray:
    face = maps["face"]
    tone = maps["tone"]
    rel_y = rel_y_mask(face)
    edge = maps["edge_strength"]
    eye_socket = face & (rel_y > 0.25) & (rel_y < 0.53) & (tone < 0.25)
    mouth_crease = face & (rel_y > 0.58) & (rel_y < 0.78) & ((tone < 0.22) | ((tone < 0.30) & (edge > 0.15)))
    under_jaw = face & (rel_y > 0.78) & (tone < 0.18)
    deep_shadow = face & (tone < 0.12)
    dark = eye_socket | mouth_crease | under_jaw | deep_shadow
    dark = ndi.binary_opening(dark, structure=np.ones((2, 2), dtype=bool), iterations=1)
    dark = ndi.binary_dilation(dark, structure=np.ones((3, 3), dtype=bool), iterations=1)
    return dark & face


def sampled_jersey_target(ref: np.ndarray, maps: dict[str, np.ndarray]) -> np.ndarray:
    ref_f = ref.astype(np.float32)
    gray = v1.luma(ref)[..., None]
    desat = gray * 0.20 + ref_f * 0.80
    red = maps["red_strength"][..., None]
    target = desat.copy()
    target[..., 0:1] = target[..., 0:1] * (1.10 + red * 0.32) + red * 22.0
    target[..., 1:2] = target[..., 1:2] * 0.74
    target[..., 2:3] = target[..., 2:3] * 0.76
    cream = (maps["tone"] > 0.50) & maps["jersey"]
    target[cream] = ref_f[cream] * 0.76 + np.array([232.0, 220.0, 198.0], dtype=np.float32) * 0.24
    return np.clip(target, 0, 255)


def apply_face_stencil(raw_layer: Image.Image, ref: np.ndarray, maps: dict[str, np.ndarray], visibility: np.ndarray) -> np.ndarray:
    face = maps["face"]
    raw = v1.alpha_to_rgb(v1.clip_layer(raw_layer, face))
    rel_y = rel_y_mask(face)
    tone = maps["tone"]
    midtone_recovery = (
        face
        & ~maps["dark_zone"]
        & (tone > 0.13)
        & (tone < 0.46)
        & (rel_y > 0.28)
        & (rel_y < 0.98)
    )
    curve = np.clip(0.40 + 1.58 * np.power(maps["enhanced_luma"], 0.82) + maps["edge_strength"] * 0.24, 0.18, 2.05)
    curve[midtone_recovery] += 0.52
    curve[maps["highlight"] & face] = np.maximum(curve[maps["highlight"] & face], 1.16)
    face_visibility = np.clip(0.58 + visibility * 0.72, 0.0, 1.24)
    out = raw * curve[..., None] * face_visibility[..., None]
    ref_gray = v1.luma(ref)[..., None]
    floor = ref_gray * np.clip(0.18 + visibility * 0.54 + maps["edge_strength"] * 0.28, 0.0, 0.82)[..., None]
    floor_mask = face & ~maps["dark_zone"] & (maps["tone"] > 0.10)
    out[floor_mask] = np.maximum(out[floor_mask], floor[floor_mask])
    out[maps["dark_zone"] & face] *= 0.18
    out[~face] = 0
    return np.clip(out, 0, 255)


def apply_jersey_stencil(raw_layer: Image.Image, ref: np.ndarray, maps: dict[str, np.ndarray], visibility: np.ndarray) -> np.ndarray:
    jersey = maps["jersey"]
    raw = v1.alpha_to_rgb(v1.clip_layer(raw_layer, jersey))
    raw_l = np.clip(v1.luma(raw) / 92.0, 0.0, 1.78)
    red_floor = maps["red_strength"] * np.clip(0.68 + visibility * 0.95 + maps["edge_strength"] * 0.52, 0.0, 1.55)
    raw_l = np.maximum(raw_l, red_floor)
    target = sampled_jersey_target(ref, maps)
    curve = np.clip(0.42 + np.power(maps["enhanced_luma"], 0.90) * 1.18 + maps["red_strength"] * 0.48, 0.12, 1.92)
    vis = np.clip(0.38 + visibility * 0.78, 0.0, 1.10)
    out = target * raw_l[..., None] * curve[..., None] * vis[..., None] * 1.58
    shadow = jersey & (maps["tone"] < 0.12)
    out[shadow] *= 0.38
    out[~jersey] = 0
    return np.clip(out, 0, 255)


def apply_body_stencil(raw_layer: Image.Image, maps: dict[str, np.ndarray], visibility: np.ndarray) -> np.ndarray:
    body = maps["shoulder"]
    raw = v1.alpha_to_rgb(v1.clip_layer(raw_layer, body))
    curve = np.clip(0.18 + maps["enhanced_luma"] * 1.10 + maps["edge_strength"] * 0.16, 0.08, 1.34)
    out = raw * curve[..., None] * np.clip(0.20 + visibility * 0.74, 0.0, 1.0)[..., None]
    out[~body] = 0
    return np.clip(out, 0, 255)


def render_integrated_jersey_anchors(ref: np.ndarray, maps: dict[str, np.ndarray], visibility: np.ndarray) -> np.ndarray:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    anchors = [
        ("BULLS", 1464, 949, 88, (232, 218, 196), 172, -12),
        ("23", 1549, 846, 108, (236, 225, 204), 168, -8),
    ]
    for text, x, y, size, color, alpha, angle in anchors:
        v1.draw_text(layer, x, y, text, size, color, alpha, angle, True)
    raw = v1.alpha_to_rgb(v1.clip_layer(layer, maps["jersey"]))
    anchor_l = np.clip(v1.luma(raw) / 210.0, 0.0, 1.0)
    target = sampled_jersey_target(ref, maps)
    cream = np.array([236.0, 224.0, 203.0], dtype=np.float32)
    color = target * 0.34 + cream * 0.66
    mod = np.clip(0.28 + maps["enhanced_luma"] * 0.86 + visibility * 0.28, 0.08, 1.18)
    out = color * anchor_l[..., None] * mod[..., None]
    out[maps["tone"] < 0.12] *= 0.58
    out[~maps["jersey"]] = 0
    return np.clip(out, 0, 255)


def render_face_anchors(maps: dict[str, np.ndarray], visibility: np.ndarray) -> np.ndarray:
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    anchors = [
        ("NBA Rookie of the Year", 1402, 220, 24, (220, 220, 226), 150, 5),
        ("MVP", 1411, 285, 38, (230, 230, 236), 158, -5),
        ("Air Jordan", 1370, 365, 27, (214, 214, 220), 150, 4),
        ("Dedication", 1375, 454, 27, (152, 152, 160), 122, 24),
        ("Dominance", 1343, 590, 26, (214, 214, 222), 145, 69),
        ("Scoring", 1565, 535, 24, (218, 218, 224), 142, -8),
    ]
    for text, x, y, size, color, alpha, angle in anchors:
        v1.draw_text(layer, x, y, text, size, color, alpha, angle, True)
    raw = v1.alpha_to_rgb(v1.clip_layer(layer, maps["face"]))
    mod = np.clip(0.22 + maps["enhanced_luma"] * 0.82 + visibility * 0.18, 0.0, 1.0)
    out = raw * mod[..., None]
    out[maps["dark_zone"]] *= 0.06
    out[~maps["face"]] = 0
    return np.clip(out, 0, 255)


def final_corrections(arr: np.ndarray, ref: np.ndarray, maps: dict[str, np.ndarray]) -> np.ndarray:
    out = arr.copy()
    face = maps["face"]
    ref_l = v1.luma(ref)
    out_l = v1.luma(out)
    dark = maps["dark_zone"]
    face_recover = face & ~dark & (maps["tone"] > 0.07)
    ref_gray = np.repeat(ref_l[..., None], 3, axis=2)
    out[face_recover] = out[face_recover] * 0.58 + ref_gray[face_recover] * 0.42

    ref_red = v1.red_mask(ref, maps["subject"])
    red_recover = ref_red & maps["jersey"] & (maps["red_strength"] > 0.12) & (maps["tone"] > 0.075)
    if np.any(red_recover):
        recovered_r = np.maximum(out[..., 0][red_recover], ref[..., 0][red_recover].astype(np.float32) * 0.72 + 20.0)
        channel_cap = np.maximum(0.0, recovered_r - 24.0)
        out[..., 0][red_recover] = recovered_r
        out[..., 1][red_recover] = np.minimum(out[..., 1][red_recover], channel_cap)
        out[..., 2][red_recover] = np.minimum(out[..., 2][red_recover], channel_cap)

    out_l = v1.luma(out)
    rel_y = rel_y_mask(face)
    lower = face & (rel_y > 0.55) & (rel_y < 0.92)
    overbright = lower & (out_l > ref_l + 23.0) & (out_l > 54.0)
    if np.any(overbright):
        ratio = (ref_l[overbright] + 7.0) / np.maximum(out_l[overbright], 1.0)
        out[overbright] *= np.clip(ratio, 0.20, 0.88)[:, None]

    out_l = v1.luma(out)
    leaking = dark & (out_l > 42.0)
    out[leaking] *= 0.42
    deep = dark & (maps["tone"] < 0.15)
    out[deep] *= 0.68
    out[~maps["subject"]] = 0
    return np.clip(out, 0, 255)


def registration_overlay(ref: np.ndarray, maps: dict[str, np.ndarray], rec: np.ndarray) -> Image.Image:
    ref_mask = maps["subject"]
    rec_mask = (v1.luma(rec) > 9.0) & (np.broadcast_to(np.arange(W), ref_mask.shape) > int(W * v1.RIGHT_CUTOFF))
    base = Image.fromarray((ref.astype(np.float32) * 0.45).astype(np.uint8), "RGB").convert("RGBA")
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    overlay[ref_mask] = (0, 180, 255, 105)
    overlay[rec_mask] = (255, 70, 150, 105)
    overlay[ref_mask & rec_mask] = (245, 245, 245, 150)
    out = Image.alpha_composite(base, Image.fromarray(overlay, "RGBA"))
    d = ImageDraw.Draw(out)
    d.text((18, 18), "cyan: reference shape  magenta: output visible  white: overlap", font=v1.font(22, True), fill=(255, 255, 255, 245))
    return out.convert("RGB")


def compute_metrics(ref: np.ndarray, rec: np.ndarray, maps: dict[str, np.ndarray]) -> dict[str, float | int | list[int]]:
    base = v1.compute_metrics(ref, rec, maps)
    diff = np.abs(ref.astype(np.float32) - rec.astype(np.float32))
    jersey = maps["jersey"]
    out = {
        "mae_full_rgb": base["mae_full_rgb"],
        "mae_subject_rgb": base["mae_subject_rgb"],
        "mae_face_rgb": base["mae_face_rgb"],
        "mae_jersey_rgb": base["mae_jersey_rgb"],
        "face_luma_mae": base["face_luma_mae"],
        "jersey_luma_mae": base["jersey_luma_mae"],
        "mae_jersey_red_channel": float(diff[..., 0][jersey].mean()),
        "edge_overlap_face": base["edge_overlap_face"],
        "red_mask_iou": base["red_mask_iou"],
        "gray_slab_penalty": base["gray_slab_penalty"],
        "mouth_banding_penalty": base["mouth_banding_penalty"],
        "protected_dark_zone_fill_ratio": base["protected_dark_zone_fill_ratio"],
        "subject_coverage": float(maps["subject"].mean()),
        "jersey_coverage": float(maps["jersey"].mean()),
        "output_resolution": [W, H],
        "seed": SEED,
    }
    return out


def save_l(path: Path, data: np.ndarray) -> None:
    Image.fromarray(np.clip(data, 0, 255).astype(np.uint8), "L").save(path)


def main() -> None:
    t0 = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    ref = v1.load_reference()
    maps = v1.build_target_maps(ref)
    maps["dark_zone"] = refine_protected_dark_zone(maps)
    visibility = build_visibility_mask(ref, maps)

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

    face_after = apply_face_stencil(face_raw, ref, maps, visibility)
    jersey_after = apply_jersey_stencil(jersey_raw, ref, maps, visibility)
    body_after = apply_body_stencil(shoulder_raw, maps, visibility)
    jersey_anchors = render_integrated_jersey_anchors(ref, maps, visibility)
    face_anchors = render_face_anchors(maps, visibility)

    final_arr = np.clip(body_after + face_after + jersey_after + jersey_anchors + face_anchors, 0, 255)
    final_arr = final_corrections(final_arr, ref, maps)
    final_img = Image.fromarray(final_arr.astype(np.uint8), "RGB").convert("RGBA")
    v1.draw_text(final_img, 795, 514, "change the game.", 28, (188, 188, 202), 220)
    final_rgb = final_img.convert("RGB")
    rec = np.array(final_rgb, dtype=np.uint8)

    metrics = compute_metrics(ref, rec, maps)
    metrics["render_time_seconds"] = round(perf_counter() - t0, 3)

    save_l(OUT / "stencil_v2_shape_mask.png", maps["subject"] * 255)
    save_l(OUT / "stencil_v2_visibility_mask.png", visibility * 255)
    Image.fromarray(face_after.astype(np.uint8), "RGB").save(OUT / "stencil_v2_face_after_luma.png")
    Image.fromarray(jersey_after.astype(np.uint8), "RGB").save(OUT / "stencil_v2_jersey_after_luma.png")
    final_rgb.save(OUT / "stencil_v2_final.png")
    v1.side_by_side(ref, final_rgb).save(OUT / "stencil_v2_side_by_side.png")
    registration_overlay(ref, maps, rec).save(OUT / "stencil_v2_registration_overlay.png")
    (OUT / "stencil_v2_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
