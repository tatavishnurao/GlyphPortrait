from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch import recreate_jordan_reference_stencil as v1
from scratch import recreate_jordan_reference_stencil_v2 as v2
from scratch import recreate_jordan_reference_stencil_v3 as v3
from scratch import recreate_jordan_reference_stencil_v4 as v4
from scratch import recreate_jordan_stencil_v5_face_only as face_v5

OUT = ROOT / "examples" / "reference_recreation"
W, H = v1.W, v1.H


def ensure_artifact(path: Path, command: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], tuple[int, int, int, int]]:
    ensure_artifact(OUT / "stencil_v4_final.png", [sys.executable, "scratch/recreate_jordan_reference_stencil_v4.py"])
    ensure_artifact(OUT / "stencil_v5_face_only.png", [sys.executable, "scratch/recreate_jordan_stencil_v5_face_only.py"])
    ref = v1.load_reference()
    v4_full = np.array(Image.open(OUT / "stencil_v4_final.png").convert("RGB"), dtype=np.float32)
    face_crop = np.array(Image.open(OUT / "stencil_v5_face_only.png").convert("RGB"), dtype=np.float32)
    maps = v1.build_target_maps(ref)
    maps["dark_zone"] = v3.refine_protected_dark_zone_v3(maps)
    box = face_v5.face_crop_box(maps["face"])
    return ref, v4_full, face_crop, maps, box


def build_face_insert_mask(maps: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    face = maps["face"]
    jersey_exclusion = ndi.binary_dilation(maps["jersey"], structure=np.ones((31, 31), dtype=bool), iterations=1)
    shoulder_exclusion = ndi.binary_dilation(maps["shoulder"], structure=np.ones((11, 11), dtype=bool), iterations=1)
    insert = face & ~jersey_exclusion
    # Preserve the strongest collar/shoulder transition from v4.
    insert &= ~(shoulder_exclusion & (maps["tone"] > 0.18))
    insert = ndi.binary_opening(insert, structure=np.ones((3, 3), dtype=bool), iterations=1)
    insert = ndi.binary_closing(insert, structure=np.ones((5, 5), dtype=bool), iterations=1)
    feather = np.array(Image.fromarray(insert.astype(np.uint8) * 255, "L").filter(ImageFilter.GaussianBlur(5.2)), dtype=np.float32)
    feather = np.clip(feather / 255.0, 0.0, 1.0)
    feather[maps["jersey"]] = 0.0
    feather[~maps["subject"]] = 0.0
    core = ndi.binary_erosion(insert, structure=np.ones((9, 9), dtype=bool), iterations=1)
    feather[core] = np.maximum(feather[core], 0.96)
    return insert, feather.astype(np.float32)


def paste_face_crop(face_crop: np.ndarray, box: tuple[int, int, int, int], shape: tuple[int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    out = np.zeros(shape, dtype=np.float32)
    expected = (y1 - y0, x1 - x0)
    crop = face_crop
    if face_crop.shape[:2] != expected:
        crop = np.array(Image.fromarray(face_crop.astype(np.uint8), "RGB").resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS), dtype=np.float32)
    out[y0:y1, x0:x1] = crop
    return out


def composite_face(v4_full: np.ndarray, face_full: np.ndarray, feather: np.ndarray) -> np.ndarray:
    alpha = feather[..., None]
    return np.clip(v4_full * (1.0 - alpha) + face_full * alpha, 0, 255)


def cleanup_fragments(arr: np.ndarray, maps: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    out = arr.copy()
    visible = (v1.luma(out) > 8.0) & ~maps["subject"]
    labels, count = ndi.label(visible, structure=np.ones((3, 3), dtype=bool))
    removed = 0
    for label in range(1, count + 1):
        region = labels == label
        if int(region.sum()) < 30:
            out[region] = 0
            removed += 1
    out[~maps["subject"] & (v1.luma(out) < 16)] = 0
    return out, removed


def final_face_corrections(arr: np.ndarray, ref: np.ndarray, maps: dict[str, np.ndarray]) -> np.ndarray:
    out = arr.copy()
    ref_l = v1.luma(ref)
    ref_gray = np.repeat(ref_l[..., None], 3, axis=2)
    edge_midtone = maps["face"] & ~maps["dark_zone"] & (maps["edge_strength"] > 0.035)
    out[edge_midtone] = out[edge_midtone] * 0.94 + ref_gray[edge_midtone] * 0.06
    out_l = v1.luma(out)
    dark = maps["dark_zone"]
    interior = dark & (maps["edge_strength"] < 0.12)
    leak = interior & (out_l > 48.0)
    out[leak] *= 0.62

    rel_y = v2.rel_y_mask(maps["face"])
    lower = maps["face"] & (rel_y > 0.55) & (rel_y < 0.92)
    out_l = v1.luma(out)
    overbright = lower & (out_l > ref_l + 22.0) & (out_l > 54.0)
    if np.any(overbright):
        ratio = (ref_l[overbright] + 8.0) / np.maximum(out_l[overbright], 1.0)
        out[overbright] *= np.clip(ratio, 0.22, 0.90)[:, None]
    out[~maps["subject"]] = 0
    return np.clip(out, 0, 255)


def seam_overlay(ref: np.ndarray, final: np.ndarray, insert: np.ndarray, feather: np.ndarray) -> Image.Image:
    base = Image.fromarray(final.astype(np.uint8), "RGB").convert("RGBA")
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    boundary = ndi.binary_dilation(insert, structure=np.ones((9, 9), dtype=bool), iterations=1) ^ ndi.binary_erosion(
        insert, structure=np.ones((9, 9), dtype=bool), iterations=1
    )
    soft_band = (feather > 0.05) & (feather < 0.95)
    overlay[soft_band] = (255, 210, 40, 90)
    overlay[boundary] = (0, 210, 255, 150)
    out = Image.alpha_composite(base, Image.fromarray(overlay, "RGBA"))
    ImageDraw.Draw(out).text((18, 18), "cyan: insert boundary  yellow: feather band", font=v1.font(22, True), fill=(255, 255, 255, 245))
    return out.convert("RGB")


def seam_artifact_score(v4_full: np.ndarray, final: np.ndarray, insert: np.ndarray, maps: dict[str, np.ndarray]) -> float:
    boundary = ndi.binary_dilation(insert, structure=np.ones((7, 7), dtype=bool), iterations=1) ^ ndi.binary_erosion(
        insert, structure=np.ones((7, 7), dtype=bool), iterations=1
    )
    boundary &= maps["subject"] & ~maps["jersey"]
    if not np.any(boundary):
        return 0.0
    diff = np.abs(v1.luma(v4_full) - v1.luma(final))
    return float(diff[boundary].mean())


def compute_metrics(ref: np.ndarray, rec: np.ndarray, maps: dict[str, np.ndarray], removed_fragments: int, seam_score: float) -> dict[str, float | int | list[int]]:
    base = v2.compute_metrics(ref, rec, maps)
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
        "floating_fragment_count": removed_fragments,
        "seam_artifact_score": seam_score,
        "output_resolution": [W, H],
    }
    return out


def main() -> None:
    start = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    ref, v4_full, face_crop, maps, box = load_inputs()
    insert, feather = build_face_insert_mask(maps)
    face_full = paste_face_crop(face_crop, box, v4_full.shape)
    merged = composite_face(v4_full, face_full, feather)
    merged = final_face_corrections(merged, ref, maps)
    merged, removed_fragments = cleanup_fragments(merged, maps)
    rec = np.clip(merged, 0, 255).astype(np.uint8)
    seam_score = seam_artifact_score(v4_full, rec.astype(np.float32), insert, maps)
    metrics = compute_metrics(ref, rec, maps, removed_fragments, seam_score)
    metrics["render_time_seconds"] = round(perf_counter() - start, 3)
    metrics["face_crop_box"] = list(box)

    final = Image.fromarray(rec, "RGB")
    final.save(OUT / "stencil_v5_final.png")
    v1.side_by_side(ref, final).save(OUT / "stencil_v5_side_by_side.png")
    Image.fromarray(np.clip(feather * 255, 0, 255).astype(np.uint8), "L").save(OUT / "stencil_v5_face_insert_mask.png")
    seam_overlay(ref, rec.astype(np.float32), insert, feather).save(OUT / "stencil_v5_seam_overlay.png")
    x0, y0, x1, y1 = box
    final.crop(box).save(OUT / "stencil_v5_face_crop_inserted.png")
    (OUT / "stencil_v5_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
