from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .regions import ReferenceRegions


def compose_right_subject_canvas(
    target_rgb: np.ndarray,
    regions: ReferenceRegions,
    output_size: tuple[int, int],
    right_margin_ratio: float = 0.03,
) -> tuple[np.ndarray, ReferenceRegions]:
    out_w, out_h = output_size
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    x0, y0, x1, y1 = regions.subject_bbox
    crop_rgb = target_rgb[y0:y1, x0:x1]
    crop_subject = regions.subject_mask[y0:y1, x0:x1]
    crop_face = regions.face_mask[y0:y1, x0:x1]
    crop_jersey = regions.jersey_mask[y0:y1, x0:x1]
    crop_lum = regions.luminance_map[y0:y1, x0:x1]
    crop_edge = regions.edge_map[y0:y1, x0:x1]

    sh, sw = crop_rgb.shape[:2]
    scale = min((out_h * 0.96) / max(1, sh), (out_w * 0.42) / max(1, sw))
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))

    rs_rgb = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    rs_subject = cv2.resize(crop_subject, (nw, nh), interpolation=cv2.INTER_NEAREST)
    rs_face = cv2.resize(crop_face, (nw, nh), interpolation=cv2.INTER_NEAREST)
    rs_jersey = cv2.resize(crop_jersey, (nw, nh), interpolation=cv2.INTER_NEAREST)
    rs_lum = cv2.resize(crop_lum, (nw, nh), interpolation=cv2.INTER_AREA)
    rs_edge = cv2.resize(crop_edge, (nw, nh), interpolation=cv2.INTER_AREA)

    x = int(out_w - nw - (out_w * right_margin_ratio))
    y = int((out_h - nh) * 0.52)
    y = max(0, min(y, out_h - nh))
    x = max(0, min(x, out_w - nw))

    dst = canvas[y : y + nh, x : x + nw]
    mask = rs_subject > 0
    dst[mask] = rs_rgb[mask]

    subject_out = np.zeros((out_h, out_w), dtype=np.uint8)
    face_out = np.zeros((out_h, out_w), dtype=np.uint8)
    jersey_out = np.zeros((out_h, out_w), dtype=np.uint8)
    lum_out = np.zeros((out_h, out_w), dtype=np.float32)
    edge_out = np.zeros((out_h, out_w), dtype=np.float32)

    subject_out[y : y + nh, x : x + nw] = rs_subject
    face_out[y : y + nh, x : x + nw] = rs_face
    jersey_out[y : y + nh, x : x + nw] = rs_jersey
    lum_out[y : y + nh, x : x + nw] = rs_lum
    edge_out[y : y + nh, x : x + nw] = rs_edge
    neg_out = (subject_out == 0).astype(np.uint8) * 255

    new_regions = ReferenceRegions(
        subject_mask=subject_out,
        face_mask=face_out,
        jersey_mask=jersey_out,
        negative_space_mask=neg_out,
        luminance_map=lum_out,
        edge_map=edge_out,
        subject_bbox=(x, y, x + nw, y + nh),
    )
    return canvas, new_regions


def to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")
