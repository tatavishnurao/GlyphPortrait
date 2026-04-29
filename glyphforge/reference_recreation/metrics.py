from __future__ import annotations

from typing import Dict

import numpy as np


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = float((aa & bb).sum())
    union = float((aa | bb).sum())
    return 0.0 if union == 0 else inter / union


def luminance_mae_inside_mask(
    target_luminance: np.ndarray, output_luminance: np.ndarray, subject_mask: np.ndarray
) -> float:
    mask = subject_mask > 0
    if not mask.any():
        return 0.0
    diff = np.abs(target_luminance[mask] - output_luminance[mask])
    return float(diff.mean())


def edge_overlap_score(target_edges: np.ndarray, output_edges: np.ndarray) -> float:
    ta = target_edges > 0.2
    oa = output_edges > 0.2
    denom = float((ta | oa).sum())
    if denom == 0:
        return 0.0
    return float((ta & oa).sum()) / denom


def build_metrics(
    payload: Dict[str, float | int | str],
) -> Dict[str, float | int | str]:
    expected = {
        "subject_iou",
        "jersey_iou",
        "luminance_mae_subject",
        "edge_overlap",
        "face_words_placed",
        "jersey_words_placed",
        "anchors_placed",
        "render_ms",
    }
    for key in expected:
        payload.setdefault(key, 0)
    return payload
