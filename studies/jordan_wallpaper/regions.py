from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .target_analysis import (
    build_target_edge_map,
    build_target_luminance_map,
    extract_face_body_mask,
    extract_nonblack_subject_mask,
    extract_red_jersey_mask,
    find_subject_bbox,
)


@dataclass
class ReferenceRegions:
    subject_mask: np.ndarray
    face_mask: np.ndarray
    jersey_mask: np.ndarray
    negative_space_mask: np.ndarray
    luminance_map: np.ndarray
    edge_map: np.ndarray
    subject_bbox: tuple[int, int, int, int]


def build_reference_regions(rgb: np.ndarray) -> ReferenceRegions:
    subject_mask = extract_nonblack_subject_mask(rgb)
    jersey_mask = extract_red_jersey_mask(rgb, subject_mask)
    face_mask = extract_face_body_mask(subject_mask, jersey_mask)
    negative_space_mask = (subject_mask == 0).astype(np.uint8) * 255
    luminance = build_target_luminance_map(rgb)
    gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = build_target_edge_map(gray_u8)
    bbox = find_subject_bbox(subject_mask)
    return ReferenceRegions(
        subject_mask=subject_mask,
        face_mask=face_mask,
        jersey_mask=jersey_mask,
        negative_space_mask=negative_space_mask,
        luminance_map=luminance,
        edge_map=edges,
        subject_bbox=bbox,
    )
