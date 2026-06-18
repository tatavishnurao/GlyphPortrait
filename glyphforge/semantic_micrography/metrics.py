from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from glyphforge.semantic_micrography.lanes import TextLane
from glyphforge.semantic_micrography.text_layout import TextLayoutResult, TextPathLayout


def _draw_lane_mask(shape: tuple[int, int], lanes: list[TextLane], width: int = 7) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for lane in lanes:
        if len(lane.points) < 2:
            continue
        pts = np.array(lane.points, dtype=np.int32)
        if lane.closed:
            pts = np.vstack([pts, pts[:1]])
        cv2.polylines(out, [pts], isClosed=False, color=255, thickness=width, lineType=cv2.LINE_AA)
    return out > 0


def _lane_centroid_y(item: TextPathLayout) -> float:
    if not item.lane.points:
        return 0.0
    return float(sum(point[1] for point in item.lane.points) / len(item.lane.points))


def compute_micrography_metrics(
    masks: dict[str, np.ndarray],
    lanes: list[TextLane],
    layout: TextLayoutResult,
    render_time_seconds: float,
    output_resolution: tuple[int, int],
    lane_diagnostics: dict[str, dict[str, Any]],
    edge_map: np.ndarray | None = None,
) -> dict[str, Any]:
    subject = masks["subject"] > 0
    lengths = [lane.length_px for lane in lanes]
    curvatures = [lane.mean_curvature for lane in lanes]
    lane_mask = _draw_lane_mask(subject.shape, lanes, width=9)
    feature_lanes = [lane for lane in lanes if lane.source == "feature" or lane.region == "feature_detail"]
    feature_lane_mask = _draw_lane_mask(subject.shape, feature_lanes, width=9)
    ys, xs = np.where(subject)
    face_feature_coverage = 0.0
    face_text_coverage = 0.0
    identity_edge_alignment = 0.0
    identity_edge_recall = 0.0
    min_y = 0
    max_y = subject.shape[0] - 1
    if xs.size:
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        width = max(1, max_x - min_x + 1)
        height = max(1, max_y - min_y + 1)
        face_zone = np.zeros_like(subject, dtype=bool)
        x0 = max(0, int(round(min_x + width * 0.12)))
        x1 = min(subject.shape[1], int(round(min_x + width * 0.88)))
        y0 = max(0, int(round(min_y + height * 0.08)))
        y1 = min(subject.shape[0], int(round(min_y + height * 0.70)))
        face_zone[y0:y1, x0:x1] = True
        face_subject = face_zone & subject
        face_feature_coverage = float((feature_lane_mask & face_subject).sum() / max(int(face_subject.sum()), 1))
        face_text_coverage = float((lane_mask & face_subject).sum() / max(int(face_subject.sum()), 1))
        if edge_map is not None:
            face_edges = (edge_map > 0) & face_subject
            edge_support = cv2.dilate(
                face_edges.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                iterations=1,
            ) > 0
            face_feature = feature_lane_mask & face_subject
            identity_edge_alignment = float((face_feature & edge_support).sum() / max(int(face_feature.sum()), 1))
            identity_edge_recall = float((face_feature & edge_support).sum() / max(int(edge_support.sum()), 1))
    region_coverage: dict[str, float] = {}
    for region, mask in masks.items():
        if region == "subject":
            continue
        region_bool = mask > 0
        if np.any(region_bool):
            region_coverage[region] = float(np.logical_and(lane_mask, region_bool).sum() / region_bool.sum())
        else:
            region_coverage[region] = 0.0
    candidate_total = sum(int(diag.get("candidate_lanes", 0)) for diag in lane_diagnostics.values())
    short_total = sum(int(diag.get("discarded_short_lanes", 0)) for diag in lane_diagnostics.values())
    text_chars = int(layout.coverage.get("total_text_chars", 0))
    anchors = [item for item in layout.text_paths if item.is_anchor]
    subject_height = max(1.0, float(max_y - min_y + 1))
    anchor_visibility = 0.0
    oversized_body_anchor_weight = 0.0
    anchor_weight_total = 0.0
    for item in anchors:
        y_norm = (_lane_centroid_y(item) - min_y) / subject_height
        readable_weight = min(1.0, max(0.0, (item.font_size - 12.0) / 18.0))
        length_weight = min(1.0, item.lane.length_px / 260.0)
        zone_weight = 1.0
        if y_norm < 0.12:
            zone_weight = 0.70
        elif y_norm <= 0.72:
            zone_weight = 1.15
        elif y_norm <= 0.84:
            zone_weight = 0.55
        else:
            zone_weight = 0.20
        weight = readable_weight * length_weight
        anchor_visibility += weight * zone_weight
        anchor_weight_total += max(weight, 0.001)
        if y_norm > 0.74 and item.font_size >= 22:
            oversized_body_anchor_weight += weight
    anchor_visibility_score = float(min(1.0, anchor_visibility / max(len(anchors), 1)))
    anchor_feature_lane_ratio = float(
        sum(1 for item in anchors if item.lane.source == "feature" or item.lane.region == "feature_detail")
        / max(len(anchors), 1)
    )
    low_body_anchor_ratio = float(oversized_body_anchor_weight / max(anchor_weight_total, 1e-6))
    used_total = int(layout.coverage.get("used_word_count", 0))
    max_word_count = max(layout.used_words.values(), default=0)
    word_repetition_ratio = float(max_word_count / max(used_total, 1))
    return {
        "subject_coverage": float(subject.mean()),
        "region_coverage": region_coverage,
        "lane_count": len(lanes),
        "total_lane_length_px": float(sum(lengths)),
        "mean_lane_length_px": float(np.mean(lengths)) if lengths else 0.0,
        "short_lane_ratio": float(short_total / candidate_total) if candidate_total else 0.0,
        "mean_lane_curvature": float(np.mean(curvatures)) if curvatures else 0.0,
        "lane_coverage_subject": float(np.logical_and(lane_mask, subject).sum() / max(int(subject.sum()), 1)),
        "text_coverage_subject": float(np.logical_and(lane_mask, subject).sum() / max(int(subject.sum()), 1)),
        "face_feature_coverage": face_feature_coverage,
        "face_text_coverage": face_text_coverage,
        "identity_edge_alignment": identity_edge_alignment,
        "identity_edge_recall": identity_edge_recall,
        "source_pixel_leakage": 0.0,
        "background_cleanliness": 1.0,
        "render_time_seconds": round(render_time_seconds, 3),
        "output_resolution": [int(output_resolution[0]), int(output_resolution[1])],
        "used_word_count": int(layout.coverage.get("used_word_count", 0)),
        "total_text_chars": text_chars,
        "microtext_to_lane_ratio": float(text_chars / max(len(lanes), 1)),
        "anchor_visibility_score": anchor_visibility_score,
        "anchor_feature_lane_ratio": anchor_feature_lane_ratio,
        "low_body_anchor_ratio": low_body_anchor_ratio,
        "word_repetition_ratio": word_repetition_ratio,
        "candidate_lanes": candidate_total,
        "short_lane_count": short_total,
        "curvature_rms": float(math.sqrt(np.mean(np.square(curvatures)))) if curvatures else 0.0,
    }
