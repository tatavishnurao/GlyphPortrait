from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from glyphforge.semantic_micrography.config import MicrographyStyleConfig
from glyphforge.semantic_micrography.vectorize import ContourPath


@dataclass(frozen=True)
class TextLane:
    id: str
    region: str
    points: list[tuple[float, float]]
    length_px: float
    mean_curvature: float
    closed: bool = False
    order_index: int = 0
    source: str = "scanline"


@dataclass
class LaneDiagnostics:
    region: str
    candidate_lanes: int = 0
    accepted_lanes: int = 0
    discarded_short_lanes: int = 0
    discarded_curvy_lanes: int = 0
    discarded_spacing_lanes: int = 0
    mean_lane_length_px: float = 0.0
    mean_lane_curvature: float = 0.0
    short_lane_ratio: float = 0.0
    total_lane_length_px: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | int | list[str] | str]:
        return {
            "region": self.region,
            "candidate_lanes": self.candidate_lanes,
            "accepted_lanes": self.accepted_lanes,
            "discarded_short_lanes": self.discarded_short_lanes,
            "discarded_curvy_lanes": self.discarded_curvy_lanes,
            "discarded_spacing_lanes": self.discarded_spacing_lanes,
            "mean_lane_length_px": self.mean_lane_length_px,
            "mean_lane_curvature": self.mean_lane_curvature,
            "short_lane_ratio": self.short_lane_ratio,
            "total_lane_length_px": self.total_lane_length_px,
            "notes": self.notes,
        }


REGION_SPACING_SCALE = {
    "dark_hair_or_shadow": 0.82,
    "skin_or_warm": 0.95,
    "clothing_primary": 1.05,
    "clothing_secondary": 1.05,
    "highlight": 0.90,
    "outline_or_edge": 0.62,
}


def lane_length(points: list[tuple[float, float]], closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    pairs = list(zip(points[:-1], points[1:]))
    if closed and len(points) > 2:
        pairs.append((points[-1], points[0]))
    return float(sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in pairs))


def mean_curvature(points: list[tuple[float, float]], closed: bool = False) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    count = 0
    n = len(points)
    indices = range(n) if closed else range(1, n - 1)
    for idx in indices:
        if not closed and (idx <= 0 or idx >= n - 1):
            continue
        x0, y0 = points[(idx - 1) % n]
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % n]
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        l1 = math.hypot(v1x, v1y)
        l2 = math.hypot(v2x, v2y)
        if l1 < 1e-3 or l2 < 1e-3:
            continue
        dot = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (l1 * l2)))
        total += math.acos(dot) / max((l1 + l2) * 0.5, 1.0)
        count += 1
    return total / max(count, 1)


def simplify_polyline(points: list[tuple[float, float]], epsilon: float = 1.5) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=False).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in approx]


def _dominant_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size < 16:
        return 0.0
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, int(np.argmax(vals))]
    angle = math.degrees(math.atan2(float(vy), float(vx)))
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    if abs(angle) > 52.0:
        angle = 0.0
    return float(angle)


def _line_mask(shape: tuple[int, int], points: list[tuple[float, float]], width: int) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    if len(points) < 2:
        return canvas
    cv2.polylines(canvas, [np.array(points, dtype=np.int32)], isClosed=False, color=255, thickness=max(1, width), lineType=cv2.LINE_AA)
    return canvas > 0


def _extract_runs(samples: list[tuple[float, float, bool]], min_points: int) -> list[list[tuple[float, float]]]:
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for x, y, inside in samples:
        if inside:
            current.append((x, y))
            continue
        if len(current) >= min_points:
            runs.append(current)
        current = []
    if len(current) >= min_points:
        runs.append(current)
    return runs


def _scanline_candidates(mask: np.ndarray, region: str, spacing: float, min_length: float) -> list[list[tuple[float, float]]]:
    h, w = mask.shape
    angle = _dominant_angle(mask)
    theta = math.radians(angle)
    dx, dy = math.cos(theta), math.sin(theta)
    nx, ny = -dy, dx
    cx, cy = w * 0.5, h * 0.5
    diag = math.hypot(w, h)
    step = 4.0
    samples_per_line = int(diag / step) + 3
    offsets = np.arange(-diag * 0.55, diag * 0.55 + spacing, spacing)
    candidates: list[list[tuple[float, float]]] = []
    for offset in offsets:
        line_samples: list[tuple[float, float, bool]] = []
        for idx in range(samples_per_line):
            t = -diag * 0.55 + idx * step
            x = cx + nx * offset + dx * t
            y = cy + ny * offset + dy * t
            ix, iy = int(round(x)), int(round(y))
            inside = 0 <= ix < w and 0 <= iy < h and bool(mask[iy, ix] > 0)
            line_samples.append((x, y, inside))
        for run in _extract_runs(line_samples, min_points=4):
            simplified = simplify_polyline(run, epsilon=1.2)
            if lane_length(simplified) >= min_length:
                candidates.append(simplified)
    if region == "highlight" and len(candidates) > 1:
        candidates = candidates[::2]
    return candidates


def _contour_candidates(contours: list[ContourPath], min_length: float) -> list[list[tuple[float, float]]]:
    candidates: list[list[tuple[float, float]]] = []
    for contour in contours[:16]:
        pts = contour.points
        if len(pts) < 6:
            continue
        length = lane_length(pts, closed=True)
        if length < min_length:
            continue
        candidates.append(pts)
    return candidates


def generate_lanes_for_region(
    region: str,
    mask: np.ndarray,
    contours: list[ContourPath],
    style: MicrographyStyleConfig,
    occupancy: np.ndarray | None = None,
) -> tuple[list[TextLane], LaneDiagnostics, np.ndarray]:
    spacing = style.lane_spacing_px * REGION_SPACING_SCALE.get(region, 1.0)
    min_length = max(style.min_lane_length_px, spacing * 3.2)
    max_curvature = style.max_lane_curvature
    diagnostics = LaneDiagnostics(region=region)
    if occupancy is None:
        occupancy = np.zeros(mask.shape, dtype=bool)
    if int((mask > 0).sum()) < max(20, int(min_length)):
        diagnostics.notes.append("region too small")
        return [], diagnostics, occupancy

    candidates = _scanline_candidates(mask, region, spacing, min_length)
    if region == "outline_or_edge":
        candidates.extend(_contour_candidates(contours, min_length=min_length * 0.75))
    diagnostics.candidate_lanes = len(candidates)

    scored: list[tuple[float, list[tuple[float, float]], bool]] = []
    for points in candidates:
        closed = region == "outline_or_edge" and len(points) > 8 and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < spacing * 1.5
        length = lane_length(points, closed=closed)
        if length < min_length:
            diagnostics.discarded_short_lanes += 1
            continue
        curvature = mean_curvature(points, closed=closed)
        if curvature > max_curvature:
            diagnostics.discarded_curvy_lanes += 1
            continue
        scored.append((length / (1.0 + curvature * 24.0), points, closed))
    scored.sort(key=lambda item: item[0], reverse=True)

    lanes: list[TextLane] = []
    mark_width = max(2, int(round(spacing * 0.50)))
    for _, points, closed in scored:
        lane_mask = _line_mask(mask.shape, points + ([points[0]] if closed else []), mark_width)
        if bool(np.any(lane_mask & occupancy)):
            diagnostics.discarded_spacing_lanes += 1
            continue
        occupancy |= _line_mask(mask.shape, points + ([points[0]] if closed else []), max(2, int(round(spacing * 0.82))))
        lanes.append(
            TextLane(
                id=f"{region}_lane_{len(lanes):04d}",
                region=region,
                points=points,
                length_px=lane_length(points, closed=closed),
                mean_curvature=mean_curvature(points, closed=closed),
                closed=closed,
                source="contour" if closed else "scanline",
            )
        )

    lengths = [lane.length_px for lane in lanes]
    curvatures = [lane.mean_curvature for lane in lanes]
    diagnostics.accepted_lanes = len(lanes)
    diagnostics.total_lane_length_px = float(sum(lengths))
    diagnostics.mean_lane_length_px = float(np.mean(lengths)) if lengths else 0.0
    diagnostics.mean_lane_curvature = float(np.mean(curvatures)) if curvatures else 0.0
    diagnostics.short_lane_ratio = float(diagnostics.discarded_short_lanes / diagnostics.candidate_lanes) if diagnostics.candidate_lanes else 0.0
    return lanes, diagnostics, occupancy


def generate_lanes(
    masks: dict[str, np.ndarray],
    contours: dict[str, list[ContourPath]],
    style: MicrographyStyleConfig,
) -> tuple[list[TextLane], dict[str, dict[str, float | int | list[str] | str]]]:
    all_lanes: list[TextLane] = []
    diagnostics: dict[str, dict[str, float | int | list[str] | str]] = {}
    occupancy = np.zeros(next(iter(masks.values())).shape, dtype=bool)
    for region in [
        "skin_or_warm",
        "clothing_primary",
        "clothing_secondary",
        "dark_hair_or_shadow",
        "highlight",
        "outline_or_edge",
    ]:
        if region not in masks:
            continue
        if region in {"dark_hair_or_shadow", "highlight", "outline_or_edge"}:
            detail_occupancy = np.zeros_like(occupancy)
            lanes, diag, _ = generate_lanes_for_region(region, masks[region], contours.get(region, []), style, detail_occupancy)
        else:
            lanes, diag, occupancy = generate_lanes_for_region(region, masks[region], contours.get(region, []), style, occupancy)
        all_lanes.extend(lanes)
        diagnostics[region] = diag.to_dict()
    return all_lanes, diagnostics
