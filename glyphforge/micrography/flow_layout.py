from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


DEFAULT_REGION_CONFIG: dict[str, dict[str, float | int | str]] = {
    "hair": {
        "spacing_px": 28.0,
        "min_length_px": 180.0,
        "size": 17,
        "alpha": 196,
        "max_lanes": 18,
        "curvature_limit": 0.32,
        "min_width_px": 14.0,
    },
    "skin": {
        "spacing_px": 30.0,
        "min_length_px": 160.0,
        "size": 15,
        "alpha": 182,
        "max_lanes": 12,
        "curvature_limit": 0.28,
        "min_width_px": 12.0,
    },
    "blue_undershirt": {
        "spacing_px": 32.0,
        "min_length_px": 150.0,
        "size": 16,
        "alpha": 186,
        "max_lanes": 10,
        "curvature_limit": 0.24,
        "min_width_px": 12.0,
    },
    "orange_gi": {
        "spacing_px": 34.0,
        "min_length_px": 170.0,
        "size": 17,
        "alpha": 190,
        "max_lanes": 14,
        "curvature_limit": 0.24,
        "min_width_px": 14.0,
    },
    "outline": {
        "spacing_px": 26.0,
        "min_length_px": 130.0,
        "size": 14,
        "alpha": 204,
        "max_lanes": 8,
        "curvature_limit": 0.34,
        "min_width_px": 8.0,
    },
}


def _merge_region_config(config: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    merged = {key: dict(value) for key, value in DEFAULT_REGION_CONFIG.items()}
    if not config:
        return merged
    for region, overrides in config.items():
        merged.setdefault(region, {})
        merged[region].update(overrides)
    return merged


def _contour_length(points: np.ndarray, closed: bool) -> float:
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    length = float(np.linalg.norm(diffs, axis=1).sum())
    if closed:
        length += float(np.linalg.norm(points[0] - points[-1]))
    return length


def _smooth_contour(points: np.ndarray, closed: bool, window: int = 11) -> np.ndarray:
    if len(points) < max(5, window):
        return points.astype(np.float32)
    window = max(5, window | 1)
    radius = window // 2
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel /= kernel.sum()
    if closed:
        padded = np.vstack([points[-radius:], points, points[:radius]])
    else:
        padded = np.vstack([np.repeat(points[:1], radius, axis=0), points, np.repeat(points[-1:], radius, axis=0)])
    smoothed = padded.copy().astype(np.float32)
    for _ in range(2):
        xs = np.convolve(smoothed[:, 0], kernel, mode="same")
        ys = np.convolve(smoothed[:, 1], kernel, mode="same")
        smoothed = np.column_stack([xs, ys]).astype(np.float32)
    return smoothed[radius:-radius]


def _resample_polyline(points: np.ndarray, step_px: float, closed: bool) -> np.ndarray:
    if len(points) < 2:
        return points.astype(np.float32)
    pts = points.astype(np.float32)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= step_px:
        return points.astype(np.float32)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = np.arange(0.0, total, max(step_px, 3.0), dtype=np.float32)
    out: list[np.ndarray] = []
    idx = 0
    for target in targets:
        while idx + 1 < len(cumulative) and cumulative[idx + 1] < target:
            idx += 1
        length = seg_len[min(idx, len(seg_len) - 1)]
        if length <= 1e-6:
            out.append(pts[idx].copy())
            continue
        ratio = float((target - cumulative[idx]) / length)
        out.append(pts[idx] + seg[idx] * ratio)
    if not closed and (not out or np.linalg.norm(out[-1] - points[-1]) > 1.0):
        out.append(points[-1].astype(np.float32))
    return np.vstack(out).astype(np.float32)


def _mean_curvature(points: np.ndarray, closed: bool) -> float:
    if len(points) < 3:
        return 0.0
    pts = points.astype(np.float32)
    if closed:
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
    else:
        prev_pts = np.vstack([pts[:1], pts[:-1]])
        next_pts = np.vstack([pts[1:], pts[-1:]])
    v1 = pts - prev_pts
    v2 = next_pts - pts
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-3) & (n2 > 1e-3)
    if not np.any(valid):
        return 0.0
    u1 = np.zeros_like(v1)
    u2 = np.zeros_like(v2)
    u1[valid] = v1[valid] / n1[valid, None]
    u2[valid] = v2[valid] / n2[valid, None]
    dot = np.clip(np.sum(u1 * u2, axis=1), -1.0, 1.0)
    turn = np.arccos(dot[valid])
    step = np.maximum((n1[valid] + n2[valid]) * 0.5, 1.0)
    return float(np.mean(turn / step))


def extract_region_contours(mask: np.ndarray, min_area_px: float = 64.0) -> list[np.ndarray]:
    boundary = mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    labels, count = ndi.label(boundary, structure=np.ones((3, 3), dtype=bool))
    contours: list[np.ndarray] = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if xs.size < min_area_px:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        angles = np.arctan2(ys - cy, xs - cx)
        order = np.argsort(angles)
        points = np.column_stack([xs[order], ys[order]]).astype(np.float32)
        if len(points) < 8:
            continue
        contours.append(points)
    return contours


def compute_boundary_tangent_directions(points: np.ndarray, closed: bool = True) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(0, dtype=np.float32)
    pts = points.astype(np.float32)
    prev_pts = np.roll(pts, 1, axis=0) if closed else np.vstack([pts[:1], pts[:-1]])
    next_pts = np.roll(pts, -1, axis=0) if closed else np.vstack([pts[1:], pts[-1:]])
    delta = next_pts - prev_pts
    return np.degrees(np.arctan2(delta[:, 1], delta[:, 0])).astype(np.float32)


def _draw_polyline(mask_shape: tuple[int, int], points: np.ndarray, width: int, closed: bool) -> np.ndarray:
    canvas = Image.new("L", (mask_shape[1], mask_shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    xy = [(float(x), float(y)) for x, y in points]
    if closed and len(xy) > 2:
        xy = xy + [xy[0]]
    draw.line(xy, fill=255, width=max(1, width), joint="curve")
    return np.array(canvas, dtype=np.uint8)


def _occupancy_overlap(points: np.ndarray, occupancy: np.ndarray, spacing_px: float, closed: bool) -> bool:
    if not np.any(occupancy):
        return False
    lane_mask = _draw_polyline(occupancy.shape, points, int(round(spacing_px * 0.45)), closed)
    return bool(np.any((lane_mask > 0) & occupancy))


def _mark_occupancy(points: np.ndarray, occupancy: np.ndarray, spacing_px: float, closed: bool) -> None:
    lane_mask = _draw_polyline(occupancy.shape, points, int(round(spacing_px * 0.60)), closed)
    occupancy |= lane_mask > 0


def _lane_score(length_px: float, curvature: float, dist_level: float) -> float:
    return length_px * (1.0 / (1.0 + curvature * 28.0)) * (1.0 + dist_level * 0.01)


def _sample_vector(field_x: np.ndarray, field_y: np.ndarray, x: float, y: float) -> tuple[float, float]:
    h, w = field_x.shape
    if x < 0.0 or y < 0.0 or x >= w - 1 or y >= h - 1:
        return 0.0, 0.0
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    tx = x - x0
    ty = y - y0
    fx = (
        field_x[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + field_x[y0, x1] * tx * (1.0 - ty)
        + field_x[y1, x0] * (1.0 - tx) * ty
        + field_x[y1, x1] * tx * ty
    )
    fy = (
        field_y[y0, x0] * (1.0 - tx) * (1.0 - ty)
        + field_y[y0, x1] * tx * (1.0 - ty)
        + field_y[y1, x0] * (1.0 - tx) * ty
        + field_y[y1, x1] * tx * ty
    )
    return float(fx), float(fy)


def _trace_streamline(
    seed_x: float,
    seed_y: float,
    tangent_x: np.ndarray,
    tangent_y: np.ndarray,
    mask: np.ndarray,
    band: np.ndarray,
    direction: float,
    step_px: float,
    max_steps: int,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    x = float(seed_x)
    y = float(seed_y)
    prev: tuple[float, float] | None = None
    for _ in range(max_steps):
        ix = int(round(x))
        iy = int(round(y))
        if ix < 0 or iy < 0 or iy >= mask.shape[0] or ix >= mask.shape[1] or not mask[iy, ix] or not band[iy, ix]:
            break
        points.append((x, y))
        vx, vy = _sample_vector(tangent_x, tangent_y, x, y)
        norm = math.hypot(vx, vy)
        if norm < 1e-4:
            break
        vx /= norm
        vy /= norm
        vx *= direction
        vy *= direction
        if prev is not None and vx * prev[0] + vy * prev[1] < 0.0:
            vx *= -1.0
            vy *= -1.0
        if prev is not None:
            vx = prev[0] * 0.72 + vx * 0.28
            vy = prev[1] * 0.72 + vy * 0.28
            norm = math.hypot(vx, vy)
            if norm < 1e-4:
                break
            vx /= norm
            vy /= norm
        nx = x + vx * step_px
        ny = y + vy * step_px
        if len(points) > 8 and math.hypot(nx - points[0][0], ny - points[0][1]) < step_px * 0.9:
            break
        prev = (vx, vy)
        x = nx
        y = ny
    return points


def _trace_band_lane(
    seed_x: float,
    seed_y: float,
    tangent_x: np.ndarray,
    tangent_y: np.ndarray,
    mask: np.ndarray,
    band: np.ndarray,
    step_px: float,
    max_steps: int,
) -> np.ndarray:
    backward = _trace_streamline(seed_x, seed_y, tangent_x, tangent_y, mask, band, -1.0, step_px, max_steps)
    forward = _trace_streamline(seed_x, seed_y, tangent_x, tangent_y, mask, band, 1.0, step_px, max_steps)
    merged = list(reversed(backward[1:])) + forward
    if len(merged) < 3:
        merged = backward + forward[1:]
    if len(merged) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(merged, dtype=np.float32)


def generate_region_lanes(
    region: str,
    mask: np.ndarray,
    words: str | list[str],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spacing_px = float(config["spacing_px"])
    min_length_px = float(config["min_length_px"])
    curvature_limit = float(config["curvature_limit"])
    min_width_px = float(config["min_width_px"])
    max_lanes = int(config["max_lanes"])

    dist = ndi.distance_transform_edt(mask)
    max_dist = float(dist.max())
    smoothed_dist = ndi.gaussian_filter(dist.astype(np.float32), sigma=2.0)
    grad_y, grad_x = np.gradient(smoothed_dist)
    tangent_x = -grad_y
    tangent_y = grad_x
    tangent_norm = np.hypot(tangent_x, tangent_y)
    tangent_norm[tangent_norm < 1e-4] = 1.0
    tangent_x = tangent_x / tangent_norm
    tangent_y = tangent_y / tangent_norm

    region_contours = extract_region_contours(mask)
    contour_lengths = [_contour_length(points, True) for points in region_contours]
    contour_tangent_samples = int(sum(len(points) for points in region_contours))

    diagnostics: dict[str, Any] = {
        "region": region,
        "boundary_contours": len(region_contours),
        "boundary_tangent_samples": contour_tangent_samples,
        "boundary_mean_length_px": float(np.mean(contour_lengths)) if contour_lengths else 0.0,
        "candidate_lanes": 0,
        "generated_lanes": 0,
        "discarded_short_lanes": 0,
        "discarded_curvy_lanes": 0,
        "discarded_spacing_lanes": 0,
        "mean_lane_length_px": 0.0,
        "mean_lane_curvature": 0.0,
        "short_lane_ratio": 0.0,
        "lane_pixel_coverage": 0.0,
    }
    if max_dist < min_width_px:
        return [], diagnostics

    occupancy = np.zeros(mask.shape, dtype=bool)
    candidates: list[tuple[float, dict[str, Any]]] = []
    level = max(spacing_px * 0.75, min_width_px)
    lane_index = 0
    while level < max_dist:
        band = mask & (np.abs(dist - level) <= max(2.5, spacing_px * 0.42))
        labels, count = ndi.label(band, structure=np.ones((3, 3), dtype=bool))
        for label in range(1, count + 1):
            component = labels == label
            if int(component.sum()) < max(24, int(min_length_px * 0.35)):
                continue
            diagnostics["candidate_lanes"] += 1
            ys, xs = np.where(component)
            center_idx = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
            seed_x = float(xs[center_idx])
            seed_y = float(ys[center_idx])
            traced = _trace_band_lane(
                seed_x,
                seed_y,
                tangent_x,
                tangent_y,
                mask,
                component,
                step_px=max(3.0, spacing_px * 0.34),
                max_steps=max(80, int(min_length_px // 2)),
            )
            if len(traced) < 4:
                diagnostics["discarded_short_lanes"] += 1
                continue
            smoothed = _smooth_contour(traced, False)
            sampled = _resample_polyline(smoothed, max(spacing_px * 0.35, 8.0), False)
            length_px = _contour_length(sampled, False)
            if length_px < min_length_px:
                diagnostics["discarded_short_lanes"] += 1
                continue
            curvature = _mean_curvature(sampled, False)
            if curvature > curvature_limit:
                diagnostics["discarded_curvy_lanes"] += 1
                continue
            tangents = compute_boundary_tangent_directions(sampled, False).tolist()
            candidates.append(
                (
                    _lane_score(length_px, curvature, level),
                    {
                        "id": f"{region}_auto_{lane_index}",
                        "region": region,
                        "points_px": [[float(x), float(y)] for x, y in sampled],
                        "tangents_deg": [float(angle) for angle in tangents],
                        "words": words,
                        "spacing": float(spacing_px),
                        "size": int(config["size"]),
                        "alpha": int(config["alpha"]),
                        "closed": False,
                        "source": "auto",
                        "distance_level_px": float(level),
                        "length_px": float(length_px),
                        "mean_curvature": float(curvature),
                    },
                )
            )
            lane_index += 1
        level += spacing_px

    candidates.sort(key=lambda item: item[0], reverse=True)
    lanes: list[dict[str, Any]] = []
    for _, lane in candidates:
        points = np.array(lane["points_px"], dtype=np.float32)
        if _occupancy_overlap(points, occupancy, float(lane["spacing"]), bool(lane["closed"])):
            diagnostics["discarded_spacing_lanes"] += 1
            continue
        _mark_occupancy(points, occupancy, float(lane["spacing"]), bool(lane["closed"]))
        lanes.append(lane)
        if len(lanes) >= max_lanes:
            break

    lengths = [float(lane["length_px"]) for lane in lanes]
    curvatures = [float(lane["mean_curvature"]) for lane in lanes]
    diagnostics["generated_lanes"] = len(lanes)
    diagnostics["mean_lane_length_px"] = float(np.mean(lengths)) if lengths else 0.0
    diagnostics["mean_lane_curvature"] = float(np.mean(curvatures)) if curvatures else 0.0
    if diagnostics["candidate_lanes"]:
        diagnostics["short_lane_ratio"] = float(diagnostics["discarded_short_lanes"] / diagnostics["candidate_lanes"])
    diagnostics["lane_pixel_coverage"] = float(np.mean(occupancy))
    return lanes, diagnostics


def generate_flow_layout(
    maps: dict[str, np.ndarray],
    flow_config: dict[str, Any] | None,
    profile_words: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    config = flow_config or {}
    regions = config.get("regions", {})
    merged_regions = _merge_region_config(regions)
    enabled_regions = [str(region) for region in config.get("enabled_regions", merged_regions.keys())]
    lanes: list[dict[str, Any]] = []
    region_diagnostics: dict[str, dict[str, Any]] = {}
    subject = maps["subject"]
    coverage_mask = np.zeros(subject.shape, dtype=bool)
    for region in enabled_regions:
        if region not in maps:
            continue
        words = region if profile_words and region in profile_words else region
        region_lanes, diagnostics = generate_region_lanes(region, maps[region], words, merged_regions[region])
        lanes.extend(region_lanes)
        region_diagnostics[region] = diagnostics
        for lane in region_lanes:
            points = np.array(lane["points_px"], dtype=np.float32)
            lane_mask = _draw_polyline(subject.shape, points, max(1, int(round(float(lane["spacing"]) * 0.45))), bool(lane.get("closed", False)))
            coverage_mask |= lane_mask > 0

    lengths = [float(lane["length_px"]) for lane in lanes]
    curvatures = [float(lane["mean_curvature"]) for lane in lanes]
    candidate_total = sum(int(diag["candidate_lanes"]) for diag in region_diagnostics.values())
    short_total = sum(int(diag["discarded_short_lanes"]) for diag in region_diagnostics.values())
    diagnostics = {
        "enabled": bool(config.get("enabled", False)),
        "generated_lanes": len(lanes),
        "candidate_lanes": candidate_total,
        "mean_lane_length_px": float(np.mean(lengths)) if lengths else 0.0,
        "mean_lane_curvature": float(np.mean(curvatures)) if curvatures else 0.0,
        "short_lane_ratio": float(short_total / candidate_total) if candidate_total else 0.0,
        "lane_coverage_subject": float(np.logical_and(coverage_mask, subject).sum() / max(int(subject.sum()), 1)),
        "regions": region_diagnostics,
    }
    return {"lanes": lanes, "diagnostics": diagnostics}
