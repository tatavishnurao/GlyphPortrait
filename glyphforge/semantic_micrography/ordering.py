from __future__ import annotations

from glyphforge.semantic_micrography.lanes import TextLane


def _lane_key(lane: TextLane) -> tuple[str, float, float, float]:
    xs = [point[0] for point in lane.points]
    ys = [point[1] for point in lane.points]
    min_y = min(ys) if ys else 0.0
    min_x = min(xs) if xs else 0.0
    mean_y = sum(ys) / max(len(ys), 1)
    return lane.region, min_y, min_x, mean_y


def orient_lane_for_readability(lane: TextLane) -> TextLane:
    if len(lane.points) < 2:
        return lane
    first = lane.points[0]
    last = lane.points[-1]
    reverse = False
    if abs(last[0] - first[0]) >= abs(last[1] - first[1]):
        reverse = last[0] < first[0]
    else:
        reverse = last[1] < first[1]
    if not reverse:
        return lane
    return TextLane(
        id=lane.id,
        region=lane.region,
        points=list(reversed(lane.points)),
        length_px=lane.length_px,
        mean_curvature=lane.mean_curvature,
        closed=lane.closed,
        order_index=lane.order_index,
        source=lane.source,
    )


def order_lanes(lanes: list[TextLane]) -> list[TextLane]:
    ordered: list[TextLane] = []
    for index, lane in enumerate(sorted((orient_lane_for_readability(lane) for lane in lanes), key=_lane_key)):
        ordered.append(
            TextLane(
                id=lane.id,
                region=lane.region,
                points=lane.points,
                length_px=lane.length_px,
                mean_curvature=lane.mean_curvature,
                closed=lane.closed,
                order_index=index,
                source=lane.source,
            )
        )
    return ordered
