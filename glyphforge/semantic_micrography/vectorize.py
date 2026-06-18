from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ContourPath:
    id: str
    region: str
    points: list[tuple[float, float]]
    closed: bool
    area_px: float


def vectorize_mask(region: str, mask: np.ndarray, min_area_px: float = 64.0, simplify_epsilon_px: float = 2.0) -> list[ContourPath]:
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    paths: list[ContourPath] = []
    for idx, contour in enumerate(contours):
        area = float(abs(cv2.contourArea(contour)))
        if area < min_area_px:
            continue
        approx = cv2.approxPolyDP(contour, epsilon=simplify_epsilon_px, closed=True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue
        paths.append(
            ContourPath(
                id=f"{region}_contour_{idx}",
                region=region,
                points=[(float(x), float(y)) for x, y in pts],
                closed=True,
                area_px=area,
            )
        )
    paths.sort(key=lambda path: path.area_px, reverse=True)
    return paths


def vectorize_regions(masks: dict[str, np.ndarray], min_area_px: float = 64.0) -> dict[str, list[ContourPath]]:
    return {
        region: vectorize_mask(region, mask, min_area_px=min_area_px)
        for region, mask in masks.items()
        if region != "subject"
    }
