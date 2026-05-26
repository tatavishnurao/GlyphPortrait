from __future__ import annotations

import html
from pathlib import Path

from glyphforge.semantic_micrography.config import MicrographyStyleConfig
from glyphforge.semantic_micrography.text_layout import TextLayoutResult
from glyphforge.semantic_micrography.vectorize import ContourPath


def _path_d(points: list[tuple[float, float]], closed: bool = False) -> str:
    if not points:
        return ""
    chunks = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    chunks.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    if closed:
        chunks.append("Z")
    return " ".join(chunks)


def _contour_paths(contours: list[ContourPath]) -> str:
    return " ".join(_path_d(path.points, closed=True) for path in contours)


def render_master_svg(
    layout: TextLayoutResult,
    contours: dict[str, list[ContourPath]],
    canvas_size: tuple[int, int],
    style: MicrographyStyleConfig,
    out_path: Path,
) -> Path:
    w, h = canvas_size
    defs: list[str] = []
    body: list[str] = []
    for item in layout.text_paths:
        lane = item.lane
        defs.append(f'<path id="{html.escape(lane.id)}" d="{_path_d(lane.points, lane.closed)}" />')
    for region, paths in contours.items():
        d = _contour_paths(paths)
        if d:
            defs.append(f'<clipPath id="clip_{html.escape(region)}" clipPathUnits="userSpaceOnUse"><path d="{d}" /></clipPath>')

    body.append(f'<rect width="{w}" height="{h}" fill="rgb{style.background_color}" />')
    if style.edge_stroke:
        body.append('<g id="subtle_lane_edges" opacity="0.18">')
        for item in layout.text_paths:
            stroke_width = "1.2" if item.is_anchor else "0.7"
            body.append(f'<path d="{_path_d(item.lane.points, item.lane.closed)}" fill="none" stroke="{item.fill}" stroke-width="{stroke_width}" />')
        body.append("</g>")

    for region in sorted({item.lane.region for item in layout.text_paths}):
        clip = f' clip-path="url(#clip_{html.escape(region)})"' if contours.get(region) else ""
        body.append(f'<g id="region_{html.escape(region)}"{clip}>')
        normal_items = [item for item in layout.text_paths if item.lane.region == region and not item.is_anchor]
        anchor_items = [item for item in layout.text_paths if item.lane.region == region and item.is_anchor]
        for item in normal_items:
            escaped_text = html.escape(item.text)
            body.append(
                f'<text font-family="{html.escape(style.font_family)}" font-size="{item.font_size}" '
                f'font-weight="{item.font_weight}" fill="{item.fill}" opacity="{item.opacity:.2f}" '
                f'letter-spacing="{item.letter_spacing:.2f}" text-anchor="start">'
                f'<textPath href="#{html.escape(item.lane.id)}" startOffset="{item.start_offset}">{escaped_text}</textPath>'
                "</text>"
            )
        for item in anchor_items:
            escaped_text = html.escape(item.text.upper())
            body.append(
                f'<text font-family="{html.escape(style.font_family)}" font-size="{item.font_size}" '
                f'font-weight="{item.font_weight}" fill="{item.fill}" opacity="{item.opacity:.2f}" '
                f'letter-spacing="{item.letter_spacing:.2f}" text-anchor="middle">'
                f'<textPath href="#{html.escape(item.lane.id)}" startOffset="{item.start_offset}">{escaped_text}</textPath>'
                "</text>"
            )
        body.append("</g>")

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        "<desc>Semantic digital micrography: visible portrait material is SVG text along generated lanes.</desc>\n"
        "<defs>\n" + "\n".join(defs) + "\n</defs>\n" + "\n".join(body) + "\n</svg>\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return out_path


def render_lane_overlay_svg(
    layout: TextLayoutResult,
    canvas_size: tuple[int, int],
    out_path: Path,
) -> Path:
    w, h = canvas_size
    colors = {
        "dark_hair_or_shadow": "#68d8ff",
        "skin_or_warm": "#ffd166",
        "clothing_primary": "#ff4d4d",
        "clothing_secondary": "#a0c4ff",
        "highlight": "#ffffff",
        "outline_or_edge": "#80ffdb",
    }
    body = [f'<rect width="{w}" height="{h}" fill="#000000" />']
    region_counts: dict[str, int] = {}
    for item in layout.text_paths:
        lane = item.lane
        color = colors.get(lane.region, "#cccccc")
        region_counts[lane.region] = region_counts.get(lane.region, 0) + 1
        width = 3.8 if item.is_anchor else (2.4 if item.is_hero else 1.6)
        opacity = "0.98" if item.is_anchor else "0.86"
        body.append(f'<path d="{_path_d(lane.points, lane.closed)}" fill="none" stroke="{color}" stroke-width="{width:.1f}" opacity="{opacity}" />')
        if lane.points:
            x, y = lane.points[0]
            label = f'{lane.order_index} | {int(round(lane.length_px))}px'
            body.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="10" fill="{color}" opacity="0.95">{label}</text>')
    body.append('<rect x="14" y="14" width="385" height="178" fill="#060606" opacity="0.72" stroke="#444444" stroke-width="1" />')
    body.append('<text x="26" y="38" font-size="16" fill="#f0f0f0">Lane Overlay Diagnostics</text>')
    body.append('<text x="26" y="58" font-size="12" fill="#d0d0d0">thick: anchor lanes | medium: hero lanes</text>')
    y = 80
    for region in [
        "dark_hair_or_shadow",
        "skin_or_warm",
        "clothing_primary",
        "clothing_secondary",
        "highlight",
        "outline_or_edge",
    ]:
        count = region_counts.get(region, 0)
        color = colors.get(region, "#cccccc")
        body.append(f'<rect x="26" y="{y-10}" width="14" height="7" fill="{color}" />')
        body.append(f'<text x="48" y="{y-4}" font-size="12" fill="#ededed">{region}: {count}</text>')
        y += 20
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n' + "\n".join(body) + "\n</svg>\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return out_path
