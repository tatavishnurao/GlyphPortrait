# Semantic Digital Micrography

GlyphPortrait implements a practical semantic digital micrography pipeline for
raster portraits. Unlike the earlier target-aware stencil studies, this engine
treats coherent text streamlines as the core primitive. The input portrait is
converted into simplified subject/region masks and vector-like contours; text
lanes are generated inside those regions; user-provided semantic words are
placed along SVG text paths; and the result is exported as SVG and PNG.

## Thesis

Words are not decorative overlays. In this pipeline, words are the visible
material of the portrait. The renderer aims for readable, low-curvature text
streamlines that fill meaningful portrait regions instead of random text dust.

## Difference From Stencil Studies

The older Jordan wallpaper study is a target-aware reconstruction workflow. It
uses reference-derived masks, luminance, hand-authored anchors, and compositing
to recreate one fixed wallpaper. That work remains useful as quality evidence,
but it is not the foundation of this engine.

The semantic micrography engine is generic and SVG-first. It does not use
original subject pixels as the visible base. It extracts regions, generates
lanes, assigns words, and writes inspectable SVG text paths.

## Digital Micrography Inspiration

This is inspired by Digital Micrography: text should follow coherent curves and
become the image structure. The current implementation is a practical raster
approximation of Digital Micrography. It does not yet implement the full
graph-cut boundary condition optimization or full 2-RoSy vector field solver
from the paper.

## Algorithm Stages

1. Load the portrait, resize to a canvas, extract luminance and edge maps.
2. Build a generic subject mask, optionally using a user-provided mask.
3. Split the subject into semantic raster regions: dark/shadow, warm/skin,
   primary clothing, secondary clothing, highlights, and outline/edge.
4. Vectorize masks into simplified contours for clipping and diagnostics.
5. Generate region-internal lanes with dominant-direction scanlines and
   contour-following outline lanes.
6. Orient and order lanes for stable reading.
7. Assign hero, region, texture, and anchor words to SVG text paths.
8. Render `current_best.svg` as the master artifact.
9. Rasterize to `current_best.png` for preview/export and save metrics/debug
   outputs.

## Why SVG First

SVG keeps the result inspectable. Lanes are paths, words are real text, and the
document can be zoomed or audited without reverse-engineering raster pixels.
PNG export is secondary.

## Current Limitations

The lane field is heuristic. It uses PCA-oriented scanlines, spacing filters,
simple contour lanes, and curvature gates. It does not yet solve a global vector
field, does not optimize text readability against all boundaries, and may need
profile-specific masks for difficult backgrounds.

## Roadmap

- Add stronger subject segmentation options and mask refinement controls.
- Add true boundary-parallel offset lanes for complex contours.
- Add a 2-RoSy-like vector field approximation.
- Improve SVG/PDF typography controls and font embedding.
- Add visual regression tests for canonical studies.
