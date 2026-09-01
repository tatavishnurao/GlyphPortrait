# Architecture diagrams

## What the system does
GlyphPortrait is a semantic digital micrography engine that reconstructs a person portrait using readable user-provided words. It supports a Gradio app, a CLI, and SVG/PNG output paths.

## Problem addressed
The system preserves portrait structure while distributing meaningful text along coherent visual geometry rather than treating the task as generic image generation.

## Overview
[System overview](system-overview.svg) shows portrait and word inputs flowing through preprocessing/masks, region and contour geometry, streamline generation, text placement, and raster/SVG composition.

## Critical path
[Critical path](critical-path.svg) shows the three-way constraint and the implementation bridge: image-derived masks/regions guide vectorized lanes, then words are placed on text paths before rendering.

## Evidence used
README.md; docs/architecture.md, algorithm.md, semantic_micrography.md, limitations.md; app.py and cli.py; glyphforge/semantic_micrography/pipeline.py, preprocess.py, regions.py, vectorize.py, lanes.py, text_layout.py, render_svg.py; glyphforge/image/masks.py, segment.py, export.py; tests/test_semantic_micrography.py, test_masks.py, test_layout.py, test_export.py, test_reference_recreation.py; scripts/render_semantic_micrography.py; outputs/vishnu_demo/lane_overlay.svg.

## Limitations and uncertainty
The diagrams describe the implemented semantic micrography path only. They do not assert a learned segmentation model, diffusion model, or other unverified model integration. Region names and exact placement behavior are implementation-dependent.

