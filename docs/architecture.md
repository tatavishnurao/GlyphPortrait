# GlyphForge Architecture

This repository is intentionally split into reusable engine code and
target-specific study code.

## Repository Roles

- `glyphforge/`
  Reusable core engine for portrait preprocessing, mask handling, typography
  layout, and generic rendering.
- `studies/jordan_wallpaper/`
  Intentionally overfit reference-recreation study for Jordan-style wallpaper
  reverse engineering.
- `experiments/`
  Iteration notes, failures, and tradeoff logs.
- `examples/`
  Generated artifacts and outputs.
- `scripts/`
  Thin wrappers for local execution of workflows.

## Core Engine Pipeline (`glyphforge/`)

1. Input portrait loaded via Gradio/CLI.
2. Preprocessing resizes and crops to target output ratio.
3. Segmentation attempts:
   - `rembg` (when available)
   - OpenCV GrabCut fallback
   - Otsu threshold fallback
4. Mask cleanup with morphology + blur thresholding.
5. Keyword parser converts text to weighted words.
6. Layout engine places words with:
   - priority by weight
   - grayscale/edge-guided importance sampling
   - deterministic seed behavior
   - bbox-in-mask and occupancy-grid collision checks
7. Renderer applies theme-aware text coloring.
8. Export writes high-resolution PNG.

## Study Pipeline (`studies/jordan_wallpaper/`)

The study pipeline is not a generic feature path. It is a diagnostic,
target-aware reconstruction track:

1. Decompose reference image into subject/face/jersey regions.
2. Compose right-side 16:9 layout with left negative space.
3. Render multi-pass typography (micro, structure, jersey, anchors, slogan).
4. Save decomposition masks and reconstruction metrics.

## Design Intent

- Keep `glyphforge/` small and reusable.
- Keep target-specific logic in `studies/`.
- Keep experiments and visual artifacts explicit so iteration history is
  auditable.
