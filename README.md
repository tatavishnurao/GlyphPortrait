# GlyphPortrait

Typographic portrait generator. Turns a portrait into poster/wallpaper
artwork built from real, readable words placed inside the subject silhouette.

Pipeline: mask subject -> build importance map -> place words with
collision-aware layout -> render themed PNG.

## Aim

Reconstruct a recognizable portrait using nothing but typography, where
every placed word remains legible and the overall silhouette,
luminance, and edge structure match the reference.

## Core Engine

- `glyphforge/` — reusable pipeline: segmentation, importance map, word
  placement layout engine, theme rendering.
- Gradio app (`app.py`), CLI (`cli.py`), export presets (`1:1`, `4:5`,
  `16:9`, `9:16`).

## Studies

### `studies/jordan_wallpaper/`

Reference-stencil reconstruction of a Michael Jordan wallpaper.

Strategy: use the reference itself as a dual stencil. The reference
luminance map and edge map guide every word placement. Text passes are
split by region (face, jersey), each with region-specific word sets and
luminance-driven color picking. Art-directed anchors are placed at fixed
facial coordinates to recover key structural lines (brow, jaw, cheek).
A composited right-aligned subject canvas matches the original wallpaper
composition.

### `studies/goku_wallpaper/`

Reference-stencil reconstruction of a Goku portrait for black-background
tribute posters.

Strategy: multi-mask region decomposition. A JSON profile defines
semantic regions (hair, skin, neck, orange gi, blue undershirt,
outline/shadow) with region-specific word sets. Anchors are placed at
exact normalized coordinates with per-word angle and alpha. Lanes
place words along curved paths following anatomical contours (hair
spikes, brow line, jawline, collar curve, gi folds, shoulder curve).
The subject is composited over a black tribute-poster background with
a rim visibility pass.

## Local Setup

```bash
uv venv inferenceimg
source inferenceimg/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
python app.py          # Gradio UI
python cli.py ...      # CLI
pytest -q              # tests
```

## Docs

- [docs/algorithm.md](docs/algorithm.md)
- [docs/limitations.md](docs/limitations.md)
- [docs/gpu-roadmap.md](docs/gpu-roadmap.md)
- [docs/roadmap.md](docs/roadmap.md)
