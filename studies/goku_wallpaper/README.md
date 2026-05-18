# Goku Wallpaper Reference-Stencil Study

This folder contains a target-aware semantic word-tribute reconstruction study
for one Goku reference. It follows the same study pattern as the Jordan
reference-stencil work, but uses the reusable `glyphforge.recipes.word_tribute`
recipe and a color-region profile.

This is not a generic anime generator. The study profile is intentionally
reference-specific: it isolates Goku, replaces the background with black tribute
poster styling, and reconstructs the subject using region-specific typography.

## Reproduce

From the repository root:

```bash
python studies/goku_wallpaper/recreate_stencil.py \
  --input reference_img/goku-sky-reference.jpg \
  --out-dir studies/goku_wallpaper/outputs
```

The reusable recipe can also be called directly:

```bash
python -m glyphforge.recipes.word_tribute \
  --input reference_img/goku-sky-reference.jpg \
  --profile studies/goku_wallpaper/goku_profile.json \
  --background black \
  --out-dir studies/goku_wallpaper/outputs
```

The reference image is expected to stay local. `reference_img/goku*.jpg` is
ignored so source artwork is not committed by accident.

## Outputs

The one-shot pipeline writes canonical artifacts:

- `outputs/current_best.png`
- `outputs/current_best_side_by_side.png`
- `outputs/current_best_metrics.json`
- `outputs/mask_panel.png`
- `outputs/anchor_overlay.png`
- `outputs/lane_overlay.png`

Per-mask diagnostics are written under `outputs/diagnostics/`.

## Pipeline

1. Load the full reference image and preserve canvas registration.
2. Build masks for subject, hair, skin/face, orange gi, blue undershirt,
   outline/shadow, and sky background.
3. Generate region-specific typography texture layers from `goku_profile.json`.
4. Clip text to masks and modulate it by local reference color, luminance, and
   edge structure.
5. Add integrated anchors and contour-lane words for hair spikes, brow, cheek,
   jaw, neck, collar, gi folds, and shoulder curves.
6. Composite the typographic subject over a black tribute-poster background with
   a small rim/edge visibility pass.
7. Save diagnostics and reconstruction metrics.

## Current Direction

The public study exposes one configured pipeline and one canonical result, not a
sequence of versioned visual experiments. Internal tuning should happen through
`goku_profile.json` or targeted edits to `glyphforge.recipes.word_tribute`, then
the same canonical output names should be regenerated.
