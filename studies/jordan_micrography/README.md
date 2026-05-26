# Jordan Semantic Micrography Study

This is the fresh semantic digital micrography implementation for the Jordan
reference image. It is separate from the older target-aware stencil study in
`studies/jordan_wallpaper/`.

The goal here is readable streamlines and SVG-first output. The portrait is
constructed from generated text lanes clipped to semantic regions; the PNG is a
preview/export of the SVG master, not the primary artifact.

The old Jordan wallpaper study remains a historical and quality reference. It
is intentionally not used as the foundation for this engine.

## Run

```bash
python scripts/render_semantic_micrography.py \
  --input reference_img/Michael-Jordan-Wallpaper-Desktop-1.jpg \
  --profile studies/jordan_micrography/jordan_profile.json \
  --out-dir studies/jordan_micrography/outputs \
  --background black \
  --style tribute_dark
```

Canonical outputs:

- `outputs/current_best.svg`
- `outputs/current_best.png`
- `outputs/current_best_metrics.json`
- `outputs/regions_panel.png`
- `outputs/lane_overlay.svg`
- `outputs/lane_overlay.png`
- `outputs/debug_summary.json`
