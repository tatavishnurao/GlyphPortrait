# Jordan Wallpaper Reference-Stencil Study

This folder contains the target-aware Michael Jordan typography reconstruction
study. It is intentionally separate from the reusable `glyphforge/` package:

- `glyphforge/` is the generic portrait typography engine.
- `studies/jordan_wallpaper/` is a forensic reconstruction workflow for one
  fixed reference image.

The current best artifact is the locked stencil v5 result:

- `outputs/current_best.png`
- `outputs/current_best_side_by_side.png`
- `outputs/current_best_metrics.json`

## Current Metrics

Latest full-poster v5 metrics:

```text
mae_full_rgb: 14.79
mae_subject_rgb: 29.12
mae_face_rgb: 22.71
mae_jersey_rgb: 28.84
face_luma_mae: 22.50
jersey_luma_mae: 27.53
mae_jersey_red_channel: 38.85
edge_overlap_face: 0.208
red_mask_iou: 0.875
gray_slab_penalty: 0.0
mouth_banding_penalty: 0.072
protected_dark_zone_fill_ratio: 0.053
floating_fragment_count: 0
```

## Reproduce

From the repository root:

```bash
python studies/jordan_wallpaper/recreate_stencil.py
```

To only refresh the canonical study copies from existing `examples/` artifacts:

```bash
python studies/jordan_wallpaper/recreate_stencil.py --skip-render
```

The wrapper delegates to the locked scratch v5 renderer to preserve visual
behavior exactly, then copies the canonical artifacts into
`studies/jordan_wallpaper/outputs/`.

## Progression

1. Naive text rendering established that generic word placement did not recover
   the target portrait.
2. Optimizer runs exposed stable failure modes: gray lower-face slabs and mouth
   banding.
3. Reference-stencil v1 split texture from structure by using the reference as a
   luminance and shape stencil.
4. Stencil v2 calibrated registration, dual masks, dark-zone control, and jersey
   red behavior.
5. Stencil v3 added face-detail recovery and soft protected dark-zone handling.
6. Stencil v4 added art-directed facial anchors and contour lanes.
7. Face-only v5 improved typographic anatomy, then full v5 merged that face into
   the frozen v4 poster while preserving the jersey.

## Scope

This is not a general portrait generator. The scripts here document the
reference-specific strategy that produced the current Jordan wallpaper study
artifact.
