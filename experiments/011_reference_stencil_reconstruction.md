# 011 Reference-Stencil Reconstruction

## Summary

The Jordan wallpaper reconstruction now has a locked current-best result:
`stencil_v5_final.png`. This pass should be treated as the canonical artifact
for the current study, not as another tuning checkpoint.

Final v5 metrics:

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
seam_artifact_score: 2.88
```

## What Failed First

The initial renderer family tried to make words act as both material and
structure:

```text
place words -> hope portrait emerges
```

That approach repeatedly produced gray lower-face slabs, weak facial structure,
and mouth banding. The optimizer confirmed the issue was architectural, not just
parameter selection: the 50 + 50 run produced no eligible results, and the best
near misses continued to fail gray-slab and mouth-banding gates.

## Pivot

The successful pivot was reference-stencil compositing:

```text
generate typography texture sheets
clip them to reference-derived masks
modulate them through reference luminance and shadow maps
add hand-authored anchors and contour lanes
apply final corrections
```

This split the work correctly:

- words provide texture and identity
- the reference provides shape, tone, shadows, and highlights
- manual anchors provide poster hierarchy

## Iteration Notes

`stencil v1` proved the architecture. It removed the lower-face gray slab by
using the reference luminance as a stencil, but the face was too dark and the
jersey was still synthetic.

`stencil v2` calibrated registration, dual masks, protected dark zones, jersey
red sampling, and anchor integration. Jersey quality improved substantially, but
the face became too stencil-carved.

`stencil v3` added face-detail recovery, edge-ring masks, and soft dark-zone
visibility. This improved face luma and edge overlap without breaking jersey
metrics.

`stencil v4` added hand-authored facial anchors and contour lanes. This was the
first pass where the full poster had strong facial structure while preserving
the v3/v4 jersey.

`stencil v5 face-only` focused only on the face crop. It strengthened tiered
anchors and lane-based typography around the skull, brow, nose, cheek, mouth,
jaw, and neck.

`stencil v5 full` merged the v5 face into the v4 full poster using a feathered
face insertion mask that excluded the jersey/collar region. This preserved the
strong jersey while adopting the improved face.

## Current Canonical Artifacts

The presentable study artifacts are copied into:

```text
studies/jordan_wallpaper/outputs/current_best.png
studies/jordan_wallpaper/outputs/current_best_side_by_side.png
studies/jordan_wallpaper/outputs/current_best_metrics.json
```

The reproducibility entrypoint is:

```bash
python studies/jordan_wallpaper/recreate_stencil.py
```

## Next Recommendation

Stop scratch visual tuning for now. The next useful work is packaging and
documentation:

- keep `stencil_v5_final.png` as the current best artifact
- leave scratch files intact for auditability
- use `studies/jordan_wallpaper/recreate_stencil.py` as the study entrypoint
- only resume visual iteration if a clear, targeted visual problem is selected
