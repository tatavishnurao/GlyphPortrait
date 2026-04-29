# Experiment 005: Reference Wallpaper Decomposition

## Goal

Decompose the Jordan-style wallpaper into explicit layers before further
generalization work.

## Decomposition Layers

1. Black background / negative space
2. Subject silhouette from non-black pixels
3. Red-dominant jersey mask inside the subject
4. Face/body mask as subject minus jersey
5. Luminance map and edge map for tone guidance

## Why Generic Single-Pass Fails

- One mask and one pass cannot capture different design rules for face and jersey.
- Anchor words in the reference are intentionally placed, not random.
- Composition requires right-side subject + left-side slogan.

## Outputs

The recreation script saves:

- `extracted_subject_mask.png`
- `extracted_jersey_mask.png`
- `extracted_face_mask.png`
- `target_luminance.png`
- `target_edges.png`
