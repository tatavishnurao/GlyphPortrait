# 009 Face Reconstruction Notes

Output: `examples/reference_recreation/face_study_v2.png`.

Side-by-side: `examples/reference_recreation/face_study_v2_side_by_side.png`.

## What Was Observed

- The face is built from dense grayscale text, not from clean continuous shading.
- Major anatomy anchors are concentrated in the forehead, cheek, jaw, and mouth/chin.
- The eye sockets must stay dark even though they contain text.
- The nose and mouth need bright contour words; the cheek and jaw need larger structural words.

## What That Implies

- A face-only renderer needs separate passes for dark microtext, surface words, contour words, and manual anchors.
- Anchor placement should be art-directed. Pure random text fill cannot place `MVP`, `Air Jordan`, `Dedication`, or `Dominance` correctly.
- Luminance modulation is required after placement so bright text does not destroy the eye sockets.

## What Was Built

- `scratch/recreate_jordan_forensic_v4.py` builds a face/neck mask from the reference silhouette and red jersey exclusion.
- The face pass uses:
  - low-alpha microtext for dark mass,
  - tone-sampled readable filler words,
  - contour-tangent words for nose/mouth/jaw,
  - explicit anatomy anchors for the major visible phrases.
- The resulting crop is exported from the full v4 reconstruction so face-only and full-poster behavior remain consistent.

## What Still Fails

- Forehead anchors are too bright and broad compared with the tighter reference arcs.
- The reference has more curved/radial crown text than the current straight rotated words.
- Eye socket protection works directionally, but the generated texture lacks the fine nested dark bands of the source.
- The nose bridge needs more hand-authored contour phrases rather than random contour sampling alone.

## Next Tuning Step

Replace the current crown/forehead random pass with 8-12 explicit curved bands, then add protected eye masks that reduce anchor alpha inside the brow and sockets.
