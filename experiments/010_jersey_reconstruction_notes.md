# 010 Jersey Reconstruction Notes

Output: `examples/reference_recreation/jersey_study_v1.png`.

Side-by-side: `examples/reference_recreation/jersey_study_v1_side_by_side.png`.

## What Was Observed

- The jersey is a separate red typographic material, not a recolored face pass.
- Red text carries most of the chest mass. White/cream text defines trim, highlights, and the `23` number zone.
- The collar and shoulder seams are long bright text strips.
- The red field uses stronger diagonal/vertical orientations than the face.

## What That Implies

- Jersey reconstruction needs its own mask, palette, word inventory, density map, and orientation grammar.
- The `23` and trim should be art-directed instead of emerging from random fill.
- Red shadows should use deep red text, not black or gray face words.

## What Was Built

- `scratch/recreate_jordan_forensic_v4.py` extracts a red jersey mask using color dominance inside the subject silhouette.
- The jersey pass uses:
  - red microtext for body fill,
  - readable red identity words,
  - contour-oriented text along internal edges,
  - manual anchors for `Named to the All-Star Team`, `NBA Most Valuable Player`, `Chicago Bulls`, `BULLS`, and `23`.
- The crop is exported as a controlled jersey study from the same full-poster v4 render.

## What Still Fails

- The white `23` geometry is too coarse and too label-like; the reference integrates it into text and jersey folds.
- The current renderer uses a large `BULLS` anchor that is useful diagnostically but not yet faithful to the reference balance.
- Collar trim needs explicit curved/seam-following phrase bands.
- The right shoulder is still treated as a generic gray region; it needs separate shoulder anchors and stronger vertical text columns.

## Next Tuning Step

Build a dedicated jersey-number mask and seam masks, then render white text only inside those masks before compositing red fill underneath.
