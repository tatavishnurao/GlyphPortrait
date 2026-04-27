# GlyphForge Algorithm Notes

This document explains the current MVP algorithm used for local typographic
portrait generation.

## 1. Segmentation fallback chain

GlyphForge uses a practical fallback strategy so the pipeline still works
without heavyweight model dependencies:

1. `rembg` foreground extraction (when runtime backend is available).
2. OpenCV GrabCut with a centered rectangle initialization.
3. Otsu threshold fallback on the grayscale portrait.

The first successful non-empty mask is used.

## 2. Mask cleanup

Raw masks are post-processed to improve contour quality:

1. Convert to strict binary (`0` or `255`).
2. Morphological open to remove small isolated noise.
3. Morphological close to fill small holes.
4. Light Gaussian blur + threshold for stable edges.

This improves text placement acceptance and silhouette coherence.

## 3. Occupancy-grid collision detection

Layout uses a pixel occupancy grid (`H x W`):

1. Each candidate word computes a text bounding box.
2. Candidate is rejected if any occupancy pixel is already set in that box.
3. Accepted placement marks its bounding region in occupancy.

This keeps words readable and avoids overlap artifacts.

## 4. Weighted word priority

Words are sorted by descending importance weight:

- Earlier words and repeated words get higher effective weights.
- Weight controls nominal font size (before small random jitter).
- High-value terms get larger and are attempted earlier.

This makes key terms more visually prominent.

## 5. Grayscale/edge-guided placement

Placement is not purely uniform random:

1. Compute darkness map: `(255 - gray) / 255`.
2. Compute edge map with Canny, then smooth it.
3. Blend both maps into an importance map:
   - darkness captures structure in shadowed regions
   - edges capture face/contour detail
4. Multiply by the subject mask and smooth.
5. Sample candidate centers from the normalized importance distribution
   (deterministic under fixed seed), then convert to top-left positions.

Result: denser text in visually meaningful portrait regions and less fill in
flat areas.

## 6. Limitations

- Axis-aligned text only (no rotation/curved baseline yet).
- Dense regions can still underfill when long words dominate.
- Fallback threshold segmentation can fail on complex backgrounds.
- Current renderer does not explicitly preserve landmarks (eyes/nose/mouth).

## 7. Future improvements

- Multi-angle layout and local orientation fields.
- Face landmark-aware importance boosting.
- Connected-component aware packing constraints.
- Smarter phrase splitting for long terms.
- Optional GPU segmentation and post-stylization hooks.
