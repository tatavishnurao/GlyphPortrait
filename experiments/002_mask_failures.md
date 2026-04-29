# Experiment 002: Mask Failures

## Goal

Record where the current segmentation fallback works and where it breaks.

## Input

- Image: `reference_img/Michael-Jordan-Wallpaper-Desktop-1.jpg`
- Preview:
  - [examples/evolution/00_original_crop.png](../examples/evolution/00_original_crop.png)
  - [examples/evolution/01_mask.png](../examples/evolution/01_mask.png)

## What worked

- The current reference image is forgiving because the subject is already
  separated by a dark background.
- GrabCut fallback is usually good enough on this image even without a fully
  configured `rembg` runtime.

## What failed

- Loose shoulder and torso contours can still merge into background depending on
  crop ratio.
- Threshold fallback is fragile on cluttered or bright backgrounds.
- Small mask errors cascade into bad text placement because placement is
  hard-constrained by the silhouette.

## What changed next

- Keep mask preview visible in Gradio.
- Treat segmentation backend quality as the first real bottleneck, not a minor
  preprocessing detail.
