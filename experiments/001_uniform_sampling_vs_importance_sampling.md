# Experiment 001: Uniform Sampling vs Importance Sampling

## Goal

Check whether the renderer still looks like a generic word cloud when text is
sampled uniformly inside the mask.

## Input

- Image: `reference_img/Michael-Jordan-Wallpaper-Desktop-1.jpg`
- Ratio: `16:9`
- Resolution: `960x540`
- Seed: `42`
- Theme: `monochrome_dark`

## Comparison

- Uniform inside mask:
  - [examples/evolution/02_uniform_sampling_bad.png](../examples/evolution/02_uniform_sampling_bad.png)
- Importance-guided:
  - [examples/evolution/03_importance_sampling_better.png](../examples/evolution/03_importance_sampling_better.png)

## What failed

- Uniform sampling fills the silhouette, but face structure gets muddy fast.
- Cheeks, forehead, and jersey all compete for the same word density.
- The result reads as "word cloud in a cutout" instead of portrait typography.

## What changed next

- I switched candidate sampling to an importance map:
  - subject mask
  - darkness map
  - edge map
- This does not solve likeness fully, but it pushes more text into meaningful
  facial structure.
