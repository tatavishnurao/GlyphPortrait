# Reference Density Map

Generated visual: `reference_analysis/reference_density_map.png`.

## Dense Regions

- Forehead/crown: very dense, with curved bands of Tier 2-4 white/gray text. The top skull shape is almost entirely typography.
- Eye sockets: dense but dark. Text remains present, but contrast is deliberately suppressed to preserve the eyes.
- Cheek and mouth: medium-to-high density with brighter contour words on nose, cheekbone, lip, and chin ridges.
- Neck: dense vertical/curved gray text; the neck column is mostly shadow with bright edge words.
- Red jersey chest: extremely dense. Red words create the fill, white/cream words create seam and number highlights.
- Right shoulder: dense gray/white vertical texture, with a few large readable shoulder anchors.

## Sparse Regions

- Black background: almost entirely silent except for the centered phrase `change the game.`.
- Deep interior shadows: not empty, but near-micro and low-contrast. The important distinction is dark density rather than no text.
- Extreme clipped silhouette edges: words are partially visible and cut sharply.

## Bright Regions

- Crown top rim and left skull edge.
- Nose bridge, nostril, upper lip, chin highlights.
- Jersey number/trim and right shoulder highlights.

## Dark Regions

- Eye sockets, center brow, under cheek, underside of jaw, neck center.
- Jersey lower folds and red shadow bands.

## Renderer Implications

- Density cannot be uniform. Fill probability and alpha must vary by anatomical region and luminance.
- Dark regions still require text; they should use low-alpha dark gray microtext instead of blank black.
- Bright ridges need a separate contour/edge pass so highlights stay text-shaped rather than blurred.
- The negative space must be protected. Random text must never leak into the black field except the deliberate caption.
