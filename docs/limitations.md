# GlyphForge Limitations

This project is in a strong MVP state, but there are known technical limits.

## Current limitations

- Output quality depends heavily on mask quality.
- Text placement is currently axis-aligned (no rotation or curved baselines).
- Face landmarks are not explicitly preserved yet.
- Rendering is raster-based PNG output, not true vector typography export.
- Long words/phrases reduce packing efficiency and can lower density.
- Fallback segmentation can underperform on complex or cluttered backgrounds.

## Practical impact

- Great outputs are possible with strong input portraits and curated word lists.
- Harder backgrounds or long phrase-heavy inputs may need tuning:
  - lower font size range
  - adjust density
  - increase placement attempts
