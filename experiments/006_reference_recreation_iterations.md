# Experiment 006: Reference Recreation Iterations

## Goal

Run target-aware reconstruction iterations against a local Jordan-style reference
to understand what still blocks reference-grade output.

## Pipeline Used

- Subject/jersey/face decomposition from the target image
- 16:9 composition with right-side subject placement
- Multi-pass typography:
  - microtext face shading
  - medium structure words
  - jersey pass
  - manual anchor pass
  - left-side slogan

## Iteration Artifacts

- `examples/reference_recreation/recreation_v1.png`
- `examples/reference_recreation/recreation_v2.png`
- `examples/reference_recreation/recreation_v3.png`
- `examples/reference_recreation/side_by_side_v3.png`
- `examples/reference_recreation/recreation_metrics.json`

## Remaining Gaps

- Axis-aligned text still limits contour flow and anatomy wrapping.
- Anchor placement is manual and tuned only for this target composition.
- Face/jersey pass separation works, but the result still lacks refined poster
  art-direction.
