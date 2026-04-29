# Experiment 007: Visual Quality Pass

## Goal

Improve likeness to the Jordan-style reference without adding product features
or changing repository structure.

## Changes in this pass

- Reserved anchor regions before stochastic placement so anchor words no longer
  fight random micro/structure text.
- Strengthened face microtext placement toward darker and edge-heavy zones.
- Separated face and jersey word banks and tuned jersey density/color behavior.
- Added staged output artifacts for diagnosis:
  - `stage_01_masks.png`
  - `stage_02_microtext.png`
  - `stage_03_structure_words.png`
  - `stage_04_jersey_words.png`
  - `stage_05_anchors.png`
  - `stage_06_final.png`
  - `side_by_side_final.png`

## Metrics additions

- `anchors_reserved`
- pass-level render timing (`stage_*_ms`)
- existing placement and luminance/edge metrics retained

## Remaining gaps

- Text is still axis-aligned.
- Face landmark protection is still approximate, not explicit landmark-driven.
- Anchor coordinates remain manual and target-specific.
