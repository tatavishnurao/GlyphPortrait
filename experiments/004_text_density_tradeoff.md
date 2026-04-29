# Experiment 004: Text Density Tradeoff

## Goal

Measure how quickly the renderer becomes unreadable when density is pushed
higher.

## Machine

- GPU: RTX 4060 Laptop GPU
- Environment: local `uv` environment (`inferenceimg`)

## Sample observations

Using the current Jordan reference image at `960x540`:

- Moderate density (`0.62`) with `8-24px` fonts:
  - around `26` words placed
  - around `136` placement attempts
  - around `1.1s` render time
- Higher density with longer words:
  - placement attempts climb quickly
  - readability drops before silhouette fidelity improves

## What failed

- More density does not automatically make the portrait better.
- Long words burn layout area too aggressively.
- Pushing density without region logic mostly creates clutter.

## What changed next

- Keep density moderate for demos.
- Treat better placement policy as more important than "more words."
