# Experiment 003: Reference Recreation Attempt

## Goal

Try to move closer to the Jordan-style reference wallpaper instead of only
producing a clean generic output.

## Output

- [examples/evolution/04_reference_sports_attempt.png](../examples/evolution/04_reference_sports_attempt.png)

## What worked

- Red/black theme separation gives the jersey more identity than monochrome.
- Importance-guided placement helps the face read better than uniform filling.

## What still feels wrong

- Text is still axis-aligned, so it does not wrap anatomy or fabric direction.
- The face and jersey are not treated as different typography zones yet.
- Anchor words are not intentionally placed in semantically important regions.

## Takeaway

This is the point where the project stops being "word placement in a mask" and
starts needing art-direction logic:

- face-aware density
- protected eye/nose/mouth regions
- jersey-specific color and scale behavior
