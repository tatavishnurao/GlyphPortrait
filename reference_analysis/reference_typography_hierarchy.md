# Reference Typography Hierarchy

## Tier 1: Major Anchors

- Size: roughly 45-95 px equivalent in the 1920x1080 image.
- Words: `MVP`, `Air Jordan`, `Dedication`, `Dominance`, `Give Up`, large jersey `23` mass.
- Function: anatomical landmarks and identity hits. These are manually positioned, not randomly distributed.
- Regions: forehead, cheek, jaw/neck, right shoulder, jersey number zone.

## Tier 2: Major Structure Words

- Size: roughly 24-44 px.
- Words: `NBA Rookie of the Year`, `Chicago Bulls`, `Love of the Game`, `Named to the All-Star Team`, `NBA Most Valuable Player`.
- Function: bridge between portrait contours and readable narrative. These words define skull arcs, cheek planes, jersey trim, and shoulder sweeps.
- Regions: crown, temple, cheek, collar/trim, jersey chest.

## Tier 3: Readable Fillers

- Size: roughly 13-23 px.
- Words: `NBA Champion`, `Finals MVP`, `Slam Dunk`, `All American Game`, `game winner`, `scoring`, `rookie`.
- Function: mid-frequency texture. These fill the subject mass while remaining readable at normal poster scale.
- Regions: all subject regions, especially forehead, neck, red jersey, and shoulders.

## Tier 4: Microtext

- Size: roughly 6-12 px.
- Words: statistics, dates, repeated achievements, quote fragments.
- Function: tone-building. This is where the portrait becomes image-like rather than a set of labels.
- Regions: eye sockets, cheek shadow, jaw shadow, jersey red field, shoulder gray field.

## Tier 5: Near-Micro Dark Texture

- Size: below comfortable readability or rendered at very low contrast.
- Function: dark mass preservation. The eye sockets, under-cheek, under-neck, and jersey shadow need text-like texture without becoming bright.
- Renderer implication: this pass must be low-alpha and color-sampled from target tone. If it is too bright, the likeness collapses.

## Weight and Typeface

The reference behaves like condensed bold sans typography. It uses tight word packing, short leading, and frequent uppercase or title-case. The dominant feeling is poster/commercial rather than literary typesetting.

Renderer implication: use condensed bold fonts where available, avoid wide geometric fonts, and allow overlaps/clipping instead of preserving clean text layout.
