# 008 Reference Visual Forensics

Reference: `reference_img/Michael-Jordan-Wallpaper-Desktop-1.jpg`.

## Core Observations

- The poster is not a generic text fill. It is a manually composed typographic portrait with automated-looking texture underneath.
- The subject occupies the right half of a 1920x1080 black field. The left half is deliberately silent negative space.
- The portrait has three major material systems: gray/white face text, red jersey text, and gray/white shoulder text.
- Readable anchor words are used as anatomical structure: `MVP`, `Air Jordan`, `Dedication`, `Dominance`, `NBA Rookie of the Year`, `Give Up`, `23`.
- Microtext remains visible in shadow regions, but it is dark enough to preserve eyes, under-cheek, jaw, and jersey folds.
- The face and jersey are clipped sharply to the subject silhouette. Words frequently terminate at the skull, shoulder, and jersey edges.

## Word Inventory

See `reference_analysis/reference_word_inventory.md`.

The inventory is a factual/biographical vocabulary: achievements, dates, statistics, Chicago Bulls identity, Jordan brand terms, and quote fragments. Most words repeat across tiers rather than appearing once.

## Word Role Classification

- Anchor words: `MVP`, `Air Jordan`, `Dedication`, `Dominance`, `NBA Rookie of the Year`, `Chicago Bulls`, `Give Up`, `23`.
- Structure words: `All American Game`, `Love of the Game`, `Named to the All-Star Team`, `NBA Most Valuable Player`, `Finals MVP`.
- Texture words: `NBA Champion`, `game winner`, `rookie`, `Slam Dunk`, dates, statistics.
- Contour words: `Dominance`, `Scoring`, vertical temple text, collar/trim phrases, right-shoulder vertical phrases.
- Shadow words: low-contrast repeats inside eye sockets, central face, neck center, and jersey folds.
- Highlight words: crown rim, nose bridge, lip/chin, collar trim, jersey number, right shoulder highlights.
- Region-specific words: red `Named to the All-Star Team` and `NBA Most Valuable Player` on jersey; `Give Up` and quote fragments on right shoulder.

## Region-by-Region Breakdown

- Forehead: high-density curved bands. Major words include `NBA Rookie of the Year`, `MVP`, `Chicago Bulls`, `All American Game`, dates, and statistics.
- Eye area: protected dark sockets. Text exists but is low-contrast; large anchors avoid over-brightening the pupils and brow.
- Nose bridge: contour-following small/medium text creates the ridge and nostril highlight.
- Cheek: large `Air Jordan`, `Dedication`, and `Love of the Game` sit over dark gray tonal text.
- Mouth/chin: short bright words trace lip edge and chin form; surrounding filler remains dark.
- Jaw: steep/vertical words, especially `Dominance`, create the jawline and left neck edge.
- Neck: dense gray text with vertical flow. It is a shadow column, not a bright label zone.
- Shoulder: left shoulder uses gray/white text; right shoulder has large rotated phrases and high contrast.
- Jersey chest: red typography is the base material. White/cream text and number shapes are overlaid for trim and `23`.
- Jersey trim: bright white curved text strips define collar and arm seams.
- Background / negative space: nearly empty black with centered `change the game.` caption.

## Typography Hierarchy

See `reference_analysis/reference_typography_hierarchy.md`.

The reference uses five tiers: major anchors, structure words, readable fillers, microtext, and near-micro dark texture. The hierarchy is spatially meaningful: large words define anatomy, small words carry tone.

## Orientation Grammar

See `reference_analysis/reference_orientation_map.png` and `reference_analysis/reference_orientation_map.md`.

Text orientation is region-dependent. Crown text follows skull curvature; face fillers are mostly horizontal; nose/cheek/jaw words rotate with anatomy; jersey words follow seams and fabric diagonals.

## Clipping and Masking Behavior

- Skull edge: many crown and temple words are clipped by the silhouette.
- Nose and cheek: words cross tonal regions, but brightness is clipped/modulated by the local face shadow.
- Eye sockets: text is not absent; it is shadow-clipped and low alpha.
- Jersey trim: white words cross red/black boundaries but are clipped to seam bands.
- Right shoulder: large words are partially clipped by the frame and subject outline.
- Background: random texture never bleeds into the black field.

## Density Map

See `reference_analysis/reference_density_map.png` and `reference_analysis/reference_density_map.md`.

The densest readable regions are the crown, jersey, neck, and shoulders. The darkest dense region is the eye socket mass. The negative-space field is intentionally sparse.

## Color and Contrast Roles

- Bright white: skull rim, nose, mouth/chin, shoulder highlights, jersey trim/number.
- Mid-gray: main face structure and readable anchors.
- Dark gray: shadow microtext in eyes, cheek, jaw, and neck.
- Red: jersey identity and body fill.
- Deep red: jersey folds and shadowed red mass.

## Inferred Layer Order

1. Black background and caption.
2. Subject silhouette, head/neck/body masks.
3. Dark shadow microtext clipped to subject.
4. Surface texture text sampled by local tone.
5. Contour words aligned to edge tangents.
6. Manual anatomy anchors.
7. Separate jersey red typography pass.
8. Jersey trim/number highlight pass.
9. Final tonal modulation and silhouette clipping.

## Renderer Implications

| Observation | Required renderer capability |
|---|---|
| Large words along cheek/jaw | Manual anatomy anchors with explicit x/y/rotation/scale. |
| Dense tiny shadow text | Low-alpha microtext pass driven by shadow density. |
| Red jersey identity | Separate jersey pass with red palette and seam-aware orientation. |
| Eyes remain dark | Protected dark eye socket masks and capped alpha/brightness. |
| Strong clipping at silhouette | Final mask clipping after text rotation. |
| Crown follows skull arc | Region-specific orientation grammar or manual curved anchor bands. |
| Nose and mouth are bright contours | Edge/tangent contour pass with brighter small text. |
| Background is silent | Negative-space mask that blocks all fill text. |
| Jersey trim is white and curved | Dedicated seam/trim anchor layer, not generic random fill. |
| Large right-shoulder words differ from face | Shoulder-specific anchors and vertical/rotated orientation rules. |
| Microtext repeats biography terms | Weighted word inventory with repeated terms across tiers. |
| Dark areas still contain words | Tone modulation must darken text rather than skip placement. |

## Automated vs Art-Directed

- Automate: subject/jersey masks, luminance guide, density guide, contour guide, microtext fill, small surface texture, approximate tangent alignment.
- Art-direct: major face anchors, eye protection, cheek/jaw words, jersey number and identity, trim/seam phrases, background caption placement.
- Hybrid: contour words around nose, mouth, collar, and shoulders can be sampled automatically but need region-specific filtering and post-tuning.

## Current Reconstruction Implications

The v4 scratch script follows the inferred layer order and produces face, jersey, and full-poster studies. It still underperforms the reference in three places: face anchors are too typographically dominant in the forehead, jersey number/trim geometry is too coarse, and the real poster has more precise curved text bands than the current tangent sampler.
