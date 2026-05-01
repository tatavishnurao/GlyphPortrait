# Reference Orientation Map

Generated visual: `reference_analysis/reference_orientation_map.png`.

## Orientation Grammar

- Crown/forehead: curved horizontal and shallow diagonal bands follow the skull dome.
- Left skull edge: near-vertical small words define the silhouette edge.
- Eye area: mostly horizontal words, but dark and tightly packed so the eye sockets remain recessed.
- Nose bridge: diagonal and vertical words follow the ridge from brow to nostril.
- Cheek: diagonal words define the cheek plane; large anchors sit at shallow rotations.
- Mouth/chin: horizontal and slightly diagonal words trace lip and chin highlights.
- Jawline/neck: vertical and steep diagonal words follow the jaw drop and neck column.
- Jersey trim: long white words sweep along the collar seam and shoulder seam.
- Jersey chest: red words run diagonally and vertically, creating fabric direction and compression.
- Right shoulder: large vertical/rotated typography follows the shoulder mass rather than the face.

## Renderer Implications

- A single horizontal text painter cannot reproduce the poster.
- Manual anchor rotations are required for cheek, jaw, neck, collar, and shoulder.
- Automated contour sampling should estimate local edge tangents for medium words.
- Jersey orientation should be its own grammar; red body text should not inherit face rules.
- Silhouette-adjacent words need clipping after rotation, otherwise the outline becomes fuzzy.
