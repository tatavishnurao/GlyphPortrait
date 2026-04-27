# GlyphForge

GlyphForge is a local-first GPU-assisted typographic portrait generator.
It turns a portrait into export-quality poster/wallpaper artwork built from
real, readable words inside a subject silhouette.

## Why this project
- Python-first CV + rendering pipeline (not a thin image-gen wrapper)
- Real text placement with mask-awareness and collision control
- Local Gradio app with deterministic seed and high-res PNG export

## MVP features
- Upload portrait
- Parse words (comma/newline separated)
- Live stage previews in Gradio:
  - preprocessed portrait
  - subject mask
  - final typographic output
  - metrics JSON
- Theme selection:
  - monochrome dark
  - minimal grayscale
  - sports red/black
  - gold/black tribute
- Mask generation with fallback if segmentation model is unavailable
- Typography layout engine:
  - weighted word priority
  - varied font sizes
  - occupancy-grid collision avoidance
  - grayscale/edge-guided placement bias
  - deterministic seed option
- Export presets: `1:1`, `4:5`, `16:9`, `9:16`

## Quickstart (`uv`)
Create a dedicated env named `inferenceimg`:

```bash
uv venv inferenceimg
source inferenceimg/bin/activate
uv pip install -r requirements.txt
```

Run app:

```bash
python app.py
```

Run CLI:

```bash
python cli.py \
  --input reference_img/Michael-Jordan-Wallpaper-Desktop-1.jpg \
  --words "MVP, Champion, Leader, Legacy, Focus, Discipline" \
  --theme sports_red_black \
  --ratio 16:9 \
  --output examples/outputs/sample.png
```

Run tests:

```bash
pytest -q
```
