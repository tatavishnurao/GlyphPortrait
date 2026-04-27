# GlyphForge

Local-first GPU-assisted typographic portrait generator.

GlyphForge turns a portrait into export-quality poster/wallpaper artwork made from real readable words shaped by the subject silhouette.

## Why this project
- Python-first CV + rendering pipeline (not a thin image-gen wrapper)
- Real text placement with mask-awareness and collision control
- Local Gradio app with deterministic seed and high-res PNG export

## MVP features
- Upload portrait
- Parse words (comma/newline separated)
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
