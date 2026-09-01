# GlyphPortrait

GlyphPortrait is a semantic digital micrography engine for portrait reconstruction. It takes a person image and a user-provided word prompt, extracts/vectorizes visual regions, generates coherent text streamlines, and renders the person using readable text.

## Run

```bash
python app.py          # Gradio UI
python cli.py --input photo.jpg --words "word1, word2" --output out.png
```

## Docs

- [docs/algorithm.md](docs/algorithm.md)
- [docs/limitations.md](docs/limitations.md)

## Architecture

- [System overview](docs/architecture/system-overview.svg)
- [Critical path](docs/architecture/critical-path.svg)
- [Evidence notes](docs/architecture/README.md)
