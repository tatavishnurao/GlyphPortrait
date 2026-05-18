# GlyphPortrait

Turns a portrait into typographic poster/wallpaper using real words inside the subject silhouette.

## Run

```bash
python app.py          # Gradio UI
python cli.py --input photo.jpg --words "word1, word2" --output out.png
```

## Studies

- `studies/jordan_wallpaper/` — Jordan reference-stencil (use ref as luminance/edge stencil)
- `studies/goku_wallpaper/` — Goku word-tribute (multi-mask regions + anchor lanes)

## Docs

- [docs/algorithm.md](docs/algorithm.md)
- [docs/limitations.md](docs/limitations.md)
