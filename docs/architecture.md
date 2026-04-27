# GlyphForge Architecture

## Pipeline
1. Input portrait loaded via Gradio/CLI.
2. Preprocessing resizes and center-crops to target output ratio.
3. Segmentation attempts:
   - `rembg` (if installed and model available)
   - OpenCV GrabCut fallback
   - Otsu threshold fallback
4. Mask cleanup with morphology + blur thresholding.
5. Keyword parser converts input text into weighted word list.
6. Layout engine places words inside mask:
   - priority by weight
   - random sampling with deterministic seed
   - bbox-in-mask check
   - occupancy-grid collision rejection
7. Renderer colors text by theme + local grayscale tone cues.
8. PNG exporter writes 300 DPI output.

## Design choices
- Core pipeline is local and deterministic.
- Heavy AI modules are optional and non-blocking.
- Typography readability is favored over maximal fill.
- Simple modules keep extension points obvious.

## Extension points
- Replace segmentation backend with custom Torch model.
- Add rotated text and line-fitting for more organic typography.
- Integrate local LLM keyword suggestion.
- Add optional diffusion post-pass in `glyphforge/stylize/`.
