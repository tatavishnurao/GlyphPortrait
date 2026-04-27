from __future__ import annotations

import json
from datetime import datetime

import gradio as gr

from glyphforge.config import AppConfig
from glyphforge.image.export import save_png
from glyphforge.keywords.suggest import suggest_default_keywords
from glyphforge.typography.render import render_typographic_portrait
from glyphforge.typography.themes import THEMES
from glyphforge.utils.paths import ensure_dir

CFG = AppConfig()
THEME_CHOICES = list(THEMES.keys())
RATIO_CHOICES = ["1:1", "4:5", "16:9", "9:16"]


def _render(
    image,
    words_text: str,
    theme_name: str,
    ratio_label: str,
    density: float,
    seed: int,
    long_edge: int,
    min_font_size: int,
    max_font_size: int,
    attempts_per_word: int,
):
    if image is None:
        raise gr.Error("Please upload a portrait image.")
    if not words_text.strip():
        words_text = ", ".join(suggest_default_keywords("athlete"))

    result = render_typographic_portrait(
        image_input=image,
        words_text=words_text,
        theme_name=theme_name,
        ratio_label=ratio_label,
        density=density,
        seed=seed,
        long_edge=long_edge,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        attempts_per_word=attempts_per_word,
        config=CFG,
    )

    out_dir = ensure_dir(CFG.outputs_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        out_dir / f"glyphforge_{theme_name}_{ratio_label.replace(':', 'x')}_{ts}.png"
    )
    save_png(result.image, out_path, dpi=300)

    metrics_text = json.dumps(result.metrics, indent=2)
    return (
        result.preprocessed_preview,
        result.mask_preview,
        result.image,
        metrics_text,
        str(out_path),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="GlyphForge") as demo:
        gr.Markdown("""
# GlyphForge
Generate a typographic portrait from your local image using real readable words.
""")
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Portrait", type="pil")
                words_input = gr.Textbox(
                    label="Words (comma or newline separated)",
                    lines=8,
                    placeholder="MVP, Champion, Leader, Legacy, Focus, Discipline",
                )
                theme_input = gr.Dropdown(
                    choices=THEME_CHOICES, value="monochrome_dark", label="Theme"
                )
                ratio_input = gr.Dropdown(
                    choices=RATIO_CHOICES,
                    value="4:5",
                    label="Aspect Ratio",
                )
                density_input = gr.Slider(
                    0.2, 0.95, value=0.65, step=0.01, label="Density"
                )
                seed_input = gr.Number(value=42, precision=0, label="Seed")
                long_edge_input = gr.Slider(
                    960,
                    3200,
                    value=1600,
                    step=32,
                    label="Long Edge (px)",
                )
                min_font_input = gr.Slider(
                    8, 48, value=12, step=1, label="Min Font Size"
                )
                max_font_input = gr.Slider(
                    18, 110, value=64, step=1, label="Max Font Size"
                )
                attempts_input = gr.Slider(
                    30, 500, value=180, step=5, label="Attempts/Word"
                )
                render_btn = gr.Button("Generate Poster", variant="primary")
            with gr.Column(scale=1):
                preprocessed_preview = gr.Image(
                    label="Preprocessed Preview", type="pil"
                )
                mask_preview = gr.Image(label="Subject Mask Preview", type="pil")
                output_image = gr.Image(label="Final Typographic Output", type="pil")
                output_metrics = gr.Code(label="Metrics", language="json")
                output_file = gr.Textbox(label="Saved PNG Path")

        render_btn.click(
            fn=_render,
            inputs=[
                image_input,
                words_input,
                theme_input,
                ratio_input,
                density_input,
                seed_input,
                long_edge_input,
                min_font_input,
                max_font_input,
                attempts_input,
            ],
            outputs=[
                preprocessed_preview,
                mask_preview,
                output_image,
                output_metrics,
                output_file,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
