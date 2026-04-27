import numpy as np
from PIL import ImageFont

from glyphforge.typography.layout import generate_layout


def test_layout_is_deterministic_for_seed():
    mask = np.ones((240, 180), dtype=np.uint8) * 255
    words = [("Leader", 1.0), ("Legacy", 0.9), ("Focus", 0.8), ("Grit", 0.7)]

    def font_loader(size: int):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    a_layout, _ = generate_layout(
        words=words,
        mask=mask,
        min_size=10,
        max_size=30,
        density=0.4,
        attempts_per_word=60,
        seed=1234,
        font_loader=font_loader,
    )
    b_layout, _ = generate_layout(
        words=words,
        mask=mask,
        min_size=10,
        max_size=30,
        density=0.4,
        attempts_per_word=60,
        seed=1234,
        font_loader=font_loader,
    )

    assert [(w.word, w.x, w.y, w.size) for w in a_layout] == [
        (w.word, w.x, w.y, w.size) for w in b_layout
    ]
