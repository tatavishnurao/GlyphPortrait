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


def test_layout_boxes_stay_inside_mask():
    mask = np.zeros((220, 220), dtype=np.uint8)
    mask[30:190, 40:180] = 255
    words = [("Leader", 1.0), ("Legacy", 0.9), ("Focus", 0.8), ("Grit", 0.7)] * 8

    def font_loader(size: int):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    layout, _ = generate_layout(
        words=words,
        mask=mask,
        min_size=10,
        max_size=20,
        density=0.55,
        attempts_per_word=120,
        seed=99,
        font_loader=font_loader,
    )

    for item in layout:
        region = mask[item.y : item.y + item.height, item.x : item.x + item.width]
        assert region.size > 0
        assert float((region > 0).mean()) >= 0.86


def test_density_behavior_increases_fill_ratio():
    mask = np.ones((280, 220), dtype=np.uint8) * 255
    words = [("Champion", 1.0), ("Legacy", 0.95), ("Focus", 0.9), ("Drive", 0.88)] * 50

    def font_loader(size: int):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

    _, low_stats = generate_layout(
        words=words,
        mask=mask,
        min_size=8,
        max_size=24,
        density=0.25,
        attempts_per_word=100,
        seed=7,
        font_loader=font_loader,
    )
    _, high_stats = generate_layout(
        words=words,
        mask=mask,
        min_size=8,
        max_size=24,
        density=0.75,
        attempts_per_word=100,
        seed=7,
        font_loader=font_loader,
    )

    assert high_stats.fill_ratio >= low_stats.fill_ratio
