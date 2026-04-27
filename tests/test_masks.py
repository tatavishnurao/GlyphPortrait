import numpy as np

from glyphforge.image.masks import cleanup_mask


def test_cleanup_mask_produces_binary_mask():
    raw = np.zeros((64, 64), dtype=np.uint8)
    raw[20:40, 20:40] = 255
    out = cleanup_mask(raw)
    assert out.shape == raw.shape
    assert out.dtype == np.uint8
    values = set(np.unique(out).tolist())
    assert values.issubset({0, 255})
