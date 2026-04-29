import numpy as np

from studies.jordan_wallpaper.anchors import AnchorSpec, scale_anchor
from studies.jordan_wallpaper.metrics import build_metrics
from studies.jordan_wallpaper.target_analysis import (
    extract_nonblack_subject_mask,
    extract_red_jersey_mask,
)


def test_subject_mask_non_empty_for_synthetic_black_background():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[30:90, 70:130] = (220, 220, 220)
    mask = extract_nonblack_subject_mask(rgb)
    assert mask.shape == (120, 160)
    assert int((mask > 0).sum()) > 0


def test_red_jersey_mask_shape_and_range():
    rgb = np.zeros((64, 80, 3), dtype=np.uint8)
    rgb[12:52, 20:60] = (210, 40, 30)
    subject = np.zeros((64, 80), dtype=np.uint8)
    subject[8:56, 16:64] = 255
    jersey = extract_red_jersey_mask(rgb, subject)
    assert jersey.shape == subject.shape
    assert jersey.dtype == np.uint8
    assert set(np.unique(jersey).tolist()).issubset({0, 255})


def test_anchor_coordinate_scaling():
    anchor = AnchorSpec("MVP", 0.5, 0.25, 40, "face", (255, 255, 255))
    x, y, size = scale_anchor(anchor, width=1920, height=1080)
    assert x == 960
    assert y == 270
    assert size >= 8


def test_metrics_payload_contains_expected_keys():
    payload = build_metrics({"render_ms": 10.0})
    for key in [
        "subject_iou",
        "jersey_iou",
        "luminance_mae_subject",
        "edge_overlap",
        "face_words_placed",
        "jersey_words_placed",
        "anchors_placed",
        "render_ms",
    ]:
        assert key in payload
