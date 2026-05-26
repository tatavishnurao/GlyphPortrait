from pathlib import Path

import numpy as np
from PIL import Image

from glyphforge.semantic_micrography.config import CanvasConfig, MicrographyStyleConfig, PipelineConfig
from glyphforge.semantic_micrography.lanes import generate_lanes
from glyphforge.semantic_micrography.ordering import order_lanes
from glyphforge.semantic_micrography.pipeline import run_pipeline
from glyphforge.semantic_micrography.preprocess import PreprocessResult
from glyphforge.semantic_micrography.profiles import profile_from_mapping, profile_from_prompt
from glyphforge.semantic_micrography.regions import REGION_KEYS, extract_regions
from glyphforge.semantic_micrography.vectorize import vectorize_mask, vectorize_regions


def test_profile_parsing_from_prompt_and_json_shape():
    prompt_profile = profile_from_prompt("Michael Jordan, MVP, Champion")
    assert prompt_profile.subject_name == "Michael Jordan"
    assert "MVP" in prompt_profile.hero_words

    profile = profile_from_mapping({"subject": "A", "hero_words": ["Hero"], "region_words": {"skin_or_warm": ["Warm"]}})
    assert profile.words_for_region("skin_or_warm")[0] == "Warm"


def test_region_extraction_returns_expected_keys():
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    rgb[18:78, 24:72] = (210, 150, 95)
    rgb[60:82, 28:68] = (210, 30, 25)
    subject = np.zeros((96, 96), dtype=np.uint8)
    subject[18:82, 24:72] = 255
    prep = PreprocessResult(rgb, subject, np.mean(rgb, axis=2).astype(np.uint8), subject, "black", (96, 96))
    result = extract_regions(prep)
    assert set(REGION_KEYS).issubset(result.masks)
    assert result.masks["subject"].shape == (96, 96)


def test_vectorize_returns_contours_for_synthetic_mask():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 15:65] = 255
    contours = vectorize_mask("skin_or_warm", mask)
    assert contours
    assert contours[0].closed


def test_lane_generation_returns_long_enough_lanes():
    mask = np.zeros((120, 180), dtype=np.uint8)
    mask[20:100, 20:160] = 255
    contours = {"skin_or_warm": vectorize_mask("skin_or_warm", mask)}
    lanes, diagnostics = generate_lanes({"skin_or_warm": mask, "subject": mask}, contours, MicrographyStyleConfig(min_lane_length_px=40))
    assert lanes
    assert min(lane.length_px for lane in lanes) >= 40
    assert diagnostics["skin_or_warm"]["accepted_lanes"] == len(lanes)


def test_ordering_is_deterministic():
    mask = np.zeros((120, 180), dtype=np.uint8)
    mask[20:100, 20:160] = 255
    contours = vectorize_regions({"skin_or_warm": mask, "subject": mask})
    lanes_a, _ = generate_lanes({"skin_or_warm": mask, "subject": mask}, contours, MicrographyStyleConfig(min_lane_length_px=40))
    lanes_b, _ = generate_lanes({"skin_or_warm": mask, "subject": mask}, contours, MicrographyStyleConfig(min_lane_length_px=40))
    assert [(lane.id, lane.points[0]) for lane in order_lanes(lanes_a)] == [(lane.id, lane.points[0]) for lane in order_lanes(lanes_b)]


def test_pipeline_smoke_generates_svg_and_metrics(tmp_path: Path):
    image_path = tmp_path / "person.png"
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    rgb[14:82, 30:66] = (220, 155, 100)
    rgb[58:88, 24:72] = (210, 30, 24)
    Image.fromarray(rgb, "RGB").save(image_path)
    profile = profile_from_prompt("Subject, Hero, Legacy, Focus, Champion")
    config = PipelineConfig(canvas=CanvasConfig(long_edge=96), style=MicrographyStyleConfig(min_lane_length_px=28))
    metrics = run_pipeline(image_path, profile, tmp_path / "out", config=config)
    assert (tmp_path / "out" / "current_best.svg").exists()
    assert (tmp_path / "out" / "current_best_metrics.json").exists()
    assert metrics["lane_count"] > 0
