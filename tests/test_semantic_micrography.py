from pathlib import Path

import numpy as np
from PIL import Image

from glyphforge.semantic_micrography.auto import (
    CANONICAL_OUTPUTS,
    generate_candidate_specs,
    run_auto_search,
    score_candidate,
)
from glyphforge.semantic_micrography.config import CanvasConfig, MicrographyStyleConfig, PipelineConfig
from glyphforge.semantic_micrography.lanes import generate_lanes
from glyphforge.semantic_micrography.ordering import order_lanes
from glyphforge.semantic_micrography.pipeline import run_pipeline
from glyphforge.semantic_micrography.preprocess import PreprocessResult, load_and_preprocess
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


def _write_rgb_and_mask(tmp_path: Path, mask: np.ndarray) -> tuple[Path, Path]:
    image_path = tmp_path / "person.png"
    mask_path = tmp_path / "mask.png"
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[..., :] = (20, 20, 20)
    rgb[mask > 0] = (220, 150, 95)
    Image.fromarray(rgb, "RGB").save(image_path)
    Image.fromarray(mask.astype(np.uint8), "L").save(mask_path)
    return image_path, mask_path


def test_manual_mask_override_is_used(tmp_path: Path):
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[20:76, 28:68] = 255
    image_path, mask_path = _write_rgb_and_mask(tmp_path, mask)

    prep = load_and_preprocess(image_path, CanvasConfig(long_edge=96), mask_path=mask_path)

    assert prep.mask_source == "manual"
    assert prep.mask_quality["mask_quality_status"] == "ok"
    assert 0.20 < prep.mask_quality["subject_coverage"] < 0.30


def test_full_canvas_mask_warns(tmp_path: Path):
    mask = np.ones((96, 96), dtype=np.uint8) * 255
    image_path, mask_path = _write_rgb_and_mask(tmp_path, mask)

    prep = load_and_preprocess(image_path, CanvasConfig(long_edge=96), mask_path=mask_path)

    warnings = prep.mask_quality["mask_quality_warnings"]
    assert "Subject mask covers most of the canvas; background likely swallowed." in warnings


def test_tiny_mask_warns(tmp_path: Path):
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[45:49, 45:49] = 255
    image_path, mask_path = _write_rgb_and_mask(tmp_path, mask)

    prep = load_and_preprocess(image_path, CanvasConfig(long_edge=96), mask_path=mask_path)

    warnings = prep.mask_quality["mask_quality_warnings"]
    assert "Subject mask too small." in warnings


def test_fragmented_mask_warns(tmp_path: Path):
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[20:55, 20:55] = 255
    mask[65:100, 65:100] = 255
    image_path, mask_path = _write_rgb_and_mask(tmp_path, mask)

    prep = load_and_preprocess(image_path, CanvasConfig(long_edge=120), mask_path=mask_path)

    warnings = prep.mask_quality["mask_quality_warnings"]
    assert "Subject mask fragmented." in warnings
    assert prep.mask_quality["largest_component_ratio"] < 0.70


def test_normal_synthetic_mask_passes_quality(tmp_path: Path):
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[20:105, 35:85] = 255
    image_path, mask_path = _write_rgb_and_mask(tmp_path, mask)

    prep = load_and_preprocess(image_path, CanvasConfig(long_edge=120), mask_path=mask_path)

    assert prep.mask_quality["mask_quality_status"] == "ok"
    assert prep.mask_quality["mask_quality_warnings"] == []


def test_semantic_micrography_cli_accepts_auto_args():
    from scripts.render_semantic_micrography import build_parser

    args = build_parser().parse_args(
        [
            "--input",
            "person.png",
            "--prompt",
            "Person, Builder",
            "--out-dir",
            "out",
            "--auto",
            "--candidate-count",
            "7",
            "--seed",
            "99",
            "--debug-candidates",
        ]
    )

    assert args.auto is True
    assert args.candidate_count == 7
    assert args.seed == 99
    assert args.debug_candidates is True


def test_auto_candidate_generation_is_deterministic():
    style = MicrographyStyleConfig()

    specs_a = generate_candidate_specs(style, candidate_count=6, seed=123)
    specs_b = generate_candidate_specs(style, candidate_count=6, seed=123)
    specs_c = generate_candidate_specs(style, candidate_count=6, seed=124)

    assert specs_a == specs_b
    assert specs_a != specs_c
    assert specs_a[0].lane_spacing_px == style.lane_spacing_px


def test_auto_scoring_penalizes_source_leakage():
    base = {
        "text_coverage_subject": 0.35,
        "lane_coverage_subject": 0.35,
        "face_feature_coverage": 0.20,
        "microtext_to_lane_ratio": 6.5,
        "source_pixel_leakage": 0.0,
        "background_cleanliness": 1.0,
        "short_lane_ratio": 0.0,
        "lane_count": 70,
    }

    leaked = {**base, "source_pixel_leakage": 0.25}

    assert score_candidate(base) > score_candidate(leaked)


def test_auto_scoring_penalizes_high_microtext_ratio():
    base = {
        "text_coverage_subject": 0.35,
        "lane_coverage_subject": 0.35,
        "face_feature_coverage": 0.20,
        "microtext_to_lane_ratio": 6.5,
        "source_pixel_leakage": 0.0,
        "background_cleanliness": 1.0,
        "short_lane_ratio": 0.0,
        "lane_count": 70,
    }

    noisy = {**base, "microtext_to_lane_ratio": 20.0}

    assert score_candidate(base) > score_candidate(noisy)


def test_auto_scoring_rewards_feature_and_text_coverage():
    weak = {
        "text_coverage_subject": 0.20,
        "lane_coverage_subject": 0.20,
        "face_feature_coverage": 0.05,
        "microtext_to_lane_ratio": 6.5,
        "source_pixel_leakage": 0.0,
        "background_cleanliness": 1.0,
        "short_lane_ratio": 0.0,
        "lane_count": 70,
    }
    strong = {**weak, "text_coverage_subject": 0.34, "lane_coverage_subject": 0.34, "face_feature_coverage": 0.24}

    assert score_candidate(strong) > score_candidate(weak)


def test_auto_scoring_rewards_anchor_visibility():
    weak = {
        "text_coverage_subject": 0.32,
        "lane_coverage_subject": 0.32,
        "face_feature_coverage": 0.20,
        "anchor_visibility_score": 0.10,
        "low_body_anchor_ratio": 0.50,
        "word_repetition_ratio": 0.20,
        "microtext_to_lane_ratio": 6.5,
        "source_pixel_leakage": 0.0,
        "background_cleanliness": 1.0,
        "short_lane_ratio": 0.0,
        "lane_count": 70,
    }
    strong = {**weak, "anchor_visibility_score": 0.76, "low_body_anchor_ratio": 0.02}

    assert score_candidate(strong) > score_candidate(weak)


def test_auto_search_picks_highest_scoring_candidate(tmp_path: Path):
    profile = profile_from_prompt("Vishnu, AI Infra, Rust")
    config = PipelineConfig(style=MicrographyStyleConfig(), output_dir=tmp_path / "out")
    scores = [
        {"text_coverage_subject": 0.12, "face_feature_coverage": 0.04},
        {"text_coverage_subject": 0.33, "face_feature_coverage": 0.28},
        {"text_coverage_subject": 0.26, "face_feature_coverage": 0.08},
    ]
    calls: list[Path] = []

    def fake_runner(input_path, profile, candidate_dir, candidate_config, mask_path):
        index = len(calls)
        calls.append(candidate_dir)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        for name in CANONICAL_OUTPUTS:
            (candidate_dir / name).write_text(f"candidate={index}\n", encoding="utf-8")
        return {
            "lane_count": 50,
            "lane_coverage_subject": scores[index]["text_coverage_subject"],
            "text_coverage_subject": scores[index]["text_coverage_subject"],
            "face_feature_coverage": scores[index]["face_feature_coverage"],
            "anchor_visibility_score": 0.55,
            "microtext_to_lane_ratio": 6.0,
            "source_pixel_leakage": 0.0,
            "background_cleanliness": 1.0,
            "short_lane_ratio": 0.0,
            "low_body_anchor_ratio": 0.0,
            "word_repetition_ratio": 0.14,
        }

    metrics = run_auto_search(
        tmp_path / "person.png",
        profile,
        tmp_path / "out",
        config,
        candidate_count=3,
        seed=7,
        runner=fake_runner,
    )

    assert metrics["face_feature_coverage"] == 0.28
    assert (tmp_path / "out" / "current_best.svg").read_text(encoding="utf-8") == "candidate=1\n"
    summary = (tmp_path / "out" / "candidate_search.json").read_text(encoding="utf-8")
    assert "top_candidates" in summary
    assert "anchor_visibility_score" in summary
