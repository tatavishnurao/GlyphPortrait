from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from glyphforge.semantic_micrography.config import MicrographyStyleConfig, PipelineConfig
from glyphforge.semantic_micrography.pipeline import run_pipeline
from glyphforge.semantic_micrography.profiles import WordProfile


CANONICAL_OUTPUTS = (
    "current_best.svg",
    "current_best.png",
    "current_best_metrics.json",
    "regions_panel.png",
    "lane_overlay.svg",
    "lane_overlay.png",
    "debug_summary.json",
)


@dataclass(frozen=True)
class CandidateSpec:
    index: int
    lane_spacing_px: float
    anchor_font_scale: float
    filler_density: float
    min_lane_length_px: float
    feature_lane_boost: float


def _scaled_region_styles(
    style: MicrographyStyleConfig,
    filler_density: float,
    feature_lane_boost: float,
) -> dict[str, dict[str, float | int]]:
    styles = {region: values.copy() for region, values in style.region_lane_styles.items()}
    for region, values in styles.items():
        if region == "feature_detail":
            values["spacing_scale"] = float(values.get("spacing_scale", 1.0)) / max(0.75, feature_lane_boost)
            values["font_scale"] = float(values.get("font_scale", 1.0)) * max(0.90, min(1.20, feature_lane_boost))
            continue
        values["spacing_scale"] = float(values.get("spacing_scale", 1.0)) / max(0.60, min(1.55, filler_density))
    return styles


def style_for_candidate(base_style: MicrographyStyleConfig, spec: CandidateSpec) -> MicrographyStyleConfig:
    return replace(
        base_style,
        lane_spacing_px=spec.lane_spacing_px,
        anchor_font_scale=spec.anchor_font_scale,
        min_lane_length_px=spec.min_lane_length_px,
        filler_density=spec.filler_density,
        feature_lane_boost=spec.feature_lane_boost,
        region_lane_styles=_scaled_region_styles(base_style, spec.filler_density, spec.feature_lane_boost),
    )


def generate_candidate_specs(
    base_style: MicrographyStyleConfig,
    candidate_count: int,
    seed: int,
) -> list[CandidateSpec]:
    count = max(1, candidate_count)
    rng = random.Random(seed)
    specs = [
        CandidateSpec(
            index=0,
            lane_spacing_px=float(base_style.lane_spacing_px),
            anchor_font_scale=float(base_style.anchor_font_scale),
            filler_density=float(base_style.filler_density),
            min_lane_length_px=float(base_style.min_lane_length_px),
            feature_lane_boost=float(base_style.feature_lane_boost),
        )
    ]
    for index in range(1, count):
        specs.append(
            CandidateSpec(
                index=index,
                lane_spacing_px=round(base_style.lane_spacing_px * rng.uniform(0.84, 1.28), 3),
                anchor_font_scale=round(base_style.anchor_font_scale * rng.uniform(0.92, 1.22), 3),
                filler_density=round(base_style.filler_density * rng.uniform(0.58, 1.18), 3),
                min_lane_length_px=round(base_style.min_lane_length_px * rng.uniform(0.82, 1.32), 3),
                feature_lane_boost=round(base_style.feature_lane_boost * rng.uniform(0.90, 1.60), 3),
            )
        )
    return specs


def score_candidate(metrics: dict[str, Any]) -> float:
    text_coverage = float(metrics.get("text_coverage_subject", 0.0))
    lane_coverage = float(metrics.get("lane_coverage_subject", text_coverage))
    face_feature_coverage = float(metrics.get("face_feature_coverage", 0.0))
    face_text_coverage = float(metrics.get("face_text_coverage", 0.0))
    identity_edge_alignment = float(metrics.get("identity_edge_alignment", 0.0))
    identity_edge_recall = float(metrics.get("identity_edge_recall", 0.0))
    anchor_visibility = float(metrics.get("anchor_visibility_score", 0.0))
    anchor_feature_ratio = float(metrics.get("anchor_feature_lane_ratio", 0.0))
    low_body_anchor_ratio = float(metrics.get("low_body_anchor_ratio", 0.0))
    word_repetition_ratio = float(metrics.get("word_repetition_ratio", 0.0))
    background_cleanliness = float(metrics.get("background_cleanliness", 0.0))
    source_leakage = float(metrics.get("source_pixel_leakage", 0.0))
    microtext_ratio = float(metrics.get("microtext_to_lane_ratio", 0.0))
    short_ratio = float(metrics.get("short_lane_ratio", 0.0))
    lane_count = int(metrics.get("lane_count", 0))

    score = 0.0
    score += min(text_coverage, 0.42) * 62.0
    score += min(lane_coverage, 0.46) * 6.0
    score += face_feature_coverage * 330.0
    score += min(face_text_coverage, 0.46) * 70.0
    score += identity_edge_alignment * 38.0
    score += identity_edge_recall * 36.0
    score += anchor_visibility * 28.0
    score += anchor_feature_ratio * 12.0
    score += background_cleanliness * 8.0
    score += min(lane_count, 90) * 0.012
    score -= source_leakage * 240.0
    score -= max(0.0, microtext_ratio - 7.2) * 12.0
    score -= max(0.0, 5.2 - microtext_ratio) * 3.0
    score -= max(0.0, text_coverage - 0.48) * 35.0
    score -= max(0.0, 0.20 - face_feature_coverage) * 130.0
    score -= max(0.0, 0.28 - face_text_coverage) * 90.0
    score -= max(0.0, 0.18 - identity_edge_recall) * 55.0
    score -= max(0.0, 0.26 - text_coverage) * 135.0
    score -= low_body_anchor_ratio * 38.0
    score -= max(0.0, word_repetition_ratio - 0.18) * 55.0
    score -= short_ratio * 40.0
    return round(score, 6)


def _candidate_metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "lane_count": metrics.get("lane_count"),
        "face_feature_coverage": metrics.get("face_feature_coverage"),
        "face_text_coverage": metrics.get("face_text_coverage"),
        "identity_edge_alignment": metrics.get("identity_edge_alignment"),
        "identity_edge_recall": metrics.get("identity_edge_recall"),
        "text_coverage_subject": metrics.get("text_coverage_subject"),
        "microtext_to_lane_ratio": metrics.get("microtext_to_lane_ratio"),
        "anchor_visibility_score": metrics.get("anchor_visibility_score"),
        "anchor_feature_lane_ratio": metrics.get("anchor_feature_lane_ratio"),
        "source_pixel_leakage": metrics.get("source_pixel_leakage"),
        "background_cleanliness": metrics.get("background_cleanliness"),
        "short_lane_ratio": metrics.get("short_lane_ratio"),
        "low_body_anchor_ratio": metrics.get("low_body_anchor_ratio"),
        "word_repetition_ratio": metrics.get("word_repetition_ratio"),
    }


def _copy_canonical_outputs(candidate_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL_OUTPUTS:
        src = candidate_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def _write_candidate_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_search.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _write_top_candidate_contact_sheet(records: list[dict[str, Any]], out_dir: Path) -> None:
    top = sorted(records, key=lambda item: float(item["score"]), reverse=True)[:5]
    images: list[tuple[dict[str, Any], Image.Image]] = []
    for record in top:
        path = Path(str(record["_candidate_dir"])) / "current_best.png"
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        img.thumbnail((260, 346))
        images.append((record, img.copy()))
    if not images:
        return
    label_h = 54
    pad = 12
    cell_w = 284
    cell_h = 346 + label_h + pad
    sheet = Image.new("RGB", (cell_w * len(images), cell_h), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for idx, (record, img) in enumerate(images):
        x = idx * cell_w + pad
        y = pad
        sheet.paste(img, (x, y))
        metrics = record["metrics"]
        label = (
            f"#{record['index']} score={record['score']:.2f}\n"
            f"face={float(metrics.get('face_feature_coverage') or 0):.3f} "
            f"anchor={float(metrics.get('anchor_visibility_score') or 0):.3f}\n"
            f"faceText={float(metrics.get('face_text_coverage') or 0):.3f} "
            f"micro={float(metrics.get('microtext_to_lane_ratio') or 0):.2f}"
        )
        draw.multiline_text((x, 358), label, fill=(235, 235, 235), spacing=3)
    debug_dir = out_dir / "debug_candidates"
    debug_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(debug_dir / "top_candidates_contact_sheet.png")


def run_auto_search(
    input_path: Path,
    profile: WordProfile,
    out_dir: Path,
    config: PipelineConfig,
    mask_path: Path | None = None,
    candidate_count: int = 32,
    seed: int = 23,
    debug_candidates: bool = False,
    runner: Callable[[Path, WordProfile, Path, PipelineConfig, Path | None], dict[str, Any]] = run_pipeline,
) -> dict[str, Any]:
    specs = generate_candidate_specs(config.style, candidate_count, seed)
    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_dir: Path | None = None

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".candidate_search_", dir=out_dir.parent) as tmp_name:
        tmp_root = Path(tmp_name)
        for spec in specs:
            candidate_dir = tmp_root / f"candidate_{spec.index:03d}"
            candidate_config = replace(
                config,
                style=style_for_candidate(config.style, spec),
                output_dir=candidate_dir,
            )
            metrics = runner(input_path, profile, candidate_dir, candidate_config, mask_path)
            score = score_candidate(metrics)
            record = {
                "index": spec.index,
                "candidate_id": spec.index,
                "score": score,
                "spec": asdict(spec),
                "metrics": _candidate_metrics_summary(metrics),
                "_candidate_dir": str(candidate_dir),
            }
            candidates.append(record)
            if best is None or score > float(best["score"]):
                best = {**record, "metrics_full": metrics}
                best_dir = candidate_dir

        if best is None or best_dir is None:
            raise RuntimeError("Auto candidate search did not produce any candidates.")
        _copy_canonical_outputs(best_dir, out_dir)
        if debug_candidates:
            _write_top_candidate_contact_sheet(candidates, out_dir)

    top_candidates = []
    for record in sorted(candidates, key=lambda item: float(item["score"]), reverse=True)[:5]:
        metrics = record["metrics"]
        top_candidates.append(
            {
                "score": record["score"],
                "candidate_id": record["candidate_id"],
                "lane_count": metrics.get("lane_count"),
                "face_feature_coverage": metrics.get("face_feature_coverage"),
                "face_text_coverage": metrics.get("face_text_coverage"),
                "identity_edge_alignment": metrics.get("identity_edge_alignment"),
                "identity_edge_recall": metrics.get("identity_edge_recall"),
                "text_coverage_subject": metrics.get("text_coverage_subject"),
                "microtext_to_lane_ratio": metrics.get("microtext_to_lane_ratio"),
                "anchor_visibility_score": metrics.get("anchor_visibility_score"),
                "anchor_feature_lane_ratio": metrics.get("anchor_feature_lane_ratio"),
                "source_pixel_leakage": metrics.get("source_pixel_leakage"),
                "background_cleanliness": metrics.get("background_cleanliness"),
                "short_lane_ratio": metrics.get("short_lane_ratio"),
                "best_config_parameters": record["spec"],
            }
        )
    public_candidates = []
    for record in candidates:
        public_candidates.append({key: value for key, value in record.items() if not key.startswith("_")})
    summary = {
        "seed": seed,
        "candidate_count": len(specs),
        "best_index": int(best["index"]),
        "best_score": float(best["score"]),
        "best_spec": best["spec"],
        "top_candidates": top_candidates,
        "candidates": public_candidates,
    }
    _write_candidate_summary(out_dir, summary)
    return dict(best["metrics_full"])
