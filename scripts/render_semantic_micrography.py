from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glyphforge.semantic_micrography.config import PipelineConfig, RenderConfig, style_config
from glyphforge.semantic_micrography.auto import run_auto_search
from glyphforge.semantic_micrography.pipeline import run_pipeline
from glyphforge.semantic_micrography.profiles import load_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an SVG-first semantic digital micrography portrait.")
    parser.add_argument("--input", type=Path, required=True, help="Input person image.")
    parser.add_argument("--prompt", default=None, help="Comma/newline/semicolon separated semantic words.")
    parser.add_argument("--words", type=Path, default=None, help="Plain text words file.")
    parser.add_argument("--profile", type=Path, default=None, help="JSON semantic word profile.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--mask", type=Path, default=None, help="Optional binary subject mask override.")
    parser.add_argument("--subject", default=None, help="Optional subject name when using --prompt or --words.")
    parser.add_argument("--background", choices=("black", "original", "transparent"), default="black")
    parser.add_argument("--style", choices=("tribute_dark",), default="tribute_dark")
    parser.add_argument("--long-edge", type=int, default=1400)
    parser.add_argument("--auto", action="store_true", help="Search deterministic style candidates and keep the best canonical output.")
    parser.add_argument("--candidate-count", type=int, default=32, help="Number of auto-mode candidates to evaluate.")
    parser.add_argument("--seed", type=int, default=23, help="Seed for deterministic auto-mode candidate generation.")
    parser.add_argument("--debug-candidates", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--strict-mask-quality",
        action="store_true",
        help="Fail before rendering when subject-mask quality checks produce warnings.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    profile = load_profile(prompt=args.prompt, words_path=args.words, profile_path=args.profile, subject=args.subject)
    style = style_config(args.style)
    config = PipelineConfig(
        style=style,
        render=RenderConfig(
            background=args.background,
            style=args.style,
            strict_mask_quality=args.strict_mask_quality,
        ),
    )
    config = PipelineConfig(
        canvas=type(config.canvas)(long_edge=args.long_edge),
        style=config.style,
        regions=config.regions,
        render=config.render,
        seed=config.seed,
        output_dir=args.out_dir,
    )
    if args.auto:
        metrics = run_auto_search(
            args.input,
            profile,
            args.out_dir,
            config=config,
            mask_path=args.mask,
            candidate_count=args.candidate_count,
            seed=args.seed,
            debug_candidates=args.debug_candidates,
        )
    else:
        metrics = run_pipeline(args.input, profile, args.out_dir, config=config, mask_path=args.mask)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
