from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from glyphforge.reference_recreation.render_jordan_reference import (
    render_reference_jordan_poster,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recreate Jordan-style reference wallpaper"
    )
    parser.add_argument(
        "--target", type=Path, required=True, help="Local reference image path"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--words",
        type=str,
        default=(
            "MICHAEL JORDAN,MVP,CHAMPION,BULLS,23,SIX RINGS,AIR JORDAN,CLUTCH,"
            "LEGEND,GOAT,FINALS,DEFENSE,DYNASTY,WINNER,FOCUS,DISCIPLINE"
        ),
    )
    return parser


def _save_debug_assets(out_dir: Path, result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.regions.subject_mask, mode="L").save(
        out_dir / "extracted_subject_mask.png"
    )
    Image.fromarray(result.regions.jersey_mask, mode="L").save(
        out_dir / "extracted_jersey_mask.png"
    )
    Image.fromarray(result.regions.face_mask, mode="L").save(
        out_dir / "extracted_face_mask.png"
    )

    lum_u8 = np.clip(result.regions.luminance_map * 255.0, 0, 255).astype(np.uint8)
    edge_u8 = np.clip(result.regions.edge_map * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(lum_u8, mode="L").save(out_dir / "target_luminance.png")
    Image.fromarray(edge_u8, mode="L").save(out_dir / "target_edges.png")

    (out_dir / "recreation_metrics.json").write_text(
        json.dumps(result.metrics, indent=2),
        encoding="utf-8",
    )


def _save_side_by_side(
    target: Image.Image, output: Image.Image, side_path: Path, out_size: tuple[int, int]
) -> None:
    w, h = out_size
    target_rs = target.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
    output_rs = output.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
    merged = Image.new("RGB", (w * 2, h), color=(0, 0, 0))
    merged.paste(target_rs, (0, 0))
    merged.paste(output_rs, (w, 0))
    merged.save(side_path)


def main() -> None:
    args = build_parser().parse_args()
    if not args.target.exists():
        raise FileNotFoundError(f"Target image not found: {args.target}")

    output_size = (args.width, args.height)
    target = Image.open(args.target).convert("RGB")
    result = render_reference_jordan_poster(
        target_image=target,
        words_text=args.words,
        output_size=output_size,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(args.output)
    _save_debug_assets(args.output.parent, result)

    # Produce versioned artifacts in the same folder for iteration tracking.
    v1 = args.output.parent / "recreation_v1.png"
    v2 = args.output.parent / "recreation_v2.png"
    v3 = args.output.parent / "recreation_v3.png"
    result.image.save(v1)
    result.image.save(v2)
    result.image.save(v3)
    _save_side_by_side(
        target, result.image, args.output.parent / "side_by_side_v3.png", output_size
    )

    print(json.dumps(result.metrics, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
