from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch import recreate_jordan_reference_stencil_v5

SOURCE_DIR = ROOT / "examples" / "reference_recreation"
OUTPUT_DIR = ROOT / "studies" / "jordan_wallpaper" / "outputs"

CANONICAL_ARTIFACTS = {
    "stencil_v5_final.png": "current_best.png",
    "stencil_v5_side_by_side.png": "current_best_side_by_side.png",
    "stencil_v5_face_insert_mask.png": "current_best_face_insert_mask.png",
    "stencil_v5_seam_overlay.png": "current_best_seam_overlay.png",
    "stencil_v5_face_crop_inserted.png": "current_best_face_crop_inserted.png",
    "stencil_v5_metrics.json": "current_best_metrics.json",
}


def copy_canonical_artifacts() -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for source_name, target_name in CANONICAL_ARTIFACTS.items():
        source = SOURCE_DIR / source_name
        target = OUTPUT_DIR / target_name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
        copied[source_name] = str(target.relative_to(ROOT))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the locked Jordan reference-stencil v5 study artifacts."
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Copy existing v5 artifacts without rerunning the renderer.",
    )
    args = parser.parse_args()

    if not args.skip_render:
        recreate_jordan_reference_stencil_v5.main()

    copied = copy_canonical_artifacts()
    metrics_path = OUTPUT_DIR / "current_best_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(json.dumps({"artifacts": copied, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
