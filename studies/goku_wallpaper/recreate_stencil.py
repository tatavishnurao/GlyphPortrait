from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glyphforge.recipes.word_tribute import run_cli

DEFAULT_INPUT = ROOT / "reference_img" / "goku-sky-reference.jpg"
DEFAULT_OUT = ROOT / "studies" / "goku_wallpaper" / "outputs"
DEFAULT_PROFILE = Path(__file__).resolve().with_name("goku_profile.json")


def main() -> None:
    run_cli(
        default_input=DEFAULT_INPUT,
        default_out=DEFAULT_OUT,
        default_profile=DEFAULT_PROFILE,
        default_background="black",
    )


if __name__ == "__main__":
    main()
