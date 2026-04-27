from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_name: str = "GlyphForge"
    default_seed: int = 42
    default_density: float = 0.65
    default_max_words: int = 600
    default_attempts_per_word: int = 180
    default_long_edge: int = 1600
    outputs_dir: Path = Path("examples/outputs")
    fonts_dir: Path = Path("assets/fonts")
