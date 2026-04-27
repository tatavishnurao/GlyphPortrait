from pathlib import Path

from PIL import Image

from glyphforge.image.export import save_png


def test_save_png_creates_file(tmp_path: Path):
    out = tmp_path / "poster.png"
    image = Image.new("RGB", (256, 256), color=(10, 20, 30))

    result_path = save_png(image=image, output_path=out, dpi=300)

    assert result_path.exists()
    assert result_path.suffix.lower() == ".png"
