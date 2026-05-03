from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image


ICON_SIZES = [(16, 16), (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "assets" / "icons" / "tz-ollama-utils-exploded.png"
    output = repo_root / "assets" / "icons" / "tz_ollama_utils_icon.ico"
    package_output = (
        repo_root / "src" / "tz_ollama_utils" / "assets" / "icons" / "tz_ollama_utils_icon.ico"
    )

    if not source.exists():
        raise FileNotFoundError(f"Source image not found: {source}")

    with Image.open(source) as image:
        image = image.convert("RGBA")
        image.save(output, format="ICO", sizes=ICON_SIZES)

    package_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, package_output)

    print(f"Wrote {output}")
    print(f"Copied to {package_output}")
    print(f"Embedded sizes: {', '.join(f'{w}x{h}' for w, h in ICON_SIZES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
