"""Zapisuje Tabelę 7.7b (kalibracja wskaźnika 0.3.6) dla obrazu bus.jpg."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_driving_assistant.scene_readability import (  # noqa: E402
    FORMULA_VERSION,
    SceneReadabilityEstimator,
)

BUS_JPG = PROJECT_ROOT / "data" / "samples" / "bus.jpg"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "scene_readability_table_7_7_0.3.6.json"
SIGMAS = (0, 1, 2, 3, 5, 8, 10, 15, 30)
EXPOSURE = (0.20, 0.35, 0.50, 0.75, 1.50, 2.20, 3.00)


def _row(label: str, image: Image.Image, estimator: SceneReadabilityEstimator) -> dict:
    result = estimator.estimate(image)
    features = {key: round(value, 4) for key, value in result.features.items()}
    return {
        "wariant": label,
        **features,
        "S": round(result.score, 4),
        "kategoria": {
            "dobra": "dobry",
            "umiarkowana": "umiarkowany",
            "ograniczona": "ograniczony",
        }[result.level.value],
    }


def build_table(bus_path: Path = BUS_JPG) -> dict:
    estimator = SceneReadabilityEstimator()
    original = Image.open(bus_path).convert("RGB")
    array = np.asarray(original, dtype=np.float32)
    rows = []
    for sigma in SIGMAS:
        image = (
            original
            if sigma == 0
            else original.filter(ImageFilter.GaussianBlur(radius=sigma))
        )
        suffix = " (obraz nieczytelny)" if sigma == 30 else ""
        label = "oryginał (σ = 0)" if sigma == 0 else f"rozmycie σ = {sigma}{suffix}"
        rows.append(_row(label, image, estimator))
    for multiplier in EXPOSURE:
        exposed = Image.fromarray(np.clip(array * multiplier, 0, 255).astype(np.uint8))
        label = f"ekspozycja × {multiplier:.2f}".replace(".", ",")
        rows.append(_row(label, exposed, estimator))
    blur_path = bus_path.with_name("blur.png")
    if blur_path.is_file():
        rows.append(
            _row(
                "blur.png z data/samples (odniesienie)",
                Image.open(blur_path).convert("RGB"),
                estimator,
            )
        )
    original_row = next(row for row in rows if row["wariant"].startswith("oryginał"))
    sigma30 = next(row for row in rows if "σ = 30" in row["wariant"])
    over = next(row for row in rows if row["wariant"].endswith("3,00"))
    return {
        "formula_version": FORMULA_VERSION,
        "image": bus_path.name,
        "degradation": (
            "Pillow ImageFilter.GaussianBlur(radius=σ); ekspozycja = clip(RGB·k, 0, 255)"
        ),
        "acceptance": {
            "original_good": original_row["kategoria"] == "dobry",
            "sigma30_limited": sigma30["kategoria"] == "ograniczony",
            "exposure_x3_limited": over["kategoria"] == "ograniczony",
        },
        "rows": rows,
    }


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    payload = build_table()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano {out}")
    print(json.dumps(payload["acceptance"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
