from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from adaptive_driving_assistant.domain import SceneReadabilityLevel
from adaptive_driving_assistant.scene_readability import SceneReadabilityEstimator

BUS_JPG = Path(__file__).resolve().parents[1] / "data" / "samples" / "bus.jpg"


def test_scene_readability_returns_features_and_thresholds() -> None:
    image = Image.new("RGB", (128, 128), (170, 190, 210))
    draw = ImageDraw.Draw(image)
    for index in range(0, 128, 8):
        draw.line((0, index, 127, 127 - index), fill=(20, 20, 20), width=2)
        draw.line((index, 0, 127 - index, 127), fill=(220, 40, 40), width=2)

    result = SceneReadabilityEstimator().estimate(image)

    assert result.level in {SceneReadabilityLevel.GOOD, SceneReadabilityLevel.MODERATE}
    assert {
        "brightness",
        "contrast",
        "sharpness",
        "saturation",
        "edge_density",
        "clip_hi",
    } <= set(result.features)
    assert result.thresholds["good_threshold"] > result.thresholds["moderate_threshold"]


def test_scene_readability_limited_for_dark_blurred_image() -> None:
    image = Image.new("RGB", (128, 128), (20, 20, 20)).filter(ImageFilter.GaussianBlur(radius=6))

    result = SceneReadabilityEstimator().estimate(image)

    assert result.level == SceneReadabilityLevel.LIMITED
    assert result.score < result.thresholds["moderate_threshold"]


def test_scene_readability_handles_tiny_image_without_nan() -> None:
    result = SceneReadabilityEstimator().estimate(Image.new("RGB", (1, 1), (0, 0, 0)))

    assert np.isfinite(result.score)


def test_scene_readability_scales_normalized_float_array() -> None:
    image = np.full((8, 8, 3), 0.5, dtype=np.float32)

    result = SceneReadabilityEstimator().estimate(image)

    assert 0.45 < result.features["brightness"] < 0.55


def test_table_7_4_5_bus_original_is_good() -> None:
    result = SceneReadabilityEstimator().estimate(Image.open(BUS_JPG))

    assert result.level == SceneReadabilityLevel.GOOD
    assert result.score >= result.thresholds["good_threshold"]


def test_table_7_4_5_bus_sigma30_is_limited() -> None:
    blurred = Image.open(BUS_JPG).convert("RGB").filter(ImageFilter.GaussianBlur(radius=30))
    result = SceneReadabilityEstimator().estimate(blurred)

    assert result.level == SceneReadabilityLevel.LIMITED
    assert result.score < result.thresholds["moderate_threshold"]


def test_table_7_4_5_bus_overexposed_x3_is_limited() -> None:
    original = Image.open(BUS_JPG).convert("RGB")
    array = np.clip(np.asarray(original, dtype=np.float32) * 3.0, 0, 255).astype(np.uint8)
    result = SceneReadabilityEstimator().estimate(Image.fromarray(array))

    assert result.level == SceneReadabilityLevel.LIMITED
    assert result.score < result.thresholds["moderate_threshold"]
    assert result.features["clip_hi"] > 0.5
