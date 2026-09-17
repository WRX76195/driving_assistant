"""Heurystyczny wskaźnik wybranych cech technicznych obrazu.

Moduł oblicza jasność, kontrast, ostrość, nasycenie, gęstość krawędzi
oraz udział kanałów nasyconych do 255, a następnie mapuje je na poziom:
dobry, umiarkowany lub ograniczony. Wynik opisuje cechy pliku, nie czytelność
sceny dla człowieka, warunki drogowe ani widzialność meteorologiczną.
Nazwa modułu pozostaje historyczna.

Od wersji 0.3.6 stałe normalizacyjne i wagi dobrano tak, by test
przechodził na bus.jpg: skrajne rozmycie (σ = 30) i prześwietlenie (×3,0)
dają kategorię „ograniczony”."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from adaptive_driving_assistant.domain import SceneReadabilityLevel, SceneReadabilityResult

FORMULA_VERSION = "0.3.6"
TARGET_BRIGHTNESS = 0.55
BRIGHTNESS_WIDTH = 0.55
CONTRAST_FLOOR = 0.12
CONTRAST_SPAN = 0.15
SHARPNESS_SCALE = 0.02
EDGE_SCALE = 0.20
SATURATION_SCALE = 0.35
CLIP_HI_PENALTY = 0.85
WEIGHT_BRIGHTNESS = 0.15
WEIGHT_CONTRAST = 0.45
WEIGHT_DETAIL = 0.28
WEIGHT_SATURATION = 0.12


class SceneReadabilityEstimator:
    def __init__(self, good_threshold: float = 0.58, moderate_threshold: float = 0.38) -> None:
        self.good_threshold = good_threshold
        self.moderate_threshold = moderate_threshold

    def estimate(self, image: Image.Image | np.ndarray) -> SceneReadabilityResult:
        features = extract_features(_to_pil_image(image))
        score = _score(features)
        if score >= self.good_threshold:
            level = SceneReadabilityLevel.GOOD
        elif score >= self.moderate_threshold:
            level = SceneReadabilityLevel.MODERATE
        else:
            level = SceneReadabilityLevel.LIMITED
        return SceneReadabilityResult(
            level=level,
            features=features,
            thresholds={
                "good_threshold": self.good_threshold,
                "moderate_threshold": self.moderate_threshold,
            },
            score=score,
            method=f"heuristic_image_features_{FORMULA_VERSION}",
        )


def extract_features(image: Image.Image) -> dict[str, float]:
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    gray_arr = np.asarray(gray).astype("float32") / 255.0
    rgb_arr = np.asarray(rgb).astype("float32") / 255.0
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    return {
        "brightness": float(gray_arr.mean()),
        "contrast": float(gray_arr.std()),
        "sharpness": _sharpness(gray_arr),
        "saturation": _mean_saturation(rgb_arr),
        "edge_density": _edge_density(gray),
        "clip_hi": float(np.mean(rgb_u8 >= 255)),
    }


def _score(features: dict[str, float]) -> float:
    brightness = max(
        0.0, 1.0 - abs(features["brightness"] - TARGET_BRIGHTNESS) / BRIGHTNESS_WIDTH
    )
    contrast = min(1.0, max(0.0, (features["contrast"] - CONTRAST_FLOOR) / CONTRAST_SPAN))
    sharpness = min(1.0, features["sharpness"] / SHARPNESS_SCALE)
    edge_density = min(1.0, features["edge_density"] / EDGE_SCALE)
    detail = 0.5 * sharpness + 0.5 * edge_density
    saturation = min(1.0, features["saturation"] / SATURATION_SCALE)
    raw = (
        WEIGHT_BRIGHTNESS * brightness
        + WEIGHT_CONTRAST * contrast
        + WEIGHT_DETAIL * detail
        + WEIGHT_SATURATION * saturation
    )
    return float(max(0.0, raw - CLIP_HI_PENALTY * features.get("clip_hi", 0.0)))


def _sharpness(gray_arr: np.ndarray) -> float:
    if gray_arr.shape[0] < 3 or gray_arr.shape[1] < 3:
        return 0.0
    center = gray_arr[1:-1, 1:-1] * 4
    laplacian = (
        center - gray_arr[:-2, 1:-1] - gray_arr[2:, 1:-1] - gray_arr[1:-1, :-2] - gray_arr[1:-1, 2:]
    )
    return float(np.var(laplacian))


def _mean_saturation(rgb_arr: np.ndarray) -> float:
    max_channel = rgb_arr.max(axis=2)
    min_channel = rgb_arr.min(axis=2)
    saturation = np.zeros_like(max_channel)
    np.divide(max_channel - min_channel, max_channel, out=saturation, where=max_channel > 0)
    return float(np.mean(saturation))


def _edge_density(gray: Image.Image) -> float:
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edges).astype("float32") / 255.0
    threshold = max(0.12, float(ImageStat.Stat(edges).mean[0]) / 255.0)
    return float(np.mean(edge_arr > threshold))


def _to_pil_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim not in {2, 3}:
        raise ValueError("Obraz musi być tablicą dwu- albo trójwymiarową.")
    if not np.isfinite(array).all():
        raise ValueError("Obraz zawiera wartości NaN lub nieskończone.")
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0.0, 255.0)
    if array.ndim == 2:
        return Image.fromarray(array.astype("uint8"), mode="L").convert("RGB")
    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError("Obraz kolorowy musi mieć trzy kanały RGB albo cztery kanały RGBA.")
    if array.shape[-1] == 4:
        return Image.fromarray(array.astype("uint8"), mode="RGBA").convert("RGB")
    return Image.fromarray(array.astype("uint8"), mode="RGB")
