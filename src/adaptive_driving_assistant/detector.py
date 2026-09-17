"""Adapter Ultralytics YOLO dla wybranych obiektów drogowych.

Moduł ładuje lokalny model yolo11n.pt, weryfikuje jego sumę SHA-256
i mapuje wyniki inferencji na struktury domenowe projektu. Obsługiwane
klasy to m.in. person, car, bus, truck, bicycle i motorcycle.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_driving_assistant.config import (
    DEFAULT_OBJECT_CONFIDENCE,
    DEFAULT_YOLO_MODEL,
    EXPECTED_YOLO_SHA256,
    MATPLOTLIB_CONFIG_DIR,
    PROJECT_ROOT,
    SUPPORTED_OBJECT_CLASSES,
    ULTRALYTICS_CONFIG_DIR,
    YOLO_MODEL_PATH,
)
from adaptive_driving_assistant.domain import BoundingBox, Detection, DetectorManifest


class ObjectDetectorUnavailableError(RuntimeError):
    """Raised when the YOLO backend cannot be loaded."""


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_local_yolo_model(
    model_path: str | Path = YOLO_MODEL_PATH,
    expected_sha256: str | None = EXPECTED_YOLO_SHA256,
) -> Path:
    raw_path = Path(model_path)
    resolved = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise ObjectDetectorUnavailableError(
            "Brak lokalnego pliku modelu YOLO. Umieść plik "
            f"{DEFAULT_YOLO_MODEL} w katalogu projektu: {PROJECT_ROOT}"
        )
    if expected_sha256 and file_sha256(resolved) != expected_sha256.lower():
        raise ObjectDetectorUnavailableError(
            "Plik modelu YOLO ma nieoczekiwaną sumę SHA-256. "
            "Przywróć zweryfikowany plik yolo11n.pt dostarczony z projektem."
        )
    return resolved


class ObjectDetector:
    def __init__(
        self,
        model_path: str | Path = YOLO_MODEL_PATH,
        confidence_threshold: float = DEFAULT_OBJECT_CONFIDENCE,
        supported_classes: Iterable[str] = SUPPORTED_OBJECT_CLASSES,
        model: Any | None = None,
        predict_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.supported_classes = set(supported_classes)
        self._model = model
        self.predict_overrides = dict(predict_overrides or {})

    def manifest(self) -> DetectorManifest:
        raw_path = self.model_path
        resolved = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        resolved = resolved.resolve()
        return DetectorManifest(
            backend="ultralytics_yolo",
            model_name=resolved.name,
            model_sha256=file_sha256(resolved) if resolved.is_file() else None,
        )

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[Detection]:
        threshold = (
            self.confidence_threshold if confidence_threshold is None else confidence_threshold
        )
        if not 0.0 < float(threshold) <= 1.0:
            raise ValueError("Próg confidence musi należeć do przedziału (0, 1].")
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return []

        model = self._load_model()
        predict_kwargs = {
            key: value
            for key, value in self.predict_overrides.items()
            if key not in {"source", "conf", "verbose"}
        }
        results = model.predict(
            source=image,
            conf=threshold,
            verbose=False,
            **predict_kwargs,
        )
        if not results:
            return []

        names = getattr(results[0], "names", getattr(model, "names", {}))
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_numpy(boxes.xyxy)
        confidences = _to_numpy(boxes.conf)
        classes = _to_numpy(boxes.cls).astype(int)

        detections: list[Detection] = []
        image_area = float(width * height)
        if not (len(xyxy) == len(confidences) == len(classes)):
            raise ObjectDetectorUnavailableError("Model zwrócił niespójne tablice detekcji.")
        for bbox_values, confidence, class_idx in zip(xyxy, confidences, classes, strict=True):
            class_name = (
                names.get(int(class_idx), str(class_idx))
                if isinstance(names, dict)
                else str(class_idx)
            )
            if class_name not in self.supported_classes:
                continue
            raw_x1, raw_y1, raw_x2, raw_y2 = [float(value) for value in bbox_values]
            x1 = min(max(raw_x1, 0.0), float(width))
            y1 = min(max(raw_y1, 0.0), float(height))
            x2 = min(max(raw_x2, 0.0), float(width))
            y2 = min(max(raw_y2, 0.0), float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=bbox,
                    normalized_center=(((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height),
                    area_ratio=bbox.area / image_area,
                )
            )
        return detections

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
        os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
        model_path = validate_local_yolo_model(self.model_path)
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ObjectDetectorUnavailableError(
                "Biblioteka ultralytics nie jest zainstalowana. Zainstaluj requirements.txt."
            ) from exc
        self._model = YOLO(str(model_path))
        return self._model


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)
