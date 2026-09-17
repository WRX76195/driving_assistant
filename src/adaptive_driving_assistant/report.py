"""Generowanie raportu JSON z pojedynczej analizy obrazu.

Moduł serializuje wynik pipeline do pliku z metadanymi środowiska,
konfiguracją, detekcjami, heurystyką, priorytetem i komunikatem.
Sumy SHA-256 plików wejściowych i wynikowych umożliwiają weryfikację powtarzalności.
"""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from adaptive_driving_assistant.config import (
    APP_VERSION,
    PRIORITY_EXPLANATION,
    SCENE_READABILITY_EXPLANATION,
)
from adaptive_driving_assistant.detector import file_sha256
from adaptive_driving_assistant.domain import (
    Detection,
    PipelineResult,
    communication_mode_label,
    priority_level_label,
    scene_readability_level_label,
    to_jsonable,
)


def build_report(result: PipelineResult) -> dict[str, Any]:
    source_path = Path(result.source_path)
    annotated_path = Path(result.annotated_image_path)
    image_width, image_height = _image_size(source_path)
    return {
        "file_name": source_path.name,
        "file_type": result.source_type,
        "analysis_datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "application_version": APP_VERSION,
        "runtime": _runtime_manifest(),
        "configuration": {
            "communication_mode": result.config.communication_mode.value,
            "communication_mode_label": communication_mode_label(result.config.communication_mode),
            "object_confidence_threshold": result.config.object_confidence_threshold,
            "detector_backend": result.detector.backend,
            "yolo_model": result.detector.model_name,
            "yolo_model_sha256": result.detector.model_sha256,
        },
        "image_width": image_width,
        "image_height": image_height,
        "input_sha256": file_sha256(source_path) if source_path.is_file() else None,
        "detections": to_jsonable(result.detections),
        "object_counts": result.object_counts,
        "scene_readability": {
            **to_jsonable(result.scene_readability),
            "label": "heurystyczny wskaźnik cech technicznych obrazu",
            "level_label": scene_readability_level_label(result.scene_readability.level),
            "explanation": SCENE_READABILITY_EXPLANATION,
        },
        "priority": {
            **to_jsonable(result.priority),
            "level_label": priority_level_label(result.priority.level),
            "label": result.priority.label,
            "explanation": PRIORITY_EXPLANATION,
        },
        "communication_mode": communication_mode_label(result.message.mode),
        "message": {
            "text": result.message.text,
            "mode": result.message.mode.value,
            "mode_label": communication_mode_label(result.message.mode),
            "priority": result.message.priority.value,
            "priority_label": priority_level_label(result.message.priority),
            "facts": result.message.facts,
        },
        "processing_time_seconds": result.processing_time_seconds,
        "annotated_image_file": annotated_path.name,
        "annotated_image_sha256": file_sha256(annotated_path) if annotated_path.is_file() else None,
        "limitations": result.limitations,
    }


def write_report(result: PipelineResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_report(result), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def count_objects(detections: list[Detection]) -> dict[str, int]:
    return dict(sorted(Counter(detection.class_name for detection in detections).items()))


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _image_size(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return int(width), int(height)
    except OSError:
        return None, None


def _runtime_manifest() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "pillow", "streamlit", "ultralytics", "torch")
        },
    }
