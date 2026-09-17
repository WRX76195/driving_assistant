"""Potok analizy pojedynczego obrazu drogowego.

Moduł łączy detekcję YOLO, heurystyczny wskaźnik cech obrazu, priorytetyzację
i generowanie komunikatu w jednym przebiegu. Zawiera też walidację wejścia
oraz zapis oznaczonego obrazu i raportu JSON.
"""

from __future__ import annotations

import re
import time
import uuid
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from adaptive_driving_assistant.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_PIL_FORMATS,
    DEFAULT_LIMITATIONS,
    INPUT_DIR,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    MIN_IMAGE_SIDE,
    OUTPUT_DIR,
)
from adaptive_driving_assistant.detector import ObjectDetector
from adaptive_driving_assistant.domain import (
    AnalysisConfig,
    Detection,
    DetectorManifest,
    PipelineResult,
    scene_readability_level_label,
)
from adaptive_driving_assistant.messages import generate_message
from adaptive_driving_assistant.priority import assess_priority
from adaptive_driving_assistant.report import count_objects, write_report
from adaptive_driving_assistant.scene_readability import SceneReadabilityEstimator


class ImageAnalysisPipeline:
    def __init__(
        self,
        detector: ObjectDetector,
        readability_estimator: SceneReadabilityEstimator | None = None,
    ) -> None:
        self.detector = detector
        self.readability_estimator = readability_estimator or SceneReadabilityEstimator()

    def analyze(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        config: AnalysisConfig,
        write_json_report: bool = True,
    ) -> tuple[PipelineResult, Path | None]:
        start = time.perf_counter()
        # 1. Walidacja ścieżki i przygotowanie katalogu wynikowego
        path = validate_image_path(image_path)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        # 2. Wczytanie obrazu i detekcja obiektów YOLO
        image = load_validated_image(path)
        image_array = np.asarray(image)
        detections = self.detector.detect(
            image_array,
            confidence_threshold=config.object_confidence_threshold,
        )
        # 3. Heurystyczny wskaźnik cech obrazu
        scene_readability = self.readability_estimator.estimate(image)
        # 4. Wyznaczenie priorytetu komunikatu
        priority = assess_priority(detections, scene_readability)
        # 5. Generowanie komunikatu w wybranym trybie
        message = generate_message(
            detections=detections,
            scene_readability=scene_readability,
            priority=priority,
            mode=config.communication_mode,
        )

        # 6. Zapis oznaczonego obrazu
        annotated_path = output / f"{safe_stem(path.name)}_oznaczony.jpg"
        save_annotated_image(
            image=image,
            detections=detections,
            output_path=annotated_path,
            title_lines=[
                "wskaźnik cech obrazu: "
                f"{scene_readability_level_label(scene_readability.level)}"
            ],
        )
        # 7. Złożenie wyniku pipeline
        result = PipelineResult(
            source_path=str(path),
            source_type="image",
            annotated_image_path=str(annotated_path),
            detections=detections,
            object_counts=count_objects(detections),
            scene_readability=scene_readability,
            priority=priority,
            message=message,
            processing_time_seconds=round(time.perf_counter() - start, 3),
            config=config,
            detector=self._detector_manifest(),
            limitations=DEFAULT_LIMITATIONS,
        )
        report_path = None
        # 8. Opcjonalny zapis raportu JSON
        if write_json_report:
            report_path = output / f"{safe_stem(path.name)}_raport.json"
            write_report(result, report_path)
        return result, report_path

    def _detector_manifest(self) -> DetectorManifest:
        if isinstance(self.detector, ObjectDetector):
            return self.detector.manifest()
        return DetectorManifest(
            backend=type(self.detector).__name__,
            model_name=None,
            model_sha256=None,
        )


def ensure_data_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_image_path(path: str | Path) -> Path:
    raw_path = str(path)
    if "://" in raw_path:
        raise ValueError("Dozwolone są tylko lokalne pliki przesłane przez użytkownika.")
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"Plik nie istnieje albo nie jest zwykłym plikiem: {resolved}")
    if resolved.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("Obsługiwane są tylko obrazy JPG, JPEG i PNG.")
    if resolved.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError(f"Plik obrazu przekracza limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    return resolved


def write_uploaded_image(uploaded_file, destination_dir: Path = INPUT_DIR) -> Path:
    original_name = Path(uploaded_file.name).name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("Obsługiwane są tylko obrazy JPG, JPEG i PNG.")

    payload = bytes(uploaded_file.getbuffer())
    validate_image_bytes(payload, original_name)

    destination_root = Path(destination_dir).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    unique_prefix = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:6]}"
    destination = (
        destination_root / f"{unique_prefix}_{safe_stem(original_name)}{extension}"
    ).resolve()
    if destination_root not in destination.parents:
        raise ValueError("Nieprawidłowa ścieżka zapisu przesłanego obrazu.")
    destination.write_bytes(payload)
    return destination


def validate_image_bytes(payload: bytes, file_name: str) -> Image.Image:
    extension = Path(file_name).suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError("Obsługiwane są tylko obrazy JPG, JPEG i PNG.")
    if not payload:
        raise ValueError("Przesłany plik jest pusty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Plik obrazu przekracza limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as probe:
                image_format = probe.format
                width, height = probe.size
                probe.verify()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError("Plik nie jest poprawnym obrazem JPG ani PNG.") from exc

    if image_format not in ALLOWED_PIL_FORMATS:
        raise ValueError("Plik nie jest poprawnym obrazem JPG ani PNG.")
    expected_format = "PNG" if extension == ".png" else "JPEG"
    if image_format != expected_format:
        raise ValueError("Rozszerzenie pliku nie zgadza się z rzeczywistym formatem obrazu.")
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"Każdy bok obrazu musi mieć co najmniej {MIN_IMAGE_SIDE} piksele.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Obraz przekracza limit {MAX_IMAGE_PIXELS:,} pikseli.")

    try:
        with Image.open(BytesIO(payload)) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB").copy()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Nie udało się bezpiecznie odczytać obrazu.") from exc


def load_validated_image(path: str | Path) -> Image.Image:
    resolved = validate_image_path(path)
    return validate_image_bytes(resolved.read_bytes(), resolved.name)


def timestamped_output_dir(base_dir: Path = OUTPUT_DIR) -> Path:
    output = Path(base_dir) / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=True)
    return output


def safe_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9_-]+", "_", stem)
    return stem.strip("_") or "obraz"


def save_annotated_image(
    image: Image.Image,
    detections: list[Detection],
    output_path: str | Path,
    title_lines: list[str] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for detection in detections:
        color = _color_for_class(detection.class_name)
        box = detection.bbox
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=3)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_box = draw.textbbox((box.x1, box.y1), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((box.x1, box.y1), label, fill=(0, 0, 0), font=font)
    if title_lines:
        y = 6
        for line in title_lines:
            text_box = draw.textbbox((6, y), line, font=font)
            draw.rectangle(text_box, fill=(255, 255, 255))
            draw.text((6, y), line, fill=(0, 0, 0), font=font)
            y += text_box[3] - text_box[1] + 4
    annotated.save(output)
    return output


def _color_for_class(class_name: str) -> tuple[int, int, int]:
    return {
        "person": (220, 40, 40),
        "car": (40, 90, 220),
        "bus": (120, 70, 200),
        "truck": (120, 70, 200),
        "bicycle": (30, 150, 90),
        "motorcycle": (30, 150, 90),
    }.get(class_name, (255, 255, 255))
