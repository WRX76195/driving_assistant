"""Proste narzędzie CLI do analizy pojedynczego obrazu drogowego."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_driving_assistant.config import DEFAULT_OBJECT_CONFIDENCE
from adaptive_driving_assistant.detector import (
    ObjectDetector,
    ObjectDetectorUnavailableError,
    validate_local_yolo_model,
)
from adaptive_driving_assistant.domain import (
    AnalysisConfig,
    CommunicationMode,
    scene_readability_level_label,
)
from adaptive_driving_assistant.pipeline import (
    ImageAnalysisPipeline,
    ensure_data_dirs,
    timestamped_output_dir,
)

MODE_CHOICES = {
    "minimalny": CommunicationMode.MINIMAL,
    "rozszerzony": CommunicationMode.EXTENDED,
    "rozszerzony-z-objasnieniem": CommunicationMode.EXTENDED_WITH_EXPLANATION,
    "rozszerzony-z-objaśnieniem": CommunicationMode.EXTENDED_WITH_EXPLANATION,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analizuj pojedynczy obraz drogowy lokalnym potokiem YOLO."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Ścieżka do obrazu JPG, JPEG lub PNG",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_CHOICES),
        default="rozszerzony",
        help="Tryb prezentacji (domyślnie: rozszerzony)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_OBJECT_CONFIDENCE,
        help=f"Próg confidence YOLO (domyślnie: {DEFAULT_OBJECT_CONFIDENCE})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_data_dirs()

    try:
        config = AnalysisConfig(
            communication_mode=MODE_CHOICES[args.mode],
            object_confidence_threshold=args.confidence,
        )
        validate_local_yolo_model()
        pipeline = ImageAnalysisPipeline(detector=ObjectDetector())
        output_dir = timestamped_output_dir()
        result, report_path = pipeline.analyze(args.image_path, output_dir, config)
    except (ValueError, ObjectDetectorUnavailableError) as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    print("=== Podsumowanie analizy ===")
    print(f"Obraz: {result.source_path}")
    print(f"Liczba detekcji: {sum(result.object_counts.values())}")
    print(f"Obiekty: {result.object_counts or 'brak'}")
    print(
        "Wskaźnik cech obrazu: "
        f"{scene_readability_level_label(result.scene_readability.level)} "
        f"({result.scene_readability.score:.2f})"
    )
    print(f"Priorytet: {result.priority.label}")
    print(f"Komunikat: {result.message.text}")
    print(f"Oznaczony obraz: {result.annotated_image_path}")
    if report_path:
        print(f"Raport JSON: {report_path}")
    print(f"Czas przetwarzania: {result.processing_time_seconds} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
