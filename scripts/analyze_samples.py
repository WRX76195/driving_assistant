"""Uruchamia potok na mini zbiorze data/samples i zapisuje podsumowanie."""

from __future__ import annotations

import json
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

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
OUTPUT_JSON = SAMPLES_DIR / "batch_results_0.3.6.json"

THESIS_MINI_SET = (
    "bus.jpg",
    "street.jpg",
    "dark.png",
    "bright.png",
    "blur.png",
    "empty_scene.png",
)


def main() -> int:
    ensure_data_dirs()
    try:
        validate_local_yolo_model()
        pipeline = ImageAnalysisPipeline(detector=ObjectDetector())
    except ObjectDetectorUnavailableError as exc:
        print(f"Błąd: {exc}", file=sys.stderr)
        return 1

    config = AnalysisConfig(
        communication_mode=CommunicationMode.EXTENDED,
        object_confidence_threshold=DEFAULT_OBJECT_CONFIDENCE,
    )
    rows: list[dict] = []
    images = [SAMPLES_DIR / name for name in THESIS_MINI_SET]
    missing = [path.name for path in images if not path.is_file()]
    if missing:
        print(
            "Brak wymaganych obrazów mini zbioru: "
            + ", ".join(missing)
            + ". Uruchom scripts/prepare_samples.py i uzupełnij pliki lokalne.",
            file=sys.stderr,
        )
        return 1

    for path in images:
        try:
            result, _report = pipeline.analyze(path, timestamped_output_dir(), config)
        except ValueError as exc:
            print(f"{path.name}: BŁĄD — {exc}", file=sys.stderr)
            continue
        row = {
            "file": path.name,
            "counts": result.object_counts,
            "readability": result.scene_readability.level.value,
            "score": round(result.scene_readability.score, 6),
            "priority": result.priority.level.value,
            "priority_score": result.priority.score,
            "primary": result.priority.primary_detection_class,
            "message": result.message.text,
            "seconds": result.processing_time_seconds,
        }
        rows.append(row)
        counts = result.object_counts or "brak"
        print(
            f"{path.name}: {counts} | "
            f"wskaźnik={scene_readability_level_label(result.scene_readability.level)} "
            f"({row['score']}) | priorytet={row['priority']}"
        )

    OUTPUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano: {OUTPUT_JSON} ({len(rows)} obrazów)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
