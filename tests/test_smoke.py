from __future__ import annotations

import json

from PIL import Image

from adaptive_driving_assistant.domain import (
    AnalysisConfig,
    BoundingBox,
    CommunicationMode,
    Detection,
)
from adaptive_driving_assistant.pipeline import ImageAnalysisPipeline
from adaptive_driving_assistant.scene_readability import SceneReadabilityEstimator


class SmokeDetector:
    def detect(self, _image, confidence_threshold=None):
        return [
            Detection(
                class_name="car",
                confidence=0.9,
                bbox=BoundingBox(15, 15, 90, 90),
                normalized_center=(0.5, 0.5),
                area_ratio=0.18,
            )
        ]


def test_smoke_local_image_pipeline(tmp_path) -> None:
    image_path = tmp_path / "local_scene.png"
    Image.new("RGB", (128, 128), (170, 180, 190)).save(image_path)
    config = AnalysisConfig(
        communication_mode=CommunicationMode.MINIMAL,
        object_confidence_threshold=0.35,
    )
    pipeline = ImageAnalysisPipeline(SmokeDetector(), SceneReadabilityEstimator())

    result, report_path = pipeline.analyze(image_path, tmp_path / "out", config)

    assert result.object_counts == {"car": 1}
    assert "wykryto" in result.message.text.lower()
    assert "samochód" in result.message.text
    assert report_path is not None
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["configuration"]["detector_backend"] == "SmokeDetector"
    assert report["configuration"]["yolo_model"] is None
    assert report["configuration"]["yolo_model_sha256"] is None
