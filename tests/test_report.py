from __future__ import annotations

import json

from adaptive_driving_assistant.domain import (
    AnalysisConfig,
    CommunicationMode,
    DetectorManifest,
    GeneratedMessage,
    PipelineResult,
    PriorityLevel,
    PriorityResult,
    SceneReadabilityLevel,
    SceneReadabilityResult,
)
from adaptive_driving_assistant.report import write_report


def _result() -> PipelineResult:
    config = AnalysisConfig(
        communication_mode=CommunicationMode.EXTENDED,
        object_confidence_threshold=0.35,
    )
    readability = SceneReadabilityResult(
        level=SceneReadabilityLevel.GOOD,
        features={"brightness": 0.7, "contrast": 0.2},
        thresholds={"good_threshold": 0.58, "moderate_threshold": 0.38},
        score=0.7,
    )
    priority = PriorityResult(PriorityLevel.NO_MESSAGE, 0.0, [])
    message = GeneratedMessage(
        "Brak istotnego komunikatu.", CommunicationMode.EXTENDED, priority.level, []
    )
    return PipelineResult(
        source_path="input/scene.jpg",
        source_type="image",
        annotated_image_path="output/scene_oznaczony.jpg",
        detections=[],
        object_counts={},
        scene_readability=readability,
        priority=priority,
        message=message,
        processing_time_seconds=0.1,
        config=config,
        detector=DetectorManifest(
            backend="test_detector",
            model_name="test-model.pt",
            model_sha256="a" * 64,
        ),
        limitations=["test"],
    )


def test_report_contains_required_structure(tmp_path) -> None:
    path = tmp_path / "report.json"

    write_report(_result(), path)

    data = json.loads(path.read_text(encoding="utf-8"))
    for key in [
        "file_name",
        "file_type",
        "analysis_datetime",
        "application_version",
        "configuration",
        "image_width",
        "image_height",
        "detections",
        "object_counts",
        "scene_readability",
        "priority",
        "communication_mode",
        "message",
        "processing_time_seconds",
        "annotated_image_file",
        "input_sha256",
        "runtime",
        "limitations",
    ]:
        assert key in data
    assert data["image_width"] is None
    assert data["image_height"] is None
    assert data["scene_readability"]["label"] == "heurystyczny wskaźnik cech technicznych obrazu"
    assert data["scene_readability"]["level_label"] == "dobry"
    assert "Nie mierzy czytelności" in data["scene_readability"]["explanation"]
    assert data["priority"]["label"] == "Priorytet komunikatu: brak komunikatu"
    assert "Nie jest oceną ryzyka kolizji" in data["priority"]["explanation"]
    assert data["communication_mode"] == "rozszerzony"
    assert data["configuration"]["detector_backend"] == "test_detector"
    assert data["configuration"]["yolo_model"] == "test-model.pt"
    assert data["configuration"]["yolo_model_sha256"] == "a" * 64
    assert "yolo_model_name" not in data["configuration"]


def test_report_rejects_non_finite_values(tmp_path) -> None:
    result = _result()
    object.__setattr__(result.scene_readability, "score", float("nan"))

    try:
        write_report(result, tmp_path / "invalid.json")
    except ValueError as exc:
        assert "JSON" in str(exc) or "compliant" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-finite JSON value was written")
