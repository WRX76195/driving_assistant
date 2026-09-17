from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from adaptive_driving_assistant.config import DEFAULT_YOLO_MODEL
from adaptive_driving_assistant.detector import (
    ObjectDetector,
    ObjectDetectorUnavailableError,
    validate_local_yolo_model,
)


class FakeBoxes:
    xyxy = np.array([[10.0, 20.0, 50.0, 80.0], [0.0, 0.0, 20.0, 20.0]])
    conf = np.array([0.9, 0.8])
    cls = np.array([0, 99])


class FakeResult:
    names = {0: "person", 99: "dog"}
    boxes = FakeBoxes()


class FakeYoloModel:
    names = FakeResult.names

    def predict(self, source, conf, verbose):  # noqa: ARG002
        return [FakeResult()]


def test_detector_calculates_bbox_center_and_area_ratio() -> None:
    detector = ObjectDetector(model=FakeYoloModel())
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    detections = detector.detect(image, confidence_threshold=0.35)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.class_name == "person"
    assert detection.bbox.x1 == 10.0
    assert detection.bbox.y1 == 20.0
    assert detection.bbox.x2 == 50.0
    assert detection.bbox.y2 == 80.0
    assert detection.normalized_center == (0.15, 0.5)
    assert detection.area_ratio == 0.12


def test_detector_clamps_bbox_to_image_boundaries() -> None:
    boxes = SimpleNamespace(
        xyxy=np.array([[-10.0, -20.0, 250.0, 120.0]]),
        conf=np.array([0.9]),
        cls=np.array([0]),
    )
    model = Mock()
    model.predict.return_value = [SimpleNamespace(names={0: "person"}, boxes=boxes)]
    detector = ObjectDetector(model=model)

    detections = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].bbox.x1 == 0.0
    assert detections[0].bbox.y1 == 0.0
    assert detections[0].bbox.x2 == 200.0
    assert detections[0].bbox.y2 == 100.0
    assert detections[0].normalized_center == (0.5, 0.5)
    assert detections[0].area_ratio == 1.0


def test_detector_rejects_inconsistent_result_arrays() -> None:
    boxes = SimpleNamespace(
        xyxy=np.array([[1.0, 1.0, 10.0, 10.0], [2.0, 2.0, 20.0, 20.0]]),
        conf=np.array([0.9]),
        cls=np.array([0, 0]),
    )
    model = Mock()
    model.predict.return_value = [SimpleNamespace(names={0: "person"}, boxes=boxes)]
    detector = ObjectDetector(model=model)

    try:
        detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    except ObjectDetectorUnavailableError as exc:
        assert "niespójne tablice" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("inconsistent detector arrays were accepted")


def test_local_yolo_model_validation_accepts_existing_file(tmp_path) -> None:
    model_path = tmp_path / DEFAULT_YOLO_MODEL
    payload = b"local weights"
    model_path.write_bytes(payload)

    assert (
        validate_local_yolo_model(model_path, sha256(payload).hexdigest()) == model_path.resolve()
    )


def test_local_yolo_model_validation_rejects_unexpected_hash(tmp_path) -> None:
    model_path = tmp_path / DEFAULT_YOLO_MODEL
    model_path.write_bytes(b"replaced weights")

    try:
        validate_local_yolo_model(model_path, "0" * 64)
    except ObjectDetectorUnavailableError as exc:
        assert "SHA-256" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("model with an unexpected hash was accepted")


def test_local_yolo_model_validation_rejects_missing_file(tmp_path) -> None:
    try:
        validate_local_yolo_model(tmp_path / DEFAULT_YOLO_MODEL)
    except ObjectDetectorUnavailableError as exc:
        assert DEFAULT_YOLO_MODEL in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing model was accepted")
