from __future__ import annotations

import struct
import zlib
from io import BytesIO

from PIL import Image

from adaptive_driving_assistant.domain import (
    AnalysisConfig,
    BoundingBox,
    CommunicationMode,
    Detection,
    SceneReadabilityLevel,
    SceneReadabilityResult,
)
from adaptive_driving_assistant.pipeline import (
    ImageAnalysisPipeline,
    save_annotated_image,
    timestamped_output_dir,
    validate_image_bytes,
    validate_image_path,
    write_uploaded_image,
)


class FakeDetector:
    def detect(self, _image, confidence_threshold=None):
        return [
            Detection(
                class_name="person",
                confidence=0.95,
                bbox=BoundingBox(20, 20, 80, 120),
                normalized_center=(0.5, 0.55),
                area_ratio=0.2,
            )
        ]


class FakeReadability:
    def estimate(self, _image):
        return SceneReadabilityResult(
            level=SceneReadabilityLevel.MODERATE,
            features={"brightness": 0.5},
            thresholds={"good_threshold": 0.58, "moderate_threshold": 0.38},
            score=0.5,
        )


def _config() -> AnalysisConfig:
    return AnalysisConfig(
        communication_mode=CommunicationMode.EXTENDED,
        object_confidence_threshold=0.35,
    )


class FakeUpload:
    def __init__(self, name: str, payload: bytes | None = None) -> None:
        self.name = name
        if payload is None and name.lower().endswith((".jpg", ".jpeg", ".png")):
            buffer = BytesIO()
            image_format = "PNG" if name.lower().endswith(".png") else "JPEG"
            Image.new("RGB", (16, 16), (120, 130, 140)).save(buffer, format=image_format)
            payload = buffer.getvalue()
        self._payload = payload if payload is not None else b"not-an-image"

    def getbuffer(self):
        return memoryview(self._payload)


def test_image_pipeline_writes_annotated_image_and_report(tmp_path) -> None:
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (160, 160), (180, 180, 180)).save(image_path)
    pipeline = ImageAnalysisPipeline(FakeDetector(), FakeReadability())

    result, report_path = pipeline.analyze(image_path, tmp_path / "out", _config())

    assert result.source_type == "image"
    assert result.object_counts == {"person": 1}
    assert result.annotated_image_path.endswith("_oznaczony.jpg")
    assert report_path is not None
    assert report_path.exists()


def test_image_validation_rejects_remote_uri_and_non_images(tmp_path) -> None:
    text_file = tmp_path / "sample.txt"
    text_file.write_bytes(b"not an image")
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"not an image")

    for path in [text_file, pdf_file]:
        try:
            validate_image_path(path)
        except ValueError as exc:
            assert "obrazy JPG" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("unsupported input was accepted")

    try:
        validate_image_path("rt" + "sp://example.local/stream")
    except ValueError as exc:
        assert "lokalne pliki" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("remote URI was accepted")


def test_uploaded_image_name_with_parent_segments_is_sanitized(tmp_path) -> None:
    saved_path = write_uploaded_image(FakeUpload("../niebezpieczny obraz.jpg"), tmp_path)

    assert saved_path.parent == tmp_path.resolve()
    assert saved_path.suffix == ".jpg"
    assert ".." not in saved_path.name
    assert saved_path.stat().st_size > 0


def test_uploaded_images_with_same_name_get_unique_paths(tmp_path) -> None:
    first = write_uploaded_image(FakeUpload("scena.png"), tmp_path)
    second = write_uploaded_image(FakeUpload("scena.png"), tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_uploaded_image_rejects_disallowed_extension(tmp_path) -> None:
    try:
        write_uploaded_image(FakeUpload("scena.txt"), tmp_path)
    except ValueError as exc:
        assert "JPG, JPEG i PNG" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unsupported upload extension was accepted")


def test_uploaded_image_rejects_fake_jpeg(tmp_path) -> None:
    try:
        write_uploaded_image(FakeUpload("scena.jpg", b"to nie jest obraz"), tmp_path)
    except ValueError as exc:
        assert "poprawnym obrazem" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("fake JPEG was accepted")


def test_image_validation_rejects_tiny_image() -> None:
    buffer = BytesIO()
    Image.new("RGB", (1, 1), (0, 0, 0)).save(buffer, format="PNG")

    try:
        validate_image_bytes(buffer.getvalue(), "tiny.png")
    except ValueError as exc:
        assert "co najmniej" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tiny image was accepted")


def test_image_validation_rejects_decompression_bomb() -> None:
    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", 20_000, 10_000, 8, 2, 0, 0, 0)
    payload = b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header) + png_chunk(b"IEND", b"")

    try:
        validate_image_bytes(payload, "oversized.png")
    except ValueError as exc:
        assert "poprawnym obrazem" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("decompression bomb was accepted")


def test_uploaded_image_allowed_name_is_preserved_safely(tmp_path) -> None:
    saved_path = write_uploaded_image(FakeUpload("Moje zdjęcie.JPG"), tmp_path)

    assert saved_path.parent == tmp_path.resolve()
    assert saved_path.suffix == ".jpg"
    assert saved_path.name.endswith("_moje_zdj_cie.jpg")


def test_timestamped_output_dir_is_unique_for_consecutive_calls(tmp_path) -> None:
    first = timestamped_output_dir(tmp_path)
    second = timestamped_output_dir(tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_analysis_config_has_no_user_editable_model_field() -> None:
    assert "yolo_model_name" not in AnalysisConfig.__dataclass_fields__


def test_save_annotated_image_writes_readable_file(tmp_path) -> None:
    image = Image.new("RGB", (100, 80), (180, 180, 180))
    output_path = tmp_path / "annotated.jpg"
    detection = Detection(
        class_name="car",
        confidence=0.91,
        bbox=BoundingBox(10, 12, 70, 60),
        normalized_center=(0.4, 0.45),
        area_ratio=0.36,
    )

    saved_path = save_annotated_image(
        image=image,
        detections=[detection],
        output_path=output_path,
        title_lines=["wskaźnik cech obrazu: dobry"],
    )

    assert saved_path == output_path
    assert saved_path.exists()
    with Image.open(saved_path) as saved_image:
        assert saved_image.size == image.size
