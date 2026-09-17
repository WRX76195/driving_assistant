from types import SimpleNamespace

import app
from app import _analysis_result_matches_config, _cleanup_failed_analysis

from adaptive_driving_assistant.domain import AnalysisConfig, CommunicationMode


def _config(mode: CommunicationMode, confidence: float) -> AnalysisConfig:
    return AnalysisConfig(
        communication_mode=mode,
        object_confidence_threshold=confidence,
    )


def test_result_remains_current_for_identical_configuration():
    config = _config(CommunicationMode.EXTENDED, 0.35)
    result = SimpleNamespace(config=config)

    assert _analysis_result_matches_config(result, config)


def test_result_becomes_stale_after_mode_change():
    result = SimpleNamespace(config=_config(CommunicationMode.EXTENDED, 0.35))
    changed_config = _config(CommunicationMode.MINIMAL, 0.35)

    assert not _analysis_result_matches_config(result, changed_config)


def test_result_becomes_stale_after_confidence_change():
    result = SimpleNamespace(config=_config(CommunicationMode.EXTENDED, 0.35))
    changed_config = _config(CommunicationMode.EXTENDED, 0.95)

    assert not _analysis_result_matches_config(result, changed_config)


def test_failed_analysis_cleanup_removes_only_application_artifacts(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    input_path = data_root / "input" / "scene.jpg"
    output_dir = data_root / "output" / "run"
    outside_file = tmp_path / "outside.txt"
    input_path.parent.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    input_path.write_bytes(b"image")
    (output_dir / "partial.json").write_text("{}", encoding="utf-8")
    outside_file.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(app, "DATA_DIR", data_root)

    _cleanup_failed_analysis(input_path, output_dir)
    _cleanup_failed_analysis(outside_file, None)

    assert not input_path.exists()
    assert not output_dir.exists()
    assert outside_file.exists()
