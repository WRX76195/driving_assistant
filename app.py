"""Interfejs Streamlit dla uproszczonego prototypu obrazowego.

Aplikacja webowa uruchamiana lokalnie umożliwia przesłanie pojedynczego
obrazu, konfigurację trybu komunikacji i progu YOLO oraz podgląd wyniku
analizy wraz z pobraniem raportu JSON i oznaczonego obrazu.
"""

# ruff: noqa: E402

from __future__ import annotations

import logging
import shutil
import sys
from hashlib import sha256
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from adaptive_driving_assistant.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    APP_NAME,
    DATA_DIR,
    DEFAULT_LIMITATIONS,
    DEFAULT_OBJECT_CONFIDENCE,
    DEFAULT_YOLO_MODEL,
    OUTPUT_DIR,
    PRIORITY_EXPLANATION,
    SCENE_READABILITY_EXPLANATION,
    YOLO_MODEL_PATH,
)
from adaptive_driving_assistant.detector import (
    ObjectDetector,
    ObjectDetectorUnavailableError,
    file_sha256,
    validate_local_yolo_model,
)
from adaptive_driving_assistant.domain import (
    AnalysisConfig,
    CommunicationMode,
    PipelineResult,
    communication_mode_label,
    scene_readability_level_label,
)
from adaptive_driving_assistant.pipeline import (
    ImageAnalysisPipeline,
    ensure_data_dirs,
    timestamped_output_dir,
    validate_image_bytes,
    write_uploaded_image,
)

LOGGER = logging.getLogger(__name__)


def main() -> None:
    ensure_data_dirs()
    st.set_page_config(page_title=APP_NAME, layout="wide")
    st.title(APP_NAME)
    st.info(
        "Lokalny demonstrator offline warstwy komunikatów. Nie jest prototypem ADAS. "
        "Aplikacja analizuje tylko pojedynczy obraz przesłany przez użytkownika, "
        "nie steruje pojazdem i nie ocenia ryzyka kolizji."
    )

    config = _sidebar_config()
    if not _analysis_result_matches_config(st.session_state.get("last_result"), config):
        _clear_result_state()
    uploaded_image = st.file_uploader(
        "Prześlij obraz drogowy",
        type=sorted(extension.lstrip(".") for extension in ALLOWED_IMAGE_EXTENSIONS),
    )

    if uploaded_image is None:
        _clear_stale_result()
    else:
        payload = bytes(uploaded_image.getbuffer())
        upload_hash = sha256(payload).hexdigest()
        if st.session_state.get("active_upload_hash") != upload_hash:
            _clear_result_state()
            st.session_state["active_upload_hash"] = upload_hash
        try:
            preview_image = validate_image_bytes(payload, uploaded_image.name)
        except ValueError as exc:
            _clear_result_state()
            st.error(str(exc))
        else:
            st.subheader("Podgląd wejścia")
            st.image(preview_image, width=900)
            if st.button("Analizuj obraz", type="primary"):
                _run_analysis(uploaded_image, config)

    _display_result()
    _limitations_section()


def _sidebar_config() -> AnalysisConfig:
    st.sidebar.header("Konfiguracja")
    mode = st.sidebar.selectbox(
        "Tryb prezentacji",
        options=list(CommunicationMode),
        format_func=communication_mode_label,
    )
    confidence = st.sidebar.slider(
        "Próg wyniku ufności YOLO",
        0.05,
        0.95,
        DEFAULT_OBJECT_CONFIDENCE,
        0.05,
    )
    st.sidebar.caption(f"Model YOLO: {DEFAULT_YOLO_MODEL} (lokalny plik w katalogu projektu)")
    return AnalysisConfig(
        communication_mode=mode,
        object_confidence_threshold=float(confidence),
    )


def _run_analysis(uploaded_image, config: AnalysisConfig) -> None:
    input_path: Path | None = None
    output_dir: Path | None = None
    try:
        model_path = validate_local_yolo_model(YOLO_MODEL_PATH)
        input_path = write_uploaded_image(uploaded_image)
        output_dir = timestamped_output_dir()
        pipeline = ImageAnalysisPipeline(
            detector=_cached_detector(str(model_path), file_sha256(model_path))
        )
    except (ValueError, ObjectDetectorUnavailableError) as exc:
        _cleanup_failed_analysis(input_path, output_dir)
        st.error(str(exc))
        return

    with st.spinner("Analizuję obraz lokalnie..."):
        try:
            result, report_path = pipeline.analyze(input_path, output_dir, config)
        except Exception:  # noqa: BLE001
            _cleanup_failed_analysis(input_path, output_dir)
            LOGGER.exception("Nieoczekiwany błąd podczas analizy obrazu")
            st.error("Nie udało się wykonać analizy. Sprawdź plik i spróbuj ponownie.")
            return

    st.session_state["last_result"] = result
    st.session_state["last_report_path"] = str(report_path) if report_path else None


def _display_result() -> None:
    result: PipelineResult | None = st.session_state.get("last_result")
    if result is None:
        return

    st.subheader("Wynik analizy")
    preview, summary = st.columns([1.2, 1])
    with preview:
        st.image(result.annotated_image_path, caption="Oznaczony obraz", width=900)
    with summary:
        st.write("**Wykryte obiekty**")
        st.json(result.object_counts)
        if result.detections:
            st.write("**Lista detekcji**")
            st.dataframe(
                [
                    {
                        "klasa": detection.class_name,
                        "wynik_ufności": round(detection.confidence, 3),
                        "bbox": [
                            round(detection.bbox.x1, 1),
                            round(detection.bbox.y1, 1),
                            round(detection.bbox.x2, 1),
                            round(detection.bbox.y2, 1),
                        ],
                        "środek": [round(value, 3) for value in detection.normalized_center],
                        "udział_powierzchni": round(detection.area_ratio, 4),
                    }
                    for detection in result.detections
                ],
                width="stretch",
            )
        st.write(
            "**Heurystyczny wskaźnik cech obrazu:** "
            f"{scene_readability_level_label(result.scene_readability.level)} "
            f"({result.scene_readability.score:.2f})"
        )
        st.caption(SCENE_READABILITY_EXPLANATION)
        st.write(f"**{result.priority.label}**")
        st.caption(PRIORITY_EXPLANATION)
        st.write(f"**Komunikat:** {result.message.text}")

    report_path = st.session_state.get("last_report_path")
    report_file = Path(report_path) if report_path else None
    annotated_file = Path(result.annotated_image_path)
    download_report, download_image = st.columns(2)
    with download_report:
        if report_file and report_file.exists():
            st.download_button(
                "Pobierz raport JSON",
                data=report_file.read_bytes(),
                file_name=report_file.name,
                mime="application/json",
            )
    with download_image:
        if annotated_file.exists():
            st.download_button(
                "Pobierz oznaczony obraz",
                data=annotated_file.read_bytes(),
                file_name=annotated_file.name,
                mime="image/jpeg",
            )

    if st.button("Usuń wynik i jego lokalne pliki"):
        _remove_current_artifacts()
        _clear_result_state()
        st.rerun()


@st.cache_resource(show_spinner=False)
def _cached_detector(model_path: str, model_hash: str) -> ObjectDetector:  # noqa: ARG001
    return ObjectDetector(model_path=model_path)


def _clear_stale_result() -> None:
    if st.session_state.get("active_upload_hash") is not None:
        _clear_result_state()
        st.session_state.pop("active_upload_hash", None)


def _analysis_result_matches_config(
    result: PipelineResult | None,
    config: AnalysisConfig,
) -> bool:
    return result is None or result.config == config


def _clear_result_state() -> None:
    st.session_state.pop("last_result", None)
    st.session_state.pop("last_report_path", None)


def _cleanup_failed_analysis(input_path: Path | None, output_dir: Path | None) -> None:
    data_root = DATA_DIR.resolve()
    for raw_path in (output_dir, input_path):
        if raw_path is None:
            continue
        path = Path(raw_path).resolve()
        if path == data_root or not path.is_relative_to(data_root):
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _remove_current_artifacts() -> None:
    result: PipelineResult | None = st.session_state.get("last_result")
    report_path = st.session_state.get("last_report_path")
    candidates = []
    if result is not None:
        candidates.extend([result.source_path, result.annotated_image_path])
    if report_path:
        candidates.append(report_path)
    data_root = DATA_DIR.resolve()
    for raw_path in candidates:
        path = Path(raw_path).resolve()
        if path.is_file() and path.is_relative_to(data_root):
            path.unlink()
            parent = path.parent
            # Usuwaj wyłącznie puste podkatalogi wyników, nie data/input.
            if (
                parent != data_root
                and parent.is_relative_to(OUTPUT_DIR.resolve())
                and parent.is_dir()
                and not any(parent.iterdir())
            ):
                parent.rmdir()


def _limitations_section() -> None:
    with st.expander("Ograniczenia"):
        for limitation in DEFAULT_LIMITATIONS:
            st.write(f"- {limitation}")


if __name__ == "__main__":
    main()
