"""Modele danych współdzielone przez moduły prototypu.

Definiuje struktury detekcji, wyników heurystyki, priorytetu, komunikatu
oraz konfiguracji analizy. Enumy opisują poziomy wskaźnika cech obrazu,
priorytet komunikatu i tryby prezentacji dostępne w interfejsie użytkownika.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum, StrEnum
from typing import Any


class SceneReadabilityLevel(StrEnum):
    GOOD = "dobra"
    MODERATE = "umiarkowana"
    LIMITED = "ograniczona"


class PriorityLevel(StrEnum):
    NO_MESSAGE = "brak komunikatu"
    INFORMATION = "podstawowy"
    ELEVATED = "podwyższony"
    HIGH = "wysoki"


class CommunicationMode(StrEnum):
    MINIMAL = "minimalny"
    EXTENDED = "rozszerzony"
    EXTENDED_WITH_EXPLANATION = "rozszerzony z objaśnieniem"


PRIORITY_LEVEL_LABELS = {
    PriorityLevel.NO_MESSAGE: "brak komunikatu",
    PriorityLevel.INFORMATION: "podstawowy",
    PriorityLevel.ELEVATED: "podwyższony",
    PriorityLevel.HIGH: "wysoki",
}

COMMUNICATION_MODE_LABELS = {
    CommunicationMode.MINIMAL: "minimalny",
    CommunicationMode.EXTENDED: "rozszerzony",
    CommunicationMode.EXTENDED_WITH_EXPLANATION: "rozszerzony z objaśnieniem",
}

SCENE_READABILITY_LEVEL_LABELS = {
    SceneReadabilityLevel.GOOD: "dobry",
    SceneReadabilityLevel.MODERATE: "umiarkowany",
    SceneReadabilityLevel.LIMITED: "ograniczony",
}


def priority_level_label(level: PriorityLevel) -> str:
    return PRIORITY_LEVEL_LABELS[level]


def priority_display_label(level: PriorityLevel) -> str:
    return f"Priorytet komunikatu: {priority_level_label(level)}"


def communication_mode_label(mode: CommunicationMode) -> str:
    return COMMUNICATION_MODE_LABELS[mode]


def scene_readability_level_label(level: SceneReadabilityLevel) -> str:
    """Zwraca etykietę poziomu dopasowaną do rzeczownika „wskaźnik”."""
    return SCENE_READABILITY_LEVEL_LABELS[level]


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


def is_central_normalized(center_x: float, center_y: float) -> bool:
    """Ten sam predykat centralności, którego używa priorytet i ewaluacja BDD."""
    return 0.35 <= center_x <= 0.65 and 0.25 <= center_y <= 0.85


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: BoundingBox
    normalized_center: tuple[float, float]
    area_ratio: float

    @property
    def is_central(self) -> bool:
        center_x, center_y = self.normalized_center
        return is_central_normalized(center_x, center_y)


@dataclass(frozen=True)
class SceneReadabilityResult:
    level: SceneReadabilityLevel
    features: dict[str, float]
    thresholds: dict[str, float]
    score: float
    method: str = "heuristic_image_features"


@dataclass(frozen=True)
class PriorityResult:
    level: PriorityLevel
    score: float
    reasons: list[str]
    primary_detection_class: str | None = None

    @property
    def label(self) -> str:
        return priority_display_label(self.level)


@dataclass(frozen=True)
class GeneratedMessage:
    text: str
    mode: CommunicationMode
    priority: PriorityLevel
    facts: list[str]


@dataclass(frozen=True)
class AnalysisConfig:
    communication_mode: CommunicationMode
    object_confidence_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 < self.object_confidence_threshold <= 1.0:
            raise ValueError("Próg confidence musi należeć do przedziału (0, 1].")


@dataclass(frozen=True)
class DetectorManifest:
    backend: str
    model_name: str | None
    model_sha256: str | None


@dataclass(frozen=True)
class PipelineResult:
    source_path: str
    source_type: str
    annotated_image_path: str
    detections: list[Detection]
    object_counts: dict[str, int]
    scene_readability: SceneReadabilityResult
    priority: PriorityResult
    message: GeneratedMessage
    processing_time_seconds: float
    config: AnalysisConfig
    detector: DetectorManifest
    limitations: list[str]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(to_jsonable(key)): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value
