"""Jawne reguły priorytetyzacji komunikatu.

Moduł ocenia, jak wykryte obiekty wpływają na kategorię priorytetu
(podstawowy, podwyższony, wysoki). Heurystyczny wskaźnik cech obrazu jest
liczony i zapisywany osobno. Reguły są deterministyczne i współdzielone przez komunikat oraz raport JSON.
"""

from __future__ import annotations

from collections.abc import Sequence

from adaptive_driving_assistant.domain import (
    Detection,
    PriorityLevel,
    PriorityResult,
    SceneReadabilityResult,
)

CLASS_WEIGHTS = {
    "person": 3.0,
    "bicycle": 2.2,
    "motorcycle": 2.2,
    "car": 1.3,
    "bus": 1.4,
    "truck": 1.5,
}


def assess_priority(
    detections: Sequence[Detection],
    scene_readability: SceneReadabilityResult,
) -> PriorityResult:
    del scene_readability
    score = 0.0
    reasons: list[str] = []
    strongest: Detection | None = None

    if detections:
        strongest = select_primary_detection(detections)
        score += _detection_score(strongest)
        reasons.append(f"wykryto obiekt: {strongest.class_name}")
        if strongest.is_central:
            reasons.append("obiekt blisko środka obrazu")
        if strongest.area_ratio >= 0.12:
            reasons.append("duży udział obiektu w obrazie")
        elif strongest.area_ratio >= 0.04:
            reasons.append("zauważalny udział obiektu w obrazie")
        if len(detections) >= 3:
            score += 0.8
            reasons.append("wiele wykrytych obiektów")

    level = _level_from_score(score)
    return PriorityResult(
        level=level,
        score=round(max(0.0, score), 3),
        reasons=reasons,
        primary_detection_class=strongest.class_name if strongest else None,
    )


def select_primary_detection(detections: Sequence[Detection]) -> Detection:
    if not detections:
        raise ValueError("Nie można wybrać głównej detekcji z pustej sekwencji.")
    return max(
        detections,
        key=lambda detection: (
            _detection_score(detection),
            detection.confidence,
            detection.area_ratio,
            detection.class_name,
        ),
    )


def _detection_score(detection: Detection) -> float:
    score = CLASS_WEIGHTS.get(detection.class_name, 0.5)
    if detection.is_central:
        score += 1.5
    if detection.area_ratio >= 0.12:
        score += 1.5
    elif detection.area_ratio >= 0.04:
        score += 0.8
    return score


def _level_from_score(score: float) -> PriorityLevel:
    if score >= 6.0:
        return PriorityLevel.HIGH
    if score >= 4.0:
        return PriorityLevel.ELEVATED
    if score >= 1.0:
        return PriorityLevel.INFORMATION
    return PriorityLevel.NO_MESSAGE
