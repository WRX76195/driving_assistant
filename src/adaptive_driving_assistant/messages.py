"""Opisowe komunikaty po polsku w trzech trybach prezentacji.

Moduł składa zdania informacyjne na podstawie detekcji, wskaźnika cech obrazu
i priorytetu. Nie ocenia ryzyka kolizji. Tryby określają wyłącznie zakres
tekstu: minimalny, rozszerzony albo rozszerzony z objaśnieniem. Tryb z
objaśnieniem dodaje fakty o liczbie wykryć, wyniku ufności i pozostałych
klasach — nie metakomentarz o „charakterze informacyjnym” komunikatu.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from adaptive_driving_assistant.domain import (
    CommunicationMode,
    Detection,
    GeneratedMessage,
    PriorityLevel,
    PriorityResult,
    SceneReadabilityLevel,
    SceneReadabilityResult,
    scene_readability_level_label,
)
from adaptive_driving_assistant.priority import select_primary_detection

OBJECT_NAMES = {
    "person": ("osoba", "osobę"),
    "car": ("samochód", "samochód"),
    "bus": ("autobus", "autobus"),
    "truck": ("ciężarówka", "ciężarówkę"),
    "bicycle": ("rower", "rower"),
    "motorcycle": ("motocykl", "motocykl"),
}

SUPPORTED_CLASS_LIST_PL = (
    "osoba, rower, motocykl, samochód, autobus, ciężarówka"
)


def generate_message(
    detections: Sequence[Detection],
    scene_readability: SceneReadabilityResult,
    priority: PriorityResult,
    mode: CommunicationMode,
) -> GeneratedMessage:
    if not detections:
        text = _empty_scene_message(scene_readability, priority, mode)
    else:
        text = _object_message(detections, scene_readability, priority, mode)
    return GeneratedMessage(
        text=text,
        mode=mode,
        priority=priority.level,
        facts=_facts(detections, scene_readability),
    )


def _empty_scene_message(
    scene_readability: SceneReadabilityResult,
    priority: PriorityResult,
    mode: CommunicationMode,
) -> str:
    if priority.level != PriorityLevel.NO_MESSAGE:
        return _readability_message(scene_readability, mode)
    base = "Nie wykryto obiektów z obsługiwanych klas."
    if mode == CommunicationMode.MINIMAL:
        return base
    extra = _readability_context(scene_readability)
    if mode == CommunicationMode.EXTENDED_WITH_EXPLANATION:
        extra = _join(extra, f"Sprawdzane klasy: {SUPPORTED_CLASS_LIST_PL}.")
    return _join(base, extra)


def _object_message(
    detections: Sequence[Detection],
    scene_readability: SceneReadabilityResult,
    priority: PriorityResult,
    mode: CommunicationMode,
) -> str:
    main_detection = select_primary_detection(detections)
    _, accusative = OBJECT_NAMES.get(
        main_detection.class_name, (main_detection.class_name, main_detection.class_name)
    )
    base = _priority_ordered_object_sentence(main_detection, accusative, priority)
    readability = _readability_context(scene_readability)

    if mode == CommunicationMode.MINIMAL:
        return base
    if mode == CommunicationMode.EXTENDED_WITH_EXPLANATION:
        return _join(base, readability, _scene_explanation(detections, main_detection))
    return _join(base, readability)


def _scene_explanation(
    detections: Sequence[Detection],
    main_detection: Detection,
) -> str:
    count = len(detections)
    parts = [
        f"Łączna liczba wykryć obsługiwanych klas wynosi {count}.",
        (
            "Wynik ufności głównej detekcji wynosi "
            f"{_format_decimal(main_detection.confidence)}."
        ),
    ]
    others = Counter(
        detection.class_name
        for detection in detections
        if detection.class_name != main_detection.class_name
    )
    if others:
        listed = ", ".join(
            f"{OBJECT_NAMES.get(name, (name, name))[0]} ({value})"
            for name, value in sorted(others.items())
        )
        parts.append(f"W obrazie wykryto także: {listed}.")
    return " ".join(parts)


def _format_decimal(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _priority_ordered_object_sentence(
    detection: Detection,
    accusative: str,
    priority: PriorityResult,
) -> str:
    if priority.level == PriorityLevel.HIGH:
        if detection.is_central:
            return f"Wykryto {accusative} w centralnej części analizowanego obrazu."
        return f"Wykryto {accusative} w analizowanym obrazie."
    if priority.level == PriorityLevel.ELEVATED and detection.is_central:
        return f"W centralnej części analizowanego obrazu wykryto {accusative}."
    return f"W analizowanym obrazie wykryto {accusative}."


def _readability_message(
    scene_readability: SceneReadabilityResult,
    mode: CommunicationMode,
) -> str:
    if mode == CommunicationMode.MINIMAL:
        return (
            "Wskaźnik cech obrazu: poziom "
            f"{scene_readability_level_label(scene_readability.level)}."
        )
    body = (
        "Heurystyczny wskaźnik cech obrazu wskazuje poziom "
        f"{scene_readability_level_label(scene_readability.level)}."
    )
    if mode == CommunicationMode.EXTENDED_WITH_EXPLANATION:
        return _join(body, f"Sprawdzane klasy: {SUPPORTED_CLASS_LIST_PL}.")
    return body


def _readability_context(
    scene_readability: SceneReadabilityResult,
) -> str:
    if scene_readability.level == SceneReadabilityLevel.GOOD:
        return ""
    return (
        "Heurystyczny wskaźnik cech obrazu wskazuje poziom "
        f"{scene_readability_level_label(scene_readability.level)}."
    )


def _facts(
    detections: Sequence[Detection],
    scene_readability: SceneReadabilityResult,
) -> list[str]:
    facts = []
    counts = Counter(detection.class_name for detection in detections)
    facts.extend(f"{name}: {count}" for name, count in sorted(counts.items()))
    facts.append(f"wskaźnik cech obrazu: {scene_readability.level.value}")
    return facts


def _join(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())
