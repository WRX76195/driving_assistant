"""Wyczerpujące wyliczenie języka komunikatów — specyfikacja PV1."""

from __future__ import annotations

import re
from collections.abc import Iterator

from adaptive_driving_assistant.domain import (
    BoundingBox,
    CommunicationMode,
    Detection,
    PriorityLevel,
    PriorityResult,
    SceneReadabilityLevel,
    SceneReadabilityResult,
)
from adaptive_driving_assistant.messages import OBJECT_NAMES, generate_message
from adaptive_driving_assistant.priority import assess_priority

FORBIDDEN = [
    "hamuj",
    "skręć",
    "skrec",
    "jedź",
    "jedz",
    "zachowaj uwagę",
    "zachowaj uwage",
    "obserwuj sytuację",
    "obserwuj sytuacje",
    "przed tobą",
    "przed toba",
    "uważaj",
    "uwazaj",
    "charakter informacyjny",
    "nie warunki drogowe",
    "wynik ma charakter techniczny",
]

IMPERATIVE_RE = re.compile(
    r"\b(hamuj|skręć|jedź|uważaj|zachowaj|obserwuj)\b",
    re.IGNORECASE,
)


def _box() -> BoundingBox:
    return BoundingBox(10, 10, 80, 100)


def _readability(level: SceneReadabilityLevel) -> SceneReadabilityResult:
    return SceneReadabilityResult(level=level, features={}, thresholds={}, score=0.4)


def _detection(
    class_name: str,
    *,
    central: bool,
    area_ratio: float,
    extras: int = 0,
) -> list[Detection]:
    center = (0.5, 0.55) if central else (0.1, 0.2)
    main = Detection(
        class_name=class_name,
        confidence=0.9,
        bbox=_box(),
        normalized_center=center,
        area_ratio=area_ratio,
    )
    detections = [main]
    for _ in range(extras):
        detections.append(
            Detection(
                class_name="car",
                confidence=0.5,
                bbox=_box(),
                normalized_center=(0.1, 0.1),
                area_ratio=0.02,
            )
        )
    return detections


def iter_formal_texts() -> Iterator[str]:
    for class_name in OBJECT_NAMES:
        for central in (True, False):
            for level in SceneReadabilityLevel:
                for priority in PriorityLevel:
                    for mode in CommunicationMode:
                        detections = _detection(class_name, central=central, area_ratio=0.12)
                        result = PriorityResult(priority, 0.0, [])
                        yield generate_message(
                            detections, _readability(level), result, mode
                        ).text
    for level in SceneReadabilityLevel:
        for priority in PriorityLevel:
            for mode in CommunicationMode:
                yield generate_message(
                    [], _readability(level), PriorityResult(priority, 0.0, []), mode
                ).text


def iter_reachable_texts() -> Iterator[str]:
    for class_name in OBJECT_NAMES:
        for central in (True, False):
            for area_ratio in (0.02, 0.08, 0.15):
                for extras in (0, 3):
                    for level in SceneReadabilityLevel:
                        for mode in CommunicationMode:
                            detections = _detection(
                                class_name,
                                central=central,
                                area_ratio=area_ratio,
                                extras=extras,
                            )
                            priority = assess_priority(detections, _readability(level))
                            yield generate_message(
                                detections, _readability(level), priority, mode
                            ).text
    for level in SceneReadabilityLevel:
        for mode in CommunicationMode:
            priority = assess_priority([], _readability(level))
            yield generate_message([], _readability(level), priority, mode).text


def test_no_forbidden_or_meta_phrases_in_formal_language() -> None:
    texts = set(iter_formal_texts())
    joined = "\n".join(texts).lower()
    for phrase in FORBIDDEN:
        assert phrase not in joined
    assert IMPERATIVE_RE.search(joined) is None


def test_reachable_language_has_no_directive_and_no_meta_disclaimer() -> None:
    formal = list(iter_formal_texts())
    reachable = list(iter_reachable_texts())
    assert len(formal) == 468
    assert len(set(formal)) == 159
    assert len(reachable) == 657
    assert len(set(reachable)) == 135
    texts = set(reachable)
    joined = "\n".join(texts).lower()
    for phrase in FORBIDDEN:
        assert phrase not in joined
    assert any("Nie wykryto obiektów z obsługiwanych klas." in text for text in texts)
    assert any("Łączna liczba wykryć" in text for text in texts)
