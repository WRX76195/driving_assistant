from __future__ import annotations

from adaptive_driving_assistant.domain import (
    BoundingBox,
    CommunicationMode,
    Detection,
    PriorityLevel,
    PriorityResult,
    SceneReadabilityLevel,
    SceneReadabilityResult,
    communication_mode_label,
    priority_display_label,
    priority_level_label,
)
from adaptive_driving_assistant.messages import generate_message


def _detection() -> Detection:
    return Detection(
        class_name="car",
        confidence=0.92,
        bbox=BoundingBox(10, 10, 80, 100),
        normalized_center=(0.5, 0.55),
        area_ratio=0.12,
    )


def _readability(
    level: SceneReadabilityLevel = SceneReadabilityLevel.MODERATE,
) -> SceneReadabilityResult:
    return SceneReadabilityResult(
        level=level,
        features={"brightness": 0.5},
        thresholds={},
        score=0.5,
    )


def test_message_modes_have_different_detail_levels() -> None:
    priority = PriorityResult(PriorityLevel.INFORMATION, 2.5, [])
    messages = {
        mode: generate_message([_detection()], _readability(), priority, mode).text
        for mode in CommunicationMode
    }

    assert messages[CommunicationMode.MINIMAL] == "W analizowanym obrazie wykryto samochód."
    assert "wskaźnik cech obrazu" in messages[CommunicationMode.EXTENDED]
    assert "Łączna liczba wykryć obsługiwanych klas wynosi 1." in messages[
        CommunicationMode.EXTENDED_WITH_EXPLANATION
    ]
    assert "Wynik ufności głównej detekcji wynosi 0,92." in messages[
        CommunicationMode.EXTENDED_WITH_EXPLANATION
    ]
    assert "nie warunki drogowe" not in messages[CommunicationMode.EXTENDED_WITH_EXPLANATION]
    assert "charakter informacyjny" not in messages[CommunicationMode.EXTENDED_WITH_EXPLANATION]
    assert len(set(messages.values())) == 3


def test_explanation_lists_other_classes() -> None:
    car = _detection()
    bus = Detection(
        class_name="bus",
        confidence=0.8,
        bbox=BoundingBox(1, 1, 40, 40),
        normalized_center=(0.2, 0.2),
        area_ratio=0.2,
    )
    text = generate_message(
        [car, bus],
        _readability(SceneReadabilityLevel.GOOD),
        PriorityResult(PriorityLevel.HIGH, 6.0, []),
        CommunicationMode.EXTENDED_WITH_EXPLANATION,
    ).text
    assert "W obrazie wykryto także: autobus (1)." in text
    assert "Łączna liczba wykryć obsługiwanych klas wynosi 2." in text


def test_message_for_readability_without_detections() -> None:
    message = generate_message(
        [],
        _readability(),
        PriorityResult(PriorityLevel.INFORMATION, 2.0, []),
        CommunicationMode.EXTENDED,
    )

    assert message.text == "Heurystyczny wskaźnik cech obrazu wskazuje poziom umiarkowany."


def test_message_modes_without_detections() -> None:
    limited = SceneReadabilityResult(
        level=SceneReadabilityLevel.LIMITED,
        features={"brightness": 0.1},
        thresholds={},
        score=0.2,
    )
    priority = PriorityResult(PriorityLevel.NO_MESSAGE, 0.0, [])
    minimal = generate_message([], limited, priority, CommunicationMode.MINIMAL).text
    explanatory = generate_message(
        [], limited, priority, CommunicationMode.EXTENDED_WITH_EXPLANATION
    ).text

    assert minimal == "Nie wykryto obiektów z obsługiwanych klas."
    assert "Sprawdzane klasy:" in explanatory
    assert "nie warunki drogowe" not in explanatory


def test_no_message_priority_without_detections() -> None:
    message = generate_message(
        [],
        SceneReadabilityResult(
            level=SceneReadabilityLevel.GOOD,
            features={"brightness": 0.5},
            thresholds={},
            score=0.7,
        ),
        PriorityResult(PriorityLevel.NO_MESSAGE, 0.0, []),
        CommunicationMode.EXTENDED,
    )
    assert message.text == "Nie wykryto obiektów z obsługiwanych klas."


def test_priority_changes_object_message_order() -> None:
    information = PriorityResult(PriorityLevel.INFORMATION, 2.0, [])
    high = PriorityResult(PriorityLevel.HIGH, 6.0, [])

    information_text = generate_message(
        [_detection()], _readability(), information, CommunicationMode.MINIMAL
    ).text
    high_text = generate_message(
        [_detection()], _readability(), high, CommunicationMode.MINIMAL
    ).text

    assert information_text == "W analizowanym obrazie wykryto samochód."
    assert high_text == "Wykryto samochód w centralnej części analizowanego obrazu."

    off_center = Detection(
        class_name="car",
        confidence=0.92,
        bbox=BoundingBox(10, 10, 80, 100),
        normalized_center=(0.1, 0.55),
        area_ratio=0.12,
    )
    high_off_center = generate_message(
        [off_center], _readability(), high, CommunicationMode.MINIMAL
    ).text
    assert high_off_center == "Wykryto samochód w analizowanym obrazie."


def test_person_class_is_not_overinterpreted_as_pedestrian() -> None:
    person = Detection(
        class_name="person",
        confidence=0.9,
        bbox=BoundingBox(10, 10, 80, 100),
        normalized_center=(0.5, 0.55),
        area_ratio=0.12,
    )
    message = generate_message(
        [person],
        _readability(),
        PriorityResult(PriorityLevel.HIGH, 6.0, []),
        CommunicationMode.MINIMAL,
    )

    assert "osobę" in message.text
    assert "piesz" not in message.text


def test_messages_do_not_use_forbidden_directive_phrases() -> None:
    forbidden = [
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
    ]
    priority = PriorityResult(PriorityLevel.HIGH, 6.0, [])

    for mode in CommunicationMode:
        message = generate_message([_detection()], _readability(), priority, mode)
        text = message.text.lower()
        assert all(phrase not in text for phrase in forbidden)


def test_user_facing_labels_use_polish_diacritics() -> None:
    assert (
        communication_mode_label(CommunicationMode.EXTENDED_WITH_EXPLANATION)
        == "rozszerzony z objaśnieniem"
    )
    assert priority_level_label(PriorityLevel.ELEVATED) == "podwyższony"
    assert {mode.name for mode in CommunicationMode} == {
        "MINIMAL",
        "EXTENDED",
        "EXTENDED_WITH_EXPLANATION",
    }
    assert priority_display_label(PriorityLevel.HIGH) == "Priorytet komunikatu: wysoki"
