from __future__ import annotations

from adaptive_driving_assistant.domain import (
    BoundingBox,
    CommunicationMode,
    Detection,
    PriorityLevel,
    SceneReadabilityLevel,
    SceneReadabilityResult,
)
from adaptive_driving_assistant.messages import generate_message
from adaptive_driving_assistant.priority import assess_priority, select_primary_detection


def _detection(
    class_name: str = "person",
    area_ratio: float = 0.15,
    center: tuple[float, float] = (0.5, 0.55),
) -> Detection:
    return Detection(
        class_name=class_name,
        confidence=0.9,
        bbox=BoundingBox(10, 10, 80, 120),
        normalized_center=center,
        area_ratio=area_ratio,
    )


def _readability(level: SceneReadabilityLevel) -> SceneReadabilityResult:
    return SceneReadabilityResult(
        level=level,
        features={"brightness": 0.5},
        thresholds={"good_threshold": 0.58, "moderate_threshold": 0.38},
        score=0.5,
    )


def test_priority_high_for_central_large_person_without_indicator_bonus() -> None:
    limited = assess_priority([_detection()], _readability(SceneReadabilityLevel.LIMITED))
    good = assess_priority([_detection()], _readability(SceneReadabilityLevel.GOOD))

    assert limited.level == PriorityLevel.HIGH
    assert limited.score == good.score
    assert limited.score == 6.0
    assert "wskaźnik cech obrazu" not in " ".join(limited.reasons)


def test_priority_no_message_for_empty_scene_regardless_of_indicator() -> None:
    for level in SceneReadabilityLevel:
        result = assess_priority([], _readability(level))
        assert result.level == PriorityLevel.NO_MESSAGE
        assert result.score == 0.0


def test_primary_detection_uses_same_scoring_rule_as_priority() -> None:
    central_bus = _detection("bus", 0.2, (0.5, 0.5))
    small_corner_person = _detection("person", 0.01, (0.1, 0.1))

    selected = select_primary_detection([small_corner_person, central_bus])
    priority = assess_priority(
        [small_corner_person, central_bus], _readability(SceneReadabilityLevel.GOOD)
    )

    assert selected.class_name == "bus"
    assert priority.primary_detection_class == "bus"


def test_priority_and_message_use_the_same_primary_detection() -> None:
    central_bus = _detection("bus", 0.2, (0.5, 0.5))
    small_corner_person = _detection("person", 0.01, (0.1, 0.1))
    detections = [small_corner_person, central_bus]
    readability = _readability(SceneReadabilityLevel.GOOD)

    priority = assess_priority(detections, readability)
    message = generate_message(detections, readability, priority, CommunicationMode.MINIMAL)

    assert priority.primary_detection_class == "bus"
    assert "autobus" in message.text
    assert "osobę" not in message.text
