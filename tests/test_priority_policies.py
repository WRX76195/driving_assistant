from __future__ import annotations

from adaptive_driving_assistant.domain import BoundingBox, Detection
from adaptive_driving_assistant.priority import select_primary_detection
from adaptive_driving_assistant.priority_policies import (
    POLICY_AREA_FIRST,
    POLICY_CONFIDENCE_FIRST,
    POLICY_VRU_FIRST,
    select_primary_detection_for_policy,
)


def _detection(
    class_name: str,
    *,
    area_ratio: float,
    confidence: float,
    center: tuple[float, float] = (0.5, 0.55),
) -> Detection:
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(10, 10, 80, 120),
        normalized_center=center,
        area_ratio=area_ratio,
    )


def test_vru_first_matches_production_select_primary() -> None:
    detections = [
        _detection("person", area_ratio=0.01, confidence=0.4, center=(0.1, 0.1)),
        _detection("bus", area_ratio=0.20, confidence=0.9, center=(0.5, 0.5)),
    ]

    production = select_primary_detection(detections)
    policy = select_primary_detection_for_policy(detections, POLICY_VRU_FIRST)

    assert policy.class_name == production.class_name == "bus"


def test_area_first_picks_largest_box() -> None:
    detections = [
        _detection("person", area_ratio=0.02, confidence=0.99),
        _detection("car", area_ratio=0.25, confidence=0.40),
    ]

    selected = select_primary_detection_for_policy(detections, POLICY_AREA_FIRST)

    assert selected.class_name == "car"


def test_confidence_first_picks_highest_score() -> None:
    detections = [
        _detection("person", area_ratio=0.20, confidence=0.41),
        _detection("car", area_ratio=0.02, confidence=0.93),
    ]

    selected = select_primary_detection_for_policy(detections, POLICY_CONFIDENCE_FIRST)

    assert selected.class_name == "car"


def test_three_policies_can_disagree_on_the_same_detections() -> None:
    detections = [
        _detection("person", area_ratio=0.03, confidence=0.55, center=(0.1, 0.1)),
        _detection("bus", area_ratio=0.22, confidence=0.40, center=(0.5, 0.5)),
        _detection("car", area_ratio=0.08, confidence=0.97, center=(0.1, 0.1)),
    ]

    chosen = {
        policy: select_primary_detection_for_policy(detections, policy).class_name
        for policy in (POLICY_VRU_FIRST, POLICY_AREA_FIRST, POLICY_CONFIDENCE_FIRST)
    }

    assert chosen[POLICY_VRU_FIRST] == "bus"
    assert chosen[POLICY_AREA_FIRST] == "bus"
    assert chosen[POLICY_CONFIDENCE_FIRST] == "car"
    assert len(set(chosen.values())) == 2
