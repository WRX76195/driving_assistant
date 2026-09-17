"""Trzy jawne polityki wyboru obiektu głównego (Droga A).

Produkcyjna reguła 0.3.5/0.3.6 to vru_first: wagi klas, centralność i udział
powierzchni. Dwie polityki odniesienia — area_first i confidence_first —
stosuje się do tych samych detekcji, bez ponownej inferencji YOLO.
Nie kalibrują one wag; mierzą, jak często hierarchia VRU zmienia klasę
komunikatu względem powierzchni albo wyniku ufności.
"""

from __future__ import annotations

from collections.abc import Sequence

from adaptive_driving_assistant.domain import Detection
from adaptive_driving_assistant.priority import select_primary_detection

POLICY_VRU_FIRST = "vru_first"
POLICY_AREA_FIRST = "area_first"
POLICY_CONFIDENCE_FIRST = "confidence_first"
POLICY_NAMES = (POLICY_VRU_FIRST, POLICY_AREA_FIRST, POLICY_CONFIDENCE_FIRST)
VRU_CLASSES = frozenset({"person", "bicycle", "motorcycle"})


def select_primary_detection_for_policy(
    detections: Sequence[Detection],
    policy: str,
) -> Detection:
    if not detections:
        raise ValueError("Nie można wybrać głównej detekcji z pustej sekwencji.")
    if policy not in POLICY_NAMES:
        raise ValueError(f"Nieznana polityka: {policy}")
    if policy == POLICY_VRU_FIRST:
        return select_primary_detection(detections)
    return max(detections, key=lambda detection: _policy_key(detection, policy))


def _policy_key(detection: Detection, policy: str) -> tuple:
    if policy == POLICY_AREA_FIRST:
        return (detection.area_ratio, detection.confidence, detection.class_name)
    if policy == POLICY_CONFIDENCE_FIRST:
        return (detection.confidence, detection.area_ratio, detection.class_name)
    raise ValueError(f"Nieznana polityka: {policy}")
