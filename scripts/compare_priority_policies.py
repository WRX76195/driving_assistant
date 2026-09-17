"""Porównanie trzech polityk wyboru obiektu głównego na tych samych detekcjach.

Droga A z planu uzupełnień: vru_first (produkcja), area_first, confidence_first.
Skrypt nie uruchamia YOLO, jeśli podano zrzut z evaluate_bdd_pairs.py
(--save-detections). Liczy wyłącznie, jak często polityki wybierają inną klasę
główną — bez twierdzeń o lęku, spokoju ani bezpieczeństwie jazdy.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_driving_assistant.config import APP_VERSION  # noqa: E402
from adaptive_driving_assistant.domain import BoundingBox, Detection  # noqa: E402
from adaptive_driving_assistant.priority_policies import (  # noqa: E402
    POLICY_NAMES,
    VRU_CLASSES,
    select_primary_detection_for_policy,
)


def detection_from_record(record: dict) -> Detection | None:
    bbox = record.get("bbox")
    class_name = record.get("class_name")
    if not isinstance(bbox, list) or len(bbox) != 4 or not isinstance(class_name, str):
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        confidence = float(record.get("confidence", 1.0))
        area_ratio = float(record.get("area_ratio") or 0.0)
    except (TypeError, ValueError):
        return None
    center = record.get("normalized_center") or [0.5, 0.5]
    try:
        center_x, center_y = float(center[0]), float(center[1])
    except (TypeError, ValueError, IndexError):
        center_x, center_y = 0.5, 0.5
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1, y1, x2, y2),
        normalized_center=(center_x, center_y),
        area_ratio=area_ratio,
    )


def load_image_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload, dict) else payload
    if not isinstance(images, list):
        raise ValueError("Zrzut detekcji musi zawierać listę images.")
    return [item for item in images if isinstance(item, dict)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detections-json",
        type=Path,
        required=True,
        help="Zrzut z evaluate_bdd_pairs.py --save-detections albo analogiczny JSON.",
    )
    parser.add_argument(
        "--source",
        choices=("detections", "ground_truth"),
        default="detections",
        help="Źródło ramek do polityk: predykcje modelu albo etykiety GT.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "output" / "priority_policies.json",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if not args.detections_json.is_file():
        parser.error(f"Nie znaleziono zrzutu: {args.detections_json}")
    images = load_image_records(args.detections_json)
    n_empty = 0
    n_with_objects = 0
    n_with_vru = 0
    primary_counts = {policy: Counter() for policy in POLICY_NAMES}
    pairwise_disagree = Counter()
    vru_other_class = {policy: 0 for policy in POLICY_NAMES}
    triple_disagree = 0

    for item in images:
        records = item.get(args.source) or []
        detections = [
            detection
            for record in records
            if isinstance(record, dict)
            for detection in [detection_from_record(record)]
            if detection is not None
        ]
        if not detections:
            n_empty += 1
            continue
        n_with_objects += 1
        has_vru = any(detection.class_name in VRU_CLASSES for detection in detections)
        if has_vru:
            n_with_vru += 1
        chosen = {
            policy: select_primary_detection_for_policy(detections, policy)
            for policy in POLICY_NAMES
        }
        classes = {policy: chosen[policy].class_name for policy in POLICY_NAMES}
        for policy, class_name in classes.items():
            primary_counts[policy][class_name] += 1
            if has_vru and class_name not in VRU_CLASSES:
                vru_other_class[policy] += 1
        if len(set(classes.values())) == 3:
            triple_disagree += 1
        for first, second in (
            ("vru_first", "area_first"),
            ("vru_first", "confidence_first"),
            ("area_first", "confidence_first"),
        ):
            if classes[first] != classes[second]:
                pairwise_disagree[f"{first}_vs_{second}"] += 1

    def _rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    report = {
        "app_version": APP_VERSION,
        "source_file": str(args.detections_json),
        "box_source": args.source,
        "images_total": len(images),
        "images_empty": n_empty,
        "images_with_objects": n_with_objects,
        "images_with_vru": n_with_vru,
        "primary_class_counts": {
            policy: dict(sorted(counter.items())) for policy, counter in primary_counts.items()
        },
        "pairwise_primary_class_disagreement": {
            key: {
                "count": count,
                "share_of_images_with_objects": _rate(count, n_with_objects),
            }
            for key, count in pairwise_disagree.items()
        },
        "all_three_policies_disagree": {
            "count": triple_disagree,
            "share_of_images_with_objects": _rate(triple_disagree, n_with_objects),
        },
        "vru_present_but_primary_is_other_class": {
            policy: {
                "count": vru_other_class[policy],
                "share_of_images_with_vru": _rate(vru_other_class[policy], n_with_vru),
            }
            for policy in POLICY_NAMES
        },
        "note": (
            "To nie jest kalibracja wag ani ocena odbioru u ludzi. "
            "Metryka to wyłącznie zgodność klasy obiektu głównego między politykami."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano: {args.out}")
    print(json.dumps(report["pairwise_primary_class_disagreement"], ensure_ascii=False, indent=2))
    print(
        json.dumps(
            report["vru_present_but_primary_is_other_class"], ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
