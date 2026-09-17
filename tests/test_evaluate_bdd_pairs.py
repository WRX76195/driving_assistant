from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_bdd_pairs as evaluation  # noqa: E402


def _box(class_name: str, coordinates: tuple[float, float, float, float]) -> evaluation.Box:
    return evaluation.Box(class_name, *coordinates)


def test_iou_handles_identical_and_disjoint_boxes() -> None:
    first = _box("car", (0, 0, 10, 10))

    assert evaluation.iou(first, _box("car", (0, 0, 10, 10))) == 1.0
    assert evaluation.iou(first, _box("car", (20, 20, 30, 30))) == 0.0


def test_greedy_matching_is_one_to_one() -> None:
    ground_truth = [_box("car", (0, 0, 10, 10))]
    predictions = [
        _box("car", (0, 0, 10, 10)),
        _box("car", (0, 0, 10, 10)),
    ]

    assert evaluation.match_greedy(predictions, ground_truth, 0.5) == (1, 1, 0)


def test_load_gt_boxes_maps_class_attributes_and_area(tmp_path: Path) -> None:
    label = {
        "frames": [
            {
                "objects": [
                    {
                        "category": "bike",
                        "box2d": {"x1": 10, "y1": 20, "x2": 20, "y2": 30},
                        "attributes": {"occluded": True, "truncated": False},
                    },
                    {
                        "category": "pedestrian",
                        "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                        "attributes": {"occluded": False, "truncated": False},
                    },
                ]
            }
        ]
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(label), encoding="utf-8")

    boxes = evaluation.load_gt_boxes(path, (100, 100))

    assert len(boxes) == 2
    assert boxes[0].class_name == "bicycle"
    assert boxes[0].area_ratio == 0.01
    assert boxes[0].occluded is True
    assert boxes[0].truncated is False
    assert boxes[1].class_name == "person"
    assert boxes[1].area_ratio == 0.01


def test_parse_thresholds_supports_ui_range_and_custom_values() -> None:
    ui_values = evaluation.parse_thresholds("ui", 0.35)

    assert len(ui_values) == 19
    assert ui_values[0] == 0.05
    assert ui_values[-1] == 0.95
    assert evaluation.parse_thresholds("0,50;0.15;0.50", 0.35) == (0.15, 0.5)


def test_extract_pairs_uses_only_common_stems_in_sorted_order(tmp_path: Path) -> None:
    archive_path = tmp_path / "pairs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("dataset/z.jpg", b"z")
        archive.writestr("dataset/z.json", "{}")
        archive.writestr("dataset/a.jpg", b"a")
        archive.writestr("dataset/a.json", "{}")
        archive.writestr("dataset/orphan.jpg", b"x")

    pairs = evaluation.extract_pairs(archive_path, tmp_path / "extracted")

    assert [image.stem for image, _label in pairs] == ["a", "z"]
    assert [label.stem for _image, label in pairs] == ["a", "z"]
    from_dir = evaluation.load_pairs_from_dir(tmp_path / "extracted")
    assert [image.stem for image, _label in from_dir] == ["a", "z"]


def test_voc_ap_known_staircase() -> None:
    assert evaluation.voc_ap([0.5, 1.0], [1.0, 0.5]) == 0.75


def test_is_central_box_matches_domain_predicate() -> None:
    central = evaluation.Box("person", 40, 30, 60, 70)
    edge = evaluation.Box("person", 0, 0, 10, 10)

    assert evaluation.is_central_box(central, (100, 100)) is True
    assert evaluation.is_central_box(edge, (100, 100)) is False


def test_resolve_bdd_archive_uses_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.zip"
    explicit.write_bytes(b"x")

    assert evaluation.resolve_bdd_archive(explicit) == explicit


def test_resolve_bdd_archive_picks_first_existing_candidate(tmp_path: Path) -> None:
    missing = tmp_path / "missing.zip"
    present = tmp_path / "present.zip"
    present.write_bytes(b"x")

    assert evaluation.resolve_bdd_archive(None, candidates=(missing, present)) == present


def test_resolve_bdd_archive_returns_none_when_missing(tmp_path: Path) -> None:
    assert evaluation.resolve_bdd_archive(None, candidates=(tmp_path / "no.zip",)) is None


def test_ranking_ap_is_one_for_perfect_match() -> None:
    ground_truth = {"img": [_box("car", (0, 0, 10, 10))]}
    detections = {
        "img": [evaluation.Box("car", 0, 0, 10, 10, confidence=0.9)],
    }

    result = evaluation.average_precision_from_ranking(
        detections, ground_truth, iou_threshold=0.5, class_name="car"
    )

    assert result["ap"] == 1.0
    assert result["n_gt"] == 1


def test_ranking_ap_falls_when_high_confidence_false_positive_leads() -> None:
    ground_truth = {"img": [_box("car", (0, 0, 10, 10))]}
    detections = {
        "img": [
            evaluation.Box("car", 40, 40, 50, 50, confidence=0.99),
            evaluation.Box("car", 0, 0, 10, 10, confidence=0.40),
        ]
    }

    result = evaluation.average_precision_from_ranking(
        detections, ground_truth, iou_threshold=0.5, class_name="car"
    )

    assert result["ap"] < 1.0
    assert result["n_predictions"] == 2


def test_ranking_ap_is_class_aware() -> None:
    ground_truth = {"img": [_box("car", (0, 0, 10, 10))]}
    detections = {
        "img": [evaluation.Box("person", 0, 0, 10, 10, confidence=0.99)],
    }

    car = evaluation.average_precision_from_ranking(
        detections, ground_truth, iou_threshold=0.5, class_name="car"
    )
    micro = evaluation.average_precision_from_ranking(
        detections, ground_truth, iou_threshold=0.5, class_name=None
    )

    assert car["ap"] == 0.0
    assert micro["ap"] == 0.0


def test_official_labels_map_rider_and_bike() -> None:
    item = {
        "name": "a.jpg",
        "labels": [
            {
                "category": "rider",
                "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                "attributes": {"occluded": False, "truncated": True},
            },
            {"category": "bike", "box2d": {"x1": 10, "y1": 10, "x2": 20, "y2": 30}},
        ],
    }

    boxes = evaluation.load_gt_boxes_from_official_item(item, (100, 100))

    assert [box.class_name for box in boxes] == ["person", "bicycle"]
    assert boxes[0].truncated is True
    assert boxes[1].area_ratio == 0.02


def test_load_gt_boxes_accepts_root_labels_array(tmp_path: Path) -> None:
    path = tmp_path / "official-like.json"
    path.write_text(
        json.dumps(
            {
                "labels": [
                    {"category": "motor", "box2d": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}},
                ]
            }
        ),
        encoding="utf-8",
    )

    boxes = evaluation.load_gt_boxes(path, (10, 10))

    assert len(boxes) == 1
    assert boxes[0].class_name == "motorcycle"

