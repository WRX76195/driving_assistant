"""Ewaluacja detekcji na stałym podzbiorze par BDD100K.

Skrypt uruchamia model jeden raz z najniższym analizowanym progiem wyniku
ufności, a następnie filtruje te same predykcje dla kolejnych progów. Dzięki
temu można odtworzyć krzywą precision--recall bez wielokrotnej inferencji.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_driving_assistant.config import (  # noqa: E402
    APP_VERSION,
    DEFAULT_OBJECT_CONFIDENCE,
    EXPECTED_YOLO_SHA256,
    SUPPORTED_OBJECT_CLASSES,
)
from adaptive_driving_assistant.detector import (  # noqa: E402
    ObjectDetector,
    file_sha256,
    validate_local_yolo_model,
)
from adaptive_driving_assistant.domain import is_central_normalized  # noqa: E402

BDD_TO_APP = {
    "person": "person",
    "pedestrian": "person",
    "rider": "person",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
}
DEFAULT_SWEEP = tuple(round(value / 100, 2) for value in range(5, 100, 5))
SIZE_BINS = ("mały (<1%)", "średni (1–5%)", "duży (≥5%)")
BDD_ARCHIVE_NAME = "bdd100k_2000_pairs.zip"


@dataclass(frozen=True)
class Box:
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    area_ratio: float | None = None
    occluded: bool | None = None
    truncated: bool | None = None


def iou(first: Box, second: Box) -> float:
    """Zwraca intersection over union dwóch ramek."""
    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def voc_ap(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """Interpolowane AP w schemacie VOC na dyskretnych punktach krzywej."""
    pairs = sorted(zip(recalls, precisions, strict=True), key=lambda item: item[0])
    if not pairs:
        return 0.0
    recall = [0.0] + [item[0] for item in pairs] + [1.0]
    precision = [pairs[0][1]] + [item[1] for item in pairs] + [0.0]
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    area = 0.0
    for index in range(1, len(recall)):
        area += (recall[index] - recall[index - 1]) * precision[index]
    return area


def is_central_box(box: Box, image_size: tuple[int, int]) -> bool:
    """Ten sam predykat co Detection.is_central w domain.py."""
    width, height = image_size
    if width <= 0 or height <= 0:
        return False
    center_x = ((box.x1 + box.x2) / 2.0) / width
    center_y = ((box.y1 + box.y2) / 2.0) / height
    return is_central_normalized(center_x, center_y)


def average_precision_from_threshold_metrics(
    metrics_by_threshold: dict[str, dict],
    class_name: str | None = None,
) -> float:
    ordered_keys = sorted(metrics_by_threshold, key=float)
    recalls: list[float] = []
    precisions: list[float] = []
    for key in ordered_keys:
        block = (
            metrics_by_threshold[key]["micro"]
            if class_name is None
            else metrics_by_threshold[key]["per_class"][class_name]
        )
        recalls.append(float(block["recall"]))
        precisions.append(float(block["precision"]))
    return round(voc_ap(recalls, precisions), 6)


def average_precision_from_ranking(
    detections_by_image: dict[str, list[Box]],
    ground_truth_by_image: dict[str, list[Box]],
    *,
    iou_threshold: float,
    class_name: str | None = None,
) -> dict[str, float | int]:
    """Interpolowane AP VOC 2010+ z pełnego rankingu ufności.

    Dopasowanie jest 1:1, klasowo, zachłanne po IoU w kolejności malejącej
    ufności. To nie jest siatka 19 progów z average_precision_from_threshold_metrics.
    """
    ranked: list[tuple[float, str, Box]] = []
    n_gt = 0
    gt_filtered: dict[str, list[Box]] = {}
    image_ids = set(detections_by_image) | set(ground_truth_by_image)
    for image_id in image_ids:
        kept = [
            box
            for box in ground_truth_by_image.get(image_id, [])
            if class_name is None or box.class_name == class_name
        ]
        gt_filtered[image_id] = kept
        n_gt += len(kept)
        for box in detections_by_image.get(image_id, []):
            if class_name is not None and box.class_name != class_name:
                continue
            ranked.append((box.confidence, image_id, box))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2].class_name))
    if n_gt == 0:
        return {"ap": 0.0 if ranked else 1.0, "n_gt": 0, "n_predictions": len(ranked)}
    matched: dict[str, set[int]] = {image_id: set() for image_id in gt_filtered}
    tp_cum = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for index, (_confidence, image_id, detection) in enumerate(ranked, start=1):
        gts = gt_filtered.get(image_id, [])
        used = matched.setdefault(image_id, set())
        best_idx = -1
        best_iou = iou_threshold
        for gt_index, gt_box in enumerate(gts):
            if gt_index in used:
                continue
            if class_name is None and detection.class_name != gt_box.class_name:
                continue
            overlap = iou(detection, gt_box)
            if overlap >= best_iou:
                best_iou = overlap
                best_idx = gt_index
        if best_idx >= 0:
            used.add(best_idx)
            tp_cum += 1
        recalls.append(tp_cum / n_gt)
        precisions.append(tp_cum / index)
    if not recalls:
        return {"ap": 0.0, "n_gt": n_gt, "n_predictions": 0}
    return {
        "ap": round(voc_ap(recalls, precisions), 6),
        "n_gt": n_gt,
        "n_predictions": len(ranked),
    }


def ranking_average_precision_report(
    detections_by_image: dict[str, list[Box]],
    ground_truth_by_image: dict[str, list[Box]],
    iou_threshold: float,
) -> dict[str, object]:
    per_class = {
        name: average_precision_from_ranking(
            detections_by_image,
            ground_truth_by_image,
            iou_threshold=iou_threshold,
            class_name=name,
        )
        for name in sorted(SUPPORTED_OBJECT_CLASSES)
    }
    scored = [float(block["ap"]) for block in per_class.values() if int(block["n_gt"]) > 0]
    micro = average_precision_from_ranking(
        detections_by_image,
        ground_truth_by_image,
        iou_threshold=iou_threshold,
        class_name=None,
    )
    return {
        "method": (
            "interpolacja VOC 2010+ na pełnym rankingu ufności; dopasowanie 1:1 "
            "zachłanne według IoU, osobno dla klasy (mikro: klasowo w puli)"
        ),
        "iou": iou_threshold,
        "micro": micro,
        "macro_mean": round(sum(scored) / len(scored), 6) if scored else 0.0,
        "per_class": per_class,
        "note": (
            "Przy maksymalnej kompletności poniżej 1 część całki VOC do recall=1 jest zerowa. "
            "Wynik nie jest porównywalny z AP z siatki 19 progów."
        ),
    }


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    return None


def boxes_from_bdd_objects(
    objects: Sequence[object],
    image_size: tuple[int, int] | None = None,
) -> list[Box]:
    """Mapuje obiekty BDD (frames[].objects albo labels[]) na obsługiwane ramki."""
    width, height = image_size or (0, 0)
    image_area = float(width * height)
    boxes: list[Box] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        class_name = BDD_TO_APP.get(str(item.get("category", "")).lower())
        raw_box = item.get("box2d")
        if class_name is None or not isinstance(raw_box, dict):
            continue
        try:
            x1 = float(raw_box["x1"])
            y1 = float(raw_box["y1"])
            x2 = float(raw_box["x2"])
            y2 = float(raw_box["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        attributes = item.get("attributes") or {}
        area_ratio = ((x2 - x1) * (y2 - y1) / image_area) if image_area > 0 else None
        boxes.append(
            Box(
                class_name=class_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                area_ratio=area_ratio,
                occluded=_optional_bool(attributes.get("occluded")),
                truncated=_optional_bool(attributes.get("truncated")),
            )
        )
    return boxes


def load_gt_boxes(label_path: Path, image_size: tuple[int, int] | None = None) -> list[Box]:
    """Wczytuje obsługiwane ramki ground truth z etykiety BDD100K (format par JPG–JSON)."""
    data = json.loads(label_path.read_text(encoding="utf-8"))
    frames = data.get("frames") or []
    objects = frames[0].get("objects", []) if frames else data.get("labels") or []
    return boxes_from_bdd_objects(objects, image_size)


def detections_to_boxes(detections: Sequence[object]) -> list[Box]:
    """Mapuje Detection z adaptera na ramki ewaluacji."""
    return [
        Box(
            class_name=item.class_name,
            x1=item.bbox.x1,
            y1=item.bbox.y1,
            x2=item.bbox.x2,
            y2=item.bbox.y2,
            confidence=item.confidence,
            area_ratio=item.area_ratio,
        )
        for item in detections
    ]


def infer_pair(
    image_path: Path,
    label_path: Path,
    detector: ObjectDetector,
    *,
    confidence_threshold: float,
) -> tuple[str, list[Box], list[Box], tuple[int, int]] | None:
    """Zwraca (id, predykcje, GT, rozmiar) albo None przy nieczytelnym pliku."""
    try:
        with Image.open(image_path) as pil_image:
            image_rgb = pil_image.convert("RGB")
            width, height = image_rgb.size
            image = np.asarray(image_rgb)
        ground_truth = load_gt_boxes(label_path, (width, height))
        detections = detector.detect(image, confidence_threshold=confidence_threshold)
    except (OSError, UnidentifiedImageError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return image_path.stem, detections_to_boxes(detections), ground_truth, (width, height)


def load_official_bdd_frames(labels_path: Path) -> list[dict]:
    """Wczytuje oficjalny plik etykiet detekcji (tablica ramek det_val / labels_images_val)."""
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        frames = data.get("frames") or data.get("images") or data.get("labels")
        if isinstance(frames, list):
            return [item for item in frames if isinstance(item, dict)]
    raise ValueError(f"Nieobsługiwany format etykiet BDD: {labels_path}")


def load_gt_boxes_from_official_item(
    item: dict,
    image_size: tuple[int, int] | None = None,
) -> list[Box]:
    objects = item.get("labels") or item.get("objects") or []
    if not isinstance(objects, list):
        objects = []
    return boxes_from_bdd_objects(objects, image_size)


def box_record(box: Box, image_size: tuple[int, int] | None = None) -> dict[str, object]:
    width, height = image_size or (0, 0)
    center_x = ((box.x1 + box.x2) / 2.0) / width if width else None
    center_y = ((box.y1 + box.y2) / 2.0) / height if height else None
    return {
        "class_name": box.class_name,
        "confidence": box.confidence,
        "bbox": [box.x1, box.y1, box.x2, box.y2],
        "area_ratio": box.area_ratio,
        "normalized_center": [center_x, center_y] if center_x is not None else None,
        "occluded": box.occluded,
        "truncated": box.truncated,
    }


def match_greedy_indices(
    predictions: list[Box],
    ground_truth: list[Box],
    iou_threshold: float,
) -> tuple[set[int], set[int]]:
    """Dopasowuje ramki 1:1, malejąco według IoU."""
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for gt_index, gt_box in enumerate(ground_truth):
            if prediction.class_name != gt_box.class_name:
                continue
            overlap = iou(prediction, gt_box)
            if overlap >= iou_threshold:
                candidates.append((overlap, prediction_index, gt_index))
    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    for _overlap, prediction_index, gt_index in sorted(candidates, reverse=True):
        if prediction_index in used_predictions or gt_index in used_ground_truth:
            continue
        used_predictions.add(prediction_index)
        used_ground_truth.add(gt_index)
    return used_predictions, used_ground_truth


def match_greedy(
    predictions: list[Box],
    ground_truth: list[Box],
    iou_threshold: float,
) -> tuple[int, int, int]:
    used_predictions, used_ground_truth = match_greedy_indices(
        predictions, ground_truth, iou_threshold
    )
    return (
        len(used_predictions),
        len(predictions) - len(used_predictions),
        len(ground_truth) - len(used_ground_truth),
    )


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_pairs_from_dir(pairs_dir: Path) -> list[tuple[Path, Path]]:
    """Zwraca pary obraz–etykieta z katalogu `images/`+`labels/` albo z katalogu płaskiego."""
    root = Path(pairs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Nie znaleziono katalogu par: {root}")
    images_dir = root / "images"
    labels_dir = root / "labels"
    search_images = images_dir if images_dir.is_dir() else root
    search_labels = labels_dir if labels_dir.is_dir() else root
    image_files = {
        path.stem: path
        for path in search_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    label_files = {
        path.stem: path
        for path in search_labels.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    }
    return [
        (image_files[stem], label_files[stem])
        for stem in sorted(image_files.keys() & label_files.keys())
    ]


def build_predict_overrides(
    *,
    imgsz: int | None = None,
    nms_iou: float | None = None,
    max_det: int | None = None,
    agnostic_nms: bool | None = None,
) -> dict[str, object]:
    """Buduje predict_overrides tylko z jawnie podanych parametrów inferencji."""
    overrides: dict[str, object] = {}
    if imgsz is not None:
        overrides["imgsz"] = imgsz
    if nms_iou is not None:
        overrides["iou"] = nms_iou
    if max_det is not None:
        overrides["max_det"] = max_det
    if agnostic_nms is not None:
        overrides["agnostic_nms"] = agnostic_nms
    return overrides


def extract_pairs(archive_path: Path, extract_dir: Path) -> list[tuple[Path, Path]]:
    """Bezpiecznie wypakowuje płaską kopię wspólnych par JPG/JSON."""
    images_dir = extract_dir / "images"
    labels_dir = extract_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_members: dict[str, zipfile.ZipInfo] = {}
    label_members: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename).name
            suffix = Path(name).suffix.lower()
            if suffix in {".jpg", ".jpeg", ".png"}:
                image_members[Path(name).stem] = member
            elif suffix == ".json":
                label_members[Path(name).stem] = member
        common_stems = sorted(image_members.keys() & label_members.keys())
        for stem in common_stems:
            image_member = image_members[stem]
            image_destination = images_dir / Path(image_member.filename).name
            label_destination = labels_dir / Path(label_members[stem].filename).name
            if not image_destination.exists():
                image_destination.write_bytes(archive.read(image_member))
            if not label_destination.exists():
                label_destination.write_bytes(archive.read(label_members[stem]))
    return [
        (
            next(images_dir.glob(f"{stem}.*")),
            labels_dir / Path(label_members[stem].filename).name,
        )
        for stem in common_stems
    ]


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def parse_thresholds(raw: str | None, fallback: float) -> tuple[float, ...]:
    if not raw:
        return (round(fallback, 4),)
    if raw.strip().lower() == "ui":
        return DEFAULT_SWEEP
    try:
        values = tuple(float(value.strip().replace(",", ".")) for value in raw.split(";"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Progi rozdziel średnikami, np. 0.15;0.35;0.55") from exc
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("Każdy próg musi należeć do przedziału (0, 1].")
    return tuple(sorted(set(round(value, 4) for value in values)))


def _new_class_counts() -> dict[str, dict[str, int]]:
    return {class_name: {"tp": 0, "fp": 0, "fn": 0} for class_name in SUPPORTED_OBJECT_CLASSES}


def _metrics(counts: dict[str, dict[str, int]]) -> dict[str, object]:
    per_class = {name: prf(**values) for name, values in sorted(counts.items())}
    micro_counts = {
        key: sum(values[key] for values in counts.values()) for key in ("tp", "fp", "fn")
    }
    return {"micro": prf(**micro_counts), "per_class": per_class}


def _size_group(area_ratio: float | None) -> str:
    if area_ratio is None or area_ratio < 0.01:
        return SIZE_BINS[0]
    if area_ratio < 0.05:
        return SIZE_BINS[1]
    return SIZE_BINS[2]


def _bool_group(value: bool | None) -> str:
    return "tak" if value is True else "nie" if value is False else "brak danych"


def _record_recall_group(
    groups: dict[str, dict[str, dict[str, int]]],
    dimension: str,
    group: str,
    class_name: str,
    matched: bool,
) -> None:
    record = (
        groups.setdefault(dimension, {})
        .setdefault(group, {})
        .setdefault(class_name, {"tp": 0, "fn": 0})
    )
    record["tp" if matched else "fn"] += 1


def _finalize_recall_groups(
    groups: dict[str, dict[str, dict[str, int]]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for dimension, dimension_groups in groups.items():
        finalized_groups: dict[str, object] = {}
        for group, class_counts in dimension_groups.items():
            total_tp = sum(values["tp"] for values in class_counts.values())
            total_fn = sum(values["fn"] for values in class_counts.values())
            per_class = {}
            for class_name, values in sorted(class_counts.items()):
                denominator = values["tp"] + values["fn"]
                per_class[class_name] = {
                    **values,
                    "recall": round(values["tp"] / denominator, 6) if denominator else None,
                }
            denominator = total_tp + total_fn
            finalized_groups[group] = {
                "tp": total_tp,
                "fn": total_fn,
                "recall": round(total_tp / denominator, 6) if denominator else None,
                "per_class": per_class,
            }
        result[dimension] = finalized_groups
    return result


def bdd_archive_candidates() -> tuple[Path, ...]:
    """Lokalne miejsca, w których można trzymać podzbiór BDD poza ZIP źródłowym."""
    return (
        PROJECT_ROOT / "data" / BDD_ARCHIVE_NAME,
        PROJECT_ROOT / BDD_ARCHIVE_NAME,
        Path.home() / "Desktop" / BDD_ARCHIVE_NAME,
    )


def resolve_bdd_archive(
    explicit: Path | None,
    candidates: Sequence[Path] | None = None,
) -> Path | None:
    """Zwraca wskazane archiwum albo pierwsze istniejące z listy kandydatów."""
    if explicit is not None:
        return Path(explicit)
    for path in bdd_archive_candidates() if candidates is None else candidates:
        if path.is_file():
            return path
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help=(
            "Archiwum z parami JPG/JSON. Gdy pominięte, skrypt szuka "
            f"{BDD_ARCHIVE_NAME} w data/, katalogu projektu i na Pulpicie."
        ),
    )
    parser.add_argument(
        "--extract-dir", type=Path, default=PROJECT_ROOT / "data" / "bdd_eval_extracted"
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_OBJECT_CONFIDENCE)
    parser.add_argument(
        "--confidence-sweep",
        help='"ui" oznacza progi 0.05–0.95; własne progi rozdziel średnikami.',
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "data" / "output" / "bdd_eval_metrics.json"
    )
    parser.add_argument(
        "--save-detections",
        type=Path,
        default=None,
        help=(
            "Zapisuje zrzut detekcji i GT per obraz (JSON). Ten sam plik wchodzi do "
            "scripts/compare_priority_policies.py bez ponownej inferencji."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Opcjonalny limit par (do próbnego przebiegu).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Jawna rozdzielczość wejściowa YOLO. Pominięcie = domyślna Ultralytics (640).",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=None,
        dest="nms_iou",
        help="Próg IoU tłumienia niemaksymalnego. Pominięcie = domyślna Ultralytics (0,7).",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=None,
        dest="max_det",
        help="Limit detekcji na obraz. Pominięcie = domyślna Ultralytics (300).",
    )
    parser.add_argument(
        "--agnostic-nms",
        dest="agnostic_nms",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tłumienie niezależne od klasy. Pominięcie = domyślna Ultralytics (False).",
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    archive = resolve_bdd_archive(args.zip)
    if archive is None or not archive.is_file():
        if args.zip is not None:
            parser.error(f"Nie znaleziono archiwum: {args.zip}")
        searched = "; ".join(str(path) for path in bdd_archive_candidates())
        parser.error(
            "Nie znaleziono archiwum BDD100K. Podaj --zip albo umieść "
            f"{BDD_ARCHIVE_NAME} w data/ albo na Pulpicie. Szukano: {searched}"
        )
    if not 0.0 < args.iou <= 1.0:
        parser.error("Próg IoU musi należeć do przedziału (0, 1].")
    try:
        thresholds = parse_thresholds(args.confidence_sweep, args.confidence)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    analysis_threshold = 0.35 if 0.35 in thresholds else thresholds[0]
    minimum_threshold = min(thresholds)

    model_path = validate_local_yolo_model()
    model_hash = file_sha256(model_path)
    predict_overrides = build_predict_overrides(
        imgsz=args.imgsz,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
    )
    detector = ObjectDetector(
        model_path=model_path,
        confidence_threshold=minimum_threshold,
        predict_overrides=predict_overrides,
    )
    pairs = extract_pairs(archive, args.extract_dir)
    if not pairs:
        parser.error("W archiwum nie znaleziono wspólnych par obrazu i etykiety JSON.")

    counts_by_threshold = {threshold: _new_class_counts() for threshold in thresholds}
    prediction_totals = {threshold: 0 for threshold in thresholds}
    recall_groups: dict[str, dict[str, dict[str, int]]] = {}
    image_metadata = {key: Counter() for key in ("weather", "scene", "timeofday")}
    failed_images: list[dict[str, str]] = []
    images_with_supported_gt = 0
    total_gt_boxes = 0
    evaluated_images = 0
    detections_by_image: dict[str, list[Box]] = {}
    ground_truth_by_image: dict[str, list[Box]] = {}
    detection_dump: list[dict[str, object]] = []
    if args.max_images is not None:
        if args.max_images <= 0:
            parser.error("--max-images musi być liczbą dodatnią.")
        pairs = pairs[: args.max_images]

    for index, (image_path, label_path) in enumerate(pairs, start=1):
        try:
            with Image.open(image_path) as pil_image:
                image_rgb = pil_image.convert("RGB")
                width, height = image_rgb.size
                image = np.asarray(image_rgb)
            label_data = json.loads(label_path.read_text(encoding="utf-8"))
            attributes = label_data.get("attributes") or {}
            ground_truth = load_gt_boxes(label_path, (width, height))
            detections = detector.detect(image, confidence_threshold=minimum_threshold)
        except (OSError, UnidentifiedImageError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            failed_images.append({"file": image_path.name, "error": type(exc).__name__})
            continue

        evaluated_images += 1
        for key in image_metadata:
            image_metadata[key][str(attributes.get(key, "brak danych"))] += 1
        if ground_truth:
            images_with_supported_gt += 1
        total_gt_boxes += len(ground_truth)
        all_predictions = detections_to_boxes(detections)
        image_id = image_path.stem
        detections_by_image[image_id] = all_predictions
        ground_truth_by_image[image_id] = ground_truth
        detection_dump.append(
            {
                "id": image_id,
                "file": image_path.name,
                "width": width,
                "height": height,
                "attributes": {
                    key: str(attributes.get(key, "brak danych")) for key in image_metadata
                },
                "detections": [box_record(box, (width, height)) for box in all_predictions],
                "ground_truth": [box_record(box, (width, height)) for box in ground_truth],
            }
        )

        for threshold in thresholds:
            predictions = [item for item in all_predictions if item.confidence >= threshold]
            prediction_totals[threshold] += len(predictions)
            for class_name in SUPPORTED_OBJECT_CLASSES:
                class_predictions = [item for item in predictions if item.class_name == class_name]
                class_ground_truth = [
                    item for item in ground_truth if item.class_name == class_name
                ]
                tp, fp, fn = match_greedy(class_predictions, class_ground_truth, args.iou)
                counts = counts_by_threshold[threshold][class_name]
                counts["tp"] += tp
                counts["fp"] += fp
                counts["fn"] += fn

        analysis_predictions = [
            item for item in all_predictions if item.confidence >= analysis_threshold
        ]
        _used_predictions, used_gt = match_greedy_indices(
            analysis_predictions, ground_truth, args.iou
        )
        for gt_index, gt_box in enumerate(ground_truth):
            matched = gt_index in used_gt
            _record_recall_group(
                recall_groups,
                "rozmiar_ramki",
                _size_group(gt_box.area_ratio),
                gt_box.class_name,
                matched,
            )
            _record_recall_group(
                recall_groups,
                "zasłonięcie",
                _bool_group(gt_box.occluded),
                gt_box.class_name,
                matched,
            )
            _record_recall_group(
                recall_groups, "ucięcie", _bool_group(gt_box.truncated), gt_box.class_name, matched
            )
            _record_recall_group(
                recall_groups,
                "polozenie_w_kadrze",
                "centralne" if is_central_box(gt_box, (width, height)) else "pozacentralne",
                gt_box.class_name,
                matched,
            )
            for key in image_metadata:
                _record_recall_group(
                    recall_groups,
                    key,
                    str(attributes.get(key, "brak danych")),
                    gt_box.class_name,
                    matched,
                )
        if index % 100 == 0 or index == len(pairs):
            print(f"Przetworzono {index}/{len(pairs)} par.", flush=True)

    metrics_by_threshold: dict[str, object] = {}
    curve: list[dict[str, float | int]] = []
    for threshold in thresholds:
        metrics = _metrics(counts_by_threshold[threshold])
        metrics["predicted_boxes"] = prediction_totals[threshold]
        key = f"{threshold:.2f}"
        metrics_by_threshold[key] = metrics
        micro = metrics["micro"]
        curve.append(
            {
                "confidence": threshold,
                "precision": micro["precision"],
                "recall": micro["recall"],
                "f1": micro["f1"],
            }
        )
    best_point = max(curve, key=lambda row: (row["f1"], row["confidence"]))
    analysis_key = f"{analysis_threshold:.2f}"
    per_class_ap = {
        name: average_precision_from_threshold_metrics(metrics_by_threshold, name)
        for name in sorted(SUPPORTED_OBJECT_CLASSES)
    }
    average_precision = {
        "method": "interpolacja VOC na N progach wyniku ufności, nie pełny ranking detekcji",
        "n_confidence_thresholds": len(thresholds),
        "iou": args.iou,
        "micro": average_precision_from_threshold_metrics(metrics_by_threshold),
        "macro_mean": (
            round(sum(per_class_ap.values()) / len(per_class_ap), 6) if per_class_ap else 0.0
        ),
        "per_class": per_class_ap,
        "note": "Przy maksymalnej kompletności poniżej 1 część całki VOC do recall=1 jest zerowa.",
    }
    ranking_ap = ranking_average_precision_report(
        detections_by_image, ground_truth_by_image, args.iou
    )

    report = {
        "app_version": APP_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": "BDD100K fixed detection subset (2000 image-label pairs)",
            "source_archive_name": archive.name,
            "source_archive_sha256": file_sha256(archive),
            "selection_protocol": {
                "used_pairs": "wszystkie wspólne pary nazw, uporządkowane alfabetycznie",
                "random_sampling": False,
                "seed": None,
                "stratification": False,
                "source_split": "brak tej informacji w metadanych archiwum",
            },
            "pairs_found": len(pairs),
            "images_evaluated": evaluated_images,
            "failed_images": failed_images,
            "images_with_supported_ground_truth": images_with_supported_gt,
            "ground_truth_boxes": total_gt_boxes,
            "predicted_boxes_at_analysis_threshold": prediction_totals[analysis_threshold],
            "image_metadata_distribution": {
                key: dict(sorted(counter.items())) for key, counter in image_metadata.items()
            },
            "class_mapping": BDD_TO_APP,
            "citation": "Yu et al., BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning, CVPR 2020",
            "license_note": "Warunki wykorzystania danych należy sprawdzić u właściciela BDD100K.",
        },
        "protocol": {
            "model": model_path.name,
            "model_sha256": model_hash,
            "expected_model_sha256": EXPECTED_YOLO_SHA256,
            "base_prediction_confidence": minimum_threshold,
            "evaluated_confidence_thresholds": thresholds,
            "analysis_threshold": analysis_threshold,
            "iou_matching_threshold": args.iou,
            "matching": "zachłanne dopasowanie 1:1 malejąco według IoU, oddzielnie dla każdej klasy",
            "supported_classes": sorted(SUPPORTED_OBJECT_CLASSES),
            "imgsz": args.imgsz,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "agnostic_nms": args.agnostic_nms,
            "predict_overrides": predict_overrides,
        },
        "metrics_at_analysis_threshold": metrics_by_threshold[analysis_key],
        "metrics_by_confidence": metrics_by_threshold,
        "precision_recall_curve_micro": curve,
        "best_micro_f1_on_evaluated_thresholds": best_point,
        "average_precision": average_precision,
        "average_precision_ranking": ranking_ap,
        "exploratory_recall_analysis_at_analysis_threshold": _finalize_recall_groups(recall_groups),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano raport: {args.out}")
    print(json.dumps(report["metrics_at_analysis_threshold"]["micro"], ensure_ascii=False))
    print("Najlepszy punkt F1:", json.dumps(best_point, ensure_ascii=False))
    print("AP (siatka progów):", json.dumps(average_precision, ensure_ascii=False))
    print("AP (pełny ranking):", json.dumps(ranking_ap, ensure_ascii=False))
    if args.save_detections is not None:
        dump = {
            "app_version": APP_VERSION,
            "created_at_utc": report["created_at_utc"],
            "source_archive_name": archive.name,
            "iou": args.iou,
            "base_prediction_confidence": minimum_threshold,
            "images": detection_dump,
        }
        args.save_detections.parent.mkdir(parents=True, exist_ok=True)
        args.save_detections.write_text(
            json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Zapisano zrzut detekcji: {args.save_detections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
