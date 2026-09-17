"""Poza zakresem PE1: ewaluacja na oficjalnym splicie val BDD100K (10 000).

Zbiorem PE1 w pracy jest archiwum 2000 par (`evaluate_bdd_pairs.py`). Ten skrypt
pozostaje w repozytorium jako narzędzie, nie jako zapowiedziany wynik.

Wejście: katalog obrazów val oraz plik etykiet 2D (det_val.json albo
bdd100k_labels_images_val.json). Surowych zdjęć skrypt nie kopiuje do repo.
Wynik zawiera zarówno AP z siatki progów, jak i AP z pełnego rankingu ufności.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
for extra in (SRC_DIR, SCRIPTS_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import evaluate_bdd_pairs as evaluation  # noqa: E402

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

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def resolve_image_path(images_dir: Path, name: str) -> Path | None:
    candidate = Path(name)
    direct = images_dir / candidate.name
    if direct.is_file():
        return direct
    stem = candidate.stem
    for suffix in IMAGE_SUFFIXES:
        path = images_dir / f"{stem}{suffix}"
        if path.is_file():
            return path
    nested = images_dir / candidate
    if nested.is_file():
        return nested
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Oficjalny JSON detekcji val (tablica ramek z polem labels i box2d).",
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_OBJECT_CONFIDENCE)
    parser.add_argument("--confidence-sweep", help='"ui" albo progi rozdzielone średnikami.')
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "output" / "bdd_val10k_metrics.json",
    )
    parser.add_argument("--save-detections", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if not args.images_dir.is_dir():
        parser.error(f"Nie znaleziono katalogu obrazów: {args.images_dir}")
    if not args.labels.is_file():
        parser.error(f"Nie znaleziono etykiet: {args.labels}")
    if not 0.0 < args.iou <= 1.0:
        parser.error("Próg IoU musi należeć do przedziału (0, 1].")
    try:
        thresholds = evaluation.parse_thresholds(args.confidence_sweep, args.confidence)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    frames = evaluation.load_official_bdd_frames(args.labels)
    if args.max_images is not None:
        if args.max_images <= 0:
            parser.error("--max-images musi być liczbą dodatnią.")
        frames = frames[: args.max_images]
    if not frames:
        parser.error("Plik etykiet nie zawiera żadnej ramki.")

    analysis_threshold = 0.35 if 0.35 in thresholds else thresholds[0]
    minimum_threshold = min(thresholds)
    model_path = validate_local_yolo_model()
    model_hash = file_sha256(model_path)
    detector = ObjectDetector(model_path=model_path, confidence_threshold=minimum_threshold)

    counts_by_threshold = {threshold: evaluation._new_class_counts() for threshold in thresholds}
    prediction_totals = {threshold: 0 for threshold in thresholds}
    image_metadata = {key: Counter() for key in ("weather", "scene", "timeofday")}
    failed_images: list[dict[str, str]] = []
    missing_images: list[str] = []
    images_with_supported_gt = 0
    total_gt_boxes = 0
    evaluated_images = 0
    detections_by_image: dict[str, list] = {}
    ground_truth_by_image: dict[str, list] = {}
    detection_dump: list[dict] = []

    for index, frame in enumerate(frames, start=1):
        name = str(frame.get("name") or frame.get("file") or "")
        image_path = resolve_image_path(args.images_dir, name) if name else None
        if image_path is None:
            missing_images.append(name or f"<brak nazwy {index}>")
            continue
        try:
            with Image.open(image_path) as pil_image:
                image_rgb = pil_image.convert("RGB")
                width, height = image_rgb.size
                image = np.asarray(image_rgb)
            attributes = frame.get("attributes") or {}
            ground_truth = evaluation.load_gt_boxes_from_official_item(frame, (width, height))
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
        all_predictions = [
            evaluation.Box(
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
                "detections": [
                    evaluation.box_record(box, (width, height)) for box in all_predictions
                ],
                "ground_truth": [
                    evaluation.box_record(box, (width, height)) for box in ground_truth
                ],
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
                tp, fp, fn = evaluation.match_greedy(
                    class_predictions, class_ground_truth, args.iou
                )
                counts = counts_by_threshold[threshold][class_name]
                counts["tp"] += tp
                counts["fp"] += fp
                counts["fn"] += fn
        if index % 100 == 0 or index == len(frames):
            print(f"Przetworzono {index}/{len(frames)} ramek.", flush=True)

    metrics_by_threshold: dict[str, object] = {}
    curve: list[dict[str, float | int]] = []
    for threshold in thresholds:
        metrics = evaluation._metrics(counts_by_threshold[threshold])
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
        name: evaluation.average_precision_from_threshold_metrics(metrics_by_threshold, name)
        for name in sorted(SUPPORTED_OBJECT_CLASSES)
    }
    average_precision = {
        "method": "interpolacja VOC na N progach wyniku ufności, nie pełny ranking detekcji",
        "n_confidence_thresholds": len(thresholds),
        "iou": args.iou,
        "micro": evaluation.average_precision_from_threshold_metrics(metrics_by_threshold),
        "macro_mean": (
            round(sum(per_class_ap.values()) / len(per_class_ap), 6) if per_class_ap else 0.0
        ),
        "per_class": per_class_ap,
    }
    ranking_ap = evaluation.ranking_average_precision_report(
        detections_by_image, ground_truth_by_image, args.iou
    )
    report = {
        "app_version": APP_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": "BDD100K official detection validation split",
            "images_dir": str(args.images_dir),
            "labels_file": args.labels.name,
            "labels_sha256": file_sha256(args.labels),
            "frames_in_labels": len(frames),
            "images_evaluated": evaluated_images,
            "missing_images": missing_images[:50],
            "missing_images_count": len(missing_images),
            "failed_images": failed_images,
            "images_with_supported_ground_truth": images_with_supported_gt,
            "ground_truth_boxes": total_gt_boxes,
            "image_metadata_distribution": {
                key: dict(sorted(counter.items())) for key, counter in image_metadata.items()
            },
            "class_mapping": evaluation.BDD_TO_APP,
            "citation": (
                "Yu et al., BDD100K: A Diverse Driving Dataset for Heterogeneous "
                "Multitask Learning, CVPR 2020"
            ),
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
        },
        "metrics_at_analysis_threshold": metrics_by_threshold[analysis_key],
        "metrics_by_confidence": metrics_by_threshold,
        "precision_recall_curve_micro": curve,
        "best_micro_f1_on_evaluated_thresholds": best_point,
        "average_precision": average_precision,
        "average_precision_ranking": ranking_ap,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano raport: {args.out}")
    print(json.dumps(report["metrics_at_analysis_threshold"]["micro"], ensure_ascii=False))
    print("AP ranking:", json.dumps(ranking_ap["micro"], ensure_ascii=False))
    if args.save_detections is not None:
        dump = {
            "app_version": APP_VERSION,
            "created_at_utc": report["created_at_utc"],
            "labels_file": args.labels.name,
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
