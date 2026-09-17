"""Opcjonalna ablacja rozdzielczości wejściowej na rozpakowanym archiwum 2000 par.

    python ablacja_imgsz_bdd.py --pairs-dir <katalog_2000_par> --imgsz 640 960 1280 --output ablacja_imgsz.json
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for extra in (SRC_DIR, SCRIPTS_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import evaluate_bdd_pairs as evaluation

from adaptive_driving_assistant.config import APP_VERSION, DEFAULT_OBJECT_CONFIDENCE
from adaptive_driving_assistant.detector import ObjectDetector, validate_local_yolo_model
from adaptive_driving_assistant.detector import file_sha256 as model_sha256

ANALYSIS_THRESHOLD = DEFAULT_OBJECT_CONFIDENCE
CEILING_THRESHOLD = 0.05
DEFAULT_IMGSZ = (640, 960, 1280)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        required=True,
        help="Rozpakowany katalog 2000 par (images/+labels/ albo płaski zestaw JPG/JSON).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=list(DEFAULT_IMGSZ),
        help="Wartości rozdzielczości wejściowej do porównania (domyślnie 640 960 1280).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Ścieżka JSON z wynikiem.")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--confidence",
        type=float,
        default=ANALYSIS_THRESHOLD,
        help="Próg analizy. Pułap kompletności liczony dodatkowo przy 0,05.",
    )
    return parser


def _score_at_threshold(
    detections_by_image: dict[str, list[evaluation.Box]],
    ground_truth_by_image: dict[str, list[evaluation.Box]],
    *,
    threshold: float,
    iou: float,
) -> dict[str, object]:
    counts = evaluation._new_class_counts()
    predicted_boxes = 0
    small_tp = 0
    small_fn = 0
    for image_id, ground_truth in ground_truth_by_image.items():
        predictions = [
            box
            for box in detections_by_image.get(image_id, [])
            if box.confidence >= threshold
        ]
        predicted_boxes += len(predictions)
        for class_name in counts:
            class_predictions = [item for item in predictions if item.class_name == class_name]
            class_ground_truth = [item for item in ground_truth if item.class_name == class_name]
            tp, fp, fn = evaluation.match_greedy(class_predictions, class_ground_truth, iou)
            counts[class_name]["tp"] += tp
            counts[class_name]["fp"] += fp
            counts[class_name]["fn"] += fn
        _used, used_gt = evaluation.match_greedy_indices(predictions, ground_truth, iou)
        for gt_index, gt_box in enumerate(ground_truth):
            if evaluation._size_group(gt_box.area_ratio) != evaluation.SIZE_BINS[0]:
                continue
            if gt_index in used_gt:
                small_tp += 1
            else:
                small_fn += 1
    metrics = evaluation._metrics(counts)
    small_den = small_tp + small_fn
    metrics["predicted_boxes"] = predicted_boxes
    metrics["small_box_recall"] = (
        round(small_tp / small_den, 6) if small_den else None
    )
    metrics["small_box_tp"] = small_tp
    metrics["small_box_fn"] = small_fn
    return metrics


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if any(value <= 0 for value in args.imgsz):
        parser.error("Każde imgsz musi być liczbą dodatnią.")
    if not 0.0 < args.iou <= 1.0:
        parser.error("Próg IoU musi należeć do przedziału (0, 1].")
    try:
        pairs = evaluation.load_pairs_from_dir(args.pairs_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not pairs:
        parser.error(f"W {args.pairs_dir} nie znaleziono wspólnych par obrazu i etykiety.")

    model_path = validate_local_yolo_model()
    by_imgsz: dict[str, object] = {}
    for imgsz in args.imgsz:
        detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=min(CEILING_THRESHOLD, args.confidence),
            predict_overrides=evaluation.build_predict_overrides(imgsz=imgsz),
        )
        detections_by_image: dict[str, list[evaluation.Box]] = {}
        ground_truth_by_image: dict[str, list[evaluation.Box]] = {}
        failed = 0
        for image_path, label_path in pairs:
            inferred = evaluation.infer_pair(
                image_path,
                label_path,
                detector,
                confidence_threshold=min(CEILING_THRESHOLD, args.confidence),
            )
            if inferred is None:
                failed += 1
                continue
            image_id, detections, ground_truth, _size = inferred
            detections_by_image[image_id] = detections
            ground_truth_by_image[image_id] = ground_truth
        print(f"imgsz={imgsz}: przetworzono {len(detections_by_image)}/{len(pairs)} par.", flush=True)
        by_imgsz[str(imgsz)] = {
            "imgsz": imgsz,
            "images_evaluated": len(detections_by_image),
            "failed_images": failed,
            "metrics_at_analysis_threshold": _score_at_threshold(
                detections_by_image,
                ground_truth_by_image,
                threshold=args.confidence,
                iou=args.iou,
            ),
            "metrics_at_confidence_0.05": _score_at_threshold(
                detections_by_image,
                ground_truth_by_image,
                threshold=CEILING_THRESHOLD,
                iou=args.iou,
            ),
        }

    report = {
        "app_version": APP_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "note": (
            "Opcjonalna ablacja. Nie wchodzi do wyników PE1, dopóki "
            "pomiar nie zostanie wykonany i zapisany."
        ),
        "pairs_dir": str(args.pairs_dir),
        "pairs_found": len(pairs),
        "model": model_path.name,
        "model_sha256": model_sha256(model_path),
        "iou_matching_threshold": args.iou,
        "analysis_threshold": args.confidence,
        "imgsz_values": list(args.imgsz),
        "by_imgsz": by_imgsz,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
