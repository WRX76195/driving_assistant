"""Opcjonalny bootstrap na poziomie obrazu na rozpakowanym archiwum 2000 par.

Pomiaru nie wykonano w zakresie PE1. Skrypt jest w archiwum, żeby dało się
odtworzyć polecenie z Tabeli 8.4 pracy:

    python bootstrap_obrazowy_bdd.py --pairs-dir <katalog_2000_par> --replicates 2000 --seed 20260827 --output bootstrap.json

Predykcje liczone są samodzielnie (jedna inferencja na obraz, potem resampling
całych kadrów). Przedziały Cloppera-Pearsona z Tabeli 7.12 pozostają dolnym
oszacowaniem, dopóki ten pomiar nie zostanie uruchomiony.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

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

DEFAULT_REPLICATES = 2000
DEFAULT_SEED = 20260827


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-dir",
        type=Path,
        required=True,
        help="Rozpakowany katalog 2000 par (images/+labels/ albo płaski zestaw JPG/JSON).",
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=DEFAULT_OBJECT_CONFIDENCE)
    return parser


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 6),
        "p025": round(float(np.percentile(values, 2.5)), 6),
        "p975": round(float(np.percentile(values, 97.5)), 6),
    }


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error("--replicates musi być liczbą dodatnią.")
    if not 0.0 < args.iou <= 1.0:
        parser.error("Próg IoU musi należeć do przedziału (0, 1].")
    try:
        pairs = evaluation.load_pairs_from_dir(args.pairs_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not pairs:
        parser.error(f"W {args.pairs_dir} nie znaleziono wspólnych par obrazu i etykiety.")

    model_path = validate_local_yolo_model()
    detector = ObjectDetector(
        model_path=model_path,
        confidence_threshold=args.confidence,
    )
    per_image: list[tuple[int, int, int]] = []
    failed = 0
    for index, (image_path, label_path) in enumerate(pairs, start=1):
        inferred = evaluation.infer_pair(
            image_path,
            label_path,
            detector,
            confidence_threshold=args.confidence,
        )
        if inferred is None:
            failed += 1
            continue
        _image_id, detections, ground_truth, _size = inferred
        tp, fp, fn = evaluation.match_greedy(detections, ground_truth, args.iou)
        per_image.append((tp, fp, fn))
        if index % 100 == 0 or index == len(pairs):
            print(f"Przetworzono {index}/{len(pairs)} par.", flush=True)

    if not per_image:
        parser.error("Nie udało się policzyć predykcji dla żadnego obrazu.")

    counts = np.asarray(per_image, dtype=np.int64)
    observed = evaluation.prf(
        int(counts[:, 0].sum()),
        int(counts[:, 1].sum()),
        int(counts[:, 2].sum()),
    )
    rng = np.random.default_rng(args.seed)
    n_images = len(per_image)
    precision = np.empty(args.replicates, dtype=np.float64)
    recall = np.empty(args.replicates, dtype=np.float64)
    f1 = np.empty(args.replicates, dtype=np.float64)
    for replicate in range(args.replicates):
        sample = counts[rng.integers(0, n_images, size=n_images)]
        metrics = evaluation.prf(
            int(sample[:, 0].sum()),
            int(sample[:, 1].sum()),
            int(sample[:, 2].sum()),
        )
        precision[replicate] = float(metrics["precision"])
        recall[replicate] = float(metrics["recall"])
        f1[replicate] = float(metrics["f1"])

    report = {
        "app_version": APP_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "note": (
            "Opcjonalny bootstrap obrazowy z Tabeli 8.4. Resampling ze zwracaniem "
            "całych kadrów. Nie wchodzi do wyników PE1, dopóki pomiar nie zostanie "
            "wykonany i zapisany."
        ),
        "pairs_dir": str(args.pairs_dir),
        "pairs_found": len(pairs),
        "images_evaluated": n_images,
        "failed_images": failed,
        "model": model_path.name,
        "model_sha256": model_sha256(model_path),
        "confidence": args.confidence,
        "iou_matching_threshold": args.iou,
        "replicates": args.replicates,
        "seed": args.seed,
        "observed_micro": observed,
        "bootstrap_micro": {
            "precision": _percentile_interval(precision),
            "recall": _percentile_interval(recall),
            "f1": _percentile_interval(f1),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zapisano: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
