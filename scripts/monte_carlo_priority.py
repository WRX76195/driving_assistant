"""Ocena reguł priorytetu metodą Monte Carlo (odtwarzalna, ziarno stałe)."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from adaptive_driving_assistant.domain import (
    BoundingBox,
    Detection,
    PriorityLevel,
    SceneReadabilityLevel,
    SceneReadabilityResult,
)
from adaptive_driving_assistant.priority import assess_priority, select_primary_detection

CLASS_SHARES_GT = {
    "car": 0.796,
    "person": 0.132,
    "truck": 0.041,
    "bus": 0.015,
    "bicycle": 0.011,
    "motorcycle": 0.005,
}
CLASS_SHARES_DET = {
    "car": 0.82,
    "person": 0.10,
    "truck": 0.04,
    "bus": 0.02,
    "bicycle": 0.01,
    "motorcycle": 0.01,
}
SIZE_BINS = (
    (0.70, 0.002, 0.01),
    (0.20, 0.01, 0.04),
    (0.10, 0.04, 0.20),
)
VRU = {"person", "bicycle", "motorcycle"}
SEED = 20260829
N_SCENES = 20_000
N_PAIRS = 200_000


def _choose_class(rng: random.Random, shares: dict[str, float]) -> str:
    names = list(shares)
    weights = [shares[name] for name in names]
    return rng.choices(names, weights=weights, k=1)[0]


def _choose_area(rng: random.Random) -> float:
    pick = rng.random()
    cumulative = 0.0
    for weight, low, high in SIZE_BINS:
        cumulative += weight
        if pick <= cumulative:
            return rng.uniform(low, high)
    return rng.uniform(0.04, 0.20)


def _synthetic_detection(rng: random.Random, shares: dict[str, float]) -> Detection:
    area = _choose_area(rng)
    central = rng.random() < 0.484
    center = (0.5, 0.55) if central else (0.12, 0.18)
    return Detection(
        class_name=_choose_class(rng, shares),
        confidence=rng.uniform(0.4, 0.99),
        bbox=BoundingBox(0, 0, 10, 10),
        normalized_center=center,
        area_ratio=area,
    )


def _indicator(rng: random.Random) -> SceneReadabilityResult:
    pick = rng.random()
    if pick < 0.70:
        level = SceneReadabilityLevel.GOOD
    elif pick < 0.95:
        level = SceneReadabilityLevel.MODERATE
    else:
        level = SceneReadabilityLevel.LIMITED
    return SceneReadabilityResult(level=level, features={}, thresholds={}, score=0.5)


def _scene(rng: random.Random, shares: dict[str, float]) -> list[Detection]:
    count = rng.randint(0, 25)
    return [_synthetic_detection(rng, shares) for _ in range(count)]


def _evaluate_scenes(shares: dict[str, float], seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    n_with = 0
    not_largest_nor_central = 0
    main_areas: list[float] = []
    max_areas: list[float] = []
    mentioned_share: list[float] = []
    dense_share: list[float] = []
    vru_scenes = 0
    vru_other = 0
    street_pattern_den = 0
    street_pattern_num = 0
    levels_by_n: dict[int, Counter[str]] = {n: Counter() for n in range(26)}
    scores_by_n: dict[int, list[float]] = {n: [] for n in range(26)}
    scenes_meta: list[tuple[int, float, str]] = []

    for _ in range(N_SCENES):
        detections = _scene(rng, shares)
        indicator = _indicator(rng)
        priority = assess_priority(detections, indicator)
        n = len(detections)
        levels_by_n[n][priority.level.value] += 1
        scores_by_n[n].append(priority.score)
        scenes_meta.append((n, priority.score, priority.level.value))
        if not detections:
            continue
        n_with += 1
        primary = select_primary_detection(detections)
        largest = max(detections, key=lambda item: (item.area_ratio, item.confidence))
        if primary.area_ratio + 1e-12 < largest.area_ratio and not primary.is_central:
            not_largest_nor_central += 1
        main_areas.append(primary.area_ratio)
        max_areas.append(largest.area_ratio)
        mentioned_share.append(1.0 / n)
        if n > 10:
            dense_share.append(1.0 / n)
        classes = {item.class_name for item in detections}
        if classes & VRU:
            vru_scenes += 1
            if primary.class_name not in VRU:
                vru_other += 1
        large_central = any(item.is_central and item.area_ratio > 0.10 for item in detections)
        if large_central:
            street_pattern_den += 1
            if primary.area_ratio < 0.01:
                street_pattern_num += 1

    pair_rng = random.Random(seed + 1)
    inversions = 0
    extreme_inversions = 0
    extreme_pairs = 0
    for _ in range(N_PAIRS):
        left = pair_rng.choice(scenes_meta)
        right = pair_rng.choice(scenes_meta)
        if left[0] > right[0] and left[1] < right[1]:
            inversions += 1
        if left[0] >= 10 and 1 <= right[0] <= 5:
            extreme_pairs += 1
            if left[1] < right[1]:
                extreme_inversions += 1

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "n_with_detections": n_with,
        "not_largest_nor_central_pct": 100.0 * not_largest_nor_central / n_with,
        "mean_main_area": _mean(main_areas),
        "mean_max_area": _mean(max_areas),
        "street_pattern_pct": 100.0 * street_pattern_num / street_pattern_den
        if street_pattern_den
        else 0.0,
        "mentioned_share_pct": 100.0 * _mean(mentioned_share),
        "dense_mentioned_share_pct": 100.0 * _mean(dense_share),
        "vru_other_pct": 100.0 * vru_other / vru_scenes if vru_scenes else 0.0,
        "pair_inversion_pct": 100.0 * inversions / N_PAIRS,
        "extreme_inversion_pct": 100.0 * extreme_inversions / extreme_pairs
        if extreme_pairs
        else 0.0,
        "empty_no_message_pct": 100.0
        * levels_by_n[0][PriorityLevel.NO_MESSAGE.value]
        / max(1, sum(levels_by_n[0].values())),
        "levels_by_n": {str(n): dict(counter) for n, counter in levels_by_n.items()},
        "mean_score_by_n": {
            str(n): _mean(values) for n, values in scores_by_n.items() if values
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/priority_monte_carlo_0.3.5.json"),
    )
    args = parser.parse_args()
    payload = {
        "seed": SEED,
        "n_scenes": N_SCENES,
        "n_pairs": N_PAIRS,
        "note": (
            "Wskaźnik cech obrazu jest losowany, ale od 0.3.5 nie wchodzi do "
            "punktacji assess_priority."
        ),
        "gt": _evaluate_scenes(CLASS_SHARES_GT, SEED),
        "det": _evaluate_scenes(CLASS_SHARES_DET, SEED + 17),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
