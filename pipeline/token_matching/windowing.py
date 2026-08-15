from __future__ import annotations

import math

from .config import default_config
from .schemas import ComparableUnit


def _recency_weight(delta_seconds: float, decay_seconds: float) -> float:
    if decay_seconds <= 0:
        return 1.0
    return math.exp(-max(delta_seconds, 0.0) / decay_seconds)


def get_candidates_in_window(
    unit: ComparableUnit,
    all_units: list[ComparableUnit],
    config: object = default_config,
) -> list[tuple[ComparableUnit, float]]:
    window_seconds = getattr(config, "window_seconds", default_config.window_seconds)
    decay_seconds = getattr(config, "recency_decay_seconds", default_config.recency_decay_seconds)
    candidates: list[tuple[ComparableUnit, float]] = []

    for other in all_units:
        if other.unit_id == unit.unit_id:
            continue
        delta_seconds = abs(unit.start - other.start)
        if delta_seconds > window_seconds:
            continue
        candidates.append((other, _recency_weight(delta_seconds, decay_seconds)))

    candidates.sort(key=lambda item: abs(unit.start - item[0].start))
    return candidates
