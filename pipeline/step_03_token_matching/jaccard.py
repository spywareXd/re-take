from __future__ import annotations

from .schemas import ComparableUnit


def jaccard_score(a: ComparableUnit, b: ComparableUnit) -> float:
    union = a.token_set | b.token_set
    if not union:
        return 0.0
    return len(a.token_set & b.token_set) / len(union)
