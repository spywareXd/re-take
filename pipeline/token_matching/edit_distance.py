from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .schemas import ComparableUnit


@dataclass(slots=True)
class ContainmentResult:
    score: float
    direction: str | None


def sequence_similarity_score(a: ComparableUnit, b: ComparableUnit) -> float:
    return SequenceMatcher(a=a.tokens, b=b.tokens).ratio()


def containment_details(a: ComparableUnit, b: ComparableUnit) -> ContainmentResult:
    if not a.tokens or not b.tokens:
        return ContainmentResult(score=0.0, direction=None)

    shared = len(set(a.tokens) & set(b.tokens))
    denominator = min(len(a.tokens), len(b.tokens))
    score = shared / denominator if denominator else 0.0

    if len(a.tokens) < len(b.tokens):
        direction = "a_subset_of_b" if score > 0 else None
    elif len(b.tokens) < len(a.tokens):
        direction = "b_subset_of_a" if score > 0 else None
    else:
        direction = "equal_length"

    return ContainmentResult(score=score, direction=direction)


def containment_score(a: ComparableUnit, b: ComparableUnit) -> float:
    return containment_details(a, b).score
