from __future__ import annotations

from .schemas import ComparableUnit


def _strip_fillers(tokens: list[str], filler_words: set[str]) -> list[str]:
    return [token for token in tokens if token not in filler_words]


def exact_match_score(a: ComparableUnit, b: ComparableUnit) -> float:
    if len(a.tokens) != len(b.tokens):
        return 0.0
    return 1.0 if a.tokens == b.tokens else 0.0


def near_exact_match_score(
    a: ComparableUnit,
    b: ComparableUnit,
    filler_words: set[str] | None = None,
) -> float:
    filler_words = filler_words or set()
    stripped_a = _strip_fillers(a.tokens, filler_words)
    stripped_b = _strip_fillers(b.tokens, filler_words)
    if len(stripped_a) != len(stripped_b):
        return 0.0
    return 1.0 if stripped_a == stripped_b else 0.0
