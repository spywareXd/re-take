from __future__ import annotations

from .schemas import ComparableUnit


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)}


def ngram_overlap_score(a: ComparableUnit, b: ComparableUnit, n: int = 2) -> float:
    ngrams_a = _ngrams(a.tokens, n)
    ngrams_b = _ngrams(b.tokens, n)
    union = ngrams_a | ngrams_b
    if not union:
        return 0.0
    return len(ngrams_a & ngrams_b) / len(union)
