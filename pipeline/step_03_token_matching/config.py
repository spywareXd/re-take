from __future__ import annotations

from dataclasses import dataclass, field


PAUSE_SPLIT_THRESHOLD_SECONDS: float = 0.7

LEXICAL_WEIGHTS = {
    "sequence": 0.30,
    "jaccard": 0.25,
    "containment": 0.25,
    "bigram": 0.20,
}


@dataclass(slots=True)
class TokenMatchingConfig:
    min_tokens: int = 3
    window_seconds: float = 120.0
    recency_decay_seconds: float = 45.0
    candidate_threshold: float = 0.35
    pause_split_threshold_seconds: float = PAUSE_SPLIT_THRESHOLD_SECONDS
    lexical_weights: dict[str, float] = field(default_factory=lambda: dict(LEXICAL_WEIGHTS))
    restart_phrase_lexicon: tuple[str, ...] = (
        "sorry",
        "wait",
        "again",
        "let me restart",
        "cut",
        "one more time",
    )
    filler_words: tuple[str, ...] = ("um", "uh", "like", "you know")


default_config = TokenMatchingConfig()
