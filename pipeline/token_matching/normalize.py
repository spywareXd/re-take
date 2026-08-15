from __future__ import annotations

import re

from .config import default_config
from .schemas import ComparableUnit, TranscriptSegment


CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}

PUNCTUATION_RE = re.compile(r"[^\w\s']")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _expand_contractions(text: str) -> str:
    expanded = text
    for needle, replacement in CONTRACTIONS.items():
        expanded = re.sub(re.escape(needle), replacement, expanded, flags=re.IGNORECASE)
    return expanded


def normalize(text: str) -> str:
    normalized = _expand_contractions(text.lower())
    normalized = PUNCTUATION_RE.sub(" ", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize(text)
    if not normalized:
        return []
    return normalized.split()


def _split_text(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(text) if chunk.strip()]
    return chunks if chunks else ([text.strip()] if text.strip() else [])


def split_into_units(segment: TranscriptSegment, config: object = default_config) -> list[ComparableUnit]:
    min_tokens = getattr(config, "min_tokens", default_config.min_tokens)
    chunks = _split_text(segment.text)
    units: list[ComparableUnit] = []

    for index, chunk in enumerate(chunks):
        normalized_text = normalize(chunk)
        tokens = normalized_text.split() if normalized_text else []
        units.append(
            ComparableUnit(
                unit_id=f"{segment.id}:{index}",
                source_segment_id=segment.id,
                start=segment.start,
                end=segment.end,
                raw_text=chunk,
                normalized_text=normalized_text,
                tokens=tokens,
                token_set=set(tokens),
                is_below_min_token_length=len(tokens) < min_tokens,
            )
        )

    return units
