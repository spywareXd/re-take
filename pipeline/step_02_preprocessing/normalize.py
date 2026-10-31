from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Union

from .config import PAUSE_SPLIT_THRESHOLD_SECONDS, default_config
from .schemas import ComparableUnit, TranscriptSegment

try:
    from models.segment import Segment
    from models.transcript import Transcript
except ImportError:
    Segment = None  # type: ignore
    Transcript = None  # type: ignore


APOSTROPHE_VARIANTS_RE = re.compile(r"[\u2018\u2019\u201A\u201B\u2032\u00B4`‘´]")
PUNCTUATION_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Compiled regex patterns for comprehensive contraction and spoken reduction expansions
CONTRACTION_PATTERNS = [
    # Specific irregular contractions
    (re.compile(r"\bcan't\b", re.IGNORECASE), "cannot"),
    (re.compile(r"\bwon't\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bshan't\b", re.IGNORECASE), "shall not"),
    (re.compile(r"\bain't\b", re.IGNORECASE), "is not"),
    (re.compile(r"\blet's\b", re.IGNORECASE), "let us"),
    (re.compile(r"\by'all\b", re.IGNORECASE), "you all"),
    (re.compile(r"\bc'mon\b", re.IGNORECASE), "come on"),
    (re.compile(r"\bo'clock\b", re.IGNORECASE), "o clock"),
    (re.compile(r"\bma'am\b", re.IGNORECASE), "madam"),

    # Spoken reductions and informal variants
    (re.compile(r"\bgonna\b", re.IGNORECASE), "going to"),
    (re.compile(r"\bwanna\b", re.IGNORECASE), "want to"),
    (re.compile(r"\bgotta\b", re.IGNORECASE), "got to"),
    (re.compile(r"\bkinda\b", re.IGNORECASE), "kind of"),
    (re.compile(r"\bsorta\b", re.IGNORECASE), "sort of"),
    (re.compile(r"\blemme\b", re.IGNORECASE), "let me"),
    (re.compile(r"\bgimme\b", re.IGNORECASE), "give me"),
    (re.compile(r"\bdunno\b", re.IGNORECASE), "do not know"),
    (re.compile(r"\b(i'mma|imma)\b", re.IGNORECASE), "i am going to"),
    (re.compile(r"\b('cause|cuz)\b", re.IGNORECASE), "because"),
    (re.compile(r"\balright\b", re.IGNORECASE), "all right"),

    # Pronoun / Noun + 're -> are
    (re.compile(r"\b(you|we|they|who|what|where|how|there|these|those)'re\b", re.IGNORECASE), r"\1 are"),

    # Pronoun / Noun + 's -> is
    (re.compile(r"\b(it|that|what|there|here|who|how|where|when|why|he|she)'s\b", re.IGNORECASE), r"\1 is"),

    # Pronoun + 'm -> am
    (re.compile(r"\bi'm\b", re.IGNORECASE), "i am"),

    # Pronoun / Modal + 've -> have
    (re.compile(r"\b(i|you|we|they|who|what|would|should|could|must|might|may)'ve\b", re.IGNORECASE), r"\1 have"),

    # Pronoun / Noun + 'll -> will
    (re.compile(r"\b(i|you|he|she|it|we|they|that|what|who|there)'ll\b", re.IGNORECASE), r"\1 will"),

    # Pronoun / Modal + 'd -> would
    (re.compile(r"\b(i|you|he|she|it|we|they|that|who|what|how|where|why|would|should|could)'d\b", re.IGNORECASE), r"\1 would"),

    # Negative verbs (n't -> not)
    (re.compile(r"\b(do|does|did|is|are|was|were|have|has|had|would|should|could|must|need|dare)n't\b", re.IGNORECASE), r"\1 not"),

    # General fallbacks for remaining contractions
    (re.compile(r"(\w+?)n't\b", re.IGNORECASE), r"\1 not"),
    (re.compile(r"(\w+)'re\b", re.IGNORECASE), r"\1 are"),
    (re.compile(r"(\w+)'ve\b", re.IGNORECASE), r"\1 have"),
    (re.compile(r"(\w+)'ll\b", re.IGNORECASE), r"\1 will"),
    (re.compile(r"(\w+)'d\b", re.IGNORECASE), r"\1 would"),
    (re.compile(r"(\w+)'m\b", re.IGNORECASE), r"\1 am"),
]


def _expand_contractions(text: str) -> str:
    # Unify all apostrophe variants to standard ASCII single quote
    expanded = APOSTROPHE_VARIANTS_RE.sub("'", text)
    for pattern, replacement in CONTRACTION_PATTERNS:
        expanded = pattern.sub(replacement, expanded)
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


def split_into_units(
    segment: Union[TranscriptSegment, Any, dict[str, Any]],
    config: object = default_config,
) -> list[ComparableUnit]:
    """
    Splits a segment (TranscriptSegment, Pydantic Segment, or dict) into normalized comparable units.
    """
    min_tokens = getattr(config, "min_tokens", default_config.min_tokens)

    # Extract attributes whether it's dataclass, Pydantic model, or dict
    if isinstance(segment, dict):
        seg_id = segment["id"]
        start = float(segment["start"])
        end = float(segment["end"])
        text = str(segment.get("text", ""))
        speaker = str(segment.get("speaker", "speaker_0"))
        confidence = float(segment["confidence"]) if segment.get("confidence") is not None else 1.0
    else:
        seg_id = getattr(segment, "id")
        start = float(getattr(segment, "start"))
        end = float(getattr(segment, "end"))
        text = str(getattr(segment, "text", ""))
        speaker = str(getattr(segment, "speaker", "speaker_0"))
        confidence = getattr(segment, "confidence", 1.0)
        if confidence is not None:
            confidence = float(confidence)

    chunks = _split_text(text)
    units: list[ComparableUnit] = []

    for index, chunk in enumerate(chunks):
        normalized_text = normalize(chunk)
        tokens = normalized_text.split() if normalized_text else []
        units.append(
            ComparableUnit(
                unit_id=f"{seg_id}:{index}",
                source_segment_id=seg_id,
                start=start,
                end=end,
                raw_text=chunk,
                normalized_text=normalized_text,
                tokens=tokens,
                token_set=set(tokens),
                speaker=speaker,
                confidence=confidence,
                is_below_min_token_length=len(tokens) < min_tokens,
            )
        )

    return units


def load_transcript_units(
    source: Union[str, Path, dict[str, Any], Any],
    config: object = default_config,
    pause_threshold: float = PAUSE_SPLIT_THRESHOLD_SECONDS,
) -> list[ComparableUnit]:
    """
    Loads a transcript from a JSON file path, dict, or Transcript model and returns comparable units
    derived from word timestamp pause splitting, punctuation boundaries, and speaker changes.
    """
    from .unit_splitter import split_transcript_into_units

    threshold = (
        pause_threshold
        if pause_threshold is not None
        else getattr(config, "pause_split_threshold_seconds", PAUSE_SPLIT_THRESHOLD_SECONDS)
    )
    return split_transcript_into_units(source, pause_threshold=threshold, config=config)
