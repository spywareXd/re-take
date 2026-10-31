"""Data shapes of the preprocessing step: transcript segments, word timestamps and the sentence units built from them."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    speaker: str = "speaker_0"
    confidence: float | None = 1.0
    words: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            id=data["id"],
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data.get("text", "")),
            speaker=str(data.get("speaker", "speaker_0")),
            confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
            words=list(data.get("words", [])),
        )


@dataclass(slots=True)
class WordTimestamp:
    word: str
    start: float
    end: float
    confidence: float | None = 1.0
    source_segment_id: int = 1
    speaker: str = "speaker_0"


@dataclass(slots=True)
class ComparableUnit:
    unit_id: str
    source_segment_id: int
    start: float
    end: float
    raw_text: str
    normalized_text: str
    tokens: list[str]
    token_set: set[str]
    speaker: str = "speaker_0"
    confidence: float | None = 1.0
    is_below_min_token_length: bool = False
