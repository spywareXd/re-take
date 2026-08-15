from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


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


@dataclass(slots=True)
class CandidateUnitRef:
    unit_id: str
    segment_id: int
    start: float
    end: float
    raw_text: str = ""
    speaker: str = "speaker_0"
    confidence: float | None = 1.0

    @classmethod
    def from_unit(cls, unit: ComparableUnit) -> "CandidateUnitRef":
        return cls(
            unit_id=unit.unit_id,
            segment_id=unit.source_segment_id,
            start=unit.start,
            end=unit.end,
            raw_text=unit.raw_text,
            speaker=unit.speaker,
            confidence=unit.confidence,
        )


@dataclass(slots=True)
class CandidateScores:
    exact_match: float = 0.0
    near_exact_match: float = 0.0
    jaccard: float = 0.0
    sequence: float = 0.0
    containment: float = 0.0
    bigram: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class CandidatePair:
    candidate_id: int
    unit_a: CandidateUnitRef
    unit_b: CandidateUnitRef
    scores: CandidateScores
    combined_score: float
    restart_phrase_detected: bool
    containment_direction: str | None
    recency_weight: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "unit_a": asdict(self.unit_a),
            "unit_b": asdict(self.unit_b),
            "scores": self.scores.to_dict(),
            "combined_score": self.combined_score,
            "restart_phrase_detected": self.restart_phrase_detected,
            "containment_direction": self.containment_direction,
            "recency_weight": self.recency_weight,
            **self.metadata,
        }
