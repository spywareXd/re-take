from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from pipeline.step_02_preprocessing.schemas import ComparableUnit, TranscriptSegment, WordTimestamp  # noqa: F401


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
    lexical_score: float
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
            "lexical_score": self.lexical_score,
            "combined_score": self.combined_score,
            "restart_phrase_detected": self.restart_phrase_detected,
            "containment_direction": self.containment_direction,
            "recency_weight": self.recency_weight,
            **self.metadata,
        }
