from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class CandidateScores(BaseModel):
    exact_match: float = 0.0
    near_exact_match: float = 0.0
    jaccard: float = 0.0
    sequence: float = 0.0
    containment: float = 0.0
    bigram: float = 0.0


class CandidateUnitRef(BaseModel):
    unit_id: str
    segment_id: int
    start: float
    end: float
    raw_text: str = ""
    speaker: str = "speaker_0"
    confidence: Optional[float] = 1.0


class CandidatePair(BaseModel):
    candidate_id: int
    unit_a: CandidateUnitRef
    unit_b: CandidateUnitRef
    scores: CandidateScores
    combined_score: float
    lexical_score: float = 0.0
    restart_phrase_detected: bool = False
    containment_direction: Optional[str] = None
    recency_weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
