from __future__ import annotations

from dataclasses import dataclass, field

SEMANTIC_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

SEMANTIC_WEIGHTS: dict[str, float] = {
    "semantic_similarity": 0.50,
    "lexical_combined_score": 0.30,
    "containment_bonus": 0.20,
}

STAGE2_RETAKE_THRESHOLD: float = 0.65  # placeholder — tune with eval, do not treat as final
STAGE2_AMBIGUOUS_FLOOR: float = 0.35   # reuses Stage 1's discard threshold as the lower bound
DISAGREEMENT_FLAG_THRESHOLD: float = 0.35  # |lexical - semantic| above this gets flagged, regardless of stage2_score band


@dataclass(slots=True)
class SemanticMatchingConfig:
    embedding_model: str = SEMANTIC_EMBEDDING_MODEL
    semantic_weights: dict[str, float] = field(default_factory=lambda: dict(SEMANTIC_WEIGHTS))
    stage2_retake_threshold: float = STAGE2_RETAKE_THRESHOLD
    stage2_ambiguous_floor: float = STAGE2_AMBIGUOUS_FLOOR
    disagreement_flag_threshold: float = DISAGREEMENT_FLAG_THRESHOLD


default_semantic_config = SemanticMatchingConfig()
