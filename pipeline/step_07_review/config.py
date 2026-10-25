"""Configuration for the review documents (candidates / groups) the editor (SimpleCut) reads. See backend-handoff.md, schema simplecut.retakes/2."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA = "simplecut.retakes/2"
STAGE_CANDIDATES = "candidates"
STAGE_GROUPS = "groups"
GENERATOR_NAME = "retake"
GENERATOR_VERSION = "0.4.0"


@dataclass
class ReviewConfig:
    cache_dir: str = "./cache"                        # transcript + units + pair scores per media fingerprint
    labels_path: str = "./labels.jsonl"               # one JSON object per user answer (training data)
    skipped_path: str = "./labels_skipped.jsonl"      # pairs the user left to the model: same fields, label "skipped"
    max_questions: int = 25                           # ambiguous pairs emitted per video; the rest are decided by ce >= t_hi
    unsure_band_enabled: bool = True                  # False reproduces the old single-threshold (t_amb) behaviour
    scorer_model_dir: Optional[str] = None            # default models/final
    scorer_device: Optional[str] = None
    transcription: dict[str, Any] = field(default_factory=lambda: {"provider": "whisperx", "model_name": "large-v2", "device": "cuda", "compute_type": "float16"})
    generator_name: str = GENERATOR_NAME
    generator_version: str = GENERATOR_VERSION
