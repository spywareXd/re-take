"""Settings of the preprocessing step: how a transcript is cut into sentence units."""
from __future__ import annotations

from dataclasses import dataclass

PAUSE_SPLIT_THRESHOLD_SECONDS: float = 0.7


@dataclass(slots=True)
class PreprocessingConfig:
    min_tokens: int = 3
    pause_split_threshold_seconds: float = PAUSE_SPLIT_THRESHOLD_SECONDS


default_config = PreprocessingConfig()
