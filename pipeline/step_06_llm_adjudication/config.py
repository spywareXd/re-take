from __future__ import annotations

import os
from dataclasses import dataclass

ADJUDICATION_MODEL: str = os.environ.get("ADJUDICATION_MODEL", "gemini-2.5-flash-lite")
ADJUDICATION_TEMPERATURE: float = 0.0  # deterministic — classification task, not creative generation
ADJUDICATION_MAX_RETRIES: int = 2     # for transient API failures or validation retries
ADJUDICATION_DEFAULT_PROVIDER: str = os.environ.get("ADJUDICATION_PROVIDER", "gemini")


@dataclass(slots=True)
class AdjudicationConfig:
    """
    Configuration settings for Step 06 LLM Adjudication.
    """
    model: str = ADJUDICATION_MODEL
    temperature: float = ADJUDICATION_TEMPERATURE
    max_retries: int = ADJUDICATION_MAX_RETRIES
    provider: str = ADJUDICATION_DEFAULT_PROVIDER
    api_key: str | None = None
    timeout_seconds: float = 30.0


default_adjudication_config = AdjudicationConfig()
