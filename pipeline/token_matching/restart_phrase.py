from __future__ import annotations

from .config import default_config


def restart_phrase_score(text: str, config: object = default_config) -> float:
    lexicon = getattr(config, "restart_phrase_lexicon", default_config.restart_phrase_lexicon)
    lowered = text.lower()
    return 1.0 if any(phrase in lowered for phrase in lexicon) else 0.0
