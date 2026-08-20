from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Union

from .config import default_config
from .schemas import ComparableUnit, TranscriptSegment

try:
    from models.segment import Segment
    from models.transcript import Transcript
except ImportError:
    Segment = None  # type: ignore
    Transcript = None  # type: ignore


CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}

PUNCTUATION_RE = re.compile(r"[^\w\s']")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _expand_contractions(text: str) -> str:
    expanded = text
    for needle, replacement in CONTRACTIONS.items():
        expanded = re.sub(re.escape(needle), replacement, expanded, flags=re.IGNORECASE)
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
) -> list[ComparableUnit]:
    """
    Loads a transcript from a JSON file path, dict, or Transcript model and returns comparable units.
    Supports both wrapped format {"segments": [...]} and raw segment list [...].
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(source, dict):
        raw_data = source
    elif hasattr(source, "segments"):
        raw_data = {"segments": [s.model_dump() if hasattr(s, "model_dump") else s for s in source.segments]}
    else:
        raise ValueError(f"Unsupported transcript source type: {type(source)}")

    if isinstance(raw_data, dict):
        raw_segments = raw_data.get("segments", [])
    elif isinstance(raw_data, list):
        raw_segments = raw_data
    else:
        raise ValueError("Invalid transcript JSON format.")

    units: list[ComparableUnit] = []
    for raw_seg in raw_segments:
        units.extend(split_into_units(raw_seg, config=config))
    return units
