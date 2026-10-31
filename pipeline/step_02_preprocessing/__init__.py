"""Step 2: preprocessing. Turns a transcript into normalised sentence units (pause, punctuation and speaker splits) and saves them."""
from .config import PAUSE_SPLIT_THRESHOLD_SECONDS, PreprocessingConfig, default_config
from .normalize import load_transcript_units, normalize, split_into_units, tokenize
from .schemas import ComparableUnit, TranscriptSegment, WordTimestamp
from .storage import load_comparable_units, save_comparable_units
from .unit_splitter import (
    build_comparable_unit,
    extract_segment_words,
    flatten_transcript_words,
    split_transcript_into_units,
    split_words_by_pause,
)

__all__ = [
    "ComparableUnit",
    "PAUSE_SPLIT_THRESHOLD_SECONDS",
    "PreprocessingConfig",
    "TranscriptSegment",
    "WordTimestamp",
    "build_comparable_unit",
    "default_config",
    "extract_segment_words",
    "flatten_transcript_words",
    "load_comparable_units",
    "load_transcript_units",
    "normalize",
    "save_comparable_units",
    "split_into_units",
    "split_transcript_into_units",
    "split_words_by_pause",
    "tokenize",
]
