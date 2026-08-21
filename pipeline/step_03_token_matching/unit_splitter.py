from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Union

from .config import PAUSE_SPLIT_THRESHOLD_SECONDS, default_config
from .normalize import normalize, tokenize
from .schemas import ComparableUnit, TranscriptSegment, WordTimestamp

TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?]$")


def flatten_transcript_words(
    transcript: Union[str, Path, dict[str, Any], list[Any], Any],
) -> list[WordTimestamp]:
    """
    Flattens all segments' word lists into a single ordered list for the whole transcript,
    carrying source_segment_id and speaker per word.
    """
    if isinstance(transcript, (str, Path)):
        path = Path(transcript)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    elif isinstance(transcript, (dict, list)):
        raw_data = transcript
    elif hasattr(transcript, "segments"):
        raw_data = {"segments": [s.model_dump() if hasattr(s, "model_dump") else s for s in transcript.segments]}
    else:
        raise ValueError(f"Unsupported transcript source type: {type(transcript)}")

    if isinstance(raw_data, dict):
        segments = raw_data.get("segments", [])
    elif isinstance(raw_data, list):
        segments = raw_data
    else:
        raise ValueError("Invalid transcript JSON format.")

    flattened: list[WordTimestamp] = []

    for seg in segments:
        if isinstance(seg, dict):
            seg_id = int(seg.get("id", len(flattened) + 1))
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            speaker = str(seg.get("speaker", "speaker_0"))
            seg_conf = float(seg["confidence"]) if seg.get("confidence") is not None else 1.0
            words = seg.get("words", [])
            text = str(seg.get("text", ""))
        else:
            seg_id = int(getattr(seg, "id", len(flattened) + 1))
            seg_start = float(getattr(seg, "start", 0.0))
            seg_end = float(getattr(seg, "end", 0.0))
            speaker = str(getattr(seg, "speaker", "speaker_0"))
            raw_conf = getattr(seg, "confidence", 1.0)
            seg_conf = float(raw_conf) if raw_conf is not None else 1.0
            words = getattr(seg, "words", []) or []
            text = str(getattr(seg, "text", ""))

        if words:
            for w in words:
                if isinstance(w, dict):
                    w_text = str(w.get("word", "")).strip()
                    w_start = float(w.get("start", seg_start))
                    w_end = float(w.get("end", seg_end))
                    w_conf = float(w["confidence"]) if w.get("confidence") is not None else seg_conf
                else:
                    w_text = str(getattr(w, "word", "")).strip()
                    w_start = float(getattr(w, "start", seg_start))
                    w_end = float(getattr(w, "end", seg_end))
                    raw_w_conf = getattr(w, "confidence", seg_conf)
                    w_conf = float(raw_w_conf) if raw_w_conf is not None else seg_conf

                if not w_text:
                    continue

                flattened.append(
                    WordTimestamp(
                        word=w_text,
                        start=w_start,
                        end=w_end,
                        confidence=w_conf,
                        source_segment_id=seg_id,
                        speaker=speaker,
                    )
                )
        else:
            # Fallback when word-level timestamps are not present in segment
            raw_words = text.strip().split()
            if not raw_words:
                continue
            duration = max(0.0, seg_end - seg_start)
            step = duration / len(raw_words) if len(raw_words) > 0 else 0.0
            for idx, rw in enumerate(raw_words):
                w_start = round(seg_start + idx * step, 2)
                w_end = round(seg_start + (idx + 1) * step, 2)
                flattened.append(
                    WordTimestamp(
                        word=rw,
                        start=w_start,
                        end=w_end,
                        confidence=seg_conf,
                        source_segment_id=seg_id,
                        speaker=speaker,
                    )
                )

    return flattened


def split_words_by_pause(
    words: list[WordTimestamp],
    pause_threshold: float = PAUSE_SPLIT_THRESHOLD_SECONDS,
) -> list[list[WordTimestamp]]:
    """
    Splits the flattened word list into groups. A new group starts when any of the following is true:
    - the gap between current word's end and next word's start is strictly greater than pause_threshold
    - the current word ends with terminal punctuation (., !, ?)
    - the next word has a different speaker than the current word (forced split)

    Boundary rule: gap == pause_threshold does NOT split (only strictly greater does).
    Non-positive gaps (negative/zero/overlapping) never trigger a pause split.
    """
    if not words:
        return []

    groups: list[list[WordTimestamp]] = []
    current_group: list[WordTimestamp] = [words[0]]

    for i in range(len(words) - 1):
        curr_word = words[i]
        next_word = words[i + 1]

        gap = next_word.start - curr_word.end
        has_pause_split = gap > pause_threshold
        has_punctuation_split = bool(TERMINAL_PUNCTUATION_RE.search(curr_word.word.strip()))
        has_speaker_split = next_word.speaker != curr_word.speaker

        if has_pause_split or has_punctuation_split or has_speaker_split:
            groups.append(current_group)
            current_group = [next_word]
        else:
            current_group.append(next_word)

    if current_group:
        groups.append(current_group)

    return groups


def build_comparable_unit(
    word_group: list[WordTimestamp],
    unit_id: str,
    min_tokens: int = 3,
) -> ComparableUnit:
    """
    Assembles a ComparableUnit from a word group:
    - joins raw text
    - normalizes and tokenizes
    - sets start and end from first and last word in the group
    - sets speaker from group
    - sets confidence as the mean of word-level confidences
    """
    if not word_group:
        raise ValueError("Cannot build a ComparableUnit from an empty word group.")

    raw_text = " ".join(w.word for w in word_group)
    normalized_text = normalize(raw_text)
    tokens = tokenize(raw_text)
    token_set = set(tokens)

    start = word_group[0].start
    end = word_group[-1].end
    speaker = word_group[0].speaker
    source_segment_id = word_group[0].source_segment_id

    confidences = [w.confidence for w in word_group if w.confidence is not None]
    confidence = (sum(confidences) / len(confidences)) if confidences else 1.0

    return ComparableUnit(
        unit_id=unit_id,
        source_segment_id=source_segment_id,
        start=start,
        end=end,
        raw_text=raw_text,
        normalized_text=normalized_text,
        tokens=tokens,
        token_set=token_set,
        speaker=speaker,
        confidence=confidence,
        is_below_min_token_length=len(tokens) < min_tokens,
    )


def split_transcript_into_units(
    transcript: Union[str, Path, dict[str, Any], list[Any], Any],
    pause_threshold: float = PAUSE_SPLIT_THRESHOLD_SECONDS,
    config: object = default_config,
) -> list[ComparableUnit]:
    """
    End-to-end flow: flatten -> split by pause / punctuation / speaker -> build ComparableUnits.
    """
    words = flatten_transcript_words(transcript)
    if not words:
        return []

    min_tokens = getattr(config, "min_tokens", default_config.min_tokens)
    word_groups = split_words_by_pause(words, pause_threshold=pause_threshold)

    units: list[ComparableUnit] = []
    seg_counters: dict[int, int] = {}

    for group in word_groups:
        seg_id = group[0].source_segment_id
        sub_idx = seg_counters.get(seg_id, 0)
        seg_counters[seg_id] = sub_idx + 1
        unit_id = f"{seg_id}:{sub_idx}"

        unit = build_comparable_unit(group, unit_id=unit_id, min_tokens=min_tokens)
        units.append(unit)

    return units
