import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step_02_preprocessing.config import PAUSE_SPLIT_THRESHOLD_SECONDS
from pipeline.step_02_preprocessing.schemas import WordTimestamp
from pipeline.step_02_preprocessing.unit_splitter import (
    build_comparable_unit,
    extract_segment_words,
    flatten_transcript_words,
    split_transcript_into_units,
    split_words_by_pause,
)


def _wt(word: str, start: float, end: float, speaker: str = "speaker_0", confidence: float = 1.0, seg_id: int = 1) -> WordTimestamp:
    return WordTimestamp(word=word, start=start, end=end, speaker=speaker, confidence=confidence, source_segment_id=seg_id)


def test_gap_under_threshold_does_not_split():
    # Gap is strictly under threshold
    threshold = PAUSE_SPLIT_THRESHOLD_SECONDS
    words = [
        _wt("hello", 0.0, 1.0),
        _wt("world", 1.0 + threshold * 0.5, 2.0),
    ]
    groups = split_words_by_pause(words, pause_threshold=threshold)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_gap_exactly_equal_to_threshold_does_not_split():
    # Gap is exactly equal to threshold
    threshold = PAUSE_SPLIT_THRESHOLD_SECONDS
    words = [
        _wt("first", 0.0, 1.0),
        _wt("second", 1.0 + threshold, 2.8),
    ]
    groups = split_words_by_pause(words, pause_threshold=threshold)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_gap_strictly_greater_than_threshold_splits():
    # Gap is strictly greater than threshold (+0.01s)
    threshold = PAUSE_SPLIT_THRESHOLD_SECONDS
    words = [
        _wt("first", 0.0, 1.0),
        _wt("second", round(1.0 + threshold + 0.01, 4), 2.8),
    ]
    groups = split_words_by_pause(words, pause_threshold=threshold)
    assert len(groups) == 2
    assert [w.word for w in groups[0]] == ["first"]
    assert [w.word for w in groups[1]] == ["second"]


def test_custom_pause_threshold_override():
    # Custom threshold 0.5s with gap 0.6s
    words = [
        _wt("alpha", 0.0, 1.0),
        _wt("beta", 1.6, 2.0),
    ]
    groups_default = split_words_by_pause(words, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)
    assert len(groups_default) == 1

    groups_custom = split_words_by_pause(words, pause_threshold=0.5)
    assert len(groups_custom) == 2


def test_sentence_ending_punctuation_splits_with_no_pause():
    # Terminal punctuation '.', '!', '?' should split even with 0 gap
    words = [
        _wt("hello.", 0.0, 1.0),
        _wt("world", 1.0, 2.0),
        _wt("again!", 2.0, 3.0),
        _wt("ready?", 3.0, 4.0),
        _wt("yes", 4.0, 5.0),
    ]
    groups = split_words_by_pause(words, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)
    assert len(groups) == 4
    assert [w.word for w in groups[0]] == ["hello."]
    assert [w.word for w in groups[1]] == ["world", "again!"]
    assert [w.word for w in groups[2]] == ["ready?"]
    assert [w.word for w in groups[3]] == ["yes"]


def test_speaker_change_forces_split_with_zero_pause():
    words = [
        _wt("hi", 0.0, 1.0, speaker="speaker_0"),
        _wt("hello", 1.0, 2.0, speaker="speaker_1"),
    ]
    groups = split_words_by_pause(words, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)
    assert len(groups) == 2
    assert groups[0][0].speaker == "speaker_0"
    assert groups[1][0].speaker == "speaker_1"


def test_empty_word_list_handled_without_error():
    assert split_words_by_pause([], pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS) == []
    assert flatten_transcript_words({"segments": []}) == []
    assert split_transcript_into_units({"segments": []}, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS) == []


def test_single_word_transcript_handled_correctly():
    words = [_wt("hello", 1.5, 2.0)]
    groups = split_words_by_pause(words, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)
    assert len(groups) == 1
    assert len(groups[0]) == 1

    unit = build_comparable_unit(groups[0], unit_id="unit_0")
    assert unit.unit_id == "unit_0"
    assert unit.start == 1.5
    assert unit.end == 2.0
    assert unit.raw_text == "hello"
    assert unit.tokens == ["hello"]


def test_unit_start_end_from_words_not_asr_segment():
    # Words span 1.2s to 3.8s inside an ASR segment that starts at 0.0s and ends at 5.0s
    words = [
        _wt("first", 1.2, 2.0, seg_id=42),
        _wt("second", 2.2, 3.8, seg_id=42),
    ]
    unit = build_comparable_unit(words, unit_id="unit_42")
    assert unit.start == 1.2
    assert unit.end == 3.8
    assert unit.source_segment_id == 42


def test_average_confidence_computed_correctly():
    words = [
        _wt("word1", 0.0, 1.0, confidence=0.8),
        _wt("word2", 1.0, 2.0, confidence=0.6),
        _wt("word3", 2.0, 3.0, confidence=1.0),
    ]
    unit = build_comparable_unit(words, unit_id="unit_1")
    expected_confidence = (0.8 + 0.6 + 1.0) / 3.0
    assert pytest.approx(unit.confidence, rel=1e-4) == expected_confidence


def test_consecutive_segments_zero_gap_remain_separate():
    """
    Two consecutive faster-whisper segments with 0.0s gap must remain two separate units.
    """
    mock_transcript = {
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 2.0,
                "speaker": "speaker_0",
                "text": "First take here",
                "words": [
                    {"word": "First", "start": 0.0, "end": 0.6, "confidence": 0.9},
                    {"word": "take", "start": 0.6, "end": 1.2, "confidence": 0.9},
                    {"word": "here", "start": 1.2, "end": 2.0, "confidence": 0.9},
                ],
            },
            {
                "id": 2,
                "start": 2.0,
                "end": 4.0,
                "speaker": "speaker_0",
                "text": "Second take now",
                "words": [
                    {"word": "Second", "start": 2.0, "end": 2.6, "confidence": 0.9},
                    {"word": "take", "start": 2.6, "end": 3.2, "confidence": 0.9},
                    {"word": "now", "start": 3.2, "end": 4.0, "confidence": 0.9},
                ],
            },
        ]
    }

    units = split_transcript_into_units(mock_transcript, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)
    assert len(units) == 2
    assert units[0].unit_id == "1:0"
    assert units[0].source_segment_id == 1
    assert units[0].raw_text == "First take here"

    assert units[1].unit_id == "2:0"
    assert units[1].source_segment_id == 2
    assert units[1].raw_text == "Second take now"


def test_consecutive_segments_gap_under_threshold_remain_separate():
    """
    Two consecutive segments with a 0.76s gap (< 0.8s threshold) must NOT be merged.
    """
    mock_transcript = {
        "segments": [
            {
                "id": 3,
                "start": 12.0,
                "end": 13.0,
                "speaker": "speaker_0",
                "text": "Fellas you wonder why Monoxidil works for some dudes but not",
                "words": [
                    {"word": "Fellas", "start": 12.0, "end": 12.5, "confidence": 0.8},
                    {"word": "not", "start": 12.6, "end": 13.0, "confidence": 0.8},
                ],
            },
            {
                "id": 4,
                "start": 13.76,
                "end": 14.8,
                "speaker": "speaker_0",
                "text": "Fellas you wonder why some Monoxidil",
                "words": [
                    # Gap from 13.0 to 13.76 is 0.76s (under 0.8s threshold)
                    {"word": "Fellas", "start": 13.76, "end": 14.2, "confidence": 0.8},
                    {"word": "Monoxidil", "start": 14.3, "end": 14.8, "confidence": 0.8},
                ],
            },
        ]
    }

    units = split_transcript_into_units(mock_transcript, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)
    assert len(units) == 2
    assert units[0].source_segment_id == 3
    assert units[1].source_segment_id == 4
    assert units[0].unit_id == "3:0"
    assert units[1].unit_id == "4:0"
    assert units[0].raw_text == "Fellas not"
    assert units[1].raw_text == "Fellas Monoxidil"


def test_segment_with_no_internal_pause_passes_as_single_unit():
    mock_transcript = {
        "segments": [
            {
                "id": 1,
                "start": 0.0,
                "end": 2.0,
                "text": "Continuous sentence without pause",
                "words": [
                    {"word": "Continuous", "start": 0.0, "end": 0.5},
                    {"word": "sentence", "start": 0.55, "end": 1.0},
                    {"word": "without", "start": 1.05, "end": 1.5},
                    {"word": "pause", "start": 1.55, "end": 2.0},
                ],
            }
        ]
    }
    units = split_transcript_into_units(mock_transcript, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)
    assert len(units) == 1
    assert units[0].unit_id == "1:0"
    assert units[0].raw_text == "Continuous sentence without pause"


def test_output_transcript_segment_boundaries_and_splits():
    transcript_path = PROJECT_ROOT / "data" / "raw" / "transcripts" / "output_transcript.json"
    if not transcript_path.exists():
        pytest.skip("output_transcript.json not found")

    from pipeline.step_02_preprocessing.normalize import load_transcript_units

    units = load_transcript_units(transcript_path, pause_threshold=PAUSE_SPLIT_THRESHOLD_SECONDS)
    assert len(units) >= 1

    # Invariant: unit IDs are always formatted parent_seg_id:sub_idx
    for u in units:
        seg_id, sub_idx = u.unit_id.split(":")
        assert int(seg_id) == u.source_segment_id
        assert int(sub_idx) >= 0
        assert u.start <= u.end

    # Invariant: units never span across multiple source segments
    segment_ids = [u.source_segment_id for u in units]
    assert segment_ids == sorted(segment_ids)

