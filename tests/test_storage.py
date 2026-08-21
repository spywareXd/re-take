import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step_03_token_matching.config import PAUSE_SPLIT_THRESHOLD_SECONDS
from pipeline.step_03_token_matching.schemas import ComparableUnit
from pipeline.step_03_token_matching.storage import load_comparable_units, save_comparable_units


def _sample_units() -> list[ComparableUnit]:
    return [
        ComparableUnit(
            unit_id="unit_0",
            source_segment_id=1,
            start=0.0,
            end=4.22,
            speaker="speaker_0",
            raw_text="[redacted sentence] now",
            normalized_text="[redacted sentence] now",
            tokens=["[redacted sentence]", "now"],
            token_set={"[redacted sentence]", "now"},
            confidence=0.79,
            is_below_min_token_length=False,
        ),
        ComparableUnit(
            unit_id="unit_1",
            source_segment_id=1,
            start=5.92,
            end=7.92,
            speaker="speaker_0",
            raw_text="you know the answer about this new method",
            normalized_text="you know the answer about this new method",
            tokens=["you", "know", "the", "answer", "about", "this", "new", "method"],
            token_set={"you", "know", "the", "answer", "about", "this", "new", "method"},
            confidence=0.95,
            is_below_min_token_length=False,
        ),
    ]


def test_storage_round_trip(tmp_path):
    output_file = tmp_path / "test_units.json"
    original_units = _sample_units()

    save_comparable_units(
        units=original_units,
        output_path=output_file,
        source_transcript="sample_f.mp4",
        pause_split_threshold_seconds=PAUSE_SPLIT_THRESHOLD_SECONDS,
    )

    loaded_units, metadata = load_comparable_units(output_file)

    assert len(loaded_units) == len(original_units)
    for orig, loaded in zip(original_units, loaded_units):
        assert loaded.unit_id == orig.unit_id
        assert loaded.source_segment_id == orig.source_segment_id
        assert loaded.start == orig.start
        assert loaded.end == orig.end
        assert loaded.speaker == orig.speaker
        assert loaded.raw_text == orig.raw_text
        assert loaded.normalized_text == orig.normalized_text
        assert loaded.tokens == orig.tokens
        assert loaded.token_set == set(orig.tokens)
        assert loaded.confidence == orig.confidence
        assert loaded.is_below_min_token_length == orig.is_below_min_token_length

    assert metadata["source_transcript"] == "sample_f.mp4"
    assert metadata["pause_split_threshold_seconds"] == PAUSE_SPLIT_THRESHOLD_SECONDS
    assert metadata["unit_count"] == 2


def test_storage_custom_threshold_preserved(tmp_path):
    output_file = tmp_path / "custom_threshold_units.json"
    custom_threshold = 1.25
    original_units = _sample_units()

    save_comparable_units(
        units=original_units,
        output_path=output_file,
        source_transcript="custom_test.mp4",
        pause_split_threshold_seconds=custom_threshold,
    )

    raw_json = json.loads(output_file.read_text(encoding="utf-8"))
    assert raw_json["pause_split_threshold_seconds"] == 1.25

    loaded_units, metadata = load_comparable_units(output_file)
    assert metadata["pause_split_threshold_seconds"] == 1.25


def test_token_set_not_in_raw_json(tmp_path):
    output_file = tmp_path / "check_token_set.json"
    original_units = _sample_units()

    save_comparable_units(
        units=original_units,
        output_path=output_file,
        source_transcript="test.mp4",
    )

    raw_json = json.loads(output_file.read_text(encoding="utf-8"))
    for unit_dict in raw_json["units"]:
        assert "token_set" not in unit_dict
        assert "tokens" in unit_dict
