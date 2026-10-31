from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step_02_preprocessing.config import PAUSE_SPLIT_THRESHOLD_SECONDS
from pipeline.step_02_preprocessing.schemas import ComparableUnit
from pipeline.step_02_preprocessing.storage import load_comparable_units, save_comparable_units
from pipeline.step_03_token_matching.storage import load_semantic_candidates, save_semantic_candidates


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


def _sample_semantic_candidates() -> list[dict]:
    return [
        {
            "candidate_id": 5,
            "unit_a": {
                "unit_id": "2:1",
                "source_segment_id": 2,
                "start": 8.73,
                "end": 10.81,
                "speaker": "speaker_0",
                "raw_text": "Now we're going to learn about tea time, guys.",
                "normalized_text": "[redacted sentence] spider man guys",
                "tokens": ["[redacted sentence]", "spider", "man", "guys"],
                "confidence": 0.75,
                "is_below_min_token_length": False,
            },
            "unit_b": {
                "unit_id": "5:0",
                "source_segment_id": 5,
                "start": 17.71,
                "end": 21.46,
                "speaker": "speaker_0",
                "raw_text": "Hey guys, let's talk about tea time.",
                "normalized_text": "hey guys lets talk about spider man",
                "tokens": ["hey", "guys", "lets", "talk", "about", "spider", "man"],
                "confidence": 0.63,
                "is_below_min_token_length": False,
            },
            "scores": {
                "exact_match": 0.0,
                "near_exact_match": 0.0,
                "jaccard": 0.5385,
                "sequence": 0.339,
                "containment": 0.4444,
                "bigram": 0.2857,
            },
            "lexical_score": 0.43,
            "combined_score": 0.35,
            "restart_phrase_detected": False,
            "containment_direction": "b_subset_of_a",
            "recency_weight": 0.82,
            "semantic_score": 0.81,
            "stage2_score": 0.63,
            "lexical_semantic_disagreement": 0.46,
            "high_disagreement_flag": True,
            "classification": "ambiguous",
        }
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


def test_semantic_storage_round_trip(tmp_path):
    output_file = tmp_path / "test_semantic_candidates.json"
    candidates = _sample_semantic_candidates()

    save_semantic_candidates(
        candidates=candidates,
        output_path=output_file,
        source_candidates="data/processed/candidates/token_candidates.json",
        embedding_model="all-MiniLM-L6-v2",
        stage2_retake_threshold=0.65,
        stage2_ambiguous_floor=0.35,
        disagreement_flag_threshold=0.35,
    )

    loaded_candidates, metadata = load_semantic_candidates(output_file)

    assert len(loaded_candidates) == 1
    c = loaded_candidates[0]
    # Check Stage 1 fields preserved
    assert c["candidate_id"] == 5
    assert c["unit_a"]["unit_id"] == "2:1"
    assert c["unit_b"]["unit_id"] == "5:0"
    assert c["combined_score"] == 0.35
    assert c["containment_direction"] == "b_subset_of_a"
    # Check Stage 2 fields preserved
    assert c["semantic_score"] == 0.81
    assert c["stage2_score"] == 0.63
    assert c["lexical_semantic_disagreement"] == 0.46
    assert c["high_disagreement_flag"] is True
    assert c["classification"] == "ambiguous"

    # Check metadata
    assert metadata["source_candidates"] == "data/processed/candidates/token_candidates.json"
    assert metadata["embedding_model"] == "all-MiniLM-L6-v2"
    assert metadata["stage2_retake_threshold"] == 0.65
    assert metadata["stage2_ambiguous_floor"] == 0.35
    assert metadata["disagreement_flag_threshold"] == 0.35
    assert metadata["candidate_count"] == 1


def test_semantic_storage_custom_parameters(tmp_path):
    output_file = tmp_path / "custom_semantic.json"
    candidates = _sample_semantic_candidates()
    custom_weights = {"semantic_similarity": 0.60, "lexical_combined_score": 0.20, "containment_bonus": 0.20}

    save_semantic_candidates(
        candidates=candidates,
        output_path=output_file,
        source_candidates="custom_source.json",
        embedding_model="custom-minilm-v3",
        semantic_weights=custom_weights,
        stage2_retake_threshold=0.80,
        stage2_ambiguous_floor=0.40,
        disagreement_flag_threshold=0.25,
    )

    raw_json = json.loads(output_file.read_text(encoding="utf-8"))
    assert raw_json["embedding_model"] == "custom-minilm-v3"
    assert raw_json["stage2_retake_threshold"] == 0.80
    assert raw_json["stage2_ambiguous_floor"] == 0.40
    assert raw_json["disagreement_flag_threshold"] == 0.25
    assert raw_json["semantic_weights"] == custom_weights

    loaded, meta = load_semantic_candidates(output_file)
    assert meta["embedding_model"] == "custom-minilm-v3"
    assert meta["stage2_retake_threshold"] == 0.80
    assert meta["stage2_ambiguous_floor"] == 0.40
    assert meta["disagreement_flag_threshold"] == 0.25
