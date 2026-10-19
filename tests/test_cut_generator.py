"""Unit tests for Step 08: Cut Generator (consuming Step 06 Adjudicated Spans)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from models.retake import CutList, RetakeGroup, TimeSpan
from pipeline.step_08_video_edit.cut_generator import (
    compute_keep_spans,
    generate_cuts_from_adjudicated_spans,
    load_cutlist,
    merge_spans,
    save_cutlist,
)


@pytest.fixture
def sample_units():
    return [
        {"unit_id": "1:0", "start": 1.0, "end": 3.0, "raw_text": "First bad take."},
        {"unit_id": "2:0", "start": 4.0, "end": 6.5, "raw_text": "First good take."},
        {"unit_id": "3:0", "start": 7.0, "end": 9.0, "raw_text": "Middle valid sentence."},
        {"unit_id": "4:0", "start": 10.0, "end": 12.0, "raw_text": "Second bad take."},
        {"unit_id": "5:0", "start": 13.0, "end": 15.5, "raw_text": "Second good take."},
    ]


@pytest.fixture
def sample_adjudicated_spans():
    return [
        {
            "span_id": "span_0",
            "unit_ids": ["1:0", "2:0"],
            "start": 1.0,
            "end": 6.5,
            "is_genuine_retake_group": True,
            "keeper_unit_id": "2:0",
            "units_to_remove": ["1:0"],
            "adjudication_status": "not_required",
            "reasoning": "Standard retake rule",
            "candidate_pairs": [{"candidate_id": 1}],
        },
        {
            "span_id": "span_1",
            "unit_ids": ["4:0", "5:0"],
            "start": 10.0,
            "end": 15.5,
            "is_genuine_retake_group": True,
            "keeper_unit_id": "5:0",
            "units_to_remove": ["4:0"],
            "adjudication_status": "resolved",
            "reasoning": "LLM decided unit 4:0 was false start",
            "candidate_pairs": [{"candidate_id": 2}],
        },
    ]


def test_generate_cuts_basic(sample_adjudicated_spans, sample_units):
    cutlist = generate_cuts_from_adjudicated_spans(
        adjudicated_spans=sample_adjudicated_spans,
        units=sample_units,
        source="test.mp4",
        source_duration=20.0,
    )

    assert isinstance(cutlist, CutList)
    assert cutlist.source == "test.mp4"
    assert cutlist.source_duration == 20.0
    assert len(cutlist.retake_groups) == 2

    # Discard spans should be [1.0, 3.0] and [10.0, 12.0]
    assert len(cutlist.discard_spans) == 2
    assert cutlist.discard_spans[0].start == 1.0
    assert cutlist.discard_spans[0].end == 3.0
    assert cutlist.discard_spans[1].start == 10.0
    assert cutlist.discard_spans[1].end == 12.0

    # Keep spans should cover [0, 1.0], [3.0, 10.0], [12.0, 20.0]
    assert len(cutlist.keep_spans) == 3
    assert cutlist.keep_spans[0].start == 0.0
    assert cutlist.keep_spans[0].end == 1.0
    assert cutlist.keep_spans[1].start == 3.0
    assert cutlist.keep_spans[1].end == 10.0
    assert cutlist.keep_spans[2].start == 12.0
    assert cutlist.keep_spans[2].end == 20.0

    # Output duration should be 20.0 - 2.0 - 2.0 = 16.0
    assert round(cutlist.output_duration, 2) == 16.0


def test_non_retake_group_is_not_cut(sample_units):
    spans = [
        {
            "span_id": "span_false_positive",
            "unit_ids": ["1:0", "2:0"],
            "start": 1.0,
            "end": 6.5,
            "is_genuine_retake_group": False,  # Not a retake
            "keeper_unit_id": None,
            "units_to_remove": [],
            "adjudication_status": "resolved",
            "reasoning": "Intentional repetition for emphasis",
            "candidate_pairs": [{"candidate_id": 1}],
        }
    ]

    cutlist = generate_cuts_from_adjudicated_spans(
        adjudicated_spans=spans,
        units=sample_units,
        source="test.mp4",
        source_duration=20.0,
    )

    assert len(cutlist.retake_groups) == 0
    assert len(cutlist.discard_spans) == 0
    assert len(cutlist.keep_spans) == 1
    assert cutlist.keep_spans[0].start == 0.0
    assert cutlist.keep_spans[0].end == 20.0
    assert cutlist.output_duration == 20.0


def test_multi_unit_discard_group(sample_units):
    extra_units = sample_units + [
        {"unit_id": "1:1", "start": 3.1, "end": 3.8, "raw_text": "Fragment."}
    ]
    spans = [
        {
            "span_id": "span_multi",
            "unit_ids": ["1:0", "1:1", "2:0"],
            "start": 1.0,
            "end": 6.5,
            "is_genuine_retake_group": True,
            "keeper_unit_id": "2:0",
            "units_to_remove": ["1:0", "1:1"],
            "adjudication_status": "resolved",
            "reasoning": "Multi-take with short sibling",
            "candidate_pairs": [{"candidate_id": 1}],
        }
    ]

    cutlist = generate_cuts_from_adjudicated_spans(
        adjudicated_spans=spans,
        units=extra_units,
        source="test.mp4",
        source_duration=20.0,
    )

    assert len(cutlist.retake_groups) == 1
    group = cutlist.retake_groups[0]
    assert group.discarded_text == "First bad take. Fragment."
    assert group.kept_text == "First good take."


def test_save_and_load_cutlist(sample_adjudicated_spans, sample_units, tmp_path):
    cutlist = generate_cuts_from_adjudicated_spans(
        adjudicated_spans=sample_adjudicated_spans,
        units=sample_units,
        source="test.mp4",
        source_duration=20.0,
    )

    out_file = tmp_path / "cutlist_test.json"
    save_cutlist(cutlist, output_path=out_file)
    assert out_file.exists()

    loaded, data = load_cutlist(out_file)
    assert loaded.source == cutlist.source
    assert loaded.source_duration == cutlist.source_duration
    assert len(loaded.keep_spans) == len(cutlist.keep_spans)
    assert len(loaded.retake_groups) == len(cutlist.retake_groups)
