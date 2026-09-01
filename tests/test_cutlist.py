"""Tests for Step 05: Cutlist Generation."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from pipeline.step_05_cutlist import (
    CutList,
    RetakeGroup,
    TimeSpan,
    build_cutlist,
    build_retake_groups,
    compute_keep_spans,
    filter_confirmed_retakes,
    group_retake_pairs,
    merge_spans,
    save_cutlist,
    load_cutlist,
    select_take,
)


# ---------------------------------------------------------------------------
# Fixtures: Realistic semantic candidate data (mirrors test.mp4 output)
# ---------------------------------------------------------------------------


def _make_candidate(cid, ua_id, ua_start, ua_end, ub_id, ub_start, ub_end,
                    classification="likely_retake", score=0.95, raw_text_a="", raw_text_b=""):
    return {
        "candidate_id": cid,
        "unit_a": {
            "unit_id": ua_id,
            "segment_id": int(ua_id.split(":")[0]),
            "start": ua_start,
            "end": ua_end,
            "raw_text": raw_text_a or f"Text for {ua_id}",
            "speaker": "speaker_0",
            "confidence": 0.85,
        },
        "unit_b": {
            "unit_id": ub_id,
            "segment_id": int(ub_id.split(":")[0]),
            "start": ub_start,
            "end": ub_end,
            "raw_text": raw_text_b or f"Text for {ub_id}",
            "speaker": "speaker_0",
            "confidence": 0.83,
        },
        "combined_score": score,
        "stage2_score": score,
        "classification": classification,
        "semantic_score": score,
    }


@pytest.fixture
def sample_candidates():
    """Mimics the test.mp4 output: consecutive retake pairs."""
    return [
        _make_candidate(3, "2:0", 3.28, 4.26, "9:0", 33.27, 34.29),
        _make_candidate(4, "3:0", 5.30, 9.10, "10:0", 35.28, 39.09),
        _make_candidate(5, "4:0", 9.68, 15.98, "11:0", 39.67, 45.97),
        _make_candidate(6, "5:0", 16.66, 18.46, "12:0", 46.66, 48.46),
        _make_candidate(7, "6:0", 18.78, 21.88, "13:0", 48.78, 51.88),
        _make_candidate(8, "7:0", 22.34, 24.86, "14:0", 52.34, 54.86),
        _make_candidate(9, "7:1", 25.60, 27.90, "14:1", 55.60, 57.91),
        _make_candidate(10, "7:2", 28.64, 29.84, "14:2", 58.65, 59.85),
        # Ambiguous candidates — should be filtered out
        _make_candidate(11, "8:0", 30.46, 32.63, "15:0", 60.47, 62.23,
                        classification="ambiguous", score=0.82),
        _make_candidate(1, "1:0", 0.49, 2.64, "8:0", 30.46, 32.63,
                        classification="ambiguous", score=0.80),
        _make_candidate(2, "1:0", 0.49, 2.64, "15:0", 60.47, 62.23,
                        classification="ambiguous", score=0.71),
    ]


# ---------------------------------------------------------------------------
# TimeSpan tests
# ---------------------------------------------------------------------------


class TestTimeSpan:
    def test_duration(self):
        span = TimeSpan(start=1.0, end=3.5)
        assert span.duration == 2.5

    def test_duration_zero(self):
        span = TimeSpan(start=5.0, end=5.0)
        assert span.duration == 0.0

    def test_overlaps_true(self):
        a = TimeSpan(start=0.0, end=5.0)
        b = TimeSpan(start=3.0, end=8.0)
        assert a.overlaps(b)

    def test_overlaps_false(self):
        a = TimeSpan(start=0.0, end=3.0)
        b = TimeSpan(start=5.0, end=8.0)
        assert not a.overlaps(b)

    def test_overlaps_with_tolerance(self):
        a = TimeSpan(start=0.0, end=3.0)
        b = TimeSpan(start=3.2, end=8.0)
        assert not a.overlaps(b, tolerance=0.0)
        assert a.overlaps(b, tolerance=0.3)

    def test_merge(self):
        a = TimeSpan(start=0.0, end=5.0)
        b = TimeSpan(start=3.0, end=8.0)
        merged = a.merge(b)
        assert merged.start == 0.0
        assert merged.end == 8.0

    def test_serialization_roundtrip(self):
        span = TimeSpan(start=1.5, end=3.7, label="keep")
        restored = TimeSpan.from_dict(span.to_dict())
        assert restored.start == span.start
        assert restored.end == span.end
        assert restored.label == span.label


# ---------------------------------------------------------------------------
# filter_confirmed_retakes
# ---------------------------------------------------------------------------


class TestFilterConfirmedRetakes:
    def test_filters_to_likely_retakes(self, sample_candidates):
        confirmed = filter_confirmed_retakes(sample_candidates)
        assert len(confirmed) == 8  # 8 likely_retake, 3 ambiguous filtered out
        assert all(c["classification"] == "likely_retake" for c in confirmed)

    def test_custom_classifications(self, sample_candidates):
        result = filter_confirmed_retakes(sample_candidates, classifications=("ambiguous",))
        assert len(result) == 3

    def test_empty_input(self):
        assert filter_confirmed_retakes([]) == []


# ---------------------------------------------------------------------------
# group_retake_pairs
# ---------------------------------------------------------------------------


class TestGroupRetakePairs:
    def test_consecutive_pairs_form_one_group(self, sample_candidates):
        confirmed = filter_confirmed_retakes(sample_candidates)
        groups = group_retake_pairs(confirmed)
        # All 8 confirmed pairs are consecutive (gap < 5s) → should be 1 group
        assert len(groups) == 1
        assert len(groups[0]) == 8

    def test_separate_groups_with_large_gap(self):
        cands = [
            _make_candidate(1, "1:0", 0.0, 2.0, "5:0", 20.0, 22.0),
            _make_candidate(2, "2:0", 3.0, 5.0, "6:0", 23.0, 25.0),
            # Big gap on both sides
            _make_candidate(3, "10:0", 50.0, 52.0, "20:0", 100.0, 102.0),
        ]
        groups = group_retake_pairs(cands)
        assert len(groups) == 2

    def test_empty_input(self):
        assert group_retake_pairs([]) == []


# ---------------------------------------------------------------------------
# select_take
# ---------------------------------------------------------------------------


class TestSelectTake:
    def test_keep_later_default(self, sample_candidates):
        confirmed = filter_confirmed_retakes(sample_candidates)
        kept, reason = select_take(confirmed)
        assert kept == "b"

    def test_keep_earlier(self, sample_candidates):
        confirmed = filter_confirmed_retakes(sample_candidates)
        kept, _ = select_take(confirmed, strategy="keep_earlier")
        assert kept == "a"

    def test_keep_higher_confidence(self):
        cands = [
            _make_candidate(1, "1:0", 0.0, 2.0, "5:0", 20.0, 22.0),
        ]
        # unit_a confidence = 0.85, unit_b confidence = 0.83
        kept, reason = select_take(cands, strategy="keep_higher_confidence")
        assert kept == "a"
        assert "Higher ASR confidence" in reason


# ---------------------------------------------------------------------------
# merge_spans
# ---------------------------------------------------------------------------


class TestMergeSpans:
    def test_merge_overlapping(self):
        spans = [
            TimeSpan(start=0.0, end=5.0),
            TimeSpan(start=3.0, end=8.0),
        ]
        merged = merge_spans(spans)
        assert len(merged) == 1
        assert merged[0].start == 0.0
        assert merged[0].end == 8.0

    def test_no_overlap(self):
        spans = [
            TimeSpan(start=0.0, end=3.0),
            TimeSpan(start=10.0, end=15.0),
        ]
        merged = merge_spans(spans)
        assert len(merged) == 2

    def test_merge_with_tolerance(self):
        spans = [
            TimeSpan(start=0.0, end=3.0),
            TimeSpan(start=3.2, end=5.0),
        ]
        merged = merge_spans(spans, tolerance=0.3)
        assert len(merged) == 1

    def test_empty_input(self):
        assert merge_spans([]) == []


# ---------------------------------------------------------------------------
# compute_keep_spans
# ---------------------------------------------------------------------------


class TestComputeKeepSpans:
    def test_no_discards_keeps_everything(self):
        keeps = compute_keep_spans([], total_duration=60.0)
        assert len(keeps) == 1
        assert keeps[0].start == 0.0
        assert keeps[0].end == 60.0

    def test_single_discard_in_middle(self):
        discards = [TimeSpan(start=10.0, end=20.0)]
        keeps = compute_keep_spans(discards, total_duration=60.0, padding=0.0)
        assert len(keeps) == 2
        assert keeps[0].end == 10.0
        assert keeps[1].start == 20.0
        assert keeps[1].end == 60.0

    def test_discard_at_start(self):
        discards = [TimeSpan(start=0.0, end=10.0)]
        keeps = compute_keep_spans(discards, total_duration=60.0, padding=0.0)
        assert len(keeps) == 1
        assert keeps[0].start == 10.0

    def test_filters_tiny_fragments(self):
        discards = [
            TimeSpan(start=0.0, end=59.95),
        ]
        keeps = compute_keep_spans(discards, total_duration=60.0, padding=0.0)
        # Remaining fragment is only 0.05s — below 0.1s threshold
        assert len(keeps) == 0


# ---------------------------------------------------------------------------
# build_cutlist (end-to-end)
# ---------------------------------------------------------------------------


class TestBuildCutlist:
    def test_full_pipeline(self, sample_candidates):
        cutlist = build_cutlist(
            candidates=sample_candidates,
            source="test.mp4",
            source_duration=62.23,
        )

        assert isinstance(cutlist, CutList)
        assert cutlist.source == "test.mp4"
        assert cutlist.source_duration == 62.23

        # Should have at least 1 retake group
        assert len(cutlist.retake_groups) >= 1

        # Should have discard spans (the first take block)
        assert len(cutlist.discard_spans) >= 1

        # Should have keep spans
        assert len(cutlist.keep_spans) >= 1

        # Output duration should be less than source
        assert cutlist.output_duration < cutlist.source_duration

        # No overlaps in keep_spans
        for i in range(len(cutlist.keep_spans) - 1):
            assert cutlist.keep_spans[i].end <= cutlist.keep_spans[i + 1].start

    def test_no_retakes_keeps_everything(self):
        cands = [_make_candidate(1, "1:0", 0.0, 2.0, "5:0", 20.0, 22.0,
                                 classification="unlikely_retake")]
        cutlist = build_cutlist(cands, source="test.mp4", source_duration=30.0)
        assert len(cutlist.discard_spans) == 0
        assert len(cutlist.keep_spans) == 1
        assert cutlist.output_duration == 30.0

    def test_serialization_roundtrip(self, sample_candidates):
        cutlist = build_cutlist(
            candidates=sample_candidates,
            source="test.mp4",
            source_duration=62.23,
        )
        d = cutlist.to_dict()
        restored = CutList.from_dict(d)
        assert restored.source == cutlist.source
        assert len(restored.keep_spans) == len(cutlist.keep_spans)
        assert len(restored.discard_spans) == len(cutlist.discard_spans)


# ---------------------------------------------------------------------------
# Storage tests
# ---------------------------------------------------------------------------


class TestCutlistStorage:
    def test_save_and_load(self, sample_candidates, tmp_path):
        cutlist = build_cutlist(
            candidates=sample_candidates,
            source="test.mp4",
            source_duration=62.23,
        )

        save_path = tmp_path / "cutlist.json"
        save_cutlist(cutlist, output_path=save_path)
        assert save_path.exists()

        loaded, metadata = load_cutlist(save_path)
        assert loaded.source == cutlist.source
        assert len(loaded.keep_spans) == len(cutlist.keep_spans)
        assert "generated_at" in metadata

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_cutlist(tmp_path / "nonexistent.json")
