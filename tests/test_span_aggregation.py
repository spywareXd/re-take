from __future__ import annotations

import json
from pathlib import Path
import pytest

from pipeline.step_05_span_aggregation import (
    RetakeSpan,
    build_retake_spans,
    load_retake_spans,
    save_retake_spans,
)


def _make_candidate(
    candidate_id: int,
    uid_a: str,
    start_a: float,
    end_a: float,
    text_a: str,
    uid_b: str,
    start_b: float,
    end_b: float,
    text_b: str,
    classification: str = "likely_retake",
    containment_direction: str | None = None,
    stage2_score: float = 0.90,
    decision: str = "retake",
    resolved_by: str = "stacker",
    stacker_p: float = 0.95,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "unit_a": {
            "unit_id": uid_a,
            "segment_id": int(uid_a.split(":")[0]) if ":" in uid_a else 1,
            "start": start_a,
            "end": end_a,
            "raw_text": text_a,
        },
        "unit_b": {
            "unit_id": uid_b,
            "segment_id": int(uid_b.split(":")[0]) if ":" in uid_b else 1,
            "start": start_b,
            "end": end_b,
            "raw_text": text_b,
        },
        "scores": {"containment": 1.0 if containment_direction else 0.0},
        "combined_score": 0.85,
        "semantic_score": 0.90,
        "legacy_stage2_score": stage2_score,
        "containment_direction": containment_direction,
        "legacy_class": classification,
        "class": decision,                       # final decision from Stage 4: "retake" | "not_retake"
        "resolved_by": resolved_by,              # "stacker" | "cross_encoder"
        "stacker_p": stacker_p,
    }


def test_transitive_clustering():
    """
    Construct a synthetic 3-pair set where A<->B and B<->C are candidates
    but A<->C is NOT a candidate pair. All 3 must end up in one span.
    """
    pair1 = _make_candidate(1, "1:0", 1.0, 3.0, "Take A", "1:1", 4.0, 6.0, "Take B")
    pair2 = _make_candidate(2, "1:1", 4.0, 6.0, "Take B", "2:0", 7.0, 9.0, "Take C")

    spans = build_retake_spans([pair1, pair2])

    assert len(spans) == 1
    span = spans[0]
    assert span.unit_ids == ["1:0", "1:1", "2:0"]
    assert span.start == 1.0
    assert span.end == 9.0
    assert len(span.candidate_pairs) == 2
    assert span.needs_adjudication is False


def test_real_data_teatime_clustering():
    """
    Real data regression from tea time transcript:
    1:0 ("Now we're going to learn about...")
    1:1 ("Now we're going to learn about tea time.")
    2:1 ("Now we're going to learn about tea time, guys.")
    All 3 are connected via pairwise candidate pairs and must land in the same span.
    """
    pairs = [
        _make_candidate(1, "1:0", 0.82, 2.80, "Now we're going to learn about...", "1:1", 3.97, 5.95, "Now we're going to learn about tea time.", containment_direction="a_subset_of_b"),
        _make_candidate(2, "1:1", 3.97, 5.95, "Now we're going to learn about tea time.", "2:1", 8.73, 10.81, "Now we're going to learn about tea time, guys.", containment_direction="a_subset_of_b"),
        _make_candidate(3, "1:0", 0.82, 2.80, "Now we're going to learn about...", "2:1", 8.73, 10.81, "Now we're going to learn about tea time, guys.", containment_direction="a_subset_of_b"),
    ]

    spans = build_retake_spans(pairs)

    assert len(spans) == 1
    span = spans[0]
    assert span.unit_ids == ["1:0", "1:1", "2:1"]
    assert span.start == 0.82
    assert span.end == 10.81
    assert span.needs_adjudication is False
    # 2:1 is the latest unit and not marked as subset of any take
    assert span.likely_final_unit_id == "2:1"


def test_unlikely_retake_pairs_excluded_from_clustering():
    """
    An unlikely_retake pair must NOT merge two otherwise-unrelated units into a span.
    """
    pair_valid = _make_candidate(1, "1:0", 1.0, 3.0, "Take 1", "1:1", 4.0, 6.0, "Take 2", classification="likely_retake")
    pair_unlikely = _make_candidate(2, "1:1", 4.0, 6.0, "Take 2", "9:0", 50.0, 55.0, "Completely unrelated sentence", classification="unlikely_retake", decision="not_retake", resolved_by="stacker", stacker_p=0.01)

    spans = build_retake_spans([pair_valid, pair_unlikely])

    assert len(spans) == 1
    span = spans[0]
    assert span.unit_ids == ["1:0", "1:1"]
    assert "9:0" not in span.unit_ids


def test_spans_never_need_adjudication_and_record_who_decided():
    """
    Every span is already a confirmed retake: needs_adjudication is False and needs_keeper True, even when an edge was ambiguous before the
    cross-encoder resolved it. The span records its unit count, its weakest edge and where its decisions came from.
    """
    pair_stacker = _make_candidate(1, "1:0", 1.0, 3.0, "Take 1", "1:1", 4.0, 6.0, "Take 2", stacker_p=0.95)
    pair_ce = _make_candidate(2, "1:1", 4.0, 6.0, "Take 2", "2:0", 7.0, 9.0, "Paraphrase Take 3", classification="ambiguous",
                              resolved_by="cross_encoder", stacker_p=0.50)

    spans = build_retake_spans([pair_stacker, pair_ce])

    assert len(spans) == 1
    span = spans[0]
    assert span.unit_ids == ["1:0", "1:1", "2:0"]
    assert span.needs_adjudication is False and span.needs_keeper is True
    assert span.n_units == 3 and span.min_stacker_p == 0.50 and span.decision_source == "mixed"
    assert span.classification_summary == {"likely_retake": 2, "ambiguous": 0, "resolved_by_stacker": 1, "resolved_by_cross_encoder": 1}


def test_span_with_more_than_four_units_logs_a_warning(caplog):
    """A chain of 5 units is built (union-find unchanged) and a warning is logged."""
    pairs = [_make_candidate(i, f"{i}:0", float(i), i + 0.5, f"T{i}", f"{i + 1}:0", i + 1.0, i + 1.5, f"T{i + 1}") for i in range(1, 5)]
    with caplog.at_level("WARNING", logger="pipeline.step_05_span_aggregation.span_aggregation"):
        spans = build_retake_spans(pairs)
    assert len(spans) == 1 and spans[0].n_units == 5
    assert any("5 units" in r.message for r in caplog.records)


def test_pairs_decided_not_retake_are_never_unioned():
    """A pair the scorer decided 'not_retake' must not bridge two real retake pairs into one span."""
    a = _make_candidate(1, "1:0", 1.0, 3.0, "A1", "1:1", 4.0, 6.0, "A2")
    b = _make_candidate(2, "5:0", 20.0, 22.0, "B1", "5:1", 23.0, 25.0, "B2")
    bridge = _make_candidate(3, "1:1", 4.0, 6.0, "A2", "5:0", 20.0, 22.0, "B1", decision="not_retake", stacker_p=0.02)
    spans = build_retake_spans([a, b, bridge])
    assert len(spans) == 2


def test_isolated_units_excluded_from_output():
    """
    If all pairs for a unit are unlikely_retake (or if candidate list is empty),
    no spans are generated and isolated units do not appear.
    """
    pair_unlikely = _make_candidate(1, "1:0", 1.0, 3.0, "Text A", "2:0", 10.0, 12.0, "Text B", classification="unlikely_retake", decision="not_retake", stacker_p=0.01)
    spans = build_retake_spans([pair_unlikely])
    assert spans == []


def test_likely_final_unit_id_heuristic_applies_to_cross_encoder_confirmed_spans():
    """
    A span whose only edge was ambiguous and was confirmed by the cross-encoder is a confirmed retake like any other: the (unchanged)
    likely-final heuristic applies and picks the chronologically latest non-subset unit; no adjudication is needed.
    """
    pair = _make_candidate(1, "2:1", 8.73, 10.81, "Now we're going to learn about tea time, guys.", "5:0", 17.71, 21.46, "Hey guys, let's talk about tea time.",
                           classification="ambiguous", resolved_by="cross_encoder", stacker_p=0.5)

    spans = build_retake_spans([pair])

    assert len(spans) == 1
    span = spans[0]
    assert span.needs_adjudication is False and span.needs_keeper is True
    assert span.decision_source == "cross_encoder"
    assert span.likely_final_unit_id == "5:0"


def test_multiple_independent_spans_sorted_chronologically():
    """
    Multiple disjoint retake clusters form separate spans sorted by start time.
    """
    # Cluster 1 (Intro retake)
    c1 = _make_candidate(1, "1:0", 1.0, 3.0, "Intro 1", "1:1", 4.0, 6.0, "Intro 2", classification="likely_retake")
    # Cluster 2 (Outro retake)
    c2 = _make_candidate(2, "8:0", 25.0, 27.0, "Outro 1", "8:1", 28.0, 30.0, "Outro 2", classification="likely_retake")

    spans = build_retake_spans([c2, c1])  # passed in reverse order

    assert len(spans) == 2
    assert spans[0].span_id == "span_0"
    assert spans[0].unit_ids == ["1:0", "1:1"]
    assert spans[0].start == 1.0

    assert spans[1].span_id == "span_1"
    assert spans[1].unit_ids == ["8:0", "8:1"]
    assert spans[1].start == 25.0


def test_storage_save_and_load_round_trip(tmp_path):
    output_file = tmp_path / "test_retake_spans.json"

    p1 = _make_candidate(1, "1:0", 0.82, 2.80, "Now we're going to learn about...", "1:1", 3.97, 5.95, "Now we're going to learn about tea time.")
    spans = build_retake_spans([p1])

    saved_path = save_retake_spans(spans, output_path=output_file, source_candidates="source.json")
    assert saved_path == output_file
    assert output_file.exists()

    loaded_spans, metadata = load_retake_spans(output_file)

    assert len(loaded_spans) == 1
    s = loaded_spans[0]
    assert s.span_id == "span_0"
    assert s.unit_ids == ["1:0", "1:1"]
    assert s.start == 0.82
    assert s.end == 5.95
    assert s.needs_adjudication is False
    assert metadata["source_candidates"] == "source.json"
    assert metadata["span_count"] == 1


def test_sibling_short_unit_attached_to_span():
    """
    Test that a short unit (is_below_min_token_length: true) sharing a source_segment_id
    with a unit in a span (e.g. 5:0 and 5:1 in segment 5) gets attached to that span as a silent member.
    """
    # Candidate pair only involves 5:1 and 6:0
    pair = _make_candidate(
        1,
        "5:1", 12.82, 15.62, "you wonder why green tea works for some people, but not.",
        "6:0", 16.50, 23.33, "Friends, you wonder why green tea works for some people, but you're not getting good results.",
        containment_direction="a_subset_of_b",
        classification="likely_retake",
    )

    all_units = [
        {
            "unit_id": "5:0",
            "source_segment_id": 5,
            "start": 11.72,
            "end": 12.08,
            "raw_text": "Fellas,",
            "is_below_min_token_length": True,
        },
        {
            "unit_id": "5:1",
            "source_segment_id": 5,
            "start": 12.82,
            "end": 15.62,
            "raw_text": "you wonder why green tea works for some people, but not.",
            "is_below_min_token_length": False,
        },
        {
            "unit_id": "6:0",
            "source_segment_id": 6,
            "start": 16.50,
            "end": 23.33,
            "raw_text": "Friends, you wonder why green tea works for some people, but you're not getting good results.",
            "is_below_min_token_length": False,
        },
    ]

    spans = build_retake_spans([pair], all_units=all_units)

    assert len(spans) == 1
    span = spans[0]
    # 5:0 is attached as a silent member
    assert span.unit_ids == ["5:0", "5:1", "6:0"]
    # Span start expands to encompass 5:0 (11.72s)
    assert span.start == 11.72
    assert span.end == 23.33
    # Keeper take remains 6:0 (5:0 is not selected as keeper)
    assert span.likely_final_unit_id == "6:0"
    # Adjudication and classification summary are unaffected
    assert span.needs_adjudication is False
    assert span.classification_summary == {"likely_retake": 1, "ambiguous": 0, "resolved_by_stacker": 1, "resolved_by_cross_encoder": 0}


def test_unrelated_short_unit_not_attached_when_sibling_not_in_span():
    """
    Test mirror case: A short unit (is_below_min_token_length: true) whose sibling is NOT
    part of any span must NOT be attached to any span. It is silently omitted from retake_spans.json
    and preserved in the clean output video.
    """
    # Only segment 5/6 has a retake pair
    pair = _make_candidate(
        1,
        "5:1", 12.82, 15.62, "you wonder why green tea works for some people, but not.",
        "6:0", 16.50, 23.33, "Friends, you wonder why green tea works for some people, but you're not getting good results.",
        containment_direction="a_subset_of_b",
        classification="likely_retake",
    )

    all_units = [
        # Segment 2: Valid dialogue split by a natural pause, NOT part of any retake
        {
            "unit_id": "2:0",
            "source_segment_id": 2,
            "start": 2.10,
            "end": 2.45,
            "raw_text": "So,",
            "is_below_min_token_length": True,
        },
        {
            "unit_id": "2:1",
            "source_segment_id": 2,
            "start": 3.20,
            "end": 5.50,
            "raw_text": "let's dive right into today's main topic.",
            "is_below_min_token_length": False,
        },
        # Segment 5/6: Retake
        {
            "unit_id": "5:0",
            "source_segment_id": 5,
            "start": 11.72,
            "end": 12.08,
            "raw_text": "Fellas,",
            "is_below_min_token_length": True,
        },
        {
            "unit_id": "5:1",
            "source_segment_id": 5,
            "start": 12.82,
            "end": 15.62,
            "raw_text": "you wonder why green tea works for some people, but not.",
            "is_below_min_token_length": False,
        },
        {
            "unit_id": "6:0",
            "source_segment_id": 6,
            "start": 16.50,
            "end": 23.33,
            "raw_text": "Friends, you wonder why green tea works for some people, but you're not getting good results.",
            "is_below_min_token_length": False,
        },
    ]

    spans = build_retake_spans([pair], all_units=all_units)

    assert len(spans) == 1
    span = spans[0]

    # Only 5:0 is attached because 5:1 is in the span
    assert span.unit_ids == ["5:0", "5:1", "6:0"]
    # 2:0 and 2:1 are completely absent from all spans
    assert "2:0" not in span.unit_ids
    assert "2:1" not in span.unit_ids
    # Span bounds do not encompass segment 2
    assert span.start == 11.72
    assert span.end == 23.33


