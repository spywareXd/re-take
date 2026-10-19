from __future__ import annotations

import json
from pathlib import Path
import pytest

from pipeline.step_06_llm_adjudication import (
    ADJUDICATION_MAX_RETRIES,
    AdjudicationConfig,
    MockLLMClient,
    build_adjudication_prompt,
    load_adjudicated_spans,
    parse_adjudication_response,
    run_stage6,
    save_adjudicated_spans,
)


def _make_mock_span(
    span_id: str = "span_0",
    unit_ids: list[str] | None = None,
    start: float = 0.0,
    end: float = 10.0,
    needs_adjudication: bool = True,
    likely_final_unit_id: str | None = None,
) -> dict:
    uids = unit_ids or ["1:0", "1:1"]
    return {
        "span_id": span_id,
        "unit_ids": uids,
        "start": start,
        "end": end,
        "candidate_pairs": [
            {
                "candidate_id": 1,
                "unit_a": {"unit_id": uids[0], "raw_text": "First attempt at line"},
                "unit_b": {"unit_id": uids[1], "raw_text": "Second attempt at line"},
                "classification": "ambiguous" if needs_adjudication else "likely_retake",
                "stage2_score": 0.75,
                "semantic_score": 0.90,
                "lexical_score": 0.60,
                "high_disagreement_flag": needs_adjudication,
            }
        ],
        "classification_summary": {"likely_retake": 0, "ambiguous": 1} if needs_adjudication else {"likely_retake": 1, "ambiguous": 0},
        "needs_adjudication": needs_adjudication,
        "likely_final_unit_id": likely_final_unit_id,
    }


def test_pass_through_when_adjudication_not_required():
    """
    A span with needs_adjudication = False is NOT sent to the LLM.
    It receives adjudication_status = 'not_required' with keeper_unit_id derived from likely_final_unit_id.
    """
    span = _make_mock_span(
        span_id="span_0",
        unit_ids=["5:0", "5:1", "6:0"],
        start=11.72,
        end=23.33,
        needs_adjudication=False,
        likely_final_unit_id="6:0",
    )

    mock_client = MockLLMClient()
    results = run_stage6([span], llm_client=mock_client)

    # LLM must NEVER be called for confident spans
    assert mock_client.call_count == 0

    assert len(results) == 1
    res = results[0]
    assert res["adjudication_status"] == "not_required"
    assert res["keeper_unit_id"] == "6:0"
    assert res["units_to_remove"] == ["5:0", "5:1"]
    assert res["is_genuine_retake_group"] is True


def test_valid_llm_response_parsing_and_resolution():
    """
    A well-formed mocked LLM response is parsed, validated, and merged into the span.
    """
    span = _make_mock_span(
        span_id="span_1",
        unit_ids=["2:1", "5:0"],
        start=8.73,
        end=21.46,
        needs_adjudication=True,
    )

    valid_response = {
        "is_genuine_retake_group": True,
        "keeper_unit_id": "5:0",
        "units_to_remove": ["2:1"],
        "confidence": "high",
        "reasoning": "Both units express the same intended intro line; unit 5:0 is the complete final take.",
    }

    mock_client = MockLLMClient(responses=[valid_response])
    results = run_stage6([span], llm_client=mock_client)

    assert mock_client.call_count == 1
    assert len(results) == 1
    res = results[0]
    assert res["adjudication_status"] == "resolved"
    assert res["is_genuine_retake_group"] is True
    assert res["keeper_unit_id"] == "5:0"
    assert res["units_to_remove"] == ["2:1"]
    assert res["confidence"] == "high"
    assert "Both units express" in res["reasoning"]
    assert res["llm_raw_response"] == valid_response


def test_hallucinated_keeper_unit_id_triggers_retry_and_fails_on_exhaustion():
    """
    If the LLM hallucinates a unit_id not in the span, parse_adjudication_response raises
    an error, triggers retries, and marks adjudication_status = 'failed' when exhausted.
    """
    span = _make_mock_span(
        span_id="span_1",
        unit_ids=["2:1", "5:0"],
        needs_adjudication=True,
    )

    hallucinated_response = {
        "is_genuine_retake_group": True,
        "keeper_unit_id": "99:99",  # Hallucinated!
        "units_to_remove": ["2:1"],
        "confidence": "high",
        "reasoning": "Hallucinated unit id.",
    }

    # 3 identical invalid responses to exhaust max_retries=2 (total 3 attempts)
    mock_client = MockLLMClient(responses=[hallucinated_response, hallucinated_response, hallucinated_response])
    config = AdjudicationConfig(max_retries=2)

    results = run_stage6([span], config=config, llm_client=mock_client)

    assert mock_client.call_count == 3
    assert len(results) == 1
    res = results[0]
    assert res["adjudication_status"] == "failed"
    assert res["keeper_unit_id"] is None
    assert res["units_to_remove"] is None
    assert "Hallucinated keeper_unit_id" in res["adjudication_error"]


def test_self_contradictory_response_rejected():
    """
    Contradictions like is_genuine_retake_group: false with a non-null keeper_unit_id
    or keeper_unit_id listed inside units_to_remove must be rejected.
    """
    span = _make_mock_span(
        span_id="span_2",
        unit_ids=["1:0", "1:1"],
        needs_adjudication=True,
    )

    # Contradiction 1: False retake group but keeper provided
    contradiction_1 = {
        "is_genuine_retake_group": False,
        "keeper_unit_id": "1:1",
        "units_to_remove": [],
        "confidence": "high",
        "reasoning": "Contradictory output.",
    }

    with pytest.raises(ValueError, match="Self-contradictory"):
        parse_adjudication_response(contradiction_1, span=span)

    # Contradiction 2: Keeper also listed in units_to_remove
    contradiction_2 = {
        "is_genuine_retake_group": True,
        "keeper_unit_id": "1:1",
        "units_to_remove": ["1:0", "1:1"],
        "confidence": "high",
        "reasoning": "Keeper in removal list.",
    }

    with pytest.raises(ValueError, match="Self-contradictory"):
        parse_adjudication_response(contradiction_2, span=span)


def test_valid_not_a_retake_response():
    """
    When the LLM decides a span is NOT a genuine retake group (e.g. coincidental topical repetition),
    is_genuine_retake_group is False, keeper_unit_id is None, and units_to_remove is [].
    """
    span = _make_mock_span(
        span_id="span_3",
        unit_ids=["3:0", "7:0"],
        needs_adjudication=True,
    )

    not_retake_resp = {
        "is_genuine_retake_group": False,
        "keeper_unit_id": None,
        "units_to_remove": [],
        "confidence": "high",
        "reasoning": "These are two separate topical statements made at different points in the video, not a restart.",
    }

    mock_client = MockLLMClient(responses=[not_retake_resp])
    results = run_stage6([span], llm_client=mock_client)

    assert len(results) == 1
    res = results[0]
    assert res["adjudication_status"] == "resolved"
    assert res["is_genuine_retake_group"] is False
    assert res["keeper_unit_id"] is None
    assert res["units_to_remove"] == []


def test_retry_exhaustion_never_guesses_fallback():
    """
    On persistent API network failures, Step 06 never silently guesses a keeper
    or assumes keep/cut defaults. It leaves keeper_unit_id=None and marks failed.
    """
    span = _make_mock_span(
        span_id="span_fail",
        unit_ids=["1:0", "1:1"],
        needs_adjudication=True,
    )

    # Simulate 3 network errors
    mock_client = MockLLMClient(responses=[
        RuntimeError("API Connection timeout"),
        RuntimeError("API Connection timeout"),
        RuntimeError("API Connection timeout"),
    ])

    config = AdjudicationConfig(max_retries=2)
    results = run_stage6([span], config=config, llm_client=mock_client)

    assert mock_client.call_count == 3
    assert len(results) == 1
    res = results[0]
    assert res["adjudication_status"] == "failed"
    assert res["keeper_unit_id"] is None
    assert res["units_to_remove"] is None
    assert "Exhausted 3 attempts" in res["adjudication_error"]


def test_storage_save_and_load_round_trip(tmp_path):
    output_file = tmp_path / "test_adjudicated_spans.json"

    spans = [
        {
            "span_id": "span_0",
            "unit_ids": ["5:0", "5:1", "6:0"],
            "start": 11.72,
            "end": 23.33,
            "needs_adjudication": False,
            "adjudication_status": "not_required",
            "likely_final_unit_id": "6:0",
            "keeper_unit_id": "6:0",
            "units_to_remove": ["5:0", "5:1"],
            "is_genuine_retake_group": True,
        },
        {
            "span_id": "span_1",
            "unit_ids": ["2:1", "5:0"],
            "start": 8.73,
            "end": 21.46,
            "needs_adjudication": True,
            "adjudication_status": "resolved",
            "is_genuine_retake_group": True,
            "keeper_unit_id": "5:0",
            "units_to_remove": ["2:1"],
            "confidence": "high",
            "reasoning": "Paraphrased retake; keeper 5:0 selected.",
        },
    ]

    saved_path = save_adjudicated_spans(spans, output_path=output_file, model="mock-model")
    assert saved_path == output_file
    assert output_file.exists()

    loaded_spans, metadata = load_adjudicated_spans(output_file)
    assert len(loaded_spans) == 2
    assert metadata["model"] == "mock-model"
    assert metadata["span_count"] == 2
    assert metadata["resolved_count"] == 1
    assert metadata["not_required_count"] == 1
    assert metadata["failed_count"] == 0
    assert loaded_spans[0]["keeper_unit_id"] == "6:0"
    assert loaded_spans[1]["keeper_unit_id"] == "5:0"
