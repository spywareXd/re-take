"""Integration tests for the final scorer inside the pipeline (Stage 4 -> 5 -> 6 -> cutlist). No GPU, no API calls: the scorer and the LLM are mocks."""
from __future__ import annotations

import numpy as np
import pytest

from models.final.final_scorer import FinalScorer
from pipeline.step_04_semantic_matching import final_stage4
from pipeline.step_05_span_aggregation import build_retake_spans
from pipeline.step_06_llm_adjudication import AdjudicationConfig, MockLLMClient, run_stage6
from pipeline.step_08_video_edit import generate_cuts_from_adjudicated_spans


def _candidate(cid, ua, sa, ea, ta, ub, sb, eb, tb, lexical, semantic, containment, legacy_stage2):
    """a candidate as produced by step 03 + the old run_stage2 (legacy recency-adjusted score and class)"""
    return {"candidate_id": cid,
            "unit_a": {"unit_id": ua, "segment_id": int(ua.split(":")[0]), "start": sa, "end": ea, "raw_text": ta},
            "unit_b": {"unit_id": ub, "segment_id": int(ub.split(":")[0]), "start": sb, "end": eb, "raw_text": tb},
            "scores": {"containment": containment}, "lexical_score": lexical, "combined_score": lexical * 0.2, "recency_weight": 0.2,
            "semantic_score": semantic, "stage2_score": legacy_stage2, "classification": "ambiguous", "containment_direction": None,
            "lexical_semantic_disagreement": abs(lexical - semantic), "high_disagreement_flag": False}


@pytest.fixture
def mock_scorer():
    s = FinalScorer(mock=True)
    assert s.mode == "stacker" and s.t_amb is not None
    return s


def test_stage4_uses_the_plain_formula_not_the_recency_adjusted_pipeline_score(monkeypatch, mock_scorer):
    """FAILS if Stage 4 feeds the pipeline's own (recency-adjusted) stage2_score to the stacker instead of 0.5*semantic + 0.3*lexical + 0.2*containment."""
    lexical, semantic, containment = 0.80, 0.90, 0.70
    plain = 0.50 * semantic + 0.30 * lexical + 0.20 * containment            # 0.83
    legacy = 0.30                                                               # what the old pipeline would have reported (recency-decayed)
    cand = _candidate(1, "1:0", 1.0, 2.0, "Take one of the line", "2:0", 50.0, 51.0, "Take two of the line", lexical, semantic, containment, legacy)
    monkeypatch.setattr(final_stage4, "run_stage2", lambda cands: [dict(c) for c in cands])    # candidates already carry the semantic fields
    out = final_stage4.run_stage4([cand], scorer=mock_scorer)[0]

    assert out["plain_stage2_score"] == pytest.approx(plain)
    assert out["legacy_stage2_score"] == legacy and out["legacy_class"] == "ambiguous"       # kept for audit only
    assert "stage2_score" not in out and "classification" not in out                         # old keys are gone so nothing can read them by mistake
    # the stacker probability is computed from the PLAIN score; using the legacy value would give a different probability
    ce = mock_scorer.ce_probs([{"text_a": cand["unit_a"]["raw_text"], "text_b": cand["unit_b"]["raw_text"]}])
    assert out["stacker_p"] == pytest.approx(float(mock_scorer.stacker_prob(np.array([plain]), ce)[0]), abs=1e-5)
    assert abs(float(mock_scorer.stacker_prob(np.array([legacy]), ce)[0]) - out["stacker_p"]) > 0.05
    assert out["class"] in ("retake", "not_retake") and out["resolved_by"] in ("stacker", "cross_encoder")


def test_mock_end_to_end_stage4_to_cutlist(monkeypatch, mock_scorer):
    """Smoke test: Stage 4 (mock scorer) -> Stage 5 -> Stage 6 (mock LLM keeps the last unit) -> cutlist. A confirmed 3-unit retake is cut;
    a not-retake pair is neither unioned nor cut."""
    text = "Welcome to the channel everyone"
    cands = [
        _candidate(1, "1:0", 1.0, 3.0, text, "1:1", 4.0, 6.0, text, 1.0, 1.0, 1.0, 0.4),                    # identical take -> retake
        _candidate(2, "1:1", 4.0, 6.0, text, "2:0", 7.0, 9.0, text, 1.0, 1.0, 1.0, 0.4),                    # identical take -> retake (chains 3 units)
        _candidate(3, "2:0", 7.0, 9.0, text, "5:0", 30.0, 32.0, "Totally different content here", 0.05, 0.1, 0.05, 0.1),   # not a retake
    ]
    monkeypatch.setattr(final_stage4, "run_stage2", lambda c: [dict(x) for x in c])
    pairs = final_stage4.run_stage4(cands, scorer=mock_scorer)
    assert [p["class"] for p in pairs] == ["retake", "retake", "not_retake"]
    assert not any(p["class"] == "ambiguous" for p in pairs)                                                 # no pair is left ambiguous

    spans = build_retake_spans(pairs)
    assert len(spans) == 1 and spans[0].unit_ids == ["1:0", "1:1", "2:0"]                                      # the not-retake pair is not unioned
    assert spans[0].needs_adjudication is False and spans[0].needs_keeper is True

    adjudicated = run_stage6(spans, all_units_by_id=None, config=AdjudicationConfig(provider="mock"), llm_client=MockLLMClient())
    assert adjudicated[0]["adjudication_status"] == "resolved" and adjudicated[0]["keeper_unit_id"] == "2:0"

    units = [{"unit_id": u, "start": s, "end": e, "raw_text": text} for u, s, e in (("1:0", 1.0, 3.0), ("1:1", 4.0, 6.0), ("2:0", 7.0, 9.0), ("5:0", 30.0, 32.0))]
    cutlist = generate_cuts_from_adjudicated_spans(adjudicated, units, source="x.mp4", source_duration=40.0)
    assert [(round(d.start, 1), round(d.end, 1)) for d in cutlist.discard_spans] == [(1.0, 3.0), (4.0, 6.0)]
    assert len(cutlist.retake_groups) == 1
