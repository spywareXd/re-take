"""Backend handoff (schema simplecut.retakes/2): unsure band, constrained clustering, stable ids + cache, builders vs validator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from models.final.final_scorer import FinalScorer
from models.transcript import Transcript
from pipeline.step_05_span_aggregation.constrained_clustering import Edge, cluster_with_constraints
from pipeline.step_07_review import ReviewConfig, analyze_media, build_candidates, build_groups, resolve_request, validate_document
from pipeline.step_07_review import json_export

FIXTURE = Path(__file__).parent / "fixtures" / "sample_c_transcript.json"
MEDIA = {"path": "x.mp4", "duration_seconds": 100.0}


def _units(n):
    return [{"unit_id": f"{i}:0", "source_segment_id": i, "start": 10.0 * i, "end": 10.0 * i + 4.0, "raw_text": f"sentence {i}", "confidence": 0.9} for i in range(n)]


def _transcript(n):
    return {"segments": [{"id": i, "words": [{"word": "w", "start": 10.0 * i + 0.3, "end": 10.0 * i + 3.8}]} for i in range(n)]}


def _score(a, b, ce, p):
    return {"a": f"{a}:0", "b": f"{b}:0", "ce_prob": ce, "stacker_p": p, "plain_stage2": 0.5}


def test_unsure_band_maps_scores_to_likely_dropped_ambiguous():
    s = FinalScorer(mock=True)
    assert s.unsure_band_enabled and s.t_lo < s.t_hi == s.t_amb
    mid_p = (s.low + s.high) / 2                                              # every pair sits in the stacker's ambiguous tier
    mid_ce = (s.t_lo + s.t_hi) / 2
    scores = [_score(0, 1, s.t_hi, mid_p), _score(2, 3, s.t_lo, mid_p), _score(4, 5, mid_ce, mid_p), _score(0, 3, 0.99, s.high + 0.01)]
    doc = build_candidates(_units(6), scores, _transcript(6), s, MEDIA, "j_1")
    cls = {p["id"]: (p["class"], p["resolved_by"]) for p in doc["pairs"]}
    assert cls["0:0|1:0"] == ("likely", "cross_encoder")                      # ce >= t_hi
    assert "2:0|3:0" not in cls                                               # ce <= t_lo: dropped
    assert cls["4:0|5:0"] == ("ambiguous", "unsure")                          # in between: asked
    assert cls["0:0|3:0"] == ("likely", "stacker")
    old = FinalScorer(mock=True, unsure_band_enabled=False)                   # the flag reproduces the old single threshold
    assert old.resolve(np.array([mid_p]), np.array([mid_ce]))[0] == ["not_retake"]
    capped = FinalScorer(mock=True, max_questions=2)                          # cap: closest to the middle of the band stay ambiguous
    many = [_score(i, i + 1, mid_ce + 0.001 * k, mid_p) for k, i in enumerate(range(0, 5))]
    decisions, by = capped.resolve(np.array([mid_p] * 5), np.array([m["ce_prob"] for m in many]))
    assert by.count("unsure") == 2 and decisions.count("ambiguous") == 2 and by.count("cross_encoder") == 3


def test_constrained_clustering_user_labels_are_constraints():
    s = FinalScorer(mock=True)
    mid_p, mid_ce = (s.low + s.high) / 2, (s.t_lo + s.t_hi) / 2
    # A-B and B-C are ambiguous questions, A-C is a confident model edge
    scores = [_score(0, 1, mid_ce, mid_p), _score(1, 2, mid_ce, mid_p), _score(0, 2, 0.999, s.high + 0.05)]
    cand = build_candidates(_units(3), scores, _transcript(3), s, MEDIA, "j_2")
    labels = {"job": "j_2", "labels": [{"pair": "0:0|1:0", "label": "retake", "source": "user"},
                                       {"pair": "1:0|2:0", "label": "different", "source": "user"}], "skipped": []}
    groups = build_groups(cand, labels)["groups"]
    assert len(groups) == 1
    assert [x["id"] for x in groups[0]["sentences"]] == ["0:0", "1:0"]        # A, B together; C is not in the group
    assert groups[0]["likely"] is True                                        # user retake is confident; the A-C edge was rejected, so it does not count
    # the same rule at the clustering level (a different order of the edges gives the same answer)
    res = cluster_with_constraints([Edge("A", "C", 0.99)], [Edge("A", "B", 1.0, "user")], [("B", "C")])
    assert res.components == [["A", "B"]] and len(res.rejected) == 1


def _fake_transcribe(calls):
    def run(path):
        calls.append(path)
        return Transcript.model_validate(json.loads(FIXTURE.read_text(encoding="utf-8")))
    return run


def test_stable_ids_and_cache_reuse(tmp_path, monkeypatch):
    media = tmp_path / "take.mp4"
    media.write_bytes(b"not really a video, only its bytes matter for the fingerprint")
    cfg = ReviewConfig(cache_dir=str(tmp_path / "cache"), labels_path=str(tmp_path / "labels.jsonl"), skipped_path=str(tmp_path / "skipped.jsonl"))
    calls, scored = [], []
    real = json_export._score_pairs
    monkeypatch.setattr(json_export, "_score_pairs", lambda *a, **k: (scored.append(1), real(*a, **k))[1])
    scorer = FinalScorer(mock=True)
    first = analyze_media(media, duration_seconds=25.8, cfg=cfg, scorer=scorer, transcribe_fn=_fake_transcribe(calls))
    second = analyze_media(media, duration_seconds=25.8, cfg=cfg, scorer=scorer, transcribe_fn=_fake_transcribe(calls))
    assert len(calls) == 1 and len(scored) == 1                               # the second run used the cache (no transcription, no scoring)
    assert [s["id"] for s in first["sentences"]] == [s["id"] for s in second["sentences"]]
    assert [p["id"] for p in first["pairs"]] == [p["id"] for p in second["pairs"]] and first["pairs"] == second["pairs"]
    assert first["media"]["fingerprint"] == second["media"]["fingerprint"] and len(first["media"]["fingerprint"]) == 64
    trimmed = analyze_media(media, duration_seconds=25.8, used=[[8.0, 26.0]], cfg=cfg, scorer=scorer, transcribe_fn=_fake_transcribe(calls))
    assert len(calls) == 1                                                    # `used` filters AFTER scoring: ids are the same, nothing is retranscribed
    assert {s["id"] for s in trimmed["sentences"]} < {s["id"] for s in first["sentences"]}
    assert {p["id"] for p in trimmed["pairs"]} <= {p["id"] for p in first["pairs"]}


def test_builders_on_sample_c_pass_the_validator_and_group_bounds_equal_member_bounds(tmp_path):
    media = tmp_path / "sample_c.mp4"
    media.write_bytes(b"sample_c stand-in")
    cfg = ReviewConfig(cache_dir=str(tmp_path / "cache"), labels_path=str(tmp_path / "labels.jsonl"), skipped_path=str(tmp_path / "skipped.jsonl"))
    cand = analyze_media(media, duration_seconds=25.8, cfg=cfg, scorer=FinalScorer(mock=True), transcribe_fn=_fake_transcribe([]))
    assert cand["sentences"] and cand["pairs"], "the sample_c fixture should produce sentences and pairs"
    assert validate_document(cand, file_duration=25.8) == []
    answers = [{"pair": p["id"], "label": "retake", "source": "user"} for p in cand["pairs"] if p["class"] == "ambiguous"]
    request = {"path": str(media), "candidates": cand, "labels": {"job": cand["job"], "labels": answers, "skipped": []}}
    groups = resolve_request(request, cfg)
    assert groups["job"] == cand["job"] and groups["groups"], "expected at least one group"
    assert validate_document(groups, file_duration=25.8, expected_job=cand["job"]) == []
    for g in groups["groups"]:
        assert g["start"] == min(s["start"] for s in g["sentences"]) and g["end"] == max(s["end"] for s in g["sentences"])     # exact, no padding
        assert [s["start"] for s in g["sentences"]] == sorted(s["start"] for s in g["sentences"])
    lines = [json.loads(x) for x in Path(cfg.labels_path).read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(answers) and all(r["source"] == "user" and r["label"] == "retake" and r["was_asked_because"] == "unsure_band" for r in lines)
    resolve_request(request, cfg)                                             # repeatable: the same answers are not written twice
    assert len(Path(cfg.labels_path).read_text(encoding="utf-8").splitlines()) == len(answers)
