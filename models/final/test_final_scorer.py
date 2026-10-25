"""Tests for models/final/final_scorer.py. No GPU and no API calls (mock path); the optional real-model smoke test runs on CPU only if the
trained weights exist. Run:  pytest models/final/test_final_scorer.py"""
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from final_scorer import FinalScorer, classify, score_pairs, to_spans

HERE = Path(__file__).resolve().parent
OOF = HERE.parents[1] / "data" / "training" / "results_v2" / "final" / "final_oof_scores.csv"
THR = HERE / "thresholds.json"
need_artifacts = pytest.mark.skipif(not (THR.exists() and OOF.exists()), reason="final artifacts not built (run fit_final.py)")


# ---------------- fallback ----------------
def test_fallback_when_files_missing_uses_legacy_rules_if_the_tuned_file_is_missing_too(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="final_scorer"):
        s = FinalScorer(model_dir=tmp_path, tuned_thresholds_path=tmp_path / "no_such_tuned.json")
    assert s.mode == "legacy_stage2_fallback" and s.scorer_name == "legacy_stage2_fallback" and (s.high, s.low) == (0.65, 0.35)
    assert any("FALLING BACK to the legacy" in r.message for r in caplog.records)           # a warning was logged
    p = s.score_pairs([{"stage2_score": 0.80}, {"stage2_score": 0.50}, {"lexical_score": 0.2, "semantic_score": 0.2, "containment": 0.2}])
    assert p[0] == pytest.approx(0.80) and p[1] == pytest.approx(0.50) and p[2] == pytest.approx(0.20)      # the score IS the Stage 2 formula
    assert s.classify(p) == ["retake", "ambiguous", "unlikely"]
    # old-rule boundaries: >= 0.65 retake, < 0.35 unlikely
    assert s.classify(0.65) == "retake" and s.classify(0.6499999) == "ambiguous" and s.classify(0.35) == "ambiguous" and s.classify(0.3499999) == "unlikely"


@need_artifacts
def test_fallback_when_cross_encoder_weights_are_missing(tmp_path, caplog):
    shutil.copy(THR, tmp_path / "thresholds.json")                               # thresholds present, weights absent
    with caplog.at_level(logging.WARNING, logger="final_scorer"):
        s = FinalScorer(model_dir=tmp_path)
    assert s.mode == "tuned_stage2_fallback" and any("ce_seed0" in r.message for r in caplog.records)
    assert s.scorer_name == "tuned_stage2_fallback"


# ---------------- classification boundaries ----------------
@need_artifacts
def test_classification_boundaries_use_the_saved_thresholds():
    s = FinalScorer(mock=True)
    art = json.loads(THR.read_text())
    hi, lo = art["thresholds"]["high"], art["thresholds"]["low"]
    assert s.mode == "stacker" and (s.high, s.low) == (hi, lo) and lo < hi
    assert s.classify(hi) == "retake" and s.classify(hi - 1e-9) == "ambiguous"
    assert s.classify(lo) == "ambiguous" and s.classify(lo - 1e-9) == "unlikely"
    assert s.classify(np.array([0.0, (lo + hi) / 2, 1.0])) == ["unlikely", "ambiguous", "retake"]
    assert classify(1.0, scorer=s) == "retake"                                  # module-level wrapper


# ---------------- golden labelled pairs (inputs and expected tiers come from the saved out-of-fold file) ----------------
GOLDEN = [   # pair_id, label, expected tier under the FINAL stacker and thresholds
    ("sample_d__2-0__3-0", 1, "retake"),                   # identical line said twice
    ("speaker_a_5__8-0__10-0", 1, "retake"),
    ("speaker_a_6__13-0__45-0", 0, "unlikely"),            # unrelated sentences
    ("video_07__13-0__29-0", 0, "unlikely"),
    ("speaker_a_3__2-0__3-0", 1, "ambiguous"),             # false start "In the last," : the cross-encoder says no, Stage 2 says yes
    ("speaker_a_4__18-0__47-1", 0, "ambiguous"),
    ("speaker_a_3__1-0__2-1", 0, "retake"),                # a known false positive of the final scorer ("Here we go." vs "...here we go.")
]


@need_artifacts
def test_golden_labelled_pairs():
    oof = pd.read_csv(OOF, dtype={"pair_id": str}).set_index("pair_id")
    s = FinalScorer(mock=True)                           # mock only affects text-only pairs; these carry precomputed scores
    pairs = [{"pair_id": pid, "stage2_score": oof.loc[pid, "stage2_score"], "ce_prob": oof.loc[pid, "ce_prob_mean"]} for pid, _, _ in GOLDEN]
    got = s.classify(s.score_pairs(pairs))
    assert got == [g[2] for g in GOLDEN]
    assert all(int(oof.loc[pid, "label"]) == lab for pid, lab, _ in GOLDEN)


# ---------------- scores reproduce the saved out-of-fold values ----------------
@need_artifacts
def test_scores_reproduce_saved_out_of_fold_values():
    oof = pd.read_csv(OOF, dtype={"pair_id": str})
    s = FinalScorer(mock=True)
    pairs = [{"stage2_score": a, "ce_prob": b} for a, b in zip(oof.stage2_score, oof.ce_prob_mean)]
    p = s.score_pairs(pairs)
    assert np.allclose(p, oof.stacker_prob_final.values, atol=1e-9)              # the packaged stacker == the one fitted in fit_final.py
    # and it stays close to the honest fold-wise stacker probabilities that the thresholds were chosen on
    assert np.mean(np.abs(p - oof.stacker_prob_oof.values)) < 0.03
    agree = np.mean(np.array(s.classify(p)) == oof.tier_final_thr.values)
    assert agree > 0.97, agree


# ---------------- mock path and spans ----------------
@need_artifacts
def test_mock_path_runs_without_models_and_separates_obvious_pairs():
    s = FinalScorer(mock=True)
    same = {"text_a": "[redacted sentence].", "text_b": "[redacted sentence]."}
    diff = {"text_a": "The motor might spin up but you are not getting anywhere.", "text_b": "Then to give my garden the extra boost there are three fertilizers."}
    p = s.score_pairs([same, diff])
    assert s.classify(p) == ["retake", "unlikely"] and p[0] > p[1]
    assert np.allclose(p, s.score_pairs([same, diff]))                           # deterministic


@need_artifacts
def test_spans_have_status_and_keeper_is_left_unset():
    s = FinalScorer(mock=True)
    rows = [
        {"pair_id": "v__1-0__2-0", "video_id": "v", "unit_a_id": "1:0", "unit_b_id": "2:0", "start_a": 1.0, "end_a": 2.0, "start_b": 3.0, "end_b": 4.0, "status": "retake", "p": 0.99},
        {"pair_id": "v__2-0__3-0", "video_id": "v", "unit_a_id": "2:0", "unit_b_id": "3:0", "start_a": 3.0, "end_a": 4.0, "start_b": 5.0, "end_b": 6.0, "status": "ambiguous", "p": 0.5},
        {"pair_id": "v__8-0__9-0", "video_id": "v", "unit_a_id": "8:0", "unit_b_id": "9:0", "start_a": 20.0, "end_a": 21.0, "start_b": 22.0, "end_b": 23.0, "status": "retake", "p": 0.97},
        {"pair_id": "v__4-0__5-0", "video_id": "v", "unit_a_id": "4:0", "unit_b_id": "5:0", "start_a": 9.0, "end_a": 10.0, "start_b": 11.0, "end_b": 12.0, "status": "unlikely", "p": 0.01},
    ]
    spans = to_spans(rows)
    assert len(spans) == 2                                                          # the unlikely pair makes no span
    a, b = spans
    assert a["unit_ids"] == ["1:0", "2:0", "3:0"] and a["status"] == "ambiguous" and (a["start"], a["end"]) == (1.0, 6.0)
    assert b["unit_ids"] == ["8:0", "9:0"] and b["status"] == "retake"
    assert all(sp["keeper_unit_id"] is None and sp["units_to_remove"] is None for sp in spans)


# ---------------- Stage 2 consistency: runtime scores must be the SAME function as the training scores ----------------
TRAIN = HERE.parents[1] / "data" / "training"
needs_data = pytest.mark.skipif(not (TRAIN / "results_v2" / "baseline_scores.csv").exists(), reason="training data not available")


def test_supplied_pipeline_stage2_is_ignored_when_parts_are_given(caplog):
    s = FinalScorer(mock=True)
    pair = {"pair_id": "x", "lexical_score": 0.6, "semantic_score": 0.9, "containment": 0.8, "stage2_score": 0.50}     # 0.50 mimics a recency-adjusted value
    plain = 0.5 * 0.9 + 0.3 * 0.6 + 0.2 * 0.8
    with caplog.at_level(logging.WARNING, logger="final_scorer"):
        assert s.stage2_score(pair) == pytest.approx(plain)
    assert any("recency-adjusted" in r.message for r in caplog.records)
    assert s.stage2_score({"lexical_score": 0.6, "semantic_score": 0.9, "containment": 0.8, "stage2_score": plain}) == pytest.approx(plain)   # consistent -> no warning needed


@needs_data
def test_text_path_reproduces_the_training_stage2_scores():
    """Stage 2 computed by the scorer from raw text equals the plain-formula score stored with the labelled pairs (same code, within the CSV rounding)."""
    pytest.importorskip("sentence_transformers")
    base = pd.read_csv(TRAIN / "results_v2" / "baseline_scores.csv", dtype={"pair_id": str})
    txt = pd.read_csv(TRAIN / "pairs_v2.csv", dtype=str, keep_default_na=False).set_index("pair_id")
    sample = base.sample(40, random_state=0)
    s = FinalScorer(mock=False, device="cpu")
    got = np.array([s.stage2_score({"text_a": txt.loc[p, "text_a"], "text_b": txt.loc[p, "text_b"]}) for p in sample.pair_id])
    assert np.max(np.abs(got - sample.current_stage2.values)) < 2e-3        # CSV features are rounded to 3 decimals


@needs_data
def test_pipeline_candidates_reproduce_training_scores_and_expose_the_recency_gap():
    """Run the REAL pipeline code (cached transcript -> units -> candidates -> run_stage2) on one labelled video and feed its candidates to the scorer."""
    pytest.importorskip("sentence_transformers")
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    from models.transcript import Transcript
    from pipeline.step_03_token_matching import TokenMatchingConfig, generate_candidate_pairs, load_transcript_units
    from pipeline.step_04_semantic_matching import run_stage2

    video = "speaker_a_3"
    tpath = TRAIN / "processed" / video / "transcript.json"
    if not tpath.exists():
        pytest.skip("cached transcript not available")
    units = load_transcript_units(Transcript.from_json_file(str(tpath)))
    cands = run_stage2(generate_candidate_pairs(units, config=TokenMatchingConfig(candidate_threshold=0.10, window_seconds=300.0)))
    base = pd.read_csv(TRAIN / "results_v2" / "baseline_scores.csv", dtype={"pair_id": str}).set_index("pair_id").current_stage2
    s = FinalScorer(mock=True)
    diffs_plain, diffs_pipeline = [], []
    for c in cands:
        a, b = sorted((c["unit_a"], c["unit_b"]), key=lambda u: u["start"])
        pid = f"{video}__{a['unit_id'].replace(':', '-')}__{b['unit_id'].replace(':', '-')}"
        if pid not in base.index:
            continue
        pair = {"pair_id": pid, "lexical_score": c["lexical_score"], "semantic_score": c["semantic_score"], "containment": c["scores"]["containment"],
                "stage2_score": c["stage2_score"]}                         # the pipeline's own (recency-adjusted) value is passed too, and must be ignored
        diffs_plain.append(abs(s.stage2_score(pair) - base[pid]))
        diffs_pipeline.append(abs(c["stage2_score"] - base[pid]))
    assert len(diffs_plain) >= 40
    assert max(diffs_plain) < 2e-3                                           # scorer == training score
    assert max(diffs_pipeline) > 0.05                                        # documents that the raw pipeline value is NOT the training score


# ---------------- real models (CPU is fine; skipped if the weights are not there) ----------------
@pytest.mark.skipif(not (HERE / "ce_seed2" / "config.json").exists() or not THR.exists(), reason="trained cross-encoder weights not present")
def test_real_models_smoke():
    pytest.importorskip("sentence_transformers")
    s = FinalScorer(device="cpu")
    assert s.mode == "stacker"
    pairs = [{"text_a": "[redacted sentence].", "text_b": "[redacted sentence]."},
             {"text_a": "The motor might spin up, but you're not getting anywhere.", "text_b": "Then to give my garden the extra boost, there are three fertilizers I found helpful."}]
    p = s.score_pairs(pairs)
    assert s.classify(p) == ["retake", "unlikely"]


# ---------------- pipeline integration: ambiguous pairs are resolved by the cross-encoder ----------------
@need_artifacts
def test_resolve_ambiguous_pairs_by_the_cross_encoder_threshold():
    s = FinalScorer(mock=True, unsure_band_enabled=False)          # the old single-threshold behaviour, reproduced by the flag
    assert s.t_amb is not None and s.low < s.high
    mid = (s.low + s.high) / 2
    decisions, by = s.resolve(np.array([s.high, s.low - 1e-9, mid, mid]), np.array([0.0, 1.0, s.t_amb, s.t_amb - 1e-9]))
    assert decisions == ["retake", "not_retake", "retake", "not_retake"]
    assert by == ["stacker", "stacker", "cross_encoder", "cross_encoder"]


def test_fallback_resolves_ambiguous_pairs_as_not_retake(tmp_path):
    s = FinalScorer(model_dir=tmp_path, tuned_thresholds_path=tmp_path / "missing.json")          # legacy 0.65 / 0.35 fallback
    decisions, by = s.resolve(np.array([0.9, 0.5, 0.1]), np.array([np.nan] * 3))
    assert decisions == ["retake", "not_retake", "not_retake"] and by == ["stage2_fallback"] * 3
