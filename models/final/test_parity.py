"""Stage 2 parity tests on two more labelled videos (video_07 = fold 3, video_02 = another speaker); see results_v2/final/parity.md.
They run the REAL pipeline code (cached transcript -> units -> candidates -> run_stage2); no ASR, no GPU, no API calls. Every test skips cleanly
when the cached artifacts are missing.  Run:  pytest models/final/test_parity.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from final_scorer import FinalScorer

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRAIN = REPO / "data" / "training"
OOF = TRAIN / "results_v2" / "final" / "final_oof_scores.csv"
VIDEOS = ["video_07", "video_02"]


def _have(video):
    return all(p.exists() for p in (TRAIN / "processed" / video / "transcript.json", TRAIN / "pairs_v2.csv", OOF, HERE / "thresholds.json"))


@pytest.fixture(scope="module")
def tables():
    pytest.importorskip("sentence_transformers")
    sys.path.insert(0, str(REPO))
    csv = pd.read_csv(TRAIN / "pairs_v2.csv", dtype=str, keep_default_na=False)
    csv = csv[csv.is_synthetic == "0"].set_index("pair_id")
    for c in ("lexical_score", "semantic_score", "containment"):
        csv[c] = pd.to_numeric(csv[c])
    csv["csv_plain"] = 0.5 * csv.semantic_score + 0.3 * csv.lexical_score + 0.2 * csv.containment
    oof = pd.read_csv(OOF, dtype={"pair_id": str}).set_index("pair_id")
    return csv, oof


def _pipeline_candidates(video):
    from models.transcript import Transcript
    from pipeline.step_03_token_matching import TokenMatchingConfig, generate_candidate_pairs, load_transcript_units
    from pipeline.step_04_semantic_matching import run_stage2

    units = load_transcript_units(Transcript.from_json_file(str(TRAIN / "processed" / video / "transcript.json")))
    out = []
    for c in run_stage2(generate_candidate_pairs(units, config=TokenMatchingConfig(candidate_threshold=0.10, window_seconds=300.0))):
        a, b = sorted((c["unit_a"], c["unit_b"]), key=lambda u: u["start"])
        out.append({"pair_id": f"{video}__{a['unit_id'].replace(':', '-')}__{b['unit_id'].replace(':', '-')}", "lexical_score": c["lexical_score"],
                    "semantic_score": c["semantic_score"], "containment": c["scores"]["containment"], "stage2_score": c["stage2_score"]})
    return out


@pytest.mark.parametrize("video", VIDEOS)
def test_plain_formula_from_pipeline_fields_matches_pairs_v2(video, tables):
    if not _have(video):
        pytest.skip(f"cached artifacts for {video} missing")
    csv, _ = tables
    cands = [c for c in _pipeline_candidates(video) if c["pair_id"] in csv.index]
    assert len(cands) >= 100
    s = FinalScorer(mock=True)
    gaps = np.array([abs(s.stage2_score(c) - csv.loc[c["pair_id"], "csv_plain"]) for c in cands])   # scorer ignores the pipeline's own stage2_score here
    assert gaps.max() < 1e-3, gaps.max()                                                           # CSV features are rounded to 3 decimals
    # the raw pipeline value is a different (recency-adjusted) quantity: documented, and the scorer must not use it
    raw = np.array([abs(c["stage2_score"] - csv.loc[c["pair_id"], "csv_plain"]) for c in cands])
    assert raw.mean() > 0.01


@pytest.mark.parametrize("video", VIDEOS)
def test_scorer_with_saved_out_of_fold_cross_encoder_reproduces_saved_tiers(video, tables):
    if not _have(video):
        pytest.skip(f"cached artifacts for {video} missing")
    _, oof = tables
    cands = [c for c in _pipeline_candidates(video) if c["pair_id"] in oof.index]
    assert len(cands) >= 100
    s = FinalScorer(mock=True)
    pairs = [{**c, "ce_prob": oof.loc[c["pair_id"], "ce_prob_mean"]} for c in cands]        # pipeline fields + saved out-of-fold CE probability
    p = s.score_pairs(pairs)
    saved = oof.loc[[c["pair_id"] for c in cands], "stacker_prob_final"].values
    assert np.max(np.abs(p - saved)) < 5e-3                                                      # same stacker; only the 3-decimal rounding differs
    assert np.mean(np.array(s.classify(p)) == np.array(s.classify(saved))) >= 0.995


@pytest.mark.parametrize("video", VIDEOS)
def test_full_scorer_with_final_cross_encoder_is_close_to_saved_out_of_fold_values(video, tables):
    """Loose guard only: the final cross-encoder saw these rows, so it is NOT an out-of-sample test; tiers should still mostly agree."""
    if not _have(video) or not (HERE / "ce_seed2" / "config.json").exists():
        pytest.skip("cached artifacts or trained weights missing")
    csv, oof = tables
    ids = [i for i in oof.index if i.startswith(video + "__")]
    sample = pd.Series(ids).sample(60, random_state=0).tolist()
    s = FinalScorer(device="cpu")
    assert s.mode == "stacker"
    p = s.score_pairs([{"pair_id": i, "text_a": csv.loc[i, "text_a"], "text_b": csv.loc[i, "text_b"]} for i in sample])
    saved = oof.loc[sample, "stacker_prob_oof"].values
    agree = np.mean(np.array(s.classify(p)) == np.array(s.classify(saved)))
    assert agree >= 0.75, agree
    assert np.mean(np.abs(p - saved)) < 0.2
