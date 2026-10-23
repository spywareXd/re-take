from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from pipeline.step_04_semantic_matching import (
    DISAGREEMENT_FLAG_THRESHOLD,
    STAGE2_AMBIGUOUS_FLOOR,
    STAGE2_RETAKE_THRESHOLD,
    embed_candidate_units,
    get_model,
    run_stage2,
    score_candidate,
    semantic_similarity_score,
)


def test_identical_text_semantic_score_near_one():
    text = "[redacted sentence]."
    model = get_model()
    emb = model.encode([text, text])
    score = semantic_similarity_score(emb[0], emb[1])
    assert score >= 0.99


def test_unrelated_sentences_low_semantic_score():
    text1 = "Open this next lesson and I show you exactly what is really going on."
    text2 = "Astronomers recently observed a distant neutron star colliding with a black hole."
    model = get_model()
    emb = model.encode([text1, text2])
    score = semantic_similarity_score(emb[0], emb[1])
    assert score < 0.55


def test_paraphrase_semantic_score_boost():
    """
    tea time transcript paraphrase case:
    2:1 ('Now we're going to learn about tea time, guys.') vs
    5:0 ('Hey guys, let's talk about tea time.')
    Stage 1 lexical combined_score is ~0.35.
    Assert semantic_score is meaningfully higher (> 0.70) than combined_score.
    """
    text_2_1 = "Now we're going to learn about tea time, guys."
    text_5_0 = "Hey guys, let's talk about tea time."

    model = get_model()
    emb = model.encode([text_2_1, text_5_0])
    sem_score = semantic_similarity_score(emb[0], emb[1])

    lexical_combined_score = 0.35
    assert sem_score > 0.70
    assert sem_score > (lexical_combined_score + 0.30)


def test_batch_embedding_consistency():
    texts = [
        "Now we're going to learn about tea time.",
        "[redacted sentence].",
        "Friends, you wonder why green tea works for some people.",
    ]
    model = get_model()

    # Individual embeddings
    single_embs = [model.encode(t) for t in texts]

    # Batch embedding
    batch_embs = model.encode(texts, batch_size=32)

    for single, batch in zip(single_embs, batch_embs):
        assert np.allclose(single, batch, atol=1e-3)
        assert semantic_similarity_score(single, batch) > 0.9999


def test_unique_units_embedded_once():
    """
    When units appear across multiple candidate pairs (e.g. unit 1:1 in pair 1 & pair 2),
    it must be embedded only once.
    """
    candidates = [
        {
            "candidate_id": 1,
            "unit_a": {"unit_id": "1:1", "raw_text": "Today we are talking about tea time."},
            "unit_b": {"unit_id": "2:1", "raw_text": "Today we are talking about tea time, guys."},
        },
        {
            "candidate_id": 2,
            "unit_a": {"unit_id": "1:1", "raw_text": "Today we are talking about tea time."},
            "unit_b": {"unit_id": "5:0", "raw_text": "Hey guys, let's talk about tea time."},
        },
    ]

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((3, 384), dtype=np.float32)

    with patch("pipeline.step_04_semantic_matching.semantic_matching.get_model", return_value=mock_model):
        embeddings = embed_candidate_units(candidates)

        # 3 unique unit_ids: 1:1, 2:1, 5:0
        assert len(embeddings) == 3
        mock_model.encode.assert_called_once()
        encoded_texts = mock_model.encode.call_args[0][0]
        assert len(encoded_texts) == 3


def test_high_disagreement_forces_ambiguous():
    """
    Construct synthetic candidate with high combined_score (0.90) but low semantic_score (0.20).
    |0.90 - 0.20| = 0.70 > DISAGREEMENT_FLAG_THRESHOLD (0.35).
    Classification must be 'ambiguous' even if blended stage2_score >= 0.65.
    """
    # Vectors with opposite alignment to produce low cosine similarity
    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([-0.6, 0.8, 0.0], dtype=np.float32)

    embeddings = {"unit_1": emb_a, "unit_2": emb_b}
    candidate = {
        "candidate_id": 100,
        "unit_a": {"unit_id": "unit_1", "raw_text": "Synthetic take A"},
        "unit_b": {"unit_id": "unit_2", "raw_text": "Synthetic take B"},
        "scores": {"containment": 1.0},
        "combined_score": 0.90,
    }

    result = score_candidate(candidate, embeddings)

    assert result["high_disagreement_flag"] is True
    assert result["lexical_semantic_disagreement"] > DISAGREEMENT_FLAG_THRESHOLD
    assert result["classification"] == "ambiguous"


def test_classification_boundary_behavior():
    # Identical vectors (cos sim = 1.0)
    emb_same = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = {"u1": emb_same, "u2": emb_same}

    # Case 1: Exact retake threshold (stage2_score == 0.65)
    # Weights: 0.5*sem + 0.3*lex + 0.2*cont
    # If sem=0.65, lex=0.65, cont=0.65 -> stage2_score = 0.65, disagreement = 0
    candidate_boundary_high = {
        "unit_a": {"unit_id": "u1"},
        "unit_b": {"unit_id": "u2"},
        "scores": {"containment": 0.30},
        "combined_score": 1.0,  # sem=1.0, lex=1.0 -> disagreement = 0
    }
    # For precise threshold testing, pass custom weights
    res_high = score_candidate(
        candidate_boundary_high,
        embeddings,
        weights={"semantic_similarity": 0.65, "lexical_combined_score": 0.0, "containment_bonus": 0.0},
        retake_threshold=0.65,
    )
    assert res_high["stage2_score"] == 0.65
    assert res_high["classification"] == "likely_retake"

    # Case 2: Exact ambiguous floor (stage2_score == 0.35)
    # If stage2_score == floor, it is NOT < floor, so it falls into 'ambiguous'
    res_floor = score_candidate(
        candidate_boundary_high,
        embeddings,
        weights={"semantic_similarity": 0.35, "lexical_combined_score": 0.0, "containment_bonus": 0.0},
        ambiguous_floor=0.35,
    )
    assert res_floor["stage2_score"] == 0.35
    assert res_floor["classification"] == "ambiguous"

    # Case 3: Below ambiguous floor (stage2_score = 0.34 < 0.35)
    res_below_floor = score_candidate(
        candidate_boundary_high,
        embeddings,
        weights={"semantic_similarity": 0.34, "lexical_combined_score": 0.0, "containment_bonus": 0.0},
        ambiguous_floor=0.35,
    )
    assert res_below_floor["stage2_score"] == 0.34
    assert res_below_floor["classification"] == "unlikely_retake"


def test_run_stage2_end_to_end():
    candidates = [
        {
            "candidate_id": 1,
            "unit_a": {"unit_id": "1:0", "raw_text": "[redacted sentence]."},
            "unit_b": {"unit_id": "2:0", "raw_text": "[redacted sentence]."},
            "scores": {"containment": 1.0, "exact_match": 1.0},
            "combined_score": 1.0,
            "containment_direction": "equal_length",
        }
    ]

    results = run_stage2(candidates)
    assert len(results) == 1
    r = results[0]
    assert r["candidate_id"] == 1
    assert r["semantic_score"] >= 0.99
    assert r["stage2_score"] >= 0.99
    assert r["classification"] == "likely_retake"
    assert r["high_disagreement_flag"] is False


def test_disagreement_uses_lexical_not_time_decayed_combined():
    """
    Verify that lexical_semantic_disagreement computes |lexical_score - semantic_score|
    rather than using time-decayed combined_score.
    Example: lexical=0.926, combined=0.275 (recency=0.297), semantic=0.990.
    Disagreement must be ~0.064 (not |0.275 - 0.990| = 0.715).
    """
    emb_same = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    embeddings = {"u1": emb_same, "u2": emb_same}

    candidate = {
        "unit_a": {"unit_id": "u1"},
        "unit_b": {"unit_id": "u2"},
        "scores": {"containment": 0.9},
        "lexical_score": 0.926,
        "combined_score": 0.275,
    }

    result = score_candidate(candidate, embeddings)
    # Cosine sim is 1.0; disagreement is |0.926 - 1.0| = 0.074, NOT |0.275 - 1.0| = 0.725
    assert result["lexical_semantic_disagreement"] == 0.074
    assert result["high_disagreement_flag"] is False
