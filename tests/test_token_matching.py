import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step_03_token_matching import (
    containment_details,
    exact_match_score,
    generate_candidate_pairs,
    jaccard_score,
    load_transcript_units,
    near_exact_match_score,
    restart_phrase_score,
    sequence_similarity_score,
)
from pipeline.step_03_token_matching.config import TokenMatchingConfig
from pipeline.step_03_token_matching.candidate_generator import rank_similar_sentence_ids
from pipeline.step_03_token_matching.schemas import ComparableUnit


def test_load_output_transcript():
    test_path = PROJECT_ROOT / "data" / "transcripts" / "test.json"
    output_path = PROJECT_ROOT / "data" / "raw" / "transcripts" / "output_transcript.json"
    transcript_path = test_path if test_path.exists() else output_path

    units = load_transcript_units(transcript_path)
    assert len(units) > 0
    assert units[0].speaker.startswith("speaker_")
    assert units[0].confidence is not None
    print("[PASS] Loaded transcript into units successfully")


def test_similarity_matchers():
    u1 = ComparableUnit("1:0", 1, 0.0, 2.0, "Hello world", "hello world", ["hello", "world"], {"hello", "world"})
    u2 = ComparableUnit("2:0", 2, 3.0, 5.0, "Hello world", "hello world", ["hello", "world"], {"hello", "world"})
    u3 = ComparableUnit("3:0", 3, 6.0, 8.0, "Like, hello world um", "like hello world um", ["like", "hello", "world", "um"], {"like", "hello", "world", "um"})

    assert exact_match_score(u1, u2) == 1.0
    assert near_exact_match_score(u1, u3, filler_words={"like", "um"}) == 1.0
    assert jaccard_score(u1, u2) == 1.0
    assert sequence_similarity_score(u1, u2) == 1.0
    assert containment_details(u1, u3).score == 1.0
    print("[PASS] Similarity matchers executed successfully")


def test_candidate_generation():
    test_path = PROJECT_ROOT / "data" / "transcripts" / "test.json"
    output_path = PROJECT_ROOT / "data" / "raw" / "transcripts" / "output_transcript.json"
    transcript_path = test_path if test_path.exists() else output_path

    units = load_transcript_units(transcript_path)
    candidates = generate_candidate_pairs(units)
    assert len(candidates) > 0
    assert candidates[0].combined_score >= 0.35
    print(f"[PASS] Candidate generation found {len(candidates)} pairs")


def _unit(unit_id: str, start: float, text: str, *, below_minimum: bool = False) -> ComparableUnit:
    tokens = text.lower().split()
    return ComparableUnit(
        unit_id=unit_id,
        source_segment_id=int(unit_id.split(":")[0]),
        start=start,
        end=start + 1.0,
        raw_text=text,
        normalized_text=" ".join(tokens),
        tokens=tokens,
        token_set=set(tokens),
        is_below_min_token_length=below_minimum,
    )


def test_recall_first_scoring_and_contextual_restart_cues():
    config = TokenMatchingConfig(window_seconds=120.0, recency_decay_seconds=45.0, candidate_threshold=0.35)

    exact_pairs = generate_candidate_pairs(
        [_unit("1:0", 0.0, "alpha beta gamma"), _unit("2:0", 110.0, "alpha beta gamma")],
        config=config,
    )
    assert exact_pairs[0].lexical_score == 1.0
    assert exact_pairs[0].combined_score == 1.0

    near_pairs = generate_candidate_pairs(
        [_unit("1:0", 0.0, "um alpha beta gamma"), _unit("2:0", 110.0, "alpha beta gamma")],
        config=config,
    )
    assert near_pairs[0].scores.near_exact_match == 1.0
    assert near_pairs[0].lexical_score == 1.0
    assert near_pairs[0].combined_score == 1.0

    ambiguous_pairs = generate_candidate_pairs(
        [_unit("1:0", 0.0, "alpha beta gamma delta"), _unit("2:0", 110.0, "alpha beta gamma epsilon")],
        config=config,
    )
    assert ambiguous_pairs[0].lexical_score >= config.candidate_threshold
    assert ambiguous_pairs[0].combined_score < config.candidate_threshold

    weak_pairs = generate_candidate_pairs(
        [
            _unit("1:0", 0.0, "alpha beta gamma"),
            _unit("2:0", 1.0, "again", below_minimum=True),
            _unit("3:0", 2.0, "delta epsilon zeta"),
        ],
        config=config,
    )
    assert weak_pairs == []

    endpoint_cue_pairs = generate_candidate_pairs(
        [_unit("1:0", 0.0, "alpha beta again"), _unit("2:0", 10.0, "alpha beta again")],
        config=config,
    )
    assert not endpoint_cue_pairs[0].restart_phrase_detected

    contextual_cue_units = [
        _unit("1:0", 0.0, "alpha beta gamma delta"),
        _unit("2:0", 1.0, "again", below_minimum=True),
        _unit("3:0", 2.0, "alpha beta gamma epsilon"),
    ]
    contextual_cue_pair = next(
        pair
        for pair in generate_candidate_pairs(contextual_cue_units, config=config)
        if {pair.unit_a.unit_id, pair.unit_b.unit_id} == {"1:0", "3:0"}
    )
    assert contextual_cue_pair.restart_phrase_detected

    containment_units = [_unit("1:0", 0.0, "alpha beta gamma delta"), _unit("2:0", 10.0, "alpha beta gamma")]
    containment_pair = generate_candidate_pairs(containment_units, config=config)[0]
    assert containment_pair.to_dict()["lexical_score"] == containment_pair.lexical_score
    assert containment_pair.to_dict()["containment_direction"] == "b_subset_of_a"
    ranked_pair = rank_similar_sentence_ids(containment_units, config=config)[0]
    assert ranked_pair["containment_direction"] == "b_subset_of_a"
    print("[PASS] Recall-first scoring and contextual restart cues validated")


def test_expanded_contractions_and_normalization():
    from pipeline.step_03_token_matching import normalize

    # Test you're, it's, what's, let's, don't, etc.
    assert normalize("You're not seeing results") == "you are not seeing results"
    assert normalize("It's how you're using it") == "it is how you are using it"
    assert normalize("What's really going on") == "what is really going on"
    assert normalize("Let's restart") == "let us restart"
    assert normalize("We're going to start, but I can't!") == "we are going to start but i cannot"
    assert normalize("They've won't shouldn't") == "they have will not should not"

    # Test curly apostrophes / quotes
    assert normalize("you’re not gonna see it’s working") == "you are not going to see it is working"
    assert normalize("c’mon, y’all gotta check what’s up") == "come on you all got to check what is up"
    print("[PASS] Expanded contractions and normalization verified")


if __name__ == "__main__":
    test_load_output_transcript()
    test_similarity_matchers()
    test_candidate_generation()
    test_recall_first_scoring_and_contextual_restart_cues()
    test_expanded_contractions_and_normalization()
    print("\nAll token matching unit tests passed successfully!")
