import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.token_matching import (
    load_transcript_units,
    generate_candidate_pairs,
    exact_match_score,
    near_exact_match_score,
    jaccard_score,
    sequence_similarity_score,
    containment_details,
    restart_phrase_score,
)
from pipeline.token_matching.schemas import ComparableUnit


def test_load_output_transcript():
    transcript_path = PROJECT_ROOT / "data" / "raw" / "transcripts" / "output_transcript.json"
    if not transcript_path.exists():
        transcript_path = PROJECT_ROOT / "data" / "transcripts" / "test.json"

    units = load_transcript_units(transcript_path)
    assert len(units) > 0
    assert units[0].speaker == "speaker_0"
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
    transcript_path = PROJECT_ROOT / "data" / "raw" / "transcripts" / "output_transcript.json"
    if not transcript_path.exists():
        transcript_path = PROJECT_ROOT / "data" / "transcripts" / "test.json"

    units = load_transcript_units(transcript_path)
    candidates = generate_candidate_pairs(units)
    assert len(candidates) > 0
    # Highest scoring candidates should have score > 0.5
    assert candidates[0].combined_score >= 0.5
    print(f"[PASS] Candidate generation found {len(candidates)} pairs")


if __name__ == "__main__":
    test_load_output_transcript()
    test_similarity_matchers()
    test_candidate_generation()
    print("\nAll token matching unit tests passed successfully!")
