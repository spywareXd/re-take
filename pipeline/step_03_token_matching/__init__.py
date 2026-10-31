"""Step 3: token matching. Scores pairs of sentence units (from step 2) with exact, edit-distance, Jaccard and n-gram overlap and keeps the likely candidates."""
from .candidate_generator import generate_candidate_pairs, rank_similar_sentence_ids
from .config import (
    IntraUnitConfig,
    TokenMatchingConfig,
    default_config,
    default_intra_config,
)
from .edit_distance import containment_details, containment_score, sequence_similarity_score
from .exact_match import exact_match_score, near_exact_match_score
from .intra_candidates import (
    IntraUnitCandidate,
    find_intra_unit_candidates,
    load_intra_unit_candidates,
    save_intra_unit_candidates,
)
from .jaccard import jaccard_score
from .ngram_match import ngram_overlap_score
from .restart_phrase import restart_phrase_score
from .schemas import CandidatePair, CandidateScores, CandidateUnitRef
from .storage import load_semantic_candidates, save_semantic_candidates
from .windowing import get_candidates_in_window

# Backward-compatible aliases: these moved to pipeline.step_02_preprocessing. Import them from there in new code.
from pipeline.step_02_preprocessing import (  # noqa: E402,F401
    PAUSE_SPLIT_THRESHOLD_SECONDS,
    ComparableUnit,
    TranscriptSegment,
    WordTimestamp,
    build_comparable_unit,
    extract_segment_words,
    flatten_transcript_words,
    load_comparable_units,
    load_transcript_units,
    normalize,
    save_comparable_units,
    split_into_units,
    split_transcript_into_units,
    split_words_by_pause,
    tokenize,
)

__all__ = [
    "PAUSE_SPLIT_THRESHOLD_SECONDS", "ComparableUnit", "TranscriptSegment", "WordTimestamp", "build_comparable_unit", "extract_segment_words",
    "flatten_transcript_words", "load_comparable_units", "load_transcript_units", "normalize", "save_comparable_units", "split_into_units",
    "split_transcript_into_units", "split_words_by_pause", "tokenize",
    "CandidatePair",
    "IntraUnitCandidate",
    "IntraUnitConfig",
    "default_intra_config",
    "find_intra_unit_candidates",
    "load_intra_unit_candidates",
    "save_intra_unit_candidates",
    "CandidateScores",
    "CandidateUnitRef",
    "TokenMatchingConfig",
    "default_config",
    "containment_details",
    "containment_score",
    "exact_match_score",
    "generate_candidate_pairs",
    "get_candidates_in_window",
    "jaccard_score",
    "load_semantic_candidates",
    "near_exact_match_score",
    "ngram_overlap_score",
    "rank_similar_sentence_ids",
    "restart_phrase_score",
    "save_semantic_candidates",
    "sequence_similarity_score",
]
