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

__all__ = [
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
