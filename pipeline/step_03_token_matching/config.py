from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.step_02_preprocessing.config import PAUSE_SPLIT_THRESHOLD_SECONDS



LEXICAL_WEIGHTS = {
    "sequence": 0.30,
    "jaccard": 0.25,
    "containment": 0.25,
    "bigram": 0.20,
}


@dataclass(slots=True)
class TokenMatchingConfig:
    min_tokens: int = 3
    # Aligned with the settings used to export pairs_v1 / pairs_v2 (training/config.py: DEFAULT_WINDOW_SECONDS = 300.0, DEFAULT_MIN_LEXICAL = 0.10),
    # because the final scorer was fitted and evaluated on candidates generated this way. OLD default: 120.0. Still configurable per run.
    window_seconds: float = 300.0
    recency_decay_seconds: float = 45.0
    # OLD default: 0.35 (it dropped low-overlap paraphrased retakes before Stage 4). Now 0.10, same as the export. Still configurable per run.
    candidate_threshold: float = 0.10
    pause_split_threshold_seconds: float = PAUSE_SPLIT_THRESHOLD_SECONDS
    lexical_weights: dict[str, float] = field(default_factory=lambda: dict(LEXICAL_WEIGHTS))
    restart_phrase_lexicon: tuple[str, ...] = (
        "sorry",
        "wait",
        "again",
        "let me restart",
        "cut",
        "one more time",
    )
    filler_words: tuple[str, ...] = ("um", "uh", "like", "you know")


default_config = TokenMatchingConfig()


@dataclass(frozen=True, slots=True)
class IntraUnitConfig:
    """
    Settings for the OPTIONAL intra-unit retake candidate detector (suggestions for the review UI only).

    A unit is suggested only when rule R1 (prefix restart) AND rule R2 (repeated n-gram) both fire.
    Measured on a 63-row review sheet: R1+R2 together were right on 9 of 12 units; R1 alone 1/6 and
    R2 alone 0/4, so neither is used on its own. Values mirror data/training/intra_detect.py.
    """

    detector_version: str = "intra_r1r2_v1"
    min_tokens: int = 6                       # units shorter than this are never flagged
    prefix_min_tokens: int = 2                # R1: first k tokens (k = 2..5) must reappear later in the unit
    prefix_max_tokens: int = 5
    ngram_min: int = 3                        # R2: repeated token n-gram, n >= 3 ...
    ngram_max: int = 8
    short_gap_seconds: float = 0.35           # ... with a token between the copies OR a silence shorter than this
    # intentional doublings that are never a restart ("very very")
    doublings: tuple[tuple[str, str], ...] = (("very", "very"), ("no", "no"), ("bye", "bye"), ("go", "go"))
    number_words: frozenset[str] = frozenset({
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million",
    })
    # an n-gram / prefix only counts if it holds at least one word that is not in this set (and not a number)
    function_words: frozenset[str] = frozenset({
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "is", "are", "was",
        "were", "be", "it", "its", "i", "you", "he", "she", "we", "they", "that", "this", "so", "if", "as",
        "do", "have", "my", "your", "me", "um", "uh", "like",
    })
    terminal_punctuation: str = ".!?"
    # first_half_complete heuristic (ranking hint only, never a filter)
    subject_words: frozenset[str] = frozenset({"i", "you", "we", "they", "he", "she", "it", "this", "that", "there"})
    common_verbs: frozenset[str] = frozenset({
        "am", "is", "are", "was", "were", "be", "have", "has", "had", "do", "does", "did", "use", "uses", "go",
        "goes", "get", "gets", "make", "makes", "want", "need", "know", "think", "love", "like", "see", "say",
        "take", "give", "try", "feel", "can", "will", "would", "could", "should", "just", "went", "got",
    })
    verb_suffixes: tuple[str, ...] = ("ed", "ing")
    subject_verb_window: int = 3              # a verb-like word within this many tokens after a subject word


default_intra_config = IntraUnitConfig()
