from __future__ import annotations

from .config import default_config
from .edit_distance import containment_details, sequence_similarity_score
from .exact_match import exact_match_score, near_exact_match_score
from .jaccard import jaccard_score
from .ngram_match import ngram_overlap_score
from .restart_phrase import restart_phrase_score
from .schemas import CandidatePair, CandidateScores, CandidateUnitRef, ComparableUnit
from .windowing import get_candidates_in_window


def _combine_scores(scores: CandidateScores, recency_weight: float, config: object) -> float:
    weights = getattr(config, "lexical_weights", default_config.lexical_weights)
    lexical_total = (
        scores.sequence * weights.get("sequence", 0.0)
        + scores.jaccard * weights.get("jaccard", 0.0)
        + scores.containment * weights.get("containment", 0.0)
        + scores.bigram * weights.get("bigram", 0.0)
    )
    near_exact_boost = max(scores.near_exact_match, scores.exact_match)
    combined = max(lexical_total, near_exact_boost)
    combined = min(1.0, combined * recency_weight)
    return combined


def generate_candidate_pairs(
    all_units: list[ComparableUnit],
    config: object = default_config,
) -> list[CandidatePair]:
    candidate_threshold = getattr(config, "candidate_threshold", default_config.candidate_threshold)
    filler_words = set(getattr(config, "filler_words", default_config.filler_words))
    candidate_pairs: list[CandidatePair] = []
    seen_pairs: set[tuple[str, str]] = set()
    candidate_id = 1

    for unit in all_units:
        if unit.is_below_min_token_length:
            continue

        for other, recency_weight in get_candidates_in_window(unit, all_units, config):
            if other.is_below_min_token_length:
                continue

            pair_key = tuple(sorted((unit.unit_id, other.unit_id)))
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            exact = exact_match_score(unit, other)
            if exact:
                scores = CandidateScores(
                    exact_match=1.0,
                    near_exact_match=1.0,
                    jaccard=1.0,
                    sequence=1.0,
                    containment=1.0,
                    bigram=1.0,
                )
                candidate_pairs.append(
                    CandidatePair(
                        candidate_id=candidate_id,
                        unit_a=CandidateUnitRef.from_unit(unit),
                        unit_b=CandidateUnitRef.from_unit(other),
                        scores=scores,
                        combined_score=1.0,
                        restart_phrase_detected=False,
                        containment_direction="equal_length",
                        recency_weight=recency_weight,
                    )
                )
                candidate_id += 1
                continue

            near_exact = near_exact_match_score(unit, other, filler_words=filler_words)
            jaccard = jaccard_score(unit, other)
            sequence = sequence_similarity_score(unit, other)
            containment = containment_details(unit, other)
            bigram = ngram_overlap_score(unit, other, n=2)

            scores = CandidateScores(
                exact_match=exact,
                near_exact_match=near_exact,
                jaccard=jaccard,
                sequence=sequence,
                containment=containment.score,
                bigram=bigram,
            )
            restart_detected = restart_phrase_score(f"{unit.raw_text} {other.raw_text}", config=config) > 0.0
            combined_score = _combine_scores(scores, recency_weight, config)
            if restart_detected:
                combined_score = min(1.0, combined_score + 0.05)

            if combined_score < candidate_threshold:
                continue

            candidate_pairs.append(
                CandidatePair(
                    candidate_id=candidate_id,
                    unit_a=CandidateUnitRef.from_unit(unit),
                    unit_b=CandidateUnitRef.from_unit(other),
                    scores=scores,
                    combined_score=combined_score,
                    restart_phrase_detected=restart_detected,
                    containment_direction=containment.direction,
                    recency_weight=recency_weight,
                )
            )
            candidate_id += 1

    candidate_pairs.sort(key=lambda item: item.combined_score, reverse=True)
    return candidate_pairs


def rank_similar_sentence_ids(
    all_units: list[ComparableUnit],
    config: object = default_config,
    limit: int = 10,
) -> list[dict[str, object]]:
    ranked_pairs = generate_candidate_pairs(all_units, config=config)
    ranked_sentence_ids: list[dict[str, object]] = []

    unit_lookup = {unit.unit_id: unit for unit in all_units}

    for pair in ranked_pairs[:limit]:
        unit_a = unit_lookup.get(pair.unit_a.unit_id)
        unit_b = unit_lookup.get(pair.unit_b.unit_id)
        ranked_sentence_ids.append(
            {
                "sentence_a_id": pair.unit_a.unit_id,
                "sentence_b_id": pair.unit_b.unit_id,
                "sentence_a_text": unit_a.raw_text if unit_a else "",
                "sentence_b_text": unit_b.raw_text if unit_b else "",
                "segment_a_id": pair.unit_a.segment_id,
                "segment_b_id": pair.unit_b.segment_id,
                "combined_score": pair.combined_score,
                "candidate_id": pair.candidate_id,
            }
        )

    return ranked_sentence_ids
