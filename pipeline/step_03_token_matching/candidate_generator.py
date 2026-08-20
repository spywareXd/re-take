from __future__ import annotations

from .config import default_config
from .edit_distance import containment_details, sequence_similarity_score
from .exact_match import exact_match_score, near_exact_match_score
from .jaccard import jaccard_score
from .ngram_match import ngram_overlap_score
from .restart_phrase import restart_phrase_score
from .schemas import CandidatePair, CandidateScores, CandidateUnitRef, ComparableUnit
from .windowing import get_candidates_in_window


def _lexical_score(scores: CandidateScores, config: object) -> float:
    weights = getattr(config, "lexical_weights", default_config.lexical_weights)
    lexical_total = (
        scores.sequence * weights.get("sequence", 0.0)
        + scores.jaccard * weights.get("jaccard", 0.0)
        + scores.containment * weights.get("containment", 0.0)
        + scores.bigram * weights.get("bigram", 0.0)
    )
    near_exact_boost = max(scores.near_exact_match, scores.exact_match)
    return max(lexical_total, near_exact_boost)


def _restart_phrase_detected(
    all_units: list[ComparableUnit],
    first_index: int,
    second_index: int,
    config: object,
) -> bool:
    start_index, end_index = sorted((first_index, second_index))
    intervening_text = " ".join(unit.raw_text for unit in all_units[start_index + 1 : end_index])
    return bool(intervening_text) and restart_phrase_score(intervening_text, config=config) > 0.0


def generate_candidate_pairs(
    all_units: list[ComparableUnit],
    config: object = default_config,
) -> list[CandidatePair]:
    candidate_threshold = getattr(config, "candidate_threshold", default_config.candidate_threshold)
    filler_words = set(getattr(config, "filler_words", default_config.filler_words))
    candidate_pairs: list[CandidatePair] = []
    seen_pairs: set[tuple[str, str]] = set()
    unit_indexes = {unit.unit_id: index for index, unit in enumerate(all_units)}
    candidate_id = 1

    for unit_index, unit in enumerate(all_units):
        if unit.is_below_min_token_length:
            continue

        for other, recency_weight in get_candidates_in_window(unit, all_units, config):
            if other.is_below_min_token_length:
                continue

            pair_key = tuple(sorted((unit.unit_id, other.unit_id)))
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            other_index = unit_indexes[other.unit_id]
            exact = exact_match_score(unit, other)
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
            lexical_score = _lexical_score(scores, config)
            if lexical_score < candidate_threshold:
                continue

            restart_detected = _restart_phrase_detected(all_units, unit_index, other_index, config)
            is_certain_match = bool(exact or near_exact)
            combined_score = lexical_score if is_certain_match else lexical_score * recency_weight
            if restart_detected:
                combined_score = min(1.0, combined_score + 0.05)

            candidate_pairs.append(
                CandidatePair(
                    candidate_id=candidate_id,
                        unit_a=CandidateUnitRef.from_unit(unit),
                        unit_b=CandidateUnitRef.from_unit(other),
                        scores=scores,
                        lexical_score=lexical_score,
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
                "containment_direction": pair.containment_direction,
                "candidate_id": pair.candidate_id,
            }
        )

    return ranked_sentence_ids
