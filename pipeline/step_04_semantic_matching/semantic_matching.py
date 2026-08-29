from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Sequence, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    DISAGREEMENT_FLAG_THRESHOLD,
    SEMANTIC_EMBEDDING_MODEL,
    SEMANTIC_WEIGHTS,
    STAGE2_AMBIGUOUS_FLOOR,
    STAGE2_RETAKE_THRESHOLD,
    default_semantic_config,
)

_model: SentenceTransformer | None = None
_current_model_name: str | None = None


def get_model(model_name: str = SEMANTIC_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer model once per process.
    Reloads only if model_name changes.
    """
    global _model, _current_model_name
    if _model is None or _current_model_name != model_name:
        _model = SentenceTransformer(model_name)
        _current_model_name = model_name
    return _model


def _get_unit_field(unit: Any, field_name: str, default: Any = "") -> Any:
    if isinstance(unit, dict):
        return unit.get(field_name, default)
    return getattr(unit, field_name, default)


def _get_candidate_field(candidate: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(field_name, default)
    return getattr(candidate, field_name, default)


def embed_candidate_units(
    candidates: Sequence[Any],
    model_name: str = SEMANTIC_EMBEDDING_MODEL,
) -> dict[str, np.ndarray]:
    """
    Collects every unique unit_id referenced across all candidates'
    unit_a/unit_b, embeds each unit's raw_text exactly once in a single
    batched call, and returns a unit_id -> embedding lookup.

    Note: Embeds raw_text rather than normalized_text to retain casing and punctuation cues.
    """
    unique_units: dict[str, str] = {}
    for c in candidates:
        unit_a = _get_candidate_field(c, "unit_a")
        unit_b = _get_candidate_field(c, "unit_b")

        if unit_a is not None:
            uid_a = str(_get_unit_field(unit_a, "unit_id"))
            text_a = str(_get_unit_field(unit_a, "raw_text", ""))
            if uid_a and uid_a not in unique_units:
                unique_units[uid_a] = text_a

        if unit_b is not None:
            uid_b = str(_get_unit_field(unit_b, "unit_id"))
            text_b = str(_get_unit_field(unit_b, "raw_text", ""))
            if uid_b and uid_b not in unique_units:
                unique_units[uid_b] = text_b

    if not unique_units:
        return {}

    ids = list(unique_units.keys())
    texts = list(unique_units.values())

    model = get_model(model_name)
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False)

    return {uid: np.asarray(vec, dtype=np.float32) for uid, vec in zip(ids, vectors)}


def semantic_similarity_score(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Cosine similarity normalized from [-1, 1] to [0, 1] for consistency with lexical scores.
    """
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.5

    raw_cos = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
    # Clamp for floating point edge cases
    raw_cos = max(-1.0, min(1.0, raw_cos))
    return float((raw_cos + 1.0) / 2.0)


def score_candidate(
    candidate: Any,
    embeddings: dict[str, np.ndarray],
    weights: dict[str, float] = SEMANTIC_WEIGHTS,
    retake_threshold: float = STAGE2_RETAKE_THRESHOLD,
    ambiguous_floor: float = STAGE2_AMBIGUOUS_FLOOR,
    disagreement_threshold: float = DISAGREEMENT_FLAG_THRESHOLD,
) -> dict[str, Any]:
    """
    Calculates semantic similarity, combined Stage 2 score, disagreement flag, and classification.
    """
    unit_a = _get_candidate_field(candidate, "unit_a")
    unit_b = _get_candidate_field(candidate, "unit_b")

    uid_a = str(_get_unit_field(unit_a, "unit_id"))
    uid_b = str(_get_unit_field(unit_b, "unit_id"))

    if uid_a not in embeddings or uid_b not in embeddings:
        raise KeyError(f"Embeddings missing for candidate units: '{uid_a}' or '{uid_b}'")

    emb_a = embeddings[uid_a]
    emb_b = embeddings[uid_b]
    semantic_score = semantic_similarity_score(emb_a, emb_b)

    scores_obj = _get_candidate_field(candidate, "scores", {})
    if isinstance(scores_obj, dict):
        containment = float(scores_obj.get("containment", 0.0))
    else:
        containment = float(getattr(scores_obj, "containment", 0.0))

    combined_score = float(_get_candidate_field(candidate, "combined_score", 0.0))

    w_sem = weights.get("semantic_similarity", SEMANTIC_WEIGHTS["semantic_similarity"])
    w_lex = weights.get("lexical_combined_score", SEMANTIC_WEIGHTS["lexical_combined_score"])
    w_cont = weights.get("containment_bonus", SEMANTIC_WEIGHTS["containment_bonus"])

    stage2_score = (
        w_sem * semantic_score
        + w_lex * combined_score
        + w_cont * containment
    )

    disagreement = abs(combined_score - semantic_score)
    high_disagreement = disagreement > disagreement_threshold

    if stage2_score >= retake_threshold and not high_disagreement:
        classification = "likely_retake"
    elif stage2_score < ambiguous_floor and not high_disagreement:
        classification = "unlikely_retake"
    else:
        # Anything in the middle band OR any high-disagreement case is ambiguous
        classification = "ambiguous"

    return {
        "semantic_score": round(semantic_score, 4),
        "stage2_score": round(stage2_score, 4),
        "lexical_semantic_disagreement": round(disagreement, 4),
        "high_disagreement_flag": high_disagreement,
        "classification": classification,
    }


def _candidate_to_dict(c: Any) -> dict[str, Any]:
    if isinstance(c, dict):
        return dict(c)
    if hasattr(c, "to_dict") and callable(c.to_dict):
        return c.to_dict()
    if hasattr(c, "model_dump") and callable(c.model_dump):
        return c.model_dump()
    if is_dataclass(c):
        return asdict(c)
    raise TypeError(f"Cannot convert candidate of type {type(c)} to dict.")


def run_stage2(
    candidates: Sequence[Any],
    model_name: str = SEMANTIC_EMBEDDING_MODEL,
    weights: dict[str, float] = SEMANTIC_WEIGHTS,
    retake_threshold: float = STAGE2_RETAKE_THRESHOLD,
    ambiguous_floor: float = STAGE2_AMBIGUOUS_FLOOR,
    disagreement_threshold: float = DISAGREEMENT_FLAG_THRESHOLD,
) -> list[dict[str, Any]]:
    """
    Executes Stage 2 Semantic Matching on CandidatePairs produced by Stage 1.
    Preserves all Stage 1 fields and appends semantic scores and classification.
    """
    if not candidates:
        return []

    embeddings = embed_candidate_units(candidates, model_name=model_name)

    enriched_candidates: list[dict[str, Any]] = []
    for c in candidates:
        base_dict = _candidate_to_dict(c)
        semantic_info = score_candidate(
            c,
            embeddings=embeddings,
            weights=weights,
            retake_threshold=retake_threshold,
            ambiguous_floor=ambiguous_floor,
            disagreement_threshold=disagreement_threshold,
        )
        enriched_candidates.append({**base_dict, **semantic_info})

    return enriched_candidates
