from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, Union


DEFAULT_SEMANTIC_CANDIDATES_PATH = Path("data") / "processed" / "candidates" / "semantic_candidates.json"


def _serialize_candidate(c: Any) -> dict[str, Any]:
    if isinstance(c, dict):
        return dict(c)
    if hasattr(c, "to_dict") and callable(c.to_dict):
        return c.to_dict()
    if hasattr(c, "model_dump") and callable(c.model_dump):
        return c.model_dump()
    if is_dataclass(c):
        return asdict(c)
    raise TypeError(f"Cannot serialize candidate of type {type(c)}")


def save_semantic_candidates(
    candidates: Sequence[Any],
    output_path: Union[str, Path] = DEFAULT_SEMANTIC_CANDIDATES_PATH,
    source_candidates: str = "data/processed/candidates/token_candidates.json",
    embedding_model: str = "all-MiniLM-L6-v2",
    semantic_weights: dict[str, float] | None = None,
    stage2_retake_threshold: float = 0.65,
    stage2_ambiguous_floor: float = 0.35,
    disagreement_flag_threshold: float = 0.35,
    extra_envelope: dict[str, Any] | None = None,
) -> Path:
    """
    Saves Stage 2 Semantic Candidates to JSON metadata envelope.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if semantic_weights is None:
        semantic_weights = {
            "semantic_similarity": 0.50,
            "lexical_combined_score": 0.30,
            "containment_bonus": 0.20,
        }

    serialized_candidates = [_serialize_candidate(c) for c in candidates]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_candidates": str(source_candidates),
        "embedding_model": str(embedding_model),
        "semantic_weights": dict(semantic_weights),
        "stage2_retake_threshold": float(stage2_retake_threshold),
        "stage2_ambiguous_floor": float(stage2_ambiguous_floor),
        "disagreement_flag_threshold": float(disagreement_flag_threshold),
        "candidate_count": len(serialized_candidates),
        **(extra_envelope or {}),                 # Stage 4 scorer fields (scorer_name, thresholds_file, model_version, feature_version, label_version, thresholds)
        "candidates": serialized_candidates,
    }

    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_p


def load_semantic_candidates(
    input_path: Union[str, Path] = DEFAULT_SEMANTIC_CANDIDATES_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Loads Stage 2 Semantic Candidates and envelope metadata from JSON.
    """
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Semantic candidates file not found: {in_p}")

    data = json.loads(in_p.read_text(encoding="utf-8"))
    candidates = list(data.get("candidates", []))

    metadata = {
        "generated_at": data.get("generated_at"),
        "source_candidates": data.get("source_candidates"),
        "embedding_model": data.get("embedding_model"),
        "semantic_weights": data.get("semantic_weights"),
        "stage2_retake_threshold": data.get("stage2_retake_threshold"),
        "stage2_ambiguous_floor": data.get("stage2_ambiguous_floor"),
        "disagreement_flag_threshold": data.get("disagreement_flag_threshold"),
        "candidate_count": data.get("candidate_count", len(candidates)),
        **{k: data.get(k) for k in ("scorer_name", "thresholds_file", "model_version", "feature_version", "label_version", "thresholds") if k in data},
    }

    return candidates, metadata


DEFAULT_RETAKE_SPANS_PATH = Path("data") / "processed" / "candidates" / "retake_spans.json"


def save_retake_spans(
    spans: Sequence[Any],
    output_path: Union[str, Path] = DEFAULT_RETAKE_SPANS_PATH,
    source_candidates: str = "data/processed/candidates/semantic_candidates.json",
) -> Path:
    """
    Saves RetakeSpans to JSON metadata envelope.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    serialized_spans = [_serialize_candidate(s) for s in spans]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_candidates": str(source_candidates),
        "span_count": len(serialized_spans),
        "spans": serialized_spans,
    }

    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_p


def load_retake_spans(
    input_path: Union[str, Path] = DEFAULT_RETAKE_SPANS_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Loads RetakeSpans and envelope metadata from JSON.
    """
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Retake spans file not found: {in_p}")

    data = json.loads(in_p.read_text(encoding="utf-8"))
    spans = list(data.get("spans", []))

    metadata = {
        "generated_at": data.get("generated_at"),
        "source_candidates": data.get("source_candidates"),
        "span_count": data.get("span_count", len(spans)),
    }

    return spans, metadata

