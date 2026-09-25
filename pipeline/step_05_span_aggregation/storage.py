from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, Union

from .span_aggregation import RetakeSpan

DEFAULT_RETAKE_SPANS_PATH = Path("data") / "processed" / "candidates" / "retake_spans.json"


def _serialize_span(s: Any) -> dict[str, Any]:
    if isinstance(s, dict):
        return dict(s)
    if hasattr(s, "to_dict") and callable(s.to_dict):
        return s.to_dict()
    if hasattr(s, "model_dump") and callable(s.model_dump):
        return s.model_dump()
    if is_dataclass(s):
        return asdict(s)
    raise TypeError(f"Cannot serialize span of type {type(s)}")


def save_retake_spans(
    spans: Sequence[Union[RetakeSpan, dict[str, Any]]],
    output_path: Union[str, Path] = DEFAULT_RETAKE_SPANS_PATH,
    source_candidates: str = "data/processed/candidates/semantic_candidates.json",
) -> Path:
    """
    Saves RetakeSpans to JSON metadata envelope.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    serialized_spans = [_serialize_span(s) for s in spans]

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
) -> tuple[list[RetakeSpan], dict[str, Any]]:
    """
    Loads RetakeSpans and envelope metadata from JSON.
    """
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Retake spans file not found: {in_p}")

    data = json.loads(in_p.read_text(encoding="utf-8"))
    raw_spans = data.get("spans", [])

    spans: list[RetakeSpan] = []
    for item in raw_spans:
        spans.append(
            RetakeSpan(
                span_id=str(item["span_id"]),
                unit_ids=list(item["unit_ids"]),
                start=float(item["start"]),
                end=float(item["end"]),
                candidate_pairs=list(item.get("candidate_pairs", [])),
                classification_summary=dict(item.get("classification_summary", {})),
                needs_adjudication=bool(item.get("needs_adjudication", False)),
                likely_final_unit_id=item.get("likely_final_unit_id"),
            )
        )

    metadata = {
        "generated_at": data.get("generated_at"),
        "source_candidates": data.get("source_candidates"),
        "span_count": data.get("span_count", len(spans)),
    }

    return spans, metadata
