from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, Union

from .config import ADJUDICATION_MODEL
from .schemas import AdjudicatedSpan

DEFAULT_ADJUDICATED_SPANS_PATH = Path("data") / "processed" / "candidates" / "adjudicated_spans.json"


def _serialize_span(s: Any) -> dict[str, Any]:
    if isinstance(s, dict):
        return dict(s)
    if hasattr(s, "to_dict") and callable(s.to_dict):
        return s.to_dict()
    if hasattr(s, "model_dump") and callable(s.model_dump):
        return s.model_dump()
    if is_dataclass(s):
        return asdict(s)
    raise TypeError(f"Cannot serialize adjudicated span of type {type(s)}")


def save_adjudicated_spans(
    spans: Sequence[Union[AdjudicatedSpan, dict[str, Any]]],
    output_path: Union[str, Path] = DEFAULT_ADJUDICATED_SPANS_PATH,
    source_spans: str = "data/processed/candidates/retake_spans.json",
    model: str = ADJUDICATION_MODEL,
) -> Path:
    """
    Saves adjudicated spans to JSON with audit metadata envelope.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    serialized_spans = [_serialize_span(s) for s in spans]

    resolved_count = sum(1 for s in serialized_spans if s.get("adjudication_status") == "resolved")
    not_required_count = sum(1 for s in serialized_spans if s.get("adjudication_status") == "not_required")
    failed_count = sum(1 for s in serialized_spans if s.get("adjudication_status") == "failed")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_spans": str(source_spans),
        "model": str(model),
        "span_count": len(serialized_spans),
        "resolved_count": resolved_count,
        "not_required_count": not_required_count,
        "failed_count": failed_count,
        "spans": serialized_spans,
    }

    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_p


def load_adjudicated_spans(
    input_path: Union[str, Path] = DEFAULT_ADJUDICATED_SPANS_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Loads adjudicated spans and metadata envelope from JSON.
    """
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Adjudicated spans file not found: {in_p}")

    data = json.loads(in_p.read_text(encoding="utf-8"))
    spans: list[dict[str, Any]] = data.get("spans", [])

    metadata = {
        "generated_at": data.get("generated_at"),
        "source_spans": data.get("source_spans"),
        "model": data.get("model"),
        "span_count": data.get("span_count", len(spans)),
        "resolved_count": data.get("resolved_count", 0),
        "not_required_count": data.get("not_required_count", 0),
        "failed_count": data.get("failed_count", 0),
    }

    return spans, metadata
