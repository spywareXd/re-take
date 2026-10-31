"""Saving and loading the sentence units (comparable_units.json)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, Union

from .config import PAUSE_SPLIT_THRESHOLD_SECONDS
from .schemas import ComparableUnit

DEFAULT_UNITS_PATH = Path("data") / "processed" / "units" / "comparable_units.json"


def save_comparable_units(
    units: Sequence[ComparableUnit],
    output_path: Union[str, Path] = DEFAULT_UNITS_PATH,
    source_transcript: str = "",
    pause_split_threshold_seconds: float = PAUSE_SPLIT_THRESHOLD_SECONDS,
) -> Path:
    """
    Saves ComparableUnits to JSON metadata envelope, omitting non-serializable token_set.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    serialized_units: list[dict[str, Any]] = []
    for u in units:
        serialized_units.append(
            {
                "unit_id": u.unit_id,
                "source_segment_id": u.source_segment_id,
                "start": u.start,
                "end": u.end,
                "speaker": u.speaker,
                "raw_text": u.raw_text,
                "normalized_text": u.normalized_text,
                "tokens": list(u.tokens),
                "confidence": u.confidence,
                "is_below_min_token_length": u.is_below_min_token_length,
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_transcript": str(source_transcript),
        "pause_split_threshold_seconds": float(pause_split_threshold_seconds),
        "unit_count": len(serialized_units),
        "units": serialized_units,
    }

    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_p


def load_comparable_units(
    input_path: Union[str, Path] = DEFAULT_UNITS_PATH,
) -> tuple[list[ComparableUnit], dict[str, Any]]:
    """
    Loads ComparableUnits and envelope metadata from JSON, reconstructing token_set as set(tokens).
    """
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Comparable units file not found: {in_p}")

    data = json.loads(in_p.read_text(encoding="utf-8"))
    raw_units = data.get("units", [])

    units: list[ComparableUnit] = []
    for item in raw_units:
        tokens = list(item.get("tokens", []))
        units.append(
            ComparableUnit(
                unit_id=str(item["unit_id"]),
                source_segment_id=int(item["source_segment_id"]),
                start=float(item["start"]),
                end=float(item["end"]),
                raw_text=str(item.get("raw_text", "")),
                normalized_text=str(item.get("normalized_text", "")),
                tokens=tokens,
                token_set=set(tokens),
                speaker=str(item.get("speaker", "speaker_0")),
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
                is_below_min_token_length=bool(item.get("is_below_min_token_length", False)),
            )
        )

    metadata = {
        "generated_at": data.get("generated_at"),
        "source_transcript": data.get("source_transcript"),
        "pause_split_threshold_seconds": data.get("pause_split_threshold_seconds"),
        "unit_count": data.get("unit_count", len(units)),
    }

    return units, metadata
