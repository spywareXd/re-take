"""JSON persistence for cutlist results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

from .schemas import CutList

DEFAULT_CUTLIST_PATH = Path("data") / "processed" / "cutlists" / "cutlist.json"


def save_cutlist(
    cutlist: CutList,
    output_path: Union[str, Path] = DEFAULT_CUTLIST_PATH,
    source_candidates: str = "",
) -> Path:
    """
    Saves a CutList to JSON with metadata envelope.

    Args:
        cutlist: The CutList to serialize.
        output_path: Destination path for the JSON file.
        source_candidates: Path to the source semantic candidates file.

    Returns:
        The resolved output path.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_candidates": str(source_candidates),
        **cutlist.to_dict(),
    }

    out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_p


def load_cutlist(
    input_path: Union[str, Path] = DEFAULT_CUTLIST_PATH,
) -> tuple[CutList, dict[str, Any]]:
    """
    Loads a CutList and envelope metadata from JSON.

    Returns:
        Tuple of (CutList, metadata_dict).
    """
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Cutlist file not found: {in_p}")

    data = json.loads(in_p.read_text(encoding="utf-8"))

    cutlist = CutList.from_dict(data)

    metadata = {
        "generated_at": data.get("generated_at"),
        "source_candidates": data.get("source_candidates"),
    }

    return cutlist, metadata
