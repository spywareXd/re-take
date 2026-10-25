from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence, Union

from models.retake import CutList, RetakeGroup, TimeSpan

logger = logging.getLogger(__name__)

DEFAULT_CUTLIST_PATH = Path("data") / "processed" / "cutlists" / "cutlist.json"


def _extract_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "to_dict") and callable(item.to_dict):
        return item.to_dict()
    if hasattr(item, "model_dump") and callable(item.model_dump):
        return item.model_dump()
    if is_dataclass(item):
        return asdict(item)
    return {
        "unit_id": str(getattr(item, "unit_id", "")),
        "start": float(getattr(item, "start", 0.0)),
        "end": float(getattr(item, "end", 0.0)),
        "raw_text": str(getattr(item, "raw_text", "")),
    }


def merge_spans(spans: list[TimeSpan], tolerance: float = 0.05) -> list[TimeSpan]:
    """
    Sorts and merges overlapping or adjacent TimeSpans within tolerance.
    """
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[TimeSpan] = [sorted_spans[0]]

    for current in sorted_spans[1:]:
        prev = merged[-1]
        if prev.overlaps(current, tolerance=tolerance):
            merged[-1] = prev.merge(current)
        else:
            merged.append(current)

    return merged


def compute_keep_spans(
    discard_spans: list[TimeSpan],
    source_duration: float,
    min_fragment_duration: float = 0.10,
) -> list[TimeSpan]:
    """
    Inverts discard spans against [0.0, source_duration] to compute keep spans.
    """
    if not discard_spans:
        return [TimeSpan(start=0.0, end=round(source_duration, 4), label="keep_all")]

    merged_discards = merge_spans(discard_spans, tolerance=0.05)
    keep_spans: list[TimeSpan] = []
    current_time = 0.0

    for d in merged_discards:
        if d.start > current_time:
            seg_dur = d.start - current_time
            if seg_dur >= min_fragment_duration:
                keep_spans.append(
                    TimeSpan(
                        start=round(current_time, 4),
                        end=round(d.start, 4),
                        label="keep",
                    )
                )
        current_time = max(current_time, d.end)

    if current_time < source_duration:
        seg_dur = source_duration - current_time
        if seg_dur >= min_fragment_duration:
            keep_spans.append(
                TimeSpan(
                    start=round(current_time, 4),
                    end=round(source_duration, 4),
                    label="keep",
                )
            )

    return keep_spans


def generate_cuts_from_adjudicated_spans(
    adjudicated_spans: Sequence[Any],
    units: Sequence[Any],
    source: str,
    source_duration: float,
    merge_tolerance: float = 0.05,
    min_fragment_duration: float = 0.10,
) -> CutList:
    """
    Generates a CutList by directly consuming Step 06 adjudicated spans.

    Args:
        adjudicated_spans: Spans from Step 06 run_stage6().
        units: ComparableUnits extracted in Step 02/03.
        source: Source video filename/identifier.
        source_duration: Total media duration in seconds.
        merge_tolerance: Seconds tolerance for merging discard spans.
        min_fragment_duration: Minimum duration for kept video fragments.

    Returns:
        A CutList object ready for FFmpeg video editing.
    """
    unit_map: dict[str, dict[str, Any]] = {}
    for u in units:
        d = _extract_dict(u)
        uid = str(d.get("unit_id", ""))
        if uid:
            unit_map[uid] = d

    raw_discards: list[TimeSpan] = []
    retake_groups: list[RetakeGroup] = []

    for idx, raw_span in enumerate(adjudicated_spans):
        span = _extract_dict(raw_span)
        is_retake = span.get("is_genuine_retake_group")
        keeper_id = span.get("keeper_unit_id")
        bad_uids = span.get("units_to_remove", [])

        # Only spans the LLM RESOLVED (status "resolved", a keeper and a non-empty removal list) are cut. Unresolved spans are skipped with a log line.
        if span.get("adjudication_status") != "resolved":
            logger.warning("[Cutlist] span %s skipped: adjudication status %r (unresolved), nothing is cut", span.get("span_id"), span.get("adjudication_status"))
            continue
        if is_retake is not False and bad_uids:
            keeper_unit = unit_map.get(str(keeper_id)) if keeper_id else None
            keeper_span = (
                TimeSpan(start=keeper_unit["start"], end=keeper_unit["end"], label=f"keeper_{keeper_id}")
                if keeper_unit
                else TimeSpan(start=span.get("start", 0.0), end=span.get("end", 0.0), label="keeper")
            )

            discard_spans_in_group: list[TimeSpan] = []
            discarded_texts_list: list[str] = []
            for b_id in bad_uids:
                b_unit = unit_map.get(str(b_id))
                if b_unit:
                    ts = TimeSpan(start=b_unit["start"], end=b_unit["end"], label=f"cut_{b_id}")
                    raw_discards.append(ts)
                    discard_spans_in_group.append(ts)
                    if b_unit.get("raw_text"):
                        discarded_texts_list.append(b_unit["raw_text"])

            if discard_spans_in_group:
                min_start = min(s.start for s in discard_spans_in_group)
                max_end = max(s.end for s in discard_spans_in_group)
                primary_discard = TimeSpan(start=min_start, end=max_end, label=f"discard_{bad_uids}")
            else:
                primary_discard = TimeSpan(0, 0)

            candidate_ids = [p.get("candidate_id", 0) for p in span.get("candidate_pairs", [])]
            reason = str(span.get("reasoning", ""))
            if not reason and keeper_id:
                reason = f"Keeper take: [{keeper_id}] | Removed take(s): {bad_uids}"

            kept_text = keeper_unit.get("raw_text", "") if keeper_unit else ""
            discarded_text = " ".join(discarded_texts_list)

            retake_groups.append(
                RetakeGroup(
                    group_id=idx + 1,
                    candidate_ids=candidate_ids,
                    take_a_span=primary_discard,
                    take_b_span=keeper_span,
                    kept_take="b",
                    discarded_take="a",
                    discard_span=primary_discard,
                    keep_span=keeper_span,
                    reason=reason,
                    pair_count=len(span.get("candidate_pairs", [])),
                    discarded_text=discarded_text,
                    kept_text=kept_text,
                )
            )

    merged_discards = merge_spans(raw_discards, tolerance=merge_tolerance)
    keep_spans = compute_keep_spans(
        merged_discards,
        source_duration=source_duration,
        min_fragment_duration=min_fragment_duration,
    )

    return CutList(
        source=str(source),
        source_duration=round(float(source_duration), 4),
        keep_spans=keep_spans,
        discard_spans=merged_discards,
        retake_groups=retake_groups,
    )


def save_cutlist(
    cutlist: CutList,
    output_path: Union[str, Path] = DEFAULT_CUTLIST_PATH,
) -> Path:
    """Save a CutList to JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(cutlist.to_dict(), indent=2), encoding="utf-8")
    return out_p


def load_cutlist(
    input_path: Union[str, Path] = DEFAULT_CUTLIST_PATH,
) -> tuple[CutList, dict[str, Any]]:
    """Load a CutList from JSON."""
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"CutList file not found: {in_p}")
    data = json.loads(in_p.read_text(encoding="utf-8"))
    cutlist = CutList.from_dict(data)
    return cutlist, data
