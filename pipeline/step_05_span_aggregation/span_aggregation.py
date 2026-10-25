from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

MAX_UNITS_BEFORE_WARNING = 4          # spans with more units than this are logged (long chains may merge separate retakes)


@dataclass(slots=True)
class RetakeSpan:
    """
    Represents a clustered group of transitively-connected retake units.
    """
    span_id: str
    unit_ids: list[str]
    start: float
    end: float
    candidate_pairs: list[dict[str, Any]]
    classification_summary: dict[str, int]
    needs_adjudication: bool
    likely_final_unit_id: str | None
    # --- final-scorer fields (added at the end so positional construction keeps working) ---
    needs_keeper: bool = True               # every span still needs the LLM to choose the keeper
    n_units: int = 0                        # number of units in the span (after sibling attachment)
    min_stacker_p: float | None = None      # weakest stacker probability among the span's edges
    decision_source: str = ""               # "stacker", "cross_encoder" or "mixed": who confirmed the span's pairs as retakes

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "unit_ids": list(self.unit_ids),
            "start": round(float(self.start), 4),
            "end": round(float(self.end), 4),
            "candidate_pairs": self.candidate_pairs,
            "classification_summary": dict(self.classification_summary),
            "needs_adjudication": bool(self.needs_adjudication),
            "likely_final_unit_id": self.likely_final_unit_id,
            "needs_keeper": bool(self.needs_keeper),
            "n_units": int(self.n_units),
            "min_stacker_p": None if self.min_stacker_p is None else round(float(self.min_stacker_p), 6),
            "decision_source": self.decision_source,
        }


class _UnionFind:
    """
    Standard disjoint-set (union-find) with path compression and rank optimization.
    """
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            if self.rank[root_a] < self.rank[root_b]:
                self.parent[root_a] = root_b
            elif self.rank[root_a] > self.rank[root_b]:
                self.parent[root_b] = root_a
            else:
                self.parent[root_b] = root_a
                self.rank[root_a] += 1


def _extract_unit_dict(unit: Any) -> dict[str, Any]:
    if isinstance(unit, dict):
        d = dict(unit)
    elif hasattr(unit, "to_dict") and callable(unit.to_dict):
        d = unit.to_dict()
    elif hasattr(unit, "model_dump") and callable(unit.model_dump):
        d = unit.model_dump()
    elif is_dataclass(unit):
        d = asdict(unit)
    else:
        d = {
            "unit_id": str(getattr(unit, "unit_id", "")),
            "source_segment_id": getattr(unit, "source_segment_id", getattr(unit, "segment_id", None)),
            "start": float(getattr(unit, "start", 0.0)),
            "end": float(getattr(unit, "end", 0.0)),
            "raw_text": str(getattr(unit, "raw_text", "")),
            "is_below_min_token_length": bool(getattr(unit, "is_below_min_token_length", False)),
        }

    # Ensure source_segment_id is populated
    if d.get("source_segment_id") is None and d.get("segment_id") is not None:
        d["source_segment_id"] = d["segment_id"]
    elif d.get("source_segment_id") is None and ":" in str(d.get("unit_id", "")):
        try:
            d["source_segment_id"] = int(str(d["unit_id"]).split(":")[0])
        except ValueError:
            pass

    return d


def _extract_candidate_dict(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return candidate
    if hasattr(candidate, "to_dict") and callable(candidate.to_dict):
        return candidate.to_dict()
    if hasattr(candidate, "model_dump") and callable(candidate.model_dump):
        return candidate.model_dump()
    if is_dataclass(candidate):
        return asdict(candidate)
    raise TypeError(f"Cannot convert candidate of type {type(candidate)} to dict")


def _determine_likely_final_unit_id(
    unit_ids: list[str],
    unit_map: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    classification_summary: dict[str, int],
) -> str | None:
    """
    Advisory-only heuristic suggestion for the final/most complete take:
    - If the span is entirely 'ambiguous' (no likely_retake pairs), return None.
    - Exclude units that are marked as the smaller/incomplete side of a containment_direction.
    - Exclude units marked as short fragments (is_below_min_token_length).
    - From the remaining non-subset units, pick the chronologically latest unit.
    - If signals conflict or all units are subsets, return None.
    """
    likely_retake_count = classification_summary.get("likely_retake", 0)
    if likely_retake_count == 0:
        return None

    subset_units: set[str] = set()
    for p in pairs:
        direction = p.get("containment_direction")
        uid_a = str(p["unit_a"]["unit_id"])
        uid_b = str(p["unit_b"]["unit_id"])

        if direction == "a_subset_of_b":
            subset_units.add(uid_a)
        elif direction == "b_subset_of_a":
            subset_units.add(uid_b)

    # Exclude subset units and short fragments from being chosen as the final keeper
    non_subset_candidates = [
        uid for uid in unit_ids
        if uid not in subset_units and not unit_map.get(uid, {}).get("is_below_min_token_length", False)
    ]

    if not non_subset_candidates:
        return None

    # Pick the latest non-subset unit chronologically
    return max(non_subset_candidates, key=lambda uid: (unit_map[uid]["start"], unit_map[uid]["end"]))


def _attach_sibling_short_units(
    raw_spans: list[tuple[float, RetakeSpan]],
    unit_map: dict[str, dict[str, Any]],
    all_units: Sequence[Any] | None = None,
) -> None:
    """
    Second Pass:
    For every unit marked is_below_min_token_length: true in all_units, check whether it shares
    a source_segment_id with a unit that ended up in a span. If so, attach it to that
    span as a 'silent member' (included in unit_ids and span start/end bounds, but not
    altering candidate_pairs, classification_summary, or needs_adjudication).
    """
    if not all_units:
        return

    units_to_check = [_extract_unit_dict(u) for u in all_units]
    if not units_to_check:
        return

    # Index short units by source_segment_id
    short_units_by_seg: dict[int, list[dict[str, Any]]] = {}
    for u in units_to_check:
        if u.get("is_below_min_token_length", False):
            seg_id = u.get("source_segment_id")
            if seg_id is not None:
                short_units_by_seg.setdefault(int(seg_id), []).append(u)

    if not short_units_by_seg:
        return

    # Attach short sibling units to respective spans
    for span_idx in range(len(raw_spans)):
        _, span = raw_spans[span_idx]

        # Find all segment IDs in this span
        span_seg_ids: set[int] = set()
        for uid in span.unit_ids:
            if uid in unit_map and unit_map[uid].get("source_segment_id") is not None:
                span_seg_ids.add(int(unit_map[uid]["source_segment_id"]))
            elif ":" in uid:
                try:
                    span_seg_ids.add(int(uid.split(":")[0]))
                except ValueError:
                    pass

        # Check for sibling short units sharing these segment IDs
        attached_any = False
        for seg_id in span_seg_ids:
            for short_u in short_units_by_seg.get(seg_id, []):
                short_uid = str(short_u["unit_id"])
                if short_uid not in span.unit_ids:
                    span.unit_ids.append(short_uid)
                    unit_map[short_uid] = short_u
                    attached_any = True

        if attached_any:
            # Re-sort unit_ids chronologically and update span bounds
            span.unit_ids.sort(key=lambda uid: (unit_map[uid]["start"], unit_map[uid]["end"], uid))
            span.start = min(unit_map[uid]["start"] for uid in span.unit_ids)
            span.end = max(unit_map[uid]["end"] for uid in span.unit_ids)
            raw_spans[span_idx] = (span.start, span)


def build_retake_spans(
    candidates: Sequence[Any],
    all_units: Sequence[Any] | None = None,
) -> list[RetakeSpan]:
    """
    Clusters overlapping and transitively connected CandidatePairs into RetakeSpans.

    Final-scorer behaviour: ONLY pairs whose final decision (`class`) is "retake" are used; pairs decided "not_retake" are never unioned.
    Every span gets needs_adjudication=False (it is already confirmed as repeated attempts) and needs_keeper=True (the LLM only picks the keeper).

    Args:
        candidates: Sequence of Step 04 candidate pair dicts or objects.
        all_units: Optional sequence of all ComparableUnits from the transcript.
                   Used in the second pass to attach silent short-unit fragments.

    Returns:
        List of RetakeSpan instances representing grouped retake clusters.
    """
    if not candidates:
        return []

    # 1. Filter: only pairs whose FINAL decision is "retake" (set by Stage 4 after the stacker / cross-encoder resolution)
    surviving_pairs: list[dict[str, Any]] = []
    for c in candidates:
        c_dict = _extract_candidate_dict(c)
        if c_dict.get("class") == "retake":
            surviving_pairs.append(c_dict)

    if not surviving_pairs:
        return []

    # 2. Collect unit info & perform Union-Find clustering
    uf = _UnionFind()
    unit_map: dict[str, dict[str, Any]] = {}

    for pair in surviving_pairs:
        unit_a = _extract_unit_dict(pair["unit_a"])
        unit_b = _extract_unit_dict(pair["unit_b"])

        uid_a = str(unit_a["unit_id"])
        uid_b = str(unit_b["unit_id"])

        if uid_a not in unit_map:
            unit_map[uid_a] = unit_a
        if uid_b not in unit_map:
            unit_map[uid_b] = unit_b

        uf.union(uid_a, uid_b)

    # 3. Group units and candidate pairs by connected component root
    component_units: dict[str, set[str]] = {}
    component_pairs: dict[str, list[dict[str, Any]]] = {}

    for uid in unit_map:
        root = uf.find(uid)
        if root not in component_units:
            component_units[root] = set()
            component_pairs[root] = []
        component_units[root].add(uid)

    for pair in surviving_pairs:
        uid_a = str(pair["unit_a"]["unit_id"])
        root = uf.find(uid_a)
        component_pairs[root].append(pair)

    # 4. Build RetakeSpan objects
    raw_spans: list[tuple[float, RetakeSpan]] = []

    for root, uids in component_units.items():
        pairs = component_pairs[root]

        # Sort unit_ids chronologically
        sorted_uids = sorted(
            uids,
            key=lambda uid: (unit_map[uid]["start"], unit_map[uid]["end"], uid),
        )

        span_start = min(unit_map[uid]["start"] for uid in sorted_uids)
        span_end = max(unit_map[uid]["end"] for uid in sorted_uids)

        # every edge is a confirmed retake. The key "likely_retake" keeps the likely-final-unit heuristic below unchanged; the extra keys
        # record who confirmed the edges.
        classification_summary = {"likely_retake": len(pairs), "ambiguous": 0,
                                  "resolved_by_stacker": sum(1 for p in pairs if p.get("resolved_by") == "stacker"),
                                  "resolved_by_cross_encoder": sum(1 for p in pairs if p.get("resolved_by") == "cross_encoder")}
        needs_adjudication = False                        # nothing is left to adjudicate: the span is already a confirmed retake
        sources = {p.get("resolved_by") for p in pairs}
        decision_source = sources.pop() if len(sources) == 1 else "mixed"
        ps = [float(p["stacker_p"]) for p in pairs if p.get("stacker_p") is not None]

        likely_final_unit_id = _determine_likely_final_unit_id(
            sorted_uids,
            unit_map=unit_map,
            pairs=pairs,
            classification_summary=classification_summary,
        )

        span = RetakeSpan(
            span_id="",  # Assigned after sorting all spans
            unit_ids=sorted_uids,
            start=span_start,
            end=span_end,
            candidate_pairs=pairs,
            classification_summary=classification_summary,
            needs_adjudication=needs_adjudication,
            likely_final_unit_id=likely_final_unit_id,
            needs_keeper=True,
            min_stacker_p=min(ps) if ps else None,
            decision_source=decision_source,
        )
        raw_spans.append((span_start, span))

    # 5. Second Pass: Attach short sibling units sharing the same source_segment_id
    _attach_sibling_short_units(raw_spans, unit_map=unit_map, all_units=all_units)

    # Sort spans chronologically by start timestamp
    raw_spans.sort(key=lambda item: (item[0], item[1].end))

    final_spans: list[RetakeSpan] = []
    for idx, (_, span) in enumerate(raw_spans):
        span.span_id = f"span_{idx}"
        span.n_units = len(span.unit_ids)                 # counted after the sibling-attachment pass
        if span.n_units > MAX_UNITS_BEFORE_WARNING:
            logger.warning("[Stage 5] %s has %d units (> %d); a long chain may merge separate retakes (min stacker_p %.3f)",
                           span.span_id, span.n_units, MAX_UNITS_BEFORE_WARNING, span.min_stacker_p if span.min_stacker_p is not None else float("nan"))
        final_spans.append(span)

    return final_spans
