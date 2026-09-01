"""
Cutlist Builder — Converts semantic candidate pairs into an actionable cutlist.

Takes Stage 4 semantic matching results and produces:
1. Retake groups (consecutive candidate pairs forming a single retake event)
2. Discard spans (time ranges to remove from the video)
3. Keep spans (time ranges to preserve, in order)

Design decisions:
- Only `likely_retake` candidates are used for grouping (ambiguous = skip for safety)
- Later take is kept by default (speakers typically improve on retakes)
- Adjacent discard spans are merged with a small padding tolerance
"""

from __future__ import annotations

from typing import Any, Sequence

from .schemas import CutList, RetakeGroup, TimeSpan


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Candidates within this many seconds of each other are grouped together
GROUP_TEMPORAL_GAP_SECONDS = 5.0

# Padding in seconds added around discard spans to avoid cutting mid-word
CUT_PADDING_SECONDS = 0.05

# Merge tolerance: discard spans closer than this are merged into one
MERGE_TOLERANCE_SECONDS = 0.3


# ---------------------------------------------------------------------------
# Step 1: Filter candidates to only confirmed retakes
# ---------------------------------------------------------------------------


def filter_confirmed_retakes(
    candidates: Sequence[dict[str, Any]],
    classifications: Sequence[str] = ("likely_retake",),
) -> list[dict[str, Any]]:
    """
    Filter semantic candidates to only those with the specified classifications.
    Default: only 'likely_retake' candidates (safe for demo).
    """
    allowed = set(classifications)
    return [c for c in candidates if c.get("classification") in allowed]


# ---------------------------------------------------------------------------
# Step 2: Group consecutive candidate pairs into retake groups
# ---------------------------------------------------------------------------


def _unit_start(candidate: dict[str, Any], side: str) -> float:
    return float(candidate[side]["start"])


def _unit_end(candidate: dict[str, Any], side: str) -> float:
    return float(candidate[side]["end"])


def group_retake_pairs(
    candidates: list[dict[str, Any]],
    temporal_gap: float = GROUP_TEMPORAL_GAP_SECONDS,
) -> list[list[dict[str, Any]]]:
    """
    Group candidate pairs into retake groups based on temporal proximity.

    Two consecutive candidates belong to the same group if both their
    unit_a and unit_b timestamps are within `temporal_gap` of each other.
    Candidates are first sorted by unit_a start time.
    """
    if not candidates:
        return []

    # Sort by unit_a start time for consistent grouping
    sorted_cands = sorted(candidates, key=lambda c: _unit_start(c, "unit_a"))

    groups: list[list[dict[str, Any]]] = [[sorted_cands[0]]]

    for cand in sorted_cands[1:]:
        prev = groups[-1][-1]

        # Check temporal continuity on both sides
        a_gap = _unit_start(cand, "unit_a") - _unit_end(prev, "unit_a")
        b_gap = _unit_start(cand, "unit_b") - _unit_end(prev, "unit_b")

        if a_gap <= temporal_gap and b_gap <= temporal_gap:
            groups[-1].append(cand)
        else:
            groups.append([cand])

    return groups


# ---------------------------------------------------------------------------
# Step 3: For each retake group, select which take to keep
# ---------------------------------------------------------------------------


def select_take(
    group: list[dict[str, Any]],
    strategy: str = "keep_later",
) -> tuple[str, str]:
    """
    Decide which take to keep for a retake group.

    Strategies:
        - 'keep_later': Keep the chronologically later take (unit_b).
          This is the default because speakers typically improve on retakes.
        - 'keep_earlier': Keep the chronologically earlier take (unit_a).
        - 'keep_higher_confidence': Keep the take with higher avg ASR confidence.

    Returns:
        (kept_take, reason): e.g., ("b", "Later take kept (default heuristic)")
    """
    if strategy == "keep_later":
        return "b", "Later take kept (default heuristic — speakers improve on retakes)"

    if strategy == "keep_earlier":
        return "a", "Earlier take kept (user preference)"

    if strategy == "keep_higher_confidence":
        avg_conf_a = sum(
            float(c["unit_a"].get("confidence", 0.0) or 0.0) for c in group
        ) / len(group)
        avg_conf_b = sum(
            float(c["unit_b"].get("confidence", 0.0) or 0.0) for c in group
        ) / len(group)

        if avg_conf_b >= avg_conf_a:
            return "b", f"Higher ASR confidence: B={avg_conf_b:.3f} vs A={avg_conf_a:.3f}"
        else:
            return "a", f"Higher ASR confidence: A={avg_conf_a:.3f} vs B={avg_conf_b:.3f}"

    return "b", "Later take kept (fallback)"


# ---------------------------------------------------------------------------
# Step 4: Build RetakeGroup objects from grouped pairs
# ---------------------------------------------------------------------------


def build_retake_groups(
    grouped_pairs: list[list[dict[str, Any]]],
    strategy: str = "keep_later",
) -> list[RetakeGroup]:
    """Build structured RetakeGroup objects with take selection applied."""
    retake_groups: list[RetakeGroup] = []

    for group_id, group in enumerate(grouped_pairs, start=1):
        # Compute the full time span for each side
        take_a_span = TimeSpan(
            start=min(_unit_start(c, "unit_a") for c in group),
            end=max(_unit_end(c, "unit_a") for c in group),
            label="take_a",
        )
        take_b_span = TimeSpan(
            start=min(_unit_start(c, "unit_b") for c in group),
            end=max(_unit_end(c, "unit_b") for c in group),
            label="take_b",
        )

        kept_take, reason = select_take(group, strategy=strategy)
        discarded_take = "a" if kept_take == "b" else "b"

        discard_span = take_a_span if discarded_take == "a" else take_b_span
        keep_span = take_b_span if kept_take == "b" else take_a_span

        retake_groups.append(
            RetakeGroup(
                group_id=group_id,
                candidate_ids=[c.get("candidate_id", 0) for c in group],
                take_a_span=take_a_span,
                take_b_span=take_b_span,
                kept_take=kept_take,
                discarded_take=discarded_take,
                discard_span=TimeSpan(
                    start=discard_span.start,
                    end=discard_span.end,
                    label=f"discard_{discarded_take}",
                ),
                keep_span=TimeSpan(
                    start=keep_span.start,
                    end=keep_span.end,
                    label=f"keep_{kept_take}",
                ),
                reason=reason,
                pair_count=len(group),
            )
        )

    return retake_groups


# ---------------------------------------------------------------------------
# Step 5: Merge overlapping discard spans and compute keep spans
# ---------------------------------------------------------------------------


def merge_spans(
    spans: list[TimeSpan],
    tolerance: float = MERGE_TOLERANCE_SECONDS,
) -> list[TimeSpan]:
    """Merge overlapping or near-adjacent time spans into non-overlapping spans."""
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda s: s.start)
    merged: list[TimeSpan] = [sorted_spans[0]]

    for span in sorted_spans[1:]:
        if merged[-1].overlaps(span, tolerance=tolerance):
            merged[-1] = merged[-1].merge(span)
        else:
            merged.append(span)

    return merged


def compute_keep_spans(
    discard_spans: list[TimeSpan],
    total_duration: float,
    padding: float = CUT_PADDING_SECONDS,
) -> list[TimeSpan]:
    """
    Invert discard spans to produce ordered keep spans covering [0, total_duration].

    Adds padding around cut boundaries to avoid cutting mid-phoneme.
    """
    if not discard_spans:
        return [TimeSpan(start=0.0, end=total_duration, label="keep")]

    keep_spans: list[TimeSpan] = []
    cursor = 0.0

    for ds in sorted(discard_spans, key=lambda s: s.start):
        # Add padding: start discard slightly earlier, end slightly later
        cut_start = max(0.0, ds.start - padding)
        cut_end = min(total_duration, ds.end + padding)

        if cursor < cut_start:
            keep_spans.append(TimeSpan(
                start=round(cursor, 3),
                end=round(cut_start, 3),
                label="keep",
            ))
        cursor = cut_end

    # Add remaining segment after last discard
    if cursor < total_duration:
        keep_spans.append(TimeSpan(
            start=round(cursor, 3),
            end=round(total_duration, 3),
            label="keep",
        ))

    # Filter out tiny fragments (< 0.1s) that would cause FFmpeg issues
    keep_spans = [s for s in keep_spans if s.duration >= 0.1]

    return keep_spans


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_cutlist(
    candidates: Sequence[dict[str, Any]],
    source: str = "",
    source_duration: float = 0.0,
    strategy: str = "keep_later",
    classifications: Sequence[str] = ("likely_retake",),
    temporal_gap: float = GROUP_TEMPORAL_GAP_SECONDS,
    merge_tolerance: float = MERGE_TOLERANCE_SECONDS,
    cut_padding: float = CUT_PADDING_SECONDS,
) -> CutList:
    """
    End-to-end cutlist generation from Stage 4 semantic candidates.

    Pipeline:
        1. Filter to confirmed retakes
        2. Group consecutive pairs into retake groups
        3. Select which take to keep for each group
        4. Merge overlapping discard spans
        5. Compute keep spans (inverse of discard)

    Args:
        candidates: Stage 4 semantic candidate dicts (with 'classification' field).
        source: Source media filename.
        source_duration: Total duration of source media in seconds.
        strategy: Take selection strategy ('keep_later', 'keep_earlier', 'keep_higher_confidence').
        classifications: Which classifications to treat as confirmed retakes.
        temporal_gap: Max gap in seconds between pairs to group them.
        merge_tolerance: Tolerance for merging adjacent discard spans.
        cut_padding: Padding around cut boundaries.

    Returns:
        CutList with keep_spans, discard_spans, and retake_groups.
    """
    # Step 1: Filter
    confirmed = filter_confirmed_retakes(candidates, classifications=classifications)

    if not confirmed:
        # No retakes found — keep entire video
        return CutList(
            source=source,
            source_duration=source_duration,
            keep_spans=[TimeSpan(start=0.0, end=source_duration, label="keep")],
            discard_spans=[],
            retake_groups=[],
            output_duration=source_duration,
        )

    # Step 2: Group
    grouped = group_retake_pairs(confirmed, temporal_gap=temporal_gap)

    # Step 3: Build retake groups with take selection
    retake_groups = build_retake_groups(grouped, strategy=strategy)

    # Step 4: Collect and merge discard spans
    raw_discard_spans = [g.discard_span for g in retake_groups]
    merged_discards = merge_spans(raw_discard_spans, tolerance=merge_tolerance)

    # Step 5: Compute keep spans
    keep_spans = compute_keep_spans(
        merged_discards,
        total_duration=source_duration,
        padding=cut_padding,
    )

    return CutList(
        source=source,
        source_duration=source_duration,
        keep_spans=keep_spans,
        discard_spans=merged_discards,
        retake_groups=retake_groups,
    )
