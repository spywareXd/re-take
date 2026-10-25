"""
Apply USER decisions on intra-unit retake candidates to a cut list.

Candidates are suggestions (status="suggested"). Nothing is cut unless a decision says so:
    {"candidate_id": "...", "action": "accept"}                          -> discard [proposed_discard_start, proposed_discard_end]
    {"candidate_id": "...", "action": "adjust", "split_word_index": 5}   -> discard [first word start, start of word #5]
    {"candidate_id": "...", "action": "reject"}                          -> nothing
The existing merge / invert logic of cut_generator is reused (0.05 s merge tolerance, 0.10 s minimum
keep fragment), so the returned cut list stays valid. The input cut list is never mutated.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from models.retake import CutList, RetakeGroup, TimeSpan
from pipeline.step_03_token_matching.intra_candidates import IntraUnitCandidate

from .cut_generator import compute_keep_spans, merge_spans

logger = logging.getLogger(__name__)

MERGE_TOLERANCE_SECONDS = 0.05
MIN_FRAGMENT_SECONDS = 0.10
VALID_ACTIONS = {"accept", "reject", "adjust"}


def _span_for_decision(cand: IntraUnitCandidate, decision: dict[str, Any]) -> Optional[tuple[TimeSpan, str, str]]:
    """Discard span (+ discarded/kept text) for one decision, or None for 'reject'."""
    action = decision.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unknown action {action!r} for candidate {cand.candidate_id!r}; expected one of {sorted(VALID_ACTIONS)}")
    if action == "reject":
        return None

    if action == "accept":
        start, end = cand.proposed_discard_start, cand.proposed_discard_end
        discard, kept = cand.discard_text, cand.kept_text
    else:  # adjust: recompute from the chosen word using the word timestamps stored on the candidate
        idx = decision.get("split_word_index")
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 < idx < len(cand.words):
            raise ValueError(
                f"'adjust' for {cand.candidate_id!r} needs an integer split_word_index in 1..{len(cand.words) - 1}, got {idx!r}"
            )
        start, end = float(cand.words[0]["start"]), float(cand.words[idx]["start"])
        discard = " ".join(w["word"] for w in cand.words[:idx])
        kept = " ".join(w["word"] for w in cand.words[idx:])

    if end <= start:
        raise ValueError(f"Empty discard span for {cand.candidate_id!r}: [{start}, {end}]")
    return TimeSpan(start=start, end=end, label=f"intra_{cand.candidate_id}"), discard, kept


def apply_intra_decisions(
    cutlist: CutList,
    decisions: Sequence[dict[str, Any]],
    candidates: Sequence[IntraUnitCandidate],
    units: Optional[Sequence[Any]] = None,
) -> CutList:
    """
    Return a NEW CutList with the accepted/adjusted intra-unit discards added.

    Args:
        cutlist: existing cut list (not modified).
        decisions: [{candidate_id, action: accept|reject|adjust, split_word_index (adjust only)}].
        candidates: the suggested candidates; needed to resolve candidate_ids and their word timestamps.
        units: optional ComparableUnits; if given, a candidate whose unit_id is not among them is refused.

    Raises:
        ValueError: unknown candidate_id, unknown action, or an invalid adjust index. Nothing is guessed.
    """
    by_id = {c.candidate_id: c for c in candidates}
    unit_ids = {str(u.get("unit_id") if isinstance(u, dict) else getattr(u, "unit_id")) for u in units} if units else None

    new_discards: list[TimeSpan] = []
    new_groups: list[RetakeGroup] = []
    next_group_id = max((g.group_id for g in cutlist.retake_groups), default=0) + 1

    for decision in decisions:
        cid = decision.get("candidate_id")
        cand = by_id.get(cid)
        if cand is None:
            raise ValueError(f"Unknown candidate_id {cid!r}")
        if unit_ids is not None and cand.unit_id not in unit_ids:
            raise ValueError(f"Candidate {cid!r} refers to unit {cand.unit_id!r} which is not in the given units")
        resolved = _span_for_decision(cand, decision)
        if resolved is None:                       # reject: leave the candidate alone
            continue
        span, discard_text, kept_text = resolved
        new_discards.append(span)
        new_groups.append(
            RetakeGroup(
                group_id=next_group_id,
                candidate_ids=[],
                take_a_span=span,
                take_b_span=TimeSpan(start=span.end, end=cand.unit_end, label="kept_part"),
                kept_take="b",
                discarded_take="a",
                discard_span=span,
                keep_span=TimeSpan(start=span.end, end=cand.unit_end, label="kept_part"),
                reason=f"Intra-unit restart ({cand.rules_fired}) in unit {cand.unit_id}, user {decision['action']}ed",
                pair_count=0,
                discarded_text=discard_text,
                kept_text=kept_text,
            )
        )
        next_group_id += 1

    # copy existing discard spans, add the new ones, then reuse the existing merge / invert logic
    discards = [TimeSpan(s.start, s.end, s.label) for s in cutlist.discard_spans] + new_discards
    merged = merge_spans(discards, tolerance=MERGE_TOLERANCE_SECONDS)
    keep = compute_keep_spans(merged, source_duration=cutlist.source_duration, min_fragment_duration=MIN_FRAGMENT_SECONDS)

    return CutList(
        source=cutlist.source,
        source_duration=cutlist.source_duration,
        keep_spans=keep,
        discard_spans=merged,
        retake_groups=list(cutlist.retake_groups) + new_groups,
    )
