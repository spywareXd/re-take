from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Sequence


def _extract_unit_dict(unit: Any) -> dict[str, Any]:
    if isinstance(unit, dict):
        return dict(unit)
    if hasattr(unit, "to_dict") and callable(unit.to_dict):
        return unit.to_dict()
    if hasattr(unit, "model_dump") and callable(unit.model_dump):
        return unit.model_dump()
    if is_dataclass(unit):
        return asdict(unit)
    return {
        "unit_id": str(getattr(unit, "unit_id", "")),
        "start": float(getattr(unit, "start", 0.0)),
        "end": float(getattr(unit, "end", 0.0)),
        "raw_text": str(getattr(unit, "raw_text", "")),
        "is_below_min_token_length": bool(getattr(unit, "is_below_min_token_length", False)),
    }


def _extract_span_dict(span: Any) -> dict[str, Any]:
    if isinstance(span, dict):
        return dict(span)
    if hasattr(span, "to_dict") and callable(span.to_dict):
        return span.to_dict()
    if hasattr(span, "model_dump") and callable(span.model_dump):
        return span.model_dump()
    if is_dataclass(span):
        return asdict(span)
    return {
        "span_id": str(getattr(span, "span_id", "")),
        "unit_ids": list(getattr(span, "unit_ids", [])),
        "start": float(getattr(span, "start", 0.0)),
        "end": float(getattr(span, "end", 0.0)),
        "candidate_pairs": list(getattr(span, "candidate_pairs", [])),
        "classification_summary": dict(getattr(span, "classification_summary", {})),
        "needs_adjudication": bool(getattr(span, "needs_adjudication", False)),
        "likely_final_unit_id": getattr(span, "likely_final_unit_id", None),
    }


def build_adjudication_prompt(
    span: Any,
    all_units_by_id: dict[str, Any] | Sequence[Any] | None = None,
) -> str:
    """
    Builds a comprehensive LLM adjudication prompt for a single ambiguous retake span.

    Args:
        span: RetakeSpan instance or dict.
        all_units_by_id: Mapping from unit_id to ComparableUnit, or list of units.

    Returns:
        Formatted prompt string instructing the LLM to return structured JSON.
    """
    s_dict = _extract_span_dict(span)

    # Normalize unit lookup map
    unit_map: dict[str, dict[str, Any]] = {}
    if isinstance(all_units_by_id, dict):
        for k, v in all_units_by_id.items():
            unit_map[str(k)] = _extract_unit_dict(v)
    elif isinstance(all_units_by_id, (list, tuple)):
        for u in all_units_by_id:
            d = _extract_unit_dict(u)
            unit_map[str(d["unit_id"])] = d

    # Also harvest unit data from candidate_pairs inside the span if available
    for p in s_dict.get("candidate_pairs", []):
        for key in ("unit_a", "unit_b"):
            if key in p:
                u_data = _extract_unit_dict(p[key])
                uid = str(u_data.get("unit_id", ""))
                if uid and uid not in unit_map:
                    unit_map[uid] = u_data

    # Collect member units in chronological order
    member_uids = s_dict.get("unit_ids", [])
    units_info: list[dict[str, Any]] = []
    for uid in member_uids:
        u_info = unit_map.get(uid, {
            "unit_id": uid,
            "start": s_dict.get("start", 0.0),
            "end": s_dict.get("end", 0.0),
            "raw_text": f"[Unit {uid}]",
        })
        units_info.append(u_info)

    # Sort units chronologically by start timestamp
    units_info.sort(key=lambda u: (u.get("start", 0.0), u.get("end", 0.0), str(u.get("unit_id", ""))))

    # Format spoken units section
    units_formatted_lines = []
    for u in units_info:
        uid = u.get("unit_id", "")
        start_t = u.get("start", 0.0)
        end_t = u.get("end", 0.0)
        raw_text = u.get("raw_text", "")
        short_flag = " (Short fragment/stutter)" if u.get("is_below_min_token_length") else ""
        units_formatted_lines.append(f'  • [{uid}] ({start_t:.2f}s -> {end_t:.2f}s){short_flag}: "{raw_text}"')

    units_section = "\n".join(units_formatted_lines)

    # Format pairwise scoring evidence section
    pairs = s_dict.get("candidate_pairs", [])
    pairs_formatted_lines = []
    if pairs:
        for idx, p in enumerate(pairs, 1):
            uid_a = p.get("unit_a", {}).get("unit_id", "?")
            uid_b = p.get("unit_b", {}).get("unit_id", "?")
            sem = p.get("semantic_score", 0.0)
            lex = p.get("lexical_score", p.get("combined_score", 0.0))
            plain = p.get("plain_stage2_score", 0.0)
            prob = p.get("stacker_p")
            direction = p.get("containment_direction", "none")
            prob_str = f", Stacker probability = {prob:.3f}" if prob is not None else ""

            pairs_formatted_lines.append(
                f'  {idx}. Pair [{uid_a}] <-> [{uid_b}]:\n'
                f'     - Scores: Plain Stage 2 = {plain:.3f}, Semantic Similarity = {sem:.3f}, Lexical Score = {lex:.3f}{prob_str}\n'
                f'     - Containment Direction: {direction}'
            )
    else:
        pairs_formatted_lines.append("  (No pairwise candidate details available)")

    pairs_section = "\n".join(pairs_formatted_lines)

    span_id = s_dict.get("span_id", "span")
    span_start = s_dict.get("start", 0.0)
    span_end = s_dict.get("end", 0.0)

    prompt = f"""You are an expert video editing and speech transcript assistant.
A scoring model has ALREADY confirmed that the spoken units below are repeated attempts at the same intended line (a retake group): the speaker restarted, stumbled or re-said the same sentence/thought. Do NOT judge whether this is a retake. Your only task is to choose which single unit to KEEP as the final take and which units to REMOVE.

======================================================================
SPAN CONTEXT
======================================================================
Span ID: {span_id} (Time range: {span_start:.2f}s to {span_end:.2f}s)
Member Unit IDs: {json.dumps(member_uids)}

SPOKEN UNITS (IN CHRONOLOGICAL ORDER):
{units_section}

SUPPORTING EVIDENCE (SCORES FROM THE SCORING MODEL):
{pairs_section}

======================================================================
INSTRUCTIONS
======================================================================
1. KEEPER SELECTION:
   - Read the raw text of each spoken unit in chronological order.
   - Select exactly ONE unit from Member Unit IDs as "keeper_unit_id" (typically the final, most complete, fluent, and correct delivery).

2. REMOVALS:
   - Set "units_to_remove" to include all OTHER member unit IDs in the span (the keeper must NOT appear in it).
   - Short fragments/stutters (e.g. "Fellas,") belong in "units_to_remove" if their full sentence attempt succeeded later.

3. CONFIDENCE & REASONING:
   - Set "confidence" to "high", "medium", or "low".
   - Provide 1 to 2 concise sentences in "reasoning" explaining why this unit is the best final take.

======================================================================
OUTPUT FORMAT CONTRACT
======================================================================
Return ONLY a valid JSON object matching this exact schema:
{{
  "keeper_unit_id": "<unit_id>",
  "units_to_remove": ["<unit_id>", ...],
  "confidence": "high" | "medium" | "low",
  "reasoning": "<1-2 sentence explanation>"
}}"""

    return prompt
