from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Sequence

from .config import AdjudicationConfig, default_adjudication_config
from .llm_client import LLMClient, get_llm_client
from .prompt_builder import build_adjudication_prompt

logger = logging.getLogger(__name__)


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


def parse_adjudication_response(raw_response: dict[str, Any] | str, span: dict[str, Any]) -> dict[str, Any]:
    """
    Validates the LLM response against the span it was asked about:
    - keeper_unit_id (if not null) must be one of span.unit_ids
    - units_to_remove must all be members of span.unit_ids
    - keeper_unit_id must never appear in units_to_remove
    - if is_genuine_retake_group is False, keeper_unit_id must be null and
      units_to_remove must be empty — reject/retry if the model contradicts itself here.
    Raises ValueError on validation failure.
    """
    if isinstance(raw_response, str):
        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse LLM response as JSON: {exc}") from exc
    elif isinstance(raw_response, dict):
        data = dict(raw_response)
    else:
        raise TypeError(f"Expected dict or JSON string, got {type(raw_response)}")

    # Check required fields
    if "is_genuine_retake_group" not in data:
        raise ValueError("Missing required field 'is_genuine_retake_group' in LLM response")

    is_genuine = data["is_genuine_retake_group"]
    if not isinstance(is_genuine, bool):
        if str(is_genuine).lower() in {"true", "1"}:
            is_genuine = True
        elif str(is_genuine).lower() in {"false", "0"}:
            is_genuine = False
        else:
            raise ValueError(f"Invalid boolean value for 'is_genuine_retake_group': {is_genuine}")

    keeper_unit_id = data.get("keeper_unit_id")
    if keeper_unit_id is not None:
        keeper_unit_id = str(keeper_unit_id).strip()
        if keeper_unit_id.lower() in {"null", "none", ""}:
            keeper_unit_id = None

    raw_units_to_remove = data.get("units_to_remove", [])
    if not isinstance(raw_units_to_remove, list):
        raise ValueError(f"'units_to_remove' must be a list, got {type(raw_units_to_remove)}")
    units_to_remove = [str(uid).strip() for uid in raw_units_to_remove]

    span_unit_ids = [str(uid) for uid in span.get("unit_ids", [])]

    # Validate based on genuine retake flag
    if not is_genuine:
        # Self-contradiction checks
        if keeper_unit_id is not None:
            raise ValueError(
                f"Self-contradictory LLM response: is_genuine_retake_group is False, but keeper_unit_id is '{keeper_unit_id}' (must be null)"
            )
        if len(units_to_remove) > 0:
            raise ValueError(
                f"Self-contradictory LLM response: is_genuine_retake_group is False, but units_to_remove is {units_to_remove} (must be empty [])"
            )
    else:
        # Must select a keeper
        if keeper_unit_id is None:
            raise ValueError("is_genuine_retake_group is True, but keeper_unit_id is null/missing")

        # Keeper must belong to span
        if keeper_unit_id not in span_unit_ids:
            raise ValueError(
                f"Hallucinated keeper_unit_id '{keeper_unit_id}' is not in span unit_ids {span_unit_ids}"
            )

        # Removed units must belong to span
        invalid_removed = [uid for uid in units_to_remove if uid not in span_unit_ids]
        if invalid_removed:
            raise ValueError(
                f"Hallucinated units_to_remove {invalid_removed} are not in span unit_ids {span_unit_ids}"
            )

        # Keeper must not be in removal list
        if keeper_unit_id in units_to_remove:
            raise ValueError(
                f"Self-contradictory response: keeper_unit_id '{keeper_unit_id}' is also listed in units_to_remove {units_to_remove}"
            )

    confidence = str(data.get("confidence", "high")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    reasoning = str(data.get("reasoning", "")).strip()

    return {
        "is_genuine_retake_group": is_genuine,
        "keeper_unit_id": keeper_unit_id,
        "units_to_remove": units_to_remove,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def adjudicate_single_span(
    span: dict[str, Any],
    all_units_by_id: dict[str, Any] | Sequence[Any] | None = None,
    config: AdjudicationConfig | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """
    Adjudicates a single ambiguous span using the LLM with validation and retry logic.
    """
    cfg = config or default_adjudication_config
    llm = client or get_llm_client(cfg)

    prompt = build_adjudication_prompt(span, all_units_by_id=all_units_by_id)

    last_error: Exception | None = None
    max_attempts = max(1, cfg.max_retries + 1)

    for attempt in range(max_attempts):
        try:
            logger.debug(f"[Adjudication] Querying LLM for span {span.get('span_id')} (Attempt {attempt + 1}/{max_attempts})")
            raw_response = llm.generate_json(prompt)
            verdict = parse_adjudication_response(raw_response, span=span)

            return {
                "adjudication_status": "resolved",
                "is_genuine_retake_group": verdict["is_genuine_retake_group"],
                "keeper_unit_id": verdict["keeper_unit_id"],
                "units_to_remove": verdict["units_to_remove"],
                "confidence": verdict["confidence"],
                "reasoning": verdict["reasoning"],
                "llm_raw_response": raw_response,
            }
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"[Adjudication] Attempt {attempt + 1}/{max_attempts} for span {span.get('span_id')} failed: {exc}"
            )

    # Exhausted retries — never guess
    logger.error(
        f"[Adjudication] All {max_attempts} attempts failed for span {span.get('span_id')}. Marking as failed."
    )
    return {
        "adjudication_status": "failed",
        "is_genuine_retake_group": None,
        "keeper_unit_id": None,
        "units_to_remove": None,
        "confidence": None,
        "reasoning": None,
        "adjudication_error": f"Exhausted {max_attempts} attempts: {last_error}",
    }


def run_stage6(
    spans: Sequence[Any],
    all_units_by_id: dict[str, Any] | Sequence[Any] | None = None,
    config: AdjudicationConfig | None = None,
    llm_client: LLMClient | None = None,
) -> list[dict[str, Any]]:
    """
    Pipeline entry point for Step 06 (LLM Adjudication):
    - For spans where needs_adjudication == False:
      Pass through untouched with adjudication_status = "not_required", normalizing
      likely_final_unit_id into keeper_unit_id and computing units_to_remove.
    - For spans where needs_adjudication == True:
      Call LLM adjudication, validate response, and attach resolved verdict or failed status.

    Args:
        spans: Sequence of Step 05 RetakeSpan objects or dicts.
        all_units_by_id: Mapping from unit_id to ComparableUnit, or sequence of units.
        config: AdjudicationConfig instance.
        llm_client: Optional explicit LLMClient instance (e.g. MockLLMClient for testing).

    Returns:
        List of adjudicated span dictionaries adhering to the unified Step 06 output contract.
    """
    cfg = config or default_adjudication_config
    results: list[dict[str, Any]] = []

    for item in spans:
        span_dict = _extract_span_dict(item)
        needs_adj = span_dict.get("needs_adjudication", False)

        if not needs_adj:
            # Pass-through path: already resolved in Step 05
            likely_final = span_dict.get("likely_final_unit_id")
            unit_ids = span_dict.get("unit_ids", [])
            units_to_remove = [uid for uid in unit_ids if uid != likely_final] if likely_final else []

            out_span = dict(span_dict)
            out_span["adjudication_status"] = "not_required"
            out_span["keeper_unit_id"] = likely_final
            out_span["units_to_remove"] = units_to_remove
            out_span["is_genuine_retake_group"] = True if likely_final else False
            results.append(out_span)
        else:
            # Ambiguous path: execute LLM adjudication
            adj_result = adjudicate_single_span(
                span_dict,
                all_units_by_id=all_units_by_id,
                config=cfg,
                client=llm_client,
            )

            out_span = dict(span_dict)
            out_span.update(adj_result)
            results.append(out_span)

    return results
