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
    Validates the LLM response against the span it was asked about (the span is already a confirmed retake group):
    - keeper_unit_id must be present and one of span.unit_ids
    - units_to_remove must all be members of span.unit_ids
    - keeper_unit_id must never appear in units_to_remove
    Raises ValueError on validation failure. is_genuine_retake_group is always True in the result.
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

    # Every span reaching Stage 6 is already a confirmed retake group, so the model's own verdict (if it sends one) is ignored and
    # is_genuine_retake_group is always True. The consistency checks below are unchanged.
    is_genuine = True

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

    # Must select a keeper
    if keeper_unit_id is None:
        raise ValueError("keeper_unit_id is null/missing")

    # Keeper must belong to span
    if keeper_unit_id not in span_unit_ids:
        raise ValueError(f"Hallucinated keeper_unit_id '{keeper_unit_id}' is not in span unit_ids {span_unit_ids}")

    # Removed units must belong to span
    invalid_removed = [uid for uid in units_to_remove if uid not in span_unit_ids]
    if invalid_removed:
        raise ValueError(f"Hallucinated units_to_remove {invalid_removed} are not in span unit_ids {span_unit_ids}")

    # Keeper must not be in removal list
    if keeper_unit_id in units_to_remove:
        raise ValueError(f"Self-contradictory response: keeper_unit_id '{keeper_unit_id}' is also listed in units_to_remove {units_to_remove}")

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
        "is_genuine_retake_group": True,
        "keeper_unit_id": None,
        "units_to_remove": None,
        "confidence": None,
        "reasoning": None,
        "adjudication_error": f"Exhausted {max_attempts} attempts: {last_error}",
    }


def _api_key_available(cfg: AdjudicationConfig) -> bool:
    """True if the configured provider can authenticate (mock needs no key). Keys are read, never printed."""
    import os
    provider = (cfg.provider or "").lower()
    if provider == "mock":
        return True
    if cfg.api_key:
        return True
    if provider in {"openai", "groq"}:
        return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY"))
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def run_stage6(
    spans: Sequence[Any],
    all_units_by_id: dict[str, Any] | Sequence[Any] | None = None,
    config: AdjudicationConfig | None = None,
    llm_client: LLMClient | None = None,
) -> list[dict[str, Any]]:
    """
    Pipeline entry point for Step 06 (keeper selection).

    Every span reaching this step is already a CONFIRMED retake group (Stage 4 decided each pair, Stage 5 clustered them), so every span
    goes to the LLM, and the LLM only chooses the keeper unit and the units to remove. There is no pass-through path.
    A span the LLM cannot resolve (no API key, API failure, invalid answer after the retries) gets adjudication_status "failed", is recorded
    in the output and is NOT cut; a clear log line says so and no exception is raised.

    Args:
        spans: Step 05 RetakeSpan objects or dicts.
        all_units_by_id: Mapping from unit_id to ComparableUnit, or sequence of units.
        config: AdjudicationConfig instance (provider and model come from config / environment variables).
        llm_client: Optional explicit LLMClient instance (e.g. MockLLMClient for testing).

    Returns:
        List of span dictionaries with the Step 06 output contract (resolved or failed).
    """
    cfg = config or default_adjudication_config
    results: list[dict[str, Any]] = []
    key_ok = llm_client is not None or _api_key_available(cfg)
    if not key_ok:
        logger.warning("[Stage 6] no API key available for provider '%s': all %d spans stay UNRESOLVED and will not be cut", cfg.provider, len(spans))

    for item in spans:
        span_dict = _extract_span_dict(item)
        if key_ok:
            adj_result = adjudicate_single_span(span_dict, all_units_by_id=all_units_by_id, config=cfg, client=llm_client)
        else:
            adj_result = {"adjudication_status": "failed", "is_genuine_retake_group": True, "keeper_unit_id": None, "units_to_remove": None,
                          "confidence": None, "reasoning": None, "adjudication_error": f"no API key available for provider '{cfg.provider}'"}
        if adj_result["adjudication_status"] != "resolved":
            logger.warning("[Stage 6] span %s unresolved (%s): skipped, nothing is cut", span_dict.get("span_id"), adj_result.get("adjudication_error"))
        out_span = dict(span_dict)
        out_span.update(adj_result)
        results.append(out_span)
    return results
