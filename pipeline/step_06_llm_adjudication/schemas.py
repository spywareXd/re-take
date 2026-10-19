from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal
from pydantic import BaseModel, Field


class AdjudicationVerdict(BaseModel):
    """
    Structured output contract returned by the LLM for an ambiguous retake span.
    """
    is_genuine_retake_group: bool = Field(
        description="True if this span represents multiple attempts/restarts of the same line, False if distinct valid thoughts."
    )
    keeper_unit_id: str | None = Field(
        default=None,
        description="The single unit_id from span.unit_ids to retain as the clean final take. Must be null if is_genuine_retake_group is False."
    )
    units_to_remove: list[str] = Field(
        default_factory=list,
        description="List of unit_ids in the span that should be cut out. Must be empty if is_genuine_retake_group is False."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description="Self-reported model confidence in this adjudication decision."
    )
    reasoning: str = Field(
        default="",
        description="1-2 sentences explaining why this unit is kept or why this is/isn't a retake (audit trail only)."
    )


@dataclass(slots=True)
class AdjudicatedSpan:
    """
    Represents a RetakeSpan after Step 06 adjudication (either resolved, pass-through, or failed).
    """
    span_id: str
    unit_ids: list[str]
    start: float
    end: float
    needs_adjudication: bool
    adjudication_status: Literal["resolved", "not_required", "failed"]
    keeper_unit_id: str | None
    units_to_remove: list[str] | None
    is_genuine_retake_group: bool | None = None
    confidence: str | None = None
    reasoning: str | None = None
    likely_final_unit_id: str | None = None
    candidate_pairs: list[dict[str, Any]] | None = None
    classification_summary: dict[str, int] | None = None
    llm_raw_response: dict[str, Any] | None = None
    adjudication_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "span_id": self.span_id,
            "unit_ids": list(self.unit_ids),
            "start": round(float(self.start), 4),
            "end": round(float(self.end), 4),
            "needs_adjudication": bool(self.needs_adjudication),
            "adjudication_status": self.adjudication_status,
            "keeper_unit_id": self.keeper_unit_id,
            "units_to_remove": list(self.units_to_remove) if self.units_to_remove is not None else None,
        }
        if self.is_genuine_retake_group is not None:
            d["is_genuine_retake_group"] = bool(self.is_genuine_retake_group)
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        if self.likely_final_unit_id is not None:
            d["likely_final_unit_id"] = self.likely_final_unit_id
        if self.candidate_pairs is not None:
            d["candidate_pairs"] = self.candidate_pairs
        if self.classification_summary is not None:
            d["classification_summary"] = self.classification_summary
        if self.llm_raw_response is not None:
            d["llm_raw_response"] = self.llm_raw_response
        if self.adjudication_error is not None:
            d["adjudication_error"] = self.adjudication_error
        return d
