"""Data models for cutlist generation and retake grouping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class TimeSpan:
    """A continuous time interval in the source media."""

    start: float
    end: float
    label: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlaps(self, other: "TimeSpan", tolerance: float = 0.0) -> bool:
        """Check if two spans overlap or are within `tolerance` seconds of each other."""
        return self.start <= (other.end + tolerance) and other.start <= (self.end + tolerance)

    def merge(self, other: "TimeSpan") -> "TimeSpan":
        """Merge two overlapping spans into a single span."""
        return TimeSpan(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            label=self.label or other.label,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeSpan":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            label=str(data.get("label", "")),
        )


@dataclass(slots=True)
class RetakeGroup:
    """
    A group of consecutive candidate pairs that form a single retake event.

    Example: The speaker reads sentences A–E, then re-reads them as A'–E'.
    The group contains pairs (A, A'), (B, B'), ..., (E, E') and identifies
    which take (the A–E block or the A'–E' block) should be kept.
    """

    group_id: int
    candidate_ids: list[int]
    take_a_span: TimeSpan
    take_b_span: TimeSpan
    kept_take: str  # "a" or "b"
    discarded_take: str  # "a" or "b"
    discard_span: TimeSpan
    keep_span: TimeSpan
    reason: str = ""  # Why this take was selected
    pair_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "candidate_ids": self.candidate_ids,
            "take_a_span": self.take_a_span.to_dict(),
            "take_b_span": self.take_b_span.to_dict(),
            "kept_take": self.kept_take,
            "discarded_take": self.discarded_take,
            "discard_span": self.discard_span.to_dict(),
            "keep_span": self.keep_span.to_dict(),
            "reason": self.reason,
            "pair_count": self.pair_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetakeGroup":
        return cls(
            group_id=int(data["group_id"]),
            candidate_ids=list(data["candidate_ids"]),
            take_a_span=TimeSpan.from_dict(data["take_a_span"]),
            take_b_span=TimeSpan.from_dict(data["take_b_span"]),
            kept_take=str(data["kept_take"]),
            discarded_take=str(data["discarded_take"]),
            discard_span=TimeSpan.from_dict(data["discard_span"]),
            keep_span=TimeSpan.from_dict(data["keep_span"]),
            reason=str(data.get("reason", "")),
            pair_count=int(data.get("pair_count", 0)),
        )


@dataclass(slots=True)
class CutList:
    """
    The final output of cutlist generation: a set of time spans to keep
    and discard, along with the retake groups that produced them.
    """

    source: str
    source_duration: float
    keep_spans: list[TimeSpan]
    discard_spans: list[TimeSpan]
    retake_groups: list[RetakeGroup]
    output_duration: float = 0.0

    def __post_init__(self):
        if self.output_duration == 0.0:
            self.output_duration = sum(s.duration for s in self.keep_spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_duration": self.source_duration,
            "output_duration": self.output_duration,
            "time_saved": round(self.source_duration - self.output_duration, 3),
            "retake_groups_count": len(self.retake_groups),
            "keep_spans": [s.to_dict() for s in self.keep_spans],
            "discard_spans": [s.to_dict() for s in self.discard_spans],
            "retake_groups": [g.to_dict() for g in self.retake_groups],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CutList":
        return cls(
            source=str(data["source"]),
            source_duration=float(data["source_duration"]),
            keep_spans=[TimeSpan.from_dict(s) for s in data.get("keep_spans", [])],
            discard_spans=[TimeSpan.from_dict(s) for s in data.get("discard_spans", [])],
            retake_groups=[RetakeGroup.from_dict(g) for g in data.get("retake_groups", [])],
            output_duration=float(data.get("output_duration", 0.0)),
        )
