from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field
from models.segment import Segment


class Transcript(BaseModel):
    source: str
    duration: float
    language: str = "en"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    segments: List[Segment] = Field(default_factory=list)

    def to_json_file(self, filepath: str) -> None:
        """Saves transcript to a JSON file matching the canonical schema."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def from_json_file(cls, filepath: str) -> "Transcript":
        """Loads transcript from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())
