from typing import List, Optional
from pydantic import BaseModel, Field


class Word(BaseModel):
    word: str
    start: float
    end: float
    confidence: Optional[float] = None


class Segment(BaseModel):
    id: int
    start: float
    end: float
    speaker: str = "speaker_0"
    text: str
    confidence: Optional[float] = 1.0
    words: Optional[List[Word]] = Field(default_factory=list)
