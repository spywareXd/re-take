import os
from typing import Optional
from models.transcript import Transcript
from models.segment import Segment, Word

try:
    from .base import BaseTranscriber  # type: ignore
except ImportError:
    from pipeline.transcription.base import BaseTranscriber  # type: ignore


class MockTranscriber(BaseTranscriber):
    """
    Mock STT Adapter for fast offline testing and development.
    Can load an existing test JSON or return realistic synthetic transcript segments.
    """

    def __init__(self, sample_json_path: Optional[str] = None, **kwargs):
        super().__init__(sample_json_path=sample_json_path, **kwargs)
        self.sample_json_path = sample_json_path

    def transcribe(self, audio_path: str, source_name: Optional[str] = None, **kwargs) -> Transcript:
        json_path = kwargs.get("sample_json_path") or self.sample_json_path

        if json_path and os.path.exists(json_path):
            transcript = Transcript.from_json_file(json_path)
            if source_name:
                transcript.source = source_name
            return transcript

        # Fallback to default mock retake scenario
        source = source_name or os.path.basename(audio_path)
        segments = [
            Segment(
                id=1,
                start=0.0,
                end=4.2,
                speaker="speaker_0",
                text="Hello everyone, welcome back to the channel.",
                confidence=0.96,
                words=[
                    Word(word="Hello", start=0.0, end=0.8, confidence=0.98),
                    Word(word="everyone,", start=0.9, end=1.5, confidence=0.96),
                    Word(word="welcome", start=1.6, end=2.2, confidence=0.97),
                    Word(word="back", start=2.3, end=2.8, confidence=0.96),
                    Word(word="to", start=2.9, end=3.1, confidence=0.99),
                    Word(word="the", start=3.2, end=3.4, confidence=0.99),
                    Word(word="channel.", start=3.5, end=4.2, confidence=0.95),
                ],
            ),
            Segment(
                id=2,
                start=4.3,
                end=8.9,
                speaker="speaker_0",
                text="Today we're going to discuss JWT authentication.",
                confidence=0.94,
            ),
            Segment(
                id=3,
                start=9.1,
                end=10.4,
                speaker="speaker_0",
                text="Wait, let me restart.",
                confidence=0.88,
            ),
            Segment(
                id=4,
                start=10.7,
                end=15.3,
                speaker="speaker_0",
                text="Today we're going to discuss JWT authentication.",
                confidence=0.97,
            ),
        ]
        return Transcript(
            source=source,
            duration=15.3,
            language="en",
            segments=segments,
        )
