from abc import ABC, abstractmethod
from typing import Optional
from models.transcript import Transcript


class BaseTranscriber(ABC):
    """Abstract Base Class for all STT (Speech-To-Text) Transcription Adapters."""

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def transcribe(self, audio_path: str, source_name: Optional[str] = None, **kwargs) -> Transcript:
        """
        Transcribes an audio file into a canonical Transcript model.

        Args:
            audio_path: Path to the input audio file (WAV/MP3/etc.).
            source_name: Optional original source filename (e.g. video.mp4).
            **kwargs: Extra runtime options (e.g., language, prompt, diarization).

        Returns:
            Transcript: Validated Pydantic Transcript object.
        """
        pass
