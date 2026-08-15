import os
import math
from typing import Optional
from models.transcript import Transcript
from models.segment import Segment, Word

try:
    from .base import BaseTranscriber  # type: ignore
except ImportError:
    from pipeline.transcription.base import BaseTranscriber  # type: ignore


class WhisperTranscriber(BaseTranscriber):
    """
    STT Adapter using faster-whisper (CTranslate2).
    Provides fast, local execution with word-level timestamps and VAD filtering.
    """

    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "default", **kwargs):
        super().__init__(model_size=model_size, device=device, compute_type=compute_type, **kwargs)
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper is not installed. Install it with: pip install faster-whisper"
                )
            self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio_path: str, source_name: Optional[str] = None, **kwargs) -> Transcript:
        model = self._get_model()
        language = kwargs.get("language", "en")
        vad_filter = kwargs.get("vad_filter", True)
        word_timestamps = kwargs.get("word_timestamps", True)

        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )

        segments_list = []
        seg_id = 1
        for seg in segments_iter:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        Word(
                            word=w.word.strip(),
                            start=round(w.start, 2),
                            end=round(w.end, 2),
                            confidence=round(w.probability, 2) if hasattr(w, "probability") else None,
                        )
                    )

            confidence = round(math.exp(seg.avg_logprob), 2) if hasattr(seg, "avg_logprob") else 1.0
            confidence = min(max(confidence, 0.0), 1.0)

            segments_list.append(
                Segment(
                    id=seg_id,
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    speaker="speaker_0",
                    text=seg.text.strip(),
                    confidence=confidence,
                    words=words,
                )
            )
            seg_id += 1

        source = source_name or os.path.basename(audio_path)
        duration = round(info.duration, 2) if hasattr(info, "duration") and info.duration else (
            segments_list[-1].end if segments_list else 0.0
        )

        return Transcript(
            source=source,
            duration=duration,
            language=info.language if hasattr(info, "language") else language,
            segments=segments_list,
        )
